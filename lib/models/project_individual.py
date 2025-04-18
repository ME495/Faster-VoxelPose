# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cameras import project_pose
from utils.transforms import affine_transform_pts_cuda as do_transform

class ProjectLayer(nn.Module):
    """
    个体投影层：为每个检测到的人体构建单独的特征体素
    
    与全局投影层(project_whole.py)不同，这个版本为每个人体单独构建一个小体素，而不是一个大的全局体素。
    这样做的好处是：
    1. 减少计算量和内存消耗
    2. 专注于特定人体区域，提高精度
    3. 便于处理多人场景
    
    工作流程：
    1. 根据检测到的人体中心位置，确定个体体素在全局空间中的位置
    2. 构建采样网格，从2D热图中提取特征
    3. 处理体素边界，避免越界
    4. 返回个体特征体素和位置偏移
    """
    def __init__(self, cfg):
        """
        初始化个体投影层
        
        Args:
            cfg: 配置对象，包含空间参数和体素参数
        """
        super(ProjectLayer, self).__init__()
        self.device = torch.device(cfg.DEVICE)  # 计算设备
        self.image_size = cfg.DATASET.IMAGE_SIZE  # 输入图像尺寸
        self.heatmap_size = cfg.DATASET.HEATMAP_SIZE  # 热图尺寸
        self.ori_image_size = cfg.DATASET.ORI_IMAGE_SIZE  # 原始图像尺寸

        # 反投影所需的常量参数
        self.whole_space_center = torch.tensor(cfg.CAPTURE_SPEC.SPACE_CENTER, device=self.device)  # 全局空间中心
        self.whole_space_size = torch.tensor(cfg.CAPTURE_SPEC.SPACE_SIZE, device=self.device)  # 全局空间大小
        self.ind_space_size = torch.tensor(cfg.INDIVIDUAL_SPEC.SPACE_SIZE, device=self.device)  # 个体空间大小
        self.voxels_per_axis = torch.tensor(cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS, device=self.device, dtype=torch.int32)  # 个体体素每轴体素数
        
        # 计算在全局空间中对应的精细体素尺寸
        # 将个体体素映射到全局空间时，需要更精细的体素以保证精度
        self.fine_voxels_per_axis = (self.whole_space_size / self.ind_space_size * (self.voxels_per_axis - 1)).int() + 1
        '''
        这行代码计算精细体素的数量，其核心思路是保持个体空间和全局空间的体素密度一致。让我详细解释：
        1. 比例关系：
            - self.whole_space_size / self.ind_space_size：计算全局空间和个体空间的大小比例
            - 例如，如果全局空间是8000mm，个体空间是2000mm，那么比例就是4
        2. 体素密度映射：
            - voxels_per_axis - 1：表示个体空间中相邻体素之间的间隔数
            - 将这个间隔数乘以空间比例，确保在全局空间中保持相同的体素密度
            - 例如，如果个体空间每轴有32个体素，间隔数是31，那么在全局空间中对应的间隔数就是31 * 4 = 124
        3. 边界处理：
            - .int() + 1：将计算结果向下取整后加1
            - 这样做是为了确保有足够的体素覆盖整个空间，避免因为取整造成的空间覆盖不完整
        举个具体例子：
            假设：
            - 全局空间大小 whole_space_size = 8000mm
            - 个体空间大小 ind_space_size = 2000mm
            - 个体空间体素数 voxels_per_axis = 32

            计算过程：
            1. 空间比例 = 8000/2000 = 4
            2. 间隔映射 = 4 * (32-1) = 4 * 31 = 124
            3. 最终体素数 = 124 + 1 = 125
        这样设计的好处是：
        1. 精度一致性：确保在全局空间和个体空间中的体素具有相同的物理尺寸
        2. 无缝映射：便于在全局空间和个体空间之间进行坐标转换
        3. 避免信息丢失：通过合理的取整和边界处理，确保空间被完整覆盖
        这是一个巧妙的设计，它在保持计算效率的同时，确保了特征提取的精度和空间表示的连续性。
        '''

        # 计算坐标转换参数
        # scale: 从全局体素索引到个体体素索引的缩放因子
        # bias: 从全局体素索引到个体体素索引的偏移量
        self.scale = (self.fine_voxels_per_axis.float() - 1)  / self.whole_space_size  # 缩放因子
        self.bias = - self.ind_space_size / 2.0 / self.whole_space_size * (self.fine_voxels_per_axis - 1)\
                    - self.scale * (self.whole_space_center - self.whole_space_size / 2.0)  # 偏移量

        # 预先计算网格
        self.save_grid() 
        self.sample_grid = {}  # 缓存不同序列的采样网格

    def save_grid(self):
        """
        预先计算并保存3D网格，用于特征采样
        
        计算两种网格：
        1. center_grid: 用于软性坐标回归的中心网格，包含三个正交平面(xy, xz, yz)
        2. fine_grid: 用于精细采样的全局网格
        """
        print("=> save the 3D grid for feature sampling")
        # 计算个体体素的网格
        grid = self.compute_grid(self.ind_space_size, self.whole_space_center, self.voxels_per_axis, device=self.device)
        grid = grid.view(self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2], 3)
        
        # 提取三个正交平面(xy, xz, yz)的2D网格，用于后续的软性坐标回归
        self.center_grid = torch.stack([grid[:, :, 0, :2].reshape(-1, 2),  # xy平面
                                        grid[:, 0, :, ::2].reshape(-1, 2),  # xz平面
                                        grid[0, :, :, 1:].reshape(-1, 2)])  # yz平面
        
        # 计算精细网格，用于特征采样
        self.fine_grid = self.compute_grid(self.whole_space_size, self.whole_space_center, self.fine_voxels_per_axis, device=self.device)
        return

    def compute_grid(self, boxSize, boxCenter, nBins, device):
        """
        计算3D体素空间中的网格点坐标
        
        Args:
            boxSize: 空间大小
            boxCenter: 空间中心
            nBins: 每个轴上的体素数量
            device: 计算设备
            
        Returns:
            grid: 形状为(N, 3)的张量，表示所有体素中心点的3D坐标
        """
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        # 创建x, y, z轴上的一维网格
        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        
        # 使用meshgrid创建3D网格
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
            indexing='ij'
        )
        
        # 将三个坐标轴的网格点整合为一个网格
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def project_grid(self, camera, w, h, nbins, resize_transform, device):
        """
        将3D网格点投影到2D图像平面上
        
        Args:
            camera: 相机参数
            w: 热图宽度
            h: 热图高度
            nbins: 体素总数
            resize_transform: 图像缩放变换矩阵
            device: 计算设备
            
        Returns:
            sample_grid: 用于网格采样的坐标网格，范围在[-1.1, 1.1]之间
        """
        # 使用相机参数将3D点投影到2D平面
        xy = project_pose(self.fine_grid, camera)
        
        # 裁剪坐标范围
        xy = torch.clamp(xy, -1.0, max(self.ori_image_size[0], self.ori_image_size[1]))
        
        # 应用仿射变换（调整图像尺寸）
        xy = do_transform(xy, resize_transform)
        
        # 将坐标缩放到热图尺寸
        xy = xy * torch.tensor(
            [w, h], dtype=torch.float, device=device) / torch.tensor(
            self.image_size, dtype=torch.float, device=device)

        # 将坐标归一化到[-1, 1]范围，用于grid_sample函数
        sample_grid = xy / torch.tensor(
            [w - 1, h - 1], dtype=torch.float,
            device=device) * 2.0 - 1.0
            
        # 调整尺寸并裁剪坐标范围
        sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)
        return sample_grid
        
    '''
    only compute the projected 2D finer grid once for each sequence
    '''
    def compute_sample_grid(self, heatmaps, meta, i, voxels_per_axis, seq, cameras, resize_transform):
        """
        计算采样网格，并缓存以提高效率
        
        对每个序列，只计算一次投影的2D精细网格，然后缓存起来重复使用
        
        Args:
            heatmaps: 输入热图
            meta: 元数据信息
            i: 批次索引
            voxels_per_axis: 每轴体素数
            seq: 序列ID
            cameras: 相机参数
            resize_transform: 图像缩放变换矩阵
        """
        device = heatmaps.device
        nbins = voxels_per_axis[0] * voxels_per_axis[1] * voxels_per_axis[2]  # 体素总数
        n = heatmaps.shape[1]  # 视角数量
        w, h = self.heatmap_size  # 热图尺寸

        # 计算采样网格
        sample_grids = torch.zeros(n, 1, 1, nbins, 2, device=device)
        curr_seq = meta['seq'][i]
        for c in range(n):
            sample_grid = self.project_grid(cameras[curr_seq][c], w, h, nbins, resize_transform, device)
            sample_grids[c] = sample_grid
        
        # 缓存采样网格，调整形状以便后续使用
        self.sample_grid[seq] = sample_grids.view(n, voxels_per_axis[0], voxels_per_axis[1], voxels_per_axis[2], 2)

    def forward(self, heatmaps, index, meta, proposal_centers, cameras, resize_transform):
        """
        前向传播函数：构建个体特征体素
        
        Args:
            heatmaps: 多视角输入热图，形状为[batch_size, num_views, num_joints, height, width]
            index: 批次索引
            meta: 元数据信息
            proposal_centers: 人体中心位置，形状为[num_people, 7]
            cameras: 相机参数
            resize_transform: 图像缩放变换矩阵
            
        Returns:
            cubes: 个体特征体素，形状为[num_people, num_joints, voxels_x, voxels_y, voxels_z]
            offset: 位置偏移，用于将相对坐标转换为全局坐标
        """
        device = heatmaps.device
        num_people = proposal_centers.shape[0]  # 人数
        n = heatmaps.shape[1]  # 视角数
        num_joints = heatmaps.shape[2]  # 关节数
        
        # 初始化输出体素
        cubes = torch.zeros(num_people, num_joints, self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2], device=device)

        # 获取当前序列ID
        curr_seq = meta['seq'][index]
        # 如果当前序列的采样网格尚未计算，则计算并缓存
        if curr_seq not in self.sample_grid:
            print("=> save the sampling grid in JLN for sequence", curr_seq)
            self.compute_sample_grid(heatmaps, meta, index, self.fine_voxels_per_axis, curr_seq, cameras, resize_transform)

        # 计算每个人体的个体体素在全局精细网格中的顶点索引
        # proposal_centers[:, :3]包含人体中心的3D坐标
        centers_tl = torch.round(proposal_centers[:, :3].float() * self.scale + self.bias).int()
        
        # 计算位置偏移，用于将个体体素中的相对坐标转换为全局坐标
        offset = centers_tl.float() / (self.fine_voxels_per_axis - 1) * self.whole_space_size - self.whole_space_size / 2.0 + self.ind_space_size / 2.0
        
        # 根据边界框大小创建掩码，过滤体素外的区域
        # proposal_centers[:, 5:7]包含边界框的宽和高
        mask = ((1 - proposal_centers[:, 5:7]) / 2 * (self.voxels_per_axis[0:2] - 1)).int()
        mask[mask < 0] = 0
        # 垂直方向(z轴)的边界框长度固定为2000mm
        mask = torch.cat([mask, torch.zeros((num_people, 1), device=device, dtype=torch.int32)], dim=1)

        # 计算有效范围，避免越界
        start = torch.where(centers_tl + mask >= 0, centers_tl + mask, torch.zeros_like(centers_tl))
        end = torch.where(centers_tl + self.voxels_per_axis - mask <= self.fine_voxels_per_axis, centers_tl + self.voxels_per_axis - mask, self.fine_voxels_per_axis)

        # 构建每个人的特征体素
        for i in range(num_people):
            # 如果有效范围无效，则跳过
            if torch.sum(start[i] >= end[i]) > 0:
                continue
                
            # 获取当前序列的采样网格
            sample_grid = self.sample_grid[curr_seq]
            # 提取当前人体有效范围内的采样网格
            sample_grid = sample_grid[:, start[i, 0]:end[i, 0], start[i, 1]:end[i, 1], start[i, 2]:end[i, 2]].reshape(n, 1, -1, 2)

            # 使用grid_sample从多视角热图中采样特征，然后取平均
            accu_cubes = torch.mean(F.grid_sample(heatmaps[index], sample_grid, align_corners=True), dim=0).view(num_joints, end[i, 0]-start[i, 0], end[i, 1]-start[i, 1], end[i, 2]-start[i, 2])
            
            # 将采样结果放入对应的位置
            cubes[i, :, start[i, 0]-centers_tl[i, 0]:end[i, 0]-centers_tl[i, 0], start[i, 1]-centers_tl[i, 1]:end[i, 1]-centers_tl[i, 1], start[i, 2]-centers_tl[i, 2]:end[i, 2]-centers_tl[i, 2]] = accu_cubes
            del sample_grid

        # 裁剪值到[0, 1]范围
        cubes = cubes.clamp(0.0, 1.0)
        # 释放不再需要的变量，节省内存
        del centers_tl, mask, start, end
        return cubes, offset