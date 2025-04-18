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
    投影层：将2D热图投影到3D体素空间
    
    该层的主要功能是将多个相机视角下的2D关节热图投影并融合成3D体素表示。
    它首先在3D空间中创建一个体素网格，然后将每个体素投影到每个相机视角下的2D平面上，
    通过采样2D热图的相应位置来填充3D体素。
    """
    def __init__(self, cfg):
        """
        初始化投影层
        
        Args:
            cfg: 配置对象，包含各种参数设置
        """
        super(ProjectLayer, self).__init__()
        self.device = torch.device(cfg.DEVICE)  # 计算设备（CPU或GPU）
        self.image_size = cfg.DATASET.IMAGE_SIZE  # 输入图像尺寸
        self.heatmap_size = cfg.DATASET.HEATMAP_SIZE  # 热图尺寸
        self.ori_image_size = cfg.DATASET.ORI_IMAGE_SIZE  # 原始图像尺寸

        # 3D空间参数设置
        self.space_size = cfg.CAPTURE_SPEC.SPACE_SIZE  # 3D空间的大小
        self.space_center = cfg.CAPTURE_SPEC.SPACE_CENTER  # 3D空间的中心点
        self.voxels_per_axis = cfg.CAPTURE_SPEC.VOXELS_PER_AXIS  # 每个轴上的体素数量

        # 计算3D空间中的网格点
        self.grid = self.compute_grid(self.space_size, self.space_center, self.voxels_per_axis, device=self.device)
        self.sample_grid = {}  # 缓存每个序列的采样网格
    
    def compute_grid(self, boxSize, boxCenter, nBins, device):
        """
        计算3D体素空间中的网格点坐标
        
        Args:
            boxSize: 3D空间的大小
            boxCenter: 3D空间的中心点
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
        xy = project_pose(self.grid, camera)
        
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
        
    def forward(self, heatmaps, meta, cameras, resize_transform):
        """
        前向传播函数：将多视角2D热图投影到3D体素空间
        
        Args:
            heatmaps: 2D热图，形状为(batch_size, num_views, num_joints, h, w)
            meta: 元数据信息
            cameras: 相机参数字典
            resize_transform: 图像缩放变换矩阵
            
        Returns:
            cubes: 3D体素表示，形状为(batch_size, num_joints, width, height, depth)
        """
        device = heatmaps.device
        batch_size = heatmaps.shape[0]
        n = heatmaps.shape[1]  # 视角数量
        num_joints = heatmaps.shape[2]  # 关节数量
        nbins = self.voxels_per_axis[0] * self.voxels_per_axis[1] * self.voxels_per_axis[2]  # 体素总数
        
        # 初始化输出体素
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, device=device)
        w, h = self.heatmap_size
        
        for i in range(batch_size):
            # 获取当前序列ID
            curr_seq = meta['seq'][i]
            
            # 检查相机参数是否存在
            assert curr_seq in cameras.keys(), "missing camera parameters for the current sequence"
            assert len(cameras[curr_seq]) == n, "inconsistent number of cameras"
            
            # 如果当前序列的采样网格尚未计算，则计算并缓存
            if curr_seq not in self.sample_grid:
                print("=> save the sampling grid in HDN for sequence", curr_seq)
                sample_grids = torch.zeros(n, 1, nbins, 2, device=device)
                for c in range(n):
                    sample_grids[c] = self.project_grid(cameras[curr_seq][c], w, h, nbins, resize_transform, device).squeeze(0)
                self.sample_grid[curr_seq] = sample_grids

            # 使用缓存的采样网格
            shared_sample_grid = self.sample_grid[curr_seq]
            
            # 对每个视角的热图进行网格采样，然后求平均
            cubes[i] = torch.mean(F.grid_sample(heatmaps[i], shared_sample_grid, align_corners=True), dim=0).squeeze(0)
            del shared_sample_grid
            
        # 裁剪值到[0, 1]范围，并重塑为最终的体素形状
        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_joints, self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2]) 
        return cubes