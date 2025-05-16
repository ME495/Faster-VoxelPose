# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

# 测试时需要添加
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from utils.cameras import project_pose
from utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayerTS(nn.Module):
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
    
    image_size: List[int]
    heatmap_size: List[int]
    ori_image_size: List[int]
    whole_space_center: torch.Tensor
    whole_space_size: torch.Tensor
    ind_space_size: torch.Tensor
    voxels_per_axis: torch.Tensor
    fine_voxels_per_axis: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor
    center_grid: torch.Tensor
    
    def __init__(self, image_size: List[int], heatmap_size: List[int], ori_image_size: List[int], 
                 space_center: List[float], space_size: List[float], 
                 individual_space_size: List[float], voxels_per_axis: List[int]):
        """
        初始化个体投影层
        
        Args:
            image_size: 输入图像尺寸 [width, height]
            heatmap_size: 热图尺寸 [width, height]
            ori_image_size: 原始图像尺寸 [width, height]
            space_center: 全局空间中心 [x, y, z]
            space_size: 全局空间大小 [x, y, z]
            individual_space_size: 个体空间大小 [x, y, z]
            voxels_per_axis: 个体体素每轴体素数 [x, y, z]
        """
        super(ProjectLayerTS, self).__init__()
        self.image_size = image_size  # 输入图像尺寸
        self.heatmap_size = heatmap_size  # 热图尺寸
        self.ori_image_size = ori_image_size  # 原始图像尺寸

        # 反投影所需的常量参数
        self.register_buffer('whole_space_center', torch.tensor(space_center))  # 全局空间中心
        self.register_buffer('whole_space_size', torch.tensor(space_size))  # 全局空间大小
        self.register_buffer('ind_space_size', torch.tensor(individual_space_size))  # 个体空间大小
        self.register_buffer('voxels_per_axis', torch.tensor(voxels_per_axis, dtype=torch.int32))  # 个体体素每轴体素数
        
        # 计算在全局空间中对应的精细体素尺寸
        self.register_buffer('fine_voxels_per_axis', 
                           (self.whole_space_size / self.ind_space_size * (self.voxels_per_axis - 1)).int() + 1)

        # 计算坐标转换参数
        self.register_buffer('scale', (self.fine_voxels_per_axis.float() - 1) / self.whole_space_size)  # 缩放因子
        self.register_buffer('bias', - self.ind_space_size / 2.0 / self.whole_space_size * (self.fine_voxels_per_axis - 1)\
                    - self.scale * (self.whole_space_center - self.whole_space_size / 2.0))  # 偏移量
        
        # 预先计算网格
        self._save_grid() 

    def _save_grid(self):
        """
        预先计算并保存3D网格，用于特征采样
        
        计算网格：
        1. center_grid: 用于软性坐标回归的中心网格，包含三个正交平面(xy, xz, yz)
        """
        print("=> save the 3D grid for feature sampling")
        # 计算个体体素的网格
        grid = self._compute_grid(self.ind_space_size, self.whole_space_center, self.voxels_per_axis)
        grid = grid.view(self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2], 3)
        
        # 提取三个正交平面(xy, xz, yz)的2D网格，用于后续的软性坐标回归
        self.register_buffer('center_grid', torch.stack([grid[:, :, 0, :2].reshape(-1, 2),  # xy平面
                                        grid[:, 0, :, ::2].reshape(-1, 2),  # xz平面
                                        grid[0, :, :, 1:].reshape(-1, 2)]))  # yz平面

    def _compute_grid(self, boxSize: torch.Tensor, boxCenter: torch.Tensor, nBins: torch.Tensor):
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
        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0])
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1])
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2])
        
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

    def forward(self, heatmaps: torch.Tensor, fine_sample_grids: torch.Tensor, 
               index: int, proposal_centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数：构建个体特征体素
        
        Args:
            heatmaps: 多视角输入热图，形状为[batch_size, num_views, num_joints, height, width]
            fine_sample_grids: 精细采样网格，形状为[batch_size, num_views, voxels_per_axisx, voxels_per_axisy, voxels_per_axisz, 2]
            index: 批次索引
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
        voxels_x = self.voxels_per_axis[0].item()
        voxels_y = self.voxels_per_axis[1].item()
        voxels_z = self.voxels_per_axis[2].item()
        cubes = torch.zeros(num_people, num_joints, voxels_x, voxels_y, voxels_z, 
                           device=device, dtype=heatmaps.dtype)

        # 计算每个人体的个体体素在全局精细网格中的顶点索引
        # proposal_centers[:, :3]包含人体中心的3D坐标
        centers_tl = torch.round(proposal_centers[:, :3].float() * self.scale + self.bias).int()
        
        # 计算位置偏移，用于将个体体素中的相对坐标转换为全局坐标
        
        offset = centers_tl.float() / (self.fine_voxels_per_axis - 1) * self.whole_space_size - self.whole_space_size / 2.0 + self.ind_space_size / 2.0
        
        # 根据边界框大小创建掩码，过滤体素外的区域
        # proposal_centers[:, 5:7]包含边界框的宽和高
        voxels_xy = torch.stack([self.voxels_per_axis[0], self.voxels_per_axis[1]]) - 1
        mask = ((1 - proposal_centers[:, 5:7]) / 2 * voxels_xy).int()
        mask = torch.where(mask >= 0, mask, torch.zeros_like(mask))
        # 垂直方向(z轴)的边界框长度固定为2000mm
        mask = torch.cat([mask, torch.zeros((num_people, 1), device=device, dtype=torch.int32)], dim=1)

        # 计算有效范围，避免越界
        zeros = torch.zeros_like(centers_tl)
        start = torch.where(centers_tl + mask >= 0, centers_tl + mask, zeros)
        end = torch.where(centers_tl + self.voxels_per_axis - mask <= self.fine_voxels_per_axis, 
                         centers_tl + self.voxels_per_axis - mask, 
                         self.fine_voxels_per_axis)

        # 构建每个人的特征体素
        for i in range(num_people):
            # 如果有效范围无效，则跳过
            if torch.any(start[i] >= end[i]):
                continue
                
            # 提取当前人体有效范围内的采样网格
            start_x = start[i, 0].item()
            start_y = start[i, 1].item()
            start_z = start[i, 2].item()
            end_x = end[i, 0].item()
            end_y = end[i, 1].item()
            end_z = end[i, 2].item()
            
            sample_grid = fine_sample_grids[:, start_x:end_x, start_y:end_y, start_z:end_z].reshape(n, 1, -1, 2)

            # 使用grid_sample从多视角热图中采样特征，然后取平均
            accu_cubes = torch.mean(F.grid_sample(heatmaps[index], sample_grid, align_corners=True), dim=0)
            accu_cubes = accu_cubes.view(num_joints, end_x-start_x, end_y-start_y, end_z-start_z)
            
            # 将采样结果放入对应的位置
            centers_tl_x = centers_tl[i, 0].item()
            centers_tl_y = centers_tl[i, 1].item()
            centers_tl_z = centers_tl[i, 2].item()
            
            x_start = start_x - centers_tl_x
            y_start = start_y - centers_tl_y
            z_start = start_z - centers_tl_z
            x_end = end_x - centers_tl_x
            y_end = end_y - centers_tl_y
            z_end = end_z - centers_tl_z
            
            cubes[i, :, x_start:x_end, y_start:y_end, z_start:z_end] = accu_cubes

        # 裁剪值到[0, 1]范围
        cubes = torch.clamp(cubes, 0.0, 1.0)

        return cubes, offset


# 测试代码
if __name__ == "__main__":
    # 测试参数
    image_size = [256, 256]
    heatmap_size = [64, 64]
    ori_image_size = [1280, 720]
    space_center = [0.0, 0.0, 1000.0]
    space_size = [2000.0, 2000.0, 2000.0]
    individual_space_size = [1000.0, 1000.0, 1000.0]
    voxels_per_axis = [32, 32, 32]
    
    print("=== 测试 ProjectLayerTS 模型 ===")
    
    try:
        # 创建模型实例
        model = ProjectLayerTS(
            image_size=image_size,
            heatmap_size=heatmap_size,
            ori_image_size=ori_image_size,
            space_center=space_center,
            space_size=space_size,
            individual_space_size=individual_space_size,
            voxels_per_axis=voxels_per_axis
        )
        
        print("模型创建成功!")
        
        # 创建测试输入
        batch_size = 2
        num_views = 3
        num_joints = 17
        num_people = 2
        h, w = 64, 64
        
        # 模拟热图输入
        heatmaps = torch.rand(batch_size, num_views, num_joints, h, w)
        
        # 模拟采样网格
        fine_voxels_x = model.fine_voxels_per_axis[0].item()
        fine_voxels_y = model.fine_voxels_per_axis[1].item()
        fine_voxels_z = model.fine_voxels_per_axis[2].item()
        fine_sample_grids = torch.rand(batch_size, num_views, fine_voxels_x, fine_voxels_y, fine_voxels_z, 2) * 2 - 1
        
        # 模拟人体中心位置
        proposal_centers = torch.rand(num_people, 7)
        proposal_centers[:, :3] = torch.tensor([0.0, 0.0, 1000.0]) + torch.randn(num_people, 3) * 100
        proposal_centers[:, 5:7] = torch.rand(num_people, 2) * 0.3 + 0.5  # 边界框大小
        
        print("测试输入准备完成!")
        
        # 测试前向传播
        index = 0  # 批次索引
        outputs = model.forward(heatmaps, fine_sample_grids, index, proposal_centers)
        
        print(f"输出形状:")
        print(f"- cubes: {outputs[0].shape}")
        print(f"- offset: {outputs[1].shape}")
        
        # 尝试导出为 TorchScript
        # 注意：由于模型依赖外部函数 project_pose 和 do_transform，
        # 可能需要将这些函数也转换为 TorchScript 兼容的形式
        print("\n尝试导出为 TorchScript...")
            
        # 尝试使用 torch.jit.script
        try:
            scripted_model = torch.jit.script(model)
            print("模型成功通过 torch.jit.script 导出!")
            
            # 保存模型
            model_path = "project_layer_ts.pt"
            scripted_model.save(model_path)
            print(f"模型已保存到: {model_path}")
            
            # 加载导出的模型
            print("\n加载导出的模型并验证...")
            loaded_model = torch.jit.load(model_path)
            
            # 使用相同的输入测试原始模型和加载的模型
            print("运行原始模型...")
            original_outputs = model.forward(heatmaps, fine_sample_grids, index, proposal_centers)
            
            print("运行转换后的模型...")
            converted_outputs = loaded_model.forward(heatmaps, fine_sample_grids, index, proposal_centers)
            
            # 验证输出是否一致
            cubes_match = torch.allclose(original_outputs[0], converted_outputs[0], rtol=1e-5, atol=1e-5)
            offset_match = torch.allclose(original_outputs[1], converted_outputs[1], rtol=1e-5, atol=1e-5)
            
            print("\n模型输出验证结果:")
            print(f"- cubes 一致性: {'通过' if cubes_match else '失败'}")
            print(f"- offset 一致性: {'通过' if offset_match else '失败'}")
            
            if cubes_match and offset_match:
                print("\n模型已成功导出为 TorchScript 格式并通过验证。")
            else:
                print("\n警告: 转换后的模型输出与原始模型不一致，可能存在兼容性问题。")
            
        except Exception as e:
            print(f"使用 torch.jit.script 导出失败: {e}")
            print("注意: 要完全支持 TorchScript，可能需要修改 project_pose 和 do_transform 函数")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")