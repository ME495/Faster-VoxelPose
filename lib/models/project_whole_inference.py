# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cameras import project_pose
from utils.transforms import affine_transform_pts_cuda as do_transform

class ProjectLayerInference(nn.Module):
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
        super(ProjectLayerInference, self).__init__()
        self.device = torch.device(cfg.DEVICE)  # 计算设备（CPU或GPU）
        self.image_size = cfg.DATASET.IMAGE_SIZE  # 输入图像尺寸
        self.heatmap_size = cfg.DATASET.HEATMAP_SIZE  # 热图尺寸
        self.ori_image_size = cfg.DATASET.ORI_IMAGE_SIZE  # 原始图像尺寸

        # 3D空间参数设置
        self.space_size = cfg.CAPTURE_SPEC.SPACE_SIZE  # 3D空间的大小
        self.space_center = cfg.CAPTURE_SPEC.SPACE_CENTER  # 3D空间的中心点
        self.voxels_per_axis = cfg.CAPTURE_SPEC.VOXELS_PER_AXIS  # 每个轴上的体素数量
        
    def forward(self, heatmaps, sample_grid):
        """
        前向传播函数：将多视角2D热图投影到3D体素空间
        
        Args:
            heatmaps: 2D热图，形状为(batch_size, num_views, num_joints, h, w)
            cam_params: 相机参数字典
            resize_transform: 图像缩放变换矩阵
            
        Returns:
            cubes: 3D体素表示，形状为(batch_size, num_joints, width, height, depth)
        """
        device = heatmaps.device
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[2]  # 关节数量
        nbins = self.voxels_per_axis[0] * self.voxels_per_axis[1] * self.voxels_per_axis[2]  # 体素总数
        
        # 初始化输出体素
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, device=device)
        
        for i in range(batch_size):
            # 对每个视角的热图进行网格采样，然后求平均
            cubes[i] = torch.mean(F.grid_sample(heatmaps[i], sample_grid[i], align_corners=True), dim=0).squeeze(0)
            
        # 裁剪值到[0, 1]范围，并重塑为最终的体素形状
        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_joints, self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2]) 
        return cubes