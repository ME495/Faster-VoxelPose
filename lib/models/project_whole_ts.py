# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ProjectLayerTS(nn.Module):
    """
    投影层：将2D热图投影到3D体素空间
    
    该层的主要功能是将多个相机视角下的2D关节热图投影并融合成3D体素表示。
    它将体素网格中的每个体素投影到每个相机视角下的2D平面上，
    通过采样2D热图的相应位置来填充3D体素。
    """
    voxels_per_axis: List[int]
    
    def __init__(self, voxels_per_axis: List[int]):
        """
        初始化投影层
        
        Args:
            voxels_per_axis: 每个轴上的体素数量
        """
        super(ProjectLayerTS, self).__init__()
        self.voxels_per_axis = voxels_per_axis  # 每个轴上的体素数量

        
    def forward(self, heatmaps: torch.Tensor, sample_grids: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数：将多视角2D热图投影到3D体素空间
        
        Args:
            heatmaps: 2D热图，形状为(batch_size, num_views, num_joints, h, w)
            sample_grids: 采样网格，形状为(batch_size, num_views, 1, nbins, 2)
            
        Returns:
            cubes: 3D体素表示，形状为(batch_size, num_joints, width, height, depth)
        """
        batch_size: int = heatmaps.shape[0]
        n: int = heatmaps.shape[1]  # 视角数量
        num_joints: int = heatmaps.shape[2]  # 关节数量
        h: int = int(heatmaps.shape[3])
        w: int = int(heatmaps.shape[4])
        
        # 计算体素总数
        nbins: int = self.voxels_per_axis[0] * self.voxels_per_axis[1] * self.voxels_per_axis[2]
        
        # 向量化操作：直接对整个批次进行网格采样
        heatmaps_reshaped: torch.Tensor = heatmaps.reshape(batch_size*n, num_joints, h, w)
        sample_grids_reshaped: torch.Tensor = sample_grids.reshape(batch_size*n, 1, nbins, 2)
        
        sampled: torch.Tensor = F.grid_sample(
            heatmaps_reshaped, 
            sample_grids_reshaped, 
            align_corners=True
        )
        
        sampled = sampled.view(batch_size, n, num_joints, 1, nbins)
        cubes: torch.Tensor = torch.mean(sampled, dim=1)  # 对视角维度求平均
        
        # 裁剪值到[0, 1]范围，并重塑为最终的体素形状
        cubes = torch.clamp(cubes, 0.0, 1.0)
        cubes = cubes.view(
            batch_size, 
            num_joints, 
            self.voxels_per_axis[0], 
            self.voxels_per_axis[1], 
            self.voxels_per_axis[2]
        ) 
        return cubes


if __name__ == "__main__":
    # 测试代码
    voxels_per_axis = [16, 16, 16]
    model = ProjectLayerTS(voxels_per_axis)
    heatmaps = torch.randn(2, 3, 17, 17, 17)
    sample_grids = torch.randn(2, 3, 1, 16*16*16, 2)
    model_scripted = torch.jit.script(model)
    cubes = model_scripted(heatmaps, sample_grids)
    print(cubes.shape)
    
