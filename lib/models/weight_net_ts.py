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
from typing import List, Tuple


class WeightNetTS(nn.Module):
    """
    权重网络：计算每个关节点的可信度权重
    
    该网络通过分析2D热图特征，为每个关节点分配一个权重值，
    用于在多视角融合时表示该关节点的可信度。
    """
    voxels_per_axis: List[int]
    num_joints: int
    num_channel_joint_feat: int
    num_channel_joint_hidden: int
    
    def __init__(self, voxels_per_axis: List[int], num_joints: int, num_channel_joint_feat: int, num_channel_joint_hidden: int):
        """
        初始化权重网络
        
        Args:
            voxels_per_axis: 每个轴上的体素数量 [x, y, z]
            num_joints: 关节点数量
            num_channel_joint_feat: 关节特征通道数
            num_channel_joint_hidden: 隐藏层通道数
        """
        super(WeightNetTS, self).__init__()
        self.voxels_per_axis = voxels_per_axis
        self.num_joints = num_joints
        self.num_channel_joint_feat = num_channel_joint_feat
        self.num_channel_joint_hidden = num_channel_joint_hidden
        
        # 热图特征提取网络
        self.heatmap_feature_net = nn.Sequential(
            nn.Conv2d(1, self.num_channel_joint_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channel_joint_feat),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        # 输出层：将特征映射到权重值
        self.output = nn.Sequential(
            nn.Linear(self.num_channel_joint_feat, self.num_channel_joint_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_channel_joint_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入热图，形状为 [batch_size, 3, num_joints, voxels_x, voxels_y]
            
        Returns:
            权重值，形状为 [batch_size, num_joints, 1]
        """
        # 将批次和关节维度展平
        x_flat: torch.Tensor = torch.flatten(x, 0, 1)
        batch_size: int = x_flat.shape[0]
        num_joints: int = self.num_joints
        
        # 重塑为卷积网络输入格式
        x_reshaped: torch.Tensor = x_flat.view(batch_size * num_joints, 1, self.voxels_per_axis[0], self.voxels_per_axis[1])
        
        # 通过特征提取网络
        feat: torch.Tensor = self.heatmap_feature_net(x_reshaped)
        
        # 全局平均池化
        feat_pooled: torch.Tensor = F.adaptive_avg_pool2d(feat, 1)
        
        # 展平为全连接层输入
        feat_flat: torch.Tensor = feat_pooled.view(batch_size * num_joints, -1)
        
        # 通过输出层
        weights: torch.Tensor = self.output(feat_flat)
        
        # 重塑为最终输出形状
        weights_reshaped: torch.Tensor = weights.view(batch_size, num_joints, 1)
        
        return weights_reshaped

    def _initialize_weights(self) -> None:
        """
        初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


# 测试代码
if __name__ == "__main__":
    # 测试参数
    voxels_per_axis = [64, 64, 64]
    num_joints = 17
    num_channel_joint_feat = 16
    num_channel_joint_hidden = 8
    
    print("=== 测试 WeightNetTS 模型 ===")
    # 创建模型实例
    model = WeightNetTS(voxels_per_axis, num_joints, num_channel_joint_feat, num_channel_joint_hidden)
    
    # 创建测试输入
    batch_size = 2
    x = torch.rand(batch_size, 3, num_joints, voxels_per_axis[0], voxels_per_axis[1])
    
    # 测试前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    try:
        # 导出为 TorchScript
        model_scripted = torch.jit.script(model)
        
        # 使用导出的模型进行推理
        scripted_output = model_scripted(x)
        
        # 验证结果是否一致
        is_close = torch.allclose(output, scripted_output)
        print(f"原始模型和TorchScript模型输出一致性检查: {'通过' if is_close else '失败'}")
        
        # 保存模型
        model_scripted.save("weight_net_ts.pt")
        print("模型已成功导出到: weight_net_ts.pt")
        
        # 加载模型测试
        loaded_model = torch.jit.load("weight_net_ts.pt")
        loaded_output = loaded_model(x)
        is_close = torch.allclose(output, loaded_output)
        print(f"加载后的模型输出一致性检查: {'通过' if is_close else '失败'}")
        
    except Exception as e:
        print(f"导出模型时出错: {e}")