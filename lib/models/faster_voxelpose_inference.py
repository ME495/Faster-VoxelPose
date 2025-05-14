# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet
from models.human_detection_net_inference import HumanDetectionNetInference
from models.joint_localization_net_inference import JointLocalizationNetInference

class FasterVoxelPoseNet(nn.Module):
    """
    Faster VoxelPose 网络：用于多视角3D人体姿态估计的端到端模型
    
    主要包含两个子网络：
    1. 人体检测网络(HumanDetectionNet)：检测3D空间中的人体位置
    2. 关节定位网络(JointLocalizationNet)：预测每个人体的关节位置
    
    工作流程：输入多视角图像 -> 提取2D热图 -> 检测人体位置 -> 定位关节点 -> 输出3D姿态
    """
    def __init__(self, cfg):
        """
        初始化Faster VoxelPose网络
        
        Args:
            cfg: 配置对象，包含网络参数和训练参数
        """
        super(FasterVoxelPoseNet, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE  # 最大人数
        self.num_joints = cfg.DATASET.NUM_JOINTS  # 关节点数量
        self.device = torch.device(cfg.DEVICE)  # 计算设备
       
        # 初始化两个主要子网络
        self.pose_net = HumanDetectionNetInference(cfg)  # 人体检测网络
        self.joint_net = JointLocalizationNetInference(cfg)  # 关节定位网络


    def forward(self, input_heatmaps, cam_params, sample_grid, resize_transform):
        """
        前向传播函数
        
        Args:
            input_heatmaps: 预计算的输入热图（如果为None，则从views通过backbone计算）
            cam_params: 相机参数
            resize_transform: 图像缩放变换矩阵
            
        Returns:
            fused_poses: 融合后的3D关节位置，形状为[batch_size, max_proposals, num_joints, 3]
        """
        batch_size = input_heatmaps.shape[0]
 
        # 使用人体检测网络检测人体位置
        # proposal_centers: 候选人体中心点信息
        proposal_centers = self.pose_net(input_heatmaps, cam_params, sample_grid, resize_transform)
        # 创建有效提案的掩码（proposal_centers[:, :, 3]>=0表示该提案匹配到了真实人体）
        mask = (proposal_centers[:, :, 3] >= 0)

        # 使用关节定位网络预测关节位置
        # fused_poses: 融合后的3D关节位置
        fused_poses = self.joint_net(input_heatmaps, proposal_centers.detach(), mask, cam_params, resize_transform)

        # 将置信度和匹配索引信息拼接到融合姿态结果中
        fused_poses = torch.cat([fused_poses, proposal_centers[:, :, 3:5].reshape(batch_size,\
                                 -1, 1, 2).repeat(1, 1, self.num_joints, 1)], dim=3)

        return fused_poses


def get(cfg):
    """
    创建Faster VoxelPose模型的工厂函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        model: 初始化后的模型
    """
    model = FasterVoxelPoseNet(cfg)
    return model