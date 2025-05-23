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
from models.human_detection_net import HumanDetectionNet
from models.joint_localization_net import JointLocalizationNet

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
        self.pose_net = HumanDetectionNet(cfg)  # 人体检测网络
        self.joint_net = JointLocalizationNet(cfg)  # 关节定位网络

        # 各个损失函数的权重
        self.lambda_loss_2d = cfg.TRAIN.LAMBDA_LOSS_2D  # 2D热图损失权重
        self.lambda_loss_1d = cfg.TRAIN.LAMBDA_LOSS_1D  # 1D热图损失权重
        self.lambda_loss_bbox = cfg.TRAIN.LAMBDA_LOSS_BBOX  # 边界框损失权重
        self.lambda_loss_fused = cfg.TRAIN.LAMBDA_LOSS_FUSED  # 融合姿态损失权重


    def forward(self, backbone=None, views=None, meta=None, targets=None, input_heatmaps=None, cameras=None, resize_transform=None):
        """
        前向传播函数
        
        Args:
            backbone: 主干特征提取网络
            views: 多视角RGB图像，形状为[batch_size, num_views, channels, height, width]
            meta: 元数据，包含真实3D关节位置等信息
            targets: 目标热图和索引
            input_heatmaps: 预计算的输入热图（如果为None，则从views通过backbone计算）
            cameras: 相机参数
            resize_transform: 图像缩放变换矩阵
            
        Returns:
            fused_poses: 融合后的3D关节位置，形状为[batch_size, max_proposals, num_joints, 3]
            plane_poses: 三个正交平面上的关节投影，形状为[3, batch_size, max_proposals, num_joints, 2]
            proposal_centers: 人体中心位置，形状为[batch_size, max_proposals, 7]
                其中7个值分别为：3D中心坐标 (x,y,z)、匹配的真实标注索引、置信度、边界框尺寸预测 (宽,高)
            input_heatmaps: 输入热图，形状为[batch_size, num_views, num_joints, height, width]
            loss_dict: 损失字典（训练模式）
        """
        # 如果提供了RGB图像，使用backbone网络从图像生成热图
        if views is not None:
            num_views = views.shape[1]
            input_heatmaps = torch.stack([backbone(views[:, c]) for c in range(num_views)], dim=1)
            # print(input_heatmaps.shape, views.shape)
        
        batch_size = input_heatmaps.shape[0]
 
        # 使用人体检测网络检测人体位置
        # proposal_heatmaps_2d: 2D候选热图
        # proposal_heatmaps_1d: 1D候选热图
        # proposal_centers: 候选人体中心点信息
        # bbox_preds: 边界框预测
        proposal_heatmaps_2d, proposal_heatmaps_1d, proposal_centers, \
                              bbox_preds = self.pose_net(input_heatmaps, meta, cameras, resize_transform)
        # 创建有效提案的掩码（proposal_centers[:, :, 3]>=0表示该提案匹配到了真实人体）
        mask = (proposal_centers[:, :, 3] >= 0)

        # 使用关节定位网络预测关节位置
        # fused_poses: 融合后的3D关节位置
        # plane_poses: 三个正交平面(xy, xz, yz)上的关节投影
        fused_poses, plane_poses = self.joint_net(meta, input_heatmaps, proposal_centers.detach(), mask, cameras, resize_transform)

        # 训练阶段：计算损失函数
        if self.training:
            assert targets is not None, 'proposal ground truth not set'
            # 获取proposal到ground truth的匹配关系
            proposal2gt = proposal_centers[:, :, 3]
            proposal2gt = torch.where(proposal2gt >= 0, proposal2gt, torch.zeros_like(proposal2gt))

            # 计算2D热图损失：预测热图与真实热图的MSE损失
            loss_2d = self.lambda_loss_2d * F.mse_loss(proposal_heatmaps_2d[:, 0], targets['2d_heatmaps'], reduction='mean')
            
            # 计算1D热图损失：提取匹配的1D目标热图，计算MSE损失
            matched_heatmaps_1d = torch.gather(targets['1d_heatmaps'], dim=1, index=proposal2gt.long()\
                                               .unsqueeze(2).repeat(1, 1, proposal_heatmaps_1d.shape[2]))
            loss_1d = self.lambda_loss_1d * F.mse_loss(proposal_heatmaps_1d[mask], matched_heatmaps_1d[mask], reduction='mean')
            
            # 计算边界框回归损失：仅对匹配到真实人体的位置计算L1损失
            bbox_preds = torch.gather(bbox_preds, 1, targets['index'].long().view(batch_size, -1, 1).repeat(1, 1, 2))
            loss_bbox = self.lambda_loss_bbox * F.l1_loss(bbox_preds[targets['mask']], targets['bbox'][targets['mask']], reduction='mean')

            # 释放不再需要的变量，节省内存
            del proposal_heatmaps_2d, proposal_heatmaps_1d, bbox_preds
            
            # 如果没有有效提案，返回只有检测损失的结果
            if torch.sum(mask) == 0:  # no valid proposals
                loss_dict = {
                    "2d_heatmaps": loss_2d,
                    "1d_heatmaps": loss_1d,
                    "bbox": loss_bbox,
                    "joint": torch.zeros(1, device=input_heatmaps.device),
                    "total": loss_2d + loss_1d + loss_bbox
                }
                return None, None, proposal_centers, input_heatmaps, loss_dict
                
            # 计算关节定位损失：加权L1损失
            # 1. 提取匹配的真实3D关节位置和可见性标志
            gt_joints_3d = meta['joints_3d'].float().to(self.device)
            gt_joints_3d_vis = meta['joints_3d_vis'].float().to(self.device)
            joints_3d = torch.gather(gt_joints_3d, dim=1, index=proposal2gt.long().view\
                                     (batch_size, -1, 1, 1).repeat(1, 1, self.num_joints, 3))[mask]
            joints_vis = torch.gather(gt_joints_3d_vis, dim=1, index=proposal2gt.long().view\
                                     (batch_size, -1, 1).repeat(1, 1, self.num_joints))[mask].unsqueeze(2)
            
            # 2. 计算三个平面投影和融合姿态的L1损失，仅考虑可见关节
            # plane_poses[0]: xy平面投影
            # plane_poses[1]: xz平面投影
            # plane_poses[2]: yz平面投影
            loss_joint = F.l1_loss(plane_poses[0][mask] * joints_vis, joints_3d[:, :, :2] * joints_vis, reduction="mean") +\
                         F.l1_loss(plane_poses[1][mask] * joints_vis, joints_3d[:, :, ::2] * joints_vis, reduction="mean") +\
                         F.l1_loss(plane_poses[2][mask] * joints_vis, joints_3d[:, :, 1:] * joints_vis, reduction="mean") +\
                         self.lambda_loss_fused * F.l1_loss(fused_poses[mask] * joints_vis, joints_3d * joints_vis, reduction="mean")

            # 构建损失字典
            loss_dict = {
                "2d_heatmaps": loss_2d,
                "1d_heatmaps": loss_1d,
                "bbox": loss_bbox,
                "joint": loss_joint,
                "total": loss_2d + loss_1d + loss_bbox + loss_joint
            }
        else:
            loss_dict = None

        # 将置信度和匹配索引信息拼接到融合姿态结果中
        # proposal_centers[:, :, 3:5]包含匹配索引和置信度
        fused_poses = torch.cat([fused_poses, proposal_centers[:, :, 3:5].reshape(batch_size,\
                                 -1, 1, 2).repeat(1, 1, self.num_joints, 1)], dim=3)

        return fused_poses, plane_poses, proposal_centers, input_heatmaps, loss_dict


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