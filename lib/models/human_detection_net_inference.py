# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from models.cnns_2d import CenterNet
from models.cnns_1d import C2CNet
from models.project_whole_inference import ProjectLayerInference
from core.proposal import nms2D

class ProposalLayerInference(nn.Module):
    """
    候选框层：负责处理和过滤人体检测的候选位置
    
    主要功能：
    1. 将体素坐标转换为实际的3D空间坐标
    2. 在训练时将候选框与真实标注匹配
    3. 过滤置信度较低的候选框
    """
    def __init__(self, cfg):
        """
        初始化候选框层
        
        Args:
            cfg: 配置对象，包含检测参数
        """
        super(ProposalLayerInference, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE  # 最大人数
        self.min_score = cfg.CAPTURE_SPEC.MIN_SCORE  # 最小置信度阈值
        self.device = torch.device(cfg.DEVICE)  # 计算设备

        # 用于将体素坐标转换为实际3D坐标的常量
        self.scale = (torch.tensor(cfg.CAPTURE_SPEC.SPACE_SIZE) / (torch.tensor(cfg.CAPTURE_SPEC.VOXELS_PER_AXIS) - 1)).to(self.device) 
        self.bias = (torch.tensor(cfg.CAPTURE_SPEC.SPACE_CENTER) - torch.tensor(cfg.CAPTURE_SPEC.SPACE_SIZE) / 2.0).to(self.device)

    def filter_proposal(self, topk_index, bbox_preds, gt_3d, gt_bbox, num_person):
        """
        根据真实标注过滤和匹配候选框
        
        Args:
            topk_index: 候选框的3D坐标，形状为(batch_size, max_people, 3)
            bbox_preds: 预测的边界框大小，形状为(batch_size, max_people, 2)
            gt_3d: 真实的3D人体位置，形状为(batch_size, max_people, 3)
            gt_bbox: 真实的边界框大小，形状为(batch_size, max_people, 2)
            num_person: 每个样本中的人数，形状为(batch_size)
            
        Returns:
            proposal2gt: 候选框与真实标注的匹配关系
        """
        batch_size = topk_index.shape[0]
        proposal2gt = torch.zeros(batch_size, self.max_people, device=topk_index.device)
        
        for i in range(batch_size):
            # 重塑张量以便计算距离
            proposals = topk_index[i].reshape(self.max_people, 1, -1) # (max_people, 1, 3)
            gt = gt_3d[i, :num_person[i]].reshape(1, num_person[i], -1) # (1, num_person, 3)
            
            # 计算每个候选框到每个真实标注的欧氏距离
            dist = torch.sqrt(torch.sum((proposals - gt)**2, dim=-1)) # (max_people, num_person)
            
            # 找到每个候选框最近的真实标注
            # min_dist: (max_people)
            # min_gt: (max_people)
            min_dist, min_gt = torch.min(dist, dim=-1)
            proposal2gt[i] = min_gt
            
            # 距离过大的候选框标记为负样本(-1)
            proposal2gt[i][min_dist > 500.0] = -1.0
            
            # 调整边界框预测值，使其接近真实标注
            for k in range(self.max_people):
                threshold = 0.1
                if proposal2gt[i, k] < 0:
                    continue
                if torch.sum(bbox_preds[i, k] < gt_bbox[i, proposal2gt[i, k].long()] - threshold): # 如果候选框的边界框预测值小于真实标注的边界框值
                    bbox_preds[i, k] = gt_bbox[i, proposal2gt[i, k].long()]
        return proposal2gt

    def forward(self, topk_index, topk_confs, match_bbox_preds):
        """
        前向传播函数
        
        Args:
            topk_index: 候选框的体素坐标，形状为(batch_size, max_people, 3)
            topk_confs: 候选框的置信度，形状为(batch_size, max_people)
            match_bbox_preds: 匹配的边界框预测，形状为(batch_size, max_people, 2)
            
        Returns:
            proposal_centers: 包含位置、置信度、匹配关系和边界框的完整候选框信息，形状为[batch_size, max_people, 7]
                其中7个值分别为：3D中心坐标 (x,y,z)、匹配的真实标注索引、置信度、边界框尺寸预测 (宽,高)
                如果meta中包含root_3d和bbox，则使用真实标注过滤和匹配候选框，否则使用置信度阈值过滤候选框
        """
        device = topk_index.device
        batch_size = topk_index.shape[0]

        # 将体素坐标转换为实际3D坐标
        topk_index = topk_index.float() * self.scale + self.bias

        # 初始化候选框中心点信息，包含7个值：
        # 0-2：3D中心坐标 (x,y,z)
        # 3：匹配的真实标注索引
        # 4：置信度
        # 5-6：边界框预测 (宽,高)
        proposal_centers = torch.zeros(batch_size, self.max_people, 7, device=device)
        proposal_centers[:, :, 0:3] = topk_index
        proposal_centers[:, :, 4] = topk_confs

        # 测试阶段：根据置信度阈值过滤候选框
        proposal_centers[:, :, 3] = (topk_confs > self.min_score).float() - 1.0  # if ground-truths are not available.
        
        # 添加边界框预测
        proposal_centers[: ,:, 5:7] = match_bbox_preds
        return proposal_centers

class HumanDetectionNetInference(nn.Module):
    """
    人体检测网络：检测3D空间中的人体位置
    
    主要组件：
    1. 投影层：将2D热图投影到3D体素空间
    2. CenterNet：2D人体中心点检测网络
    3. C2CNet：1D高度检测网络
    4. 候选框层：处理和过滤候选框
    
    工作流程：
    1. 将多视角2D热图投影到3D体素空间
    2. 使用2D CNN检测人体中心点的平面位置(x,y)
    3. 使用1D CNN检测人体的高度(z)
    4. 组合(x,y,z)得到完整的3D人体位置
    """
    def __init__(self, cfg):
        """
        初始化人体检测网络
        
        Args:
            cfg: 配置对象
        """
        super(HumanDetectionNetInference, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE  # 最大人数
        self.project_layer = ProjectLayerInference(cfg)  # 投影层
        self.center_net = CenterNet(cfg.DATASET.NUM_JOINTS, 1)  # 2D中心点检测网络
        self.c2c_net = C2CNet(cfg.DATASET.NUM_JOINTS, 1)  # 1D高度检测网络
        self.proposal_layer = ProposalLayerInference(cfg)  # 候选框处理层
        
    def forward(self, heatmaps, cam_params, sample_grid, resize_transform):
        """
        前向传播函数
        
        Args:
            heatmaps: 多视角的2D关节热图
            cam_params: 相机参数
            resize_transform: 图像缩放变换矩阵
            
        Returns:
            proposal_centers: 候选人体中心点信息，形状为[batch_size, max_proposals, 7]
                其中7个值分别为：3D中心坐标 (x,y,z)、匹配的真实标注索引、置信度、边界框尺寸预测 (宽,高)
        """
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[2]
        
        # 步骤1：构建3D特征体素
        # (batch_size, num_joints, height, width, depth) 
        feature_cubes = self.project_layer(heatmaps, sample_grid)                                 
        
        # 步骤2：生成2D平面上的候选框
        # proposal_heatmaps_2d: (batch_size, 1, height, width)
        # bbox_preds: (batch_size, 2, height, width)
        proposal_heatmaps_2d, bbox_preds = self.center_net(feature_cubes)
        # topk_2d_confs: (batch_size, max_people, 1)
        # topk_2d_index: (batch_size, max_people, 2)
        # topk_2d_flatten_index: (batch_size, max_people)
        topk_2d_confs, topk_2d_index, topk_2d_flatten_index = nms2D(proposal_heatmaps_2d.detach(), self.max_people) 
        
        # 步骤3：提取匹配的边界框预测
        bbox_preds = torch.flatten(bbox_preds, 2, 3).permute(0, 2, 1) # (batch_size, height * width, 2)
        match_bbox_preds = torch.gather(bbox_preds, dim=1, index=topk_2d_flatten_index.unsqueeze(2).repeat(1, 1, 2)) # (batch_size, max_people, 2)
        
        # 步骤4：提取匹配的1D特征并送入1D CNN
        # (batch_size, num_joints, width, height, depth) 
        # -> (batch_size, num_joints, width * height, depth)
        # -> (batch_size, width * height, num_joints, depth)
        # -> (batch_size, max_people, num_joints, depth)
        feature_1d = torch.gather(torch.flatten(feature_cubes, 2, 3).permute(0, 2, 1, 3), dim=1,\
                                  index=topk_2d_flatten_index.view(batch_size, -1, 1, 1).repeat(1, 1, num_joints, feature_cubes.shape[4]))
        # (batch_size, max_people, num_joints, depth)
        # -> (batch_size * max_people, num_joints, depth)
        # -> (batch_size * max_people, 1, depth)
        # -> (batch_size, max_people, depth)
        proposal_heatmaps_1d = self.c2c_net(torch.flatten(feature_1d, 0, 1)).view(batch_size, self.max_people, -1)
        # topk_1d_confs: (batch_size, max_people, 1)
        # topk_1d_index: (batch_size, max_people, 1)
        topk_1d_confs, topk_1d_index = proposal_heatmaps_1d.detach().topk(1)

        # 步骤5：组装最终预测结果
        # 组合(x,y)和z坐标得到完整的3D坐标
        topk_index = torch.cat([topk_2d_index, topk_1d_index], dim=2) # (batch_size, max_people, 3)
        
        # 计算最终置信度：2D置信度 × 1D置信度
        topk_confs = topk_2d_confs * topk_1d_confs.squeeze(2) # (batch_size, max_people)
        
        # 通过候选框层处理最终的候选人体位置
        proposal_centers = self.proposal_layer(topk_index, topk_confs, match_bbox_preds)

        return proposal_heatmaps_2d, proposal_heatmaps_1d, proposal_centers, bbox_preds