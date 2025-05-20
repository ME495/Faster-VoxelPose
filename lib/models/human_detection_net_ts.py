# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

# 测试时需要添加
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from models.cnns_2d import CenterNet
from models.cnns_1d import C2CNet
from models.project_whole_ts import ProjectLayerTS

# 将 nms2D 相关函数直接集成到模块中，以便 TorchScript 兼容
def get_index2D(indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    将一维展平索引转换为二维坐标索引
    
    Args:
        indices: 展平后的一维索引, 形状为 [batch_size, num_people]
        height: 原始2D热图的高度
        width: 原始2D热图的宽度
        
    Returns:
        二维坐标索引, 形状为 [batch_size, num_people, 2]，每个点包含 [x, y] 坐标
    """
    batch_size = indices.shape[0]
    num_people = indices.shape[1]
    # 计算x坐标（行索引）：通过整除得到
    indices_x = torch.div(indices, width, rounding_mode='trunc').reshape(batch_size, num_people, -1)
    # 计算y坐标（列索引）：通过取余得到
    indices_y = (indices % width).reshape(batch_size, num_people, -1)
    # 拼接x和y坐标
    indices = torch.cat([indices_x, indices_y], dim=2)
    return indices

def max_pool2D(inputs: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    执行最大池化操作，用于非极大值抑制
    
    通过与局部最大值比较，保留局部最大值点，抑制非最大值点
    
    Args:
        inputs: 输入热图，形状为 [batch_size, channels, height, width]
        kernel: 池化核大小，默认为3
        
    Returns:
        经过非极大值抑制后的热图，只保留局部最大值点
    """
    padding = (kernel - 1) // 2  # 计算填充大小，保持输出尺寸不变
    # 执行最大池化，得到每个点的局部最大值
    max_vals = F.max_pool2d(inputs, kernel_size=kernel, stride=1, padding=padding)
    # 比较原始值和局部最大值，只保留等于局部最大值的点（即局部最大值点）
    keep = (inputs == max_vals).float()
    return keep * inputs  # 只保留局部最大值点的原始值

def nms2D_ts(prob_map: torch.Tensor, max_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对概率热图执行2D非极大值抑制并选取前K个点 (TorchScript兼容版本)
    
    Args:
        prob_map: 概率热图，形状为 [batch_size, channels, height, width]
        max_num: 需要选取的最大点数
        
    Returns:
        topk_values: 前K个点的概率值，形状为 [batch_size, max_num]
        topk_index: 前K个点的2D坐标索引，形状为 [batch_size, max_num, 2]
        topk_flatten_index: 前K个点的1D展平索引，形状为 [batch_size, max_num]
    """
    batch_size = prob_map.shape[0]
    height = prob_map.shape[2]
    width = prob_map.shape[3]
    
    # 步骤1: 执行非极大值抑制，只保留局部最大值点
    prob_map_nms = max_pool2D(prob_map)
    # 步骤2: 将热图展平为1D张量
    prob_map_nms_reshape = prob_map_nms.reshape(batch_size, -1)
    # 步骤3: 选取前K个最大值及其索引
    topk_values, topk_flatten_index = prob_map_nms_reshape.topk(max_num)
    # 步骤4: 将1D索引转换回2D坐标
    topk_index = get_index2D(topk_flatten_index, height, width)
    return topk_values, topk_index, topk_flatten_index


class ProposalLayerTS(nn.Module):
    """
    候选框层：负责处理和过滤人体检测的候选位置
    
    主要功能：
    1. 将体素坐标转换为实际的3D空间坐标
    2. 在训练时将候选框与真实标注匹配
    3. 过滤置信度较低的候选框
    """
    max_people: int
    min_score: float
    scale: torch.Tensor
    bias: torch.Tensor
    
    def __init__(self, max_people: int, min_score: float, space_size: List[float], voxels_per_axis: List[int], space_center: List[float]):
        """
        初始化候选框层
        
        Args:
            max_people: 最大人数
            min_score: 最小置信度阈值
            space_size: 3D空间大小 [x, y, z]
            voxels_per_axis: 每个轴上的体素数量 [x, y, z]
            space_center: 3D空间中心点 [x, y, z]
        """
        super(ProposalLayerTS, self).__init__()
        self.max_people = max_people  # 最大人数
        self.min_score = min_score  # 最小置信度阈值

        # 用于将体素坐标转换为实际3D坐标的常量
        self.register_buffer('scale', torch.tensor(space_size) / (torch.tensor(voxels_per_axis) - 1.0))
        self.register_buffer('bias', torch.tensor(space_center) - torch.tensor(space_size) / 2.0)

    def forward(self, topk_index: torch.Tensor, topk_confs: torch.Tensor, match_bbox_preds: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            topk_index: 候选框的体素坐标，形状为(batch_size, max_people, 3)
            topk_confs: 候选框的置信度，形状为(batch_size, max_people)
            match_bbox_preds: 匹配的边界框预测，形状为(batch_size, max_people, 2)
            
        Returns:
            proposal_centers: 包含位置、置信度、匹配关系和边界框的完整候选框信息，形状为[batch_size, max_people, 7]
                其中7个值分别为：3D中心坐标 (x,y,z)、匹配的真实标注索引、置信度、边界框尺寸预测 (宽,高)
        """
        device: torch.device = topk_index.device
        batch_size: int = topk_index.shape[0]

        # 将体素坐标转换为实际3D坐标
        topk_index_float: torch.Tensor = topk_index.float()
        scale_device: torch.Tensor = self.scale.to(device)
        bias_device: torch.Tensor = self.bias.to(device)
        
        # 执行坐标转换
        topk_index_transformed: torch.Tensor = topk_index_float * scale_device + bias_device

        # 初始化候选框中心点信息，包含7个值：
        # 0-2：3D中心坐标 (x,y,z)
        # 3：匹配的真实标注索引
        # 4：置信度
        # 5-6：边界框预测 (宽,高)
        proposal_centers: torch.Tensor = torch.zeros(batch_size, self.max_people, 7, device=device)
        proposal_centers[:, :, 0:3] = topk_index_transformed
        proposal_centers[:, :, 4] = topk_confs

        # 根据置信度阈值过滤候选框
        mask: torch.Tensor = (topk_confs > self.min_score).float() - 1.0
        proposal_centers[:, :, 3] = mask
        
        # 添加边界框预测
        proposal_centers[:, :, 5:7] = match_bbox_preds
        return proposal_centers


class HumanDetectionNetTS(nn.Module):
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
    max_people: int
    
    def __init__(self, max_people: int, min_score: float, space_size: List[float], 
                 voxels_per_axis: List[int], space_center: List[float], num_joints: int):
        """
        初始化人体检测网络
        
        Args:
            max_people: 最大人数
            min_score: 最小置信度阈值
            space_size: 3D空间大小 [x, y, z]
            voxels_per_axis: 每个轴上的体素数量 [x, y, z]
            space_center: 3D空间中心点 [x, y, z]
            num_joints: 关节数量
        """
        super(HumanDetectionNetTS, self).__init__()
        self.max_people = max_people  # 最大人数
        self.project_layer = ProjectLayerTS(voxels_per_axis)  # 投影层
        self.center_net = CenterNet(num_joints, 1)  # 2D中心点检测网络
        self.c2c_net = C2CNet(num_joints, 1)  # 1D高度检测网络
        self.proposal_layer = ProposalLayerTS(max_people, min_score, space_size, voxels_per_axis, space_center)  # 候选框处理层
        
    def forward(self, heatmaps: torch.Tensor, sample_grids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播函数
        
        Args:
            heatmaps: 多视角的2D关节热图，形状为(batch_size, num_views, num_joints, h, w)
            sample_grids: 采样网格，形状为(batch_size, num_views, voxels_x, voxels_y, voxels_z, 2)
            
        Returns:
            proposal_heatmaps_2d: 2D候选热图，形状为[batch_size, 1, height, width]
            proposal_heatmaps_1d: 1D候选热图，形状为[batch_size, max_people, depth]
            proposal_centers: 候选人体中心点信息，形状为[batch_size, max_proposals, 7]
                其中7个值分别为：3D中心坐标 (x,y,z)、匹配的真实标注索引、置信度、边界框尺寸预测 (宽,高)
            bbox_preds: 过滤前边界框尺寸预测，形状为[batch_size, 2, height, width]
        """
        batch_size: int = heatmaps.shape[0]
        num_views: int = heatmaps.shape[1]
        num_joints: int = heatmaps.shape[2]
        
        # 步骤1：构建3D特征体素
        # (batch_size, num_joints, height, width, depth) 
        feature_cubes: torch.Tensor = self.project_layer(heatmaps, sample_grids.view(batch_size, num_views, 1, -1, 2))                                 
        
        # 步骤2：生成2D平面上的候选框
        # proposal_heatmaps_2d: (batch_size, 1, height, width)
        # bbox_preds: (batch_size, 2, height, width)
        proposal_heatmaps_2d, bbox_preds = self.center_net(feature_cubes)
        
        # 获取热图尺寸
        height: int = proposal_heatmaps_2d.shape[2]
        width: int = proposal_heatmaps_2d.shape[3]
        
        # 使用TorchScript兼容版本的nms2D
        # topk_2d_confs: (batch_size, max_people, 1)
        # topk_2d_index: (batch_size, max_people, 2)
        # topk_2d_flatten_index: (batch_size, max_people)
        topk_2d_confs, topk_2d_index, topk_2d_flatten_index = nms2D_ts(proposal_heatmaps_2d.detach(), self.max_people) 
        
        # 步骤3：提取匹配的边界框预测
        bbox_preds_flat: torch.Tensor = torch.flatten(bbox_preds, 2, 3).permute(0, 2, 1) # (batch_size, height * width, 2)
        repeat_index: torch.Tensor = topk_2d_flatten_index.unsqueeze(2).repeat(1, 1, 2)
        match_bbox_preds: torch.Tensor = torch.gather(bbox_preds_flat, dim=1, index=repeat_index) # (batch_size, max_people, 2)
        
        # 步骤4：提取匹配的1D特征并送入1D CNN
        # (batch_size, num_joints, width, height, depth) 
        # -> (batch_size, num_joints, width * height, depth)
        # -> (batch_size, width * height, num_joints, depth)
        # -> (batch_size, max_people, num_joints, depth)
        feature_cubes_flat: torch.Tensor = torch.flatten(feature_cubes, 2, 3).permute(0, 2, 1, 3)
        depth: int = feature_cubes.shape[4]
        repeat_index_1d: torch.Tensor = topk_2d_flatten_index.view(batch_size, -1, 1, 1).repeat(1, 1, num_joints, depth)
        feature_1d: torch.Tensor = torch.gather(feature_cubes_flat, dim=1, index=repeat_index_1d)
        
        # (batch_size, max_people, num_joints, depth)
        # -> (batch_size * max_people, num_joints, depth)
        # -> (batch_size * max_people, 1, depth)
        # -> (batch_size, max_people, depth)
        feature_1d_flat: torch.Tensor = torch.flatten(feature_1d, 0, 1)
        proposal_heatmaps_1d_flat: torch.Tensor = self.c2c_net(feature_1d_flat)
        proposal_heatmaps_1d: torch.Tensor = proposal_heatmaps_1d_flat.view(batch_size, self.max_people, -1)
        
        # topk_1d_confs: (batch_size, max_people, 1)
        # topk_1d_index: (batch_size, max_people, 1)
        topk_1d_confs, topk_1d_index = proposal_heatmaps_1d.detach().topk(1)

        # 步骤5：组装最终预测结果
        # 组合(x,y)和z坐标得到完整的3D坐标
        topk_index: torch.Tensor = torch.cat([topk_2d_index, topk_1d_index], dim=2) # (batch_size, max_people, 3)
        
        # 计算最终置信度：2D置信度 × 1D置信度
        topk_confs: torch.Tensor = topk_2d_confs * topk_1d_confs.squeeze(2) # (batch_size, max_people)
        
        # 通过候选框层处理最终的候选人体位置
        proposal_centers: torch.Tensor = self.proposal_layer(topk_index, topk_confs, match_bbox_preds)

        return proposal_heatmaps_2d, proposal_heatmaps_1d, proposal_centers, bbox_preds


# 导出TorchScript模型的辅助函数
def export_torchscript_model(model: torch.nn.Module, model_path: str) -> None:
    """
    将模型导出为TorchScript格式
    
    Args:
        model: 要导出的模型
        model_path: 保存路径
    """
    try:
        # 使用torch.jit.script导出模型
        scripted_model = torch.jit.script(model)
        scripted_model.save(model_path)
        print(f"模型已成功导出到: {model_path}")
        return True
    except Exception as e:
        print(f"导出模型时出错: {e}")
        return False


if __name__ == "__main__":
    # 测试参数
    max_people = 10
    min_score = 0.3
    space_size = [2000.0, 2000.0, 2000.0]
    voxels_per_axis = [64, 64, 64]
    space_center = [0.0, 0.0, 1000.0]
    num_joints = 17
    
    print("=== 测试 ProposalLayerTS 模型 ===")
    # 创建 ProposalLayerTS 模型实例
    proposal_model = ProposalLayerTS(max_people, min_score, space_size, voxels_per_axis, space_center)
    
    # 创建测试输入
    batch_size = 2
    topk_index = torch.randint(0, 64, (batch_size, max_people, 3))
    topk_confs = torch.rand(batch_size, max_people)
    match_bbox_preds = torch.rand(batch_size, max_people, 2)
    
    # 导出为 TorchScript
    proposal_model_scripted = torch.jit.script(proposal_model)
    
    # 使用导出的模型进行推理
    proposal_centers_scripted = proposal_model_scripted(topk_index, topk_confs, match_bbox_preds)
    
    print(f"ProposalLayerTS 输出形状: {proposal_centers_scripted.shape}")
    print("ProposalLayerTS 模型导出成功!")
    
    print("\n=== 测试 HumanDetectionNetTS 模型 ===")
    try:
        # 创建 HumanDetectionNetTS 模型实例
        detection_model = HumanDetectionNetTS(max_people, min_score, space_size, voxels_per_axis, space_center, num_joints)
        
        # 创建测试输入
        num_views = 3
        h, w = 64, 64
        
        heatmaps = torch.rand(batch_size, num_views, num_joints, h, w)
        sample_grids = torch.rand(batch_size, num_views, voxels_per_axis[0], voxels_per_axis[1], voxels_per_axis[2], 2) * 2 - 1  # 归一化到[-1, 1]范围
        
        # 测试前向传播
        outputs = detection_model(heatmaps, sample_grids)
        print(f"HumanDetectionNetTS 输出形状:")
        print(f"- proposal_heatmaps_2d: {outputs[0].shape}")
        print(f"- proposal_heatmaps_1d: {outputs[1].shape}")
        print(f"- proposal_centers: {outputs[2].shape}")
        print(f"- bbox_preds: {outputs[3].shape}")
        
        # 导出为 TorchScript
        export_success = export_torchscript_model(detection_model, "human_detection_net_ts.pt")
        
        if export_success:
            # 加载导出的模型
            loaded_model = torch.jit.load("human_detection_net_ts.pt")
            
            # 使用导出的模型进行推理
            scripted_outputs = loaded_model(heatmaps, sample_grids)
            
            # 验证结果是否一致
            for i in range(len(outputs)):
                is_close = torch.allclose(outputs[i], scripted_outputs[i])
                print(f"输出 {i} 一致性检查: {'通过' if is_close else '失败'}")
            
            print("HumanDetectionNetTS 模型导出并验证成功!")
        
    except Exception as e:
        print(f"测试 HumanDetectionNetTS 模型时出错: {e}")
        