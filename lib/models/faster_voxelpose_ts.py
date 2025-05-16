# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 测试时，需要添加项目根目录到Python路径
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from models import resnet
from models.human_detection_net_ts import HumanDetectionNetTS
from models.joint_localization_net_ts import JointLocalizationNetTS

class FasterVoxelPoseNetTS(nn.Module):
    """
    Faster VoxelPose 网络：用于多视角3D人体姿态估计的端到端模型
    
    主要包含两个子网络：
    1. 人体检测网络(HumanDetectionNet)：检测3D空间中的人体位置
    2. 关节定位网络(JointLocalizationNet)：预测每个人体的关节位置
    
    工作流程：输入多视角图像 -> 提取2D热图 -> 检测人体位置 -> 定位关节点 -> 输出3D姿态
    """
    max_people: int
    num_joints: int
    
    def __init__(self, cfg):
        """
        初始化Faster VoxelPose网络
        
        Args:
            cfg: 配置对象，包含网络参数和训练参数
        """
        super(FasterVoxelPoseNetTS, self).__init__()
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE  # 最大人数
        self.num_joints = cfg.DATASET.NUM_JOINTS  # 关节点数量
       
        # 初始化两个主要子网络
        self.pose_net = HumanDetectionNetTS(cfg.CAPTURE_SPEC.MAX_PEOPLE,
                                            cfg.CAPTURE_SPEC.MIN_SCORE,
                                            cfg.CAPTURE_SPEC.SPACE_SIZE,
                                            cfg.CAPTURE_SPEC.VOXELS_PER_AXIS,
                                            cfg.CAPTURE_SPEC.SPACE_CENTER,
                                            cfg.DATASET.NUM_JOINTS)  # 人体检测网络
        self.joint_net = JointLocalizationNetTS(cfg.DATASET.NUM_JOINTS,
                                                cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS,
                                                cfg.NETWORK.NUM_CHANNEL_JOINT_FEAT,
                                                cfg.NETWORK.NUM_CHANNEL_JOINT_HIDDEN,
                                                cfg.DATASET.IMAGE_SIZE,
                                                cfg.DATASET.HEATMAP_SIZE,
                                                cfg.DATASET.ORI_IMAGE_SIZE,
                                                cfg.CAPTURE_SPEC.SPACE_CENTER,
                                                cfg.CAPTURE_SPEC.SPACE_SIZE,
                                                cfg.CAPTURE_SPEC.VOXELS_PER_AXIS,
                                                cfg.NETWORK.BETA)  # 关节定位网络


    def forward(self, input_heatmaps: torch.Tensor, sample_grids: torch.Tensor, fine_sample_grids: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            input_heatmaps: 预计算的输入热图，形状为[batch_size, num_views, num_joints, height, width]
            sample_grids: 用于特征采样的网格，形状为(batch_size, num_views, voxels_x, voxels_y, voxels_z, 2)
            fine_sample_grids: 用于精细采样的网格，形状为(batch_size, num_views, fine_voxels_x, fine_voxels_y, fine_voxels_z, 2)
            
        Returns:
            fused_poses: 融合后的3D关节位置，形状为[batch_size, max_proposals, num_joints, 5]
                包含：3D坐标(x,y,z)、匹配索引、置信度
        """
        
        batch_size = input_heatmaps.shape[0]
 
        # 使用人体检测网络检测人体位置
        # proposal_heatmaps_2d: 2D候选热图
        # proposal_heatmaps_1d: 1D候选热图
        # proposal_centers: 候选人体中心点信息
        # bbox_preds: 边界框预测
        proposal_heatmaps_2d, proposal_heatmaps_1d, proposal_centers, \
                              bbox_preds = self.pose_net(input_heatmaps, sample_grids)
        # 创建有效提案的掩码（proposal_centers[:, :, 3]>=0表示该提案匹配到了真实人体）
        mask = (proposal_centers[:, :, 3] >= 0)

        # 使用关节定位网络预测关节位置
        # fused_poses: 融合后的3D关节位置
        # plane_poses: 三个正交平面(xy, xz, yz)上的关节投影
        fused_poses, plane_poses = self.joint_net(input_heatmaps, fine_sample_grids, proposal_centers, mask)

        # 将置信度和匹配索引信息拼接到融合姿态结果中
        # proposal_centers[:, :, 3:5]包含匹配索引和置信度
        fused_poses = torch.cat([fused_poses, proposal_centers[:, :, 3:5].reshape(batch_size,\
                                 -1, 1, 2).repeat(1, 1, self.num_joints, 1)], dim=3)

        return fused_poses


# 测试代码
if __name__ == "__main__":
    print("=== 测试 FasterVoxelPoseNetTS 模型 ===")
    
    # 创建一个简单的配置类
    class SimpleConfig:
        class CaptureSpec:
            MAX_PEOPLE = 10
            MIN_SCORE = 0.1
            SPACE_SIZE = [2000.0, 2000.0, 2000.0]
            VOXELS_PER_AXIS = [64, 64, 64]
            SPACE_CENTER = [0.0, 0.0, 1000.0]
            
        class Dataset:
            NUM_JOINTS = 17
            IMAGE_SIZE = [256, 256]
            HEATMAP_SIZE = [64, 64]
            ORI_IMAGE_SIZE = [1280, 720]
            
        class IndividualSpec:
            VOXELS_PER_AXIS = [32, 32, 32]
            
        class Network:
            NUM_CHANNEL_JOINT_FEAT = 32
            NUM_CHANNEL_JOINT_HIDDEN = 64
            BETA = 100.0
            
        CAPTURE_SPEC = CaptureSpec()
        DATASET = Dataset()
        INDIVIDUAL_SPEC = IndividualSpec()
        NETWORK = Network()
    
    # 创建模型实例
    cfg = SimpleConfig()
    model = FasterVoxelPoseNetTS(cfg)
    print("模型创建成功!")
    
    # 创建测试输入
    batch_size = 2
    num_views = 3
    num_joints = cfg.DATASET.NUM_JOINTS
    h, w = 64, 64
    
    # 模拟热图输入
    input_heatmaps = torch.rand(batch_size, num_views, num_joints, h, w)
    
    # 模拟采样网格
    voxels_x = cfg.CAPTURE_SPEC.VOXELS_PER_AXIS[0]
    voxels_y = cfg.CAPTURE_SPEC.VOXELS_PER_AXIS[1]
    voxels_z = cfg.CAPTURE_SPEC.VOXELS_PER_AXIS[2]
    sample_grids = torch.rand(batch_size, num_views, voxels_x, voxels_y, voxels_z, 2) * 2 - 1
    
    # 模拟精细采样网格
    fine_voxels_x = cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS[0]
    fine_voxels_y = cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS[1]
    fine_voxels_z = cfg.INDIVIDUAL_SPEC.VOXELS_PER_AXIS[2]
    fine_sample_grids = torch.rand(batch_size, num_views, fine_voxels_x, fine_voxels_y, fine_voxels_z, 2) * 2 - 1
    
    print("测试输入准备完成!")
    
    # 测试前向传播
    try:
        print("运行前向传播...")
        outputs = model(input_heatmaps, sample_grids, fine_sample_grids)
        print(f"输出形状: {outputs.shape}")
        print("前向传播成功!")
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 尝试导出为 TorchScript
    try:
        print("\n尝试导出为 TorchScript...")
        
        # 尝试使用 torch.jit.script
        model_path = "faster_voxelpose_net_ts.pt"
        scripted_model = torch.jit.script(model)
        print("模型成功通过 torch.jit.script 导出!")
        
        # 保存模型
        scripted_model.save(model_path)
        print(f"模型已保存到: {model_path}")
        
        # 加载导出的模型
        print("\n加载导出的模型并验证...")
        loaded_model = torch.jit.load(model_path)
        
        # 使用相同的输入测试原始模型和加载的模型
        print("运行原始模型...")
        original_outputs = model(input_heatmaps, sample_grids, fine_sample_grids)
        
        print("运行转换后的模型...")
        converted_outputs = loaded_model(input_heatmaps, sample_grids, fine_sample_grids)
        
        # 验证输出是否一致
        match = torch.allclose(original_outputs, converted_outputs, rtol=1e-5, atol=1e-5)
        
        print("\n模型输出验证结果:")
        print(f"- 输出一致性: {'通过' if match else '失败'}")
        
        if match:
            print("\n模型已成功导出为 TorchScript 格式并通过验证。")
        else:
            print("\n警告: 转换后的模型输出与原始模型不一致，可能存在兼容性问题。")
            
    except Exception as e:
        print(f"导出为 TorchScript 失败: {e}")
        import traceback
        traceback.print_exc()