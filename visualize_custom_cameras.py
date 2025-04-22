#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import json_tricks as json
import argparse

def visualize_custom_cameras(dataset_dir, sequence):
    """
    可视化自定义数据集的世界坐标系和相机位置
    
    Args:
        dataset_dir: 数据集根目录
        sequence: 序列名称，如 'Take_036'
    """
    # 加载相机参数
    cam_file = osp.join(dataset_dir, sequence, "calibration.json")
    with open(cam_file, "r") as f:
        calib = json.load(f)
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制世界坐标系
    origin = np.zeros(3)
    ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='r', length=500, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='g', length=500, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='b', length=500, arrow_length_ratio=0.1)
    
    # 提取并可视化相机位置
    camera_positions = []
    camera_orientations = []
    
    for cam_name, cam_params in calib.items():
        # 构建相机内参矩阵K
        K = np.array([
            [cam_params['k'][0], 0, cam_params['k'][2]],
            [0, cam_params['k'][1], cam_params['k'][3]],
            [0, 0, 1]
        ])
        
        # 获取投影矩阵
        proj_mat = np.array(cam_params['p']).reshape(3, 4)
        
        # 计算相机外参
        T_cam_world = np.linalg.inv(K).dot(proj_mat)
        R = T_cam_world[:3, :3]
        t = T_cam_world[:3, 3].reshape(3, 1)
        
        # 计算相机中心在世界坐标系中的位置
        # C = -R^T * t
        C = -np.dot(R.T, t).flatten()
        
        # 计算相机朝向 (z轴)
        z_axis = R[2, :] * 300  # 放大以便可视化，与Panoptic脚本一致
        
        camera_positions.append(C)
        camera_orientations.append(z_axis)
        
        # 打印相机位置
        print(f"相机 {cam_name} 位置: {C}")
        
        # 绘制相机位置
        ax.scatter(C[0], C[1], C[2], color='black', s=100, label=f'Camera {cam_name}')
        
        # 绘制相机朝向
        ax.quiver(C[0], C[1], C[2], z_axis[0], z_axis[1], z_axis[2], color='purple', length=1.0)
    
    # 设置图形属性
    ax.set_xlabel('X 轴', fontsize=14)
    ax.set_ylabel('Y 轴', fontsize=14)
    ax.set_zlabel('Z 轴', fontsize=14)
    ax.set_title(f'自定义数据集 {sequence} 相机位置可视化', fontsize=16)
    
    # 设置坐标轴范围，确保所有相机都可见
    camera_positions = np.array(camera_positions)
    min_pos = np.min(camera_positions, axis=0) - 1000
    max_pos = np.max(camera_positions, axis=0) + 1000
    mid_pos = (np.min(camera_positions, axis=0) + np.max(camera_positions, axis=0)) / 2
    print(mid_pos)
    
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])
    # ax.set_zlim(min_pos[2], max_pos[2])
    ax.set_zlim(0, max_pos[2])  # 保持z轴从0开始，与Panoptic脚本一致
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{sequence}_custom_cameras.png', dpi=300)
    plt.show()
    
    print(f"图像已保存为 {sequence}_custom_cameras.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化自定义数据集的相机配置')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='数据集根目录路径')
    parser.add_argument('--sequence', type=str, default='Take_036',
                        help='要可视化的序列名称')
    
    args = parser.parse_args()
    
    visualize_custom_cameras(args.dataset_dir, args.sequence) 