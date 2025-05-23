import os
import numpy as np
import pickle
import json
import glob
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='验证SPACE_CENTER的设置依据')
    parser.add_argument('--config', default='configs/panoptic/jln64.yaml', help='配置文件路径')
    parser.add_argument('--datadir', default='data/Panoptic', help='数据集目录')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化图表')
    args = parser.parse_args()
    return args

def load_config(config_file):
    """加载配置文件"""
    import yaml
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # 获取配置中的SPACE_CENTER
    space_center = cfg['CAPTURE_SPEC']['SPACE_CENTER']
    print(f"配置文件中的SPACE_CENTER: {space_center}")
    
    # 定义训练序列列表 (与Panoptic数据集一致)
    TRAIN_LIST = [
        '160422_ultimatum1',
        '160224_haggling1',
        '160226_haggling1',
        '161202_haggling1',
        '160906_ian1',
        '160906_ian2',
        '160906_ian3',
        '160906_band1',
        '160906_band2'
    ]
    
    VAL_LIST = [
        '160906_pizza1',
        '160422_haggling1',
        '160906_ian5',
        '160906_band4',
    ]
    
    # 合并所有序列
    ALL_SEQUENCES = TRAIN_LIST + VAL_LIST
    
    # 定义根关节ID (通常是骨盆或mid-hip)
    ROOT_JOINT_ID = cfg['DATASET']['ROOT_JOINT_ID']  # 注意，有些数据集可能是列表
    
    # 存储所有人体的中心坐标
    all_centers = []
    
    # 遍历所有序列
    for seq in tqdm(ALL_SEQUENCES, desc="处理序列"):
        # 确定序列数据目录
        curr_anno = os.path.join(args.datadir, seq, 'hdPose3d_stage1_coco19')
        if not os.path.exists(curr_anno):
            print(f"警告: 找不到序列 {seq} 的数据目录")
            continue
            
        # 获取所有标注文件
        anno_files = sorted(glob.glob(f'{curr_anno}/*.json'))
        
        # 设置采样间隔
        interval = 3  # 用于训练集的采样间隔
        
        # 遍历标注文件
        for i, anno_file in enumerate(anno_files):
            if i % interval == 0:  # 按间隔采样
                try:
                    with open(anno_file, "r") as f:
                        bodies = json.load(f)["bodies"]
                    
                    if len(bodies) == 0:
                        continue
                        
                    # 处理每个人体
                    for body in bodies:
                        pose3d = np.array(body['joints19']).reshape((-1, 4))
                        
                        # 根关节为单一ID
                        if isinstance(ROOT_JOINT_ID, int):
                            # 检查可见性
                            if pose3d[ROOT_JOINT_ID, -1] > 0.1:
                                # 坐标变换 (与原代码一致)
                                M = np.array([[1.0, 0.0, 0.0],
                                              [0.0, 0.0, -1.0],
                                              [0.0, 1.0, 0.0]])
                                pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
                                
                                # 获取根关节坐标作为人体中心 (乘以10以匹配原代码处理)
                                center = pose3d[ROOT_JOINT_ID, 0:3] * 10.0
                                all_centers.append(center)
                        
                        # 如果根关节是列表 (如骨盆由两个关节点表示)
                        elif isinstance(ROOT_JOINT_ID, list):
                            # 检查可见性 (至少一个关节点可见)
                            if any(pose3d[j, -1] > 0.1 for j in ROOT_JOINT_ID):
                                # 坐标变换
                                M = np.array([[1.0, 0.0, 0.0],
                                              [0.0, 0.0, -1.0],
                                              [0.0, 1.0, 0.0]])
                                pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
                                
                                # 计算多个关节点的平均值作为中心
                                visible_joints = [j for j in ROOT_JOINT_ID if pose3d[j, -1] > 0.1]
                                if visible_joints:
                                    center = np.mean([pose3d[j, 0:3] for j in visible_joints], axis=0) * 10.0
                                    all_centers.append(center)
                                
                except Exception as e:
                    print(f"处理文件 {anno_file} 时出错: {e}")
    
    # 计算所有人体中心的均值
    if all_centers:
        all_centers = np.array(all_centers)
        
        # 基本统计信息
        mean_center = np.mean(all_centers, axis=0)
        std_center = np.std(all_centers, axis=0)
        min_center = np.min(all_centers, axis=0)
        max_center = np.max(all_centers, axis=0)
        
        # 计算中点值 (最大值+最小值)/2
        midpoint = (min_center + max_center) / 2
        
        print(f"处理了 {len(all_centers)} 个人体")
        print("\n坐标统计信息:")
        print(f"均值: {mean_center}")
        print(f"标准差: {std_center}")
        print(f"最小值: {min_center}")
        print(f"最大值: {max_center}")
        print(f"中点值 (最大值+最小值)/2: {midpoint}")
        
        # 比较配置的SPACE_CENTER与中点的差异
        print("\n验证猜想 - SPACE_CENTER是否为坐标最大最小值的中点:")
        midpoint_diff = midpoint - np.array(space_center)
        print(f"SPACE_CENTER: {space_center}")
        print(f"坐标中点值: {midpoint}")
        print(f"差异 (中点 - SPACE_CENTER): {midpoint_diff}")
        
        # 计算相对误差
        relative_error = 100 * np.abs(midpoint_diff / midpoint)
        print(f"相对误差 (%): {relative_error}")
        
        # 判断是否接近
        threshold = 5.0  # 5%的相对误差阈值
        xy_is_close = np.all(relative_error[:2] < threshold)
        
        if xy_is_close:
            print("\n结论: SPACE_CENTER的x和y非常接近坐标中点值，猜想很可能正确！")
        else:
            print("\n结论: SPACE_CENTER的x和y与坐标中点值存在较大差异，猜想可能不成立。")
        
        # 可视化分析
        if args.visualize:
            plt.figure(figsize=(15, 10))
            
            # 散点图: x-y平面上的人体分布
            plt.subplot(2, 2, 1)
            plt.scatter(all_centers[:, 0], all_centers[:, 1], alpha=0.1, s=1)
            plt.scatter(space_center[0], space_center[1], color='red', label='SPACE_CENTER', s=100, marker='x')
            plt.scatter(midpoint[0], midpoint[1], color='green', label='计算的中点', s=100, marker='+')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
            plt.grid(alpha=0.3)
            plt.title('人体中心在X-Y平面的分布')
            plt.xlabel('X轴 (mm)')
            plt.ylabel('Y轴 (mm)')
            plt.legend()
            
            # 散点图: x-z平面上的人体分布
            plt.subplot(2, 2, 2)
            plt.scatter(all_centers[:, 0], all_centers[:, 2], alpha=0.1, s=1)
            plt.scatter(space_center[0], space_center[2], color='red', label='SPACE_CENTER', s=100, marker='x')
            plt.scatter(midpoint[0], midpoint[2], color='green', label='计算的中点', s=100, marker='+')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
            plt.grid(alpha=0.3)
            plt.title('人体中心在X-Z平面的分布')
            plt.xlabel('X轴 (mm)')
            plt.ylabel('Z轴 (mm)')
            plt.legend()
            
            # 直方图: X坐标分布
            plt.subplot(2, 2, 3)
            plt.hist(all_centers[:, 0], bins=50, alpha=0.7)
            plt.axvline(x=space_center[0], color='red', linestyle='-', label='SPACE_CENTER')
            plt.axvline(x=midpoint[0], color='green', linestyle='--', label='计算的中点')
            plt.grid(alpha=0.3)
            plt.title('X坐标分布直方图')
            plt.xlabel('X轴 (mm)')
            plt.ylabel('频率')
            plt.legend()
            
            # 直方图: Y坐标分布
            plt.subplot(2, 2, 4)
            plt.hist(all_centers[:, 1], bins=50, alpha=0.7)
            plt.axvline(x=space_center[1], color='red', linestyle='-', label='SPACE_CENTER')
            plt.axvline(x=midpoint[1], color='green', linestyle='--', label='计算的中点')
            plt.grid(alpha=0.3)
            plt.title('Y坐标分布直方图')
            plt.xlabel('Y轴 (mm)')
            plt.ylabel('频率')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('space_center_analysis.png', dpi=300)
            print(f"可视化结果已保存为 'space_center_analysis.png'")
    else:
        print("未找到有效的人体数据")

if __name__ == "__main__":
    main() 