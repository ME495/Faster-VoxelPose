# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import argparse
import os
import time
import cv2
import numpy as np
import json

import _init_paths
from core.config import config, update_config
from utils.utils import create_logger
from utils.transforms import get_affine_transform, get_scale
from utils.vis import test_vis_all
from utils.rtsp_utils import RTSPReader, RTSPWriter

import models

def render_result_on_image(config, meta, cameras, resize_transform, image, input_heatmaps, fused_poses):
    """
    在图像上渲染3D姿态估计结果
    
    Args:
        config: 配置对象
        meta: 元数据
        cameras: 相机参数
        resize_transform: 缩放变换矩阵
        image: 输入图像
        input_heatmaps: 输入热图
        fused_poses: 融合后的3D姿态
        
    Returns:
        image: 渲染后的图像
    """
    from utils.cameras import project_pose
    from utils.transforms import affine_transform_pts_cuda as do_transform
    
    batch_size, max_people, num_joints, _ = fused_poses.shape
    
    # 获取骨架连接
    if num_joints == 15:
        limbs = LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
                 [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]
    elif num_joints == 14:
        limbs = LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
                  [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]
    else:
        limbs = LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
                [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]
    
    # 可视化颜色
    colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
    color_map = {
        'b': (255, 0, 0),       # 蓝色 (BGR格式)
        'g': (0, 255, 0),       # 绿色
        'c': (255, 255, 0),     # 青色
        'y': (0, 255, 255),     # 黄色
        'm': (255, 0, 255),     # 洋红色
        'orange': (0, 165, 255),# 橙色
        'pink': (203, 192, 255),# 粉色
        'royalblue': (225, 105, 65), # 宝蓝色
        'lightgreen': (144, 238, 144), # 浅绿色
        'gold': (0, 215, 255)   # 金色
    }
    
    height, width = image.shape[:2]
    
    # 姿态估计结果可视化
    detected_count = 0
    for i in range(batch_size):
        curr_seq = meta['seq'][i]
        
        for n in range(max_people):
            # 检查置信度
            if fused_poses[i, n, 0, 4] < config.CAPTURE_SPEC.MIN_SCORE:
                continue
                
            detected_count += 1
            color = color_map[colors[5]]
            
            # 对每个视角渲染结果
            for c, camera in enumerate(cameras[curr_seq]):
                # 计算视角偏移
                row = c // 2
                col = c % 2
                y_offset = row * (height // 2)
                x_offset = col * (width // 2)
                
                # 投影3D姿态到2D视图
                pose_2d = project_pose(fused_poses[i, n, :, :3], camera)
                pose_2d = do_transform(pose_2d, resize_transform)
                
                # 绘制关节点
                for j in range(num_joints):
                    if is_valid_coord(pose_2d[j], width // 2, height // 2):
                        x = int(pose_2d[j][0] + x_offset)
                        y = int(pose_2d[j][1] + y_offset)
                        cv2.circle(image, (x, y), 8, color, -1)
                
                # 绘制骨架连接
                for limb in limbs:
                    parent = pose_2d[limb[0]]
                    child = pose_2d[limb[1]]
                    
                    if not is_valid_coord(parent, width // 2, height // 2) or \
                       not is_valid_coord(child, width // 2, height // 2):
                        continue
                    
                    px = int(parent[0] + x_offset)
                    py = int(parent[1] + y_offset)
                    cx = int(child[0] + x_offset)
                    cy = int(child[1] + y_offset)
                    
                    cv2.line(image, (px, py), (cx, cy), color, 4)
    
    # 添加检测到的人数信息
    cv2.putText(image, f"Detect {detected_count} people", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

def is_valid_coord(joint, width, height):
    """
    检查关节坐标是否有效
    
    Args:
        joint: 关节坐标 [x, y]
        width: 图像宽度
        height: 图像高度
        
    Returns:
        valid: 是否有效
    """
    valid_x = joint[0] >= 0 and joint[0] < width
    valid_y = joint[1] >= 0 and joint[1] < height
    return valid_x and valid_y

def parse_args():
    parser = argparse.ArgumentParser(description='基于RTSP的3D姿态估计服务')
    parser.add_argument('--cfg', help='配置文件路径', 
                        default='configs/custom/jln64.yaml', type=str)
    parser.add_argument('--rtsp_url', help='RTSP视频流URL', 
                        default='rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream', type=str)
    parser.add_argument('--output_url', help='输出RTSP流URL（可选）', 
                        default='', type=str)
    parser.add_argument('--view_mode', help='可视化模式：none, show, save, rtsp', 
                        default='show', type=str)
    parser.add_argument('--output_dir', help='结果保存目录（view_mode为save时使用）', 
                        default='output/results', type=str)
    parser.add_argument('--view_type', help='可视化类型，用逗号分隔',
                        default='image_with_poses', type=str)
    parser.add_argument('--log_dir', help='日志目录',
                        default='log', type=str)
    parser.add_argument('--calibration_file', help='相机标定文件路径',
                        default='', type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)
    
    return args


class CustomCamera:
    """创建相机参数"""
    def __init__(self, camera_id, R, T, fx, fy, cx, cy, k=None, p=None):
        """
        初始化相机参数
        Args:
            camera_id: 相机ID
            R: 旋转矩阵 (3x3)
            T: 平移向量 (3x1)
            fx, fy: 焦距
            cx, cy: 主点坐标
            k: 径向畸变系数 (默认为0)
            p: 切向畸变系数 (默认为0)
        """
        self.camera = {}
        self.camera['R'] = R
        self.camera['T'] = T
        self.camera['fx'] = fx
        self.camera['fy'] = fy
        self.camera['cx'] = cx
        self.camera['cy'] = cy
        
        if k is None:
            self.camera['k'] = np.zeros((3, 1))
        else:
            self.camera['k'] = k
            
        if p is None:
            self.camera['p'] = np.zeros((2, 1))
        else:
            self.camera['p'] = p
            
        self.id = camera_id
        
    def to_dict(self):
        return self.camera


def split_frame(frame, image_size, num_views=4):
    """
    将一个包含多视角图像的帧拆分为多个单独的视图
    
    Args:
        frame: 输入的完整帧
        image_size: 要求的的原始图像尺寸
        num_views: 视图数量(默认为4)
    
    Returns:
        views: 列表，包含拆分后的各个视图
        is_valid: 布尔值，指示分割是否有效
    """
    height, width = frame.shape[:2]
    expected_height = image_size[1] * 2  # 两行图像
    expected_width = image_size[0] * 2   # 两列图像
    
    # 检查整体尺寸是否符合预期
    if height != expected_height or width != expected_width:
        return None, False
    
    # 确定每个视图的尺寸
    if num_views == 4:  # 2x2布局
        view_height = height // 2
        view_width = width // 2
        
        views = [
            frame[0:view_height, 0:view_width],                          # 左上
            frame[0:view_height, view_width:width],                      # 右上
            frame[view_height:height, 0:view_width],                     # 左下
            frame[view_height:height, view_width:width]                  # 右下
        ]
        
    else:
        raise ValueError(f"不支持的视图数量: {num_views}")
    
    return views, True


def prepare_input(views, transform):
    """
    准备输入数据
    
    Args:
        views: 多视角视图列表
        transform: 数据变换
    
    Returns:
        inputs: 处理后的输入张量
    """
    processed_views = []
    
    for view in views:
        # 应用变换
        if transform:
            view = transform(view)
            
        processed_views.append(view)
    
    # 堆叠为批次
    inputs = torch.stack(processed_views, dim=0)
    inputs = inputs.unsqueeze(0)  # 添加批次维度
    
    return inputs


def setup_cameras(calibration_file=None, num_views=4):
    """
    根据标定文件设置相机参数
    
    Args:
        calibration_file: 相机标定文件路径
        num_views: 视图数量
    
    Returns:
        cameras: 相机参数字典
    """
    cameras = {}
    cameras['default'] = []
    
    # 如果提供了标定文件，从文件中加载相机参数
    if calibration_file and os.path.exists(calibration_file):
        with open(calibration_file, "r") as f:
            calib = json.load(f)
        
        # 假设标定文件中相机ID为字符串的键值
        for cam in calib.keys():
            # 构造相机参数字典
            cam_param = {}
            cam_param['fx'] = calib[cam]['k'][0]
            cam_param['fy'] = calib[cam]['k'][1]
            cam_param['cx'] = calib[cam]['k'][2]
            cam_param['cy'] = calib[cam]['k'][3]
            cam_param['k'] = np.array([calib[cam]['d'][0], calib[cam]['d'][1], calib[cam]['d'][4]]).reshape(3, 1)
            cam_param['p'] = np.array([calib[cam]['d'][2], calib[cam]['d'][3]]).reshape(2, 1)
            
            # 计算相机外参
            proj_mat = np.array(calib[cam]['p']).reshape(3, 4)
            K = np.array(
                [
                    [cam_param['fx'], 0, cam_param['cx']],
                    [0, cam_param['fy'], cam_param['cy']],
                    [0, 0, 1]
                ]
            )
            
            T_cam_world = np.linalg.inv(K).dot(proj_mat)
            R = T_cam_world[:3, :3]
            t = T_cam_world[:3, 3].reshape(3, 1)
            cam_param['R'] = R  # 世界坐标系到相机坐标系的旋转矩阵
            cam_param['T'] = -np.dot(R.T, t)  # 相机在世界坐标系中的位置
            
            cameras['default'].append(cam_param)
        
        if len(cameras['default']) != num_views:
            raise ValueError(f"标定文件中的相机数量({len(cameras['default'])})与配置的视图数量({num_views})不匹配！")
            
        return cameras
    else:
        # 如果没有提供标定文件，直接报错退出
        raise FileNotFoundError(f"错误：未提供相机标定文件或文件不存在！请使用--calibration_file参数指定有效的标定文件。")


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'service')
    logger.info(f'启动RTSP姿态估计服务 - 输入流: {args.rtsp_url}')

    # 设置设备
    logger.info(f'使用设备: {config.DEVICE}')
    
    # 准备输入数据处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # 设置CUDA相关配置
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    # 构建模型
    logger.info('=> 构建模型...')
    model = eval('models.' + config.MODEL + '.get')(config)
    model = model.to(config.DEVICE)
    
    # 加载特征提取网络
    if config.NETWORK.PRETRAINED_BACKBONE:
        backbone = eval('models.' + config.BACKBONE + '.get')(config)
        logger.info('=> 加载预训练特征提取网络权重')
        backbone.load_state_dict(torch.load(config.NETWORK.PRETRAINED_BACKBONE))
        backbone = backbone.to(config.DEVICE)
        backbone.eval()
    else:
        backbone = None
    
    # 加载模型权重
    model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(model_file):
        logger.info(f'=> 加载模型权重 {model_file}')
        model.load_state_dict(torch.load(model_file))
    else:
        logger.error('模型文件不存在！')
        return
    
    # 设置为评估模式
    model.eval()
    
    # 设置相机参数
    try:
        cameras = setup_cameras(calibration_file=args.calibration_file, num_views=config.DATASET.CAMERA_NUM)
        logger.info(f'=> 成功加载相机参数，相机数量: {len(cameras["default"])}')
    except Exception as e:
        logger.error(str(e))
        logger.error('程序无法继续，退出中...')
        return
    
    # 计算缩放变换矩阵
    ori_image_size = np.array(config.DATASET.ORI_IMAGE_SIZE)
    image_size = np.array(config.DATASET.IMAGE_SIZE)
    c = np.array([ori_image_size[0] / 2.0, ori_image_size[1] / 2.0])
    s = get_scale(ori_image_size, image_size)
    r = 0
    trans = get_affine_transform(c, s, r, image_size)
    resize_transform = torch.as_tensor(trans, dtype=torch.float, device=config.DEVICE)
    
    # 初始化视频流读取器
    rtsp_reader = RTSPReader(args.rtsp_url).start()
    time.sleep(1.0)  # 等待视频流初始化
    
    # 初始化输出流（如果需要）
    rtsp_writer = None
    if args.view_mode == 'rtsp' and args.output_url:
        output_width = image_size[0] * 2
        output_height = image_size[1] * 2
        rtsp_writer = RTSPWriter(args.output_url, output_width, output_height).start()
    
    # 可视化类型
    vis_types = args.view_type.split(',')
    
    # 统计FPS
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    # 主循环
    logger.info('=> 开始处理视频流')
    frame_count = 0
    skipped_count = 0
    
    try:
        while True:
            # 读取一帧
            frame = rtsp_reader.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # 将单个帧拆分为多个视图
            views, is_valid = split_frame(frame, image_size, config.DATASET.CAMERA_NUM)
            
            # 检查图像尺寸是否符合要求
            if not is_valid:
                skipped_count += 1
                logger.warning(f"警告: 接收到的图像尺寸与配置不匹配，已跳过。当前帧: {frame_count}, 已跳过: {skipped_count}")
                continue
            
            # 记录起始时间
            start_time = time.time()
            
            # 准备输入数据
            inputs = prepare_input(views, transform)
            inputs = inputs.to(config.DEVICE)
            
            # 创建元数据
            meta = {'seq': ['default']}
            
            # 前向推理
            with torch.no_grad():
                fused_poses, plane_poses, proposal_centers, input_heatmaps, _ = model(
                    backbone=backbone, 
                    views=inputs, 
                    meta=meta, 
                    cameras=cameras, 
                    resize_transform=resize_transform
                )
            
            # 计算推理时间
            inference_time = time.time() - start_time
            
            # 计算FPS
            fps_count += 1
            if fps_count >= 10:
                fps = fps_count / (time.time() - fps_time)
                fps_time = time.time()
                fps_count = 0
            
            # 处理结果可视化
            if args.view_mode != 'none':
                # 构建可视化图像
                if 'image_with_poses' in vis_types:
                    # 合并四个视图并在其上绘制结果
                    vis_image = np.zeros((image_size[1] * 2, image_size[0] * 2, 3), dtype=np.uint8)
                    
                    # 将拆分的视图放回原位
                    for i, view in enumerate(views):
                        if view.shape[0] != image_size[1] or view.shape[1] != image_size[0]:
                            view = cv2.resize(view, (image_size[0], image_size[1]))
                        
                        row = i // 2
                        col = i % 2
                        y_start = row * image_size[1]
                        x_start = col * image_size[0]
                        vis_image[y_start:y_start+image_size[1], x_start:x_start+image_size[0]] = view
                    
                    # 在图像上渲染3D姿态结果
                    vis_image = render_result_on_image(config, meta, cameras, resize_transform, vis_image, input_heatmaps, fused_poses)
                    
                    # 添加FPS信息
                    cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(vis_image, f"Inference time: {inference_time*1000:.1f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 根据可视化模式处理
                    if args.view_mode == 'show':
                        cv2.imshow('3D Pose Estimation', vis_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    elif args.view_mode == 'save':
                        os.makedirs(args.output_dir, exist_ok=True)
                        timestamp = int(time.time() * 1000)
                        cv2.imwrite(os.path.join(args.output_dir, f'pose_{timestamp}.jpg'), vis_image)
                    elif args.view_mode == 'rtsp' and rtsp_writer is not None:
                        rtsp_writer.write(vis_image)
            
            # 输出当前状态信息
            if fps_count == 0:
                logger.info(f'FPS: {fps:.1f}, 推理时间: {inference_time*1000:.1f}ms')
    
    except KeyboardInterrupt:
        logger.info('服务被用户中断')
    finally:
        # 清理资源
        logger.info('正在清理资源...')
        rtsp_reader.stop()
        if rtsp_writer is not None:
            rtsp_writer.stop()
        if args.view_mode == 'show':
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
