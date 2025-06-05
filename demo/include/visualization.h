#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "types.h"

namespace fs = std::filesystem;

void visualize_3d_pose(
    cv::Mat& image,
    const torch::Tensor& poses_3d,
    const CameraParams& camera_params,
    const torch::Tensor& resize_transform,
    const std::vector<std::pair<int, int>>& limbs,
    const std::vector<int>& original_image_size
);

// 保存带有3D姿态投影的图像
void save_image_with_poses_cpp(
    const std::string& output_prefix_str,
    const torch::Tensor& image_tensor_chw, // (Channels, Height, Width), RGB, float
    const torch::Tensor& poses_3d,         // (1, MaxPeople, NumJoints, 5)
    const CameraParams& camera_params,
    const torch::Tensor& resize_transform, // 2x3仿射变换矩阵
    int view_idx,
    float min_pose_score,
    const std::vector<std::pair<int, int>>& limbs,
    const std::vector<int>& original_image_size // 原始图像大小{ori_w, ori_h}
); 