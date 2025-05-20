#pragma once

#include <torch/torch.h>
#include <vector>
#include "types.h"

// 计算3D体素网格
torch::Tensor compute_grid_cpp(
    const std::vector<float>& boxSize, 
    const std::vector<float>& boxCenter, 
    const std::vector<int>& nBins, 
    torch::Device device);

// 3D点投影到2D图像平面
torch::Tensor project_point_cpp(
    torch::Tensor x, 
    const torch::Tensor& R, 
    const torch::Tensor& T, 
    const torch::Tensor& f, 
    const torch::Tensor& c, 
    const torch::Tensor& k_dist, 
    const torch::Tensor& p_dist);

// 仿射变换点
torch::Tensor affine_transform_pts_cpp(
    torch::Tensor pts, 
    torch::Tensor t);

// 3D网格投影到2D图像平面
torch::Tensor project_grid_cpp(
    torch::Tensor grid, 
    const std::vector<int>& ori_image_size_vec,
    const std::vector<int>& image_size_vec,
    const CameraParams& camera, 
    int heatmap_w, 
    int heatmap_h, 
    int nbins, 
    torch::Tensor resize_transform, 
    torch::Device device);

// 投影3D姿态到2D图像平面
torch::Tensor project_pose_cpp(
    const torch::Tensor& poses_3d_person,
    const CameraParams& camera);

// 检查坐标点是否在图像范围内
bool is_valid_coord_cpp(
    const torch::Tensor& pt, 
    int width, 
    int height); 