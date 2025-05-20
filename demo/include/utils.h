#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// 列出目录中的图像文件，按字母顺序排序
std::vector<std::string> list_image_files(const std::string& dir_path);

// 获取调整大小的仿射变换矩阵
torch::Tensor get_resize_affine_transform_cpp(
    const std::vector<int>& ori_size, 
    const std::vector<int>& new_size, 
    torch::Device device);

// 定义可视化颜色
extern const std::vector<cv::Scalar> VIS_COLORS; 