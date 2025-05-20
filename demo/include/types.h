#pragma once

#include <torch/torch.h>
#include <string>

// 相机参数数据结构
struct CameraParams {
    torch::Tensor R;         // 旋转矩阵 (3x3), 世界坐标到相机坐标
    torch::Tensor T;         // 平移向量 (3x1), 世界坐标系中的相机中心 (C_w)
                            // x_cam = R @ (x_world - T)
    torch::Tensor K_intrinsic; // 内参矩阵 (3x3)
    torch::Tensor f;         // 焦距 (fx, fy) (2x1)
    torch::Tensor c;         // 主点 (cx, cy) (2x1)
    torch::Tensor k_dist;    // 径向畸变系数 (k1, k2, k3) (3x1)
    torch::Tensor p_dist;    // 切向畸变系数 (p1, p2) (2x1)
    std::string id;          // 相机ID，例如 "44310001"

    CameraParams();
    CameraParams to(torch::Device device);
};

// 关节连接定义（肢体连接）
extern const std::vector<std::pair<int, int>> LIMBS17;
extern const std::vector<std::pair<int, int>> LIMBS15;
std::vector<std::pair<int, int>> get_limbs_definition(int num_joints); 