#include "types.h"

// CameraParams构造函数实现
CameraParams::CameraParams() :
    R(torch::eye(3, torch::kFloat32)),
    T(torch::zeros({3,1}, torch::kFloat32)),
    K_intrinsic(torch::eye(3, torch::kFloat32)),
    f(torch::ones({2,1}, torch::kFloat32) * 1000.0),
    c(torch::zeros({2,1}, torch::kFloat32)),
    k_dist(torch::zeros({3,1}, torch::kFloat32)),
    p_dist(torch::zeros({2,1}, torch::kFloat32)),
    id("unknown")
{}

// 将相机参数移动到指定设备
CameraParams CameraParams::to(torch::Device device) {
    R = R.to(device);
    T = T.to(device);
    K_intrinsic = K_intrinsic.to(device);
    f = f.to(device);
    c = c.to(device);
    k_dist = k_dist.to(device);
    p_dist = p_dist.to(device);
    return *this;
}

// COCO 17关节点模型的肢体连接定义
const std::vector<std::pair<int, int>> LIMBS17 = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},         // 头部 (鼻子到左/右眼, 左/右眼到左/右耳)
    {5, 6},                                 // 肩膀
    {5, 7}, {7, 9},                         // 左臂
    {6, 8}, {8, 10},                        // 右臂
    {11, 12},                               // 臀部
    {5, 11}, {6, 12},                       // 躯干
    {11, 13}, {13, 15},                     // 左腿
    {12, 14}, {14, 16}                      // 右腿
};

// 15关节点模型的肢体连接定义
const std::vector<std::pair<int, int>> LIMBS15 = {
    {0, 1}, {0, 2}, {0, 3}, {3, 4}, {4, 5}, 
    {0, 9}, {9, 10}, {10, 11}, {2, 6}, {2, 12}, 
    {6, 7}, {7, 8}, {12, 13}, {13, 14}
};

// 根据关节点数量获取对应的肢体连接定义
std::vector<std::pair<int, int>> get_limbs_definition(int num_joints) {
    if (num_joints == 17) {
        return LIMBS17;
    } else if (num_joints == 15) {
        return LIMBS15;
    }
    // 未找到对应定义时返回空列表
    return {};
} 