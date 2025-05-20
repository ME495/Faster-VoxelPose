#include "camera.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;

std::vector<CameraParams> load_cameras_from_json(const std::string& json_path, torch::Device device) {
    std::vector<CameraParams> cameras;
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        std::cerr << "错误：无法打开标定文件: " << json_path << std::endl;
        return cameras;
    }
    json calib_data;
    try {
        calib_data = json::parse(ifs);
    } catch (json::parse_error& e) {
        std::cerr << "JSON解析错误: " << e.what() << std::endl;
        return cameras;
    }

    for (auto& [cam_id_str, cam_json] : calib_data.items()) {
        CameraParams cam_p;
        cam_p.id = cam_id_str;
        try {
            std::vector<float> k_vec = cam_json["k"].get<std::vector<float>>();
            cam_p.f = torch::tensor({k_vec[0], k_vec[1]}, torch::kFloat32).to(device).view({2,1});
            cam_p.c = torch::tensor({k_vec[2], k_vec[3]}, torch::kFloat32).to(device).view({2,1});
            cam_p.K_intrinsic = torch::eye(3, torch::kFloat32).to(device);
            cam_p.K_intrinsic[0][0] = k_vec[0];
            cam_p.K_intrinsic[1][1] = k_vec[1];
            cam_p.K_intrinsic[0][2] = k_vec[2];
            cam_p.K_intrinsic[1][2] = k_vec[3];

            std::vector<float> d_vec = cam_json["d"].get<std::vector<float>>();
            cam_p.k_dist = torch::tensor({d_vec[0], d_vec[1], d_vec[4]}, torch::kFloat32).to(device).view({3,1});
            cam_p.p_dist = torch::tensor({d_vec[2], d_vec[3]}, torch::kFloat32).to(device).view({2,1});

            std::vector<float> p_mat_vec = cam_json["p"].get<std::vector<float>>();
            torch::Tensor P_proj = torch::tensor(p_mat_vec, torch::kFloat32).to(device).view({3, 4});
            torch::Tensor K_inv = torch::inverse(cam_p.K_intrinsic);
            torch::Tensor RT_compound = torch::mm(K_inv, P_proj);
            cam_p.R = RT_compound.slice(1, 0, 3);
            torch::Tensor t_world_to_cam = RT_compound.slice(1, 3, 4);
            cam_p.T = -torch::mm(cam_p.R.transpose(0,1), t_world_to_cam);
            cameras.push_back(cam_p);
        } catch (const std::exception& e) {
            std::cerr << "解析相机 " << cam_id_str << " 时出错: " << e.what() << std::endl;
        }
    }
    
    // 按ID字符串排序相机，确保处理顺序一致
    std::sort(cameras.begin(), cameras.end(), [](const CameraParams& a, const CameraParams& b) {
        return a.id < b.id;
    });
    
    return cameras;
} 