#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

/**
 * 相机参数结构体
 */
struct CameraParams {
    // 内参矩阵
    cv::Mat K;  // 3x3 相机内参矩阵
    // 外参矩阵
    cv::Mat R;  // 3x3 旋转矩阵
    cv::Mat t;  // 3x1 平移向量
    // 畸变参数
    cv::Mat distCoeffs;  // 畸变系数
};

/**
 * 相机工具类：用于处理相机参数和图像加载
 */
class CameraUtils {
public:
    /**
     * 加载相机参数
     * 
     * @param camera_file 相机参数文件路径
     * @return 相机参数列表
     */
    static std::vector<CameraParams> loadCameraParams(const std::string& camera_file) {
        std::vector<CameraParams> cameras;
        
        try {
            // 读取相机参数文件
            // 这里假设相机参数文件是JSON或YAML格式
            cv::FileStorage fs(camera_file, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                throw std::runtime_error("无法打开相机参数文件: " + camera_file);
            }
            
            // 读取相机数量
            int num_cameras = 0;
            fs["num_cameras"] >> num_cameras;
            
            // 读取每个相机的参数
            for (int i = 0; i < num_cameras; i++) {
                CameraParams camera;
                std::string prefix = "camera_" + std::to_string(i);
                
                fs[prefix + "_K"] >> camera.K;
                fs[prefix + "_R"] >> camera.R;
                fs[prefix + "_t"] >> camera.t;
                fs[prefix + "_distCoeffs"] >> camera.distCoeffs;
                
                cameras.push_back(camera);
            }
            
            fs.release();
        }
        catch (const std::exception& e) {
            std::cerr << "加载相机参数时出错: " << e.what() << std::endl;
            throw;
        }
        
        return cameras;
    }
    
    /**
     * 将相机参数转换为Torch张量
     * 
     * @param cameras 相机参数列表
     * @return Torch张量列表，每个张量包含一个相机的参数
     */
    static std::vector<torch::Tensor> convertToTorchTensors(const std::vector<CameraParams>& cameras) {
        std::vector<torch::Tensor> tensors;
        
        for (const auto& camera : cameras) {
            // 创建相机参数张量
            // 假设模型需要的相机参数格式为：[K|R|t]
            // 根据实际模型需求可能需要调整
            
            // 转换内参矩阵K
            torch::Tensor K_tensor = torch::from_blob(
                const_cast<float*>(reinterpret_cast<const float*>(camera.K.data)),
                {3, 3},
                torch::kFloat32
            ).clone();
            
            // 转换旋转矩阵R
            torch::Tensor R_tensor = torch::from_blob(
                const_cast<float*>(reinterpret_cast<const float*>(camera.R.data)),
                {3, 3},
                torch::kFloat32
            ).clone();
            
            // 转换平移向量t
            torch::Tensor t_tensor = torch::from_blob(
                const_cast<float*>(reinterpret_cast<const float*>(camera.t.data)),
                {3, 1},
                torch::kFloat32
            ).clone();
            
            // 组合参数
            // 根据模型的实际需求可能需要调整
            std::vector<torch::Tensor> params = {K_tensor, R_tensor, t_tensor};
            tensors.push_back(torch::cat(params, 1));  // 沿第二维拼接
        }
        
        return tensors;
    }
    
    /**
     * 加载图像
     * 
     * @param image_dir 图像目录
     * @param num_cameras 相机数量
     * @param frame_idx 帧索引
     * @return 图像列表
     */
    static std::vector<cv::Mat> loadImages(const std::string& image_dir, int num_cameras, int frame_idx) {
        std::vector<cv::Mat> images;
        
        try {
            for (int i = 0; i < num_cameras; i++) {
                // 构建图像文件名
                // 假设图像文件名格式为：camera_<camera_idx>_<frame_idx>.jpg
                std::string image_file = image_dir + "/camera_" + std::to_string(i) + "_" + 
                                        std::to_string(frame_idx) + ".jpg";
                
                // 检查文件是否存在
                if (!fs::exists(image_file)) {
                    throw std::runtime_error("图像文件不存在: " + image_file);
                }
                
                // 读取图像
                cv::Mat image = cv::imread(image_file);
                if (image.empty()) {
                    throw std::runtime_error("无法读取图像: " + image_file);
                }
                
                images.push_back(image);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "加载图像时出错: " << e.what() << std::endl;
            throw;
        }
        
        return images;
    }
}; 