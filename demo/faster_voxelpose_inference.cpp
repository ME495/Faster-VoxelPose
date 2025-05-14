/**
 * Faster VoxelPose 推理实现
 * 使用libtorch (C++ PyTorch API) 实现Faster VoxelPose模型的推理
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "camera_utils.hpp"

// 配置结构体，对应YAML配置
struct FasterVoxelPoseConfig {
    // 设备配置
    std::string device = "cuda:0";
    
    // 数据集配置
    int num_joints = 15;
    int root_joint_id = 2;
    int camera_num = 5;
    std::vector<int> image_size = {960, 512};
    std::vector<int> heatmap_size = {240, 128};
    
    // 网络配置
    float sigma = 3.0f;
    float beta = 100.0f;
    
    // 空间配置
    std::vector<float> space_size = {8000.0f, 8000.0f, 2000.0f};
    std::vector<float> space_center = {0.0f, -500.0f, 800.0f};
    std::vector<int> voxels_per_axis = {80, 80, 20};
    int max_people = 10;
    float min_score = 0.3f;
    
    // 个体空间配置
    std::vector<float> individual_space_size = {2000.0f, 2000.0f, 2000.0f};
    std::vector<int> individual_voxels_per_axis = {64, 64, 64};
};

class FasterVoxelPoseInference {
public:
    /**
     * 初始化FasterVoxelPose推理模型
     * 
     * @param model_path 模型路径
     * @param backbone_path 骨干网络模型路径
     * @param config 配置参数
     */
    FasterVoxelPoseInference(const std::string& model_path, 
                             const std::string& backbone_path,
                             const FasterVoxelPoseConfig& config) 
        : config_(config) {
        
        // 设置设备
        device_ = torch::Device(config.device);
        
        try {
            // 加载骨干网络
            backbone_module_ = torch::jit::load(backbone_path);
            backbone_module_.to(device_);
            backbone_module_.eval();
            
            // 加载主模型
            model_module_ = torch::jit::load(model_path);
            model_module_.to(device_);
            model_module_.eval();
            
            std::cout << "模型加载成功" << std::endl;
        }
        catch (const c10::Error& e) {
            std::cerr << "模型加载失败: " << e.what() << std::endl;
            throw;
        }
    }
    
    /**
     * 处理多视角图像，进行3D姿态估计
     * 
     * @param images 多视角图像列表
     * @param cameras 相机参数列表
     * @return 返回检测到的人体3D姿态
     */
    std::vector<torch::Tensor> inference(const std::vector<cv::Mat>& images, 
                                        const std::vector<torch::Tensor>& cameras) {
        
        // 检查输入
        if (images.size() != config_.camera_num) {
            throw std::runtime_error("输入图像数量与配置的相机数量不匹配");
        }
        
        // 预处理图像
        std::vector<torch::Tensor> processed_images;
        for (const auto& image : images) {
            processed_images.push_back(preprocess_image(image));
        }
        
        // 将图像组合为批次
        torch::Tensor batch_images = torch::stack(processed_images, 1);  // [1, num_views, 3, H, W]
        batch_images = batch_images.to(device_);
        
        // 准备相机参数
        torch::Tensor camera_params = torch::stack(cameras, 0).unsqueeze(0).to(device_);  // [1, num_views, ...]
        
        // 创建元数据字典
        std::unordered_map<std::string, torch::Tensor> meta;
        
        // 创建缩放变换矩阵（如果需要）
        torch::Tensor resize_transform = torch::eye(3, torch::kFloat32).unsqueeze(0).repeat({config_.camera_num, 1, 1}).to(device_);
        
        // 使用torch::NoGradGuard禁用梯度计算，提高推理速度
        torch::NoGradGuard no_grad;
        
        try {
            // 第一步：使用骨干网络提取特征
            std::vector<torch::Tensor> view_features;
            for (int i = 0; i < config_.camera_num; i++) {
                // 提取每个视角的特征
                torch::Tensor view_image = batch_images.select(1, i);  // [1, 3, H, W]
                torch::Tensor feature = backbone_module_.forward({view_image}).toTensor();
                view_features.push_back(feature);
            }
            
            // 组合所有视角的特征
            torch::Tensor input_heatmaps = torch::stack(view_features, 1);  // [1, num_views, num_joints, h, w]
            
            // 第二步：使用FasterVoxelPose模型进行推理
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(torch::IValue());  // backbone参数为空，因为我们已经提取了特征
            inputs.push_back(torch::IValue());  // views参数为空，因为我们已经提取了特征
            inputs.push_back(c10::impl::toIValue(meta, c10::TensorType::get()));  // meta
            inputs.push_back(torch::IValue());  // targets为空，因为是推理模式
            inputs.push_back(input_heatmaps);   // input_heatmaps
            inputs.push_back(camera_params);    // cameras
            inputs.push_back(resize_transform); // resize_transform
            
            // 执行模型推理
            auto outputs = model_module_.forward(inputs).toTuple()->elements();
            
            // 解析输出
            torch::Tensor fused_poses = outputs[0].toTensor();       // [batch_size, max_people, num_joints, 5]
            torch::Tensor plane_poses = outputs[1].toTensor();       // [3, batch_size, max_people, num_joints, 2]
            torch::Tensor proposal_centers = outputs[2].toTensor();  // [batch_size, max_people, 7]
            
            // 处理结果：提取有效的人体姿态
            torch::Tensor mask = proposal_centers.select(2, 4) > config_.min_score;  // 置信度大于阈值的掩码
            
            // 提取有效的3D姿态（前3列是xyz坐标）
            torch::Tensor valid_poses = fused_poses.select(0, 0).index({mask, torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
            
            // 返回结果
            std::vector<torch::Tensor> result;
            result.push_back(valid_poses);
            result.push_back(proposal_centers.select(0, 0).index({mask}));  // 返回有效的人体中心点信息
            
            return result;
        }
        catch (const c10::Error& e) {
            std::cerr << "推理过程中出错: " << e.what() << std::endl;
            throw;
        }
    }
    
    /**
     * 可视化3D姿态
     * 
     * @param poses 3D姿态
     * @param output_file 输出文件路径
     */
    void visualizePoses(const torch::Tensor& poses, const std::string& output_file) {
        // 这里只是一个简单的示例，将姿态保存为CSV文件
        // 实际应用中可以使用OpenGL或其他3D可视化库进行可视化
        
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法打开输出文件: " << output_file << std::endl;
            return;
        }
        
        // 写入标题
        file << "person_id,joint_id,x,y,z" << std::endl;
        
        // 写入姿态数据
        for (int i = 0; i < poses.size(0); i++) {
            for (int j = 0; j < config_.num_joints; j++) {
                float x = poses[i][j][0].item<float>();
                float y = poses[i][j][1].item<float>();
                float z = poses[i][j][2].item<float>();
                
                file << i << "," << j << "," << x << "," << y << "," << z << std::endl;
            }
        }
        
        file.close();
        std::cout << "姿态已保存到: " << output_file << std::endl;
    }
    
private:
    // 图像预处理函数
    torch::Tensor preprocess_image(const cv::Mat& image) {
        // 调整图像大小
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(config_.image_size[0], config_.image_size[1]));
        
        // 转换为RGB格式（如果是BGR）
        cv::Mat rgb_image;
        cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
        
        // 转换为浮点数并归一化
        cv::Mat float_image;
        rgb_image.convertTo(float_image, CV_32F, 1.0/255.0);
        
        // 标准化
        cv::Mat channels[3];
        cv::split(float_image, channels);
        
        // 使用ImageNet均值和标准差
        channels[0] = (channels[0] - 0.485) / 0.229;  // R
        channels[1] = (channels[1] - 0.456) / 0.224;  // G
        channels[2] = (channels[2] - 0.406) / 0.225;  // B
        
        cv::merge(channels, 3, float_image);
        
        // 转换为torch::Tensor，形状为[C, H, W]
        torch::Tensor tensor_image = torch::from_blob(float_image.data, 
                                                     {float_image.rows, float_image.cols, 3}, 
                                                     torch::kFloat32).permute({2, 0, 1});
        
        // 添加批次维度
        return tensor_image.unsqueeze(0);  // [1, C, H, W]
    }
    
    FasterVoxelPoseConfig config_;
    torch::Device device_;
    torch::jit::script::Module backbone_module_;
    torch::jit::script::Module model_module_;
};

// 示例用法
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "用法: " << argv[0] << " <模型路径> <骨干网络路径> <相机参数文件> <图像目录> [帧索引] [输出文件]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string backbone_path = argv[2];
    std::string camera_file = argv[3];
    std::string image_dir = argv[4];
    
    // 可选参数
    int frame_idx = (argc > 5) ? std::stoi(argv[5]) : 0;
    std::string output_file = (argc > 6) ? argv[6] : "poses.csv";
    
    try {
        // 创建配置
        FasterVoxelPoseConfig config;
        
        // 加载相机参数
        std::vector<CameraParams> camera_params = CameraUtils::loadCameraParams(camera_file);
        if (camera_params.size() != config.camera_num) {
            std::cerr << "警告: 相机参数数量(" << camera_params.size() 
                      << ")与配置的相机数量(" << config.camera_num << ")不匹配" << std::endl;
            config.camera_num = camera_params.size();
        }
        
        // 转换相机参数为Torch张量
        std::vector<torch::Tensor> camera_tensors = CameraUtils::convertToTorchTensors(camera_params);
        
        // 加载图像
        std::vector<cv::Mat> images = CameraUtils::loadImages(image_dir, config.camera_num, frame_idx);
        
        // 初始化推理模型
        FasterVoxelPoseInference inference(model_path, backbone_path, config);
        
        // 执行推理
        std::vector<torch::Tensor> results = inference.inference(images, camera_tensors);
        
        // 处理结果
        if (!results.empty()) {
            torch::Tensor poses = results[0];
            torch::Tensor centers = results[1];
            
            std::cout << "检测到 " << poses.size(0) << " 个人体" << std::endl;
            
            // 输出每个人的姿态
            for (int i = 0; i < poses.size(0); i++) {
                std::cout << "人体 #" << i << ":" << std::endl;
                std::cout << "  中心位置: " 
                          << centers[i][0].item<float>() << ", " 
                          << centers[i][1].item<float>() << ", " 
                          << centers[i][2].item<float>() << std::endl;
                std::cout << "  置信度: " << centers[i][4].item<float>() << std::endl;
            }
            
            // 可视化姿态
            inference.visualizePoses(poses, output_file);
        } else {
            std::cout << "未检测到人体" << std::endl;
        }
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
} 