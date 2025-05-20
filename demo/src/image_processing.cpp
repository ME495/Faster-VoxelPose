#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

torch::Tensor load_and_preprocess_image(
    const std::string& image_path, 
    const std::vector<int>& target_image_size, // {width, height}
    torch::Device device) {
    
    // 读取图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "错误：无法加载图像: " << image_path << std::endl;
        return torch::Tensor();
    }

    // 调整图像大小到target_image_size（这是主干网络期望的输入尺寸）
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(target_image_size[0], target_image_size[1]));

    // 转换为RGB（OpenCV默认是BGR）并归一化到[0,1]范围
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);

    // 将OpenCV的Mat转换为torch::Tensor
    torch::Tensor img_tensor = torch::from_blob(
        resized_img.data, 
        {resized_img.rows, resized_img.cols, 3}, 
        torch::kFloat32
    ).to(device);
    
    // 调整通道顺序：HWC转CHW
    img_tensor = img_tensor.permute({2, 0, 1});

    // 标准化（ImageNet预训练模型通常使用的均值和标准差）
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}, device).view({3, 1, 1});
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}, device).view({3, 1, 1});
    img_tensor = (img_tensor - mean) / std;

    // 添加批次维度
    return img_tensor.unsqueeze(0);
} 