#include "visualization.h"
#include "projection.h"
#include "utils.h"
#include <iostream>

void visualize_3d_pose(
    cv::Mat& image,
    const torch::Tensor& poses_3d,
    const CameraParams& camera_params,
    const torch::Tensor& resize_transform,
    const std::vector<std::pair<int, int>>& limbs,
    const std::vector<int>& original_image_size
) {
    // 检查姿态数据是否有效
    if (poses_3d.numel() == 0 || poses_3d.size(0) == 0) {
        return; // 没有姿态可绘制，返回原图
    }

    // 获取图像尺寸
    int img_h = image.rows;
    int img_w = image.cols;

    // --- 姿态处理 ---
    // poses_3d形状为(Batch=1, MaxPeople, NumJoints, 5)
    // 最后一维: 0:x, 1:y, 2:z, 3:score, 4:visibility
    
    auto poses_cpu = poses_3d.cpu();
    auto poses_accessor = poses_cpu.accessor<float, 4>(); // B, P, J, Dims
    int max_people = poses_3d.size(1);
    int num_joints = poses_3d.size(2);
    
    float min_pose_score = 0.2f; // 最小姿态置信度阈值

    for (int p = 0; p < max_people; ++p) {
        // 检查是否是真实人体（第4维度 >= 0）
        float is_real_person = poses_accessor[0][p][0][3]; // 所有关节点的值相同
        if (is_real_person < 0.0f) {
            continue; // 跳过非真实人体
        }
        
        // 获取人体置信度
        float person_confidence = poses_accessor[0][p][0][4]; // 所有关节点的值相同

        // 过滤掉低置信度的人体
        if (person_confidence < min_pose_score) {
            continue;
        }

        // 获取跟踪器分配的人体ID（存储在第4维度）
        int tracked_person_id = static_cast<int>(is_real_person);
        
        // 为每个人使用不同颜色（基于跟踪ID而不是数组索引）
        cv::Scalar color = VIS_COLORS[tracked_person_id % VIS_COLORS.size()];

        // 提取当前人的3D姿态点
        torch::Tensor current_pose_3d_pts = poses_3d.select(0,0).select(0,p).slice(1,0,3).contiguous(); // (NumJoints, 3)
        
        // 投影到2D并应用图像变换
        torch::Tensor pose_2d_projected = project_pose_cpp(current_pose_3d_pts, camera_params); // (NumJoints, 2)
        torch::Tensor pose_2d_transformed = affine_transform_pts_cpp(pose_2d_projected, resize_transform); // (NumJoints, 2)
        
        // 绘制关节点 - 使用更大的关节点以便区分不同的人
        int joint_radius = 5; // 增大关节点半径
        for (int j = 0; j < num_joints; ++j) {
            torch::Tensor pt_tensor = pose_2d_transformed[j];
            if (is_valid_coord_cpp(pt_tensor, img_w, img_h)) {
                cv::Point center(static_cast<int>(pt_tensor[0].item<float>()), static_cast<int>(pt_tensor[1].item<float>()));
                cv::circle(image, center, joint_radius, color, -1); // 填充圆圈
                // 添加白色边框使关节点更突出
                cv::circle(image, center, joint_radius, cv::Scalar(255, 255, 255), 1);
            }
        }

        // 绘制肢体（关节连接）- 使用更粗的线条
        int line_thickness = 3; // 增加线条粗细
        for (const auto& limb : limbs) {
            if (limb.first >= num_joints || limb.second >= num_joints) continue;

            torch::Tensor p1_tensor = pose_2d_transformed[limb.first];
            torch::Tensor p2_tensor = pose_2d_transformed[limb.second];

            if (is_valid_coord_cpp(p1_tensor, img_w, img_h) && is_valid_coord_cpp(p2_tensor, img_w, img_h)) {
                cv::Point p1(static_cast<int>(p1_tensor[0].item<float>()), static_cast<int>(p1_tensor[1].item<float>()));
                cv::Point p2(static_cast<int>(p2_tensor[0].item<float>()), static_cast<int>(p2_tensor[1].item<float>()));
                cv::line(image, p1, p2, color, line_thickness); // 更粗的线条
            }
        }

        // 添加人员编号标识 - 显示跟踪器分配的真实ID
        if (num_joints > 0) {
            torch::Tensor head_pt_tensor = pose_2d_transformed[0]; // 假设关节0是头部或颈部
            if (is_valid_coord_cpp(head_pt_tensor, img_w, img_h)) {
                cv::Point text_pos(
                    static_cast<int>(head_pt_tensor[0].item<float>()) - 10,
                    static_cast<int>(head_pt_tensor[1].item<float>()) - 15
                );
                std::string person_id = "ID" + std::to_string(tracked_person_id);
                
                // 绘制白色背景的文本，使其更加醒目
                cv::putText(image, person_id, text_pos, cv::FONT_HERSHEY_SIMPLEX, 
                           0.6, cv::Scalar(255, 255, 255), 4, cv::LINE_AA); // 白色背景
                cv::putText(image, person_id, text_pos, cv::FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2, cv::LINE_AA); // 彩色前景
            }
        }
    }
}

void save_image_with_poses_cpp(
    const std::string& output_prefix_str,
    const torch::Tensor& image_tensor_chw, // (Channels, Height, Width), RGB, float (model input)
    const torch::Tensor& poses_3d,         // (1, MaxPeople, NumJoints, 5 for x,y,z,score,visibility_or_similar)
    const CameraParams& camera_params,
    const torch::Tensor& resize_transform, // 2x3 affine transform matrix
    int view_idx,
    float min_pose_score,
    const std::vector<std::pair<int, int>>& limbs,
    const std::vector<int>& original_image_size // {ori_w, ori_h} for un-doing normalization if needed or context
) {
    // 创建输出目录
    fs::path output_prefix(output_prefix_str);
    fs::path dirname = output_prefix.parent_path() / "image_with_poses";
    if (!fs::exists(dirname)) {
        fs::create_directories(dirname);
    }
    std::string full_prefix = (dirname / output_prefix.filename()).string();
    std::string file_name = full_prefix + "_view_" + std::to_string(view_idx + 1) + ".jpg";

    // --- 图像张量准备 ---
    // image_tensor_chw是(C, H, W)，RGB格式，浮点型，已为模型输入标准化
    torch::Tensor vis_img_tensor = image_tensor_chw.clone();

    // 从模型输入反标准化（使用ImageNet标准均值/标准差）
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}, vis_img_tensor.options()).view({3, 1, 1});
    torch::Tensor std_dev = torch::tensor({0.229, 0.224, 0.225}, vis_img_tensor.options()).view({3, 1, 1});
    vis_img_tensor = vis_img_tensor * std_dev + mean;
    
    // 转换为适合OpenCV显示的格式
    vis_img_tensor = vis_img_tensor.mul(255.0f).clamp(0.0f, 255.0f).to(torch::kU8);
    vis_img_tensor = vis_img_tensor.permute({1, 2, 0}).cpu().contiguous(); // CHW -> HWC

    int img_h = vis_img_tensor.size(0);
    int img_w = vis_img_tensor.size(1);
    int channels = vis_img_tensor.size(2);

    // 转换为OpenCV格式
    cv::Mat vis_mat;
    if (channels == 3) {
        vis_mat = cv::Mat(img_h, img_w, CV_8UC3, vis_img_tensor.data_ptr<uint8_t>());
        cv::cvtColor(vis_mat, vis_mat, cv::COLOR_RGB2BGR); // 转换RGB为BGR（OpenCV格式）
    } else if (channels == 1) {
        vis_mat = cv::Mat(img_h, img_w, CV_8UC1, vis_img_tensor.data_ptr<uint8_t>());
    } else {
        std::cerr << "不支持的通道数: " << channels << std::endl;
        return;
    }
    cv::Mat output_display = vis_mat.clone(); // 在副本上操作

    visualize_3d_pose(output_display, poses_3d, camera_params, resize_transform, limbs, original_image_size);
    
    // 保存结果图像
    cv::imwrite(file_name, output_display);
} 