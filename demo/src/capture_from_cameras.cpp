#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <future>
#include <algorithm>
#include <iomanip>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handle.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <opencv2/opencv.hpp>

// GigE Camera includes
#include <stdio.h>
#include <conio.h>
#include "GigELib.h"
#pragma comment(lib, "ws2_32.lib")
#include <windows.h>
#include <omp.h>

// VoxelPose includes
#include "types.h"
#include "camera.h"
#include "projection.h"
#include "utils.h"
#include "visualization.h"
#include "image_processing.h"
#include "tensorrt_inference.h"
#include "person_tracker.h"

using namespace QUICKCAM;
using namespace cv;
using namespace std;

int main(int argc, const char* argv[]) {
    if (argc < 5) {
        std::cerr << "用法: capture_from_cameras <path_to_scripted_backbone.pt> "
                     "<path_to_scripted_model.pt> <path_to_calibration.json> "
                     "<device (cpu/cuda)> [bind_ip_address]" << std::endl;
        return -1;
    }

    std::string backbone_path = argv[1];
    std::string model_path = argv[2];
    std::string calib_json_path = argv[3];
    std::string device_str = argv[4];
    std::string bind_ip = (argc > 5) ? argv[5] : "192.168.1.250";

    // 设置设备(CPU/CUDA)
    torch::DeviceType device_type;
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "使用CUDA设备." << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "使用CPU设备." << std::endl;
    }
    torch::Device device(device_type);

    // 设置torch线程数
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads > 8) {
        torch::set_num_threads(8);
        torch::set_num_interop_threads(8);
        std::cout << "已设置torch线程数: " << 8 << std::endl;
    }

    // --- 配置参数 ---
    std::vector<int> ori_image_size_cfg = {2048, 1544};      
    std::vector<int> image_size_cfg = {1024, 784};           
    std::vector<int> heatmap_size_cfg = {256, 200};          
    const int NUM_CAMERAS = 4;
    unsigned int framerate = 30;

    std::vector<float> space_size_cfg = {8000.0f, 8000.0f, 2000.0f}; 
    std::vector<float> space_center_cfg = {0.0f, -300.0f, 800.0f};  
    std::vector<int> voxels_per_axis_cfg = {80, 80, 20};           
    std::vector<float> individual_space_size_cfg = {2000.0f, 2000.0f, 2000.0f};
    std::vector<int> ind_voxels_per_axis_cfg = {64, 64, 64};      

    // 加载相机参数
    std::vector<CameraParams> cameras = load_cameras_from_json(calib_json_path, device);
    if (cameras.size() < NUM_CAMERAS) {
        std::cerr << "错误: 加载了 " << cameras.size() << " 个相机, 需要 " << NUM_CAMERAS << std::endl;
        return -1;
    }
    if (cameras.size() > NUM_CAMERAS) {
         cameras.resize(NUM_CAMERAS);
    }

    // 加载模型
    std::unique_ptr<TensorRTInference> backbone_trt;
    torch::jit::script::Module model_module;
    try {
        if (backbone_path.substr(backbone_path.find_last_of(".") + 1) == "trt") {
            std::cout << "加载 TensorRT backbone 引擎: " << backbone_path << std::endl;
            backbone_trt = std::make_unique<TensorRTInference>(backbone_path, device);
            if (!backbone_trt->isInitialized()) {
                throw std::runtime_error("TensorRT backbone 初始化失败");
            }
        } else {
            throw std::runtime_error("backbone 文件必须是 .trt 格式");
        }
        
        model_module = torch::jit::load(model_path, device);
        model_module.eval();
    } catch (const std::exception& e) {
        std::cerr << "加载模型时出错:\n" << e.what() << std::endl;
        return -1;
    }

    // --- 预计算网格 ---
    torch::Tensor resize_transform_tensor = get_resize_affine_transform_cpp(ori_image_size_cfg, image_size_cfg, device);
    torch::Tensor grid_coarse = compute_grid_cpp(space_size_cfg, space_center_cfg, voxels_per_axis_cfg, device);
    int nbins_coarse = voxels_per_axis_cfg[0] * voxels_per_axis_cfg[1] * voxels_per_axis_cfg[2];
    std::vector<torch::Tensor> sample_grids_coarse_list_static;

    torch::Tensor world_space_size_tensor = torch::tensor(space_size_cfg, device);
    torch::Tensor ind_space_size_tensor = torch::tensor(individual_space_size_cfg, device);
    torch::Tensor ind_voxels_per_axis_tensor = torch::tensor(ind_voxels_per_axis_cfg, torch::kInt).to(device);
    torch::Tensor fine_voxels_per_axis_float = (world_space_size_tensor / ind_space_size_tensor * 
                                             (ind_voxels_per_axis_tensor.to(torch::kFloat) - 1.0f)) + 1.0f;
    std::vector<int> fine_voxels_per_axis_cfg(3);
    fine_voxels_per_axis_cfg[0] = static_cast<int>(fine_voxels_per_axis_float[0].item<float>());
    fine_voxels_per_axis_cfg[1] = static_cast<int>(fine_voxels_per_axis_float[1].item<float>());
    fine_voxels_per_axis_cfg[2] = static_cast<int>(fine_voxels_per_axis_float[2].item<float>());
    torch::Tensor grid_fine = compute_grid_cpp(space_size_cfg, space_center_cfg, fine_voxels_per_axis_cfg, device);
    int nbins_fine = fine_voxels_per_axis_cfg[0] * fine_voxels_per_axis_cfg[1] * fine_voxels_per_axis_cfg[2];
    std::vector<torch::Tensor> sample_grids_fine_list_static;

    for (int i = 0; i < NUM_CAMERAS; ++i) {
        torch::Tensor sg_coarse_cam = project_grid_cpp(grid_coarse, ori_image_size_cfg, image_size_cfg, 
                                                        cameras[i], heatmap_size_cfg[0], heatmap_size_cfg[1],
                                                        nbins_coarse, resize_transform_tensor, device);
        sample_grids_coarse_list_static.push_back(sg_coarse_cam.view({voxels_per_axis_cfg[0], 
                                                                     voxels_per_axis_cfg[1], 
                                                                     voxels_per_axis_cfg[2], 2}));

        torch::Tensor sg_fine_cam = project_grid_cpp(grid_fine, ori_image_size_cfg, image_size_cfg, 
                                                      cameras[i], heatmap_size_cfg[0], heatmap_size_cfg[1],
                                                      nbins_fine, resize_transform_tensor, device);
        sample_grids_fine_list_static.push_back(sg_fine_cam.view({fine_voxels_per_axis_cfg[0], 
                                                                 fine_voxels_per_axis_cfg[1], 
                                                                 fine_voxels_per_axis_cfg[2], 2}));
    }
    torch::Tensor final_sample_grid_coarse = torch::stack(sample_grids_coarse_list_static, 0).unsqueeze(0);
    torch::Tensor final_sample_grid_fine = torch::stack(sample_grids_fine_list_static, 0).unsqueeze(0);

    // --- 初始化人体跟踪器 ---
    PersonTracker person_tracker(10, 500.0f);
    std::cout << "已初始化人体跟踪器 (最大丢失帧数: 10, 距离阈值: 500.0)" << std::endl;

    // --- 初始化GigE相机 ---
    QCGuid guid;
    GigECamera *cam[10] = {NULL};
    BusManager busMgr;
    Property ppt;
    Image img[10];
    vector<Mat> cvImgs(10);
    Mat imgConcat;

    unsigned int camNum = 0, R3Num = 0;
    
    // 启动相机管理器
    BusManager::StartManager(htonl(inet_addr(bind_ip.c_str())));
    busMgr.RequestEnumerateCameras();
    busMgr.GetNumOfCameras(&camNum);
    
    if(camNum == 0) {
        printf("没有找到ChingMU GigE相机!\n");
        goto camera_cleanup;
    } else {
        printf("找到 %d 个相机:\n", camNum);
        for(int i=0; i<camNum; ++i) {
            uint32_t camSerial = 0;
            busMgr.GetCameraSerialNumberFromIndex(i, &camSerial);
            printf("%08d\n", camSerial);
        }
    }

    // 连接R3相机
    printf("连接R3相机:\n");
    for(int i=0; i<camNum && i<200; ++i) {
        DeviceInfo devInfo;
        busMgr.GetCameraFromIndex(i, &guid);
        busMgr.GetDeviceInfo(&guid, &devInfo);
        if(!strcmp((char*)devInfo.user_defined_name, "R3"))
            cam[R3Num] = new GigECamera();
        else
            continue;
    
        if(cam[R3Num]->Connect(&guid) != GEV_STATUS_SUCCESS) {
            printf("连接相机[%08s] 失败!\n", devInfo.serial_number);
            goto camera_cleanup;
        }
    
        // 设置相机帧率
        ppt.type = PT_FRAME_RATE;
        ppt.valueA = framerate;
        if(cam[R3Num]->SetProperty(&ppt) != GEV_STATUS_SUCCESS) {
            printf("设置相机[%08s] fps[%d] 失败!\n", devInfo.serial_number, ppt.valueA);
            goto camera_cleanup;
        }

        // 设置相机传输图像
        bool lock = true;
        cam[R3Num]->SetTransmitImageLock(&lock);

        // 开始捕获图像
        if(cam[R3Num]->StartCapture()!= GEV_STATUS_SUCCESS) {
            printf("启动相机[%08s] 捕获图像失败\n", devInfo.serial_number);
            goto camera_cleanup;
        }

        R3Num++;
        if(R3Num >= NUM_CAMERAS) break; // 只使用需要的相机数量
    }

    if(R3Num < NUM_CAMERAS) {
        printf("错误: 只找到 %d 个R3相机, 需要 %d 个\n", R3Num, NUM_CAMERAS);
        goto camera_cleanup;
    }

    camNum = R3Num;
    
    // 创建OpenCV窗口
    namedWindow("Pose Estimation Results", WINDOW_NORMAL);
    resizeWindow("Pose Estimation Results", 1024, 768);
    
    printf("开始捕获和处理图像，按 Q 键退出\n");
    
    // 初始化FPS计算变量
    int frames = 0;
    double fps = 0.0;
    double freq = getTickFrequency();
    int64 lastTime = getTickCount();
    int frame_idx = 0;
    
    // 初始化性能统计变量
    long long total_duration_ms = 0;
    long long total_pose_estimation_ms = 0;
    int frames_processed_for_fps = 0;
    int warmup_frames = 10;
    
    // --- 主处理循环 ---
    {
        torch::NoGradGuard no_grad;

        while(1) {
            if(_kbhit() && _getch() == 'q')
                break;

            auto overall_frame_start_time = std::chrono::high_resolution_clock::now();
            
            long long timestamp = 0x7FFFFFFFFFFFFFFF;

            // 1. 获取所有相机图像
            for(int i=0; i<camNum; ++i) {
                if(cam[i]->RetrieveBuffer(&img[i])!= GEV_STATUS_SUCCESS)
                    continue;
                
                // 按时间戳分组帧
                if(timestamp > img[i].GetEmbeddedTimestamp())
                    timestamp = img[i].GetEmbeddedTimestamp();
            }

            // 2. 取消较新的帧
            for(int i=0; i<camNum; ++i) {
                if(img[i].GetEmbeddedTimestamp() - timestamp > 1000000/framerate/2)
                    cam[i]->CancelLastRetrievedBuffer();
            }

            // 3. 处理图像并转换为tensor
            std::vector<torch::Tensor> batch_tensors;
            bool all_images_valid = true;
            
            for(int i=0; i<camNum; ++i) {
                if(img[i].GetData() && img[i].GetDataSize()) {
                    int width = img[i].GetCols();
                    int height = img[i].GetRows();
                    
                    if(width == ori_image_size_cfg[0] && height == ori_image_size_cfg[1]) {
                        // 创建OpenCV Mat
                        Mat temp(height, width, CV_8UC3, img[i].GetData(), img[i].GetStride());
                        
                        // 等比例缩放到目标尺寸，不足部分用0填充
                        Mat resized_img = Mat::zeros(image_size_cfg[1], image_size_cfg[0], CV_8UC3);
                        
                        // 计算缩放比例，取较小的比例以确保图像完全在目标尺寸内
                        double scale_x = static_cast<double>(image_size_cfg[0]) / width;
                        double scale_y = static_cast<double>(image_size_cfg[1]) / height;
                        double scale = min(scale_x, scale_y);
                        
                        // 计算缩放后的尺寸
                        int new_width = static_cast<int>(width * scale);
                        int new_height = static_cast<int>(height * scale);
                        
                        // 缩放图像
                        Mat scaled_img;
                        resize(temp, scaled_img, Size(new_width, new_height));
                        
                        // 计算在目标图像中的位置（居中）
                        int x_offset = (image_size_cfg[0] - new_width) / 2;
                        int y_offset = (image_size_cfg[1] - new_height) / 2;
                        
                        // 将缩放后的图像复制到目标图像的中央
                        scaled_img.copyTo(resized_img(Rect(x_offset, y_offset, new_width, new_height)));
                        
                        // 归一化并转换为tensor
                        Mat normalized_img;
                        // 使用ImageNet标准归一化：先归一化到[0,1]，再减均值除以标准差
                        resized_img.convertTo(normalized_img, CV_32FC3, 1.0f / 255.0f);
                        std::vector<cv::Mat> channels(3);
                        cv::split(normalized_img, channels);
                        const float mean[3] = {0.485f, 0.456f, 0.406f};
                        const float std[3] = {0.229f, 0.224f, 0.225f};
                        for (int c = 0; c < 3; ++c) {
                            channels[c] = (channels[c] - mean[c]) / std[c];
                        }
                        cv::merge(channels, normalized_img);
                        
                        torch::Tensor img_tensor = torch::from_blob(
                            normalized_img.data,
                            {normalized_img.rows, normalized_img.cols, 3},
                            torch::kFloat32
                        ).clone().permute({2, 0, 1}).unsqueeze(0).to(device);
                        
                        batch_tensors.push_back(img_tensor);
                        
                        // 保存原图用于可视化
                        if(cvImgs[i].empty() || cvImgs[i].size() != resized_img.size()) {
                            cvImgs[i] = Mat(temp.size(), CV_8UC3);
                        }
                        resized_img.copyTo(cvImgs[i]);
                    } else {
                        all_images_valid = false;
                        break;
                    }
                } else {
                    all_images_valid = false;
                    break;
                }
            }

            if(!all_images_valid || batch_tensors.size() != NUM_CAMERAS) {
                continue; // 跳过这一帧
            }

            // 4. 姿态估计
            auto pose_estimation_start_time = std::chrono::high_resolution_clock::now();
            
            // Heatmap提取
            torch::Tensor batch_input = torch::cat(batch_tensors, 0);  // [NUM_CAMERAS, 3, H, W]
            batch_input = batch_input.to(torch::kHalf);
            torch::Tensor batch_heatmaps = backbone_trt->forward(batch_input);
            batch_heatmaps = batch_heatmaps.to(torch::kFloat);
            batch_heatmaps = batch_heatmaps.unsqueeze(0);

            // 姿态估计
            std::vector<torch::jit::IValue> model_inputs;
            model_inputs.push_back(batch_heatmaps);
            model_inputs.push_back(final_sample_grid_coarse); 
            model_inputs.push_back(final_sample_grid_fine);
            torch::Tensor fused_poses = model_module.forward(model_inputs).toTensor();
            
            // 人体跟踪
            if (fused_poses.defined() && fused_poses.numel() > 0) {
                fused_poses = person_tracker.update(fused_poses, frame_idx);
            }
            
            auto pose_estimation_end_time = std::chrono::high_resolution_clock::now();
            if (device_type == torch::kCUDA) {
                torch::cuda::synchronize(); 
                pose_estimation_end_time = std::chrono::high_resolution_clock::now();
            }

            // 5. 可视化
            Mat visualization_img;
            if (fused_poses.defined() && fused_poses.numel() > 0 && fused_poses.dim() == 4) {
                // 选择第一个相机视角进行可视化
                visualization_img = cvImgs[0];
                
                // 获取关节点定义
                int num_joints_from_pose = fused_poses.size(2);
                std::vector<std::pair<int, int>> current_limbs = get_limbs_definition(num_joints_from_pose);
                
                // 使用visualization.cpp中的函数进行3D姿态可视化
                visualize_3d_pose(
                    visualization_img,
                    fused_poses,
                    cameras[0],  // 使用第一个相机的参数
                    resize_transform_tensor,
                    current_limbs,
                    ori_image_size_cfg
                );
            } else {
                // 没有检测到姿态时显示原图
                visualization_img = cvImgs[0];
            }

            // 添加FPS信息
            frames++;
            if (frames % 10 == 0) {
                int64 now = getTickCount();
                double seconds = (now - lastTime) / freq;
                fps = 10.0 / seconds;
                lastTime = now;
            }
            
            // 在图像上显示FPS和跟踪信息
            std::string fps_text = "FPS: " + std::to_string((int)fps);
            std::string person_text = "Persons: " + std::to_string(person_tracker.get_tracked_person_count());
            putText(visualization_img, fps_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            putText(visualization_img, person_text, Point(10, 70), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            
            // 显示结果
            imshow("Pose Estimation Results", visualization_img);
            
            // 检查按键
            int key = waitKey(1);
            if(key == 'q' || key == 'Q' || key == 27) // q, Q or ESC to exit
                break;

            auto overall_frame_end_time = std::chrono::high_resolution_clock::now();
            
            // 计时统计
            if (frame_idx >= warmup_frames) {
                auto pose_estimation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    pose_estimation_end_time - pose_estimation_start_time);
                total_pose_estimation_ms += pose_estimation_duration.count();
                auto overall_frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    overall_frame_end_time - overall_frame_start_time);
                total_duration_ms += overall_frame_duration.count();
                frames_processed_for_fps++;
            }
            
            frame_idx++;
            
            // 每100帧输出一次统计信息
            if (frame_idx % 100 == 0 && frames_processed_for_fps > 0) {
                double avg_overall_time_per_frame_ms = static_cast<double>(total_duration_ms) / frames_processed_for_fps;
                double avg_pose_estimation_ms = static_cast<double>(total_pose_estimation_ms) / frames_processed_for_fps;
                double processing_fps = 1000.0 / avg_overall_time_per_frame_ms;
                
                std::cout << "帧 " << frame_idx << ": 平均处理时间 " << std::fixed << std::setprecision(2) 
                         << avg_overall_time_per_frame_ms << " ms, 姿态估计 " << avg_pose_estimation_ms 
                         << " ms, 处理FPS: " << processing_fps << std::endl;
            }
        }
    } // torch::NoGradGuard 作用域结束

    // 输出最终统计
    if (frames_processed_for_fps > 0) {
        double avg_overall_time_per_frame_ms = static_cast<double>(total_duration_ms) / frames_processed_for_fps;
        double avg_pose_estimation_ms = static_cast<double>(total_pose_estimation_ms) / frames_processed_for_fps;
        double processing_fps = 1000.0 / avg_overall_time_per_frame_ms;
        
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "最终统计 (热身后 " << frames_processed_for_fps << " 帧):" << std::endl;
        std::cout << "平均每帧处理时间: " << std::fixed << std::setprecision(2) 
                 << avg_overall_time_per_frame_ms << " 毫秒" << std::endl;
        std::cout << "平均姿态估计时间: " << std::fixed << std::setprecision(2) 
                 << avg_pose_estimation_ms << " 毫秒" << std::endl;
        std::cout << "处理FPS: " << std::fixed << std::setprecision(2) << processing_fps << std::endl;
        std::cout << "-------------------------------------" << std::endl;
    }

camera_cleanup:
    // 清理相机资源
    for(int i=0; i<camNum; ++i) {
        if(cam[i] && cam[i]->IsConnected()) {
            bool lock = false;
            cam[i]->SetTransmitImageLock(&lock);
            cam[i]->StopCapture();
            cam[i]->Disconnect();
            delete cam[i];
            cam[i] = NULL;
        }
    }
    
    // 释放OpenCV资源
    cvImgs.clear();
    imgConcat.release();
    destroyAllWindows();
    
    std::cout << "程序结束." << std::endl;
    
    return 0;
}
