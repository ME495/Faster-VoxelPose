#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // For cv::imwrite
#include <opencv2/imgproc.hpp>   // For cv::circle, cv::line, cv::cvtColor
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>   // For std::ifstream
#include <chrono>    // For timing
#include <filesystem> // For directory iteration (C++17)
#include <algorithm> // For std::sort, std::min
#include <iomanip>   // For std::fixed, std::setprecision
#include <future>    // For std::async
#include <map>       // For LIMBS definition
#include <cmath>     // For M_PI

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// Helper struct for Camera Parameters
struct CameraParams {
    torch::Tensor R; // Rotation matrix (3x3), world to camera
    torch::Tensor T; // Translation vector (3x1), camera center in world coordinates (C_w)
                     // such that x_cam = R @ (x_world - T)
    torch::Tensor K_intrinsic; // Intrinsics matrix (3x3)
    torch::Tensor f; // Focal length (fx, fy) (2x1)
    torch::Tensor c; // Principal point (cx, cy) (2x1)
    torch::Tensor k_dist; // Radial distortion coefficients (k1, k2, k3) (3x1)
    torch::Tensor p_dist; // Tangential distortion coefficients (p1, p2) (2x1)
    std::string id; // camera id like "44310001"

    CameraParams() :
        R(torch::eye(3, torch::kFloat32)),
        T(torch::zeros({3,1}, torch::kFloat32)),
        K_intrinsic(torch::eye(3, torch::kFloat32)),
        f(torch::ones({2,1}, torch::kFloat32) * 1000.0),
        c(torch::zeros({2,1}, torch::kFloat32)),
        k_dist(torch::zeros({3,1}, torch::kFloat32)),
        p_dist(torch::zeros({2,1}, torch::kFloat32)),
        id("unknown")
    {}

    CameraParams to(torch::Device device) {
        R = R.to(device);
        T = T.to(device);
        K_intrinsic = K_intrinsic.to(device);
        f = f.to(device);
        c = c.to(device);
        k_dist = k_dist.to(device);
        p_dist = p_dist.to(device);
        return *this;
    }
};

// Function to list image files in a directory, sorted alphabetically
std::vector<std::string> list_image_files(const std::string& dir_path) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff") {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << " while accessing path: " << dir_path << std::endl;
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Function to get affine transform for resizing (mimics utils.transforms.get_affine_transform behavior for r=0, shift=0)
torch::Tensor get_resize_affine_transform_cpp(const std::vector<int>& ori_size, 
                                              const std::vector<int>& new_size, 
                                              torch::Device device) {
    // ori_size: {width, height} of original image
    // new_size: {width, height} of target image

    // 1. Calculate center of original image
    std::vector<float> center = {
        static_cast<float>(ori_size[0]) / 2.0f,
        static_cast<float>(ori_size[1]) / 2.0f
    };

    // 2. Calculate scale based on the Python get_scale logic
    float w = static_cast<float>(ori_size[0]);
    float h = static_cast<float>(ori_size[1]);
    float w_resized = static_cast<float>(new_size[0]);
    float h_resized = static_cast<float>(new_size[1]);
    
    float w_pad, h_pad;
    if (w / w_resized < h / h_resized) {
        w_pad = h / h_resized * w_resized;
        h_pad = h;
    } else {
        w_pad = w;
        h_pad = w / w_resized * h_resized;
    }
    
    std::vector<float> scale = {w_pad / 200.0f, h_pad / 200.0f};
    
    // Define output size and other parameters aligned with Python implementation
    float rot = 0.0f; // No rotation for resize operation
    std::vector<float> shift = {0.0f, 0.0f}; // No shift
    bool inv = false;

    // Follow the Python implementation logic
    std::vector<float> scale_tmp = {scale[0] * 200.0f, scale[1] * 200.0f};
    float src_w = scale_tmp[0];
    float src_h = scale_tmp[1];
    float dst_w = static_cast<float>(new_size[0]);
    float dst_h = static_cast<float>(new_size[1]);

    float rot_rad = M_PI * rot / 180.0f;
    std::vector<float> src_dir, dst_dir;
    
    if (src_w >= src_h) {
        // Calculate direction vector based on rotation
        src_dir = {
            static_cast<float>(0.0f - std::sin(rot_rad) * src_w * 0.5f),
            static_cast<float>(0.0f - std::cos(rot_rad) * src_w * 0.5f)
        };
        dst_dir = {0.0f, dst_w * -0.5f};
    } else {
        src_dir = {
            static_cast<float>(std::cos(rot_rad) * src_h * 0.5f),
            static_cast<float>(-std::sin(rot_rad) * src_h * 0.5f)
        };
        dst_dir = {dst_h * -0.5f, 0.0f};
    }

    // Setup source and destination points for affine transform
    std::vector<cv::Point2f> src_pts(3);
    std::vector<cv::Point2f> dst_pts(3);

    // Point 1: center point
    src_pts[0] = cv::Point2f(center[0] + scale_tmp[0] * shift[0], center[1] + scale_tmp[1] * shift[1]);
    dst_pts[0] = cv::Point2f(dst_w * 0.5f, dst_h * 0.5f);

    // Point 2: center + direction
    src_pts[1] = cv::Point2f(center[0] + src_dir[0] + scale_tmp[0] * shift[0], 
                            center[1] + src_dir[1] + scale_tmp[1] * shift[1]);
    dst_pts[1] = cv::Point2f(dst_w * 0.5f + dst_dir[0], dst_h * 0.5f + dst_dir[1]);

    // Point 3: get third point to form a triangle (similar to get_3rd_point in Python)
    src_pts[2] = cv::Point2f(src_pts[0].x - src_pts[1].y + src_pts[0].y,
                            src_pts[0].y + src_pts[1].x - src_pts[0].x);
    dst_pts[2] = cv::Point2f(dst_pts[0].x - dst_pts[1].y + dst_pts[0].y,
                            dst_pts[0].y + dst_pts[1].x - dst_pts[0].x);

    // Use OpenCV's getAffineTransform (equivalent to Python's cv2.getAffineTransform)
    cv::Mat trans_cv;
    if (inv) {
        trans_cv = cv::getAffineTransform(dst_pts, src_pts);
    } else {
        trans_cv = cv::getAffineTransform(src_pts, dst_pts);
    }

    std::cout << trans_cv << std::endl;

    // Convert OpenCV Mat to torch::Tensor
    torch::Tensor trans = torch::zeros({2, 3}, torch::kFloat32);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            trans[i][j] = trans_cv.at<double>(i, j);
        }
    }

    std::cout << trans << std::endl;

    trans = trans.to(device);

    return trans;
}

std::vector<CameraParams> load_cameras_from_json(const std::string& json_path, torch::Device device) {
    std::vector<CameraParams> cameras;
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        std::cerr << "Error opening calibration file: " << json_path << std::endl;
        return cameras;
    }
    json calib_data;
    try {
        calib_data = json::parse(ifs);
    } catch (json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
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
            std::cerr << "Error parsing camera " << cam_id_str << ": " << e.what() << std::endl;
        }
    }
    // Sort cameras by ID string to ensure consistent order for processing
    std::sort(cameras.begin(), cameras.end(), [](const CameraParams& a, const CameraParams& b) {
        return a.id < b.id;
    });
    return cameras;
}

// Forward declarations (some already exist)
torch::Tensor compute_grid_cpp(const std::vector<float>& boxSize, const std::vector<float>& boxCenter, const std::vector<int>& nBins, torch::Device device);
torch::Tensor project_point_cpp(torch::Tensor x, const torch::Tensor& R, const torch::Tensor& T, const torch::Tensor& f, const torch::Tensor& c, const torch::Tensor& k, const torch::Tensor& p);
torch::Tensor affine_transform_pts_cpp(torch::Tensor pts, torch::Tensor t);
torch::Tensor project_grid_cpp(torch::Tensor grid, const std::vector<int>& ori_image_size_vec, const std::vector<int>& image_size_vec, const CameraParams& camera, int heatmap_w, int heatmap_h, int nbins, torch::Tensor resize_transform, torch::Device device);
torch::Tensor load_and_preprocess_image(const std::string& image_path, const std::vector<int>& target_image_size, torch::Device device);

// ------------- Visualization Code Start -------------

// Define limb connections based on number of joints.
// This is a placeholder. You should define it based on your dataset/model's joint definition.
// Example for 17 COCO-style joints:
const std::vector<std::pair<int, int>> LIMBS17 = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},         // Head (Nose to L/R Eye, L/R Eye to L/R Ear)
    {5, 6},                                 // Shoulders
    {5, 7}, {7, 9},                         // Left Arm
    {6, 8}, {8, 10},                        // Right Arm
    {11, 12},                               // Hips
    {5, 11}, {6, 12},                       // Torso
    {11, 13}, {13, 15},                     // Left Leg
    {12, 14}, {14, 16}                      // Right Leg
};

const std::vector<std::pair<int, int>> LIMBS15 = {
    {0, 1}, {0, 2}, {0, 3}, {3, 4}, {4, 5}, 
    {0, 9}, {9, 10}, {10, 11}, {2, 6}, {2, 12}, 
    {6, 7}, {7, 8}, {12, 13}, {13, 14}
};

std::vector<std::pair<int, int>> get_limbs_definition(int num_joints) {
    if (num_joints == 17) {
        return LIMBS17;
    } else if (num_joints == 15) {
        return LIMBS15;
    }
    std::cerr << "Warning: Limb definition for " << num_joints << " joints not found. Returning empty limbs." << std::endl;
    return {};
}

// Define a simple color palette (BGR format for OpenCV)
const std::vector<cv::Scalar> VIS_COLORS = {
    cv::Scalar(255, 0, 0),   // Blue
    cv::Scalar(0, 255, 0),   // Green
    cv::Scalar(0, 0, 255),   // Red
    cv::Scalar(255, 255, 0), // Cyan
    cv::Scalar(255, 0, 255), // Magenta
    cv::Scalar(0, 255, 255), // Yellow
};

bool is_valid_coord_cpp(const torch::Tensor& pt, int width, int height) {
    float x = pt[0].item<float>();
    float y = pt[1].item<float>();
    return x >= 0 && x < width && y >= 0 && y < height;
}

// Project a set of 3D joints to 2D image plane
// poses_3d_person: (NumJoints, 3) tensor
// Returns: (NumJoints, 2) tensor
torch::Tensor project_pose_cpp(
    const torch::Tensor& poses_3d_person,
    const CameraParams& camera) {
    
    // 直接将整个关节点张量传入，批量处理所有关节点
    // poses_3d_person 形状为 (NumJoints, 3)
    return project_point_cpp(
        poses_3d_person, camera.R, camera.T, camera.f, camera.c, camera.k_dist, camera.p_dist);
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
    fs::path output_prefix(output_prefix_str);
    fs::path dirname = output_prefix.parent_path() / "image_with_poses";
    if (!fs::exists(dirname)) {
        fs::create_directories(dirname);
    }
    std::string full_prefix = (dirname / output_prefix.filename()).string();
    std::string file_name = full_prefix + "_view_" + std::to_string(view_idx + 1) + ".jpg";

    // --- Image Tensor Preparation ---
    // image_tensor_chw is (C, H, W), RGB, float, normalized for model input
    // Based on Python: make_grid(normalize=True) -> then mul(255).clamp().byte().permute()
    // normalize=True in make_grid shifts to [0,1] range.
    
    torch::Tensor vis_img_tensor = image_tensor_chw.clone();

    // Denormalize from model input (example for standard ImageNet mean/std)
    // This step is crucial if you want to see images that are not just the normalized ones.
    // If load_and_preprocess_image applies normalization, we should approximately reverse it.
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}, vis_img_tensor.options()).view({3, 1, 1});
    torch::Tensor std_dev = torch::tensor({0.229, 0.224, 0.225}, vis_img_tensor.options()).view({3, 1, 1});
    vis_img_tensor = vis_img_tensor * std_dev + mean;
    
    vis_img_tensor = vis_img_tensor.mul(255.0f).clamp(0.0f, 255.0f).to(torch::kU8);
    vis_img_tensor = vis_img_tensor.permute({1, 2, 0}).cpu().contiguous(); // CHW -> HWC

    int img_h = vis_img_tensor.size(0);
    int img_w = vis_img_tensor.size(1);
    int channels = vis_img_tensor.size(2);

    cv::Mat vis_mat;
    if (channels == 3) {
        vis_mat = cv::Mat(img_h, img_w, CV_8UC3, vis_img_tensor.data_ptr<uint8_t>());
        cv::cvtColor(vis_mat, vis_mat, cv::COLOR_RGB2BGR); // Convert RGB to BGR for OpenCV
    } else if (channels == 1) {
        vis_mat = cv::Mat(img_h, img_w, CV_8UC1, vis_img_tensor.data_ptr<uint8_t>());
    } else {
        std::cerr << "Unsupported channel count for visualization: " << channels << std::endl;
        return;
    }
    cv::Mat output_display = vis_mat.clone(); // Work on a copy

    // --- Poses Processing ---
    // poses_3d shape is (Batch=1, MaxPeople, NumJoints, 5)
    // Last dim: 0:x, 1:y, 2:z, 3:score, 4:visibility (assumption, Python used poses[...,0,4] for score)
    
    if (poses_3d.size(0) == 0) return; // No poses to draw

    auto poses_cpu = poses_3d.cpu();
    auto poses_accessor = poses_cpu.accessor<float, 4>(); // B, P, J, Dims
    int max_people = poses_3d.size(1);
    int num_joints = poses_3d.size(2);

    for (int p = 0; p < max_people; ++p) {
        // Use score of a prominent joint (e.g., joint 0) or a dedicated person score if available
        // Python code: poses[i, n, 0, 4] < config.CAPTURE_SPEC.MIN_SCORE
        // Assuming index 3 for score here based on typical (x,y,z,score) or 4 if (x,y,z,score_joint,vis_person_score)
        float person_score = (num_joints > 0 && poses_3d.size(3) > 3) ? poses_accessor[0][p][0][3] : 0.0f; // Example: score of joint 0
        if (poses_3d.size(3) > 4) { // If 5th element exists, use it as per Python's poses[i,n,0,4]
             person_score = poses_accessor[0][p][0][4];
        }


        if (person_score < min_pose_score) {
            continue;
        }

        cv::Scalar color = VIS_COLORS[p % VIS_COLORS.size()];

        torch::Tensor current_pose_3d_pts = poses_3d.select(0,0).select(0,p).slice(1,0,3).contiguous(); // (NumJoints, 3)
        
        torch::Tensor pose_2d_projected = project_pose_cpp(current_pose_3d_pts, camera_params); // (NumJoints, 2)
        torch::Tensor pose_2d_transformed = affine_transform_pts_cpp(pose_2d_projected, resize_transform); // (NumJoints, 2)
        
        // Draw joints
        for (int j = 0; j < num_joints; ++j) {
            torch::Tensor pt_tensor = pose_2d_transformed[j];
            if (is_valid_coord_cpp(pt_tensor, img_w, img_h)) {
                cv::Point center(static_cast<int>(pt_tensor[0].item<float>()), static_cast<int>(pt_tensor[1].item<float>()));
                cv::circle(output_display, center, 4, color, -1); // Radius 4
            }
        }

        // Draw limbs
        for (const auto& limb : limbs) {
            if (limb.first >= num_joints || limb.second >= num_joints) continue;

            torch::Tensor p1_tensor = pose_2d_transformed[limb.first];
            torch::Tensor p2_tensor = pose_2d_transformed[limb.second];

            if (is_valid_coord_cpp(p1_tensor, img_w, img_h) && is_valid_coord_cpp(p2_tensor, img_w, img_h)) {
                cv::Point p1(static_cast<int>(p1_tensor[0].item<float>()), static_cast<int>(p1_tensor[1].item<float>()));
                cv::Point p2(static_cast<int>(p2_tensor[0].item<float>()), static_cast<int>(p2_tensor[1].item<float>()));
                cv::line(output_display, p1, p2, color, 2); // Thickness 2
            }
        }
    }
    cv::imwrite(file_name, output_display);
    // std::cout << "Saved visualization to " << file_name << std::endl;
}

// ------------- Visualization Code End -------------

int main(int argc, const char* argv[]) {
    if (argc < 6) { // Prog_name, backbone, model, calib.json, image_base_dir, device
        std::cerr << "Usage: faster_voxelpose_demo <path_to_scripted_backbone.pt> \
                     <path_to_scripted_model.pt> <path_to_calibration.json> \
                     <image_sequence_base_dir> <device (cpu/cuda)> [num_frames_to_process]" << std::endl;
        return -1;
    }

    std::string backbone_path = argv[1];
    std::string model_path = argv[2];
    std::string calib_json_path = argv[3];
    std::string image_base_dir = argv[4];
    std::string device_str = argv[5];
    int num_frames_to_process_arg = -1; // Process all if not specified
    if (argc > 6) {
        try {
            num_frames_to_process_arg = std::stoi(argv[6]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid number for num_frames_to_process: " << argv[6] << std::endl;
            return -1;
        }
    }

    torch::DeviceType device_type;
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "Using CUDA device." << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "Using CPU device." << std::endl;
    }
    torch::Device device(device_type);

    // --- Configuration values from configs/custom/jln64.yaml --- 
    std::vector<int> ori_image_size_cfg = {2048, 1544};      
    std::vector<int> image_size_cfg = {1024, 784};           
    std::vector<int> heatmap_size_cfg = {256, 200};          
    const int NUM_CAMERAS = 4; // Defined by requirement

    std::vector<float> space_size_cfg = {8000.0f, 8000.0f, 2000.0f}; 
    std::vector<float> space_center_cfg = {0.0f, -300.0f, 800.0f};  
    std::vector<int> voxels_per_axis_cfg = {80, 80, 20};           
    std::vector<float> individual_space_size_cfg = {2000.0f, 2000.0f, 2000.0f};
    std::vector<int> ind_voxels_per_axis_cfg = {64, 64, 64};      
    // --- End Configuration ---

    // Load Camera Parameters
    std::vector<CameraParams> cameras = load_cameras_from_json(calib_json_path, device);
    if (cameras.size() < NUM_CAMERAS) {
        std::cerr << "Error: Loaded " << cameras.size() << " cameras, expected " << NUM_CAMERAS << std::endl;
        return -1;
    }
    // Ensure we only use NUM_CAMERAS, in case json has more but sorted by ID.
    if (cameras.size() > NUM_CAMERAS) {
         cameras.resize(NUM_CAMERAS);
    }

    // List image files for each camera
    std::vector<std::vector<std::string>> camera_image_files(NUM_CAMERAS);
    size_t min_frames = std::string::npos;

    for (int i = 0; i < NUM_CAMERAS; ++i) {
        fs::path cam_dir = fs::path(image_base_dir) / cameras[i].id;
        camera_image_files[i] = list_image_files(cam_dir.string());
        if (camera_image_files[i].empty()) {
            std::cerr << "Error: No image files found in directory: " << cam_dir.string() << std::endl;
            return -1;
        }
        if (min_frames == std::string::npos || camera_image_files[i].size() < min_frames) {
            min_frames = camera_image_files[i].size();
        }
        std::cout << "Found " << camera_image_files[i].size() << " images for camera " << cameras[i].id << std::endl;
    }

    if (min_frames == 0 || min_frames == std::string::npos) {
        std::cerr << "Error: No frames to process or inconsistent frame counts across cameras." << std::endl;
        return -1;
    }
    std::cout << "Processing a sequence of " << min_frames << " synchronized frames." << std::endl;

    int frames_to_run = (num_frames_to_process_arg > 0 && num_frames_to_process_arg < min_frames) ? num_frames_to_process_arg : min_frames;
    std::cout << "Will run for " << frames_to_run << " frames." << std::endl;

    torch::jit::script::Module backbone_module, model_module;
    try {
        backbone_module = torch::jit::load(backbone_path, device);
        backbone_module.eval();
        model_module = torch::jit::load(model_path, device);
        model_module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model(s):\n" << e.what() << std::endl;
        return -1;
    }

    // --- Precompute Grids (these are static for the sequence) ---
    torch::Tensor resize_transform_tensor = get_resize_affine_transform_cpp(ori_image_size_cfg, image_size_cfg, device);
    torch::Tensor grid_coarse = compute_grid_cpp(space_size_cfg, space_center_cfg, voxels_per_axis_cfg, device);
    int nbins_coarse = voxels_per_axis_cfg[0] * voxels_per_axis_cfg[1] * voxels_per_axis_cfg[2];
    std::vector<torch::Tensor> sample_grids_coarse_list_static;

    torch::Tensor world_space_size_tensor = torch::tensor(space_size_cfg, device);
    torch::Tensor ind_space_size_tensor = torch::tensor(individual_space_size_cfg, device);
    torch::Tensor ind_voxels_per_axis_tensor = torch::tensor(ind_voxels_per_axis_cfg, torch::kInt).to(device);
    torch::Tensor fine_voxels_per_axis_float = (world_space_size_tensor / ind_space_size_tensor * (ind_voxels_per_axis_tensor.to(torch::kFloat) - 1.0f)) + 1.0f;
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
        sample_grids_coarse_list_static.push_back(sg_coarse_cam.view({voxels_per_axis_cfg[0], voxels_per_axis_cfg[1], voxels_per_axis_cfg[2], 2}));

        torch::Tensor sg_fine_cam = project_grid_cpp(grid_fine, ori_image_size_cfg, image_size_cfg, 
                                                      cameras[i], heatmap_size_cfg[0], heatmap_size_cfg[1],
                                                      nbins_fine, resize_transform_tensor, device);
        sample_grids_fine_list_static.push_back(sg_fine_cam.view({fine_voxels_per_axis_cfg[0], fine_voxels_per_axis_cfg[1], fine_voxels_per_axis_cfg[2], 2}));
    }
    torch::Tensor final_sample_grid_coarse = torch::stack(sample_grids_coarse_list_static, 0).unsqueeze(0);
    torch::Tensor final_sample_grid_fine = torch::stack(sample_grids_fine_list_static, 0).unsqueeze(0);
    // --- End Precompute Grids ---

    torch::NoGradGuard no_grad;
    long long total_duration_ms = 0;
    long long total_image_processing_ms = 0;
    long long total_heatmap_extraction_ms = 0;
    long long total_pose_estimation_ms = 0;
    int frames_processed_for_fps = 0;
    int warmup_frames = 10;

    std::cout << "Starting processing loop for " << frames_to_run << " frames..." << std::endl;

    for (int frame_idx = 0; frame_idx < frames_to_run; ++frame_idx) {
        if (frame_idx % 50 == 0 || frame_idx == frames_to_run -1) {
             std::cout << "Processing frame " << frame_idx + 1 << "/" << frames_to_run << std::endl;
        }

        auto overall_frame_start_time = std::chrono::high_resolution_clock::now();

        // --- 1. 图像处理计时开始 ---
        auto image_processing_start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::future<torch::Tensor>> image_futures;
        for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
            image_futures.push_back(std::async(std::launch::async, load_and_preprocess_image, 
                                    camera_image_files[cam_idx][frame_idx], image_size_cfg, device));
        }

        std::vector<torch::Tensor> batch_tensors;
        bool has_error = false;
        for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
            torch::Tensor img_tensor = image_futures[cam_idx].get();
            if (img_tensor.numel() == 0) {
                std::cerr << "Skipping frame due to image load error for camera " << cam_idx << std::endl;
                has_error = true;
                break;
            }
            batch_tensors.push_back(img_tensor);
        }

        if (has_error || batch_tensors.size() != NUM_CAMERAS) {
            std::cerr << "Error: Not enough images loaded for frame " << frame_idx << std::endl;
            continue;
        }
        // --- 1. 图像处理计时结束 ---
        auto image_processing_end_time = std::chrono::high_resolution_clock::now();
        if (device_type == torch::kCUDA) { // 确保所有可能的to(device)操作完成
            torch::cuda::synchronize();
            image_processing_end_time = std::chrono::high_resolution_clock::now(); // 重新获取时间
        }

        // --- 2. Heatmap提取计时开始 ---
        auto heatmap_extraction_start_time = std::chrono::high_resolution_clock::now();

        torch::Tensor batch_input = torch::cat(batch_tensors, 0);  // [NUM_CAMERAS, 3, H, W]
        torch::Tensor batch_heatmaps = backbone_module.forward({batch_input}).toTensor().unsqueeze(0);
        
        // --- 2. Heatmap提取计时结束 ---
        auto heatmap_extraction_end_time = std::chrono::high_resolution_clock::now();
        if (device_type == torch::kCUDA) {
            torch::cuda::synchronize();
            heatmap_extraction_end_time = std::chrono::high_resolution_clock::now(); // 重新获取时间
        }

        // --- 3. 姿态估计计时开始 ---
        auto pose_estimation_start_time = std::chrono::high_resolution_clock::now();

        std::vector<torch::jit::IValue> model_inputs;
        model_inputs.push_back(batch_heatmaps);
        model_inputs.push_back(final_sample_grid_coarse); 
        model_inputs.push_back(final_sample_grid_fine);

        torch::Tensor fused_poses = model_module.forward(model_inputs).toTensor();
        
        // --- 3. 姿态估计计时结束 ---
        auto pose_estimation_end_time = std::chrono::high_resolution_clock::now();
        if (device_type == torch::kCUDA) {
            torch::cuda::synchronize(); 
            pose_estimation_end_time = std::chrono::high_resolution_clock::now(); // 重新获取时间
        }

        auto overall_frame_end_time = std::chrono::high_resolution_clock::now();
        
        if (frame_idx >= warmup_frames) { // Start timing after warmup
            auto image_processing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(image_processing_end_time - image_processing_start_time);
            total_image_processing_ms += image_processing_duration.count();

            auto heatmap_extraction_duration = std::chrono::duration_cast<std::chrono::milliseconds>(heatmap_extraction_end_time - heatmap_extraction_start_time);
            total_heatmap_extraction_ms += heatmap_extraction_duration.count();

            auto pose_estimation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pose_estimation_end_time - pose_estimation_start_time);
            total_pose_estimation_ms += pose_estimation_duration.count();
            
            auto overall_frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_frame_end_time - overall_frame_start_time);
            total_duration_ms += overall_frame_duration.count();
            frames_processed_for_fps++;
        }

        // --- Visualization Call (example: save every 20 frames after warmup) ---
        if (frame_idx >= warmup_frames && frame_idx % 20 == 0) {
            // fused_poses is assumed (1, MaxPeople, NumJoints, 5)
            // batch_tensors contains NUM_CAMERAS tensors, each (1, C, H, W)
            // These are the preprocessed images fed to the backbone.
			// fused_poses = fused_poses.to(torch::kCPU); // Move to CPU for visualization
            if (fused_poses.defined() && fused_poses.numel() > 0 && fused_poses.dim() == 4 && batch_tensors.size() == NUM_CAMERAS) {
                int num_joints_from_pose = fused_poses.size(2);
                std::vector<std::pair<int, int>> current_limbs = get_limbs_definition(num_joints_from_pose);

                for (int view_idx = 0; view_idx < NUM_CAMERAS; ++view_idx) {
                    std::string vis_output_prefix = "output_frame_" + std::to_string(frame_idx);
                    // batch_tensors[view_idx] is (1, C, H, W). Squeeze to (C,H,W) for the function.
                    // Or modify save_image_with_poses_cpp to take (B,C,H,W) and handle B internally if needed.
                    // For now, assume B=1 for visualization.
                    if (batch_tensors[view_idx].size(0) == 1) { // Ensure batch size is 1 for this image
                         save_image_with_poses_cpp(
                            vis_output_prefix,
                            batch_tensors[view_idx].squeeze(0), // Pass (C,H,W)
                            fused_poses,                        // Pass (1, MaxPeople, NumJoints, 5)
                            cameras[view_idx],
                            resize_transform_tensor,            // This is the 2x3 affine transform
                            view_idx,
                            0.2f,                               // Min pose score example
                            current_limbs,
                            ori_image_size_cfg                  // Pass original image size for context
                        );
                    }
                }
            }
        }
    }

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Finished processing image sequence." << std::endl;
    if (frames_processed_for_fps > 0) {
        double avg_overall_time_per_frame_ms = static_cast<double>(total_duration_ms) / frames_processed_for_fps;
        double avg_image_processing_ms = static_cast<double>(total_image_processing_ms) / frames_processed_for_fps;
        double avg_heatmap_extraction_ms = static_cast<double>(total_heatmap_extraction_ms) / frames_processed_for_fps;
        double avg_pose_estimation_ms = static_cast<double>(total_pose_estimation_ms) / frames_processed_for_fps;
        double fps = 1000.0 / avg_overall_time_per_frame_ms;

        std::cout << "Frames processed for FPS: " << frames_processed_for_fps << std::endl;
        std::cout << "Total processing time (after warmup): " << total_duration_ms << " ms" << std::endl;
        std::cout << "Average time per frame (Overall): " << std::fixed << std::setprecision(2) << avg_overall_time_per_frame_ms << " ms" << std::endl;
        std::cout << "  Average Image Processing time: " << std::fixed << std::setprecision(2) << avg_image_processing_ms << " ms" << std::endl;
        std::cout << "  Average Heatmap Extraction time: " << std::fixed << std::setprecision(2) << avg_heatmap_extraction_ms << " ms" << std::endl;
        std::cout << "  Average Pose Estimation time: " << std::fixed << std::setprecision(2) << avg_pose_estimation_ms << " ms" << std::endl;
        std::cout << "FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
    } else {
        std::cout << "Not enough frames processed after warmup to calculate FPS." << std::endl;
    }
    std::cout << "-------------------------------------" << std::endl;

    return 0;
}

// --- Implementations for Helper Functions --- 
// (compute_grid_cpp, project_point_cpp, affine_transform_pts_cpp, project_grid_cpp as previously defined)

torch::Tensor compute_grid_cpp(const std::vector<float>& boxSize_vec, 
                               const std::vector<float>& boxCenter_vec, 
                               const std::vector<int>& nBins_vec, 
                               torch::Device device) {
    torch::Tensor boxSize = torch::tensor(boxSize_vec, device);
    torch::Tensor boxCenter = torch::tensor(boxCenter_vec, device);
    torch::Tensor grid1Dx = torch::linspace(-boxSize[0].item<float>() / 2.0f, boxSize[0].item<float>() / 2.0f, nBins_vec[0], device);
    torch::Tensor grid1Dy = torch::linspace(-boxSize[1].item<float>() / 2.0f, boxSize[1].item<float>() / 2.0f, nBins_vec[1], device);
    torch::Tensor grid1Dz = torch::linspace(-boxSize[2].item<float>() / 2.0f, boxSize[2].item<float>() / 2.0f, nBins_vec[2], device);
    std::vector<torch::Tensor> grids_v = torch::meshgrid({grid1Dx + boxCenter[0].item<float>(), 
                                                          grid1Dy + boxCenter[1].item<float>(), 
                                                          grid1Dz + boxCenter[2].item<float>()}, "ij");
    torch::Tensor gridx = grids_v[0].contiguous().view({-1, 1});
    torch::Tensor gridy = grids_v[1].contiguous().view({-1, 1});
    torch::Tensor gridz = grids_v[2].contiguous().view({-1, 1});
    return torch::cat({gridx, gridy, gridz}, 1);
}

torch::Tensor project_point_cpp(torch::Tensor x, 
                              const torch::Tensor& R, const torch::Tensor& T, 
                              const torch::Tensor& f, const torch::Tensor& c, 
                              const torch::Tensor& k_dist, const torch::Tensor& p_dist
                              ) {
    torch::Tensor x_minus_T = x.transpose(0, 1) - T;
    torch::Tensor xcam = torch::mm(R, x_minus_T);
    torch::Tensor y = xcam.slice(0, 0, 2) / (xcam.slice(0, 2, 3) + 1e-5);
    torch::Tensor r_sq = torch::sum(y * y, 0);
    torch::Tensor d_radial = 1.0f + k_dist[0] * r_sq + k_dist[1] * r_sq * r_sq + k_dist[2] * r_sq * r_sq * r_sq;
    torch::Tensor y0 = y.slice(0, 0, 1);
    torch::Tensor y1 = y.slice(0, 1, 2);
    torch::Tensor u_tangential = 2 * p_dist[0] * y0 * y1 + p_dist[1] * (r_sq + 2 * y0 * y0);
    torch::Tensor v_tangential = 2 * p_dist[1] * y0 * y1 + p_dist[0] * (r_sq + 2 * y1 * y1);
    torch::Tensor u = y0 * d_radial + u_tangential;
    torch::Tensor v = y1 * d_radial + v_tangential;
    torch::Tensor y_distorted = torch::cat({u, v}, 0);
    torch::Tensor ypixel = f * y_distorted + c;
    return ypixel.transpose(0, 1);
}

torch::Tensor affine_transform_pts_cpp(torch::Tensor pts, torch::Tensor t) {
    int64_t npts = pts.size(0);
    torch::Tensor ones = torch::ones({npts, 1}, pts.options());
    torch::Tensor pts_homo = torch::cat({pts, ones}, 1);
    torch::Tensor out_homo = torch::mm(t, pts_homo.transpose(0, 1)); 
    return out_homo.slice(0,0,2).transpose(0,1);
}

torch::Tensor project_grid_cpp(torch::Tensor grid, 
                             const std::vector<int>& ori_image_size_vec,
                             const std::vector<int>& image_size_vec,
                             const CameraParams& camera, 
                             int heatmap_w, int heatmap_h, 
                             int nbins, 
                             torch::Tensor resize_transform, 
                             torch::Device device) {
    torch::Tensor xy = project_point_cpp(grid, camera.R, camera.T, camera.f, camera.c, camera.k_dist, camera.p_dist);
    float clamp_max = static_cast<float>(std::max(ori_image_size_vec[0], ori_image_size_vec[1]));
    xy = torch::clamp(xy, -1.0f, clamp_max);
    xy = affine_transform_pts_cpp(xy, resize_transform);
    torch::Tensor wh_heatmap = torch::tensor({static_cast<float>(heatmap_w), static_cast<float>(heatmap_h)}, device);
    torch::Tensor wh_image_size = torch::tensor({static_cast<float>(image_size_vec[0]), static_cast<float>(image_size_vec[1])}, device);
    xy = xy * wh_heatmap / wh_image_size;
    torch::Tensor wh_minus_1 = torch::tensor({static_cast<float>(heatmap_w - 1), static_cast<float>(heatmap_h - 1)}, device);
    torch::Tensor sample_grid_normalized = (xy / wh_minus_1) * 2.0f - 1.0f;
    sample_grid_normalized = sample_grid_normalized.view({1, 1, nbins, 2}); // This shape is for F.grid_sample compatibility
    sample_grid_normalized = torch::clamp(sample_grid_normalized, -1.1f, 1.1f);
    return sample_grid_normalized;
}

// load_and_preprocess_image function remains as previously defined.

torch::Tensor load_and_preprocess_image(const std::string& image_path, 
                                        const std::vector<int>& target_image_size, // {width, height}
                                        torch::Device device) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        return torch::Tensor();
    }

    // Resize image to target_image_size (this is what backbone expects)
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(target_image_size[0], target_image_size[1]));

    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);

    torch::Tensor img_tensor = torch::from_blob(resized_img.data, {resized_img.rows, resized_img.cols, 3}, torch::kFloat32).to(device);
    img_tensor = img_tensor.permute({2, 0, 1}); // HWC to CHW

    // Normalization (values from validate_ts.py)
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}, device).view({3, 1, 1});
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}, device).view({3, 1, 1});
    img_tensor = (img_tensor - mean) / std;

    return img_tensor.unsqueeze(0); // Add batch dimension
} 
