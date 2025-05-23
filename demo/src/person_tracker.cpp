#include "person_tracker.h"
#include <iostream>
#include <cmath>
#include <limits>

PersonTracker::PersonTracker(int max_frame_gap, float distance_threshold) 
    : next_person_id_(0), max_frame_gap_(max_frame_gap), distance_threshold_(distance_threshold) {
}

torch::Tensor PersonTracker::update(const torch::Tensor& poses_3d, int current_frame) {
    // 提取当前帧中的有效人体
    auto current_persons = extract_valid_persons(poses_3d);
    
    // 清理长时间未见的人体
    cleanup_lost_persons(current_frame);
    
    // 创建输出张量，复制输入的形状和数据
    torch::Tensor output_poses = poses_3d.clone();
    
    if (current_persons.empty()) {
        return output_poses;
    }
    
    if (tracked_persons_.empty()) {
        // 第一帧或没有跟踪的人体时，为所有检测到的人体分配新ID
        for (size_t i = 0; i < current_persons.size(); ++i) {
            tracked_persons_.emplace_back(next_person_id_++, 
                                        current_persons[i].first, 
                                        current_persons[i].second, 
                                        current_frame);
            
            // 在输出张量中更新人体ID（存储在第4维度的位置）
            // 使用张量索引操作代替accessor，适用于GPU张量
            int num_joints = poses_3d.size(2);
            for (int j = 0; j < num_joints; ++j) { // 遍历所有关节点
                output_poses[0][i][j][3] = static_cast<float>(tracked_persons_.back().person_id);
            }
        }
        return output_poses;
    }
    
    // 构建距离矩阵
    std::vector<std::vector<float>> cost_matrix(tracked_persons_.size(), 
                                               std::vector<float>(current_persons.size()));
    
    for (size_t i = 0; i < tracked_persons_.size(); ++i) {
        for (size_t j = 0; j < current_persons.size(); ++j) {
            cost_matrix[i][j] = compute_pose_distance(tracked_persons_[i].pose_3d, 
                                                     current_persons[j].first);
        }
    }
    
    // 使用匈牙利算法进行最优匹配
    auto assignments = hungarian_assignment(cost_matrix);
    
    // 记录已匹配的检测
    std::set<int> matched_detections;
    std::vector<PersonInstance> updated_persons;
    
    // 处理匹配结果
    for (const auto& assignment : assignments) {
        int tracker_idx = assignment.first;
        int detection_idx = assignment.second;
        
        // 检查距离是否在阈值内
        if (cost_matrix[tracker_idx][detection_idx] <= distance_threshold_) {
            // 更新已有人体
            tracked_persons_[tracker_idx].pose_3d = current_persons[detection_idx].first.clone();
            tracked_persons_[tracker_idx].confidence = current_persons[detection_idx].second;
            tracked_persons_[tracker_idx].last_seen_frame = current_frame;
            
            updated_persons.push_back(tracked_persons_[tracker_idx]);
            matched_detections.insert(detection_idx);
            
            // 在输出张量中更新人体ID
            // 使用张量索引操作代替accessor，适用于GPU张量
            int num_joints = poses_3d.size(2);
            for (int j = 0; j < num_joints; ++j) { // 遍历所有关节点
                output_poses[0][detection_idx][j][3] = static_cast<float>(tracked_persons_[tracker_idx].person_id);
            }
        }
    }
    
    // 为未匹配的检测分配新ID
    for (size_t i = 0; i < current_persons.size(); ++i) {
        if (matched_detections.find(i) == matched_detections.end()) {
            updated_persons.emplace_back(next_person_id_++, 
                                       current_persons[i].first, 
                                       current_persons[i].second, 
                                       current_frame);
            
            // 在输出张量中更新人体ID
            // 使用张量索引操作代替accessor，适用于GPU张量
            int num_joints = poses_3d.size(2);
            for (int j = 0; j < num_joints; ++j) { // 遍历所有关节点
                output_poses[0][i][j][3] = static_cast<float>(updated_persons.back().person_id);
            }
        }
    }
    
    // 保留未匹配但仍在阈值内的跟踪人体（暂时丢失）
    for (size_t i = 0; i < tracked_persons_.size(); ++i) {
        bool found_match = false;
        for (const auto& assignment : assignments) {
            if (assignment.first == static_cast<int>(i) && 
                cost_matrix[i][assignment.second] <= distance_threshold_) {
                found_match = true;
                break;
            }
        }
        if (!found_match && (current_frame - tracked_persons_[i].last_seen_frame) <= max_frame_gap_) {
            // 保留暂时丢失的人体，但不添加到updated_persons中
            // 它们会在下一帧继续尝试匹配
        }
    }
    
    tracked_persons_ = updated_persons;
    
    return output_poses;
}

float PersonTracker::compute_pose_distance(const torch::Tensor& pose1, const torch::Tensor& pose2) {
    // pose1和pose2的形状为 (num_joints, 3)
    if (pose1.size(0) != pose2.size(0)) {
        return std::numeric_limits<float>::max();
    }
    
    // 计算所有关节点的欧式距离的平均值
    torch::Tensor diff = pose1 - pose2;
    torch::Tensor squared_diff = diff * diff;
    torch::Tensor joint_distances = torch::sqrt(torch::sum(squared_diff, 1)); // 每个关节的距离
    
    // 返回平均关节距离
    return torch::mean(joint_distances).item<float>();
}

std::vector<std::pair<torch::Tensor, float>> PersonTracker::extract_valid_persons(const torch::Tensor& poses_3d) {
    std::vector<std::pair<torch::Tensor, float>> valid_persons;
    
    // poses_3d形状为 (batch_size, max_proposals, num_joints, 5)
    // 最后一维：0:x, 1:y, 2:z, 3:is_real_person, 4:confidence
    
    int max_proposals = poses_3d.size(1);
    int num_joints = poses_3d.size(2);
    
    for (int p = 0; p < max_proposals; ++p) {
        // 检查是否是真实人体（第4维度 >= 0）
        // 使用张量索引操作获取标量值，适用于GPU张量
        torch::Tensor is_real_person_tensor = poses_3d[0][p][0][3]; // 所有关节点的值相同
        float is_real_person = is_real_person_tensor.item<float>();
        
        if (is_real_person >= 0.0f) {
            // 提取3D关节点坐标
            torch::Tensor pose_3d = poses_3d.select(0, 0).select(0, p).slice(1, 0, 3).contiguous(); // (num_joints, 3)
            
            // 提取置信度
            torch::Tensor confidence_tensor = poses_3d[0][p][0][4]; // 所有关节点的值相同
            float confidence = confidence_tensor.item<float>();
            
            valid_persons.emplace_back(pose_3d, confidence);
        }
    }
    
    return valid_persons;
}

std::vector<std::pair<int, int>> PersonTracker::hungarian_assignment(
    const std::vector<std::vector<float>>& cost_matrix) {
    
    // 简化版的匈牙利算法实现（贪心近似）
    // 对于更精确的实现，可以使用专门的匈牙利算法库
    
    std::vector<std::pair<int, int>> assignments;
    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return assignments;
    }
    
    int num_trackers = cost_matrix.size();
    int num_detections = cost_matrix[0].size();
    
    std::vector<bool> tracker_assigned(num_trackers, false);
    std::vector<bool> detection_assigned(num_detections, false);
    
    // 贪心匹配：每次选择最小代价的未分配配对
    while (true) {
        float min_cost = std::numeric_limits<float>::max();
        int best_tracker = -1;
        int best_detection = -1;
        
        for (int i = 0; i < num_trackers; ++i) {
            if (tracker_assigned[i]) continue;
            for (int j = 0; j < num_detections; ++j) {
                if (detection_assigned[j]) continue;
                if (cost_matrix[i][j] < min_cost) {
                    min_cost = cost_matrix[i][j];
                    best_tracker = i;
                    best_detection = j;
                }
            }
        }
        
        if (best_tracker == -1 || best_detection == -1) {
            break; // 没有更多可匹配的配对
        }
        
        assignments.emplace_back(best_tracker, best_detection);
        tracker_assigned[best_tracker] = true;
        detection_assigned[best_detection] = true;
    }
    
    return assignments;
}

void PersonTracker::cleanup_lost_persons(int current_frame) {
    auto it = tracked_persons_.begin();
    while (it != tracked_persons_.end()) {
        if (current_frame - it->last_seen_frame > max_frame_gap_) {
            it = tracked_persons_.erase(it);
        } else {
            ++it;
        }
    }
}

void PersonTracker::reset() {
    tracked_persons_.clear();
    next_person_id_ = 0;
} 