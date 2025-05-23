#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>
#include <algorithm>
#include <set>

// 人体实例结构
struct PersonInstance {
    int person_id;                      // 唯一的人体ID
    torch::Tensor pose_3d;             // 3D姿态 (num_joints, 3)
    float confidence;                  // 人体置信度
    int last_seen_frame;               // 最后看到的帧号
    
    PersonInstance() : person_id(-1), confidence(0.0f), last_seen_frame(-1) {}
    PersonInstance(int id, const torch::Tensor& pose, float conf, int frame) 
        : person_id(id), pose_3d(pose.clone()), confidence(conf), last_seen_frame(frame) {}
};

// 人体跟踪器类
class PersonTracker {
private:
    int next_person_id_;                           // 下一个分配的人体ID
    std::vector<PersonInstance> tracked_persons_;  // 当前跟踪的人体列表
    int max_frame_gap_;                            // 最大丢失帧数，超过则移除人体
    float distance_threshold_;                     // 距离阈值，用于匹配人体
    
public:
    PersonTracker(int max_frame_gap = 10, float distance_threshold = 500.0f);
    
    // 更新跟踪器，输入新一帧的姿态数据，返回分配了ID的姿态数据
    torch::Tensor update(const torch::Tensor& poses_3d, int current_frame);
    
    // 计算两个3D姿态之间的距离（基于关节点的欧式距离）
    float compute_pose_distance(const torch::Tensor& pose1, const torch::Tensor& pose2);
    
    // 从姿态张量中提取有效人体（置信度>=0）
    std::vector<std::pair<torch::Tensor, float>> extract_valid_persons(const torch::Tensor& poses_3d);
    
    // 使用匈牙利算法进行人体匹配
    std::vector<std::pair<int, int>> hungarian_assignment(
        const std::vector<std::vector<float>>& cost_matrix);
    
    // 清理长时间未见的人体
    void cleanup_lost_persons(int current_frame);
    
    // 获取当前跟踪的人体数量
    int get_tracked_person_count() const { return tracked_persons_.size(); }
    
    // 重置跟踪器
    void reset();
}; 