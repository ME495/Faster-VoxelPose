#pragma once

#include <torch/torch.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <fstream>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path, const torch::Device& device);
    ~TensorRTInference();
    
    // 执行推理
    torch::Tensor forward(const torch::Tensor& input);
    
    // 检查是否初始化成功
    bool isInitialized() const { return initialized_; }

private:
    bool loadEngine(const std::string& engine_path);
    bool setupBindings();
    void cleanUp();
    
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Logger logger_;
    
    std::vector<void*> bindings_;
    std::vector<size_t> binding_sizes_;
    
    // 输入输出信息
    int input_index_;
    int output_index_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    
    // CUDA 流
    cudaStream_t stream_;
    
    torch::Device device_;
    bool initialized_;
}; 