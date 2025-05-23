#include "tensorrt_inference.h"
#include <iostream>
#include <cassert>
#include <cuda_fp16.h>

TensorRTInference::TensorRTInference(const std::string& engine_path, const torch::Device& device)
    : device_(device), initialized_(false), input_index_(-1), output_index_(-1) {
    
    // 创建 CUDA 流
    cudaStreamCreate(&stream_);
    
    // 加载引擎
    if (loadEngine(engine_path)) {
        if (setupBindings()) {
            initialized_ = true;
            std::cout << "TensorRT 引擎初始化成功" << std::endl;
        }
    }
    
    if (!initialized_) {
        cleanUp();
        std::cerr << "TensorRT 引擎初始化失败" << std::endl;
    }
}

TensorRTInference::~TensorRTInference() {
    cleanUp();
}

bool TensorRTInference::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "无法打开引擎文件: " << engine_path << std::endl;
        return false;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取引擎数据
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    // 创建 runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "创建 TensorRT runtime 失败" << std::endl;
        return false;
    }
    
    // 反序列化引擎
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "反序列化 TensorRT 引擎失败" << std::endl;
        return false;
    }
    
    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "创建 TensorRT 执行上下文失败" << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTInference::setupBindings() {
    int num_bindings = engine_->getNbBindings();
    if (num_bindings != 2) {
        std::cerr << "期望 2 个绑定 (输入和输出), 但得到 " << num_bindings << std::endl;
        return false;
    }
    
    bindings_.resize(num_bindings);
    binding_sizes_.resize(num_bindings);
    
    for (int i = 0; i < num_bindings; ++i) {
        if (engine_->bindingIsInput(i)) {
            input_index_ = i;
            input_dims_ = engine_->getBindingDimensions(i);
        } else {
            output_index_ = i;
            output_dims_ = engine_->getBindingDimensions(i);
        }
        
        // 计算绑定的内存大小
        auto dims = engine_->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        
        // 假设使用 FP16
        size *= sizeof(__half);
        binding_sizes_[i] = size;
        
        // 分配 GPU 内存
        cudaMalloc(&bindings_[i], size);
    }
    
    if (input_index_ == -1 || output_index_ == -1) {
        std::cerr << "未找到有效的输入或输出绑定" << std::endl;
        return false;
    }
    
    std::cout << "输入维度: ";
    for (int i = 0; i < input_dims_.nbDims; ++i) {
        std::cout << input_dims_.d[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "输出维度: ";
    for (int i = 0; i < output_dims_.nbDims; ++i) {
        std::cout << output_dims_.d[i] << " ";
    }
    std::cout << std::endl;
    
    return true;
}

torch::Tensor TensorRTInference::forward(const torch::Tensor& input) {
    if (!initialized_) {
        throw std::runtime_error("TensorRT 引擎未初始化");
    }
    
    // 确保输入在正确的设备上并且是半精度
    auto input_half = input.to(device_).to(torch::kHalf).contiguous();
    
    // 验证输入形状
    auto input_shape = input_half.sizes();
    if (input_shape.size() != input_dims_.nbDims) {
        throw std::runtime_error("输入维度不匹配");
    }
    for (int i = 0; i < input_dims_.nbDims; ++i) {
        if (input_shape[i] != input_dims_.d[i]) {
            throw std::runtime_error("输入形状不匹配");
        }
    }
    
    // 复制输入数据到 GPU
    cudaMemcpyAsync(bindings_[input_index_], input_half.data_ptr(), 
                    binding_sizes_[input_index_], cudaMemcpyDeviceToDevice, stream_);
    
    // 执行推理
    bool success = context_->enqueueV2(bindings_.data(), stream_, nullptr);
    if (!success) {
        throw std::runtime_error("TensorRT 推理执行失败");
    }
    
    // 创建输出张量
    std::vector<int64_t> output_shape;
    for (int i = 0; i < output_dims_.nbDims; ++i) {
        output_shape.push_back(output_dims_.d[i]);
    }
    
    auto output = torch::empty(output_shape, torch::TensorOptions()
                              .dtype(torch::kHalf)
                              .device(device_));
    
    // 复制输出数据
    cudaMemcpyAsync(output.data_ptr(), bindings_[output_index_], 
                    binding_sizes_[output_index_], cudaMemcpyDeviceToDevice, stream_);
    
    // 同步流
    cudaStreamSynchronize(stream_);
    
    // 转换为 float32 返回
    return output.to(torch::kFloat);
}

void TensorRTInference::cleanUp() {
    // 释放 GPU 内存
    for (void* binding : bindings_) {
        if (binding) {
            cudaFree(binding);
        }
    }
    bindings_.clear();
    binding_sizes_.clear();
    
    // 销毁 CUDA 流
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    // 重置智能指针会自动释放资源
    context_.reset();
    engine_.reset();
    runtime_.reset();
} 