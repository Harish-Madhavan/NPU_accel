#include "include/ops.h"
#include "include/device.h"
#include <iostream>
#include <map>
#include <string>
#include <sstream>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>

// Simple compilation cache: (OpName + InputShapes) -> CompiledModel
static std::map<std::string, ov::CompiledModel>* g_model_cache = nullptr;
static ov::Core* g_core = nullptr;

std::string get_shape_key(const std::string& op_name, const torch::Tensor& a, const torch::Tensor& b) {
    std::stringstream ss;
    ss << op_name << "_";
    for (auto s : a.sizes()) ss << s << "x";
    ss << "_";
    for (auto s : b.sizes()) ss << s << "x";
    return ss.str();
}

ov::CompiledModel get_or_compile_binary_op(const std::string& op_name, const torch::Tensor& a, const torch::Tensor& b) {
    std::string key = get_shape_key(op_name, a, b);
    
    if (g_model_cache == nullptr) {
        g_model_cache = new std::map<std::string, ov::CompiledModel>();
    }

    if (g_model_cache->find(key) != g_model_cache->end()) {
        return (*g_model_cache)[key];
    }

    // Cache Cleanup Strategy: Simple eviction if too big
    if (g_model_cache->size() > 100) {
        g_model_cache->clear();
    }

    initialize_npu();
    
    if (g_core == nullptr) {
        g_core = new ov::Core();
    }

    ov::Shape shape;
    for (auto d : a.sizes()) {
        shape.push_back(d);
    }
    // For matmul, shapes might differ, but for element-wise they are same.
    // Matmul implementation will handle shapes differently below.
    
    std::shared_ptr<ov::Model> model;
    
    if (op_name == "add" || op_name == "sub" || op_name == "mul" || op_name == "div") {
        auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        std::shared_ptr<ov::Node> op_node;
        
        if (op_name == "add") op_node = std::make_shared<ov::opset1::Add>(arg_a, arg_b);
        else if (op_name == "sub") op_node = std::make_shared<ov::opset1::Subtract>(arg_a, arg_b);
        else if (op_name == "mul") op_node = std::make_shared<ov::opset1::Multiply>(arg_a, arg_b);
        else if (op_name == "div") op_node = std::make_shared<ov::opset1::Divide>(arg_a, arg_b);
        
        model = std::make_shared<ov::Model>(ov::NodeVector{op_node}, ov::ParameterVector{arg_a, arg_b});
    } else if (op_name == "matmul") {
        ov::Shape shape_a, shape_b;
        for(auto d : a.sizes()) shape_a.push_back(d);
        for(auto d : b.sizes()) shape_b.push_back(d);
        
        auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_a);
        auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
        auto op_node = std::make_shared<ov::opset1::MatMul>(arg_a, arg_b);
        model = std::make_shared<ov::Model>(ov::NodeVector{op_node}, ov::ParameterVector{arg_a, arg_b});
    }

    std::string device = "NPU";
    // std::cout << "[Intel NPU Lib] Compiling " << op_name << " for " << key << "..." << std::endl;
    auto compiled_model = g_core->compile_model(model, device);
    (*g_model_cache)[key] = compiled_model;
    return compiled_model;
}

torch::Tensor execute_binary_op(const std::string& op_name, torch::Tensor a, torch::Tensor b) {
    auto compiled_model = get_or_compile_binary_op(op_name, a, b);
    auto infer_request = compiled_model.create_infer_request();

    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();
    
    // Determine shapes for OpenVINO tensor wrapping
    ov::Shape shape_a, shape_b;
    for(auto d : a.sizes()) shape_a.push_back(d);
    for(auto d : b.sizes()) shape_b.push_back(d);

    ov::Tensor input_tensor_a(ov::element::f32, shape_a, a_contig.data_ptr<float>());
    ov::Tensor input_tensor_b(ov::element::f32, shape_b, b_contig.data_ptr<float>());

    infer_request.set_input_tensor(0, input_tensor_a);
    infer_request.set_input_tensor(1, input_tensor_b);
    
    infer_request.infer();
    
    auto output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    
    // Convert output shape to torch sizes
    std::vector<int64_t> torch_shape;
    for(auto d : output_shape) torch_shape.push_back(d);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor result = torch::empty(torch_shape, options);
    
    // Copy data back (or wrap if possible, but copy is safer for now with ownership)
    std::memcpy(result.data_ptr<float>(), output_tensor.data<float>(), output_tensor.get_byte_size());

    return result;
}

torch::Tensor npu_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for element-wise add");
    return execute_binary_op("add", a, b);
}

torch::Tensor npu_sub(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for element-wise sub");
    return execute_binary_op("sub", a, b);
}

torch::Tensor npu_mul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for element-wise mul");
    return execute_binary_op("mul", a, b);
}

torch::Tensor npu_div(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for element-wise div");
    return execute_binary_op("div", a, b);
}

torch::Tensor npu_matmul(torch::Tensor a, torch::Tensor b) {
    // Basic check for matrix multiplication: (..., M, K) x (..., K, N)
    TORCH_CHECK(a.dim() >= 2 && b.dim() >= 2, "Tensors must be at least 2D for matmul");
    TORCH_CHECK(a.size(-1) == b.size(-2), "Incompatible dimensions for matmul");
    return execute_binary_op("matmul", a, b);
}