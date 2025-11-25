#include "include/ops.h"
#include "include/device.h"
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset4.hpp> // For Swish (SiLU)
#include <openvino/opsets/opset8.hpp> // For GELU

// Global state
static std::map<std::string, ov::CompiledModel>* g_model_cache = nullptr;
static ov::Core* g_core = nullptr;

// --- Helper Functions for Key Generation ---

std::string get_shape_str(const torch::Tensor& t) {
    std::stringstream ss;
    for (auto s : t.sizes()) ss << s << "x";
    return ss.str();
}

std::string get_key(const std::string& op_name, const std::vector<torch::Tensor>& inputs, const std::string& extra_args = "") {
    std::stringstream ss;
    ss << op_name << "_";
    for (const auto& t : inputs) {
        ss << get_shape_str(t) << "_";
    }
    ss << extra_args;
    return ss.str();
}

// --- Core Compilation & Execution Logic ---

ov::CompiledModel get_or_compile_model(const std::string& key, std::shared_ptr<ov::Model> model) {
    if (g_model_cache == nullptr) {
        g_model_cache = new std::map<std::string, ov::CompiledModel>();
    }

    if (g_model_cache->find(key) != g_model_cache->end()) {
        return (*g_model_cache)[key];
    }

    // Cache Cleanup
    if (g_model_cache->size() > 100) {
        g_model_cache->clear();
    }

    initialize_npu();
    
    if (g_core == nullptr) {
        g_core = new ov::Core();
    }

    std::string device = "NPU";
    // std::cout << "[Intel NPU Lib] Compiling " << key << "..." << std::endl;
    auto compiled_model = g_core->compile_model(model, device);
    (*g_model_cache)[key] = compiled_model;
    return compiled_model;
}

torch::Tensor execute_op(const std::string& key, std::shared_ptr<ov::Model> model, const std::vector<torch::Tensor>& inputs) {
    auto compiled_model = get_or_compile_model(key, model);
    auto infer_request = compiled_model.create_infer_request();

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto t = inputs[i].contiguous();
        ov::Shape shape;
        for(auto d : t.sizes()) shape.push_back(d);
        
        ov::Tensor input_tensor(ov::element::f32, shape, t.data_ptr<float>());
        infer_request.set_input_tensor(i, input_tensor);
    }
    
    infer_request.infer();
    
    auto output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    
    std::vector<int64_t> torch_shape;
    for(auto d : output_shape) torch_shape.push_back(d);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor result = torch::empty(torch_shape, options);
    
    std::memcpy(result.data_ptr<float>(), output_tensor.data<float>(), output_tensor.get_byte_size());

    return result;
}

// --- Op Implementations ---

torch::Tensor npu_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for add");
    std::string key = get_key("add", {a, b});
    
    // Define Graph (Lazy construction, strictly only if not cached? 
    // Actually, we construct it to pass to compiler, but compiler function checks cache first. 
    // Optimally we check cache before constructing model, but for simplicity we construct model here or check cache here.
    // To avoid constructing model every time, we should check cache first.)
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a, b}); // Model not needed if cached
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Add>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_sub(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for sub");
    std::string key = get_key("sub", {a, b});

    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a, b});
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Subtract>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_mul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for mul");
    std::string key = get_key("mul", {a, b});

    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a, b});
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Multiply>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_div(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for div");
    std::string key = get_key("div", {a, b});

    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a, b});
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Divide>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() >= 2 && b.dim() >= 2, "Tensors must be at least 2D for matmul");
    std::string key = get_key("matmul", {a, b});
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a, b});
    }

    ov::Shape shape_a, shape_b;
    for(auto d : a.sizes()) shape_a.push_back(d);
    for(auto d : b.sizes()) shape_b.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_a);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
    auto op = std::make_shared<ov::opset1::MatMul>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_relu(torch::Tensor a) {
    std::string key = get_key("relu", {a});
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a});
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Relu>(arg_a);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_gelu(torch::Tensor a) {
    std::string key = get_key("gelu", {a});
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a});
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    // Gelu in opset7, default 'erf' approximation mode usually
    auto op = std::make_shared<ov::opset8::Gelu>(arg_a);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_silu(torch::Tensor a) {
    std::string key = get_key("silu", {a});
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a});
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset4::Swish>(arg_a); // Swish is SiLU in OpenVINO opset4
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_rmsnorm(torch::Tensor input, torch::Tensor weight, float epsilon) {
    std::string key = get_key("rmsnorm", {input, weight}, std::to_string(epsilon));
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {input, weight});
    }

    ov::Shape input_shape, weight_shape;
    for(auto d : input.sizes()) input_shape.push_back(d);
    for(auto d : weight.sizes()) weight_shape.push_back(d);
    
    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    auto arg_weight = std::make_shared<ov::opset1::Parameter>(ov::element::f32, weight_shape);

    // 1. Square the input
    auto input_squared = std::make_shared<ov::opset1::Multiply>(arg_input, arg_input);

    // 2. Calculate Mean Square
    auto last_dim_idx = input_shape.size() - 1;
    auto axes = ov::opset1::Constant::create(ov::element::i64, {1}, {last_dim_idx});
    auto mean_square = std::make_shared<ov::opset1::ReduceMean>(input_squared, axes, true); // Keep dims

    // 3. Add Epsilon
    auto epsilon_const = ov::opset1::Constant::create(ov::element::f32, {}, {epsilon});
    auto variance = std::make_shared<ov::opset1::Add>(mean_square, epsilon_const);

    // 4. Square Root
    auto std_dev = std::make_shared<ov::opset1::Sqrt>(variance);

    // 5. Normalize
    auto x_normalized = std::make_shared<ov::opset1::Divide>(arg_input, std_dev);

    // 6. Apply Weight (Gain)
    auto output = std::make_shared<ov::opset1::Multiply>(x_normalized, arg_weight);
    
    auto model = std::make_shared<ov::Model>(ov::NodeVector{output}, ov::ParameterVector{arg_input, arg_weight});
    
    return execute_op(key, model, {input, weight});
}

torch::Tensor npu_softmax(torch::Tensor a, int64_t dim) {
    // Handle negative dim
    if (dim < 0) dim += a.dim();
    
    std::string key = get_key("softmax", {a}, std::to_string(dim));
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {a});
    }

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Softmax>(arg_a, dim);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Linear: y = x * A^T + b
    // Input: (..., in_features)
    // Weight: (out_features, in_features)
    // Bias: (out_features)
    
    std::string key = get_key("linear", {input, weight, bias});
    
    if (g_model_cache && g_model_cache->find(key) != g_model_cache->end()) {
        return execute_op(key, nullptr, {input, weight, bias});
    }

    ov::Shape shape_in, shape_w, shape_b;
    for(auto d : input.sizes()) shape_in.push_back(d);
    for(auto d : weight.sizes()) shape_w.push_back(d);
    for(auto d : bias.sizes()) shape_b.push_back(d);
    
    auto arg_in = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_in);
    auto arg_w = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_w);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
    
    // MatMul: Weight needs to be transposed usually for Linear? 
    // PyTorch Linear: input x weight.T
    // OpenVINO MatMul can transpose_b=true
    auto matmul = std::make_shared<ov::opset1::MatMul>(arg_in, arg_w, false, true);
    auto add = std::make_shared<ov::opset1::Add>(matmul, arg_b);
    
    auto model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{arg_in, arg_w, arg_b});
    
    return execute_op(key, model, {input, weight, bias});
}
