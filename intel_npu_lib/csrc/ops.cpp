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
#include <openvino/opsets/opset13.hpp> // For ScaledDotProductAttention

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

torch::Tensor execute_op(const std::string& key, std::shared_ptr<ov::Model> model, const std::vector<torch::Tensor>& inputs) {
    try {
        auto compiled_model = NPUBackend::getInstance().getOrCompileModel(key, model);
        auto infer_request = compiled_model.create_infer_request();

        // Keep tensors alive during inference
        std::vector<torch::Tensor> held_tensors; 

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto t = inputs[i].contiguous();
            held_tensors.push_back(t); // extend lifetime
            
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

    } catch (const std::exception& e) {
        // Fallback or rethrow
        // For now, rethrow as a torch error
        TORCH_CHECK(false, "NPU Execution Failed: " + std::string(e.what()));
    }
}

// --- Op Implementations ---

torch::Tensor npu_add(torch::Tensor a, torch::Tensor b) {
    std::string key = get_key("add", {a, b});
    
    ov::Shape shape_a, shape_b;
    for(auto d : a.sizes()) shape_a.push_back(d);
    for(auto d : b.sizes()) shape_b.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_a);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
    auto op = std::make_shared<ov::opset1::Add>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_sub(torch::Tensor a, torch::Tensor b) {
    std::string key = get_key("sub", {a, b});

    ov::Shape shape_a, shape_b;
    for(auto d : a.sizes()) shape_a.push_back(d);
    for(auto d : b.sizes()) shape_b.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_a);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
    auto op = std::make_shared<ov::opset1::Subtract>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_neg(torch::Tensor a) {
    std::string key = get_key("neg", {a});

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Negative>(arg_a);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_mul(torch::Tensor a, torch::Tensor b) {
    std::string key = get_key("mul", {a, b});

    ov::Shape shape_a, shape_b;
    for(auto d : a.sizes()) shape_a.push_back(d);
    for(auto d : b.sizes()) shape_b.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_a);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
    auto op = std::make_shared<ov::opset1::Multiply>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_div(torch::Tensor a, torch::Tensor b) {
    std::string key = get_key("div", {a, b});

    ov::Shape shape_a, shape_b;
    for(auto d : a.sizes()) shape_a.push_back(d);
    for(auto d : b.sizes()) shape_b.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_a);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
    auto op = std::make_shared<ov::opset1::Divide>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a, arg_b});
    
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() >= 2 && b.dim() >= 2, "Tensors must be at least 2D for matmul");
    std::string key = get_key("matmul", {a, b});

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

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Relu>(arg_a);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_gelu(torch::Tensor a) {
    std::string key = get_key("gelu", {a});

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

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset4::Swish>(arg_a); // Swish is SiLU in OpenVINO opset4
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_rmsnorm(torch::Tensor input, torch::Tensor weight, float epsilon) {
    std::string key = get_key("rmsnorm", {input, weight}, std::to_string(epsilon));

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

    ov::Shape shape;
    for(auto d : a.sizes()) shape.push_back(d);
    
    auto arg_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto op = std::make_shared<ov::opset1::Softmax>(arg_a, dim);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_a});
    
    return execute_op(key, model, {a});
}

torch::Tensor npu_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Linear: y = x * A^T + b
    std::string key = get_key("linear", {input, weight, bias});

    ov::Shape shape_in, shape_w, shape_b;
    for(auto d : input.sizes()) shape_in.push_back(d);
    for(auto d : weight.sizes()) shape_w.push_back(d);
    for(auto d : bias.sizes()) shape_b.push_back(d);
    
    auto arg_in = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_in);
    auto arg_w = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_w);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape_b);
    
    auto matmul = std::make_shared<ov::opset1::MatMul>(arg_in, arg_w, false, true);
    auto add = std::make_shared<ov::opset1::Add>(matmul, arg_b);
    
    auto model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{arg_in, arg_w, arg_b});
    
    return execute_op(key, model, {input, weight, bias});
}

torch::Tensor npu_transpose(torch::Tensor input, std::vector<int64_t> permutation) {
    std::stringstream ss;
    ss << "transpose_";
    for (auto p : permutation) ss << p << ",";
    std::string extra = ss.str();
    std::string key = get_key("transpose", {input}, extra);

    ov::Shape input_shape;
    for(auto d : input.sizes()) input_shape.push_back(d);

    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    
    auto perm_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{permutation.size()}, permutation);
    
    auto op = std::make_shared<ov::opset1::Transpose>(arg_input, perm_const);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_input});

    return execute_op(key, model, {input});
}

torch::Tensor npu_reshape(torch::Tensor input, std::vector<int64_t> shape) {
    std::stringstream ss;
    ss << "reshape_";
    for (auto s : shape) ss << s << ",";
    std::string extra = ss.str();
    std::string key = get_key("reshape", {input}, extra);

    ov::Shape input_shape;
    for(auto d : input.sizes()) input_shape.push_back(d);

    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);

    auto shape_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape);

    auto op = std::make_shared<ov::opset1::Reshape>(arg_input, shape_const, false); 
    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_input});

    return execute_op(key, model, {input});
}

torch::Tensor npu_scaled_dot_product_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor attn_mask,
    double dropout_p,
    bool is_causal,
    double scale
) {
    std::stringstream ss;
    ss << "sdpa_" << is_causal << "_" << scale << "_";
    if (attn_mask.defined()) ss << "mask_";
    
    std::string extra = ss.str();
    std::vector<torch::Tensor> key_inputs = {query, key, value};
    if (attn_mask.defined()) key_inputs.push_back(attn_mask);
    
    std::string op_key = get_key("sdpa", key_inputs, extra);

    ov::Shape q_shape, k_shape, v_shape;
    for(auto d : query.sizes()) q_shape.push_back(d);
    for(auto d : key.sizes()) k_shape.push_back(d);
    for(auto d : value.sizes()) v_shape.push_back(d);
    
    auto arg_q = std::make_shared<ov::opset1::Parameter>(ov::element::f32, q_shape);
    auto arg_k = std::make_shared<ov::opset1::Parameter>(ov::element::f32, k_shape);
    auto arg_v = std::make_shared<ov::opset1::Parameter>(ov::element::f32, v_shape);
    
    ov::ParameterVector params = {arg_q, arg_k, arg_v};
    std::vector<torch::Tensor> inputs = {query, key, value};
    
    std::shared_ptr<ov::Node> arg_mask = nullptr;
    if (attn_mask.defined()) {
        ov::Shape m_shape;
        for(auto d : attn_mask.sizes()) m_shape.push_back(d);
        auto p_mask = std::make_shared<ov::opset1::Parameter>(ov::element::f32, m_shape);
        arg_mask = p_mask;
        params.push_back(p_mask);
        inputs.push_back(attn_mask);
    }
    
    std::shared_ptr<ov::Node> arg_scale = nullptr;
    // Torch SDPA usually defaults scale to 1/sqrt(head_dim) if not provided?
    // Here we assume the caller passes the correct scale if they want to override, 
    // or we might pass 0/negative to indicate "default"?
    // But OpenVINO SDPA doesn't compute default scale internally if input is missing? 
    // Wait, documentation says "If scale is not provided, it defaults to 1 / sqrt(query.shape[-1])".
    // So if scale <= 0, we skip providing it?
    // Let's assume if scale > 0, we provide it.
    
    if (scale > 0) {
         // Scale is scalar? Or tensor?
         // SDPA expects scale as input 4 (after mask).
         arg_scale = ov::opset1::Constant::create(ov::element::f32, {}, {scale});
    }

    // Constructor: ScaledDotProductAttention (const Output< Node > &query, const Output< Node > &key, const Output< Node > &value, const Output< Node > &attention_mask, const Output< Node > &scale, bool causal=false)
    // Note: If args are null/missing, there are different constructors or we pass empty?
    // Actually, C++ API usually has overloads.
    // If we want to skip mask but provide scale: we pass null/dummy?
    // opset13::SDPA constructor takes optional args?
    // Let's check constructor signatures.
    // Explicit constructor: query, key, value, attn_mask, scale, causal.
    
    std::shared_ptr<ov::Node> op;
    if (arg_mask && arg_scale) {
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask, arg_scale, is_causal);
    } else if (arg_mask && !arg_scale) {
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask, is_causal);
    } else if (!arg_mask && arg_scale) {
         // How to pass scale without mask? Pass empty/null for mask?
         // Usually undefined/empty constant?
         // Or overload without mask?
         // Documentation says: "attention_mask (optional)"
         // "If not provided, ..."
         // The C++ API might require passing something.
         // Let's assume we can pass a dummy empty node or there is an overload.
         // Actually, if we look at python: query, key, value, attention_mask=None, scale=None...
         // In C++, often `ov::Output<Node>()` or similar.
         // But `make_shared` might not like that.
         // Safe bet: if mask is missing but scale is present, pass an empty constant for mask? or nullptr?
         // OpenVINO C++ nodes usually don't take nullptr for Input.
         // They take Output<Node>.
         // Let's try passing a dummy (empty shape?) or look for overload.
         // If no overload, usually we construct the node and set inputs.
         
         // Let's assume there is a constructor `ScaledDotProductAttention(q, k, v, causal)` 
         // and we can set inputs later?
         // Or `ScaledDotProductAttention(q, k, v, mask, scale, causal)`
         
         // If I don't know the exact API, this is risky.
         // However, I can create the node with basic args and then `.set_argument(3, scale)`?
         
         // Let's fallback to: if scale is provided, we MUST provide mask? 
         // Or provide a "dummy" mask (all zeros? or booleans?).
         // If mask is missing in SDPA, it means "attend to everything".
         // Passing a tensor of zeros (additive mask) does the same.
         // So if scale is needed but mask is missing, create a zero mask broadcastable?
         
         if (!arg_mask) {
             // Create zero mask (1,1,1,1) ?
             arg_mask = ov::opset1::Constant::create(ov::element::f32, {}, {0.0f});
         }
         op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask, arg_scale, is_causal);
    } else {
        // No mask, no scale
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, is_causal);
    }

    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, params);
    
    return execute_op(op_key, model, inputs);
}

torch::Tensor npu_conv2d(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    std::vector<int64_t> stride, 
    std::vector<int64_t> padding, 
    std::vector<int64_t> dilation, 
    int64_t groups
) {
    std::stringstream ss;
    ss << "conv2d_";
    for(auto s : stride) ss << s << ",";
    for(auto p : padding) ss << p << ",";
    for(auto d : dilation) ss << d << ",";
    ss << groups;
    
    std::string key = get_key("conv2d", {input, weight}, ss.str());
    if (bias.defined()) key += "_bias";

    ov::Shape input_shape, weight_shape;
    for(auto d : input.sizes()) input_shape.push_back(d);
    for(auto d : weight.sizes()) weight_shape.push_back(d);

    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    auto arg_weight = std::make_shared<ov::opset1::Parameter>(ov::element::f32, weight_shape);

    ov::Strides ov_strides(stride.begin(), stride.end());
    ov::CoordinateDiff ov_pads_begin(padding.begin(), padding.end());
    ov::CoordinateDiff ov_pads_end(padding.begin(), padding.end());
    ov::Strides ov_dilations(dilation.begin(), dilation.end());

    std::shared_ptr<ov::Node> conv_op;
    
    if (groups == 1) {
        conv_op = std::make_shared<ov::opset1::Convolution>(
            arg_input,
            arg_weight,
            ov_strides,
            ov_pads_begin,
            ov_pads_end,
            ov_dilations
        );
    } else {
        int64_t out_channels = weight.size(0);
        int64_t in_channels_per_group = weight.size(1);
        int64_t kH = weight.size(2);
        int64_t kW = weight.size(3);
        
        auto shape_const = ov::opset1::Constant::create(ov::element::i64, {5}, {groups, out_channels/groups, in_channels_per_group, kH, kW});
        auto reshaped_w = std::make_shared<ov::opset1::Reshape>(arg_weight, shape_const, false);
        
        conv_op = std::make_shared<ov::opset1::GroupConvolution>(
            arg_input,
            reshaped_w,
            ov_strides,
            ov_pads_begin,
            ov_pads_end,
            ov_dilations
        );
    }

    std::shared_ptr<ov::Node> result = conv_op;

    std::vector<torch::Tensor> inputs = {input, weight};
    std::shared_ptr<ov::opset1::Parameter> arg_bias = nullptr;

    if (bias.defined()) {
        inputs.push_back(bias);
        ov::Shape bias_shape;
        for(auto d : bias.sizes()) bias_shape.push_back(d);
        arg_bias = std::make_shared<ov::opset1::Parameter>(ov::element::f32, bias_shape);
        
        auto axes_const = ov::opset1::Constant::create(ov::element::i64, {3}, {0, 2, 3});
        auto bias_4d = std::make_shared<ov::opset1::Unsqueeze>(arg_bias, axes_const);
        
        result = std::make_shared<ov::opset1::Add>(result, bias_4d);
    }

    ov::ParameterVector params = {arg_input, arg_weight};
    if (arg_bias) params.push_back(arg_bias);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{result}, params);
    
    return execute_op(key, model, inputs);
}

torch::Tensor npu_max_pool2d(
    torch::Tensor input, 
    std::vector<int64_t> kernel_size, 
    std::vector<int64_t> stride, 
    std::vector<int64_t> padding, 
    std::vector<int64_t> dilation,
    bool ceil_mode
) {
    std::stringstream ss;
    ss << "maxpool2d_";
    for(auto k : kernel_size) ss << k << ",";
    for(auto s : stride) ss << s << ",";
    for(auto p : padding) ss << p << ",";
    for(auto d : dilation) ss << d << ",";
    ss << ceil_mode;

    std::string key = get_key("maxpool2d", {input}, ss.str());

    ov::Shape input_shape;
    for(auto d : input.sizes()) input_shape.push_back(d);
    
    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);

    ov::Strides ov_strides(stride.begin(), stride.end());
    ov::Shape ov_kernel(kernel_size.begin(), kernel_size.end());
    ov::Shape ov_pads_begin(padding.begin(), padding.end());
    ov::Shape ov_pads_end(padding.begin(), padding.end());
    ov::Strides ov_dilations(dilation.begin(), dilation.end());
    
    auto rounding_type = ceil_mode ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;

    auto op = std::make_shared<ov::opset1::MaxPool>(
        arg_input,
        ov_strides,
        ov_pads_begin,
        ov_pads_end,
        ov_kernel,
        rounding_type,
        ov::op::PadType::EXPLICIT
    );

    auto model = std::make_shared<ov::Model>(ov::NodeVector{op}, ov::ParameterVector{arg_input});
    
    return execute_op(key, model, {input});
}
