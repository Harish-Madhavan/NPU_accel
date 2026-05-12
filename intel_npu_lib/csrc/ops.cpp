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

// ---------------------------------------------------------------------------
// Dtype helpers
// ---------------------------------------------------------------------------

/// Map a PyTorch scalar type to the corresponding OpenVINO element type.
/// Falls back to f32 for unsupported types with a stderr warning.
static ov::element::Type torch_dtype_to_ov(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case torch::kFloat:   return ov::element::f32;
        case torch::kHalf:    return ov::element::f16;
        case torch::kBFloat16: return ov::element::bf16;
        case torch::kDouble:  return ov::element::f64;
        case torch::kInt:     return ov::element::i32;
        case torch::kLong:    return ov::element::i64;
        case torch::kShort:   return ov::element::i16;
        case torch::kChar:    return ov::element::i8;
        case torch::kByte:    return ov::element::u8;
        case torch::kBool:    return ov::element::boolean;
        default:
            std::cerr << "[Intel NPU] Unsupported dtype: "
                      << t.scalar_type() << ". Falling back to f32." << std::endl;
            return ov::element::f32;
    }
}

/// Map an OV element type back to a torch dtype for the output tensor.
static torch::Dtype ov_dtype_to_torch(ov::element::Type ov_type) {
    if (ov_type == ov::element::f32)      return torch::kFloat;
    if (ov_type == ov::element::f16)      return torch::kHalf;
    if (ov_type == ov::element::bf16)     return torch::kBFloat16;
    if (ov_type == ov::element::f64)      return torch::kDouble;
    if (ov_type == ov::element::i32)      return torch::kInt;
    if (ov_type == ov::element::i64)      return torch::kLong;
    if (ov_type == ov::element::i16)      return torch::kShort;
    if (ov_type == ov::element::i8)       return torch::kChar;
    if (ov_type == ov::element::u8)       return torch::kByte;
    if (ov_type == ov::element::boolean)  return torch::kBool;
    return torch::kFloat; // safe fallback
}

/// Make a contiguous tensor whose memory is safe to pass to OpenVINO.
/// OpenVINO shared_memory requires contiguous layout.
static torch::Tensor ensure_contiguous(const torch::Tensor& t) {
    return t.is_contiguous() ? t : t.contiguous();
}

// --- Helper Functions for Key Generation ---

static ov::Shape get_ov_shape(const torch::Tensor& t) {
    ov::Shape shape;
    for(auto d : t.sizes()) shape.push_back(d);
    return shape;
}

std::string get_shape_str(const torch::Tensor& t) {
    std::stringstream ss;
    for (auto s : t.sizes()) ss << s << "x";
    return ss.str();
}

std::string get_key(const std::string& op_name, const std::vector<torch::Tensor>& inputs, const std::string& extra_args = "") {
    std::stringstream ss;
    ss << op_name << "_";
    for (const auto& t : inputs) {
        ss << get_shape_str(t) << t.scalar_type() << "_";
    }
    ss << extra_args;
    return ss.str();
}

// --- Core Compilation & Execution Logic ---

torch::Tensor execute_op(const std::string& key, std::shared_ptr<ov::Model> model, const std::vector<torch::Tensor>& inputs) {
    try {
        auto& backend = NPUBackend::getInstance();
        auto compiled_model = backend.getOrCompileModel(key, model);
        auto infer_request = compiled_model.create_infer_request();

        // Map inputs — use the tensor's actual dtype so OV sees the right element type.
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto t = ensure_contiguous(inputs[i]);
            ov::element::Type ov_type = torch_dtype_to_ov(t);
            ov::Shape shape(t.sizes().begin(), t.sizes().end());

            // shared_memory=true: OV reads directly from the torch storage — no copy.
            ov::Tensor input_tensor(ov_type, shape, t.data_ptr());
            infer_request.set_input_tensor(i, input_tensor);
        }
        
        infer_request.infer();
        
        auto output_tensor = infer_request.get_output_tensor();
        auto output_shape = output_tensor.get_shape();
        auto output_ov_type = output_tensor.get_element_type();

        std::vector<int64_t> torch_shape(output_shape.begin(), output_shape.end());
        torch::Dtype torch_out_dtype = ov_dtype_to_torch(output_ov_type);

        auto result = torch::empty(torch_shape, torch::TensorOptions().dtype(torch_out_dtype));

        // Copy back from OV output tensor to the torch result tensor.
        std::memcpy(result.data_ptr(), output_tensor.data(), output_tensor.get_byte_size());

        return result;

    } catch (const std::exception& e) {
        TORCH_CHECK(false, "[Intel NPU] Execution Failed for key '" + key + "': " + std::string(e.what()));
    }
    // Unreachable — silences compiler warning about missing return.
    return torch::Tensor();
}

template <typename OpT>
torch::Tensor execute_unary_op_helper(const std::string& name, torch::Tensor a) {
    std::string key = get_key(name, {a});
    auto arg_a = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(a), get_ov_shape(a));
    auto op = std::make_shared<OpT>(arg_a);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_a});
    return execute_op(key, model, {a});
}

template <typename OpT>
torch::Tensor execute_binary_op_helper(const std::string& name, torch::Tensor a, torch::Tensor b) {
    std::string key = get_key(name, {a, b});
    auto arg_a = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(a), get_ov_shape(a));
    auto arg_b = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(b), get_ov_shape(b));
    auto op = std::make_shared<OpT>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_a, arg_b});
    return execute_op(key, model, {a, b});
}

// --- Op Implementations ---

torch::Tensor npu_add(torch::Tensor a, torch::Tensor b) {
    return execute_binary_op_helper<ov::opset1::Add>("add", a, b);
}

torch::Tensor npu_sub(torch::Tensor a, torch::Tensor b) {
    return execute_binary_op_helper<ov::opset1::Subtract>("sub", a, b);
}

torch::Tensor npu_neg(torch::Tensor a) {
    return execute_unary_op_helper<ov::opset1::Negative>("neg", a);
}

torch::Tensor npu_mul(torch::Tensor a, torch::Tensor b) {
    return execute_binary_op_helper<ov::opset1::Multiply>("mul", a, b);
}

torch::Tensor npu_div(torch::Tensor a, torch::Tensor b) {
    std::string key = get_key("div", {a, b});
    ov::Shape shape_a = get_ov_shape(a);
    ov::Shape shape_b = get_ov_shape(b);
    // Division result is always floating-point.
    ov::element::Type fp_type = (torch_dtype_to_ov(a) == ov::element::f16 &&
                                  torch_dtype_to_ov(b) == ov::element::f16)
                                 ? ov::element::f16 : ov::element::f32;
    auto arg_a = std::make_shared<ov::opset1::Parameter>(fp_type, shape_a);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(fp_type, shape_b);
    auto op = std::make_shared<ov::opset1::Divide>(arg_a, arg_b);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_a, arg_b});
    return execute_op(key, model, {a, b});
}

torch::Tensor npu_matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() >= 2 && b.dim() >= 2, "Tensors must be at least 2D for matmul");
    return execute_binary_op_helper<ov::opset1::MatMul>("matmul", a, b);
}

torch::Tensor npu_relu(torch::Tensor a) {
    return execute_unary_op_helper<ov::opset1::Relu>("relu", a);
}

torch::Tensor npu_gelu(torch::Tensor a) {
    return execute_unary_op_helper<ov::opset8::Gelu>("gelu", a);
}

torch::Tensor npu_silu(torch::Tensor a) {
    return execute_unary_op_helper<ov::opset4::Swish>("silu", a);
}

torch::Tensor npu_rmsnorm(torch::Tensor input, torch::Tensor weight, float epsilon) {
    std::string key = get_key("rmsnorm", {input, weight}, std::to_string(epsilon));

    ov::element::Type ov_type = torch_dtype_to_ov(input);
    ov::Shape input_shape = get_ov_shape(input);
    ov::Shape weight_shape = get_ov_shape(weight);

    auto arg_input  = std::make_shared<ov::opset1::Parameter>(ov_type, input_shape);
    auto arg_weight = std::make_shared<ov::opset1::Parameter>(ov_type, weight_shape);

    // 1. Square the input
    auto input_squared = std::make_shared<ov::opset1::Multiply>(arg_input, arg_input);

    // 2. Calculate Mean Square along the last dimension
    auto last_dim_idx = static_cast<int64_t>(input_shape.size() - 1);
    auto axes = ov::opset1::Constant::create(ov::element::i64, {1}, {last_dim_idx});
    auto mean_square = std::make_shared<ov::opset1::ReduceMean>(input_squared, axes, true);

    // 3. Add Epsilon — cast to the working dtype
    auto epsilon_const = ov::opset1::Constant::create(ov_type, {}, {epsilon});
    auto variance = std::make_shared<ov::opset1::Add>(mean_square, epsilon_const);

    // 4. Square Root
    auto std_dev = std::make_shared<ov::opset1::Sqrt>(variance);

    // 5. Normalize
    auto x_normalized = std::make_shared<ov::opset1::Divide>(arg_input, std_dev);

    // 6. Apply Weight (Gain)
    auto output = std::make_shared<ov::opset1::Multiply>(x_normalized, arg_weight);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{arg_input, arg_weight});
    return execute_op(key, model, {input, weight});
}

torch::Tensor npu_softmax(torch::Tensor a, int64_t dim) {
    if (dim < 0) dim += a.dim();
    std::string key = get_key("softmax", {a}, std::to_string(dim));
    auto arg_a = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(a), get_ov_shape(a));
    auto op = std::make_shared<ov::opset1::Softmax>(arg_a, dim);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_a});
    return execute_op(key, model, {a});
}

torch::Tensor npu_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Linear: y = x * W^T + b
    bool has_bias = bias.defined() && bias.numel() > 0;
    std::string key = get_key("linear", {input, weight}, has_bias ? "_bias" : "");
    
    ov::element::Type ov_type = torch_dtype_to_ov(input);
    ov::Shape shape_in = get_ov_shape(input);
    ov::Shape shape_w  = get_ov_shape(weight);
    
    auto arg_in = std::make_shared<ov::opset1::Parameter>(ov_type, shape_in);
    auto arg_w  = std::make_shared<ov::opset1::Parameter>(ov_type, shape_w);
    auto matmul = std::make_shared<ov::opset1::MatMul>(arg_in, arg_w, false, true);
    
    std::shared_ptr<ov::Node> result = matmul;
    ov::ParameterVector params = {arg_in, arg_w};
    std::vector<torch::Tensor> inputs = {input, weight};

    if (has_bias) {
        auto arg_b = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(bias));
        result = std::make_shared<ov::opset1::Add>(result, arg_b);
        params.push_back(arg_b);
        inputs.push_back(bias);
    }
    
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, params);
    return execute_op(key, model, inputs);
}

torch::Tensor npu_transpose(torch::Tensor input, std::vector<int64_t> permutation) {
    std::stringstream ss;
    ss << "transpose_";
    for (auto p : permutation) ss << p << ",";
    std::string key = get_key("transpose", {input}, ss.str());
    auto arg_input  = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    auto perm_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{permutation.size()}, permutation);
    auto op    = std::make_shared<ov::opset1::Transpose>(arg_input, perm_const);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}

torch::Tensor npu_reshape(torch::Tensor input, std::vector<int64_t> shape) {
    std::stringstream ss;
    ss << "reshape_";
    for (auto s : shape) ss << s << ",";
    std::string key = get_key("reshape", {input}, ss.str());
    auto arg_input   = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    auto shape_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape);
    auto op    = std::make_shared<ov::opset1::Reshape>(arg_input, shape_const, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
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
    if (attn_mask.defined() && attn_mask.numel() > 0) ss << "mask_";
    
    std::string extra = ss.str();
    std::vector<torch::Tensor> key_inputs = {query, key, value};
    if (attn_mask.defined() && attn_mask.numel() > 0) key_inputs.push_back(attn_mask);
    
    std::string op_key = get_key("sdpa", key_inputs, extra);

    ov::element::Type ov_type = torch_dtype_to_ov(query);
    auto arg_q = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(query));
    auto arg_k = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(key));
    auto arg_v = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(value));
    
    ov::ParameterVector params = {arg_q, arg_k, arg_v};
    std::vector<torch::Tensor> inputs = {query, key, value};
    
    std::shared_ptr<ov::Node> arg_mask = nullptr;
    if (attn_mask.defined() && attn_mask.numel() > 0) {
        auto p_mask = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(attn_mask));
        arg_mask = p_mask;
        params.push_back(p_mask);
        inputs.push_back(attn_mask);
    }
    
    std::shared_ptr<ov::Node> arg_scale = nullptr;
    if (scale > 0) {
         arg_scale = ov::opset1::Constant::create(ov::element::f32, {}, {scale});
    }

    std::shared_ptr<ov::Node> op;
    if (arg_mask && arg_scale) {
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask, arg_scale, is_causal);
    } else if (arg_mask && !arg_scale) {
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask, is_causal);
    } else if (!arg_mask && arg_scale) {
         arg_mask = ov::opset1::Constant::create(ov::element::f32, {}, {0.0f});
         op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask, arg_scale, is_causal);
    } else {
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, is_causal);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, params);
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
    if (bias.defined() && bias.numel() > 0) key += "_bias";

    ov::element::Type ov_type = torch_dtype_to_ov(input);
    auto arg_input  = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(input));
    auto arg_weight = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(weight));

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
    ov::ParameterVector params = {arg_input, arg_weight};

    if (bias.defined() && bias.numel() > 0) {
        inputs.push_back(bias);
        auto arg_bias = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(bias));
        params.push_back(arg_bias);
        
        auto axes_const = ov::opset1::Constant::create(ov::element::i64, {3}, {0, 2, 3});
        auto bias_4d = std::make_shared<ov::opset1::Unsqueeze>(arg_bias, axes_const);
        result = std::make_shared<ov::opset1::Add>(result, bias_4d);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, params);
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
    auto arg_input = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));

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

    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}
