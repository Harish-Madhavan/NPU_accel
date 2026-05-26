#include "include/ops.h"

#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset13.hpp>  // For ScaledDotProductAttention
#include <openvino/opsets/opset3.hpp>   // For ScatterUpdate
#include <openvino/opsets/opset4.hpp>   // For Swish (SiLU)
#include <openvino/opsets/opset8.hpp>   // For GELU
#include <sstream>
#include <string>
#include <vector>

#include "include/device.h"

// ---------------------------------------------------------------------------
// Dtype helpers
// ---------------------------------------------------------------------------

/// Map a PyTorch scalar type to the corresponding OpenVINO element type.
/// Falls back to f32 for unsupported types with a stderr warning.
static ov::element::Type torch_dtype_to_ov(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case torch::kFloat:
            return ov::element::f32;
        case torch::kHalf:
            return ov::element::f16;
        case torch::kBFloat16:
            return ov::element::bf16;
        case torch::kDouble:
            return ov::element::f64;
        case torch::kInt:
            return ov::element::i32;
        case torch::kLong:
            return ov::element::i64;
        case torch::kShort:
            return ov::element::i16;
        case torch::kChar:
            return ov::element::i8;
        case torch::kByte:
            return ov::element::u8;
        case torch::kBool:
            return ov::element::boolean;
        default:
            std::cerr << "[Intel NPU] Unsupported dtype: " << t.scalar_type()
                      << ". Falling back to f32." << std::endl;
            return ov::element::f32;
    }
}

/// Map an OV element type back to a torch dtype for the output tensor.
static torch::Dtype ov_dtype_to_torch(ov::element::Type ov_type) {
    if (ov_type == ov::element::f32) return torch::kFloat;
    if (ov_type == ov::element::f16) return torch::kHalf;
    if (ov_type == ov::element::bf16) return torch::kBFloat16;
    if (ov_type == ov::element::f64) return torch::kDouble;
    if (ov_type == ov::element::i32) return torch::kInt;
    if (ov_type == ov::element::i64) return torch::kLong;
    if (ov_type == ov::element::i16) return torch::kShort;
    if (ov_type == ov::element::i8) return torch::kChar;
    if (ov_type == ov::element::u8) return torch::kByte;
    if (ov_type == ov::element::boolean) return torch::kBool;
    return torch::kFloat;  // safe fallback
}

/// Make a contiguous tensor whose memory is safe to pass to OpenVINO.
/// OpenVINO shared_memory requires contiguous layout.
static torch::Tensor ensure_contiguous(const torch::Tensor& t) {
    return t.is_contiguous() ? t : t.contiguous();
}

// --- Helper Functions for Key Generation ---

static ov::Shape get_ov_shape(const torch::Tensor& t) {
    ov::Shape shape;
    for (auto d : t.sizes()) shape.push_back(d);
    return shape;
}

std::string get_shape_str(const torch::Tensor& t) {
    std::stringstream ss;
    for (auto s : t.sizes()) ss << s << "x";
    return ss.str();
}

std::string get_key(const std::string& op_name, const std::vector<torch::Tensor>& inputs,
                    const std::string& extra_args = "") {
    std::stringstream ss;
    ss << op_name << "_";
    for (const auto& t : inputs) {
        ss << get_shape_str(t) << t.scalar_type() << "_";
    }
    ss << extra_args;
    return ss.str();
}

// --- Core Compilation & Execution Logic ---

torch::Tensor execute_op(const std::string& key, std::shared_ptr<ov::Model> model,
                         const std::vector<torch::Tensor>& inputs) {
    try {
        auto& backend = NPUBackend::getInstance();
        backend.getOrCompileModel(key, model);
        auto infer_request = backend.getOrCachedInferRequest(key);

        // Ensure inputs are contiguous and kept alive for the duration of inference
        std::vector<torch::Tensor> contiguous_inputs;
        contiguous_inputs.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            contiguous_inputs.push_back(ensure_contiguous(inputs[i]));
        }

        // Map inputs
        for (size_t i = 0; i < contiguous_inputs.size(); ++i) {
            const auto& t = contiguous_inputs[i];
            ov::element::Type ov_type = torch_dtype_to_ov(t);
            ov::Shape shape(t.sizes().begin(), t.sizes().end());

            ov::Tensor input_tensor(ov_type, shape, t.data_ptr());
            infer_request.set_input_tensor(i, input_tensor);
        }

        // Pre-allocate output tensor to enable zero-copy if shape is static
        auto out_port = infer_request.get_compiled_model().output(0);
        auto partial_shape = out_port.get_partial_shape();

        torch::Tensor result;
        bool preallocated = false;
        if (partial_shape.is_static()) {
            auto output_shape = partial_shape.get_shape();
            auto output_ov_type = out_port.get_element_type();
            std::vector<int64_t> torch_shape(output_shape.begin(), output_shape.end());
            torch::Dtype torch_out_dtype = ov_dtype_to_torch(output_ov_type);
            result = torch::empty(torch_shape, torch::TensorOptions().dtype(torch_out_dtype));
            ov::Tensor output_tensor(output_ov_type, output_shape, result.data_ptr());
            infer_request.set_output_tensor(0, output_tensor);
            preallocated = true;
        }

        infer_request.infer();

        if (!preallocated) {
            auto output_tensor = infer_request.get_output_tensor();
            auto output_shape = output_tensor.get_shape();
            auto output_ov_type = output_tensor.get_element_type();

            std::vector<int64_t> torch_shape(output_shape.begin(), output_shape.end());
            torch::Dtype torch_out_dtype = ov_dtype_to_torch(output_ov_type);

            result = torch::empty(torch_shape, torch::TensorOptions().dtype(torch_out_dtype));

            // Fallback: Copy back from OV output tensor to the torch result tensor.
            std::memcpy(result.data_ptr(), output_tensor.data(), output_tensor.get_byte_size());
        }

        return result;

    } catch (const std::exception& e) {
        TORCH_CHECK(false,
                    "[Intel NPU] Execution Failed for key '" + key + "': " + std::string(e.what()));
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
    ov::element::Type type_a = torch_dtype_to_ov(a);
    ov::element::Type type_b = torch_dtype_to_ov(b);

    // Determine target execution precision
    ov::element::Type fp_type = (type_a == ov::element::f32 || type_b == ov::element::f32)
                                    ? ov::element::f32
                                    : ov::element::f16;

    auto arg_a = std::make_shared<ov::opset1::Parameter>(type_a, get_ov_shape(a));
    auto arg_b = std::make_shared<ov::opset1::Parameter>(type_b, get_ov_shape(b));

    std::shared_ptr<ov::Node> node_a = arg_a;
    std::shared_ptr<ov::Node> node_b = arg_b;

    if (type_a != fp_type) node_a = std::make_shared<ov::opset1::Convert>(arg_a, fp_type);
    if (type_b != fp_type) node_b = std::make_shared<ov::opset1::Convert>(arg_b, fp_type);

    auto op = std::make_shared<OpT>(node_a, node_b);
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_a, arg_b});
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
    ov::element::Type fp_type =
        (torch_dtype_to_ov(a) == ov::element::f16 && torch_dtype_to_ov(b) == ov::element::f16)
            ? ov::element::f16
            : ov::element::f32;
    auto arg_a = std::make_shared<ov::opset1::Parameter>(fp_type, shape_a);
    auto arg_b = std::make_shared<ov::opset1::Parameter>(fp_type, shape_b);
    auto op = std::make_shared<ov::opset1::Divide>(arg_a, arg_b);
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_a, arg_b});
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

    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov_type, input_shape);
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

    auto model = std::make_shared<ov::Model>(ov::OutputVector{output},
                                             ov::ParameterVector{arg_input, arg_weight});
    return execute_op(key, model, {input, weight});
}

torch::Tensor npu_layer_norm(torch::Tensor input, std::vector<int64_t> normalized_shape,
                             torch::Tensor weight, torch::Tensor bias, float epsilon) {
    std::string key = get_key("layernorm", {input, weight, bias}, std::to_string(epsilon));

    ov::element::Type ov_type = torch_dtype_to_ov(input);
    ov::Shape input_shape = get_ov_shape(input);

    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov_type, input_shape);

    std::vector<torch::Tensor> inputs = {input};
    ov::ParameterVector params = {arg_input};

    // Calculate axes for reduction based on normalized_shape
    int64_t rank = input_shape.size();
    int64_t norm_rank = normalized_shape.size();
    std::vector<int64_t> axes_vec;
    for (int64_t i = rank - norm_rank; i < rank; ++i) {
        axes_vec.push_back(i);
    }
    auto axes = ov::opset1::Constant::create(ov::element::i64, {axes_vec.size()}, axes_vec);

    auto mean = std::make_shared<ov::opset1::ReduceMean>(arg_input, axes, true);
    auto sub = std::make_shared<ov::opset1::Subtract>(arg_input, mean);
    auto sq = std::make_shared<ov::opset1::Multiply>(sub, sub);
    auto variance = std::make_shared<ov::opset1::ReduceMean>(sq, axes, true);

    auto eps_const = ov::opset1::Constant::create(ov_type, {}, {epsilon});
    auto var_eps = std::make_shared<ov::opset1::Add>(variance, eps_const);
    auto std_dev = std::make_shared<ov::opset1::Sqrt>(var_eps);

    std::shared_ptr<ov::Node> result = std::make_shared<ov::opset1::Divide>(sub, std_dev);

    if (weight.defined() && weight.numel() > 0) {
        auto arg_weight = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(weight));
        params.push_back(arg_weight);
        inputs.push_back(weight);
        result = std::make_shared<ov::opset1::Multiply>(result, arg_weight);
    }

    if (bias.defined() && bias.numel() > 0) {
        auto arg_bias = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(bias));
        params.push_back(arg_bias);
        inputs.push_back(bias);
        result = std::make_shared<ov::opset1::Add>(result, arg_bias);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, params);
    return execute_op(key, model, inputs);
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

    ov::element::Type ov_in_type = torch_dtype_to_ov(input);
    ov::element::Type ov_w_type = torch_dtype_to_ov(weight);
    ov::Shape shape_in = get_ov_shape(input);
    ov::Shape shape_w = get_ov_shape(weight);

    auto arg_in = std::make_shared<ov::opset1::Parameter>(ov_in_type, shape_in);
    auto arg_w = std::make_shared<ov::opset1::Parameter>(ov_w_type, shape_w);

    std::shared_ptr<ov::Node> weight_node = arg_w;
    if (ov_in_type != ov_w_type) {
        // Convert weight to input type (triggers W8A16 hardware acceleration on NPU if weight is
        // i8)
        weight_node = std::make_shared<ov::opset1::Convert>(arg_w, ov_in_type);
    }

    auto matmul = std::make_shared<ov::opset1::MatMul>(arg_in, weight_node, false, true);

    std::shared_ptr<ov::Node> result = matmul;
    ov::ParameterVector params = {arg_in, arg_w};
    std::vector<torch::Tensor> inputs = {input, weight};

    if (has_bias) {
        auto arg_b = std::make_shared<ov::opset1::Parameter>(ov_in_type, get_ov_shape(bias));
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
    auto arg_input =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    auto perm_const =
        ov::opset1::Constant::create(ov::element::i64, ov::Shape{permutation.size()}, permutation);
    auto op = std::make_shared<ov::opset1::Transpose>(arg_input, perm_const);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}

torch::Tensor npu_reshape(torch::Tensor input, std::vector<int64_t> shape) {
    std::stringstream ss;
    ss << "reshape_";
    for (auto s : shape) ss << s << ",";
    std::string key = get_key("reshape", {input}, ss.str());
    auto arg_input =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    auto shape_const =
        ov::opset1::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape);
    auto op = std::make_shared<ov::opset1::Reshape>(arg_input, shape_const, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}

torch::Tensor npu_squeeze(torch::Tensor input, std::vector<int64_t> dims) {
    std::stringstream ss;
    ss << "squeeze_";
    for (auto d : dims) ss << d << ",";
    std::string key = get_key("squeeze", {input}, ss.str());
    auto arg_input =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));

    std::shared_ptr<ov::Node> op;
    if (dims.empty()) {
        op = std::make_shared<ov::opset1::Squeeze>(arg_input);
    } else {
        auto axes_const =
            ov::opset1::Constant::create(ov::element::i64, ov::Shape{dims.size()}, dims);
        op = std::make_shared<ov::opset1::Squeeze>(arg_input, axes_const);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}

torch::Tensor npu_unsqueeze(torch::Tensor input, std::vector<int64_t> dims) {
    std::stringstream ss;
    ss << "unsqueeze_";
    for (auto d : dims) ss << d << ",";
    std::string key = get_key("unsqueeze", {input}, ss.str());
    auto arg_input =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    auto axes_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{dims.size()}, dims);
    auto op = std::make_shared<ov::opset1::Unsqueeze>(arg_input, axes_const);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}

torch::Tensor npu_cat(std::vector<torch::Tensor> tensors, int64_t dim) {
    if (tensors.empty()) throw std::runtime_error("npu_cat expects at least one tensor");
    if (dim < 0) dim += tensors[0].dim();
    std::string key = get_key("cat", tensors, std::to_string(dim));

    ov::ParameterVector params;
    ov::OutputVector concat_inputs;

    for (const auto& t : tensors) {
        auto param = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(t), get_ov_shape(t));
        params.push_back(param);
        concat_inputs.push_back(param);
    }

    auto concat = std::make_shared<ov::opset1::Concat>(concat_inputs, dim);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{concat}, params);
    return execute_op(key, model, tensors);
}

torch::Tensor npu_stack(std::vector<torch::Tensor> tensors, int64_t dim) {
    if (tensors.empty()) throw std::runtime_error("npu_stack expects at least one tensor");
    int64_t rank = tensors[0].dim();
    if (dim < 0) dim += rank + 1;

    std::string key = get_key("stack", tensors, std::to_string(dim));

    ov::ParameterVector params;
    ov::OutputVector concat_inputs;

    for (const auto& t : tensors) {
        auto param = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(t), get_ov_shape(t));
        params.push_back(param);

        auto unsqueeze_dim = ov::opset1::Constant::create(ov::element::i64, {1}, {dim});
        auto unsqueeze = std::make_shared<ov::opset1::Unsqueeze>(param, unsqueeze_dim);
        concat_inputs.push_back(unsqueeze);
    }

    auto concat = std::make_shared<ov::opset1::Concat>(concat_inputs, dim);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{concat}, params);
    return execute_op(key, model, tensors);
}

torch::Tensor npu_mean(torch::Tensor input, std::vector<int64_t> dim, bool keepdim) {
    std::stringstream ss;
    for (auto d : dim) ss << d << ",";
    std::string key = get_key("mean", {input}, ss.str() + std::to_string(keepdim));

    auto arg_input =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));

    if (dim.empty()) {
        int64_t rank = input.dim();
        for (int64_t i = 0; i < rank; ++i) dim.push_back(i);
    }

    auto axes = ov::opset1::Constant::create(ov::element::i64, {dim.size()}, dim);
    auto reduce = std::make_shared<ov::opset1::ReduceMean>(arg_input, axes, keepdim);

    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}

torch::Tensor npu_index_select(torch::Tensor input, int64_t dim, torch::Tensor index) {
    std::string key = get_key("index_select", {input, index}, std::to_string(dim));
    auto arg_input =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    auto arg_index =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(index), get_ov_shape(index));
    auto axis_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {dim});
    auto op = std::make_shared<ov::opset1::Gather>(arg_input, arg_index, axis_const);
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input, arg_index});
    return execute_op(key, model, {input, index});
}

torch::Tensor npu_scaled_dot_product_attention(torch::Tensor query, torch::Tensor key,
                                               torch::Tensor value, torch::Tensor attn_mask,
                                               double dropout_p, bool is_causal, double scale) {
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
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask,
                                                                      arg_scale, is_causal);
    } else if (arg_mask && !arg_scale) {
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask,
                                                                      is_causal);
    } else if (!arg_mask && arg_scale) {
        arg_mask = ov::opset1::Constant::create(ov::element::f32, {}, {0.0f});
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v, arg_mask,
                                                                      arg_scale, is_causal);
    } else {
        op = std::make_shared<ov::opset13::ScaledDotProductAttention>(arg_q, arg_k, arg_v,
                                                                      is_causal);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, params);
    return execute_op(op_key, model, inputs);
}

torch::Tensor npu_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         std::vector<int64_t> stride, std::vector<int64_t> padding,
                         std::vector<int64_t> dilation, int64_t groups) {
    std::stringstream ss;
    ss << "conv2d_";
    for (auto s : stride) ss << s << ",";
    for (auto p : padding) ss << p << ",";
    for (auto d : dilation) ss << d << ",";
    ss << groups;

    std::string key = get_key("conv2d", {input, weight}, ss.str());
    if (bias.defined() && bias.numel() > 0) key += "_bias";

    ov::element::Type ov_type = torch_dtype_to_ov(input);
    auto arg_input = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(input));
    auto arg_weight = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(weight));

    ov::Strides ov_strides(stride.begin(), stride.end());
    ov::CoordinateDiff ov_pads_begin(padding.begin(), padding.end());
    ov::CoordinateDiff ov_pads_end(padding.begin(), padding.end());
    ov::Strides ov_dilations(dilation.begin(), dilation.end());

    std::shared_ptr<ov::Node> conv_op;

    if (groups == 1) {
        conv_op = std::make_shared<ov::opset1::Convolution>(
            arg_input, arg_weight, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations);
    } else {
        int64_t out_channels = weight.size(0);
        int64_t in_channels_per_group = weight.size(1);
        int64_t kH = weight.size(2);
        int64_t kW = weight.size(3);

        auto shape_const = ov::opset1::Constant::create(
            ov::element::i64, {5}, {groups, out_channels / groups, in_channels_per_group, kH, kW});
        auto reshaped_w = std::make_shared<ov::opset1::Reshape>(arg_weight, shape_const, false);

        conv_op = std::make_shared<ov::opset1::GroupConvolution>(
            arg_input, reshaped_w, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations);
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

torch::Tensor npu_max_pool2d(torch::Tensor input, std::vector<int64_t> kernel_size,
                             std::vector<int64_t> stride, std::vector<int64_t> padding,
                             std::vector<int64_t> dilation, bool ceil_mode) {
    std::stringstream ss;
    ss << "maxpool2d_";
    for (auto k : kernel_size) ss << k << ",";
    for (auto s : stride) ss << s << ",";
    for (auto p : padding) ss << p << ",";
    for (auto d : dilation) ss << d << ",";
    ss << ceil_mode;

    std::string key = get_key("maxpool2d", {input}, ss.str());
    auto arg_input =
        std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));

    ov::Strides ov_strides(stride.begin(), stride.end());
    ov::Shape ov_kernel(kernel_size.begin(), kernel_size.end());
    ov::Shape ov_pads_begin(padding.begin(), padding.end());
    ov::Shape ov_pads_end(padding.begin(), padding.end());
    ov::Strides ov_dilations(dilation.begin(), dilation.end());

    auto rounding_type = ceil_mode ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;

    auto op =
        std::make_shared<ov::opset1::MaxPool>(arg_input, ov_strides, ov_pads_begin, ov_pads_end,
                                              ov_kernel, rounding_type, ov::op::PadType::EXPLICIT);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{arg_input});
    return execute_op(key, model, {input});
}

torch::Tensor npu_update_kv_cache(torch::Tensor cache, torch::Tensor new_kv,
                                  torch::Tensor position) {
    std::string key = get_key("update_kv_cache", {cache, new_kv, position});

    ov::element::Type ov_type = torch_dtype_to_ov(cache);
    ov::Shape cache_shape = get_ov_shape(cache);
    ov::Shape new_kv_shape = get_ov_shape(new_kv);
    ov::Shape pos_shape = get_ov_shape(position);

    auto arg_cache = std::make_shared<ov::opset1::Parameter>(ov_type, cache_shape);
    auto arg_new_kv = std::make_shared<ov::opset1::Parameter>(ov_type, new_kv_shape);
    auto arg_pos = std::make_shared<ov::opset1::Parameter>(ov::element::i64, pos_shape);

    int64_t seq_len = new_kv.size(1);

    std::shared_ptr<ov::Node> pos_node = arg_pos;
    if (position.scalar_type() != torch::kLong) {
        pos_node = std::make_shared<ov::opset1::Convert>(arg_pos, ov::element::i64);
    }

    // Stop position = position + seq_len
    auto seq_len_const = ov::opset1::Constant::create(ov::element::i64, {}, {seq_len});
    auto stop_node = std::make_shared<ov::opset1::Add>(pos_node, seq_len_const);

    auto step_const = ov::opset1::Constant::create(ov::element::i64, {}, {1});

    // Range: start, stop, step, output_type
    auto range_op =
        std::make_shared<ov::opset4::Range>(pos_node, stop_node, step_const, ov::element::i64);

    // ScatterUpdate: data, indices, updates, axis
    auto axis_const =
        ov::opset1::Constant::create(ov::element::i64, {}, {1});  // Axis 1 (sequence length dim)

    auto scatter_op =
        std::make_shared<ov::opset3::ScatterUpdate>(arg_cache, range_op, arg_new_kv, axis_const);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{scatter_op},
                                             ov::ParameterVector{arg_cache, arg_new_kv, arg_pos});

    return execute_op(key, model, {cache, new_kv, position});
}

torch::Tensor npu_quantized_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor scale,
                                   torch::Tensor zero_point, torch::Tensor bias) {
    bool has_zp = zero_point.defined() && zero_point.numel() > 0;
    bool has_bias = bias.defined() && bias.numel() > 0;

    std::string key = get_key("quantized_linear", {input, weight, scale},
                              (has_zp ? "_zp" : "") + std::string(has_bias ? "_bias" : ""));

    ov::element::Type ov_in_type = torch_dtype_to_ov(input);
    ov::element::Type ov_w_type = torch_dtype_to_ov(weight);
    ov::element::Type ov_scale_type = torch_dtype_to_ov(scale);

    ov::Shape shape_in = get_ov_shape(input);
    ov::Shape shape_w = get_ov_shape(weight);
    ov::Shape shape_scale = get_ov_shape(scale);

    auto arg_in = std::make_shared<ov::opset1::Parameter>(ov_in_type, shape_in);
    auto arg_w = std::make_shared<ov::opset1::Parameter>(ov_w_type, shape_w);
    auto arg_scale = std::make_shared<ov::opset1::Parameter>(ov_scale_type, shape_scale);

    bool is_int4 = (weight.scalar_type() == torch::kByte && weight.dim() == 2 &&
                    weight.size(1) == input.size(-1) / 2);

    // 1. Convert/unpack weight to the input floating precision (e.g., f16 or f32)
    std::shared_ptr<ov::Node> weight_float;
    if (is_int4) {
        auto w_cast = std::make_shared<ov::opset1::Convert>(arg_w, ov_in_type);
        auto scale_16 = ov::opset1::Constant::create(ov_in_type, {}, {16.0});

        auto w_div = std::make_shared<ov::opset1::Divide>(w_cast, scale_16);
        auto w_odd = std::make_shared<ov::opset1::Floor>(w_div);

        auto w_odd_mul = std::make_shared<ov::opset1::Multiply>(w_odd, scale_16);
        auto w_even = std::make_shared<ov::opset1::Subtract>(w_cast, w_odd_mul);

        int64_t c_out = shape_w[0];
        int64_t c_in_half = shape_w[1];
        auto reshape_3d = ov::opset1::Constant::create(ov::element::i64, {3},
                                                       std::vector<int64_t>{c_out, c_in_half, 1});
        auto w_even_3d = std::make_shared<ov::opset1::Reshape>(w_even, reshape_3d, false);
        auto w_odd_3d = std::make_shared<ov::opset1::Reshape>(w_odd, reshape_3d, false);

        auto w_concat =
            std::make_shared<ov::opset1::Concat>(ov::NodeVector{w_even_3d, w_odd_3d}, 2);

        auto final_shape = ov::opset1::Constant::create(ov::element::i64, {2},
                                                        std::vector<int64_t>{c_out, c_in_half * 2});
        weight_float = std::make_shared<ov::opset1::Reshape>(w_concat, final_shape, false);
    } else {
        weight_float = std::make_shared<ov::opset1::Convert>(arg_w, ov_in_type);
    }

    ov::ParameterVector params = {arg_in, arg_w, arg_scale};
    std::vector<torch::Tensor> inputs = {input, weight, scale};

    // 2. Subtract zero point if defined
    if (has_zp) {
        ov::element::Type ov_zp_type = torch_dtype_to_ov(zero_point);
        auto arg_zp = std::make_shared<ov::opset1::Parameter>(ov_zp_type, get_ov_shape(zero_point));

        std::shared_ptr<ov::Node> zp_node = arg_zp;
        if (ov_zp_type != ov_in_type) {
            zp_node = std::make_shared<ov::opset1::Convert>(arg_zp, ov_in_type);
        }

        weight_float = std::make_shared<ov::opset1::Subtract>(weight_float, zp_node);
        params.push_back(arg_zp);
        inputs.push_back(zero_point);
    }

    // 3. Multiply by dequantization scale
    std::shared_ptr<ov::Node> scale_node = arg_scale;
    if (ov_scale_type != ov_in_type) {
        scale_node = std::make_shared<ov::opset1::Convert>(arg_scale, ov_in_type);
    }
    weight_float = std::make_shared<ov::opset1::Multiply>(weight_float, scale_node);

    // 4. Perform MatMul with transposed weights: y = x * W^T
    auto matmul = std::make_shared<ov::opset1::MatMul>(arg_in, weight_float, false, true);

    std::shared_ptr<ov::Node> result = matmul;

    // 5. Add bias if defined
    if (has_bias) {
        auto arg_bias = std::make_shared<ov::opset1::Parameter>(ov_in_type, get_ov_shape(bias));
        result = std::make_shared<ov::opset1::Add>(result, arg_bias);
        params.push_back(arg_bias);
        inputs.push_back(bias);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, params);
    return execute_op(key, model, inputs);
}
