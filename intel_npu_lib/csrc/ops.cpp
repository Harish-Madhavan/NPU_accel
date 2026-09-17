#include "include/ops.h"

#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset13.hpp>  // For ScaledDotProductAttention
#include <openvino/opsets/opset3.hpp>   // For ScatterUpdate
#include <openvino/opsets/opset4.hpp>   // For Swish (SiLU)
#include <openvino/opsets/opset5.hpp>   // For LogSoftmax
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
    static const std::unordered_map<c10::ScalarType, ov::element::Type> map = {
        {torch::kFloat, ov::element::f32}, {torch::kHalf, ov::element::f16},
        {torch::kBFloat16, ov::element::bf16}, {torch::kDouble, ov::element::f64},
        {torch::kInt, ov::element::i32}, {torch::kLong, ov::element::i64},
        {torch::kShort, ov::element::i16}, {torch::kChar, ov::element::i8},
        {torch::kByte, ov::element::u8}, {torch::kBool, ov::element::boolean}
    };
    auto it = map.find(t.scalar_type());
    if (it != map.end()) return it->second;
    std::cerr << "[Intel NPU] Unsupported torch dtype (" << t.scalar_type()
              << "); falling back to f32." << std::endl;
    return ov::element::f32;
}

static torch::Dtype ov_dtype_to_torch(ov::element::Type ov_type) {
    static const std::unordered_map<ov::element::Type_t, torch::Dtype> map = {
        {ov::element::f32, torch::kFloat}, {ov::element::f16, torch::kHalf},
        {ov::element::bf16, torch::kBFloat16}, {ov::element::f64, torch::kDouble},
        {ov::element::i32, torch::kInt}, {ov::element::i64, torch::kLong},
        {ov::element::i16, torch::kShort}, {ov::element::i8, torch::kChar},
        {ov::element::u8, torch::kByte}, {ov::element::boolean, torch::kBool}
    };
    auto it = map.find(ov_type);
    return it != map.end() ? it->second : torch::kFloat;
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
        // Fold the active compile configuration (performance hint + eager
        // device) into the cache key so a hint change can't reuse a binary
        // compiled under different settings.
        const std::string full_key = key + "_" + backend.getCompileConfigKey();
        backend.getOrCompileModel(full_key, model);
        auto infer_request = backend.getOrCachedInferRequest(full_key);

        // Ensure inputs are contiguous and kept alive for the duration of inference
        std::vector<torch::Tensor> contiguous_inputs;
        contiguous_inputs.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            contiguous_inputs.push_back(ensure_contiguous(inputs[i]));
        }

        // Map inputs (zero-copy wrapping of PyTorch tensor memory)
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
    // Unreachable — TORCH_CHECK always throws; silences C4715.
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
    return execute_binary_op_helper<ov::opset1::Divide>("div", a, b);
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

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

torch::Tensor npu_embedding(torch::Tensor weight, torch::Tensor indices) {
    std::string key = get_key("embedding", {weight, indices});
    auto arg_w = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(weight), get_ov_shape(weight));
    auto arg_idx = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(indices), get_ov_shape(indices));
    auto axis = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::opset8::Gather>(arg_w, arg_idx, axis);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{arg_w, arg_idx});
    return execute_op(key, model, {weight, indices});
}

torch::Tensor npu_embedding_backward(torch::Tensor grad_output, torch::Tensor indices, int64_t num_embeddings) {
    // Embedding gradient scatter
    auto grad_out_flat = grad_output.reshape({-1, grad_output.size(-1)});
    auto idx_flat = indices.reshape({-1});
    torch::Tensor grad_weight = torch::zeros({num_embeddings, grad_output.size(-1)}, grad_output.options());
    grad_weight.index_add_(0, idx_flat, grad_out_flat);
    return grad_weight;
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// ---------------------------------------------------------------------------

torch::Tensor npu_rotary_embedding(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    x = x.contiguous();
    cos = cos.contiguous();
    sin = sin.contiguous();

    std::string key = get_key("rotary_embedding", {x, cos, sin});
    auto ov_type = torch_dtype_to_ov(x);
    auto arg_x = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(x));
    auto arg_cos = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(cos));
    auto arg_sin = std::make_shared<ov::opset1::Parameter>(ov_type, get_ov_shape(sin));

    int64_t last_dim = x.size(-1);
    int64_t half_dim = last_dim / 2;

    // Split x into x1 and x2 along last dimension
    auto split_lengths = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {half_dim, half_dim});
    auto axis_node = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    auto split = std::make_shared<ov::opset1::VariadicSplit>(arg_x, axis_node, split_lengths);

    auto x1 = split->output(0);
    auto x2 = split->output(1);

    // out1 = x1 * cos - x2 * sin
    auto x1_cos = std::make_shared<ov::opset1::Multiply>(x1, arg_cos);
    auto x2_sin = std::make_shared<ov::opset1::Multiply>(x2, arg_sin);
    auto out1 = std::make_shared<ov::opset1::Subtract>(x1_cos, x2_sin);

    // out2 = x1 * sin + x2 * cos
    auto x1_sin = std::make_shared<ov::opset1::Multiply>(x1, arg_sin);
    auto x2_cos = std::make_shared<ov::opset1::Multiply>(x2, arg_cos);
    auto out2 = std::make_shared<ov::opset1::Add>(x1_sin, x2_cos);

    // Concat along last axis
    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{out1, out2}, -1);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{arg_x, arg_cos, arg_sin});
    return execute_op(key, model, {x, cos, sin});
}

// ---------------------------------------------------------------------------
// Backward Autograd Operators
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> npu_matmul_backward(torch::Tensor grad_output, torch::Tensor a, torch::Tensor b) {
    auto a_dim = a.dim();
    auto b_dim = b.dim();

    std::vector<int64_t> perm_a(a_dim);
    for (int64_t i = 0; i < a_dim; ++i) perm_a[i] = i;
    if (a_dim >= 2) std::swap(perm_a[a_dim - 1], perm_a[a_dim - 2]);

    std::vector<int64_t> perm_b(b_dim);
    for (int64_t i = 0; i < b_dim; ++i) perm_b[i] = i;
    if (b_dim >= 2) std::swap(perm_b[b_dim - 1], perm_b[b_dim - 2]);

    auto b_t = (b_dim >= 2) ? npu_transpose(b, perm_b) : b;
    auto a_t = (a_dim >= 2) ? npu_transpose(a, perm_a) : a;

    auto grad_a = npu_matmul(grad_output, b_t);
    auto grad_b = npu_matmul(a_t, grad_output);
    return {grad_a, grad_b};
}

std::vector<torch::Tensor> npu_linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                               bool needs_input_grad, bool needs_weight_grad, bool needs_bias_grad) {
    torch::Tensor grad_input, grad_weight, grad_bias;
    if (needs_input_grad) {
        grad_input = npu_matmul(grad_output, weight);
    }
    if (needs_weight_grad) {
        auto grad_out_2d = grad_output.reshape({-1, grad_output.size(-1)});
        auto in_2d = input.reshape({-1, input.size(-1)});
        grad_weight = npu_matmul(grad_out_2d.t(), in_2d);
    }
    if (needs_bias_grad) {
        auto grad_out_2d = grad_output.reshape({-1, grad_output.size(-1)});
        grad_bias = npu_mean(grad_out_2d, {0}, false) * static_cast<float>(grad_out_2d.size(0));
    }
    return {grad_input, grad_weight, grad_bias};
}

torch::Tensor npu_relu_backward(torch::Tensor grad_output, torch::Tensor input) {
    std::string key = get_key("relu_backward", {grad_output, input});
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad_output), get_ov_shape(grad_output));
    auto arg_in = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    auto zero = ov::opset1::Constant::create(arg_in->get_element_type(), ov::Shape{}, {0});
    auto mask = std::make_shared<ov::opset1::Greater>(arg_in, zero);
    auto zero_g = ov::opset1::Constant::create(arg_g->get_element_type(), ov::Shape{}, {0});
    auto grad_in = std::make_shared<ov::opset1::Select>(mask, arg_g, zero_g);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{grad_in}, ov::ParameterVector{arg_g, arg_in});
    return execute_op(key, model, {grad_output, input});
}

torch::Tensor npu_gelu_backward(torch::Tensor grad_output, torch::Tensor input) {
    std::string key = get_key("gelu_backward", {grad_output, input});
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad_output), get_ov_shape(grad_output));
    auto arg_x = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    
    // Exact GELU derivative: cdf = 0.5 * (1 + erf(x / sqrt(2))); pdf = (1 / sqrt(2*pi)) * exp(-0.5 * x^2)
    // deriv = cdf + x * pdf
    auto elem_type = arg_x->get_element_type();
    auto c_inv_sqrt2 = ov::opset1::Constant::create(elem_type, ov::Shape{}, {0.7071067811865475f});
    auto c_inv_sqrt2pi = ov::opset1::Constant::create(elem_type, ov::Shape{}, {0.3989422804014327f});
    auto c_half = ov::opset1::Constant::create(elem_type, ov::Shape{}, {0.5f});
    auto c_one = ov::opset1::Constant::create(elem_type, ov::Shape{}, {1.0f});
    auto c_minus_half = ov::opset1::Constant::create(elem_type, ov::Shape{}, {-0.5f});

    auto x_scaled = std::make_shared<ov::opset1::Multiply>(arg_x, c_inv_sqrt2);
    auto erf_node = std::make_shared<ov::opset1::Erf>(x_scaled);
    auto cdf = std::make_shared<ov::opset1::Multiply>(c_half, std::make_shared<ov::opset1::Add>(c_one, erf_node));

    auto x_sq = std::make_shared<ov::opset1::Multiply>(arg_x, arg_x);
    auto exp_arg = std::make_shared<ov::opset1::Multiply>(c_minus_half, x_sq);
    auto exp_node = std::make_shared<ov::opset1::Exp>(exp_arg);
    auto pdf = std::make_shared<ov::opset1::Multiply>(c_inv_sqrt2pi, exp_node);

    auto deriv = std::make_shared<ov::opset1::Add>(cdf, std::make_shared<ov::opset1::Multiply>(arg_x, pdf));
    auto grad_in = std::make_shared<ov::opset1::Multiply>(arg_g, deriv);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{grad_in}, ov::ParameterVector{arg_g, arg_x});
    return execute_op(key, model, {grad_output, input});
}

torch::Tensor npu_silu_backward(torch::Tensor grad_output, torch::Tensor input) {
    std::string key = get_key("silu_backward", {grad_output, input});
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad_output), get_ov_shape(grad_output));
    auto arg_x = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(input), get_ov_shape(input));
    
    // Exact SiLU derivative: sig = sigmoid(x); deriv = sig * (1 + x * (1 - sig))
    auto elem_type = arg_x->get_element_type();
    auto sig = std::make_shared<ov::opset1::Sigmoid>(arg_x);
    auto one = ov::opset1::Constant::create(elem_type, ov::Shape{}, {1.0f});
    auto one_minus_sig = std::make_shared<ov::opset1::Subtract>(one, sig);
    auto x_term = std::make_shared<ov::opset1::Multiply>(arg_x, one_minus_sig);
    auto bracket = std::make_shared<ov::opset1::Add>(one, x_term);
    auto deriv = std::make_shared<ov::opset1::Multiply>(sig, bracket);
    auto grad_in = std::make_shared<ov::opset1::Multiply>(arg_g, deriv);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{grad_in}, ov::ParameterVector{arg_g, arg_x});
    return execute_op(key, model, {grad_output, input});
}

torch::Tensor npu_softmax_backward(torch::Tensor grad_output, torch::Tensor output, int64_t dim) {
    if (dim < 0) dim += grad_output.dim();
    std::string key = get_key("softmax_backward", {grad_output, output}, std::to_string(dim));
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad_output), get_ov_shape(grad_output));
    auto arg_y = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(output), get_ov_shape(output));

    auto gy = std::make_shared<ov::opset1::Multiply>(arg_g, arg_y);
    auto axis = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {dim});
    auto sum_gy = std::make_shared<ov::opset1::ReduceSum>(gy, axis, true);
    auto diff = std::make_shared<ov::opset1::Subtract>(arg_g, sum_gy);
    auto grad_in = std::make_shared<ov::opset1::Multiply>(arg_y, diff);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{grad_in}, ov::ParameterVector{arg_g, arg_y});
    return execute_op(key, model, {grad_output, output});
}

std::vector<torch::Tensor> npu_rmsnorm_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, float epsilon) {
    auto go = grad_output.to(torch::kFloat32);
    auto in_f = input.to(torch::kFloat32);
    auto w_f = weight.to(torch::kFloat32);

    auto rms = torch::sqrt(in_f.pow(2).mean(-1, true) + epsilon);
    auto xn = in_f / rms;

    std::vector<int64_t> sum_dims(go.dim() - 1);
    std::iota(sum_dims.begin(), sum_dims.end(), 0);
    auto grad_w = (go * xn).sum(sum_dims).to(weight.dtype());

    auto dL_dxn = go * w_f;
    auto correction = (dL_dxn * xn).mean(-1, true);
    auto grad_in = ((dL_dxn - xn * correction) / rms).to(grad_output.dtype());

    return {grad_in, grad_w};
}

// ---------------------------------------------------------------------------
// Loss Functions
// ---------------------------------------------------------------------------

torch::Tensor npu_mse_loss(torch::Tensor pred, torch::Tensor target, int64_t reduction) {
    std::string key = get_key("mse_loss", {pred, target}, std::to_string(reduction));
    auto arg_p = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(pred), get_ov_shape(pred));
    auto arg_t = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(target), get_ov_shape(target));
    auto diff = std::make_shared<ov::opset1::Subtract>(arg_p, arg_t);
    auto sq = std::make_shared<ov::opset1::Multiply>(diff, diff);
    std::shared_ptr<ov::Node> result = sq;
    if (reduction == 1) {
        std::vector<int64_t> axes(pred.dim());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        result = std::make_shared<ov::opset1::ReduceMean>(sq, axes_c, false);
    } else if (reduction == 2) {
        std::vector<int64_t> axes(pred.dim());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        result = std::make_shared<ov::opset1::ReduceSum>(sq, axes_c, false);
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{arg_p, arg_t});
    return execute_op(key, model, {pred, target});
}

torch::Tensor npu_mse_loss_backward(torch::Tensor grad_output, torch::Tensor pred, torch::Tensor target, int64_t reduction) {
    std::string key = get_key("mse_loss_backward", {grad_output, pred, target}, std::to_string(reduction));
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad_output), get_ov_shape(grad_output));
    auto arg_p = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(pred), get_ov_shape(pred));
    auto arg_t = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(target), get_ov_shape(target));
    auto diff = std::make_shared<ov::opset1::Subtract>(arg_p, arg_t);
    
    float scale_val = 2.0f;
    if (reduction == 1) {
        scale_val = 2.0f / static_cast<float>(pred.numel());
    }
    auto scale = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {scale_val});
    auto grad = std::make_shared<ov::opset1::Multiply>(std::make_shared<ov::opset1::Multiply>(scale, diff), arg_g);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{grad}, ov::ParameterVector{arg_g, arg_p, arg_t});
    return execute_op(key, model, {grad_output, pred, target});
}

torch::Tensor npu_cross_entropy_loss(torch::Tensor pred, torch::Tensor target, int64_t reduction) {
    std::string key = get_key("cross_entropy_loss", {pred, target}, std::to_string(reduction));
    auto arg_p = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(pred), get_ov_shape(pred));
    auto arg_t = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(target), get_ov_shape(target));

    auto log_sm = std::make_shared<ov::opset5::LogSoftmax>(arg_p, -1);

    std::shared_ptr<ov::Node> nll;
    if (target.dim() == pred.dim()) {
        auto prod = std::make_shared<ov::opset1::Multiply>(arg_t, log_sm);
        auto neg_one = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {-1.0f});
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        nll = std::make_shared<ov::opset1::Multiply>(
            neg_one,
            std::make_shared<ov::opset1::ReduceSum>(prod, axes_c, false)
        );
    } else {
        int64_t num_classes = pred.size(-1);
        auto depth_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {num_classes});
        auto on_c = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {1.0f});
        auto off_c = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {0.0f});
        auto one_hot = std::make_shared<ov::opset1::OneHot>(arg_t, depth_c, on_c, off_c, -1);

        auto prod = std::make_shared<ov::opset1::Multiply>(one_hot, log_sm);
        auto neg_one = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {-1.0f});
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        nll = std::make_shared<ov::opset1::Multiply>(
            neg_one,
            std::make_shared<ov::opset1::ReduceSum>(prod, axes_c, false)
        );
    }

    std::shared_ptr<ov::Node> result = nll;
    if (reduction == 1) {
        std::vector<int64_t> axes(target.dim());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        result = std::make_shared<ov::opset1::ReduceMean>(nll, axes_c, false);
    } else if (reduction == 2) {
        std::vector<int64_t> axes(target.dim());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        result = std::make_shared<ov::opset1::ReduceSum>(nll, axes_c, false);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{arg_p, arg_t});
    return execute_op(key, model, {pred, target});
}

torch::Tensor npu_cross_entropy_loss_backward(torch::Tensor grad_output, torch::Tensor pred, torch::Tensor target, int64_t reduction) {
    std::string key = get_key("cross_entropy_loss_backward", {grad_output, pred, target}, std::to_string(reduction));
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad_output), get_ov_shape(grad_output));
    auto arg_p = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(pred), get_ov_shape(pred));
    auto arg_t = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(target), get_ov_shape(target));

    auto sm = std::make_shared<ov::opset1::Softmax>(arg_p, -1);

    std::shared_ptr<ov::Node> target_one_hot;
    if (target.dim() == pred.dim()) {
        target_one_hot = arg_t;
    } else {
        int64_t num_classes = pred.size(-1);
        auto depth_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {num_classes});
        auto on_c = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {1.0f});
        auto off_c = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {0.0f});
        target_one_hot = std::make_shared<ov::opset1::OneHot>(arg_t, depth_c, on_c, off_c, -1);
    }

    auto diff = std::make_shared<ov::opset1::Subtract>(sm, target_one_hot);

    std::shared_ptr<ov::Node> scaled_diff = diff;
    if (reduction == 1) {
        float n_val = static_cast<float>(pred.dim() > 1 ? pred.size(0) : 1);
        auto scale_c = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {1.0f / n_val});
        scaled_diff = std::make_shared<ov::opset1::Multiply>(diff, scale_c);
    }

    auto grad = std::make_shared<ov::opset1::Multiply>(scaled_diff, arg_g);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{grad}, ov::ParameterVector{arg_g, arg_p, arg_t});
    return execute_op(key, model, {grad_output, pred, target});
}

torch::Tensor npu_l1_loss(torch::Tensor pred, torch::Tensor target, int64_t reduction) {
    std::string key = get_key("l1_loss", {pred, target}, std::to_string(reduction));
    auto arg_p = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(pred), get_ov_shape(pred));
    auto arg_t = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(target), get_ov_shape(target));

    auto diff = std::make_shared<ov::opset1::Subtract>(arg_p, arg_t);
    auto abs_diff = std::make_shared<ov::opset1::Abs>(diff);

    std::shared_ptr<ov::Node> result = abs_diff;
    if (reduction == 1) {
        std::vector<int64_t> axes(pred.dim());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        result = std::make_shared<ov::opset1::ReduceMean>(abs_diff, axes_c, false);
    } else if (reduction == 2) {
        std::vector<int64_t> axes(pred.dim());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_c = ov::opset1::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        result = std::make_shared<ov::opset1::ReduceSum>(abs_diff, axes_c, false);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{arg_p, arg_t});
    return execute_op(key, model, {pred, target});
}

torch::Tensor npu_l1_loss_backward(torch::Tensor grad_output, torch::Tensor pred, torch::Tensor target, int64_t reduction) {
    std::string key = get_key("l1_loss_backward", {grad_output, pred, target}, std::to_string(reduction));
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad_output), get_ov_shape(grad_output));
    auto arg_p = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(pred), get_ov_shape(pred));
    auto arg_t = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(target), get_ov_shape(target));

    auto diff = std::make_shared<ov::opset1::Subtract>(arg_p, arg_t);
    auto sgn = std::make_shared<ov::opset1::Sign>(diff);

    std::shared_ptr<ov::Node> scaled_sgn = sgn;
    if (reduction == 1) {
        float n_val = static_cast<float>(pred.numel());
        auto scale_c = ov::opset1::Constant::create(arg_p->get_element_type(), ov::Shape{}, {1.0f / n_val});
        scaled_sgn = std::make_shared<ov::opset1::Multiply>(sgn, scale_c);
    }

    auto grad = std::make_shared<ov::opset1::Multiply>(scaled_sgn, arg_g);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{grad}, ov::ParameterVector{arg_g, arg_p, arg_t});
    return execute_op(key, model, {grad_output, pred, target});
}

// ---------------------------------------------------------------------------
// Hardware Optimizer Operators
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> npu_adam_step(torch::Tensor param, torch::Tensor grad,
                                         torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
                                         double lr, double beta1, double beta2, double eps,
                                         double weight_decay, int64_t step) {
    std::string extra = std::to_string(lr) + "_" + std::to_string(beta1) + "_" +
                        std::to_string(beta2) + "_" + std::to_string(eps) + "_" +
                        std::to_string(weight_decay) + "_" + std::to_string(step);
    std::string key = get_key("adam_step", {param, grad, exp_avg, exp_avg_sq}, extra);

    auto arg_p = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(param), get_ov_shape(param));
    auto arg_g = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(grad), get_ov_shape(grad));
    auto arg_m = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(exp_avg), get_ov_shape(exp_avg));
    auto arg_v = std::make_shared<ov::opset1::Parameter>(torch_dtype_to_ov(exp_avg_sq), get_ov_shape(exp_avg_sq));

    auto elem_type = arg_p->get_element_type();

    // Effective grad with weight decay
    std::shared_ptr<ov::Node> effective_grad = arg_g;
    if (weight_decay != 0.0) {
        auto wd_c = ov::opset1::Constant::create(elem_type, ov::Shape{}, {static_cast<float>(weight_decay)});
        effective_grad = std::make_shared<ov::opset1::Add>(arg_g, std::make_shared<ov::opset1::Multiply>(wd_c, arg_p));
    }

    // m_t = beta1 * m_{t-1} + (1 - beta1) * g
    auto b1_c = ov::opset1::Constant::create(elem_type, ov::Shape{}, {static_cast<float>(beta1)});
    auto b1_inv = ov::opset1::Constant::create(elem_type, ov::Shape{}, {static_cast<float>(1.0 - beta1)});
    auto new_m = std::make_shared<ov::opset1::Add>(
        std::make_shared<ov::opset1::Multiply>(b1_c, arg_m),
        std::make_shared<ov::opset1::Multiply>(b1_inv, effective_grad)
    );

    // v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    auto b2_c = ov::opset1::Constant::create(elem_type, ov::Shape{}, {static_cast<float>(beta2)});
    auto b2_inv = ov::opset1::Constant::create(elem_type, ov::Shape{}, {static_cast<float>(1.0 - beta2)});
    auto g_sq = std::make_shared<ov::opset1::Multiply>(effective_grad, effective_grad);
    auto new_v = std::make_shared<ov::opset1::Add>(
        std::make_shared<ov::opset1::Multiply>(b2_c, arg_v),
        std::make_shared<ov::opset1::Multiply>(b2_inv, g_sq)
    );

    // Bias corrections
    float bias_correction1 = static_cast<float>(1.0 - std::pow(beta1, step));
    float bias_correction2 = static_cast<float>(1.0 - std::pow(beta2, step));
    auto bc1_c = ov::opset1::Constant::create(elem_type, ov::Shape{}, {bias_correction1});
    auto bc2_c = ov::opset1::Constant::create(elem_type, ov::Shape{}, {bias_correction2});

    auto m_hat = std::make_shared<ov::opset1::Divide>(new_m, bc1_c);
    auto v_hat = std::make_shared<ov::opset1::Divide>(new_v, bc2_c);

    // p_t = p_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps))
    auto eps_c = ov::opset1::Constant::create(elem_type, ov::Shape{}, {static_cast<float>(eps)});
    auto sqrt_v = std::make_shared<ov::opset1::Sqrt>(v_hat);
    auto denom = std::make_shared<ov::opset1::Add>(sqrt_v, eps_c);
    auto step_val = std::make_shared<ov::opset1::Divide>(m_hat, denom);

    auto lr_c = ov::opset1::Constant::create(elem_type, ov::Shape{}, {static_cast<float>(lr)});
    auto update = std::make_shared<ov::opset1::Multiply>(lr_c, step_val);
    auto new_p = std::make_shared<ov::opset1::Subtract>(arg_p, update);

    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{new_p, new_m, new_v},
        ov::ParameterVector{arg_p, arg_g, arg_m, arg_v}
    );

    // Execute combined optimizer step
    auto res_p = execute_op(key + "_p", std::make_shared<ov::Model>(ov::OutputVector{new_p}, ov::ParameterVector{arg_p, arg_g, arg_m, arg_v}), {param, grad, exp_avg, exp_avg_sq});
    auto res_m = execute_op(key + "_m", std::make_shared<ov::Model>(ov::OutputVector{new_m}, ov::ParameterVector{arg_p, arg_g, arg_m, arg_v}), {param, grad, exp_avg, exp_avg_sq});
    auto res_v = execute_op(key + "_v", std::make_shared<ov::Model>(ov::OutputVector{new_v}, ov::ParameterVector{arg_p, arg_g, arg_m, arg_v}), {param, grad, exp_avg, exp_avg_sq});

    return {res_p, res_m, res_v};
}

std::vector<torch::Tensor> npu_sgd_step(torch::Tensor param, torch::Tensor grad,
                                        torch::Tensor momentum_buffer, double lr,
                                        double momentum, double weight_decay,
                                        double dampening, bool nesterov, bool has_momentum_buffer) {
    torch::Tensor effective_grad = grad;
    if (weight_decay != 0.0) {
        effective_grad = npu_add(grad, param * weight_decay);
    }
    torch::Tensor new_buf;
    torch::Tensor update_dir;
    if (momentum != 0.0) {
        if (!has_momentum_buffer) {
            new_buf = effective_grad.clone();
        } else {
            new_buf = npu_add(momentum_buffer * momentum, effective_grad * (1.0 - dampening));
        }
        if (nesterov) {
            update_dir = npu_add(effective_grad, new_buf * momentum);
        } else {
            update_dir = new_buf;
        }
    } else {
        update_dir = effective_grad;
    }
    torch::Tensor new_param = npu_sub(param, update_dir * lr);
    return {new_param, new_buf};
}
