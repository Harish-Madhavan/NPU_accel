import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest
import torch
import torch.nn as nn

from intel_npu_acceleration.frontend.transformations import (
    apply_pre_post_processing,
    clean_node_name,
    configure_input_ppp,
    configure_output_ppp,
    fold_scalar_parameter_inputs,
    optimize_ov_model,
    reshape_model_inputs_to_static,
    transform_quantized_weights,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("l_x_1", "x"),
        ("L__tokens_0", "tokens"),
        ("arg_0.1", "arg"),
        ("my_input", "my_input"),
    ],
)
def test_clean_node_name(raw, expected):
    assert clean_node_name(raw) == expected


class TestTransformations:
    def test_fold_scalar_parameter_inputs(self):
        # Create an OV model with 1 scalar int parameter and 1 tensor float parameter
        p_scalar = ops.parameter(ov.Shape([]), ov.Type.i64)
        p_scalar.set_friendly_name("start_pos")
        p_tensor = ops.parameter(ov.Shape([2, 4]), ov.Type.f32)
        p_tensor.set_friendly_name("x")

        # Simple computation: multiply tensor by scalar cast to float
        scalar_f32 = ops.convert(p_scalar, destination_type=ov.Type.f32)
        out = ops.multiply(p_tensor, scalar_f32)
        ov_model = ov.Model([out], [p_scalar, p_tensor], "ScalarFoldTest")

        example_inputs = (5, torch.randn(2, 4))
        placeholder_names = ["start_pos", "x"]

        folded_model = fold_scalar_parameter_inputs(ov_model, example_inputs, placeholder_names)

        # After folding, only 1 parameter (the tensor 'x') should remain!
        assert len(folded_model.get_parameters()) == 1
        assert folded_model.get_parameters()[0].get_friendly_name() == "x"

    def test_reshape_model_inputs_to_static(self):
        p_dynamic = ops.parameter(ov.PartialShape([-1, 16]), ov.Type.f32)
        p_dynamic.set_friendly_name("x")
        ov_model = ov.Model([p_dynamic], [p_dynamic], "DynamicShapeTest")

        example_inputs = (torch.randn(4, 16),)
        placeholder_names = ["x"]

        reshape_model_inputs_to_static(ov_model, example_inputs, placeholder_names)
        assert list(ov_model.inputs[0].get_shape()) == [4, 16]

    def test_ppp_configuration_helpers(self):
        param = ops.parameter(ov.Shape([1, 3, 64, 64]), ov.Type.f32, "input")
        ov_model = ov.Model([param], [param], "PPPTest")

        from openvino.preprocess import PrePostProcessor
        ppp = PrePostProcessor(ov_model)

        cfg_in = {
            "shape": (1, 64, 64, 3),
            "layout": "NHWC",
            "element_type": "u8",
            "model_layout": "NCHW",
            "color_format": "BGR",
            "model_color_format": "RGB",
            "mean": [128.0, 128.0, 128.0],
            "scale": [255.0, 255.0, 255.0],
        }
        configure_input_ppp(ppp.input(0), cfg_in)
        configure_output_ppp(ppp.output(0), {"element_type": "f32"})

        processed_model = ppp.build()
        assert list(processed_model.inputs[0].get_shape()) == [1, 64, 64, 3]
        assert processed_model.inputs[0].get_element_type() == ov.Type.u8

    def test_transform_quantized_weights_int8(self):
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 64, bias=False)

            def forward(self, x):
                return self.fc(x)

        m = LinearModel().eval()
        import intel_npu_acceleration as npu
        from intel_npu_acceleration.frontend.compiler import (
            _convert_to_ov_model,
            _dequantized_linears,
            _trace_model,
        )

        npu.nn.quantize(m, mode="int8")

        x = torch.randn(1, 32)
        traced = _trace_model(m)
        with _dequantized_linears(traced):
            ov_model = _convert_to_ov_model(m, (x,))

        transformed = transform_quantized_weights(ov_model, m)

        ops_types = [op.get_type_name() for op in transformed.get_ops()]
        assert "Convert" in ops_types
        assert "Multiply" in ops_types

        # Verify an int8 Constant exists
        int8_consts = [
            op for op in transformed.get_ops()
            if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type.i8
        ]
        assert len(int8_consts) >= 1

    def test_transform_quantized_weights_int4(self):
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 64, bias=False)

            def forward(self, x):
                return self.fc(x)

        m = LinearModel().eval()
        import intel_npu_acceleration as npu
        from intel_npu_acceleration.frontend.compiler import (
            _convert_to_ov_model,
            _dequantized_linears,
            _trace_model,
        )

        npu.nn.quantize(m, mode="int4")

        x = torch.randn(1, 32)
        traced = _trace_model(m)
        with _dequantized_linears(traced):
            ov_model = _convert_to_ov_model(m, (x,))

        transformed = transform_quantized_weights(ov_model, m)

        # Verify a uint4 Constant exists
        u4_consts = [
            op for op in transformed.get_ops()
            if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type.u4
        ]
        assert len(u4_consts) >= 1

    def test_transform_quantized_weights_int4_sym(self):
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 64, bias=False)

            def forward(self, x):
                return self.fc(x)

        m = LinearModel().eval()
        import intel_npu_acceleration as npu
        from intel_npu_acceleration.frontend.compiler import _trace_and_convert_to_ov

        npu.nn.quantize(m, mode="int4_sym")
        x = torch.randn(1, 32)
        ov_model, _, _, _ = _trace_and_convert_to_ov(m, (x,), precision="fp16")

        i4_consts = [
            op for op in ov_model.get_ops()
            if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type.i4
        ]
        assert len(i4_consts) >= 1

    def test_trace_and_convert_preserves_int8_qdq_after_optimization(self):
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 64, bias=False)

            def forward(self, x):
                return self.fc(x)

        m = LinearModel().eval()
        import intel_npu_acceleration as npu
        from intel_npu_acceleration.frontend.compiler import _trace_and_convert_to_ov

        npu.nn.quantize(m, mode="int8")
        x = torch.randn(1, 32)
        ov_model, _, _, _ = _trace_and_convert_to_ov(m, (x,), precision="fp16")

        i8_consts = [
            op for op in ov_model.get_ops()
            if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type.i8
        ]
        assert len(i8_consts) >= 1


