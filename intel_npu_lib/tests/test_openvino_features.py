import unittest
import os
import tempfile
import torch
import torch.nn as nn
import openvino as ov
import intel_npu_acceleration as npu
from intel_npu_acceleration.frontend import export_openvino_ir


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TestOpenVINOFeatures(unittest.TestCase):
    def test_export_openvino_ir(self):
        model = SimpleMLP().eval()
        x = torch.randn(2, 16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = os.path.join(tmp_dir, "simple_mlp.xml")
            bin_path = os.path.join(tmp_dir, "simple_mlp.bin")

            exported = export_openvino_ir(model, x, xml_path, bin_path)
            self.assertEqual(exported, xml_path)
            self.assertTrue(os.path.exists(xml_path))
            self.assertTrue(os.path.exists(bin_path))

            # Verify that OpenVINO can load and inspect the exported IR
            core = ov.Core()
            ov_model = core.read_model(xml_path, bin_path)
            self.assertIsNotNone(ov_model)
            self.assertEqual(len(ov_model.inputs), 1)
            self.assertEqual(len(ov_model.outputs), 1)

    def test_profiling_info_accessible(self):
        model = SimpleMLP().eval()
        x = torch.randn(2, 16)

        compiled_model = npu.compile(model, x)
        _ = compiled_model(x)

        if hasattr(compiled_model, "get_profiling_info"):
            prof_info = compiled_model.get_profiling_info(0)
            self.assertIsInstance(prof_info, list)

    def test_export_openvino_ir_with_ppp(self):
        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = ConvNet().eval()
        x_traced = torch.randn(1, 3, 32, 32)
        preprocess_config = {
            "input": {
                "shape": (1, 32, 32, 3),
                "layout": "NHWC",
                "element_type": "u8",
                "model_layout": "NCHW",
                "mean": [123.675, 116.28, 103.53],
                "scale": [58.395, 57.12, 57.375],
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = os.path.join(tmp_dir, "convnet_ppp.xml")
            bin_path = os.path.join(tmp_dir, "convnet_ppp.bin")

            export_openvino_ir(model, x_traced, xml_path, bin_path, preprocess_config=preprocess_config)
            self.assertTrue(os.path.exists(xml_path))

            core = ov.Core()
            ov_model = core.read_model(xml_path, bin_path)
            # The input tensor on the exported model should have the NHWC shape [1, 32, 32, 3] and u8 type
            self.assertEqual(list(ov_model.inputs[0].get_shape()), [1, 32, 32, 3])
            self.assertEqual(ov_model.inputs[0].get_element_type(), ov.Type.u8)

    def test_export_openvino_ir_stateful(self):
        from intel_npu_acceleration import NPUStatefulKVCache

        class StatefulDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_cache = NPUStatefulKVCache(1, 128, 4, 16)
                self.proj = nn.Linear(16, 16)

            def forward(self, x):
                cached = self.kv_cache(x)
                return self.proj(cached)

        model = StatefulDecoder().eval()
        x = torch.randn(1, 1, 4, 16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = os.path.join(tmp_dir, "stateful_model.xml")
            bin_path = os.path.join(tmp_dir, "stateful_model.bin")

            export_openvino_ir(model, x, xml_path, bin_path, stateful=True)
            self.assertTrue(os.path.exists(xml_path))

            core = ov.Core()
            ov_model = core.read_model(xml_path, bin_path)
            # Verify that state variables exist in the model
            var_ids = [v.get_info().variable_id for v in ov_model.get_variables()]
            self.assertTrue(len(var_ids) > 0)

    def test_optimize_ov_model(self):
        import numpy as np
        from intel_npu_acceleration.frontend.transformations import optimize_ov_model
        import openvino.opset13 as ops

        # Construct a graph with constant addition that can be folded: const(2) + const(3)
        const1 = ops.constant(np.array([2.0], dtype=np.float32))
        const2 = ops.constant(np.array([3.0], dtype=np.float32))
        add_node = ops.add(const1, const2)
        param = ops.parameter(ov.Shape([1]), ov.Type.f32)
        mul_node = ops.multiply(param, add_node)
        ov_model = ov.Model([mul_node], [param], "ConstantFoldTest")

        optimized = optimize_ov_model(ov_model)
        self.assertIsNotNone(optimized)
        self.assertEqual(len(optimized.get_parameters()), 1)
        self.assertEqual(len(optimized.get_results()), 1)


if __name__ == "__main__":
    unittest.main()
