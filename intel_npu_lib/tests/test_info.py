import io
import sys
import unittest

import intel_npu_acceleration as npu
from intel_npu_acceleration.info import get_system_info, print_info


class TestSystemInfoCLI(unittest.TestCase):
    def test_get_system_info_structure(self):
        info = get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn("os", info)
        self.assertIn("python_version", info)
        self.assertIn("pytorch_version", info)
        self.assertIn("openvino_available", info)
        self.assertIn("openvino_version", info)
        self.assertIn("npu_available", info)
        self.assertIn("cache_dir", info)
        self.assertIn("cache_size_mb", info)
        self.assertIn("cache_file_count", info)
        self.assertIn("in_memory_graph_cache_count", info)
        self.assertIn("torch_npu_registered", info)
        self.assertIn("torch_backends_npu_registered", info)
        self.assertIn("dynamo_npu_registered", info)
        self.assertIn("accelerator_available", info)

    def test_print_info_stdout(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            print_info()
        finally:
            sys.stdout = sys.__stdout__

        output_str = captured_output.getvalue()
        self.assertIn("INTEL NPU ACCELERATION LIBRARY - DIAGNOSTIC REPORT", output_str)
        self.assertIn("Python Version", output_str)
        self.assertIn("PyTorch Version", output_str)

    def test_cli_version_flag(self):
        from unittest.mock import patch

        from intel_npu_acceleration.__main__ import main

        captured_output = io.StringIO()
        with patch("sys.argv", ["intel_npu_acceleration", "--version"]):
            with patch("sys.stdout", captured_output):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 0)
        self.assertIn("intel_npu_acceleration", captured_output.getvalue())

    def test_cli_clear_cache_flag(self):
        from unittest.mock import patch

        from intel_npu_acceleration.__main__ import main

        captured_output = io.StringIO()
        with patch("sys.argv", ["intel_npu_acceleration", "--clear-cache"]):
            with patch("sys.stdout", captured_output):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 0)
        self.assertIn("Cleared cache directory", captured_output.getvalue())

    def test_cli_info_flag(self):
        from unittest.mock import patch

        from intel_npu_acceleration.__main__ import main

        captured_output = io.StringIO()
        with patch("sys.argv", ["intel_npu_acceleration", "--info"]):
            with patch("sys.stdout", captured_output):
                main()
        self.assertIn("INTEL NPU ACCELERATION LIBRARY - DIAGNOSTIC REPORT", captured_output.getvalue())


if __name__ == "__main__":
    unittest.main()
