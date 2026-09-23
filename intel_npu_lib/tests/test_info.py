from unittest.mock import patch

import pytest

import intel_npu_acceleration as npu
from intel_npu_acceleration.info import get_system_info, print_info


class TestSystemInfoCLI:
    def test_get_system_info_structure(self):
        info = get_system_info()
        assert isinstance(info, dict)
        for key in [
            "os",
            "python_version",
            "pytorch_version",
            "openvino_available",
            "openvino_version",
            "npu_available",
            "cache_dir",
            "cache_size_mb",
            "cache_file_count",
            "in_memory_graph_cache_count",
            "torch_npu_registered",
            "torch_backends_npu_registered",
            "dynamo_npu_registered",
            "accelerator_available",
        ]:
            assert key in info

    def test_print_info_stdout(self, capsys):
        print_info()
        output_str = capsys.readouterr().out
        assert "INTEL NPU ACCELERATION LIBRARY - DIAGNOSTIC REPORT" in output_str
        assert "Python Version" in output_str
        assert "PyTorch Version" in output_str

    def test_cli_version_flag(self, capsys):
        from intel_npu_acceleration.__main__ import main

        with patch("sys.argv", ["intel_npu_acceleration", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        assert "intel_npu_acceleration" in capsys.readouterr().out

    def test_cli_clear_cache_flag(self, capsys):
        from intel_npu_acceleration.__main__ import main

        with patch("sys.argv", ["intel_npu_acceleration", "--clear-cache"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        assert "Cleared cache directory" in capsys.readouterr().out

    def test_cli_info_flag(self, capsys):
        from intel_npu_acceleration.__main__ import main

        with patch("sys.argv", ["intel_npu_acceleration", "--info"]):
            main()
        assert "INTEL NPU ACCELERATION LIBRARY - DIAGNOSTIC REPORT" in capsys.readouterr().out
