import unittest
import torch
import torch.nn as nn
from intel_npu_acceleration.frontend import compile_to_npu


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TestCVWorkload(unittest.TestCase):
    def test_resnet_block(self):
        in_channels = 64
        out_channels = 128
        model = ResNetBlock(in_channels, out_channels, stride=2)
        model.eval()

        x = torch.randn(1, in_channels, 32, 32)

        try:
            npu_model = compile_to_npu(model, x)
        except Exception as e:
            self.fail(f"Compilation failed: {e}")

        out_npu = npu_model(x)
        out_cpu = model(x)

        # Comparison
        # CV models can have larger discrepancies due to accumulation
        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
