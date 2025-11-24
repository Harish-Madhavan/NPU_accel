import torch
import unittest
import intel_npu_acceleration

class TestIntelNPULib(unittest.TestCase):
    def test_availability(self):
        # In our stub, this should return True if extension is loaded, or False if not
        # We just ensure it doesn't crash
        available = intel_npu_acceleration.is_available()
        print(f"NPU Available: {available}")
        self.assertIsInstance(available, bool)

    def test_add_op(self):
        a = torch.randn(5)
        b = torch.randn(5)
        
        # Run through our library
        result = intel_npu_acceleration.add(a, b)
        
        # Run standard pytorch
        expected = a + b
        
        # Check closeness
        self.assertTrue(torch.allclose(result, expected), "NPU add result should match PyTorch add")

if __name__ == '__main__':
    unittest.main()
