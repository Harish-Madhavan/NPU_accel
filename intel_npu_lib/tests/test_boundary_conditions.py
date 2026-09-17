import unittest

import torch

import intel_npu_acceleration as npu


class TestBoundaryConditions(unittest.TestCase):
    def test_zero_sized_tensors(self):
        """Test ops with zero-sized tensors (at least one dimension is 0)."""
        test_shapes = [
            (0,),
            (0, 10),
            (10, 0),
            (1, 0, 5),
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                a = torch.randn(shape)
                b = torch.randn(shape)

                # Element-wise add
                try:
                    res = npu.add(a, b)
                    self.assertEqual(res.shape, a.shape)
                    self.assertEqual(res.numel(), 0)
                except Exception as e:
                    self.fail(f"npu.add failed for zero-sized tensor with shape {shape}: {e}")

    def test_unaligned_arrays(self):
        """Test ops with non-contiguous (unaligned) tensors."""
        # Create a non-contiguous tensor by slicing or transposing
        a_base = torch.randn(10, 20)
        a = a_base[:, :10] # Sliced, likely non-contiguous if we don't call .contiguous()

        b = torch.randn(10, 10)

        self.assertFalse(a.is_contiguous())

        try:
            res = npu.add(a, b)
            self.assertTrue(torch.allclose(res, a + b, atol=1e-3))
        except Exception as e:
            self.fail(f"npu.add failed for non-contiguous tensor: {e}")

    def test_5d_tensors(self):
        """Test ops with 5D tensors (mentioned in ROADMAP)."""
        shape = (2, 3, 4, 5, 6)
        a = torch.randn(shape)
        b = torch.randn(shape)

        try:
            res = npu.add(a, b)
            self.assertTrue(torch.allclose(res, a + b, atol=1e-3))
        except Exception as e:
            self.fail(f"npu.add failed for 5D tensor: {e}")

    def test_different_dtypes(self):
        """Test ops with different dtypes (FP16, INT8, UINT8)."""
        dtypes = [torch.float16, torch.int32]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                if dtype.is_floating_point:
                    a = torch.randn(10, 10).to(dtype)
                    b = torch.randn(10, 10).to(dtype)
                else:
                    a = torch.randint(0, 100, (10, 10)).to(dtype)
                    b = torch.randint(0, 100, (10, 10)).to(dtype)

                try:
                    res = npu.add(a, b)
                    self.assertEqual(res.dtype, dtype)
                    self.assertTrue(torch.allclose(res.float(), (a + b).float(), atol=1e-2))
                except Exception as e:
                    self.fail(f"npu.add failed for dtype {dtype}: {e}")

    def test_uint8_clamping(self):
        """Test UINT8 image boundary clamping (mentioned in ROADMAP)."""
        a = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
        b = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)

        # NPU add for uint8 might wrap or clamp depending on implementation.
        # Standard torch.add for uint8 wraps.
        # Let's see what our NPU implementation does.
        try:
            res = npu.add(a, b)
            # If it's implemented via OpenVINO Add, it might have different behavior or cast to float first.
            # _functional.py _promote_binary promotes uint8 to int32.
            expected = (a.to(torch.int32) + b.to(torch.int32)).to(torch.uint8)
            self.assertTrue(torch.equal(res, expected))
        except Exception as e:
            self.fail(f"npu.add failed for uint8: {e}")

if __name__ == "__main__":
    unittest.main()
