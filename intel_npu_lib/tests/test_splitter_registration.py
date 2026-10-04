"""Splitter + backend-registration regressions (no NPU crash risk)."""

import pytest
import torch
import torch.nn as nn


class TestRecursiveSplitter:
    def test_split_and_retry_recovers(self, monkeypatch):
        """A failing segment is bisected; compilable halves land on NPU."""
        import intel_npu_acceleration.frontend.partitioner as P

        monkeypatch.setattr(P, "_MIN_SPLIT_NODES", 2)
        monkeypatch.setattr(P, "_MAX_SPLIT_DEPTH", 2)

        class FlatModel(nn.Module):
            def forward(self, x, y):
                a = torch.add(x, y)
                b = torch.mul(a, x)
                c = torch.relu(b)
                d = torch.matmul(c, y)
                e = torch.neg(d)
                f = torch.nn.functional.silu(e)
                g = torch.softmax(f, dim=-1)
                return torch.transpose(g, 0, 1)

        model = FlatModel().eval()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        traced = torch.fx.symbolic_trace(model)

        calls = []

        from intel_npu_acceleration.frontend import compiler as C

        real_ctn = C.compile_to_npu

        def flaky_compile(child, inputs, *args, **kwargs):
            n = sum(1 for _ in child.graph.nodes)
            calls.append(n)
            if n > 6:
                raise RuntimeError("simulated NPU failure")
            return real_ctn(child, inputs, *args, **kwargs)

        monkeypatch.setattr(
            "intel_npu_acceleration.frontend.compiler.compile_to_npu", flaky_compile
        )

        parent = torch.fx.GraphModule(model, traced.graph)
        # wrap as split parent with single child
        from torch.fx.passes.split_module import split_module

        sp = split_module(parent, model, lambda n: 0)
        name, child = next(iter(sp.named_children()))
        inputs = (x, y)
        h = child.register_forward_pre_hook(lambda m, a: None)
        with torch.no_grad():
            try:
                sp(x, y)
            except Exception:
                pass
        h.remove()
        ctx = {
            "performance_hint": "LATENCY",
            "num_streams": 1,
            "dynamic_buckets": False,
            "bucket_sizes": None,
            "dynamic_dim": 1,
            "clone_outputs": True,
            "preprocess_config": None,
        }

        P._compile_segment(sp, name, child, {name: inputs}, {name: child(x, y)}, ctx, depth=0)
        # big child must have been replaced by split halves
        assert calls and max(calls) > 6  # the failing whole was attempted
        assert min(calls) <= 6  # halves were attempted
        out = sp(x, y)
        assert_close = torch.allclose(out, model(x, y), atol=1e-2, rtol=1e-2)
        assert assert_close

    def test_small_failure_falls_back_quietly(self, monkeypatch, caplog):
        """Segments at/below the node floor fall back to CPU without splitting."""
        import logging

        import intel_npu_acceleration.frontend.partitioner as P

        monkeypatch.setattr(P, "_MIN_SPLIT_NODES", 1000)

        model = nn.Sequential(nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        traced = torch.fx.symbolic_trace(model)
        parent = torch.fx.GraphModule(model, traced.graph)
        from torch.fx.passes.split_module import split_module

        sp = split_module(parent, model, lambda n: 0)
        name, child = next(iter(sp.named_children()))

        def always_fail(*a, **k):
            raise RuntimeError("simulated NPU failure")

        monkeypatch.setattr(
            "intel_npu_acceleration.frontend.compiler.compile_to_npu", always_fail
        )
        ctx = {
            "performance_hint": "LATENCY",
            "num_streams": 1,
            "dynamic_buckets": False,
            "bucket_sizes": None,
            "dynamic_dim": 1,
            "clone_outputs": True,
            "preprocess_config": None,
        }
        with caplog.at_level(logging.WARNING, logger="intel_npu_acceleration.frontend.partitioner"):
            P._compile_segment(sp, name, child, {name: (x,)}, {}, ctx, depth=0)
        assert "Falling back to CPU" in caplog.text
        # child untouched (still original eager module)
        assert torch.allclose(sp(x), model(x), atol=1e-5, rtol=1e-5)


class TestBackendRegistration:
    def test_no_duplicate_registration_without_prior_import(self):
        """torch.compile(backend='npu') must work in a fresh interpreter."""
        import subprocess
        import sys

        code = (
            "import torch;"
            "m = torch.nn.Linear(4, 4);"
            "c = torch.compile(m, backend='npu');"
            "print('entry-point compile OK', tuple(c(torch.randn(2, 4)).shape))"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=300)
        out = (r.stdout or b"").decode(errors="replace")
        assert "entry-point compile OK" in out, f"rc={r.returncode} out={out[-500:]}"
