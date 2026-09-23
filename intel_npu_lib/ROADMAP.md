# 🗺️ Roadmap: Intel NPU Acceleration Library → Proper PyTorch Backend

**Owner:** Intel NPU Acceleration maintainers · **Current version:** `0.2.0` · **Last updated:** 2026-09-21

This document is the single source of truth for where the backend stands and what
"proper PyTorch backend" means for this project. It replaces all prior roadmap
claims with status grounded in the current tree (`223` pytest tests, full
`torch.npu` / `torch.backends.npu` / `torch.accelerator` / TorchDynamo surface).

**Status legend:** ✅ done · 🚧 in progress · 📋 planned · 🔬 research spike first

---

## 1. Where we are (v0.2.0 — audited 2026-09-21)

### 1.1 Backend surface inventory

| PyTorch API surface | Status | Notes |
| :--- | :---: | :--- |
| `torch.device("npu")` / `"npu:0"` (PrivateUse1 rename) | ✅ | `backends/npu_backend.py::_register_privateuse1_backend` |
| `torch.npu.*` (device mgmt, streams/events, RNG, memory stubs, AMP queries) | ✅ | 14 conformance tests in `tests/test_pytorch_backend.py` |
| `torch.npu.amp.autocast` / `GradScaler` / `custom_fwd` / `custom_bwd` | ✅ | `custom_*` bound with `device_type="npu"` (torch≥2.4) |
| `torch.backends.npu.*` (availability, `is_built`, version, matmul/SDP flags) | ✅ | `version()` reads installed distribution metadata |
| `torch.accelerator.*` bridge | ✅ | Patched `current_stream`, `synchronize`, `device_count`, memory fns |
| `torch.compile(model, backend="npu" / "intel_npu")` | ✅ | Entry points + `frontend/dynamo.py`; modes/smoke covered |
| `torch.Tensor.to("npu")`, `.npu()`, `.is_npu`, factories with `device="npu"` | ✅ | Zero-copy CPU storage + NPU tagging |
| `nn.Module.to("npu")` → dynamo compile; `.to_npu()` / `.compile_npu()` helpers | ✅ | |
| Eager functional ops (`_functional/`) with CPU fallback | ✅ | ~40 ops; proxy/empty-tensor guards |
| Autograd backward on NPU (matmul, linear, conv2d, norms, activations, losses, SDPA, embedding) | ✅ | All Level Zero graphs: transposed-conv + im2col grads, SDPA recompute chain rule, one-hot embedding scatter; `tests/test_npu_backward_kernels.py` pins kernels directly |
| Fused optimizers `NPUAdam` / `NPUSGD` + `clip_grad_norm_` (= torch impl) | ✅ | |
| INT8 per-channel + packed INT4 `quantize()` / `quantized_linear` | ✅ | Eager + compiled paths tested |
| Stateful KV-cache (`NPUStatefulKVCache`, ReadValue/Assign registers) | ✅ | Single + multi-layer auto-mapping tested |
| Async infer (`infer_async`/`wait_async`/`submit`/`batch_infer`), multi-stream | ✅ | Double-buffered output leases |
| PPP offload (layout/dtype/normalize baked into graph) | ✅ | Static configs; dynamic crop/resize/pad missing (see Phase 3) |
| Disk + in-memory graph cache (`INTEL_NPU_CACHE_DIR` honored) | ✅ | LRU prune, adaptive dropped-key ladder for old drivers |

### 1.2 Quality gates (all green on NPU hardware)

- **235 pytest tests**, plain pytest style (`unittest` fully removed), parametrized shape/dtype/op matrices, `slow`/`npu` markers — `cd intel_npu_lib && pytest`.
- **Ruff lint clean** (`ruff check intel_npu_lib` from repo root); import sorting enforced.
- **CI (GitHub Actions):** Windows-only, Python 3.12, CPU torch + OpenVINO from PyPI, C++ extension build, lint, full suite with coverage, wheel artifact. No-driver runners exercise the CPU-fallback path (hardware-only tests skip via availability probes).
- **Packaging:** `pyproject.toml` is the single source of truth; `setup.py` builds only the `_C` extension; `NPU_NO_BUILD_EXT=1` yields a pure-Python wheel; `intel-npu-info` console script.

### 1.3 Known gaps (the rest of this document)

1. No PyPI release; no prebuilt wheels; Linux CI missing; Python matrix is 3.12-only in CI.
2. No per-commit performance tracking (examples benchmark, nothing gates regressions).
3. No API docs site; examples don't share `_common.py` helpers yet.
4. Backward kernels outstanding on CPU: `conv2d`, SDPA recompute, embedding (`aten` op).
5. No NNCF PTQ flow; no dynamic PPP; no distributed/oneCCL story; no profiler activity integration.

---

## 2. Definition of done — "proper backend"

The project is a *proper backend* when **all** of these hold on a clean contributor machine:

1. `pip install intel-npu-acceleration` (no repo clone) gives a working accelerated install on Windows **and** Linux, Python 3.10–3.13, with and without NPU hardware (graceful CPU fallback, truthful `is_available()`).
2. The full test suite passes on: NPU hardware, hardware-less CI, and from an installed wheel (not just editable checkout).
3. A backend-conformance matrix (op × dtype × layout × {eager, graph, autograd}) is green, including `float16`/`bfloat16`, non-contiguous inputs, and dynamic shapes.
4. Performance SLOs exist and are enforced per commit (compile-time budget, single-stream latency, throughput scaling, tokens/s on reference SLMs).
5. Versioned releases with changelog; `torch.backends.npu.version()` always equals the release.
6. Docs site + troubleshooting matrix sufficient that a PyTorch-only user (zero NPU knowledge) can install, run, and debug — matching the project's core principle.

---

## 3. Phases

### 🟢 Phase 1 — Backend conformance lockdown
*Goal: prove the shim correct, not just present. Highest value per effort.*

- [x] `torch.npu` / `torch.backends.npu` / `torch.accelerator` parity modules with conformance tests.
- [x] `amp.custom_fwd` / `custom_bwd`; metadata-driven `backends.npu.version()`; `is_built()`.
- [ ] **Op conformance matrix**: extend `tests/test_pytorch_conformance.py` into an explicit op×dtype×layout table for every eager op: dtypes `{float32, float16, bfloat16}`, contiguous + channels-last + non-contiguous strided inputs, scalar/0-dim edge cases. *Exit: matrix green on NPU; gaps filed as Phase 4/5 items, not silent.*
- [ ] **Error-path tests**: unknown device index, bad `performance_hint`, `strict=True` on unsupported graphs, missing driver — assert exception *types* (`NPUCompilationError`, `NPUDeviceError`, …), not just "raises".
- [ ] **`torch.func` smoke tests**: `torch.func.grad` / `vmap` over an NPU-compiled function where PrivateUse1 supports it; document what is unsupported.
- [ ] **Serialization**: `torch.save`/`load_state_dict` round-trip for compiled modules and `quantize()`d models; document what is/isn't portable across machines (cache dir must never be assumed portable).
- [ ] **Determinism note**: document which paths are bit-stable across runs (graph cache hits, streams) and add a determinism test (same seed → same outputs, N times).

*Exit criteria:* §2 items 2 (hardware + CI legs) hold; conformance matrix file exists and is green except explicitly triaged gaps. *Metric:* test count grows by parametrization, not by copy-paste (helpers/conftest reuse enforced in review).

### 🟢 Phase 2 — Packaging & release engineering
*Goal: `pip install` works for strangers. Unblocks adoption and Linux users.*

- [x] Single-source `pyproject.toml`; slim `setup.py`; `NPU_NO_BUILD_EXT` fallback; `intel-npu-info` script; rewritten install docs + troubleshooting.
- [ ] **PyPI release `0.3.0`**: project metadata (authors, license file, URLs), README rendering check (`twine check`), sdist completeness (`python -m build` from a clean sdist, extension included).
- [ ] **Prebuilt wheels**: `cibuildwheel` (or hand-rolled `build` matrix) for Windows + manylinux, Python 3.10–3.13, CPU-torch linkage. Pure-Python fallback wheel when no compiler is present.
- [ ] **Version automation**: version from git tags (`setuptools-scm` or a `VERSION` file + tag check in CI); `__version__`, `backends.npu.version()`, and wheel metadata can never diverge (add a test asserting all three agree).
- [ ] **CI matrix expansion**: `{windows-latest, ubuntu-22.04}` × `{3.10, 3.12, 3.13}` (3.11 optional); nightly scheduled run; coverage gate (fail under threshold) instead of report-only.
- [ ] **`CHANGELOG.md`**: Keep-a-Changelog format, updated per release; backfill 0.1.0→0.2.0 from git history.

*Exit criteria:* §2 item 1 holds; release checklist is a script, not tribal knowledge. *Risk:* NPU driver availability on CI runners — hardware legs stay on self-hosted/label-gated runners; public CI asserts the fallback contract.

### 🟡 Phase 3 — Performance engineering & regression gating
*Goal: make "fast" measurable and defended. Today's examples benchmark; nothing gates.*

- [x] Example benchmarks: matmul vs vanilla OpenVINO, async pipelining, model profiler, TinyLlama decode.
- [ ] **`benchmarks/` harness**: one command (`pytest benchmarks/ --benchmark-only` or a script) emitting machine-readable JSON (latency p50/p95, throughput, tokens/s, compile time, peak cache bytes) for a fixed model set (MLP, ResNet block, encoder block, TinyLlama-1.1B decode).
- [ ] **Per-commit tracking**: store results as CI artifacts + a rolling CSV/JSON in-repo (or GitHub Pages chart); fail CI on >10% regression vs rolling median on reference kernels.
- [ ] **Compile-time budget**: assert `compile_to_npu` wall time for reference models stays under budget on cache-miss and near-zero on cache-hit (extends `test_fx_tracing_caching`).
- [ ] **Quantization perf validation**: INT8/INT4 vs FP16 latency + accuracy tables on reference models (prove the claimed DP4A/VNNI win or remove the claim).
- [ ] **NNCF PTQ flow (research first)**: spike OpenVINO NNCF post-training quantization → `quantize(model, mode="int8")` parity; only productize if accuracy deltas are documented per model.
- [ ] **Streams autotune**: heuristic or micro-benchmark choosing `num_streams`/`performance_hint` per model shape instead of defaulting to LATENCY/1; log the choice via the existing `[NPU Intelligence]` channel.
- [ ] **Dynamic PPP**: crop/resize/pad inside the hardware PPP pipeline (currently static configs only).

*Exit criteria:* a regression that costs 15% latency cannot land silently. *Metric:* benchmark variance characterized (3+ runs, pinned runner labels) before any gate is enforced.

### 🟡 Phase 4 — Model coverage (prove it on real models)
*Goal: graduate from blocks to end-to-end models users actually run.*

- [x] TinyLlama decode example; encoder/CV workloads; Llama attention+FFN unit coverage.
- [ ] **Full SLM decode e2e**: Llama-3.2-1B (or TinyLlama-1.1B) and Qwen2.5-0.5B: prefill + stateful autoregressive decode to N tokens, tokens/s + accuracy-vs-CPU parity test (small, deterministic, CPU-fallback-runnable for shape; NPU-gated for perf).
- [ ] **SLM readiness report, automated**: turn `slm_readiness_report.md` into a generated artifact (script that probes op coverage for a target architecture and lists missing kernels) instead of a hand-written note.
- [ ] **Vision transformer e2e**: ViT-Tiny/Small classification parity + latency (exercises reshape/permute/softmax/linear at realistic sizes).
- [ ] **Dynamic shapes GA**: bucketing policies documented + tested for batch and sequence dims; failure mode (recompile storm) has a logged warning and a test.
- [ ] **Long-context KV**: stateful cache at 4k+ sequence lengths; memory-growth characterization; eviction/sliding-window policy decision (documented even if the answer is "not supported, error clearly").
- [ ] **Diffusion UNet smoke** (only if Phase 3 shows headroom): single denoising step parity; deprioritize if LLM/ViT work overflows.

*Exit criteria:* two SLMs + one ViT run e2e in CI (accuracy leg; perf leg NPU-gated). *Risk:* model weights licensing/hosting — use gated Hugging Face ids with `HF_TOKEN` secret or vendored tiny random-init configs for CI.

### 🔴 Phase 5 — Training maturity
*Goal: on-device fine-tuning users can trust, with honest boundaries.*

- [x] NPU backward for matmul/linear/norms/activations/losses; `NPUAdam`/`NPUSGD`; end-to-end training-loop tests; `clip_grad_norm_` == torch.
- [x] **Backward kernels on NPU**: `conv2d` (transposed-conv `grad_input`,
  im2col Gather+MatMul `grad_weight`, ReduceSum `grad_bias`), SDPA (Level Zero
  recompute chain rule), embedding (one-hot-matmul scatter) are all native
  graphs — no CPU recompute/`aten` in the hot path. CPU fallbacks remain only
  for out-of-scope configs (sparse/padded embedding variants).
- [x] **Direct kernel tests**: `tests/test_npu_backward_kernels.py` calls the
  native `_C` kernels with hardware gating (`skip_if_no_npu`), so a broken
  kernel cannot hide behind the CPU fallback.
- [ ] **AMP training workflows**: GradScaler + autocast end-to-end training test (loss scaling on NPU, inf/nan skip behavior matches torch).
- [ ] **LoRA/Q LoRA fine-tune demo**: small SLM adapter training script with loss-curve parity vs CPU; documents memory ceiling for client NPUs.
- [ ] **Gradient accumulation + clipping recipes**: tested example (multi-micro-batch, `clip_grad_norm_` semantics identical — already guaranteed by delegation).
- [ ] **Determinism & precision guide**: which dtypes/ops are bit-stable for training; fp16 accumulation caveats; per-op tolerance table linked from `SUPPORTED_OPS.md`.

*Exit criteria:* a user can fine-tune a LoRA adapter on NPU following only repo docs, with documented parity bounds. *Non-goal:* pre-training at scale (datacenter problem, not AI-PC).

### 🟢 Phase 6 — Docs & contributor experience
*Goal: zero-tribal-knowledge repo. Required for external contributors.*

- [x] README install/troubleshooting rewrite; `docs/getting_started.md`; `SUPPORTED_OPS.md` backend matrix.
- [ ] **API docs site** (MkDocs + mkdocstrings, deployed via GitHub Pages from docstrings — no separate hand-written API pages to rot).
- [ ] **Examples gallery pass**: all examples import shared `_common.py` helpers (seed/timer/bench), take `--device/--iters` flags, and assert (not just print) parity where they claim it.
- [ ] **Troubleshooting matrix**: driver versions × OpenVINO versions × symptoms (unknown-option warnings, `cl.exe` missing, huge CUDA torch download, cache growth) with copy-paste fixes; link from README and from `intel-npu-info` output.
- [ ] **CONTRIBUTING.md**: env setup, build/test/lint commands, "add an operator" runbook (exists as a skill — promote the stable parts), PR checklist (tests + ruff + docs touched), review SLA.
- [ ] **Issue/PR templates**: bug report (auto-attach `intel-npu-info` output), operator request (op signature + shapes + dtypes), benchmark regression.

*Exit criteria:* a new contributor can land a tested operator change following only in-repo docs. *Metric:* docs build is a CI job; broken links fail the build.

### 🔬 Phase 7 — Ecosystem & upstream (research-gated)
*Goal: stop carrying patches others should own. Nothing here starts without a spike note.*

- [ ] **Distributed/oneCCL smoke**: single-process multi-device and (later) multi-process `ccl`-backend allreduce smoke test on multi-NPU hardware; until hardware exists, keep as documented non-goal.
- [ ] **Profiler integration**: `torch.profiler` NPU activity support (or documented Kineto gap + OpenVINO profiling bridge via existing `get_profiling_info`).
- [ ] **Upstream PrivateUse1 gaps**: enumerate torch-side issues hit by this backend (accelerator patching, `_get_device_index`, factory coverage) and file/track them against pytorch/pytorch with minimal repros.
- [ ] **Driver compat matrix automation**: nightly probe of `available_devices` + dropped-key ladder across driver/OpenVINO versions; publish the matrix; pin minimums in `pyproject.toml` and install docs.
- [ ] **Linux parity GA**: full suite + benchmarks on Ubuntu 22.04/24.04 with Intel NPU driver; Windows/Linux behavioral diffs filed as bugs until closed or documented.

---

## 4. Release scheme

- **Versioning:** SemVer. `0.x` while the §2 checklist is incomplete; `1.0.0` when §2 holds end-to-end.
- **Cadence:** minor release per completed phase milestone; patch releases for driver-compat and bug fixes.
- ** Changelog:** `CHANGELOG.md` (Keep-a-Changelog); every PR that changes behavior adds an entry (CI lint checks this once Phase 2 automation lands).
- **Next release (`0.3.0`) proposed scope:** Phase 2 packaging (PyPI + wheels + version automation) + Phase 1 conformance matrix. No new ops required.

## 5. Non-goals (explicit)

- Pre-training LLMs on NPU; datacenter/Uscale hardware targets.
- Custom user-facing APIs outside standard PyTorch (see `AGENTS.md` core principles — this constraint is permanent).
- Supporting EOL Python (<3.10), EOL torch (<2.1), or OpenVINO <2024.
- Vendoring model weights in this repo.

## 6. How to propose roadmap changes

Open a PR against this file: one phase per PR, each new item needs *why*, *exit criteria*, and *metric*. Status fields (`✅/🚧/📋`) are updated by the PR that lands the work — never in advance.
