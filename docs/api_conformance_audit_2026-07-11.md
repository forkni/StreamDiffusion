# API Conformance Audit — CUDA Python / TensorRT / ONNX (2026-07-11)

> **Untracked working document — never stage or commit** (same convention as
> `docs/perf_bestpractices_audit_2026-07-10.md`).
>
> Audit of the codebase's usage of the CUDA Python, TensorRT, and ONNX-family APIs
> against official documentation and the *installed* package versions. Report only —
> no code was changed this phase. Branch `refactor/py-code-review-remediation`,
> HEAD `d5efd4d`.

---

## 1. Method and ground truth

Two evidence sources, used together:

1. **Online docs (browser, pulled this session).** NVIDIA's "latest" TensorRT docs
   are version **11.1.0** — one major version ahead of the installed 10.16. They are
   used as the authoritative statement of semantic contracts (which are stable across
   10.x→11.x) and as the *direction of travel* for deprecations/removals.
2. **Installed-package introspection (venv, authoritative for pinned versions).**
   Enum membership, method presence, signatures, and docstrings were introspected from
   the actual installed wheels (`venv/Scripts/python.exe`). Where 11.1 docs and 10.16
   bindings disagree, the introspection result is treated as ground truth for today's
   behavior and the 11.1 docs as ground truth for what an upgrade will break.

### Pinned versions → doc sources

| Package | Installed | Doc source used |
|---|---|---|
| tensorrt_cu12 | 10.16.1.11 | docs.nvidia.com TensorRT 11.1.0 (`/latest/_static/python-api/...`, `/latest/_static/c-api/...`) + venv introspection |
| cuda-python | 12.9.7 (dist 12.9.0) | venv introspection (docstrings/warnings); nvidia.github.io/cuda-python |
| onnx | 1.19.1 | venv introspection (`onnx.save_model` signature) |
| torch | 2.8.0+cu128 | venv introspection (`torch.onnx.export` signature + docstring) |
| polygraphy | 0.49.26 | venv introspection (`engine_from_bytes` signature) |
| nvidia-modelopt | 0.43.0 | venv introspection (`modelopt.onnx.quantization.quantize` signature) |
| onnxruntime-gpu | 1.24.4 | (used only indirectly by modelopt calibration — no direct call sites) |

### Key doc pages cited below

- **[TRT-CTX-PY]** `https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/ExecutionContext.html`
- **[TRT-CTX-CPP]** `https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_execution_context.html`
- **[TRT-BCFG]** `https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/BuilderConfig.html`
- **[TRT-ONNX]** `https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/parsers/Onnx/pyOnnx.html`
- **[TRT-REFIT]** `https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/Refitter.html`
- **[INTROSPECT]** venv introspection of installed packages, this session.

### Installed TRT 10.16.1.11 enum/API ground truth [INTROSPECT]

- `trt.TacticSource` members: `{CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS}`
  (11.1 docs list **only** `EDGE_MASK_CONVOLUTIONS` and `JIT_CONVOLUTIONS`, both "Enabled by default" [TRT-BCFG]).
- `trt.BuilderFlag` includes `FP16, FP8, TF32, REFIT, SPARSE_WEIGHTS`; **no `STRONGLY_TYPED`** (removed in 10.12).
- `trt.PreviewFeature` members: `{PROFILE_SHARING_0806, ALIASED_PLUGIN_IO_10_03, RUNTIME_ACTIVATION_RESIZE_10_10, MULTIDEVICE_RUNTIME_10_16}`.
- `trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH` still present (deprecated).
- `trt.OnnxParserFlag.NATIVE_INSTANCENORM` present.
- `IExecutionContext.set_device_memory(memory, size)` **exists** (two-arg, V2 semantics:
  docstring requires "256-byte aligned device memory" and size ≥ `get_device_memory_size_v2`);
  the legacy one-arg `device_memory` property setter also still exists. There is **no**
  Python attribute named `set_device_memory_v2` — the V2 API surfaces under the plain name.
- `ICudaEngine.device_memory_size_v2` exists.
- `Refitter` has `get_all`, `set_weights`, `set_named_weights`, `refit_cuda_engine`,
  `refit_cuda_engine_async`; `trt.WeightsRole` has 6 members.
- `trt.nptype(trt.DataType.FP8)` raises `TypeError` ("Could not resolve TensorRT datatype
  to an equivalent numpy datatype").
- Accessing deprecated enum members emits **no** runtime `DeprecationWarning` — deprecation
  is silent at runtime; only the docs/headers say so.

---

## 2. Verdict summary

**No CRITICAL findings.** Every documented contract with a runtime consequence
(tensor-address alignment, input-lifetime around `execute_async_v3`, timing-cache
lifetime, graph-capture rules, refit-vs-enqueued-work) was checked and the code
conforms — in several cases with explicit, comment-documented mitigations. The
findings below are deprecation debt (breaks on a future TRT 11 upgrade), missing
defensive checks, and minor idiom/perf notes.

| ID | Severity | File:line | Summary |
|---|---|---|---|
| H1 | HIGH | `src/streamdiffusion/acceleration/tensorrt/utilities.py:388-401` | `TacticSource.CUBLAS/CUBLAS_LT` removed in TRT 11 → whole tactic-scoping block silently no-ops on upgrade |
| H2 | HIGH | `utilities.py:1002` | Deprecated `context.device_memory` property setter (superseded since TRT 10.1); path is currently dead code |
| M1 | MEDIUM | `preprocessing/processors/trt_base.py:135,144,195,200,235` | `set_input_shape` / `set_tensor_address` bool returns unchecked → silent failure risk |
| M2 | MEDIUM | `utilities.py:544-570` | Zero-copy bind path has no 256-byte alignment guard (contract-safe under all default configs — proven in §5 — but unguarded against config drift) |
| M3 | MEDIUM | `src/streamdiffusion/tools/compile_depth_anything_tensorrt.py:129` | Deprecated `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` (sibling RAFT tool already fixed) |
| L1 | LOW | `utilities.py:1525` | `do_constant_folding=True` — "Deprecated option" in torch 2.8 |
| L2 | LOW | `utilities.py:1519-1530` | Legacy TorchScript ONNX exporter (`dynamo=False`); torch 2.8 recommends `dynamo=True` but keeps `False` as default |
| L3 | LOW | `utilities.py:1579` | External-data detection by `.pb` suffix scan is heuristic (safe in current flow, brittle to reordering) |
| I1–I5 | INFO | various | Redundant flag, CPU-alloc idiom, refit sync-scope note, legacy-import fallback, ctypes cudart |

Conformance confirmations (checked, no finding) are listed in §4.

---

## 3. Findings

### H1 — `TacticSource.CUBLAS/CUBLAS_LT` are gone in TRT 11; the SM_120 tactic-scoping block will silently vanish on upgrade

**File:** [utilities.py:388-401](../src/streamdiffusion/acceleration/tensorrt/utilities.py)

```python
if gpu_profile.compute_capability >= (12, 0):
    try:
        sources = (
            (1 << int(trt.TacticSource.CUBLAS))
            | (1 << int(trt.TacticSource.CUBLAS_LT))
            | (1 << int(trt.TacticSource.JIT_CONVOLUTIONS))
            | (1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS))
        )
        config.set_tactic_sources(sources)
        ...
    except (AttributeError, TypeError) as e:
        logger.debug(f"[TRT Config] set_tactic_sources not available: {e}")
```

**Docs:** [TRT-BCFG] documents `tensorrt.TacticSource` with exactly two members —
`EDGE_MASK_CONVOLUTIONS` and `JIT_CONVOLUTIONS`, each annotated "Enabled by default".
`CUBLAS`, `CUBLAS_LT`, and `CUDNN` no longer exist in 11.1. On installed 10.16 all five
members still exist [INTROSPECT], so the block works today.

**Consequence on TRT 11 upgrade:** the first expression, `trt.TacticSource.CUBLAS`,
raises `AttributeError`; the `except` swallows it at DEBUG level. Net effect: no call to
`set_tactic_sources` at all. Because the code's *primary intent* on SM_120 is to **exclude
CUDNN** (per the block comment) and CUDNN tactics no longer exist in TRT 11, the silent
no-op is behaviorally harmless there — but the comment's rationale becomes stale, the
INFO log ("CUDNN excluded for SM_120+") never fires, and cuBLAS/cuBLAS_LT scoping intent
is silently dropped rather than consciously retired.

**Remediation:** gate on `hasattr(trt.TacticSource, "CUBLAS")` instead of try/except so
the TRT 11 path is an explicit, logged branch (e.g. "TRT ≥11: default tactic sources
already exclude cuDNN/cuBLAS — nothing to scope"), and update the comment. No behavior
change on 10.16.

---

### H2 — deprecated `context.device_memory` property setter (dead path today)

**File:** [utilities.py:999-1002](../src/streamdiffusion/acceleration/tensorrt/utilities.py)

```python
def activate(self, reuse_device_memory=None):
    if reuse_device_memory:
        ...
        self.context.device_memory = reuse_device_memory
```

**Docs:** [TRT-CTX-CPP] `setDeviceMemory`: *"Deprecated in TensorRT 10.1. Superseded by
setDeviceMemoryV2()."* — with the additional caveat *"Weight streaming related scratch
memory will be allocated by TensorRT if the memory is set by this API"* (i.e. the legacy
setter's size assumption excludes weight-streaming scratch). The installed 10.16 Python
binding already exposes the successor as two-arg `set_device_memory(memory, size)` whose
docstring requires *"256-byte aligned device memory"* and *"size ... at least as large as
CudaEngine.get_device_memory_size_v2"* [INTROSPECT].

**Liveness:** grep of all 15 `activate()` call sites (runtime_engines, preprocessing
processors) shows **none pass `reuse_device_memory`** — the deprecated line is
unreachable in practice. `activate()` therefore always takes the
`create_execution_context()` branch.

**Remediation (pick one):**
1. Delete the `reuse_device_memory` parameter and branch (dead code); or
2. Migrate to `self.context.set_device_memory(ptr, self.engine.device_memory_size_v2)`
   if the memory-reuse feature is intended to come back.

Either way the deprecated one-arg setter should not survive to a TRT 11 upgrade.

---

### M1 — `trt_base.TensorRTEngine` ignores `set_input_shape` / `set_tensor_address` return values

**File:** [trt_base.py:135](../src/streamdiffusion/preprocessing/processors/trt_base.py)
(also :144, :195, :200 for `set_input_shape`; :235 for `set_tensor_address`)

```python
self.context.set_input_shape(name, input_shape)        # :135, :144, :195, :200
...
self.context.set_tensor_address(name, tensor.data_ptr())  # :235
```

**Docs:** [TRT-CTX-PY] `set_input_shape(self, name, shape) -> bool` — returns whether the
shape was set successfully (False for out-of-profile shapes, wrong rank, unknown name);
`set_tensor_address` likewise returns `bool`. A `False` return is TRT's *only* signal —
no exception is raised.

**Consequence:** an out-of-profile input shape (e.g. a preprocessing resolution outside
the engine's optimization profile) is silently accepted by Python; the failure then
surfaces downstream as an `execute_async_v3` failure or — worse — garbage output with
stale shapes. The core `Engine` class in utilities.py already does this correctly
(bool check at :1055; raises on `set_tensor_address` failure at :1209-1210); the second,
independent `TensorRTEngine` in trt_base.py (used by depth/pose/realesrgan/temporal-net
processors, i.e. live per-frame paths) does not.

**Remediation:** mirror the utilities.py pattern — raise `RuntimeError` with tensor name
and shape when either call returns `False`.

---

### M2 — zero-copy bind path (`_staging_action`) has no 256-byte alignment guard

**File:** [utilities.py:544-570](../src/streamdiffusion/acceleration/tensorrt/utilities.py)
(decision helper), bind at :1194/:1208.

```python
if name not in zero_copy_names or not is_contiguous or not dtype_match:
    return "copy"
```

**Docs:** [TRT-CTX-CPP] `setTensorAddress`: *"The pointer must have at least 256-byte
alignment."* The Sub-phase 5.6 zero-copy path binds caller-owned kvo/fio per-layer
*views* (bucket base + slot offset) directly to TRT, so view offsets must preserve
256-byte alignment.

**Current status: conformant.** §5 proves arithmetically that every per-layer view
offset is a multiple of 256 bytes under all default configurations (kvo unconditionally;
fio for every FI-eligible hidden dim under `max_fi_up_blocks ≤ 3`). No violation exists
today.

**Why still a finding:** the guarantee is *implicit* — it rests on (a) fp16 dtype,
(b) hidden dims being multiples of 64, (c) the default FI eligibility mask, and
(d) PyTorch's allocator alignment. None of these is asserted anywhere. A config drift
(`max_fi_up_blocks=4` exposing H=320 layers, combined with an odd
`maxframes·batch·seq` product — see §5) or an atypical model would silently violate the
documented contract, and TRT does not reliably diagnose misaligned addresses (undefined
behavior territory).

**Remediation:** add `cur_ptr % 256 == 0` to `_staging_action`'s eligibility test
(misaligned → fall back to `"copy"`, which is always safe). One integer modulo per input
per frame on the CPU; keeps the pure-function unit-testability. Optionally log once at
WARNING when the fallback triggers.

---

### M3 — deprecated `EXPLICIT_BATCH` flag in the Depth-Anything compile tool

**File:** [compile_depth_anything_tensorrt.py:129](../src/streamdiffusion/tools/compile_depth_anything_tensorrt.py)

```python
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
```

**Docs:** `EXPLICIT_BATCH` is deprecated since TRT 10.0 (networks are always
explicit-batch); the flag still exists in installed 10.16 [INTROSPECT] and is ignored.
The sibling tool already carries the fix and the rationale:
[compile_raft_tensorrt.py:152](../src/streamdiffusion/tools/compile_raft_tensorrt.py) —
`network = builder.create_network()  # EXPLICIT_BATCH deprecated/ignored in TRT 10.x`.

**Remediation:** same one-line change as the RAFT tool. Low blast radius (offline tool),
but it breaks outright when TRT removes the enum member.

---

### L1 — `do_constant_folding=True` is a deprecated `torch.onnx.export` option

**File:** [utilities.py:1525](../src/streamdiffusion/acceleration/tensorrt/utilities.py)

**Docs [INTROSPECT, torch 2.8.0 docstring]:** *"do_constant_folding: Deprecated
option."* (alongside `training`, `operator_export_type`, `custom_opsets`). Harmless
today; drop it whenever the export call is next touched.

### L2 — legacy TorchScript exporter (`dynamo=False`)

**File:** [utilities.py:1519-1530](../src/streamdiffusion/acceleration/tensorrt/utilities.py)

**Docs [INTROSPECT, torch 2.8.0]:** `dynamo=False` **is still the default** in 2.8 and is
not deprecated; the docstring says `dynamo=True` "is the recommended way to export models
to ONNX." No action required now — but the custom attention processors / export wrappers
were built against TS-exporter tracing semantics, so treat any future move to
`dynamo=True` as its own validation project, not a flag flip. Being explicit
(`dynamo=False`) is *better* than relying on the default, since the default is the likely
thing to change in a future torch.

### L3 — external-data detection by `.pb` suffix scan

**File:** [utilities.py:1579](../src/streamdiffusion/acceleration/tensorrt/utilities.py)

```python
external_data_files = [f for f in os.listdir(onnx_dir) if f.endswith(".pb")]
```

Safe in the current flow **only because** `export_onnx` consolidates torch's external
tensor files (named `onnx__*`, no `.pb` suffix) into a single `weights.pb` and deletes
the originals (:1545-1562) before `optimize_onnx` runs. If a >2GB model ever reached
`optimize_onnx` straight from `torch.onnx.export` (which writes extension-less external
files), the scan would report "no external data" and `onnx.load(onnx_path)` at :1611
would still succeed (external data loads by default) but the subsequent plain
`onnx.save` at :1614 would fail on the 2GB protobuf limit. Prefer
`onnx.external_data_helper`-based detection (check `TensorProto.data_location ==
EXTERNAL` on the loaded model) or document the ordering invariant at the scan site.

### INFO notes

- **I1 — redundant parser flag.** `parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)`
  at utilities.py:769/:893 — [TRT-ONNX]: *"This flag is ON by default"* (and required for
  version-/hardware-compatible engines). Not deprecated; setting it is a no-op. Keep or
  drop, cosmetic either way.
- **I2 — CPU-alloc-then-copy.** utilities.py:1066
  `torch.empty(tuple(shape), dtype=torch_dtype).to(device=device)` allocates on CPU then
  copies; trt_base.py:165 already uses the direct `device=` form (with a comment saying
  why). One-time cost at buffer (re)allocation only — perf-cosmetic.
- **I3 — refit sync scope.** [TRT-REFIT]: *"The behavior is undefined if the engine has
  pending enqueued work."* The refit path synchronizes `torch.cuda.current_stream()` at
  utilities.py:723 before `refit_cuda_engine()` — correct for the load-time-only refit
  flow (comment there documents this as defensive). Note it syncs torch's current stream,
  not the engine's polygraphy stream; if refit were ever made reachable mid-stream, sync
  the engine stream too.
- **I4 — legacy cuda-python import fallback.** utilities.py:37-41 prefers
  `from cuda.bindings import runtime as cudart` and falls back to `from cuda import cudart`.
  Installed 12.9.7 emits on the legacy import: *"The cuda.cudart module is deprecated and
  will be removed in a future release, please switch to use the cuda.bindings.runtime
  module instead."* [INTROSPECT]. The preference order is already correct; the fallback
  only fires on cuda-python < 12.x installs. Fine as is.
- **I5 — ctypes cudart in `tools/cuda_l2_cache.py`** (`cudaDeviceSetLimit`,
  `cudaStreamSetAttribute` via ctypes): bypasses cuda-python entirely; works but carries
  its own struct-layout risk. Out of scope for remediation; noted for completeness.

---

## 4. Conformance confirmations (checked, no finding)

| Contract | Doc | Code | Verdict |
|---|---|---|---|
| `execute_async_v3` input lifetime (inputs must not be modified/freed before stream sync) | [TRT-CTX-PY] | Inputs staged via `copy_()` into persistent engine buffers *on the engine stream* (utilities.py:1158-1196); zero-copy names are persistent, address-stable caches by construction | ✔ |
| Default-stream warning (`enqueueV3` on default stream ⇒ extra device sync) | [TRT-CTX-CPP] | Core engine runs on polygraphy-owned non-default stream; trt_base engines create a dedicated `torch.cuda.Stream()` with pre/post event barriers + `record_stream()` | ✔ |
| `setTensorAddress` 256-byte alignment | [TRT-CTX-CPP] | Proven for all bound tensors: full buffers are torch base allocations (512-B aligned); kvo/fio view offsets ≡ 0 mod 256 under defaults (§5). Guard recommended (M2) | ✔ |
| CUDA graph capture: quiesce streams, ThreadLocal mode, instantiate signature | cuda-python 12.9 [INTROSPECT] | 3 warmup passes, engine-stream + legacy-stream drain before `cudaStreamBeginCapture(..., ThreadLocal)`; `cudaGraphInstantiate(graph, 0)` matches documented `(graph, unsigned long long flags)`; `CUASSERT` correctly unpacks `(err, result)` tuples; graph-launch failure falls back to `execute_async_v3` and re-captures | ✔ |
| Rebind-after-capture rule (addresses baked into graph) | [TRT-CTX-CPP] | `_staging_action` returns `bind_and_reset` on pointer change with live graph → `reset_cuda_graph()`; addresses only rebound when no graph instance exists (utilities.py:1198-1212) | ✔ |
| Timing cache lifetime ("must not be destroyed until after the engine is built") | [TRT-BCFG] | Local `trt_cache` created/loaded at utilities.py:807-817 stays in scope through `build_serialized_network` | ✔ |
| `builder_optimization_level` valid range 0–5 (default 3) | [TRT-BCFG] | Profile uses 3 or 4 | ✔ |
| `avg_timing_iterations` (default 1), `max_num_tactics` (default −1) | [TRT-BCFG] | 8/4 iterations, cap 64 — valid values, guarded by try/except for older TRT | ✔ |
| `PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10` | [TRT-BCFG] + [INTROSPECT] | Present in installed 10.16 *and* still documented in 11.1 | ✔ |
| `BuilderFlag.STRONGLY_TYPED` removed in 10.12 | [INTROSPECT] | `hasattr(trt.BuilderFlag, "STRONGLY_TYPED")` guard at utilities.py:913-919 correctly skips on 10.16; FP8 path uses the network-creation flag instead | ✔ |
| `trt.nptype` has no FP8 mapping | [INTROSPECT] | `TypeError` → FP8 fallback at utilities.py:1043-1051 | ✔ |
| Refitter API (`get_all`, `set_weights`, `WeightsRole`, `refit_cuda_engine`) | [TRT-REFIT] | All still current in 11.1 docs, zero deprecation notes on the page; weight lifetime rule ("can be unset and released ... after refit_cuda_engine returns") satisfied — numpy arrays outlive the call | ✔ |
| `set_input_shape`/`set_tensor_address` return checks (core engine) | [TRT-CTX-PY] | utilities.py:1055 (bool check), :1209-1210 (raise) | ✔ (trt_base gap = M1) |
| `onnx.save_model` external-data kwargs | [INTROSPECT] onnx 1.19.1 | `save_as_external_data=True, all_tensors_to_one_file=True, location="weights.pb", convert_attribute=False` all match the installed signature | ✔ |
| `modelopt.onnx.quantization.quantize` kwargs | [INTROSPECT] 0.43.0 | `quantize_mode`, `use_external_data_format`, `high_precision_dtype` all valid; fp8_quantize.py additionally guards every kwarg via `inspect.signature` | ✔ |
| `polygraphy engine_from_bytes(serialized_engine, runtime=None)` | [INTROSPECT] 0.49.26 | Called with a single positional arg | ✔ |
| cuda-python module layout (`cuda.bindings.runtime` preferred) | [INTROSPECT] 12.9.7 | Preferred import first, legacy fallback second (utilities.py:37-41) | ✔ |

---

## 5. Appendix — kvo/fio zero-copy alignment arithmetic (checkpoint #1)

**Contract:** [TRT-CTX-CPP] `setTensorAddress`: *"The pointer must have at least 256-byte
alignment."*

**What gets bound:** `Engine.infer` binds `buf.data_ptr()` for every feed-dict entry in
`zero_copy_names` (utilities.py:1180-1194). `zero_copy_names` is exactly the kvo + fio
input names (unet_engine.py:100/:108). Those tensors are the per-layer views created in
[models/utils.py](../src/streamdiffusion/acceleration/tensorrt/models/utils.py):

- kvo (`create_kvo_cache`, :118-122): bucket `torch.zeros(L, 2, mf, B, S, H, dtype=fp16)`;
  per-layer view = `bucket[slot]`.
- fio (`create_fi_cache`, :239-251): bucket `torch.zeros(L, mf, B, S, H, dtype=fp16)`;
  per-layer view = `bucket[slot]`.

(`L` = layers in bucket, `mf` = cache_maxframes, `B` = batch, `S` = sequence length,
`H` = attention hidden dim; fp16 ⇒ 2 bytes/element.)

**Base pointers:** each bucket is a fresh `torch.zeros` CUDA allocation. PyTorch's CUDA
caching allocator returns block-granular base pointers (512-byte rounding), so
`bucket.data_ptr() % 256 == 0`. (This is allocator behavior, not a documented API
contract — one more reason for the M2 guard, which makes the whole argument
assumption-free at runtime.)

**View offsets:**

- kvo: `offset(slot) = slot · (2·mf·B·S·H elements) · 2 B = 4·slot·mf·B·S·H` bytes.
  Every attention hidden dim in the supported UNets (SD1.5/SD-Turbo/SDXL: 320, 640, 1280)
  is a multiple of 64, so `4·H ≡ 0 (mod 256)` ⇒ **every kvo view offset is a multiple of
  256 unconditionally** (independent of `mf`, `B`, `S`, `slot`).
- fio: `offset(slot) = 2·slot·mf·B·S·H` bytes ⇒ multiple of 256 iff `slot·mf·B·S·(H/64)`
  is even.
  - Default FI eligibility (`max_fi_up_blocks=2`, models/utils.py:127-164) admits only
    mid-block + first two up-blocks ⇒ `H ∈ {1280}` (SD1.5: up_blocks[0] has no
    attentions, up_blocks[1] is H=1280) or `H ∈ {1280, 640}` (SDXL). `H/64 ∈ {20, 10}`
    is even ⇒ **aligned unconditionally**.
  - `max_fi_up_blocks=3` adds H=640 layers (SD1.5) — still even ⇒ aligned.
  - Only a non-default `max_fi_up_blocks=4` (which also contradicts the function's
    documented "final up-block always excluded" rule) would admit H=320 (`H/64 = 5`,
    odd). Then alignment requires `slot·mf·B·S` even. At the H=320 level
    `S = (h/8)·(w/8)`, odd only when *both* `h/8` and `w/8` are odd (resolutions
    ≡ 8 mod 16, e.g. 968×968) — so a violation additionally needs odd `mf`, odd `B`,
    and an odd slot. Narrow, but reachable ⇒ finding M2.

**Rebound caches:** `stream_parameter_updater.py:1089` replaces `kvo_cache[i]` with a
standalone tensor — a fresh base allocation (≥256-aligned) — and the pointer change is
detected by `_staging_action` → `bind_and_reset` → graph reset. Conformant.

**Verdict: no alignment violation exists in any default configuration.** The M2 guard
converts this from an arithmetic argument into a runtime invariant.

---

## 6. Ranked fix shortlist (candidate follow-up gated commits)

1. **M1 — check `set_input_shape`/`set_tensor_address` returns in trt_base.py**
   (raise like utilities.py does). Cheapest, closes a real silent-failure hole on live
   per-frame paths.
2. **M2 — add `cur_ptr % 256 == 0` to `_staging_action` eligibility** (+ unit test rows
   in the existing pure-function test). Makes the documented alignment contract a runtime
   invariant instead of an arithmetic proof.
3. **H1 — replace try/except with `hasattr(trt.TacticSource, "CUBLAS")` version gate**
   and refresh the comment/log so the TRT 11 behavior is explicit, not accidental.
4. **H2 — delete the dead `reuse_device_memory` branch** (or migrate to
   `set_device_memory(ptr, engine.device_memory_size_v2)` if reuse is planned).
5. **M3 — `create_network()` in compile_depth_anything_tensorrt.py:129**, same one-liner
   as the RAFT tool.
6. **L1 — drop `do_constant_folding=True`** next time the export call is touched.
7. **L3 — harden external-data detection** in `optimize_onnx` (data_location check or a
   comment documenting the `export_onnx`-consolidates-first invariant).

Items 1–5 are behavior-preserving on the installed stack (TRT 10.16 / cuda-python 12.9 /
torch 2.8) and would each pass the standard gate (pytest 117+/0, ruff, pyrefly ≤72
baseline) independently.
