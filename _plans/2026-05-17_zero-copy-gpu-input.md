# True zero-copy GPU input — close the input/output asymmetry

> **Hand-off from `wiggly-finding-puzzle.md` (CUDARuntimeTypes import fix, commit `eecb9f5`).** Input direction transport now works end-to-end (logs at 21:38:15 confirm `CUDA IPC input ready: shm=StreamDiffusionTD_512-512_input_ipc`). But the input *payload* still detours through CPU. Output direction is already true zero-copy. This plan brings input to parity.

## Context

After commit `eecb9f5` (CUDARuntimeTypes import fix), both directions of the TD ↔ SD bridge use the same vendored `_compat/cuda_ipc/` package as IPC transport. But the data flow is asymmetric:

| Direction | Transport | Payload handling | Notes |
|---|---|---|---|
| SD → TD (output) | `CUDAIPCExporter` | **Stays on GPU end-to-end** | `wrapper._ipc_pack_rgba` builds BGRA on GPU → `exporter.export_frame(data_ptr, numel)` |
| TD → SD (input) | `CUDAIPCImporter` | **D2H → numpy → CPU preprocess → H2D** | `_get_input_frame_cuda_ipc` does `.cpu().numpy()`, then loop does CPU float-cast, then `VaeImageProcessor.preprocess` runs on CPU, then pinned-buffer H2D |

The input direction throws away the zero-copy GPU tensor the Importer already hands out, then re-materializes it via a CPU round-trip — pure waste.

### What's already in place (verified by Phase 1 Explore)

- **`CUDAIPCImporter.get_frame()`** (`src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py:903`) already returns a **persistent, zero-copy `torch.Tensor` on CUDA** built in `TorchBuffers.build` (lines 282-316) via `__cuda_array_interface__` + `torch.as_tensor`. Shape on the TD wire: **HWC float32 BGRA, range [0,1]**. Optional `stream=` kwarg adds `cudaStreamWaitEvent` so the consumer's stream waits GPU-side for the producer's IPC event — no CPU sync.
- **Pipeline fast-path already exists**: `StreamDiffusion.__call__` at `src/streamdiffusion/pipeline.py:1024-1039` skips `image_processor.preprocess` and the H2D staging copy when `x` is a CUDA tensor with `dtype == self.dtype` and `shape[-2:] == (self.height, self.width)`. Hit those three conditions exactly and the pipeline runs zero-copy. Every example script (`examples/benchmark/single.py:107-119`, `examples/img2img/single.py`, `examples/screen/main.py`, `examples/vid2vid/main.py`) already uses this pattern: `image_tensor = stream.preprocess_image(...); stream(image=image_tensor)`.
- **`wrapper.img2img`** (`src/streamdiffusion/wrapper.py:856-860`) already passes a `torch.Tensor` straight through to `self.stream(image)` — only `str`/`Image.Image` inputs go through `preprocess_image`. No wrapper-side changes needed.
- **GPU BGRA→RGB indexer is already inline** at `td_manager.py:687`: `gpu_frame[..., [2, 1, 0]].contiguous()`. We keep that step; we just stop scaling/casting/D2H afterward.
- **ControlNet & IPAdapter use independent SHM streams** (`self.control_memory`, `self.ipadapter_memory`) — they don't read the main input frame, so the input zero-copy plan has **no coupling** to those features. They keep their CPU paths unchanged.

### What the current code does (the waste)

`td_manager.py:684-689` (current `_get_input_frame_cuda_ipc` tail):

```python
gpu_frame = self._cuda_ipc_importer.get_frame()  # zero-copy torch.Tensor on GPU [HWC float32 BGRA, [0,1]]
if gpu_frame is None:
    return None
rgb = gpu_frame[..., [2, 1, 0]].contiguous()      # BGRA → RGB (drop alpha)  ← KEEP this step on GPU
rgb_u8 = (rgb.clamp(0, 1) * 255).to(torch.uint8)  # ← WASTE: rescale up to uint8 just to rescale back down
return rgb_u8.cpu().numpy()                        # ← WASTE: D2H + numpy (~0.5-1ms per frame, syncs the stream)
```

Followed by `td_manager.py:527-529`:

```python
if input_image.dtype == np.uint8:
    input_image = input_image.astype(np.float32) / 255.0  # ← WASTE: CPU rescale back to [0,1]
```

Followed by `pipeline.py:1034-1039`:

```python
_raw = self.image_processor.preprocess(x, self.height, self.width)  # ← WASTE: CPU resize/normalize on numpy
self._input_staging.copy_(_raw)                                       # ← WASTE: CPU→pinned copy
x = self._input_staging.to(device=self.device, non_blocking=True)     # ← WASTE: H2D copy of data we already had on GPU
```

Net effect: ~3 redundant rescales, 1 D2H, 1 H2D, 1 stream-blocking sync, per frame. The user's success logs already show occasional `total_time` jitter spikes (298→1133→709µs) and 100×-outlier memcpy spikes (~612µs at Frames 1746, 2231) which are consistent with the D2H sync interacting with WDDM scheduling — eliminating this path should shrink both the steady-state median and the tail.

## Approach

Replace the D2H tail of `_get_input_frame_cuda_ipc` with a small GPU-only transform that **lands the tensor exactly in the shape/dtype/range the pipeline fast-path expects**, then route it through `_streaming_loop` as a `torch.Tensor` instead of `np.ndarray`. Mirror exactly what the output direction already does.

### The GPU transform (single chained op, all in-flight on the current CUDA stream)

```python
# gpu_frame: HWC float32 BGRA on GPU, range [0,1]   ← from Importer.get_frame()
# target:    NCHW self.dtype on GPU, range [-1,1]   ← pipeline fast-path expects this exactly
nchw = (
    gpu_frame[..., [2, 1, 0]]    # HWC float32 RGB [0,1]            (drop alpha, channel swap — free view+gather)
    .mul(2.0).sub_(1.0)          # HWC float32 RGB [-1,1]           (matches VaeImageProcessor.normalize: img*2-1)
    .permute(2, 0, 1)            # CHW float32 RGB [-1,1]           (free view)
    .unsqueeze(0)                # NCHW float32 RGB [-1,1] (N=1)    (free view)
    .to(dtype=self.wrapper.stream.dtype, non_blocking=True)
    .contiguous()                # required: permute leaves strides non-contiguous; pipeline downstream expects contiguous NCHW
)
```

Cost: one channel-gather kernel + one elementwise scale + one dtype cast, all bandwidth-bound on a 512×512×3 tensor (~1.5MB at fp16) — **well under 100µs** on the user's RTX 4090, vs the ~0.5-1ms the D2H currently burns. Plus we delete the `astype(float32)/255.0` CPU op and the entire `VaeImageProcessor.preprocess` + pinned-buffer H2D path.

### Optional stream-sync hardening

`CUDAIPCImporter.get_frame(stream=...)` accepts a CUDA stream and inserts `cudaStreamWaitEvent` so the consumer's stream waits GPU-side for the producer's IPC event. Pass `torch.cuda.current_stream()._as_parameter_` so the wait happens inside the pipeline's own stream — eliminates the small CPU-side wait the Importer currently does internally. **Defer to Phase 2** if the v1 path lands cleanly; v1 should work without it.

## Code changes

### Patch 1 — `StreamDiffusionTD/td_manager.py:661-689` — rewrite `_get_input_frame_cuda_ipc` tail

Replace lines 661-689 with:

```python
def _get_input_frame_cuda_ipc(self) -> Optional["torch.Tensor"]:
    """Read one frame from TD's CUDAIPCExporter and return a GPU torch.Tensor
    matching the pipeline's fast-path: NCHW, self.stream.dtype, range [-1,1],
    on CUDA, shape (1, 3, height, width). Bypasses image_processor.preprocess
    and pinned-buffer H2D entirely."""
    if self._cuda_ipc_importer is None:
        if not self._probe_ipc_input_shm():
            return None
        from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter
        try:
            self._cuda_ipc_importer = CUDAIPCImporter(
                shm_name=self.cuda_ipc_input_shm_name,
                debug=False,
            )
        except Exception as e:
            logger.warning(f"CUDAIPCImporter init failed: {e}")
            self._cuda_ipc_importer = None
            return None
        if not self._cuda_ipc_importer.is_ready():
            self._cuda_ipc_importer = None
            return None
        logger.info(f"CUDA IPC input ready (zero-copy GPU): shm={self.cuda_ipc_input_shm_name}")

    gpu_frame = self._cuda_ipc_importer.get_frame()  # HWC float32 BGRA on GPU, [0,1]
    if gpu_frame is None:
        return None

    target_dtype = self.wrapper.stream.dtype
    nchw = (
        gpu_frame[..., [2, 1, 0]]    # HWC RGB float32 [0,1]
        .mul(2.0).sub_(1.0)          # HWC RGB float32 [-1,1]
        .permute(2, 0, 1)            # CHW RGB float32 [-1,1]
        .unsqueeze(0)                # NCHW (N=1)
        .to(dtype=target_dtype, non_blocking=True)
        .contiguous()
    )
    return nchw
```

Key shifts vs current code:
- Return type: `Optional[np.ndarray]` → `Optional[torch.Tensor]`
- Drops `clamp(0,1)*255 → uint8 → .cpu().numpy()` (3 ops, 1 stream-sync, 1 D2H)
- Adds `mul(2).sub_(1) → permute → unsqueeze → to(dtype)` (4 ops, all in-flight on GPU)
- Log line gains "(zero-copy GPU)" to make the success criterion trivially greppable

### Patch 2 — `StreamDiffusionTD/td_manager.py:643-659` — update dispatcher return type

The legacy SHM path still returns numpy uint8 RGB. The dispatcher needs to allow both. Change the return annotation only:

```python
def _get_input_frame(self) -> Optional[Union["np.ndarray", "torch.Tensor"]]:
    """Get input frame from TouchDesigner (platform-specific).

    Returns:
        torch.Tensor (NCHW, self.stream.dtype, [-1,1] on CUDA) — CUDA IPC path
        np.ndarray (HWC uint8 RGB) — legacy SHM / Syphon paths
    """
    # body unchanged
```

(Imports: `Union` is already in scope at top of file — verify in patch.)

### Patch 3 — `StreamDiffusionTD/td_manager.py:521-549` — branch on input type in `_streaming_loop`

Replace lines 527-529 (the uint8→float32 CPU cast) with a type guard so the CPU path runs only for numpy inputs:

```python
# img2img mode: get input frame and process
input_image = self._get_input_frame()
if input_image is None:
    time.sleep(0.001)
    continue

# CUDA IPC fast-path returns a ready-to-consume GPU tensor; legacy SHM path returns
# HWC uint8 RGB numpy which still needs the CPU float-cast.
if isinstance(input_image, np.ndarray) and input_image.dtype == np.uint8:
    input_image = input_image.astype(np.float32) / 255.0
```

No other changes in this block. `self.wrapper.img2img(input_image)` already passes `torch.Tensor` straight through to `self.stream(image)` (`wrapper.py:856-860`), which hits the pipeline fast-path because we constructed the tensor to satisfy all three checks at `pipeline.py:1028-1033`.

### What we explicitly do NOT touch

- `wrapper.img2img` — already correct; tensor passthrough is its existing behavior
- `wrapper.preprocess_image` — only used by examples, not by the TD streaming loop
- `pipeline.__call__` fast-path — already exists and is the contract we satisfy
- `_input_staging` allocation in `pipeline.prepare()` — stays allocated but unused on the IPC path; ~1.5MB pinned, not worth conditional logic
- `_process_controlnet_frame` / `_process_ipadapter_frame` — independent SHM streams, no coupling
- `_send_output_frame` / `postprocess_image` — output direction already zero-copy
- `_compat/cuda_ipc/` — no changes; the existing API is sufficient

## Verification

After applying all three patches in the running SD venv (no rebuild needed — pure Python; `Scripts/` edits live-reload per `[[project_scripts_dir_purpose]]`):

### 1. Smoke test the type contract

Before launching the .toe, run this in SD's venv to confirm the new return-type plumbing parses:

```powershell
venv\Scripts\python -c "from StreamDiffusionTD.td_manager import TouchDesignerManager; print('OK')"
```

Must print `OK`. Any `SyntaxError` / `NameError` / `ImportError` means a patch is wrong — stop and re-read the file.

### 2. Functional test — relaunch the .toe

Watch SD cmd log for:
- ✅ `CUDA IPC input ready (zero-copy GPU): shm=StreamDiffusionTD_512-512_input_ipc` (new log marker proving Patch 1 took effect)
- ✅ No `_get_input_frame:` debug exceptions
- ✅ Frames stream at ≥ the prior baseline (~20-26 FPS in success logs from 2026-05-17 21:38)
- ✅ Clean shutdown on OSC `/stop` (no leaked tensor refs / IPC handle errors)

### 3. Visual round-trip check (TD)

TD Receiver COMP should show:
- Correct colors (BGRA→RGB shuffle still happens, just on GPU now — verify no R/B swap)
- No tone shift or banding (`mul(2).sub_(1)` followed by VAE encode is the same arithmetic the CPU path performed via `astype(float32)/255 → VaeImageProcessor.normalize`, so output should be visually identical)
- No flicker / dropped frames

### 4. Performance verification

In SD log, compare these metrics against the 2026-05-17 21:38 baseline:
- **Steady-state `total_time`**: expected ~0.5-1ms lower (D2H + CPU rescale + H2D eliminated)
- **`total_time` jitter** (max/min spread): expected meaningfully tighter, since the stream-blocking `.cpu()` sync is gone
- **GPU memcpy spikes** (the ~612µs Frame 1746 / 2231 outliers): may or may not disappear — those might be unrelated allocator/WDDM events, but at minimum we've removed one possible source

Optional deeper check with `nsys`:

```powershell
nsys profile -o input_zerocopy_after --trace cuda,nvtx --capture-range cudaProfilerApi `
  venv\Scripts\python -m StreamDiffusionTD.main_sdtd ...
```

Then `nsys analyze input_zerocopy_after.nsys-rep` — the input direction should show **zero `cudaMemcpyAsync DtoH` calls** between consecutive `cudaGraphLaunch` calls.

## Commit

Per `[[feedback_pr_branch_convention]]`, branch stays at `feat/cuda-ipc-output` (current head: `eecb9f5`), PR target `SDTD_031_dev`.

```powershell
./scripts/git/commit_enhanced.sh --no-venv `
  "feat: true zero-copy GPU input via CUDA IPC (close input/output asymmetry)"
```

Then save the plan as a project file per `[[feedback_save_plans_as_project_files]]`:
- Copy this file to `StreamDiffusion/_plans/2026-05-17_zero-copy-gpu-input.md`

## Critical files

| File | Lines | Change |
|---|---|---|
| `StreamDiffusionTD/td_manager.py` | 661-689 | Rewrite `_get_input_frame_cuda_ipc` body — drop D2H, add GPU NCHW transform |
| `StreamDiffusionTD/td_manager.py` | 643-659 | Update `_get_input_frame` return-type annotation only |
| `StreamDiffusionTD/td_manager.py` | 521-549 | Branch CPU float-cast on `isinstance(np.ndarray)` |

Reused unchanged (verified by Phase 1 Explore):

- `src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py:903` — `get_frame()` already returns zero-copy GPU torch.Tensor
- `src/streamdiffusion/wrapper.py:834-873` — `img2img` already passes `torch.Tensor` straight through
- `src/streamdiffusion/pipeline.py:1024-1039` — fast-path already exists for tensors matching `is_cuda + dtype + (H,W)`
- `src/streamdiffusion/wrapper.py:921-925, 967-975` — output direction reference pattern (`_ipc_pack_rgba` + `export_frame(data_ptr, …)`)

## Out of scope

- **Stream-sync hardening** via `get_frame(stream=current_stream)`. Defer to a v2 if v1 lands stable. The Importer's internal CPU wait is microseconds and not the dominant cost; we can come back for the last 5%.
- **`_input_staging` deallocation**. Stays as ~1.5MB pinned dead weight on the IPC path; not worth a conditional in `pipeline.prepare()`.
- **ControlNet / IPAdapter zero-copy**. Independent SHM streams, independent CPU rescales. Would be a similar 3-patch refactor per feature; not blocking the main img2img path.
- **GPU memcpy spike investigation** (~612µs at Frames 1746 / 2231). If they persist after this change, that's a separate diagnosis (likely WDDM preemption or allocator pressure, not input-path related).
- **BGRA → RGB correctness audit**. The shuffle is byte-identical to the prior code; this plan only changes when (`.cpu()`) and how (`.mul().sub_().permute().unsqueeze().to(dtype)`) we use it.
- **A shared `_ipc_unpack_input` utility** in `wrapper.py` to mirror `_ipc_pack_rgba`. Could DRY the two directions; defer until a second consumer appears.
