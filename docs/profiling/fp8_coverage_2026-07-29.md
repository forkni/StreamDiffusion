# FP8 coverage deep-dive — 2026-07-29

Executes the top two recommendations of `docs/plans/fi_fusion_spike_2026-07-29.md`:
(1) raise fp8 layer coverage via MHA Q/DQ (Arm 1), (2) quantize the K/V-cache path
(Arm 2, "b′"). Every timing number below comes from a `profiler_logs/*_stats.json`
CUDA-event file produced with **no ncu attached**; node-trace (nsys sqlite) numbers
are used only for *attribution shares*, never as canonical per-frame timing.

## Setup

- **GPU:** RTX 4090 (Ada SM 8.9, WDDM), TouchDesigner closed, GPU work strictly
  sequential.
- **Workload:** sdxl-turbo, 512×512, `t_index_list: [13, 30]` (batch-2 denoise),
  cached-attn V2V, FI on, ControlNet engine loaded (runtime scale 0), img2img
  benchmark target. Baseline canonical p50: `unet_step` **27.950 ms**
  (`profiler_logs/sdtd_benchmark_20260729_125524_stats.json`, see
  `fp8_fi_gates_2026-07-29.md`).
- **Arms** (pinned configs in `configs/profiling/`, `build_engines_if_missing: false`):

  | Arm | Config | Engine | Axis |
  |---|---|---|---|
  | `fp8_fi_on` | `fp8_fi_on.yaml` | `--fp8v3--h18b4bb48d936` | baseline recipe |
  | `fp8_mhaq` | `fp8_mhaq_fi_on.yaml` | `--fp8v3-mhaq--hfbd6a0d5b858` | + MHA Q/DQ (`fp8_mha_qdq: true`) |

## Phase 0 — Per-frame GEMM-precision attribution (offline, node traces)

Method: `logs/fp8_gemm_attrib_20260729.py` on the node-granularity traces
`logs/nsys_node_fp8_fi_{on,off}_20260729.sqlite` — same replay clustering as the FI
spike (UNet graphId 5, 2 ms replay gap, skip 4 warmup replays, 19 steady replays),
kernel families from `demangledName`, classified by operand-pair markers in the
kernel name (`e4m3e4m3` = pure fp8, `e4m3f16`/`f16e4m3` = mixed, `f16f16`/`h16816`/
`hgemm` = pure f16; `_gemm_mha_v2_0x<hash>` = Myelin fused MHA, precision opaque
from the name but f16 by recipe since MHA is Q/DQ-excluded).

Per-frame median ms per class (class sums reconcile with the per-replay kernel-sum
p50 to <0.01 ms in both arms: 26.007 vs 26.014 fi_on, 23.577 vs 23.578 fi_off):

| Class | fi_on ms | fi_on % | inst | fi_off ms | inst | Δ (on−off) |
|---|---:|---:|---:|---:|---:|---:|
| `gemm_mixed` (e4m3×f16) | **7.971** | 30.6% | 547 | 7.480 | 501 | +0.491 |
| `mha_fused` (`_gemm_mha_v2`) | **5.055** | 19.4% | 140 | 5.056 | 140 | ≈0 |
| `gemm_fp8` (e4m3×e4m3) | 3.467 | 13.3% | 241 | 3.977 | 287 | −0.510 |
| `myl_other` (Myelin fused misc) | 2.998 | 11.5% | — | — | — | — |
| `conv_mixed` (e4m3-w × f16-act) | 2.718 | 10.4% | 34 | 2.710 | 34 | ≈0 |
| `fi_math` (8 FI Myelin families) | 2.448 | 9.4% | 364 | 1.385 | — | +1.063 |
| `gemm_f16` (pure f16, h16816gemm) | 0.803 | 3.1% | 51 | 0.104 | 5 | **+0.699** |
| `conv_fp8` | 0.528 | 2.0% | 9 | — | — | — |
| `conv_f16` | 0.018 | — | — | — | — | — |

Findings:

1. **Gate G0 PASSES for Arm 1** — the MHA-attributable pool (`mha_fused`,
   the `disable_mha_qdq` exclusion) is **5.06 ms/frame**, 10× the 0.5 ms bar.
   Identical across FI arms (140 instances, ±0.001 ms), i.e. orthogonal to FI.
2. **NEW: mixed-precision e4m3×f16 dominates model-wide, not just the cache path.**
   `gemm_mixed` (7.97 ms) + `conv_mixed` (2.72 ms) ≈ 10.7 ms/frame run with e4m3
   weights against **f16 activations** in *both* FI arms. Pure-e4m3 GEMMs are the
   minority (3.5 ms). Total f16-operand exposure ≈ 16.5 of 26.0 ms (64%) — the
   capture-phase "32.4% f16" in `fp8_fi_gates_2026-07-29.md` counted only pure-f16
   kernel names and missed this. Activation-side quantization coverage is therefore
   the *largest* untapped lever (~10.7 ms pool) — larger than either arm of this
   pass. Recorded as future work, not exercised here.
3. The FI precision shift from the spike doc re-confirms at node granularity:
   FI on moves 46 GEMM instances fp8→mixed (−0.51/+0.49 ms) and adds
   +0.70 ms of pure-f16 `h16816gemm` (40× 128x64 + 6× 256x128) — the b′ target
   pool.

## Phase 1 — `fp8_mha_qdq` flag plumbing + cache-tag fork

New optional bool `fp8_mha_qdq` (default **False** — default path byte-identical to
production), mirroring the `fp8_use_cached_attn` plumbing:

- `config.py` `_extract_wrapper_params` → `wrapper.py` `__init__`/`self` →
  `_unet_build_opts` → `acceleration/tensorrt/__init__.py compile_unet` (pop/pass)
  → `builder.py build()` → `quantize_onnx_fp8(disable_mha_qdq=not fp8_mha_qdq)`.
- **Cache identity:** the quantization recipe is not otherwise part of the engine
  hash, so the flag forks the tag at both `engine_manager.py` sites:
  `--fp8v3` → `--fp8v3-mhaq` (canonical prefix, which is hashed, *and* the visible
  dir tag). Verified: flag omitted vs `False` → byte-identical path
  (`sdxl-turbo--fp8v3--he06eb26efb0f…` unchanged on arbitrary args; the real
  deployed dir `h18b4bb48d936` reproduces from its canonical string); flag `True`
  → `sdxl-turbo--fp8v3-mhaq--h<newhash>…`. Golden config-extraction test updated
  and passing (4 passed).

## Recipe structure — offline ONNX inspection (b′ design input)

`logs/inspect_fp8_onnx_20260729.py` on the base-recipe
`…--fp8v3--h18b4bb48d936…/unet.fp8.onnx` (graph only, no external data):

- 70 `kvo_cache_in_*` (FLOAT16 `[2, 4, 2B, seq, dim]` — K+V paired on dim0,
  4 = cache_maxframes) and 46 `fio_cache_in_*` (FLOAT16 `[4, 2B, 256, 1280]`)
  graph inputs; 1726 QuantizeLinear + 1726 DequantizeLinear among 13,425 nodes.
- **The `_FEATURE_EXCLUDE_PATTERNS` name-excludes are nearly a no-op at node
  level**: only 2 nodes in the whole graph carry cache-pattern names (2 Casts).
  The cache path stays f16 because (a) `disable_mha_qdq=True` excludes the
  attention BMMs and (b) Slice/Concat/Transpose are data movement modelopt never
  quantizes anyway — not because of the name patterns.
- kvo chain per attention layer: `kvo_cache_in → Slice → Concat_2` (K) /
  `Slice_2 → Concat_3` (V), concat outputs feeding the *unquantized* BMMs raw.
  **Consequence: b′ has no mechanism without Arm 1** — there is no downstream Q to
  feed until the BMMs themselves are quantized. Arm 2 is therefore evaluated
  strictly on top of the Arm 1 engine (as the plan's gating anticipated).
- The FI blend output (`…/attn1/Mul_4`) already feeds `QuantizeLinear` — FI's
  downstream projections re-enter the quantized domain; the FI-attributable f16
  exposure is upstream (gather/top-k math + the BMM path).

## Phase 2 — Arm 1: MHA Q/DQ (`fp8_mha_qdq: true`)

Build: `logs/build_fp8_mhaq_20260729.py` + `logs/fp8_mhaq_build_20260729.yaml`
(scratch copy of `fp8_fi_on.yaml`). Reference signals from the base build
(`…h18b4bb48d936…/build_stats.json`): 4033 fp8 Q/DQ TRT layers, **140 fused-MHA
kernels**, 1914 engine layers, 2087 s total.

- Build signals (`…hfbd6a0d5b858…/build_stats.json`, finished 2026-07-29 20:20 UTC,
  2343 s total): `fp8_qdq_layers` **5337** vs base 4033 (**+1304** — MHA Q/DQ took
  effect); `mha_fused_kernels` **140, unchanged** vs base — **MHA stayed fused, did
  not decompose on Ada** (the feared fallback did not happen at build level);
  `total_engine_layers` 1934 (base 1914); engine 2496.7 MB (base 2495.7);
  `fp8_onnx_quantize` 1448 s (base 1220), `trt_build` 495 s (base 478). No
  `_assert_finite_qdq_scales` abort, no errors in the build log.
- Canonical A/B (`profiler_logs/sdtd_benchmark_20260729_181624_stats.json`, no
  nsys/ncu attached; engine-load confirmed via `-mhaq` engine file access time
  18:17:31 during the run + `build_engines_if_missing: false`):
  `unet_step` p50 **27.293 ms** vs baseline 27.950 ms → **−0.657 ms (−2.4%)**.
  Corroborating regions: `unet_step.engine` 27.092 vs 27.735 (−0.643),
  `predict_x0_batch` 27.486 vs 28.019 (−0.533). Speed half of G1 (≥0.3 ms) **passes**.
- PSNR (`logs/quality_fp8_mhaq_20260729.npy`, 12-frame deterministic harness):
  vs `quality_fp16_20260729.npy` **25.07 dB** (gate ≥ 23.1 dB — passes, and
  *exceeds* the base-fp8 recipe's 23.65 dB vs the same fp16 reference); vs
  `quality_fp8_20260729.npy` 23.26 dB (recipe drift — the two fp8 recipes differ
  inside attention, expected).
- **Gate G1 (keep if ≥ 0.3 ms faster AND PSNR ≥ 23.1 dB): KEEP** —
  −0.657 ms and 25.07 dB both clear their bars.

## Phase 3 — Arm 2 (b′): K/V-cache quantization

Gated on Arm 1 (see recipe-structure section — no mechanism without it). Outcome:
**b′ is fully subsumed by Arm 1 — no separate arm exists to build.** Offline
inspection of the mhaq `unet.fp8.onnx` (`logs/inspect_mhaq_cache_q_20260729.py`;
2378 Q/DQ pairs vs base 1726, +652 pairs = the +1304 TRT-layer delta):

- All **70 V-cache concats** (`attn1/Concat_3`) feed `QuantizeLinear` directly.
- All **70 K-cache concats** (`attn1/Concat_2`) reach `QuantizeLinear` immediately
  after the attention-scale `Mul` — `Concat → Mul → Q`.

Once `disable_mha_qdq` is dropped, modelopt quantizes the BMM inputs, which *are*
the cache-concat outputs — the K/V-cache path enters the fp8 domain with no extra
recipe work, exactly the mechanism the base recipe lacked. The Arm 1 A/B and PSNR
numbers above therefore already measure the quantized-cache configuration; a
standalone Arm 2 would re-measure the same engine. (The mhaq export also reshapes
the cache chain: `kvo_cache_in → Gather → Transpose → Reshape → Concat`, replacing
the base recipe's `Slice` split.)

## Verdicts & deployment recommendation

- **Arm 1 (`fp8_mha_qdq: true`): KEEP.** −0.657 ms `unet_step` p50 (27.293 vs
  27.950 ms, −2.4%), PSNR 25.07 dB vs fp16 (better than the base recipe's
  23.65 dB), MHA stayed fused (140 kernels, no Ada decomposition), build cost
  +256 s (2343 vs 2087 s).
- **Arm 2 (b′): closed as subsumed** — the Arm 1 graph already quantizes both
  K and V cache paths (Phase 3 section); no separate engine or measurement exists.
- **Deployment:** flags stay default-OFF in code (`fp8_mha_qdq` defaults `false`;
  the default `--fp8v3` path is byte-identical to production). The TD component's
  **Performance** TRT profile now carries `fp8_mha_qdq: true`, so selecting
  Performance in TD generates a config that resolves to the already-built
  `--fp8v3-mhaq--hfbd6a0d5b858` engine (matching pars: 512×512, 2 t-indexes,
  cachef4, FI on, CN scribble, batch-2 — no rebuild); other par combos rebuild
  only if `Buildifmissing` is on (~39 min). The wrapper's `[TRT] UNet engine:`
  launch line now prints `fp8_mha_qdq=` for visibility. Flipping the *production*
  `td_config.yaml` remains a recommendation only.
- **Future work (recorded, not exercised):** activation-side quantization of the
  mixed e4m3×f16 pool (~10.7 ms/frame, the largest remaining lever — Phase 0
  finding 2) and the FI gather-BMM pure-f16 pool (+0.70 ms, Phase 0 finding 3).
