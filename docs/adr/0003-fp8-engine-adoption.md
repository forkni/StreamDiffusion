# FP8 (fp8v3) stays the deployed UNet engine precision

Status: accepted

## Context & Decision

The deployed TouchDesigner preset (`StreamDiffusionTD/td_config.yaml`) builds the UNet
TensorRT engine with `fp8: true` (fp8v3 quantization, e4m3). Whether fp8 should remain
the default — versus falling back to fp16 — was an open follow-up item (3c in
`docs/plans/PMPP_Follow_Up.md`), blocked on a controlled A/B, which ran on 2026-07-29
with pinned arm configs differing only on the `fp8` flag
(`configs/profiling/fp8_fi_on.yaml` vs `fp16_ab.yaml`; full method and numbers in
`docs/profiling/fp8_fi_gates_2026-07-29.md`).

Measured on the deployed workload (SDXL-Turbo 512×512, batch 2, cached-attn/V2V + FI +
ControlNet, CUDA graphs on):

- **Latency:** `unet_step` p50 **27.950 ms (fp8) vs 29.799 ms (fp16)** — fp8 is
  **6.2% faster**; engine-level p50 27.735 vs 29.571 ms; p95 also favors fp8
  (29.636 vs 30.752 ms).
- **Quality (fp8 vs fp16 reference, fixed seed, identical input):** PSNR 23.65 dB,
  MAE 8.248, max-abs 205.0 (uint8 scale). Visible but artistically acceptable deviation
  for the live-visuals TouchDesigner use-case; this pipeline has no fidelity-critical
  consumer.

**Decision: keep fp8 as the deployed UNet engine precision.** The 6.2% p50 win is real
headroom at the frame-rate budget, and the quality delta is acceptable for this
deployment.

## Considered Options

- **fp16 UNet engine** — rejected for deployment: 6.2% slower at p50 with no consumer
  that needs the fidelity; remains the reference arm for quality comparisons
  (`configs/profiling/fp16_ab.yaml`).
- **fp8 (chosen)** — fastest measured arm; quality delta tolerable for live visuals.
- **Defer until fp8 covers more of the GEMM time** — rejected as a blocker (the win is
  already real), but retained as the named revisit lever below.

## Consequences

- **The fp8 ceiling is coverage-limited, not kernel-limited:** in the fp8 arm only
  ~36% of GPU kernel time runs as e4m3 GEMMs while ~32% remains f16 `cutlass` GEMMs
  (kernel-family shares in `docs/profiling/fp8_fi_gates_2026-07-29.md`). Raising fp8
  layer coverage in the quantization recipe is the main future latency lever — worth a
  follow-up before any deeper kernel work on the GEMM path.
- Quality should be re-checked (same fixed-seed PSNR/MAE harness,
  `logs/quality_check_20260729.py` pattern) whenever the quantization recipe or model
  changes; 23.65 dB is the accepted baseline, not a floor guaranteed by construction.
- Reproducible fp8 measurement requires the pinned arm config
  (`configs/profiling/fp8_fi_on.yaml`, engine hash `h18b4bb48d936`) — never profile
  against the live-edited `StreamDiffusionTD/td_config.yaml`.
- ncu limiter data for the e4m3 kernels (all shared-memory-limited, 2–3 blocks/SM) is
  recorded in the same results doc; the PMPP report's smem-limited occupancy thesis
  extends to the fp8 path unchanged.

## Addendum (2026-07-29): MHA Q/DQ recipe (`fp8_mha_qdq`)

The coverage follow-up named above ran the same day
(`docs/profiling/fp8_coverage_2026-07-29.md`). A new opt-in flag `fp8_mha_qdq`
(default `false`; default path byte-identical to production) drops modelopt's
`disable_mha_qdq` exclusion, quantizing the attention BMMs and — as a verified side
effect — the whole K/V-cache path (all 70 K and 70 V cache concats enter the fp8
domain; a separate cache-quantization arm is unnecessary). Engines fork the cache
tag `--fp8v3` → `--fp8v3-mhaq`.

Measured vs this ADR's fp8 arm on the identical workload: `unet_step` p50
**27.293 ms vs 27.950 ms (−2.4%)**, PSNR vs fp16 **25.07 dB** (above the 23.65 dB
accepted baseline), MHA kernels stayed fused on Ada (140, unchanged). Gate G1
passed → the recipe is **kept**: exposed through the TD component's *Performance*
TRT profile (`fp8_mha_qdq: true` there only). The deployed production
`td_config.yaml` is unchanged; adopting mhaq as the deployed default remains a
recommendation pending soak in live TD use.
