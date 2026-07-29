# FI fusion design spike — where the 2.8 ms goes, and what (not) to build

Date: 2026-07-29. Status: **measurement-complete; recommendation below**.

This executes the B2a GO proviso from `docs/profiling/fp8_fi_gates_2026-07-29.md`
("targeted ncu on the FI-attributable kernels first, before writing any plugin") and
closes follow-up item 3a in `docs/plans/PMPP_Follow_Up.md` at a written go/no-go —
no runtime code in this pass.

## 1. Method — three evidence layers

1. **Canonical timing** (unchanged): the original A/B `profiler_logs/
   sdtd_benchmark_20260729_{125524,125616}_stats.json` — `unet_step` p50
   **27.950 ms (fi_on) vs 25.118 ms (fi_off) → FI cost +2.832 ms (+11.3%)**.
2. **Graph-granularity recon** (offline, `logs/fi_recon_20260729.py` on the original
   `logs/nsys_fp8_fi_{on,off}_20260729.sqlite`): nsys's default
   `--cuda-graph-trace=graph` records CUDA-graph replays only as whole-graph
   `GRAPH_TRACE` spans — the ~123 ms of KERNEL rows in those traces are
   capture/eager-phase only.
3. **Node-granularity re-runs** (`logs/run_node_trace_20260729.sh`, nsys 2026.3.1,
   `--cuda-graph-trace=node`, same pinned configs `configs/profiling/fp8_fi_on.yaml` /
   `fi_ablation_off.yaml`): per-replay kernels fully traced —
   `logs/nsys_node_fp8_fi_{on,off}_20260729.sqlite`, analyzed by
   `logs/fi_family_diff_20260729.py`. Node tracing adds measurable overhead
   (`unet_step` p50 28.302/26.274 in the accompanying stats jsons), so **node-run and
   ncu numbers are used for attribution and structure only, never as canonical
   timing** (per `scripts/profiling/README.md` guardrails).

## 2. Where the 28 ms goes — the pipeline is GPU-bound

From the graph-granularity `GRAPH_TRACE` spans (original traces, 23 steady replays):

| CUDA graph | fi_on p50 | fi_off p50 | Identity |
|---|---:|---:|---|
| graphId 4 (UNet + ControlNet) | **26.136 ms** | **23.663 ms** | Δ **+2.473 ms** = FI's in-graph cost |
| graphId 1 | 0.838 ms | 0.837 ms | VAE (arm-identical) |
| graphId 7 | 1.006 ms | 1.005 ms | VAE (arm-identical) |

GPU busy ≈ 26.1 + 1.0 + 0.8 ≈ **27.9 ms of a ~29 ms frame → ~96% GPU-bound**. There is
no launch-overhead mystery: the earlier "123 ms captured vs 672 ms wall" gap was purely
trace granularity. The NVTX `unet_step` CPU range closes long before the GPU finishes
(CPU wall p50 3.2/2.1 ms); the 27.95 ms stats.json p50 is the CUDA-event span. FI's
cost is real per-frame GPU work — kernel-level optimization is a legitimate target,
which funded the node-granularity pass below.

Cross-check ladder (fi_on − fi_off): canonical CUDA-event Δ **+2.832 ms** ⊇ UNet-graph
span Δ **+2.473 ms** ≈ node-granularity in-graph kernel-sum Δ **+2.436 ms** ≈ family
table total Δ **+2.430 ms**. The ~0.36 ms remainder lives outside the UNet graph
(eager-phase kernels and CPU-side work).

## 3. Per-frame FI cost decomposition (node granularity, steady state)

`logs/fi_family_diff_20260729.py` restricts to kernels replayed inside the UNet graph
(graphId 5 in both node traces), clusters the 23 replays by time gap, drops 4 warmup
replays, and takes per-family per-replay medians. Kernel names are normalized across
engines (Myelin `_0x…` hashes and per-engine layer indices stripped). Steady-state
per-replay kernel sum: **26.014 ms (fi_on, 1974 kernels) vs 23.578 ms (fi_off, 1535
kernels) → Δ +2.436 ms and +439 kernels per frame.**

Category sums (per-frame median ms, fi_on − fi_off):

| Category | Δ ms/frame | Share | Constituent families (Δ ms, fi_on inst) |
|---|---:|---:|---|
| **FI similarity + blend math** | **+1.43** | 59% | `TranReshMulSum` +0.435 (46), `SqrtMaxMinReplDiv` +0.259 (46), `MulMulCastCastMulReplGat…` (gather/select blend) +0.190 (39), `Topk` +0.176 (46), `DivReshTranMulSum` +0.152 (46), `ReshSqrtMaxMinReshTranRe…` +0.126 (46), `MaxrReshEqlGtrOr` (threshold) +0.096 (45) |
| **GEMM precision shift** | **+0.68** | 28% | e4m3×f16 GEMMs +0.491 (547 vs 501 inst), new f16 `trt_ampere_h16816gemm_128x64` +0.430 (40) and `…256x128` +0.266 (6), pure-fp8 e4m3×e4m3 GEMMs **−0.510** (241 vs 287 inst) |
| **K/V concat + data movement** | **+0.10** | 4% | new `MoveTranReshTranReshConc` +0.383 (20) and `Conc` +0.089, offset by `MoveSlicReshTranReshSlic…` −0.179 and `MoveSlicReshTranReshConc` −0.193 |
| **Myelin refusion churn** | **+0.21** | 9% | elementwise chains regrouped into different fusions (e.g. `AddSqrtDivMulCastMulAddM…` +0.267 vs `CastMulAddMulCastCastMul…` −0.266, GEGLU/erf chains ±) — net noise |
| **Total** | **+2.43** | 100% | matches the +2.436 per-replay kernel-sum delta |

Two structural findings:

1. **The dominant cost is the `get_nn_feats` cosine-similarity pipeline itself**
   (`src/streamdiffusion/acceleration/tensorrt/attention_processors.py:9-35`:
   `F.normalize`×2 + `bmm` + `.max(-1)` + `gather` + `where`, called at :215) — ~1.43 ms
   spread over ~46 Myelin kernel launches per frame per family. The `[B,N,M]` cosine
   matrix is materialized, normalized, reduced, and re-read across separate kernels.
2. **FI knocks the attention GEMMs off the pure-fp8 path.** With FI/EA on, cached K/V
   (fp16) is concatenated into the attention inputs (:147-154), and ~46 pure
   e4m3×e4m3 GEMM instances become e4m3×f16 (+0.49 ms), plus 46 new plain-f16
   `h16816gemm` instances appear (+0.70 ms) — consistent with the cosine `bmm` and
   concat-fed attention running in fp16. This 0.68 ms is a *precision/layout* cost a
   fusion plugin would NOT recover.

### Correction to the results doc's framing

`docs/profiling/fp8_fi_gates_2026-07-29.md` (B2a "Where the cost lives") reported
"GEMM families are flat … the FI cost signature is elementwise/copy overhead, not
GEMM", and its family-share table quotes 123.4/116.5 ms "total captured kernel time".
Those numbers are **capture/warmup-phase composition only** (graph-granularity traces
miss all replay kernels; `graphNodeId` is 0 on every KERNEL row). At per-frame steady
state the picture inverts: the Move/Slic/Resh movement chain is net **+0.10 ms** (the
capture-phase headline "+2.72 ms run-total" family), while GEMMs are **not** flat
(+0.68 ms net precision shift). The capture-phase diff remains valid only as "which
kernel families FI adds", never as per-frame cost.

## 4. ncu limiter structure of the FI kernels

Narrowed detailed-set pass on the fp8_fi_on arm
(`logs/ncu_config_fp8_fi_on_detailed_20260729_143902.ncu-rep`, 40 launches, kernel
regex = the 8 FI families from §3; content check passed, 40 non-blank rows; ncu
serializes launches so **no ncu number below is timing** — Speed-of-Light and
occupancy ratios only, per the README guardrails). Medians over 5 launches per family:

| Family (role) | SM % | Mem/DRAM SOL % | Achieved occ. % |
|---|---:|---:|---:|
| `MoveSlicResh…Conc…` (K/V-cache concat chain) | 6.1 | **85.2** | 85.6 |
| `TranReshMulSum` (cosine `bmm` reduction) | 7.6 | **72.3** | 77.8 |
| `SqrtMaxMinReplDiv` (normalize + max) | 35.2 | **69.6** | 75.7 |
| `MulMulCastCastMulReplGath…Sele…` (gather/select blend) | 21.6 | 42.8 | 31.5 |
| `ReshSqrtMaxMinReshTranReshReplDiv` | 27.0 | 40.4 | 71.0 |
| `MaxrReshEqlGtrOr` (threshold/where) | 4.1 | 34.1 | 25.7 |
| `DivReshTranMulSum` | 9.8 | 30.8 | 26.6 |
| `Topk` (`.max(-1)`) | 8.7 | 27.3 | 30.7 |

Read-out:

- **Every FI family is memory-bound** (SM ≤ 35%, memory SOL = DRAM SOL throughout —
  the traffic goes all the way to DRAM, not caches).
- The three heavy families stream the materialized `[B,N,M]` cosine matrix through
  DRAM at 70–85% of peak: **these kernels are individually near-roofline — there is
  no per-kernel tuning win.** The only way to make this work cheaper is to *not do
  the traffic*, i.e. an online-(max,idx) fusion that never materializes the matrix
  (option b) — which is why the ~1.4 ms ceiling in §5 is credible.
- The five light families run at 26–43% SOL and 26–32% occupancy — small,
  latency-bound launches whose cost is mostly per-launch overhead; a fusion absorbs
  them for free.
- The concat chain at 85% DRAM SOL is an efficient copy; a layout restructure
  (option a) could only shrink the bytes moved, and §3 caps that prize at
  ~0.1 ms/frame net.

## 5. Options analysis

### (a) Cache-layout restructure — **demoted by the data**

Premise (from the capture-phase diff): store the FI bank / EA K-V cache pre-shaped so
the per-frame transpose/reshape/concat chain (attention_processors.py:151-154,
:213-216) leaves the exported graph. The node-granularity numbers cap this option's
prize at **~+0.10 ms/frame net** (new concat work +0.47, offset by −0.37 of movement
that FI's graph restructure removes elsewhere) — total movement-family time is
~1.4–1.5 ms/frame in *both* arms, i.e. mostly non-FI plumbing. An engine rebuild plus
export-graph surgery for ≤0.5 ms gross is poor ROI. **Not recommended.**

### (b) Custom fused TRT plugin for `get_nn_feats` — **real ceiling ~1.4 ms, not funded now**

A single-launch online-(max,idx) kernel that never materializes the `[B,N,M]` cosine
matrix would attack the 1.43 ms math category (§3) and some launch/refusion churn —
realistic recovery **~1.0–1.4 ms/frame ≈ 4–5% of `unet_step`**, not the 2.8 ms the GO
gate implied. Costs: 1–3 weeks; reintroduces the MSVC/CUDA toolchain ADR-0001
deliberately avoided; must survive all three engine-install paths
(`unet_unified_export.py:15-52`, `refresh_fi_procs` :120-139,
`_patches/diffusers_kvo_patch.py:68-140`). **No-go at current economics** — revisit
triggers below.

### (b′) K/V-cache precision experiment — new lever surfaced by this spike

The 0.68 ms GEMM precision shift is addressable without any plugin: store/quantize the
EA cached K/V (and FI bank) as e4m3 at write time so the concatenated attention inputs
stay on the pure-fp8 GEMM path. Runtime + export change, engine rebuild, no toolchain;
quality-gated (same fixed-seed PSNR harness as ADR 0003). Cheap to prototype and
composable with the fp8-coverage work. **Recommended as the next gated experiment if
FI latency matters.**

### (c) Accept the cost — **recommended default**

FI buys the StreamV2V temporal-coherence behavior (thesis Eq 3.1/3.2, docstring
attention_processors.py:38-64) for +2.4 ms in-graph ≈ 8.7% of the frame. The
**bigger prize identified by ADR 0003 is unrelated to FI**: ~32% of kernel time is
still f16 GEMM; raising fp8 layer coverage attacks several ms without new kernels or
toolchain.

## 6. Recommendation and gate

**No-go on the 3a fused plugin at current economics.** Priority order:

1. **fp8 layer-coverage raise** (ADR 0003's named lever) — largest addressable pool,
   no toolchain.
2. **(b′) e4m3 K/V-cache experiment** — targets 0.68 ms, small, quality-gated.
3. **Plugin (b)** — only if BOTH revisit triggers fire: (i) FI's per-frame share grows
   (larger feature bank / more cached frames scales the `[B,N,M]` matrix), and
   (ii) the coverage + (b′) work lands and FI's ~1.4 ms math becomes the top remaining
   line item.

**Parity design recorded for whichever implementation ever goes:** standalone harness
asserting equivalence against eager `get_nn_feats` (StreamV2V Eq 3.1/3.2), then a
frame-for-frame `_fi_cache_out` diff on fixed seed (12-frame settle, final-frame
compare) using the `logs/quality_check_20260729.py` pattern; PSNR vs the unmodified
engine must be bit-identical for a pure-layout change and ≥ the 23.65 dB ADR-0003
baseline for any precision change.

## Artifacts

| Path | Content |
|---|---|
| `logs/nsys_node_fp8_fi_{on,off}_20260729.{nsys-rep,sqlite}` | node-granularity traces (attribution only) |
| `logs/nsys_node_*_cuda_gpu_kern_sum.csv` | node-run kernel summaries |
| `logs/fi_recon_20260729.py`, `logs/fi_family_diff_20260729.py` | throwaway analysis scripts |
| `profiler_logs/sdtd_benchmark_20260729_{143023,143054}_stats.json` | node-run stats (arm-validity check only — not canonical) |
| `logs/ncu_config_fp8_fi_on_detailed_20260729_143523.*` | first ncu pass (`__myl` regex — flooded by non-FI elementwise kernels; superseded) |
| `logs/ncu_config_fp8_fi_on_detailed_20260729_143902.{ncu-rep,csv}` | narrowed ncu pass behind §4 (8 FI families, 40 launches) |
