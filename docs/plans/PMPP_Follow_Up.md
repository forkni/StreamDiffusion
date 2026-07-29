# PMPP follow-up: gap-closure research + verified improvement plan

## Context

`docs/PMPP_deep_dive_verification_2026-07-29.md` (71024be) audited StreamDiffusion against the
PMPP checklist with real `nsys`/`ncu` measurements. Two independent verification passes now
back this plan:

1. **Code verification (this session):** Explore agent + MCP semantic search confirmed all 18
   claim groups; report is substantially accurate; four cosmetic errata found.
2. **Book verification (parallel agent, spot-checked this session):** three primary sources at
   `F:\RD_PROJECTS\COMPONENTS\GPU_BOOKS\` — PMPP 5th ed. (1646 pp), The CUDA Handbook (2013),
   Inference Engineering — map onto the report's open gaps. **The pivotal claim was re-verified
   verbatim this session from the PDF (pp. 880–882):** PMPP 5e §15.7 states low occupancy is
   the *intended* operating point for register-tiled matmul ("the extreme benefit of register
   tiling … makes it worthwhile"; shared-mem pressure "can easily be alleviated" by shrinking k;
   registers are the usual binding limiter at 255/thread → 12.5% occupancy — squarely inside
   the report's measured 7.5–14%). This **contradicts the report's §3.1 emphasis** that
   occupancy is the defect; the measurement captured only shared-mem/block limiters, never the
   register limiter, so it cannot currently distinguish "defect" from "correctly-tuned CUTLASS
   tile". Resolving that is gate A1.

New code touchpoints surfaced by the parallel pass; **all re-verified this session via the MCP
code-search index (synced 2026-07-29 03:43) + call-graph + grep:**

- `get_nn_feats` has **one production call line**: `attention_processors.py:215` inside
  `CachedSTAttnProcessor2_0.__call__` (class spans 38-220); call graph confirms no other
  production callers. Its docstring cites the StreamV2V thesis §3.4.2 Eq 3.2 as the source.
- A fused kernel must survive **three** install/collect paths: `_collect_fi_processors`
  (`unet_unified_export.py:15-52`), `refresh_fi_procs` (`unet_unified_export.py:120-139`), and
  the monkey-patch layer (`_patches/diffusers_kvo_patch.py:68-140`).
- There are **two** exporter-init paths: `_lazy_init_ipc_exporter` (wrapper.py:1136-1164) and
  `_lazy_init_cn_ipc_exporter` (wrapper.py:1196-1222) — each with its **own** forced
  `export_sync=True` override (wrapper.py:1151 and :1209) and its own `exporter.export(` call
  (wrapper.py:1036 main; :1238 CN preview). Double-buffering and any sync-default flip must
  cover both.
- Engine identity: wrapper.py:2090-2117 calls `EngineManager.get_engine_path`
  (`engine_manager.py:136-254`) passing `use_feature_injection` (:2102), `fp8` (:2104),
  `static_batch_size` (:2111) — the canonical hash lives in EngineManager, not wrapper.py.

**User decisions (2026-07-29):** FP8-vs-FP16 → *investigate drift first*. FI kernel →
*measure-first gate*.

**PDF access:** repo venv lacks pymupdf; use system `python` (has fitz 1.28) via Bash heredoc.
Page numbers below are 1-based PDF indices (PyMuPDF), not printed pages.

---

## Status (2026-07-29 execution pass)

Results and verdicts live in `docs/profiling/fp8_fi_gates_2026-07-29.md`; commits
99f1cd5 / 5a546de / c532221.

- **Done:** A1 (report §3.1 stands — smem-limited), A4 (report amended), B1 (`_pack_bgra`
  helper + test), B4 (ncu guardrails + `profile_ncu.py` sanity check; fp8 drift record),
  2a (FI ablation: +2.832 ms = +11.3% on `unet_step` p50 → **GO** on 3a), 2d (fp8 vs
  fp16: fp8 6.2% faster, PSNR 23.65 dB), 3c (**closed** — fp8 stays deployed, see
  `docs/adr/0003-fp8-engine-adoption.md`).
- **Dropped:** 2c (see the dated note on the item below — the deployed VAE is TensorRT,
  so the PyTorch-side A/B has no production consumer).
- **Open:** 2b (blocking-export gate; needs a from-scratch mock cuda-link consumer —
  none exists in the repo) and 3b behind it; A2/A3 book extraction; 3a is in design
  spike (FI deep-dive: per-frame cost reconciliation + Myelin-kernel ncu, then
  plugin-vs-cache-layout-restructure decision, `docs/plans/fi_fusion_spike_2026-07-29.md`).
- **Update (2026-07-29, later same day):** the fp8-coverage lever named in ADR-0003
  executed — `fp8_mha_qdq` recipe (MHA Q/DQ + K/V-cache quantization subsumed):
  `unet_step` p50 27.293 ms (−0.657 ms vs the 3c baseline 27.950), PSNR 25.07 dB
  vs fp16, gate G1 **KEEP**; exposed via the TD *Performance* TRT profile,
  production default unchanged. Full record:
  `docs/profiling/fp8_coverage_2026-07-29.md` + ADR-0003 addendum. Next-largest
  recorded lever: activation-side quantization of the mixed e4m3×f16 pool
  (~10.7 ms/frame).

## Workstream A — Research & report correction (docs + logs only, no runtime code)

### A1. Settle the occupancy contradiction (measurement gate — run first)

One ncu pass on the same 10 kernels as the report, adding the limiter metrics it missed:
`launch__occupancy_limit_registers`, `launch__occupancy_limit_warps`,
`launch__registers_per_thread`, `sm__maximum_warps_per_active_cycle_pct`.

- Same shape as report §6: `--config StreamDiffusionTD/td_config.yaml --set detailed
  --kernel-regex "cutlass|gemm" --launch-skip 5 --launch-count 10`,
  `STREAMDIFFUSION_PROFILE_TRT=1`, background, ≥10 min timeout. Uses the already-fixed
  `scripts/profiling/profile_ncu.py` as-is.
- Guardrails: sanity-check the `.ncu-rep` with `--metrics gpu__time_duration.sum` before
  trusting it; confirm each added metric resolves in the export header (plausible names drop
  silently); quote **no timing** from this run.
- **Decision rule:** register limit ≤ shared-mem limit and regs/thread near 255 → kernels match
  §15.7's register-tiled profile; report §3.1 is wrong in *emphasis* (low occupancy = intended
  operating point, not the bottleneck). Shared memory genuinely tighter → §15.7's "shrink k"
  says TRT left something on the table and the report stands.

### A2. Targeted book extraction (priority order)

1. **`get_nn_feats`** — PMPP §20.5 flash attention (p1214-1233) + Ch. 10 reduction
   (§10.5/10.7/10.10); Handbook Ch. 15 normalized correlation (p476-497, the exact math shape —
   shows the sums-based single-pass normalization). Key simplification: FI needs only online
   **argmax**, not softmax — no running sum, no output rescaling; composition collapses to
   (max, idx) pair-merging, materially simpler than flash attention. Reconcile against the
   StreamV2V thesis §3.4.2 Eq 3.2 (the docstring's cited source), not just the current code.
2. **Occupancy** — PMPP §15.3/§15.7/§15.8, §5.6, interpreted against A1's verdict.
3. **BGRA pack** — PMPP §6.3 vector stores (p321); Handbook write-combining (p150-153), uchar4
   (p362-363). The uchar4 single-4-byte-store form is the *principled* fix; record it as the
   documented ceiling — not worth a custom kernel for a measured ≤11 µs/frame (the `flip(-1)`
   one-liner in B1 captures most of it in pure PyTorch).
4. **IPC double buffering** — PMPP §6.7 (p334-336): double buffering removes a false
   write-after-read dependence — precisely the ADR-0001 race; Handbook Ch. 6 streams/events.
5. **FP8 drift + TRT plugin cost** — Inference Engineering §5.1 quantization (p122-131), §4.3
   engines/plugins (p107, 101-111); §2.4 roofline (p63-69, states image-gen inference is
   compute-bound — corroborates the report's DRAM ≤64% finding).
6. **channels_last** — **no direct coverage in any of the three books** (zero hits for
   NHWC/NCHW/channels_last). Record general layout-for-coalescing principle (PMPP §6.1/§6.8)
   only; state the absence plainly, don't manufacture a citation. Measurement (B2c) is the
   only arbiter.

Also extract PMPP 5e §6.8 "A checklist of optimizations" (p336-344) — the 5th edition's own
checklist, superseding the podcast's 4th-ed distillation that framed the original audit.

### A3. Deliverable: `StreamDiffusion/docs/PMPP_gap_closure_research_2026-07-29.md`

- **Part 1 — findings.** One section per gap: what the report measured; what the source says
  (book + section + PDF page); confirms/contradicts/refines; revised recommendation. Open with
  the §15.7 contradiction + A1 verdict (most consequential). Include a short re-run of the
  report's concept table against the §6.8 5e checklist.
- **Part 2 — implementation plan.** Per item: change, files, risk, expected gain (from the
  report's already-measured numbers, never re-estimated), verification. Mirrors Workstream B.

### A4. Amend the verification report (single edit pass, after A1)

- §3.1: correction or confirmation note per A1's verdict + cross-reference to the new doc.
  Surgical — measurements stand; only the interpretation is in question.
- Fold in the four cosmetic errata found earlier: (i) `attention_processors.py` cites → full
  path `acceleration/tensorrt/models/...`; (ii) "pinned staging" mislabel for the D2D IPC pack
  buffer (real pinned paths: `pipeline.py:423-427`, `_output_pin_buf`); (iii) L2-persistence
  prose actually at `cuda_l2_cache.py:316-319, 352-370` — fix cites; (iv) `pipeline.py:629-634`
  is *unconditional* allocation; only 1217-1224 is inside `denoising_steps_num > 1`.

---

## Workstream B — Implementation (gated; can start B1 in parallel with A1)

### B1. Quick wins (~1 hour, one commit; hygiene-level impact)

- **BGRA pack**: [wrapper.py:1131-1133](StreamDiffusion/src/streamdiffusion/wrapper.py:1131) →
  `self._ipc_pack_buf[..., :3] = rgb_hwc.flip(-1)`; same edit in the config-gated duplicate
  `_ipc_pack_unit_rgba` (1191-1193) — fix, don't delete (`send_controlnet_preview` feature).
- **Test**: new `tests/unit/test_ipc_pack_bgra_correctness.py` — old 3-write vs `flip(-1)`,
  `torch.equal` on all channels (pure permutation, exact equality correct), parameterized over
  both functions, CUDA part behind the repo's skip idiom. Rerun
  `tests/unit/test_ipc_producer_stream.py` + full `pytest tests/unit -q`.

### B2. Measurement gates (before funding any B3 item)

- **2a. FI ablation** (first — cheapest gate for the biggest unknown): scratch config
  `configs/profiling/fi_ablation_off.yaml`, byte-identical to `td_config.yaml` except explicit
  `use_feature_injection: false` — omitting the key defaults it to **true**
  ([config.py:171-177](StreamDiffusion/src/streamdiffusion/config.py:171)). Diff the *full
  resolved config* (engine hash also encodes `static_batch_size` etc. via the
  `get_engine_path` call at wrapper.py:2090-2117 / `engine_manager.py:136-254`).
  Fresh engine build (≤37 min; shared `_timing_cache/` should shorten). A/B `unet_step` p50 via
  `venv/Scripts/python scripts/profiling/profile_nsys.py --target benchmark --config <cfg>`.
  **Gate:** <2% delta (≈0.4 ms) → no-go on 3a; ≥3-5% → go (with a targeted ncu pass on the
  FI-attributable kernels first).
- **2b. Blocking-export cost** (~half day): headless mock cuda-link consumer (mirrors TD's
  `CUDALinkBootstrap`, drains frames so `export()` returns real PUBLISHED); CUDA-event/NVTX
  timing around `exporter.export(` (wrapper.py:1036 — main path; the CN-preview export at
  :1238 is out of scope for the gate); forced-sync vs scratch-only
  `CUDALINK_EXPORT_SYNC=0` (upper bound; race-prone, never production). **Gate:**
  sub-millisecond stall → no-go on 3b (CUDA 719 history).
- **2c. `channels_last` A/B** on the PyTorch-side VAE only (filler): CUDA-event harness,
  20-warmup/200-timed/sync-bracketed. **Gate:** >5% VAE-portion win → quick win; else drop and
  record (books offer no specific support either way).
  **Dropped 2026-07-29 without measuring:** in the deployed configuration the VAE runs as
  TensorRT engines (`AutoencoderKLEngine` installed at `wrapper.py:2611-2625`); the eager
  PyTorch VAE exists only as the OOM fallback (`wrapper.py:2627-2657`), so a
  `channels_last` A/B on it has no production consumer. The PMPP layout principle itself
  is already noted under A2 item 6.
- **2d. FP8 provenance** (parallel with 2a; per user decision): reconstruct how
  `sdxl-turbo--fp8v3--h18b4bb48d936` was built — `engines/.../build_log.jsonl` (2026-07-25,
  2226.69 s), git history of `configs/profiling/profiling_fp8v3.yaml`, hash inputs. Fix config
  to resolve to the real engine or record params as unrecoverable; rebuild reproducibly; A/B
  FP8 vs FP16 (`unet_step` p50 + limiter-metrics ncu pass + output-quality check). Highest
  known-payoff path (2026-07-10 audit: FP8 GEMMs ≈45% of kernel time).

### B3. Conditional big-ticket items (each gated on B2)

- **3a. Fused `get_nn_feats` TRT plugin** (gated on 2a; 1–3 weeks, highest risk): single-launch
  online-(max,idx) kernel (never materializes `[B,N,M]`; simpler than flash attention — no
  softmax state), as custom ONNX op + registered TRT plugin. Must survive all three
  touchpoints (`_collect_fi_processors`, `refresh_fi_procs`, `diffusers_kvo_patch.py:68-140`)
  - engine rebuild; single call site limits blast radius. Parity: (i) standalone vs report
  §3.3 harness, 4 layer sizes, exact or documented fp16-order epsilon, checked against thesis
  Eq 3.2; (ii) engine-level frame-for-frame `_fi_cache_out` diff on fixed seed. Then A/B vs
  both FI-off and original engines — must recover most of 2a's delta or be dropped. Design
  spike first (repo has zero kernels; plugin reintroduces the MSVC/CUDA toolchain ADR-0001
  avoided).
- **3b. Double-buffered `_ipc_pack_buf` + async export** (gated on 2b; days + soak): 2-buffer
  ping-pong (prior art `_stock_noise_bufs`, allocated at pipeline.py:633 **and re-allocated at
  :775 in `_refresh_derived_tensors`** — the double-buffer impl must likewise survive
  re-init/resize) across **both** exporter-init paths; then flip **both** `export_sync`
  overrides (wrapper.py:1151 and :1209). PMPP §6.7 WAR-elimination is the formal
  justification. Extend `test_ipc_producer_stream.py`; **soak-test hours against a live
  receiver** (CUDA 719 class).
- **3c. FP8 adoption** — decision from 2d's data.

### B4. Hygiene

- `profile_ncu.py`: post-run `.ncu-rep` sanity check (blank `gpu__time_duration.sum` → fail
  loudly); capture guardrail prose in `scripts/profiling/README.md` or
  `docs/profiling/ncu_guardrails.md`.
- Factor shared `_pack_bgra(src, dst)` helper (kills the B1 duplication permanently).
- Close out the FP8 drift record (ADR or build-log note) regardless of the 3c decision.

---

## Verification

- **A1:** `.ncu-rep` passes the duration sanity check; every added metric appears in the export
  header; the register-vs-shared-mem question resolves one way or the other, in writing.
- **Sourcing:** every Part 1 claim carries book + section + PDF page, each actually read (no
  ToC-inferred citations); channels_last absence stated plainly.
- **Code claims:** every file:line in the new doc re-verified against the MCP code-search index
  before commit.
- **Consistency:** no number in the new doc contradicts the report without an explicit,
  reasoned correction; report amended in the same pass (A4) so the two never disagree silently.
  No `CUDA_LAUNCH_BLOCKING=1` timing ever quoted as a performance number.
- **B1:** `venv/Scripts/python -m pytest tests/unit -q` incl. the new pack test.
- **B2/B3 timing validity:** only `profiler_logs/*_stats.json` from runs with no ncu attached
  and no launch-blocking (~8.6× inflation); ≥10 min ncu timeouts, check for `.ncu-rep` on disk
  before declaring failure.
- **Repo conventions:** all git via `scripts/git/*.sh` wrappers (conventional ≤50-char
  subjects); Python via `venv/Scripts/python` (except PDF reads: system `python` has fitz).
- **Workstream A leaves `git status` showing only docs + `logs/` artifacts.**

## Risks

- Engine-hash sensitivity: scratch config differing on a non-target axis → differently-tactic'd
  engine → invalid A/B. Always diff full resolved configs.
- `use_feature_injection` defaults **true** on key absence — every ablation sets it explicitly.
- A1 metric names may silently drop — verify each resolves before interpreting.
- Real-time constraint: no added frame latency; re-verify after 3a/3b.
- 3b touches a boundary with a documented production incident (CUDA 719) — soak mandatory.
- 3a requires an MSVC + CUDA toolchain this package currently avoids.
- PMPP 5e ≠ the transcript's 4th ed. — cite the edition explicitly in the research doc.

## Recommended order

1. **A1** (occupancy ncu pass) and **B1** (quick-win commit) in parallel — both cheap.
2. **A2 → A3** (extraction, research doc); **A4** once A1's verdict lands (fold errata into the
   same report edit).
3. **B2a** (FI ablation gate) with **B2d** (FP8 provenance) in parallel; 2b/2c as capacity
   allows.
4. **B3** strictly per gates; FP8 (2d→3c) remains the highest-known-expected-value path.
