# Verification of `SDXL-Turbo_Research.md` against SDTD_040_Beta — 2026-08-16

`docs/plans/SDXL-Turbo_Research.md` is an external-literature research document (untracked at
the time of this pass, no history) that states a set of "mandatory" contracts for SDXL-Turbo —
`EulerAncestralDiscreteScheduler` + `timestep_spacing="trailing"`, `guidance_scale=0.0`, 1–4
steps, 512², fixed seed — and describes "StreamDiffusion internals" drawn from the upstream
cumulo-autumn repo, not from this fork.

The live pipeline runs `stabilityai/sdxl-turbo` (`StreamDiffusionTD/td_config.yaml:2`), so this
pass checked every claim in the research doc against `src/streamdiffusion`, the repo's own
read-only probe (`scripts/probe_schedule.py`), direct scheduler arithmetic in the project venv,
and the rest of `docs/`. **Headline: the `timestep_spacing="trailing"` requirement is already
satisfied at the live config — but by coincidence, not by design, and that coincidence is
fragile.** Four small defects fell out of the audit; all four are fixed in this pass and none of
them changes the live rendered frame (see "Effect on live output" below).

## Verdict table

| Research-doc guideline | Status in this fork | Evidence |
|---|---|---|
| `timestep_spacing="trailing"` mandatory | **Satisfied de facto, not by design** | see "The trailing finding" below |
| `EulerAncestralDiscreteScheduler` mandatory | **N/A — correctly not applicable** | only `LCMScheduler`/`TCDScheduler` exist (`pipeline.py:8,275-281`); zero hits for `EulerAncestral\|EulerDiscrete\|DPMSolver` in `src/`. The LCM path never calls `scheduler.step()` — it hand-rolls add_noise + a consistency step (`pipeline.py:1049-1056`, `1058-1079`) |
| `guidance_scale = 0.0` for Turbo | **Not enforced; live config diverges deliberately** | `td_config.yaml:15` `guidance_scale: 1.01` + `:30` `cfg_type: "self"` → RCFG-self active (`pipeline.py:1319` gates the combine on `guidance_scale > 1.0`); profiling runs pinned at 1.045 (`docs/profiling/fp8_fi_gates_2026-07-29.md:14`) |
| Model-type-aware scheduler/CFG dispatch | **Absent** | `is_turbo` is computed (`model_detection.py`) but its only behavioural consumer is FP8 calibration (`wrapper.py:2945`) |
| txt2img must be `cfg_type="none"` | **Implemented** | `wrapper.py:615-617` raises |
| Steps 1–4 | **Satisfied in config, unvalidated in code** | live `t_index_list: [4, 8, 12]` = 3 UNet calls; every profiled config is 1–2 steps; no length bound exists anywhere |
| img2img `strength`, `steps*strength ≥ 1` guard | **Absent by design** | no `strength` param exists; `t_index_list[0]` is the strength dial (`pipeline.py:1403`) |
| Negative prompts inert | **True, and broader than the doc says** | inert for `none` *and* `self`; the live-update path passes `do_classifier_free_guidance=False` (`stream_parameter_updater.py:685`), so a negative-prompt-only OSC update is a silent no-op |
| Fixed `init_noise` reused every frame | **Implemented** | drawn once in `prepare()` (`pipeline.py:651-654`) from a seeded generator (`:453-456`), reused at `:1403`; `stock_noise` ping-pong at `:1427-1441` |
| `seed = -1` → per-frame flicker | **True, but it is a TD-layer feature** | backend `wrapper.py:735-736` randomises **once at init**; true per-frame re-randomisation is `Scripts/streamdiffusionTD__Text__td_manager__td.py:189-202` + `:713-722` |
| Stochastic Similarity Filter (0.98 / 10) | **Implemented, exact defaults, off in live config** | `pipeline.py:434-437`, `image_filter.py:8-22` — **deviates from the cited paper: pixel-space MSE, not latent cosine similarity** (documented in the class docstring) |
| TAESD / taesdxl tiny VAE | **Implemented, default on, SDXL-aware** | `wrapper.py:2317-2323` picks `taesdxl` when `is_sdxl` |
| 512×512 native | **Matches, but no Turbo-specific validation** | `td_config.yaml:9-10`; `384x640` engines also exist on disk. `docs/profiling/unet_ncu_roofline_2026-05-24.md:43-52` treats 512² as the *cause* of GPU under-utilisation, not a quality target |
| TensorRT FP16 default, INT8 risky | **Moot — no INT8 path exists** | only int8 token in the TRT tree is a dtype-map entry (`utilities.py:464`). FP16/FP8 flags only. The doc's FP16 conclusion is also superseded by `docs/adr/0003-fp8-engine-adoption.md` (fp8 6.2% faster per `docs/profiling/fp8_fi_gates_2026-07-29.md`) — though the live config currently runs `fp8: false` |
| SDXL-Lightning / Hyper-SD as alternatives | **Not referenced anywhere in the backend** | grep hits only inside `calibration_prompts_sdxl.txt` |
| SDXL first-class (dual encoders, pooled embeds, `time_ids`) | **Implemented** | `pipeline.py:515-583`, `:1156`, `txt2img_sd_turbo` at `:1719` — this repo *is* one of the forks the doc says the support "lives in" |

## The trailing finding (the crux)

The doc's #1 requirement is met, but not the way the doc assumes. Three facts, all verified:

1. **The model ships `trailing` and this fork inherits it.**
   `sdxl-turbo/scheduler/scheduler_config.json` declares `"_class_name":
   "EulerAncestralDiscreteScheduler"`, `"timestep_spacing": "trailing"`.
   `_initialize_scheduler` does `LCMScheduler.from_config(self.pipe.scheduler.config, ...)`
   (`pipeline.py:276`), so the constructed `LCMScheduler.config.timestep_spacing == "trailing"`
   even with the default `sampler: "normal"`. Verified by direct load.

2. **…but `LCMScheduler` ignores `timestep_spacing` entirely.** Confirmed empirically: building
   it with `leading` vs `trailing` yields the *identical* grid `[999, 979, 959, 939, 919]`. This
   is exactly what `pipeline.py:286-288` documents, and why `_get_spaced_timesteps` (`:283-302`)
   re-implements Lin et al. Table 2 by hand.

3. **The LCM native grid *is* the trailing grid at the live settings.** With the forced
   `original_inference_steps = 100` (`pipeline.py:273`) and `num_inference_steps = 50`, the
   native grid is `999 − 20·i` — bit-identical to manual trailing, and it starts at **t = 999**.
   Verified across step counts: native == trailing at S ∈ {1, 2, 4, 10, 20, 25, 50}; they
   **diverge** at S ∈ {3, 15, 30}. `leading` (`sampler: "ddim"`) starts at **980**, i.e. exactly
   the doc's washed-out failure mode — and it is a selectable option in the TD `Sampler` menu.

So live behaviour is correct, but it rests on `S=50` dividing `original_inference_steps=100`.
Nothing in the code states or defends that invariant.

## Corollary: the Sampler menu is nearly inert at the live settings

Measured grids (exact LCM algorithm from the installed `scheduling_lcm.py`,
`original_inference_steps=100`):

| Sampler | realised t at `[4, 8, 12]`, S=50 | vs `normal` |
|---|---|---|
| `normal` (LCM native) | 919 / 839 / 759 | baseline |
| `sgm_uniform` (trailing) | 919 / 839 / 759 | **bit-identical** |
| `beta` | — | no-op, self-documented (`pipeline.py:263`) |
| `karras` | — | no-op, self-documented (`pipeline.py:264`) |
| `simple` (linspace) | 917 / 836 / 754 | Δ ≤ 5 → imperceptible |
| `ddim` (leading) | 900 / 820 / 740 | Δ ≈ 19 → small but real |

At S=50 the grid pitch is 20, so any spacing strategy can shift each t by at most one pitch. The
doc's leading-vs-trailing failure is an S≤4 phenomenon (at S=4: trailing starts 999, leading
750). Divergence grows as S shrinks — at S=15 with proportionally rescaled indices, `ddim` lands
73–81 timesteps below native while `sgm_uniform` stays within 7.

`sampler` is also **not live-updatable** — absent from both `UPDATER_PARAM_NAMES` and
`PARAM_NAMES`, it is construction-time only (`td_config.yaml:46` → `prepare()`), so changing it
needs a reload.

This downgrades defect 1 (below) from "wrong output" to "a user-selected setting is silently
discarded on the next live step-count change." Worth fixing — it is 3 lines, reusing an existing
helper — but it was never a rendering bug at S=50.

Secondary correctness note: `c_skip ≈ 3e-9`, `c_out ≈ 1.0` at every timestep in use (measured via
`scripts/probe_schedule.py`), so the LCM consistency step numerically degenerates to the raw x₀
prediction the ADD student expects. The doc's implied concern about the LCM boundary-condition
math is a non-issue here.

## CFG rationale (documented, not changed)

`guidance_scale: 1.01` + `cfg_type: "self"` on SDXL-Turbo contradicts the doc's "CFG must be 0",
but it is the deliberate RCFG-self technique the doc itself acknowledges for img2img. At
`gs=1.01` the RCFG term contributes ~1% — nearly a no-op that still costs the `stock_noise`
bookkeeping — and `pipeline.py:1319` gates the combine on `guidance_scale > 1.0`, so `gs=1.0`
would disable it entirely. **This pass does not change `td_config.yaml` and does not add a
warning log**, per explicit instruction; this section is the record of why the divergence is
intentional.

## Defects fixed this pass

1. **Live `num_inference_steps` change silently discarded the spacing override.**
   `stream_parameter_updater.py` (previously `:380-382`) called `scheduler.set_timesteps(...)`
   directly instead of `param_schema.materialise_timestep_grid(...)` — the only direct
   `set_timesteps` call left in `src/` outside the helper itself. For
   `sampler ∈ {simple, sgm_uniform, ddim}` the grid built in `prepare()` reverted to the LCM
   native one on the first runtime step change — precisely the failure mode
   `param_schema.py`'s `materialise_timestep_grid` docstring warns about. **Fix:** route through
   the shared helper (now imported alongside the module's other `param_schema` symbols).

2. **`is_turbo` discriminator in `model_detection.py` was wrong in both directions.** The old
   comment claimed "Base SDXL has `time_cond_proj_dim` … Turbo has it `None`". That field is the
   *LCM-distillation* guidance-embedding dim; stock SDXL-Base-1.0 also has it `null`, so
   SDXL-Base was flagged `is_turbo=True`. Conversely the non-SDXL UNet branch never set
   `is_turbo` at all, so `sd-turbo` (verified locally: `addition_embed_type: null`,
   `cross_attention_dim: 1024`) was classified `SD2.1`, `is_turbo=False`. The duplicated
   ControlNet branch had the identical bug. **Fix:** added `_detect_turbo_from_scheduler(pipe)`,
   which discriminates via the *source* pipeline's scheduler class + `timestep_spacing`
   (`EulerAncestralDiscreteScheduler` + `trailing` ⇒ Turbo), evaluated before StreamDiffusion
   swaps in its own `LCMScheduler`/`TCDScheduler`. Wired into all three branches (SDXL UNet,
   non-SDXL UNet, ControlNet); `time_cond_proj_dim is not None` is kept as what it actually
   means (LCM-distilled), not repurposed as a Turbo signal.

3. **`examples/txt2img/spacing_compare.py` rested on a false premise.** The "19 mod 20" /
   `original_inference_steps=50` assumption was baked into the docstring, the `STEP_COUNTS`
   comment, the `sgm_uniform` label, the `on_grid` predicate (`int(t) % 20 == 19`), and the
   `is_divisor = 50 % S == 0` test. The pipeline forces `original_inference_steps = 100`, so the
   distillation grid is `t ≡ 9 (mod 10)` and native==trailing coincidence is keyed to divisors of
   100, not 50. **Fix:** corrected all five sites to the `mod 10` / `divides 100` facts and added
   a docstring note that the script has not been run. **Not run as part of this pass** — the
   empirical A/B stays out of scope.

4. **Dead scheduler API.** `set_scheduler` and `_uses_lcm_logic` in `pipeline.py` had zero
   callers — confirmed via `find_connections` (no `direct_callers` for either) and a repo-wide
   grep matching only the two `def` lines. `set_scheduler` would additionally have left
   `timesteps`/`sub_timesteps`/`c_skip`/`c_out`/alpha-beta stale if it were ever called (it
   doesn't recompute sub-timesteps or CDF-style boundary coefficients the way `prepare()` does).
   **Fix:** deleted both methods.

## Effect on live output

**All four fixes leave the live rendered frame bit-identical.** None touches the path the
current config (`sdxl-turbo`, `sampler: "normal"`, `S=50`, `t=[4,8,12]`, `fp8: false`) executes.

| Fix | Live effect now | Only changes output when… | Magnitude then |
|---|---|---|---|
| 1 — spacing helper in updater | **None.** `materialise_timestep_grid` applies its override only for `sampler ∈ {simple, sgm_uniform, ddim}`; at `"normal"` it degenerates to the exact two lines it replaces | sampler is set to `simple`/`sgm_uniform`/`ddim` at build time **and** `num_inference_steps` is later changed live over OSC | Today the sampler choice is silently discarded at that moment; after this fix, it persists. Measured Δ is small: at S=50 `sgm_uniform` is bit-identical to `normal`, `simple` differs by ≤5, `ddim` by ≈19 timesteps. Grows at lower S |
| 2 — `is_turbo` | **None, twice over.** Only consumer `wrapper.py:2945` is inside `if fp8:` and live `fp8: false`; and sdxl-turbo already resolved `True` before the fix, unchanged by it | an **fp8 engine build** for `sd-turbo` (False→True) or `sdxl-base-1.0` (True→False) | Calibration guidance flips 7.5↔0.0, feeding the capture `pipe()` call (`fp8_quantize.py:608`) → different activations → different FP8 scales → different engine. Materialises only on rebuild |
| 3 — `spacing_compare.py` | **None.** Standalone example, never imported, never run | never (offline analysis script) | — |
| 4 — delete dead API | **None.** Zero callers in `StreamDiffusion/` *and* zero in the TD `Scripts/` layer | never | — |

**No engine invalidation.** `is_turbo` is not part of the TRT engine path or cache key — its
remaining uses are logging and diagnostics dicts. Cached engines stay valid; no rebuild is
forced by this pass.

**The only knob that would change live output is CFG**, and it is deliberately left alone (see
"CFG rationale" above).

## Logged, not fixed (out of scope)

- Negative-prompt-only live update is a no-op (`stream_parameter_updater.py:456,466,685`).
- Dead config key `use_taesd` in `configs/sdxl_multicontrol.yaml.example:40` — `config.py` only
  reads `use_tiny_vae`.
- `configs/td_config.yaml.example:89-92` recommends `enable_similar_image_filter: true` @ 0.9996
  while the live config has it `false` @ 0.9998.

## Where the research document is wrong about this fork

Its "StreamDiffusion internals" section describes upstream cumulo-autumn. Stale/misleading here:
"first-class SDXL-Turbo support lives in forks" (this *is* that fork); Euler-a framed as
"mandatory" for a pipeline that never runs a diffusers sampler loop; `seed = -1` presented as a
backend behaviour (it is TD-layer); the INT8 caution (no INT8 path exists); and the FP16
recommendation (superseded by the ADR-0003 fp8 measurement). The `t_index_list=[0,16,32,45]` and
`seed=2` defaults it quotes *are* this fork's defaults (`param_schema.py:78-93`) — the live
config simply overrides them to `[4, 8, 12]` and `4717776`. The `guidance_rescale`/Karras/
high-order-solver "dead knobs" are correctly absent — there is nothing to remove.

## Reproduction

Run from `D:\dev\SDTD_040_Beta\StreamDiffusion`:

1. **Schedule unchanged on the live config** (regression guard for defect 1):
   `venv/Scripts/python.exe scripts/probe_schedule.py`
   Expect exactly: `sampler=normal steps=50`, `t_index_list=[4, 8, 12]`,
   `t = 919 / 839 / 759`, `a_sqrt = 0.1076 / 0.1608 / 0.2285`, `do_add_noise EFFECTIVE=True`.

2. **Spacing override now survives a live step change**: construct with
   `sampler="sgm_uniform"`, call `update_stream_params(num_inference_steps=15)`, and assert
   `stream.timesteps[:4] == [999, 932, 866, 799]` (manual trailing) rather than
   `[999, 939, 869, 799]` (LCM native). Those two grids differ only at S ∉ divisors of 100 —
   S=15 is the discriminating case; testing at S=50 would pass either way.

3. **`is_turbo` detection**: `sd-turbo → True`, `sdxl-turbo → True`, `sdxl-base-1.0 → False`
   (`tests/unit/test_model_detection.py`).

4. **No behavioural regression in the app**: launch the TD pipeline unchanged and confirm the
   `[schedule:prepare]` log line still reports `scheduler=LCMScheduler sampler=normal steps=50
   t_index_list=[4, 8, 12]`.

5. **Existing suite**: `venv/Scripts/python.exe -m pytest tests/unit -q`.
