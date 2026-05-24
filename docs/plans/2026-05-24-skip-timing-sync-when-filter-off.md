# Skip GPU timing sync entirely when the similar image filter is off

## Context

`pipeline.py __call__` maintains `inference_time_ema` via CUDA timing events to power the
similar-image-filter sleep heuristic: when a frame is similar enough to skip, the pipeline
sleeps for `inference_time_ema * similar_filter_sleep_fraction` so the GPU doesn't spin idle.

The P2 optimization (already in place) throttled the per-frame `end.synchronize()` to every
16th frame, eliminating ~15/16 of the per-frame host stalls. The question: **can we skip
synchronization entirely when the similar filter is off?**

**Yes.** Grep across the whole file confirms `inference_time_ema` is consumed at exactly one
call site — `pipeline.py:1059`, inside the `if self.similar_image_filter:` branch at `:1055`.
When the filter is off, the EMA has no reader; the entire `start.record()` / `end.record()` /
`end.synchronize()` / EMA-update path is dead work.

---

## Implementation (completed 2026-05-24)

Two edits in `src/streamdiffusion/pipeline.py`, both in `__call__`:

### 1. Gate `start.record()` behind the filter flag (`:1031-1034`)

```python
start = self._timing_start
end = self._timing_end
if self.similar_image_filter:
    start.record()
```

### 2. Gate `end.record()` + sync + EMA update (`:1098-1108`)

```python
# P2: the timing path exists only to maintain inference_time_ema, which is consumed
# solely by the similar-filter sleep heuristic (see the similar_image_filter branch
# above). When the filter is off the EMA has no reader, so skip the records AND the
# blocking end.synchronize() entirely. When on, retain the 16-frame sample cadence
# (eliminates ~15/16 per-frame host stalls).
# Grounding: CUDA HB §6.1 — per-frame device sync ~100 µs vs ~3.4 µs amortized.
if self.similar_image_filter:
    end.record()
    self._sync_counter += 1
    if self._sync_counter % 16 == 0:
        end.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
```

Both `record()` calls are gated identically, so `start.elapsed_time(end)` always sees a matched
pair. The gate is dynamic (runtime check of `self.similar_image_filter`) because the flag is
toggleable at runtime via `enable/disable_similar_image_filter()`.

### Runtime-toggle behavior

When toggled on after running with the filter off, `inference_time_ema` is `0` (or stale) and
the sleep warms up over the first 16 non-skipped frames — identical to cold-start behavior.

### Skip-frame path unaffected

Frames skipped by the filter return early at `:1060`, before `end.record()`, as today.
`_sync_counter` still counts only non-skipped frames.

---

## Verification

- **Filter OFF:** confirm `end.synchronize()` is unreachable with the default config. FPS
  should be unchanged or marginally improved.
- **Filter ON:** enable, feed near-static frames, confirm skip + sleep still fires and
  `inference_time_ema` populates. No `elapsed_time` exceptions.
- **Toggle mid-run:** enable then disable during a live run; confirm no crash.
