"""
Unit tests for the on/off resolver in streamdiffusion.tools.cuda_l2_cache.

Regression coverage for the bug where SDTD_L2_PERSIST was read into a module-level
constant at import time, so a value set after the module first imported (e.g. by
TouchDesigner's embedded Python) had no effect -- setup_l2_persistence kept running
unconditionally in TRT/Performance mode. The fix reads the env var at call time and
layers it under an explicit `enabled` kwarg (the wrapper's `l2_persist` config key) and
a mode-aware default. All tests are CPU-only: reserve_l2_persisting_cache (the only
CUDA-touching call) is monkeypatched to a spy, matching the repo convention of patching
the real imported module object (see test_diagnostics.py).
"""

from streamdiffusion.tools import cuda_l2_cache


def _spy_reserve(monkeypatch, return_value=True):
    calls = []

    def _fake(persist_mb=None):
        calls.append(persist_mb)
        return return_value

    monkeypatch.setattr(cuda_l2_cache, "reserve_l2_persisting_cache", _fake)
    return calls


class TestSetupL2PersistenceResolver:
    def test_explicit_enabled_false_short_circuits(self, monkeypatch):
        calls = _spy_reserve(monkeypatch)
        result = cuda_l2_cache.setup_l2_persistence(object(), enabled=False, acceleration="tensorrt")
        assert result is False
        assert calls == []  # Tier 1 must never be attempted when disabled

    def test_env_zero_disables_when_enabled_unset(self, monkeypatch):
        calls = _spy_reserve(monkeypatch)
        monkeypatch.setenv("SDTD_L2_PERSIST", "0")
        result = cuda_l2_cache.setup_l2_persistence(object(), acceleration="none")
        assert result is False
        assert calls == []

    def test_env_one_enables_when_enabled_unset(self, monkeypatch):
        calls = _spy_reserve(monkeypatch, return_value=True)
        monkeypatch.setenv("SDTD_L2_PERSIST", "1")
        result = cuda_l2_cache.setup_l2_persistence(object(), acceleration="tensorrt")
        assert result is True
        assert calls == [64]  # default SDTD_L2_PERSIST_MB

    def test_tensorrt_mode_defaults_off_when_config_and_env_both_unset(self, monkeypatch):
        """The reported bug: TD's Performance profile (acceleration='tensorrt') never sets
        SDTD_L2_PERSIST (TD's embedded Python can't reach shell env vars), and previously
        L2_PERSIST_ENABLED was frozen True at import time regardless. It must now default off."""
        calls = _spy_reserve(monkeypatch)
        monkeypatch.delenv("SDTD_L2_PERSIST", raising=False)
        result = cuda_l2_cache.setup_l2_persistence(object(), acceleration="tensorrt")
        assert result is False
        assert calls == []

    def test_non_tensorrt_mode_defaults_on_when_config_and_env_both_unset(self, monkeypatch):
        """Legacy behavior preserved for torch/xformers, where Tier 1/2 are not inert."""
        calls = _spy_reserve(monkeypatch, return_value=True)
        monkeypatch.delenv("SDTD_L2_PERSIST", raising=False)
        result = cuda_l2_cache.setup_l2_persistence(object(), acceleration="xformers")
        assert result is True
        assert calls == [64]

    def test_explicit_enabled_true_wins_over_disabling_env(self, monkeypatch):
        """Config wins over env (the inverse of gpu_profiler's precedence) -- TD cannot set
        env vars, so an explicit l2_persist: true in config must not be silently vetoed by a
        stale SDTD_L2_PERSIST=0 left in the process environment."""
        calls = _spy_reserve(monkeypatch, return_value=True)
        monkeypatch.setenv("SDTD_L2_PERSIST", "0")
        result = cuda_l2_cache.setup_l2_persistence(object(), enabled=True, acceleration="tensorrt")
        assert result is True
        assert calls == [64]

    def test_persist_mb_env_overrides_default(self, monkeypatch):
        calls = _spy_reserve(monkeypatch, return_value=True)
        monkeypatch.setenv("SDTD_L2_PERSIST_MB", "32")
        cuda_l2_cache.setup_l2_persistence(object(), enabled=True, acceleration="tensorrt")
        assert calls == [32]

    def test_explicit_persist_mb_wins_over_env(self, monkeypatch):
        calls = _spy_reserve(monkeypatch, return_value=True)
        monkeypatch.setenv("SDTD_L2_PERSIST_MB", "32")
        cuda_l2_cache.setup_l2_persistence(object(), enabled=True, persist_mb=16, acceleration="tensorrt")
        assert calls == [16]

    def test_tier1_failure_skips_tier2(self, monkeypatch):
        _spy_reserve(monkeypatch, return_value=False)
        pin_calls = []
        monkeypatch.setattr(cuda_l2_cache, "pin_hot_unet_weights", lambda *a, **k: pin_calls.append((a, k)) or 0)
        result = cuda_l2_cache.setup_l2_persistence(object(), enabled=True, tier2=True, acceleration="none")
        assert result is False
        assert pin_calls == []

    def test_tier2_disabled_by_default(self, monkeypatch):
        _spy_reserve(monkeypatch, return_value=True)
        pin_calls = []
        monkeypatch.setattr(cuda_l2_cache, "pin_hot_unet_weights", lambda *a, **k: pin_calls.append((a, k)) or 0)
        monkeypatch.delenv("SDTD_L2_PERSIST_TIER2", raising=False)
        cuda_l2_cache.setup_l2_persistence(object(), enabled=True, acceleration="none")
        assert pin_calls == []

    def test_tier2_explicit_true_invokes_pin(self, monkeypatch):
        _spy_reserve(monkeypatch, return_value=True)
        pin_calls = []
        unet = object()
        monkeypatch.setattr(cuda_l2_cache, "pin_hot_unet_weights", lambda u, *a, **k: pin_calls.append((u, k)) or 0)
        cuda_l2_cache.setup_l2_persistence(unet, enabled=True, tier2=True, acceleration="none")
        assert len(pin_calls) == 1
        assert pin_calls[0][0] is unet


class TestEnvReaders:
    def test_env_enabled_tri_state(self, monkeypatch):
        monkeypatch.delenv("SDTD_L2_PERSIST", raising=False)
        assert cuda_l2_cache._env_enabled() is None
        monkeypatch.setenv("SDTD_L2_PERSIST", "1")
        assert cuda_l2_cache._env_enabled() is True
        monkeypatch.setenv("SDTD_L2_PERSIST", "0")
        assert cuda_l2_cache._env_enabled() is False

    def test_env_persist_mb_default_and_override(self, monkeypatch):
        monkeypatch.delenv("SDTD_L2_PERSIST_MB", raising=False)
        assert cuda_l2_cache._env_persist_mb() == 64
        monkeypatch.setenv("SDTD_L2_PERSIST_MB", "128")
        assert cuda_l2_cache._env_persist_mb() == 128

    def test_env_tier2_default_and_override(self, monkeypatch):
        monkeypatch.delenv("SDTD_L2_PERSIST_TIER2", raising=False)
        assert cuda_l2_cache._env_tier2() is False
        monkeypatch.setenv("SDTD_L2_PERSIST_TIER2", "1")
        assert cuda_l2_cache._env_tier2() is True
