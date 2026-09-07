"""
Registry-gap regression guard for `_ensure_preprocessor_engines`.

Verification test for plan i-ve-noticed-that-in-graceful-kettle.md, Change 6.

The bug this guards against: `pose_tensorrt` requires a config-supplied, prebuilt
`preprocessor_params.engine_path` and has no self-build capability -- but it was absent from
`streamdiffusionTD/td_manager.py`'s `PREPROCESSOR_ENGINE_BUILD_REGISTRY`. That gap made
`_ensure_preprocessor_engines()` silently skip it at startup (`if preprocessor not in
build_registry: continue`, run *before* the engine_path was ever checked). Startup logged
nothing wrong; the failure only surfaced ~17 minutes later as a `FileNotFoundError` on every
single ControlNet frame, at ~22 fps.

The correct seam is the registry itself, not any one preprocessor: this test asserts that every
preprocessor requiring a prebuilt, config-supplied engine is either a registry key or in a short,
commented exempt set with a stated reason. It intentionally does NOT use "not a
SelfBuildingTRTPreprocessor subclass yet raises 'TensorRT engine not found'" as the predicate --
that reads as obviously right but produces two false positives:

  - `realesrgan_trt.RealESRGANProcessor` raises that exact string, but it is fully self-sufficient
    (_download_file -> _export_to_onnx -> _build_tensorrt_engine) and computes engine_path
    internally from models_dir, never from a config-supplied preprocessor_params.engine_path.
  - `temporal_net_tensorrt.TemporalNetTensorRTPreprocessor` takes a config-supplied engine_path,
    but checks it eagerly in __init__ and raises with the exact build command in the message --
    already the fail-fast behaviour Change 3 gives pose_tensorrt/depth_tensorrt. It fails loudly
    at wrapper-construction time (td_manager.py's create_wrapper_from_config call, which runs
    *before* _ensure_preprocessor_engines), never silently 17 minutes later.

So the correct predicate is: engine_path comes from `self.params.get("engine_path")` (i.e. is
config-supplied, checked lazily in an `engine` property) AND the class has no self-build method.
Today that set is exactly {depth_tensorrt, pose_tensorrt} -- both required to be registry keys.

`streamdiffusionTD/` is gitignored, so recursive Grep/ripgrep silently skips it (a real trap hit
while writing this plan -- see the plan's own note). Load td_manager.py by explicit file path
instead, mirroring the importlib.util.spec_from_file_location pattern already used in
tests/unit/test_cn_preprocessor_residency.py for the sibling Scripts/ mirror.

Run with: pytest tests/unit/test_preprocessor_engine_build_registry.py -v
"""

import importlib.util
import os

import pytest

# ---------------------------------------------------------------------------
# Preprocessors that require a prebuilt, config-supplied `engine_path` and cannot self-build.
# Every name here MUST be a key in PREPROCESSOR_ENGINE_BUILD_REGISTRY.
# ---------------------------------------------------------------------------
REQUIRES_PREBUILT_CONFIG_ENGINE = {
    "depth_tensorrt",
    "pose_tensorrt",
}

# Preprocessors that raise "TensorRT engine not found" (or similar) but are deliberately NOT in
# the set above -- see the module docstring for why each is a false positive under the naive
# "not SelfBuildingTRTPreprocessor" predicate.
EXEMPT_WITH_REASON = {
    "temporal_net_tensorrt": (
        "engine_path is config-supplied, but TemporalNetTensorRTPreprocessor.__init__ checks it "
        "eagerly and raises (with the build command) at wrapper-construction time -- before "
        "_ensure_preprocessor_engines ever runs. Already fail-fast; no registry entry needed."
    ),
    "realesrgan_trt": (
        "RealESRGANProcessor computes its engine_path internally from models_dir and is fully "
        "self-sufficient (_download_file -> _export_to_onnx -> _build_tensorrt_engine). It never "
        "reads a config-supplied preprocessor_params.engine_path, so the registry gap this test "
        "guards against cannot occur for it."
    ),
}


def _load_td_manager_module():
    """
    Load streamdiffusionTD/td_manager.py by explicit path.

    Not a normal import: the package is gitignored (kept out of git, per
    docs/adr/0002-touchdesigner-script-deployment-topology.md) but still present on disk in a
    dev checkout, so importlib.util.spec_from_file_location reaches it the same way
    test_cn_preprocessor_residency.py reaches the sibling Scripts/ mirror.
    """
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    td_manager_path = os.path.join(repo_root, "streamdiffusionTD", "td_manager.py")
    td_manager_path = os.path.normpath(td_manager_path)

    spec = importlib.util.spec_from_file_location("td_manager_under_test", td_manager_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"streamdiffusionTD/td_manager.py not found at {td_manager_path} -- dev-only module")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"streamdiffusionTD/td_manager.py failed to import ({e}) -- dev-only module, TD-side deps")

    return mod


def test_every_prebuilt_config_engine_preprocessor_is_registered():
    """
    Every preprocessor in REQUIRES_PREBUILT_CONFIG_ENGINE must be a key in
    PREPROCESSOR_ENGINE_BUILD_REGISTRY -- otherwise _ensure_preprocessor_engines silently skips
    it, and it fails on the first real ControlNet frame instead of at startup.

    Fails on the pre-fix tree (pose_tensorrt absent) and passes after Change 2 registers it --
    that red-to-green ordering is the whole point of this test; see Verification step 1 in the
    plan.
    """
    mod = _load_td_manager_module()
    registry = mod.PREPROCESSOR_ENGINE_BUILD_REGISTRY

    missing = sorted(name for name in REQUIRES_PREBUILT_CONFIG_ENGINE if name not in registry)

    assert not missing, (
        f"Preprocessor(s) {missing} require a prebuilt, config-supplied engine_path but have no "
        "entry in PREPROCESSOR_ENGINE_BUILD_REGISTRY (streamdiffusionTD/td_manager.py). "
        "_ensure_preprocessor_engines will silently skip them at startup, and they will raise "
        "FileNotFoundError on the first ControlNet frame instead -- exactly the pose_tensorrt/"
        "yolonas_pose.engine incident this test guards against. Add a registry entry."
    )


def test_exempt_preprocessors_are_not_accidentally_registered_too():
    """
    Sanity check on the exempt set itself: if someone "fixes" temporal_net_tensorrt or
    realesrgan_trt by adding a registry entry, that's harmless but pointless (see the reasons in
    EXEMPT_WITH_REASON) and signals the exemption comment should be revisited or removed rather
    than silently going stale.
    """
    mod = _load_td_manager_module()
    registry = mod.PREPROCESSOR_ENGINE_BUILD_REGISTRY

    unexpectedly_registered = sorted(name for name in EXEMPT_WITH_REASON if name in registry)

    assert not unexpectedly_registered, (
        f"{unexpectedly_registered} are both exempt (per EXEMPT_WITH_REASON) AND now registered "
        "in PREPROCESSOR_ENGINE_BUILD_REGISTRY. That's not a failure by itself, but it means the "
        "exemption reason in this test is stale -- update EXEMPT_WITH_REASON (or move the name "
        "into REQUIRES_PREBUILT_CONFIG_ENGINE) to match reality."
    )
