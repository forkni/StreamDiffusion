"""
Regression test for IP-Adapter/FaceID scoping of the TensorRT engine cache key.

``EngineManager.get_engine_path`` (acceleration/tensorrt/engine_manager.py) only folds
``is_faceid`` / ``ipadapter_tokens`` into the cache key inside the
``if engine_type == EngineType.UNET:`` block (:160-178). VAE-encoder and VAE-decoder engines
are IP-Adapter-agnostic (they never see cross-attention weights), so those two params must be
accepted but ignored for ``EngineType.VAE_ENCODER`` / ``EngineType.VAE_DECODER`` — otherwise
toggling FaceID or changing the token count would force a redundant VAE rebuild.

This is also the B3 regression test: bumping the marker from ``--fid`` to ``--fid2`` (a plain
generation bump, done because B2's LoRA fusion changes the exported UNet graph and would
otherwise let a stale ``--fid`` engine be silently reused) must still fork the UNet cache key
whenever ``is_faceid`` flips, independent of the literal marker string. The UNet prefix is
hashed (``engine_manager.py:240-247``) before becoming a directory name, so this test compares
*paths*, not literal substrings — pinning ``--fid2`` itself would break the moment the marker
is bumped again.

Reuses the ``_make_engine_manager`` / GPU-less-import pattern from test_engine_path_length.py.

Run with: pytest tests/unit/test_engine_path_ipadapter_suffixes.py -v
"""

from pathlib import Path

import pytest

try:
    from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager, EngineType

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.engine_manager not importable",
)

_ENGINE_DIR = r"C:\Users\deswh\Documents\sdtd040\StreamDiffusion\engines\td"

_BASE_KWARGS = {
    "model_id_or_path": "stabilityai/sdxl-turbo",
    "max_batch_size": 4,
    "min_batch_size": 1,
    "mode": "img2img",
    "use_tiny_vae": True,
}

# Extra filenames beyond UNET (test_engine_path_length.py's _make_engine_manager only wires
# UNET), so VAE_ENCODER/VAE_DECODER paths can be built too.
_FILENAMES = {
    EngineType.UNET: "unet.engine",
    EngineType.VAE_ENCODER: "vae_encoder.engine",
    EngineType.VAE_DECODER: "vae_decoder.engine",
}


def _make_engine_manager(engine_dir: str = _ENGINE_DIR) -> EngineManager:
    """Build an EngineManager without running __init__'s heavy compile-fn imports."""
    em = EngineManager.__new__(EngineManager)
    em.engine_dir = Path(engine_dir)
    em._configs = {etype: {"filename": fname} for etype, fname in _FILENAMES.items()}
    return em


class TestUnetPathForksOnFaceIdAndTokens:
    def test_unet_path_changes_when_faceid_toggled(self):
        em = _make_engine_manager()
        path_regular = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=False, **_BASE_KWARGS)
        path_faceid = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, **_BASE_KWARGS)

        assert path_regular != path_faceid
        assert path_regular.parent != path_faceid.parent

    def test_unet_path_changes_with_ipadapter_tokens(self):
        em = _make_engine_manager()
        path_4 = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, ipadapter_tokens=4, **_BASE_KWARGS)
        path_77 = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, ipadapter_tokens=77, **_BASE_KWARGS)

        assert path_4 != path_77

    def test_unet_path_is_deterministic(self):
        """Same FaceID/token config -> same path, so a previously-built engine is reused."""
        em = _make_engine_manager()
        kwargs = dict(_BASE_KWARGS, engine_type=EngineType.UNET, is_faceid=True, ipadapter_tokens=4)

        path_a = em.get_engine_path(**kwargs)
        path_b = em.get_engine_path(**kwargs)

        assert path_a == path_b


class TestVaePathsIgnoreIpAdapterFlags:
    def test_vae_encoder_path_unaffected_by_faceid(self):
        em = _make_engine_manager()
        path_regular = em.get_engine_path(engine_type=EngineType.VAE_ENCODER, is_faceid=False, **_BASE_KWARGS)
        path_faceid = em.get_engine_path(engine_type=EngineType.VAE_ENCODER, is_faceid=True, **_BASE_KWARGS)

        assert path_regular == path_faceid

    def test_vae_decoder_path_unaffected_by_faceid(self):
        em = _make_engine_manager()
        path_regular = em.get_engine_path(engine_type=EngineType.VAE_DECODER, is_faceid=False, **_BASE_KWARGS)
        path_faceid = em.get_engine_path(engine_type=EngineType.VAE_DECODER, is_faceid=True, **_BASE_KWARGS)

        assert path_regular == path_faceid

    def test_vae_paths_unaffected_by_ipadapter_tokens(self):
        em = _make_engine_manager()
        path_4 = em.get_engine_path(
            engine_type=EngineType.VAE_DECODER, is_faceid=True, ipadapter_tokens=4, **_BASE_KWARGS
        )
        path_77 = em.get_engine_path(
            engine_type=EngineType.VAE_DECODER, is_faceid=True, ipadapter_tokens=77, **_BASE_KWARGS
        )

        assert path_4 == path_77

    def test_vae_encoder_and_decoder_paths_differ_from_each_other(self):
        """Sanity check that the two VAE engine types don't collide via a shared filename bug."""
        em = _make_engine_manager()
        path_enc = em.get_engine_path(engine_type=EngineType.VAE_ENCODER, is_faceid=True, **_BASE_KWARGS)
        path_dec = em.get_engine_path(engine_type=EngineType.VAE_DECODER, is_faceid=True, **_BASE_KWARGS)

        assert path_enc != path_dec
