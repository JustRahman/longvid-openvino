"""Tests for the tiled decoder. Run:  OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests -q

Uses a toy decoder and a tiny random-weight LTX VAE (same architecture, few channels), so it runs in seconds.
"""
import copy
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLLTXVideo

from longvid_ov import metrics
from longvid_ov.bench_worker import convert_decoder
from longvid_ov.tiled_decode import MODES, OVDecoder, TiledDecoder, TilingConfig, blend, blend_weights

ROOT = Path(__file__).resolve().parent.parent
SMALL_TILES = dict(tile_sample_min_height=128, tile_sample_min_width=128,
                   tile_sample_stride_height=96, tile_sample_stride_width=96)  # temporal: defaults 16 / 8
LATENT_SHAPE = (1, 8, 5, 5, 8)  # -> video [1, 3, 33, 160, 256]: 2x3 spatial tiles, 4 temporal chunks
VIDEO_SHAPE = (1, 3, 33, 160, 256)


def toy_decoder(z):
    """Pointwise stand-in with the real shape rule: latent frame 0 -> 1 frame, others -> 8; pixels x32.
    Each output pixel depends on one latent pixel only, so a correct tiling must reproduce it exactly."""
    v = z[:, :3]
    v = np.concatenate([v[:, :, :1], np.repeat(v[:, :, 1:], 8, axis=2)], axis=2)
    return np.repeat(np.repeat(v, 32, axis=3), 32, axis=4)


@pytest.fixture(scope="module")
def latents():
    return np.random.default_rng(0).standard_normal(LATENT_SHAPE).astype(np.float32)


@pytest.fixture(scope="module")
def vae():
    torch.manual_seed(0)
    ch = (8, 16, 32, 32)
    return AutoencoderKLLTXVideo(latent_channels=LATENT_SHAPE[1], block_out_channels=ch, decoder_block_out_channels=ch,
                                 layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1, 1)).eval()


@pytest.fixture(scope="module")
def ov_decoder(vae, tmp_path_factory):
    ir = tmp_path_factory.mktemp("ir") / "vae_decoder.xml"
    # ARM: OpenVINO has no fast 5D convolution there (step 1), so export Conv3d as Conv2d like the local runs do
    convert_decoder(copy.deepcopy(vae), str(ir), conv3d_as_2d=platform.machine() in ("arm64", "aarch64"))
    return OVDecoder(ir, threads=4)


def test_blend_weights_sum_to_one():
    for extent in (1, 2, 8, 64):
        w_prev, w_new = blend_weights(extent)
        np.testing.assert_allclose(w_prev + w_new, 1.0, atol=1e-7)
        assert w_prev[0] == 1.0 and w_new[0] == 0.0  # the overlap starts as the previous tile
    # blending two equal tiles changes nothing; an edge tile shorter than the overlap is handled
    a = np.full((1, 1, 4, 4, 4), 3.0, np.float32)
    for axis in (2, 3, 4):
        np.testing.assert_allclose(blend(a, a.copy(), 8, axis), a, atol=1e-6)


@pytest.mark.parametrize("rule", ["drop_first", "diffusers"])
@pytest.mark.parametrize("mode", MODES)
def test_constant_stays_constant(mode, rule):
    """End-to-end form of 'weights sum to 1': tiling a constant video gives the same constant, no dips at seams."""
    dec = TiledDecoder(toy_decoder, TilingConfig(first_frame_rule=rule, **SMALL_TILES))
    out = dec.decode(np.ones(LATENT_SHAPE, np.float32), mode)
    assert out.shape == VIDEO_SHAPE
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_pointwise_decoder_is_reproduced_exactly(mode, latents):
    """Every tile lands in the right place (space and time), including the causal first frame of each chunk."""
    dec = TiledDecoder(toy_decoder, TilingConfig(**SMALL_TILES))
    np.testing.assert_allclose(dec.decode(latents, mode), toy_decoder(latents), atol=1e-6)
    assert dec.calls == {"none": 1, "spatial": 6, "temporal": 4, "spatial+temporal": 24}[mode]


def test_diffusers_first_frame_rule_is_one_frame_late(latents):
    """Documents the diffusers 0.39 behaviour that `drop_first` fixes: after the first chunk, frame i shows frame i-1."""
    ref = toy_decoder(latents)
    out = TiledDecoder(toy_decoder, TilingConfig(first_frame_rule="diffusers", **SMALL_TILES)).decode(latents, "temporal")
    np.testing.assert_allclose(out[:, :, :10], ref[:, :, :10], atol=1e-6)  # first chunk + first blended frame are fine
    np.testing.assert_allclose(out[:, :, 17:], ref[:, :, 16:-1], atol=1e-6)  # later frames: the previous frame
    assert np.abs(out - ref).max() > 0.1


@pytest.mark.parametrize("mode", MODES)
def test_shapes_openvino(mode, ov_decoder, latents):
    for rule in ("drop_first", "diffusers"):
        out = TiledDecoder(ov_decoder, TilingConfig(first_frame_rule=rule, **SMALL_TILES)).decode(latents, mode)
        assert out.shape == VIDEO_SHAPE and out.dtype == np.float32 and np.isfinite(out).all()
    # a shape that does not divide into whole tiles / chunks
    odd = latents[:, :, :4, :, :7]
    assert TiledDecoder(ov_decoder, TilingConfig(**SMALL_TILES)).decode(odd, mode).shape == (1, 3, 25, 160, 224)


@pytest.mark.parametrize("mode", MODES)
def test_tile_larger_than_input_equals_untiled(mode, ov_decoder, latents):
    """Tiles of 512 px / chunks of 64 frames on a 256x160, 33-frame video: one decoder call, same output."""
    dec = TiledDecoder(ov_decoder, TilingConfig(tile_sample_min_num_frames=64, tile_sample_stride_num_frames=32))
    out = dec.decode(latents, mode)
    assert dec.calls == 1
    np.testing.assert_allclose(out, ov_decoder(latents), atol=1e-5)


@pytest.mark.parametrize("mode", MODES[1:])
def test_openvino_port_matches_diffusers_tiling(mode, vae, ov_decoder, latents):
    """The port check: our tiling on the OpenVINO IR == diffusers' own tiling in PyTorch, same settings."""
    ref_vae = copy.deepcopy(vae)
    ref_vae.use_tiling = "spatial" in mode
    ref_vae.use_framewise_decoding = "temporal" in mode
    ref_vae.tile_sample_min_height = ref_vae.tile_sample_min_width = 128
    ref_vae.tile_sample_stride_height = ref_vae.tile_sample_stride_width = 96
    with torch.no_grad():
        ref = ref_vae.decode(torch.from_numpy(latents), None, return_dict=False)[0].numpy()
        untiled = vae.decode(torch.from_numpy(latents), None, return_dict=False)[0].numpy()
    out = TiledDecoder(ov_decoder, TilingConfig(first_frame_rule="diffusers", **SMALL_TILES)).decode(latents, mode)
    assert np.abs(ref - untiled).max() > 1e-3, "tiling did not kick in, the test would prove nothing"
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_seam_layout():
    dec = TiledDecoder(toy_decoder, TilingConfig(**SMALL_TILES))
    assert dec.seam_layout(LATENT_SHAPE, "none") == {"frames": [], "rows": [], "cols": []}
    assert dec.seam_layout(LATENT_SHAPE, "spatial+temporal") == {
        "frames": [8, 16, 24], "rows": [95, 127], "cols": [95, 127, 191, 223]}
    assert TiledDecoder(toy_decoder).seam_layout(LATENT_SHAPE, "spatial")["rows"] == []  # smaller than one 512 px tile


def test_metrics_detect_a_seam_and_a_shift():
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 0.01, (1, 3, 33, 32, 32)), axis=2).astype(np.float32)  # slowly changing video
    same = metrics.compare(base, base.copy(), {"frames": [8, 16, 24], "rows": [15], "cols": []})
    assert same["psnr_db"] == float("inf") and same["ssim"] == pytest.approx(1.0) and same["drift"]["drop_db"] == 0.0
    jump = base.copy()
    jump[:, :, 17:] += 0.3  # a jump at the chunk boundary 16 -> 17
    jump[:, :, :, 16:] += 0.2  # and a line between rows 15 and 16
    m = metrics.compare(base, jump, {"frames": [8, 16, 24], "rows": [15], "cols": []})
    assert m["temporal_seam"]["tiled"]["score"] > 2 * m["temporal_seam"]["untiled"]["score"]
    assert m["spatial_seam"]["tiled"]["score"] > 2 * m["spatial_seam"]["untiled"]["score"]
    assert m["drift"]["drop_db"] > 0  # only the later frames were damaged
    late = np.concatenate([base[:, :, :1], base[:, :, :-1]], axis=2)  # every frame shown one frame late
    m = metrics.compare(base, late)
    assert m["psnr_frames_2plus_vs_previous_ref_frame_db"] == float("inf") > m["psnr_frames_2plus_db"]


def test_kaggle_script_embeds_current_sources():
    sys.path.insert(0, str(ROOT / "scripts"))
    import s3_embed

    assert s3_embed.embedded_block() in s3_embed.TARGET.read_text(), "run scripts/s3_embed.py"
