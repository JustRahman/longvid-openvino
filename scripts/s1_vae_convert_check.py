"""Step 1 of LongVid-OV: convert the LTX-Video VAE to OpenVINO IR and prove it decodes correctly.

Pipeline (re-runnable; downloads and IRs are cached, pass --force to reconvert):
  1. download only vae/ of Lightricks/LTX-Video
  2. load it with diffusers AutoencoderKLLTXVideo (PyTorch, CPU, float32) = reference
  3. convert encoder + decoder to OpenVINO IR (dynamic frames/height/width)
  4. build a real 25-frame 512x320 test clip, encode with PyTorch, decode with PyTorch and OpenVINO
  5. compare (max abs diff, PSNR, SSIM), write contact sheet, mp4s, metrics JSON and the report

Run from the project root:  .venv/bin/python scripts/s1_vae_convert_check.py
  --force             reconvert the IRs even if they exist
  --conv3d-baseline   also time the naive (native Conv3d) decoder export, ~1.5 min extra; see Conv3dAs2d
"""
import argparse
import contextlib
import json
import platform
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import openvino as ov
import psutil
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLLTXVideo
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent
PT_DIR = ROOT / "models" / "ltx_vae_pt"
OV_DIR = ROOT / "models" / "ltx_vae_ov"
DATA_DIR = ROOT / "data" / "clip"  # downloaded media lives in its own folder
RES_DIR = ROOT / "results"

REPO = "Lightricks/LTX-Video"
# Public sample video used in the diffusers docs (CogVideoX video-to-video example).
CLIP_REPO, CLIP_FILE = "huggingface/documentation-images", "diffusers/hiker.mp4"
FRAMES, HEIGHT, WIDTH, FPS = 25, 320, 512, 24  # frames must be 8k+1, H/W multiples of 32
# Second, different shape: proves the IRs are really dynamic, not baked to the trace shape.
FRAMES2, HEIGHT2, WIDTH2 = 9, 192, 256


class PeakRSS:
    """Samples process RSS in a background thread; .peak is the max seen (bytes)."""

    def __init__(self, interval=0.05):
        self.proc, self.interval, self.peak, self._stop = psutil.Process(), interval, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self._t.join()
        return self.peak


class EncoderWrapper(torch.nn.Module):
    """video [B,3,F,H,W] in [-1,1] -> latent [B,128,f,h,w] (posterior mean, i.e. latent_dist.mode())."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, sample):
        return self.vae.encode(sample).latent_dist.mode()


class DecoderWrapper(torch.nn.Module):
    """latent [B,128,f,h,w] -> video [B,3,F,H,W] in ~[-1,1]. Same call diffusers makes: vae.decode(z, None)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, None, return_dict=False)[0]


class Conv3dAs2d(torch.nn.Module):
    """Exact re-expression of an nn.Conv3d (no temporal padding, zero spatial padding) as ONE Conv2d.

    Why: the OpenVINO CPU plugin on ARM64 has no optimised 5D convolution and falls back to its reference
    kernel (exec type `ref_any`), which makes the VAE ~50x slower than PyTorch. 2D convolutions do get
    optimised ARM kernels. A 3D conv with temporal kernel kt is a 2D conv over kt time-shifted copies of
    the input stacked along channels, with frames folded into the batch dimension:
        [B,C,T,H,W] -> kt taps [B,C,T',H,W] -> cat -> [B*T', kt*C, H, W] -> conv2d -> [B,O,T',H',W']
    Same weights (reshaped), same math; only the float summation order can differ.
    """

    def __init__(self, conv):
        super().__init__()
        assert conv.padding[0] == 0 and conv.padding_mode == "zeros" and conv.dilation == (1, 1, 1) and conv.groups == 1
        o, c, kt, kh, kw = conv.weight.shape
        self.kt, self.stride_t = kt, conv.stride[0]
        self.stride, self.padding = conv.stride[1:], conv.padding[1:]
        self.weight = torch.nn.Parameter(conv.weight.detach().permute(0, 2, 1, 3, 4).reshape(o, kt * c, kh, kw))
        self.bias = conv.bias

    def forward(self, x):
        # tap k = input frames k, k+stride_t, ... that a valid (unpadded) temporal conv would read
        taps = [x[:, :, k : (k - self.kt + 1) or None : self.stride_t] for k in range(self.kt)]
        x = torch.cat(taps, dim=1).permute(0, 2, 1, 3, 4)  # [B,T',kt*C,H,W]
        y = F.conv2d(x.flatten(0, 1), self.weight, self.bias, self.stride, self.padding)
        return y.unflatten(0, (x.shape[0], -1)).permute(0, 2, 1, 3, 4)


@contextlib.contextmanager
def conv3d_as_2d(model):
    """Temporarily swap every nn.Conv3d for Conv3dAs2d (export only; the PyTorch reference stays stock)."""
    swapped = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, torch.nn.Conv3d):
            parent, attr = model.get_submodule(name.rpartition(".")[0]), name.rpartition(".")[2]
            swapped.append((parent, attr, mod))
            setattr(parent, attr, Conv3dAs2d(mod))
    try:
        yield len(swapped)
    finally:
        for parent, attr, mod in swapped:
            setattr(parent, attr, mod)


def log(msg):
    print(f"[s1] {msg}", flush=True)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def make_clip():
    """Returns (uint8 frames [F,H,W,3], source description). Real video if possible, synthetic fallback."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(CLIP_REPO, CLIP_FILE, repo_type="dataset", local_dir=DATA_DIR)
        reader = imageio.get_reader(path, format="ffmpeg")
        frames = []
        for i, fr in enumerate(reader):
            if i >= FRAMES:
                break
            # center-crop to the target aspect ratio, then resize
            h, w = fr.shape[:2]
            ch = min(h, int(round(w * HEIGHT / WIDTH)))
            cw = min(w, int(round(ch * WIDTH / HEIGHT)))
            top, left = (h - ch) // 2, (w - cw) // 2
            img = Image.fromarray(fr[top : top + ch, left : left + cw]).resize((WIDTH, HEIGHT), Image.LANCZOS)
            frames.append(np.asarray(img))
        src_meta = reader.get_meta_data()
        reader.close()
        if len(frames) != FRAMES:
            raise RuntimeError(f"only {len(frames)} frames in {path}")
        src = (
            f"real video: HF dataset {CLIP_REPO}/{CLIP_FILE} (source {src_meta.get('size')} @ {src_meta.get('fps')} fps), "
            f"first {FRAMES} frames, center-cropped + Lanczos-resized to {WIDTH}x{HEIGHT}"
        )
        return np.stack(frames), src
    except Exception as e:  # fallback: synthetic moving pattern
        print(f"[clip] real video unavailable ({e!r}); using synthetic moving pattern")
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        frames = []
        for t in range(FRAMES):
            r = 0.5 + 0.5 * np.sin((xx + 6 * t) / 23.0)
            g = 0.5 + 0.5 * np.sin((yy - 4 * t) / 17.0)
            b = 0.5 + 0.5 * np.sin((xx + yy + 8 * t) / 31.0)
            frames.append((np.stack([r, g, b], -1) * 255).astype(np.uint8))
        return np.stack(frames), f"SYNTHETIC fallback (moving sine pattern), reason: {e!r}"


def to_tensor(frames_u8):
    """uint8 [F,H,W,3] -> float32 [1,3,F,H,W] in [-1,1] (the range the VAE works in)."""
    return torch.from_numpy(frames_u8).float().div(127.5).sub(1.0).permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def to_unit(video):
    """[1,3,F,H,W] in [-1,1] (numpy) -> float [F,H,W,3] clipped to [0,1]."""
    return np.clip(video[0].transpose(1, 2, 3, 0) / 2 + 0.5, 0, 1)


def psnr(a, b, data_range=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(data_range**2 / mse)


def ssim(a, b):
    """Mean per-frame SSIM over [F,H,W,3] arrays in [0,1]."""
    return float(np.mean([structural_similarity(x, y, channel_axis=-1, data_range=1.0) for x, y in zip(a, b)]))


def convert(wrapper, example, input_shape, xml_path, out_name):
    """Trace wrapper (Conv3d swapped for Conv2d) with `example`, convert to OV with dynamic `input_shape`, save FP32 IR."""
    with torch.no_grad(), conv3d_as_2d(wrapper):
        m = ov.convert_model(wrapper, example_input=example, input=[ov.PartialShape(input_shape)])
    m.outputs[0].get_tensor().set_names({out_name})
    # FP32 on purpose: this step is the fidelity reference; weight compression comes in later steps.
    ov.save_model(m, xml_path, compress_to_fp16=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reconvert the IRs even if they exist")
    ap.add_argument("--conv3d-baseline", action="store_true", help="also time the naive native-Conv3d decoder export")
    args = ap.parse_args()
    for d in (OV_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    mem = PeakRSS()
    T = {}  # timings in seconds
    R = {}  # everything that goes into the metrics JSON / report

    # 1. download ONLY the VAE ------------------------------------------------------------------
    _, T["download_or_cache_check"] = timed(
        lambda: snapshot_download(REPO, allow_patterns=["vae/*"], local_dir=PT_DIR)
    )
    R["downloaded_files"] = {
        str(p.relative_to(PT_DIR)): p.stat().st_size for p in sorted((PT_DIR / "vae").glob("*")) if p.is_file()
    }

    log(f"VAE files ready in {PT_DIR.relative_to(ROOT)}")

    # 2. PyTorch reference ----------------------------------------------------------------------
    vae, T["pt_load"] = timed(
        lambda: AutoencoderKLLTXVideo.from_pretrained(PT_DIR / "vae", torch_dtype=torch.float32).eval()
    )
    cfg = vae.config
    # Decode-time inputs. Newer LTX VAEs (0.9.1+) take a timestep and get noise mixed into the latents
    # by the pipeline; this checkpoint has timestep_conditioning=False, so diffusers calls
    # vae.decode(latents, None) and adds no noise. The decoder IR therefore has a single input.
    if cfg.timestep_conditioning or any(cfg.decoder_inject_noise):
        raise NotImplementedError("this VAE needs a decode timestep / noise injection; extend DecoderWrapper")
    R["vae_config"] = {
        k: cfg[k]
        for k in ("latent_channels", "patch_size", "patch_size_t", "block_out_channels", "spatio_temporal_scaling",
                  "encoder_causal", "decoder_causal", "timestep_conditioning", "decoder_inject_noise", "scaling_factor")
    }
    R["vae_params_M"] = round(sum(p.numel() for p in vae.parameters()) / 1e6, 2)
    R["compression"] = {"spatial": vae.spatial_compression_ratio, "temporal": vae.temporal_compression_ratio}
    # Per-channel latent stats: the *pipeline* (not the VAE) de-normalises transformer latents with these
    # before decode: z = z * std / scaling_factor + mean. Saved next to the IR for the later steps.
    np.savez(OV_DIR / "latents_stats.npz", mean=vae.latents_mean.numpy(), std=vae.latents_std.numpy(),
             scaling_factor=np.float32(cfg.scaling_factor))
    R["latents_mean_range"] = [float(vae.latents_mean.min()), float(vae.latents_mean.max())]
    R["latents_std_range"] = [float(vae.latents_std.min()), float(vae.latents_std.max())]

    # 4a. test clip (needed before conversion: it is the tracing example) ------------------------
    clip_u8, R["clip_source"] = make_clip()
    video = to_tensor(clip_u8)  # [1,3,25,320,512]
    orig = clip_u8.astype(np.float32) / 255.0
    lat_f, lat_h, lat_w = (FRAMES - 1) // 8 + 1, HEIGHT // 32, WIDTH // 32

    # 3. convert to OpenVINO IR -----------------------------------------------------------------
    enc_xml, dec_xml = OV_DIR / "vae_encoder.xml", OV_DIR / "vae_decoder.xml"
    R["converted_this_run"] = args.force or not (enc_xml.exists() and dec_xml.exists())
    if R["converted_this_run"]:
        log("converting encoder + decoder to OpenVINO IR ...")
        _, T["convert_encoder"] = timed(
            lambda: convert(EncoderWrapper(vae), video, [1, 3, -1, -1, -1], enc_xml, "latent")
        )
        ex_latent = torch.randn(1, cfg.latent_channels, lat_f, lat_h, lat_w)
        _, T["convert_decoder"] = timed(
            lambda: convert(DecoderWrapper(vae), ex_latent, [1, cfg.latent_channels, -1, -1, -1], dec_xml, "sample")
        )
    R["ir_files"] = {p.name: p.stat().st_size for p in sorted(OV_DIR.glob("vae_*"))}

    core = ov.Core()
    ov_cfg = {"INFERENCE_PRECISION_HINT": "f32"}  # ARM CPUs may default to f16; keep the check apples-to-apples
    ov_enc, T["ov_compile_encoder"] = timed(lambda: core.compile_model(enc_xml, "CPU", ov_cfg))
    ov_dec, T["ov_compile_decoder"] = timed(lambda: core.compile_model(dec_xml, "CPU", ov_cfg))
    R["ov_io"] = {
        "encoder_in": str(ov_enc.input(0).partial_shape), "encoder_out": str(ov_enc.output(0).partial_shape),
        "decoder_in": str(ov_dec.input(0).partial_shape), "decoder_out": str(ov_dec.output(0).partial_shape),
        "inference_precision": str(ov_dec.get_property("INFERENCE_PRECISION_HINT")),
    }

    log(f"IRs compiled on CPU: {R['ov_io']}")

    # 4b. encode with PyTorch, decode with PyTorch and OpenVINO ----------------------------------
    with torch.no_grad():
        latents, T["pt_encode"] = timed(lambda: vae.encode(video).latent_dist.mode())
        pt_dec, T["pt_decode"] = timed(lambda: vae.decode(latents, None, return_dict=False)[0])
    latents_np, pt_dec = latents.numpy(), pt_dec.numpy()
    log(f"PyTorch encode {T['pt_encode']:.1f} s, decode {T['pt_decode']:.1f} s; running OpenVINO ...")
    ov_dec_out, T["ov_decode_first"] = timed(lambda: ov_dec(latents_np)[0])  # includes shape-specific setup
    ov_dec_out, T["ov_decode"] = timed(lambda: ov_dec(latents_np)[0])
    ov_lat, T["ov_encode_first"] = timed(lambda: ov_enc(video.numpy())[0])
    assert latents_np.shape == (1, cfg.latent_channels, lat_f, lat_h, lat_w), latents_np.shape
    assert pt_dec.shape == ov_dec_out.shape == tuple(video.shape), (pt_dec.shape, ov_dec_out.shape)
    R["shapes"] = {
        "video_in": list(video.shape), "latents": list(latents_np.shape),
        "pt_decoded": list(pt_dec.shape), "ov_decoded": list(ov_dec_out.shape),
    }
    R["latent_stats"] = {"min": float(latents_np.min()), "max": float(latents_np.max()),
                         "mean": float(latents_np.mean()), "std": float(latents_np.std())}

    log(f"OpenVINO decode {T['ov_decode']:.1f} s (first call {T['ov_decode_first']:.1f} s)")

    # 5. verify ---------------------------------------------------------------------------------
    pt_u, ov_u = to_unit(pt_dec), to_unit(ov_dec_out)
    R["ov_vs_pt_decode"] = {
        "max_abs_diff_raw": float(np.abs(pt_dec - ov_dec_out).max()),  # raw decoder output, range ~[-1,1]
        "mean_abs_diff_raw": float(np.abs(pt_dec - ov_dec_out).mean()),
        "psnr_db": psnr(pt_u, ov_u),  # on clipped [0,1] frames
        "max_abs_diff_uint8": int(np.abs(np.round(pt_u * 255) - np.round(ov_u * 255)).max()),
    }
    R["ov_vs_pt_encode"] = {
        "max_abs_diff": float(np.abs(latents_np - ov_lat).max()),
        "mean_abs_diff": float(np.abs(latents_np - ov_lat).mean()),
    }
    R["pt_vs_original"] = {"psnr_db": psnr(orig, pt_u), "ssim": ssim(orig, pt_u)}
    R["ov_vs_original"] = {"psnr_db": psnr(orig, ov_u), "ssim": ssim(orig, ov_u)}

    # Dynamic-shape check: a different frames/height/width through the same compiled IRs.
    small_u8 = np.stack([np.asarray(Image.fromarray(f).resize((WIDTH2, HEIGHT2), Image.LANCZOS))
                         for f in clip_u8[:FRAMES2]])
    small = to_tensor(small_u8)
    with torch.no_grad():
        lat2 = vae.encode(small).latent_dist.mode()
        pt2 = vae.decode(lat2, None, return_dict=False)[0].numpy()
    ov_lat2, ov2 = ov_enc(small.numpy())[0], ov_dec(lat2.numpy())[0]
    R["dynamic_shape_check"] = {
        "video_in": list(small.shape), "latents": list(lat2.shape), "ov_decoded": list(ov2.shape),
        "decode_max_abs_diff_raw": float(np.abs(pt2 - ov2).max()),
        "decode_psnr_db": psnr(to_unit(pt2), to_unit(ov2)),
        "encode_max_abs_diff": float(np.abs(lat2.numpy() - ov_lat2).max()),
    }

    # Optional: measure what the Conv3dAs2d rewrite buys. Converts the stock decoder (native Conv3d) in memory
    # and times it on the small shape next to the shipped IR. Saved separately so later runs keep the result.
    if args.conv3d_baseline:
        log("timing the naive native-Conv3d decoder export (slow on ARM) ...")
        with torch.no_grad():
            naive = ov.convert_model(DecoderWrapper(vae), example_input=lat2,
                                     input=[ov.PartialShape([1, cfg.latent_channels, -1, -1, -1])])
        req = core.compile_model(naive, "CPU", {**ov_cfg, "PERF_COUNT": "YES"}).create_infer_request()
        _, t_naive = timed(lambda: req.infer({0: lat2.numpy()}))
        conv = [p for p in req.get_profiling_info() if p.node_type == "Convolution"]
        _, t_pt = timed(lambda: vae.decode(lat2, None))
        _, t_ours = timed(lambda: ov_dec(lat2.numpy()))
        (RES_DIR / "s1_conv3d_baseline.json").write_text(json.dumps({
            "latents": list(lat2.shape), "naive_conv3d_ir_decode_s": round(t_naive, 2),
            "naive_conv_layers": len(conv), "naive_conv_exec_types": sorted({p.exec_type for p in conv}),
            "naive_conv_time_s": round(sum(p.real_time.total_seconds() for p in conv), 2),
            "naive_max_abs_diff_vs_pt": float(np.abs(req.get_output_tensor(0).data - pt2).max()),
            "conv2d_ir_decode_s": round(t_ours, 2), "pytorch_decode_s": round(t_pt, 2), "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        del req, naive

    # Outputs: mp4s + contact sheet (rows: original / PyTorch / OpenVINO, 4 frames)
    for name, arr in (("s1_original", orig), ("s1_decoded_pytorch", pt_u), ("s1_decoded_openvino", ov_u)):
        imageio.mimwrite(RES_DIR / f"{name}.mp4", np.round(arr * 255).astype(np.uint8), fps=FPS, quality=9,
                         macro_block_size=16)
    idx = [0, 8, 16, 24]
    label_w, pad = 90, 4
    sheet = Image.new("RGB", (label_w + len(idx) * (WIDTH + pad), 3 * (HEIGHT + pad) + 16), "white")
    draw = ImageDraw.Draw(sheet)
    for c, i in enumerate(idx):
        draw.text((label_w + c * (WIDTH + pad) + 4, 2), f"frame {i}", fill="black")
    for r, (label, arr) in enumerate((("original", orig), ("PyTorch", pt_u), ("OpenVINO", ov_u))):
        draw.text((6, 16 + r * (HEIGHT + pad) + HEIGHT // 2), label, fill="black")
        for c, i in enumerate(idx):
            tile = Image.fromarray(np.round(arr[i] * 255).astype(np.uint8))
            sheet.paste(tile, (label_w + c * (WIDTH + pad), 16 + r * (HEIGHT + pad)))
    sheet.save(RES_DIR / "s1_check.png")

    R["peak_rss_gb"] = round(mem.stop() / 2**30, 2)
    R["timings_s"] = {k: round(v, 2) for k, v in T.items()}
    R["versions"] = {
        "python": platform.python_version(), "openvino": ov.__version__, "torch": torch.__version__,
        "diffusers": __import__("diffusers").__version__, "numpy": np.__version__,
        "machine": f"{platform.machine()} / {platform.platform()}", "cpu_threads": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
    }
    R["run_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RES_DIR / "s1_metrics.json").write_text(json.dumps(R, indent=2, default=str))
    write_report(R)
    print_summary(R)


def write_report(R):
    """results/s1_report.md, generated from this run's measurements (no hand-typed numbers)."""
    s, t, d, v = R["shapes"], R["timings_s"], R["ov_vs_pt_decode"], R["versions"]
    dyn, c = R["dynamic_shape_check"], R["vae_config"]
    gb = lambda n: f"{n / 2**30:.2f} GB"
    conv = (f"encoder {t['convert_encoder']} s, decoder {t['convert_decoder']} s" if R["converted_this_run"]
            else "IRs reused from an earlier run (use --force to re-time conversion)")
    md = f"""# Step 1 report: LTX-Video VAE in OpenVINO

Generated by `scripts/s1_vae_convert_check.py` on {R['run_time']}. Every number below was measured in that run
(raw values: `results/s1_metrics.json`).

## What was done
1. Downloaded only `vae/` of `{REPO}` with `huggingface_hub.snapshot_download(allow_patterns=["vae/*"])`
   into `models/ltx_vae_pt/` ({', '.join(f'{k}: {gb(n) if n > 1e6 else str(n) + " B"}' for k, n in R['downloaded_files'].items())}).
   The repo was not gated; the community-IR fallback was not needed.
2. Loaded it with diffusers `AutoencoderKLLTXVideo` (PyTorch, CPU, float32, {R['vae_params_M']} M parameters) as the reference.
3. Converted encoder and decoder with `openvino.convert_model` to FP32 IR in `models/ltx_vae_ov/`
   ({', '.join(f'{k}: {gb(n) if n > 1e6 else str(n) + " B"}' for k, n in R['ir_files'].items())}).
4. Test clip: {R['clip_source']}.
   Encoded with the PyTorch encoder (posterior mean, `latent_dist.mode()`), decoded with PyTorch and with OpenVINO (CPU).
5. Compared the decodes and wrote `s1_check.png`, `s1_original.mp4`, `s1_decoded_pytorch.mp4`, `s1_decoded_openvino.mp4`.

## Shapes
| tensor | shape |
|---|---|
| input video (B,C,F,H,W), range [-1,1] | {s['video_in']} |
| latents | {s['latents']} |
| PyTorch decode | {s['pt_decoded']} |
| OpenVINO decode | {s['ov_decoded']} |

IR signatures (dynamic frames/height/width, batch fixed to 1):
- encoder: `{R['ov_io']['encoder_in']}` -> `{R['ov_io']['encoder_out']}`
- decoder: `{R['ov_io']['decoder_in']}` -> `{R['ov_io']['decoder_out']}`

Dynamic shapes were verified by running a second, different shape through the same compiled models:
video {dyn['video_in']} -> latents {dyn['latents']} -> decode {dyn['ov_decoded']};
OV vs PyTorch decode max abs diff {dyn['decode_max_abs_diff_raw']:.3e}, PSNR {dyn['decode_psnr_db']:.2f} dB;
encode max abs diff {dyn['encode_max_abs_diff']:.3e}.

## VAE latent shape math (as observed)
- Spatial: /{R['compression']['spatial']} = patchify {c['patch_size']}x{c['patch_size']} (pixel-unshuffle, no weights)
  x 2x2x2 from the {sum(c['spatio_temporal_scaling'])} spatio-temporal down blocks. {HEIGHT}x{WIDTH} -> {s['latents'][3]}x{s['latents'][4]}.
- Temporal: /{R['compression']['temporal']}, causal on the encoder side: the first frame gets its own latent frame, so
  `latent_frames = (frames - 1) / 8 + 1` and `frames = (latent_frames - 1) * 8 + 1`. {FRAMES} -> {s['latents'][2]}; {FRAMES2} -> {dyn['latents'][2]}.
  That is why frames must be 8k+1 (and H, W multiples of 32).
- Channels: 3 -> {c['latent_channels']}. Overall {s['video_in'][1] * s['video_in'][2] * s['video_in'][3] * s['video_in'][4]} values ->
  {s['latents'][1] * s['latents'][2] * s['latents'][3] * s['latents'][4]} values
  ({s['video_in'][1] * s['video_in'][2] * s['video_in'][3] * s['video_in'][4] / (s['latents'][1] * s['latents'][2] * s['latents'][3] * s['latents'][4]):.1f}x compression for this clip).

## Decode-time inputs (handled like diffusers)
- `timestep_conditioning = {c['timestep_conditioning']}`, `decoder_inject_noise = {c['decoder_inject_noise']}` for this checkpoint,
  so `LTXPipeline` calls `vae.decode(latents, None)` with no decode timestep and mixes in no decode noise.
  The decoder IR therefore has exactly one input (`latent`). The script raises if a VAE with timestep conditioning
  (LTX 0.9.1+) is loaded, instead of silently converting it wrong.
- Causality: `encoder_causal = {c['encoder_causal']}`, `decoder_causal = {c['decoder_causal']}`. It is a constructor/config flag
  (first-frame replicate padding in every temporal conv), not a runtime input, so it is baked into the IR graph.
- Latent de-normalisation is done by the pipeline, outside the VAE: `z = z * latents_std / scaling_factor + latents_mean`
  (scaling_factor = {c['scaling_factor']}; mean range {R['latents_mean_range']}, std range {R['latents_std_range']}).
  It is NOT inside the IR; the vectors are saved to `models/ltx_vae_ov/latents_stats.npz` for the next steps.
  It was not needed here because latents came straight from the encoder.
- Tiling / framewise decoding are off by default in diffusers and were not used.

## Results
OpenVINO vs PyTorch (same latents in, inference precision `{R['ov_io']['inference_precision']}`):
| metric | value |
|---|---|
| decode max abs diff (raw output, range ~[-1,1]) | {d['max_abs_diff_raw']:.3e} |
| decode mean abs diff (raw) | {d['mean_abs_diff_raw']:.3e} |
| decode PSNR OV vs PT (frames clipped to [0,1]) | {d['psnr_db']:.2f} dB |
| decode max diff after uint8 rounding | {d['max_abs_diff_uint8']} / 255 |
| encoder latents max abs diff (latent std {R['latent_stats']['std']:.3f}) | {R['ov_vs_pt_encode']['max_abs_diff']:.3e} |

Reconstruction vs the original clip ({FRAMES} frames, {WIDTH}x{HEIGHT}; SSIM = mean per-frame, RGB):
| decode | PSNR | SSIM |
|---|---|---|
| PyTorch | {R['pt_vs_original']['psnr_db']:.2f} dB | {R['pt_vs_original']['ssim']:.4f} |
| OpenVINO | {R['ov_vs_original']['psnr_db']:.2f} dB | {R['ov_vs_original']['ssim']:.4f} |

## Time and memory
| stage | seconds |
|---|---|
| PyTorch load | {t['pt_load']} |
| convert + save IR | {conv} |
| OV compile (CPU) | encoder {t['ov_compile_encoder']}, decoder {t['ov_compile_decoder']} |
| PyTorch encode | {t['pt_encode']} |
| PyTorch decode | {t['pt_decode']} |
| OpenVINO decode | first call {t['ov_decode_first']}, second call {t['ov_decode']} |
| OpenVINO encode | first call {t['ov_encode_first']} |

Single runs, not a benchmark. Peak process RSS (psutil, sampled every 50 ms, whole script with PyTorch and both
OpenVINO models loaded at once): **{R['peak_rss_gb']} GB**.

## Versions
Python {v['python']}, openvino {v['openvino']}, torch {v['torch']}, diffusers {v['diffusers']}, numpy {v['numpy']};
{v['machine']}, {v['cpu_threads']} threads, {v['ram_total_gb']} GB RAM, OpenVINO device CPU.

## What broke / notes
{notes()}
"""
    (RES_DIR / "s1_report.md").write_text(md)


def notes():
    """Hand-written observations from developing this step; numbers only come from s1_conv3d_baseline.json."""
    base = RES_DIR / "s1_conv3d_baseline.json"
    if base.exists():
        b = json.loads(base.read_text())
        measured = (
            f"Measured ({b['run_time']}, latents {b['latents']}, `--conv3d-baseline`): naive IR decode "
            f"{b['naive_conv3d_ir_decode_s']} s, of which {b['naive_conv_time_s']} s in {b['naive_conv_layers']} Convolution "
            f"layers with exec type {b['naive_conv_exec_types']}; PyTorch {b['pytorch_decode_s']} s; "
            f"Conv2d-rewritten IR {b['conv2d_ir_decode_s']} s. The naive IR was still numerically correct "
            f"(max abs diff vs PyTorch {b['naive_max_abs_diff_vs_pt']:.3e}), just slow."
        )
    else:
        measured = "Not measured in this results folder; run the script with `--conv3d-baseline` to record the timings."
    return f"""- **Native Conv3d is unusably slow in OpenVINO on this ARM CPU (main finding).** A straight
  `convert_model` of the diffusers VAE gives a correct IR, but the CPU plugin executes every 5D convolution
  with its unoptimised reference kernel. The first end-to-end run of this script was killed at a 10-minute limit
  while still inside OpenVINO inference; per-layer profiling (`PERF_COUNT`) showed where the time went.
  Reshaping the IR to static shapes did not change the exec type or the time, so it is not a dynamic-shape issue.
  {measured}
  **Fix:** at export time only, every `nn.Conv3d` is swapped for `Conv3dAs2d` (see the script): the same weights
  applied as one 2D convolution over time-shifted copies of the input stacked along channels, with frames folded
  into the batch. This is mathematically identical (only float summation order differs), runs on the optimised
  ARM 2D kernels, and keeps frames/height/width dynamic. The PyTorch reference is the untouched diffusers model.
- **ARM inference precision.** `INFERENCE_PRECISION_HINT` is set to `f32` explicitly so the OV-vs-PyTorch
  comparison is float32 against float32; the IRs are also saved uncompressed (FP32). FP16 / weight compression
  is left for a later step and was not evaluated here.
- **Batch is fixed to 1** in the IRs; frames, height and width are dynamic. Valid inputs still need
  frames = 8k+1 and H, W multiples of 32 on the encoder side (the patchify reshape fails otherwise).
- The encoder IR returns the posterior mean (deterministic), not a random sample of the posterior.
- Not covered in this step: VAE tiling / framewise decoding, LTX 0.9.1+ VAEs with decode timestep and noise
  injection (the script refuses those), and any shape other than the two tested.
"""


def print_summary(R):
    d, t, s = R["ov_vs_pt_decode"], R["timings_s"], R["shapes"]
    print("\n".join([
        "=== STEP 1 SUMMARY ===",
        f"1. VAE: {REPO}/vae only ({R['vae_params_M']} M params), PyTorch float32 CPU reference",
        f"2. IR: {OV_DIR.relative_to(ROOT)}/vae_encoder.xml + vae_decoder.xml (FP32), decoder in {R['ov_io']['decoder_in']}",
        f"3. Clip: {R['clip_source'][:90]}",
        f"4. Shapes: video {s['video_in']} -> latents {s['latents']} -> decode {s['ov_decoded']}",
        f"5. OV vs PT decode: max abs diff {d['max_abs_diff_raw']:.3e}, PSNR {d['psnr_db']:.2f} dB, uint8 max diff {d['max_abs_diff_uint8']}",
        f"6. vs original: PT {R['pt_vs_original']['psnr_db']:.2f} dB / SSIM {R['pt_vs_original']['ssim']:.4f}; "
        f"OV {R['ov_vs_original']['psnr_db']:.2f} dB / SSIM {R['ov_vs_original']['ssim']:.4f}",
        f"7. Dynamic shape check {R['dynamic_shape_check']['video_in']}: decode max abs diff "
        f"{R['dynamic_shape_check']['decode_max_abs_diff_raw']:.3e}",
        f"8. Decode time: PyTorch {t['pt_decode']} s, OpenVINO {t['ov_decode']} s (first call {t['ov_decode_first']} s)",
        f"9. Peak RSS: {R['peak_rss_gb']} GB",
        "10. Outputs: results/s1_report.md, s1_metrics.json, s1_check.png, s1_*.mp4",
    ]))


if __name__ == "__main__":
    main()
