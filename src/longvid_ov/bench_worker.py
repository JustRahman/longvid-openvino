"""s3 worker: one job per process, so every measurement starts from a clean heap.

    python bench_worker.py tiny | convert | encode | decode | metrics  ... --out result.json

Progress and results go to --out as JSON (the parent reads it even if this process gets killed).
The VAE loading, IR conversion and the Conv3dAs2d ARM workaround are the step 1 / step 2 code.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tiled_decode.py and metrics.py sit next to this file
from tiled_decode import MODES, OVDecoder, TiledDecoder, TilingConfig  # noqa: E402

R = {}  # the result JSON
OUT = None


def rss_gb():
    return round(psutil.Process().memory_info().rss / 2**30, 3)


def save(**kw):
    R.update(kw)
    R.setdefault("rss_at_phase_gb", {})[R["phase"]] = rss_gb()  # RSS when each phase starts
    if OUT:
        with open(OUT + ".tmp", "w") as f:
            json.dump(R, f)
        os.replace(OUT + ".tmp", OUT)


def load_vae(vae_dir, part, threads=0):
    """One half ("decoder" or "encoder") of the diffusers LTX VAE: PyTorch, float32, eval.

    Built on the meta device and filled straight from the safetensors file, so loading holds one copy of the
    weights of that half only (a plain from_pretrained can hold two copies of everything).
    """
    import torch
    from diffusers import AutoencoderKLLTXVideo
    from safetensors import safe_open

    if threads:
        torch.set_num_threads(threads)
    with torch.device("meta"):
        vae = AutoencoderKLLTXVideo.from_config(AutoencoderKLLTXVideo.load_config(vae_dir))
    if vae.config.timestep_conditioning or any(vae.config.decoder_inject_noise):
        raise NotImplementedError("this VAE needs a decode timestep / noise injection (LTX 0.9.1+)")
    other = "encoder" if part == "decoder" else "decoder"
    setattr(vae, other, None)
    with safe_open(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), "pt") as f:
        weights = {k: f.get_tensor(k) for k in f.keys() if not k.startswith(other + ".")}
    vae.load_state_dict(weights, strict=True, assign=True)
    on_meta = [n for n, t in [*vae.named_parameters(), *vae.named_buffers()] if t.device.type == "meta"]
    if on_meta:
        raise RuntimeError(f"tensors not filled from the checkpoint: {on_meta[:5]}")
    return vae.eval()


def convert_decoder(vae, ir_path, conv3d_as_2d=False):
    """diffusers VAE decoder -> OpenVINO IR (FP32, dynamic frames/H/W). Returns how many Conv3d were rewritten.

    conv3d_as_2d: the step 1 workaround for ARM CPUs, where OpenVINO runs 5D convolutions on a slow reference
    kernel: every nn.Conv3d becomes ONE Conv2d over time-shifted copies of the input stacked on channels.
    Same weights, same math. Leaves `vae` rewritten, so convert last.
    """
    import openvino as ov
    import torch
    import torch.nn.functional as F

    class DecoderWrapper(torch.nn.Module):
        """latent [1,C,f,h,w] -> video [1,3,F,H,W]. Same call diffusers makes: vae.decode(z, None)."""

        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            return self.vae.decode(latent, None, return_dict=False)[0]

    class Conv3dAs2d(torch.nn.Module):
        def __init__(self, conv):
            super().__init__()
            assert conv.padding[0] == 0 and conv.padding_mode == "zeros" and conv.dilation == (1, 1, 1) and conv.groups == 1
            o, c, kt, kh, kw = conv.weight.shape
            self.kt, self.stride_t = kt, conv.stride[0]
            self.stride, self.padding = conv.stride[1:], conv.padding[1:]
            self.weight = torch.nn.Parameter(conv.weight.detach().permute(0, 2, 1, 3, 4).reshape(o, kt * c, kh, kw))
            self.bias = conv.bias

        def forward(self, x):
            taps = [x[:, :, k : (k - self.kt + 1) or None : self.stride_t] for k in range(self.kt)]
            x = torch.cat(taps, dim=1).permute(0, 2, 1, 3, 4)
            y = F.conv2d(x.flatten(0, 1), self.weight, self.bias, self.stride, self.padding)
            return y.unflatten(0, (x.shape[0], -1)).permute(0, 2, 1, 3, 4)

    swapped = 0
    if conv3d_as_2d:
        for name, mod in list(vae.named_modules()):
            if isinstance(mod, torch.nn.Conv3d):
                parent, _, attr = name.rpartition(".")
                setattr(vae.get_submodule(parent), attr, Conv3dAs2d(mod))
                swapped += 1
    channels = vae.config.latent_channels
    example = torch.randn(1, channels, 3, 5, 8)  # differs from every benchmark shape: a baked-in dim would show up
    with torch.no_grad():
        m = ov.convert_model(DecoderWrapper(vae), example_input=example, input=[ov.PartialShape([1, channels, -1, -1, -1])])
    m.outputs[0].get_tensor().set_names({"sample"})
    ov.save_model(m, ir_path, compress_to_fp16=False)
    return swapped


def read_clip(path, frames, height, width):
    """First `frames` frames of the video, center-cropped + Lanczos-resized: uint8 [F,H,W,3].

    A clip that is too short is extended by playing it forward-backward-forward... (ping-pong), which keeps the
    motion continuous. Returns (frames, how many frames the file has).
    """
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(path, format="ffmpeg")
    src, in_file = [], 0
    for fr in reader:
        in_file += 1
        if len(src) == frames:
            continue  # only counting the rest
        h, w = fr.shape[:2]
        ch = min(h, int(round(w * height / width)))
        cw = min(w, int(round(ch * width / height)))
        top, left = (h - ch) // 2, (w - cw) // 2
        src.append(np.asarray(Image.fromarray(fr[top : top + ch, left : left + cw]).resize((width, height), Image.LANCZOS)))
    reader.close()
    n = len(src)
    cycle = list(range(n)) + list(range(n - 2, 0, -1))  # 0..n-1, n-2..1, then again
    return np.stack([src[cycle[i % len(cycle)]] for i in range(frames)]), in_file


def tiling_config(a):
    return TilingConfig(
        tile_sample_min_height=a.tile_hw[0], tile_sample_min_width=a.tile_hw[0],
        tile_sample_stride_height=a.tile_hw[1], tile_sample_stride_width=a.tile_hw[1],
        tile_sample_min_num_frames=a.tile_t[0], tile_sample_stride_num_frames=a.tile_t[1],
        first_frame_rule=a.rule)


def job_tiny(a):
    """Smoke only: a small random-weight VAE of the same architecture (same 32x / 8x compression)."""
    import torch
    from diffusers import AutoencoderKLLTXVideo

    save(phase="build")
    torch.manual_seed(0)
    ch = (16, 32, 64, 64)
    AutoencoderKLLTXVideo(block_out_channels=ch, decoder_block_out_channels=ch).save_pretrained(a.vae_dir)
    save(phase="done")


def job_convert(a):
    save(phase="load")
    vae = load_vae(a.vae_dir, "decoder", a.threads)
    save(phase="convert")
    t = time.perf_counter()
    swapped = convert_decoder(vae, a.ir, a.conv3d_as_2d)
    save(phase="done", convert_s=round(time.perf_counter() - t, 2), conv3d_rewritten_as_2d=swapped,
         ir_bytes=os.path.getsize(a.ir[:-4] + ".bin"))


def job_encode(a):
    """Real latents: encode the test clip with the PyTorch encoder (posterior mean)."""
    import torch

    save(phase="clip")
    clip, src_frames = read_clip(a.clip, a.frames, a.height, a.width)
    video = torch.from_numpy(clip).float().div(127.5).sub(1.0).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    save(phase="load", clip_frames_in_file=src_frames, clip_extended_by_ping_pong=src_frames < a.frames,
         video_shape=list(video.shape))
    vae = load_vae(a.vae_dir, "encoder", a.threads)
    if a.tiled_encode:  # fallback when the one-shot encode does not fit in memory (diffusers defaults)
        vae.enable_tiling()
        vae.use_framewise_encoding = True
    save(phase="encode", tiled_encode=a.tiled_encode)
    t = time.perf_counter()
    with torch.no_grad():
        z = vae.encode(video).latent_dist.mode().numpy()
    np.save(a.save, z)
    save(phase="done", encode_s=round(time.perf_counter() - t, 2), latent_shape=list(z.shape),
         latent_std=float(z.std()))


def job_decode(a):
    """One decode of the first --frames frames of the latents. Backends:
    ov         our TiledDecoder, each tile decoded by the OpenVINO IR
    pt         our TiledDecoder, each tile decoded by the PyTorch decoder (same algorithm, other runtime)
    diffusers  diffusers' own tiling in PyTorch: the reference the port is checked against
    """
    z = np.load(a.latents)[:, :, : (a.frames - 1) // 8 + 1]
    cfg = tiling_config(a)
    dec = None
    save(phase="load")
    t = time.perf_counter()
    if a.backend == "ov":
        ov_dec = OVDecoder(a.ir, a.threads)
        dec = TiledDecoder(ov_dec, cfg)
        threads = int(ov_dec.compiled.get_property("INFERENCE_NUM_THREADS"))
        run = lambda: dec.decode(z, a.mode)
    else:
        import torch

        vae = load_vae(a.vae_dir, "decoder", a.threads)
        threads = torch.get_num_threads()
        if a.backend == "pt":
            dec = TiledDecoder(lambda lat: vae.decoder(torch.from_numpy(lat), None).numpy(), cfg)
            call = lambda: dec.decode(z, a.mode)
        else:
            assert a.rule == "diffusers", "diffusers has no other first-frame rule"
            vae.use_tiling = a.mode in ("spatial", "spatial+temporal")
            vae.use_framewise_decoding = a.mode in ("temporal", "spatial+temporal")
            vae.tile_sample_min_height = vae.tile_sample_min_width = a.tile_hw[0]
            vae.tile_sample_stride_height = vae.tile_sample_stride_width = a.tile_hw[1]
            vae.tile_sample_min_num_frames, vae.tile_sample_stride_num_frames = a.tile_t
            call = lambda: vae.decode(torch.from_numpy(z), None, return_dict=False)[0].numpy()

        def run():
            with torch.no_grad():
                return call()

    save(phase="decode", load_s=round(time.perf_counter() - t, 2), rss_after_load_gb=rss_gb(), threads=threads,
         latent_shape=list(z.shape), layout=TiledDecoder(None, cfg).seam_layout(z.shape, a.mode))
    t = time.perf_counter()
    out = run()
    decode_s = time.perf_counter() - t
    save(phase="save", decode_s=round(decode_s, 2), output_shape=list(out.shape), rss_after_decode_gb=rss_gb(),
         decoder_calls=dec.calls if dec else None)
    if a.save:
        np.save(a.save, out)
    save(phase="done")


def job_metrics(a):
    """Compare two saved decodes (memory-mapped, frame by frame)."""
    from metrics import compare

    save(phase="metrics")
    ref, test = np.load(a.ref, mmap_mode="r"), np.load(a.test, mmap_mode="r")
    assert ref.shape == test.shape, (ref.shape, test.shape)
    save(phase="done", **compare(ref, test, json.loads(a.layout) if a.layout else None))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["tiny", "convert", "encode", "decode", "metrics"])
    ap.add_argument("--out")
    ap.add_argument("--vae-dir")
    ap.add_argument("--ir")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--conv3d-as-2d", action="store_true")
    ap.add_argument("--clip")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--tiled-encode", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--latents")
    ap.add_argument("--backend", choices=["ov", "pt", "diffusers"], default="ov")
    ap.add_argument("--mode", choices=MODES, default="none")
    ap.add_argument("--rule", default="drop_first")
    ap.add_argument("--tile-hw", type=int, nargs=2, default=[512, 448], metavar=("MIN", "STRIDE"))
    ap.add_argument("--tile-t", type=int, nargs=2, default=[16, 8], metavar=("MIN", "STRIDE"))
    ap.add_argument("--ref")
    ap.add_argument("--test")
    ap.add_argument("--layout")
    a = ap.parse_args()
    OUT = a.out

    try:  # Linux: make this process the kernel OOM killer's first choice, not the parent
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    {"tiny": job_tiny, "convert": job_convert, "encode": job_encode, "decode": job_decode, "metrics": job_metrics}[a.job](a)


# The guard matters: on macOS/Windows a helper process spawned during conversion re-imports this file.
if __name__ == "__main__":
    main()
