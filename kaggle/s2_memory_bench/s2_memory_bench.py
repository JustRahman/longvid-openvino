"""Step 2 of LongVid-OV: measure how much memory the LTX-Video VAE decode needs as the video gets longer.

Kaggle "script" kernel (CPU, internet on). Single file: the per-config worker is embedded below (WORKER_SRC)
and written to disk at start, because every measurement runs in a FRESH subprocess.

What it does
  1. pip installs pinned deps (Kaggle only), records versions / lscpu / RAM -> s2_env.txt
  2. downloads ONLY vae/ of Lightricks/LTX-Video, converts the DECODER to OpenVINO IR (FP32, dynamic
     frames/H/W, NATIVE Conv3d; --conv3d-as-2d re-enables the ARM-only workaround from step 1)
  3. OpenVINO grid: 512x320, 768x512, 1216x704 (WxH) x frames 9..257, one untiled decode per config
  4. reference: diffusers PyTorch decode at 768x512, frames 25/49/97: untiled, enable_tiling() (spatial
     tiles), enable_tiling() + use_framewise_decoding (spatial + temporal tiles)
  5. writes s2_results.csv/.json (after every config), s2_memory_vs_frames.png, s2_env.txt, s2_report.md

How memory is measured: the parent polls the child's RSS every 50 ms (plus the kernel's own high-water mark
VmHWM on Linux, which cannot miss a spike) and the system swap. A child is "oom" if the kernel SIGKILLs it,
if it dies with an allocation error, or if its RSS passes the watchdog limit (RAM limit - 1.5 GB; the
watchdog kills it first so the kernel OOM killer does not take the whole Kaggle session down).

Latents are random normal, scaled with the VAE's real per-channel latents_mean / latents_std (what the
pipeline feeds the decoder). Decode memory and time do not depend on the latent VALUES: the decoder is a
fixed stack of convolutions / norms / pixel-shuffles, so the same ops run on the same tensor shapes whatever
the numbers are. Only the shape (frames, H, W) matters, and that is what the grid varies.

Local smoke test (same code path; tiny shapes and a tiny random-weight VAE of the same architecture, because
the real decoder passes 4 GB in OpenVINO on an ARM Mac even at 9 frames 256x160; 6 threads; watchdog at 3.7 GB):
  OMP_NUM_THREADS=6 .venv/bin/python kaggle/s2_memory_bench/s2_memory_bench.py --smoke --conv3d-as-2d
  (--vae-dir models/ltx_vae_pt/vae runs the smoke shapes on the real VAE instead)
The full grid refuses to run outside Kaggle.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = "Lightricks/LTX-Video"
PINS = ["openvino==2026.4.1", "openvino-genai==2026.4.1.0"]
RESOLUTIONS = [(512, 320), (768, 512), (1216, 704)]  # (W, H), multiples of 32
FRAMES = [9, 25, 49, 97, 161, 257]  # 8k+1; latent frames = (f-1)/8+1
PT_RES, PT_FRAMES = (768, 512), [25, 49, 97]
PT_TILINGS = ["none", "spatial", "spatial+temporal"]
SAMPLE_S = 0.05
WATCHDOG_HEADROOM_GB = 1.5
MIN_AVAIL_GB = 1.0  # watchdog also fires when the whole system has less than this available
GB = 2**30

WORKER_SRC = r'''
"""s2 worker: one job per process. Modes: tiny | convert | verify | ov | pt. Writes progress/results as JSON to --out."""
import argparse, json, os, time

import numpy as np
import psutil

a = None  # parsed args, set in main()
R = {}


def rss_gb():
    return round(psutil.Process().memory_info().rss / 2**30, 3)


def save(**kw):
    R.update(kw)
    R.setdefault("rss_at_phase_gb", {})[R["phase"]] = rss_gb()  # RSS when each phase starts
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f)
    os.replace(tmp, a.out)


def make_latents(frames, height, width):
    """Random normal latents de-normalised like LTXPipeline does: z * std / scaling_factor + mean."""
    s = np.load(a.stats)
    mean, std = s["mean"].astype(np.float32), s["std"].astype(np.float32)
    shape = (1, mean.size, (frames - 1) // 8 + 1, height // 32, width // 32)
    z = np.random.default_rng(0).standard_normal(shape, dtype=np.float32)
    return z * (std / float(s["scaling_factor"])).reshape(1, -1, 1, 1, 1) + mean.reshape(1, -1, 1, 1, 1)


def ov_config():
    cfg = {"INFERENCE_PRECISION_HINT": "f32"}  # f32 everywhere (some CPUs default to bf16/f16)
    if a.threads:
        cfg["INFERENCE_NUM_THREADS"] = a.threads
    return cfg


def load_vae():
    """Decoder half of the diffusers LTX VAE (PyTorch, float32, eval); the encoder is never loaded.

    Built on the meta device and filled straight from the safetensors file, so loading holds one copy of the
    decoder weights. Plain from_pretrained holds two (random init + checkpoint) when accelerate is missing,
    and that load spike would pollute the peak-RSS measurement.
    """
    import torch
    from diffusers import AutoencoderKLLTXVideo
    from safetensors import safe_open

    if a.threads:
        torch.set_num_threads(a.threads)
    with torch.device("meta"):
        vae = AutoencoderKLLTXVideo.from_config(AutoencoderKLLTXVideo.load_config(a.vae_dir))
    if vae.config.timestep_conditioning or any(vae.config.decoder_inject_noise):
        raise NotImplementedError("this VAE needs a decode timestep / noise injection (LTX 0.9.1+)")
    vae.encoder = None
    with safe_open(os.path.join(a.vae_dir, "diffusion_pytorch_model.safetensors"), "pt") as f:
        weights = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("encoder.")}
    vae.load_state_dict(weights, strict=True, assign=True)
    on_meta = [n for n, t in [*vae.named_parameters(), *vae.named_buffers()] if t.device.type == "meta"]
    if on_meta:
        raise RuntimeError(f"tensors not filled from the checkpoint: {on_meta[:5]}")
    return vae.eval()


def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tiny", "convert", "verify", "ov", "pt"])
    ap.add_argument("--vae-dir")
    ap.add_argument("--ir")
    ap.add_argument("--stats")
    ap.add_argument("--out")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tiling", default="none")
    ap.add_argument("--tile-min", type=int, default=0)
    ap.add_argument("--conv3d-as-2d", action="store_true")
    a = ap.parse_args()

    try:  # Linux: make this process the kernel OOM killer's first choice, not the parent
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass

    if a.mode == "convert":
        import openvino as ov
        import torch
        import torch.nn.functional as F

        class DecoderWrapper(torch.nn.Module):
            """latent [1,128,f,h,w] -> video [1,3,F,H,W]. Same call diffusers makes: vae.decode(z, None)."""

            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, latent):
                return self.vae.decode(latent, None, return_dict=False)[0]

        class Conv3dAs2d(torch.nn.Module):
            """Exact nn.Conv3d as ONE Conv2d over time-shifted copies stacked on channels (step 1, ARM-only fix)."""

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

        save(phase="load")
        vae = load_vae()
        np.savez(a.stats, mean=vae.latents_mean.numpy(), std=vae.latents_std.numpy(),
                 scaling_factor=np.float32(vae.config.scaling_factor))
        channels = vae.config.latent_channels

        # PyTorch reference on a tiny shape, taken from the stock model before any rewrite
        save(phase="reference")
        z = make_latents(9, 128, 192)
        with torch.no_grad():
            ref = vae.decode(torch.from_numpy(z), None, return_dict=False)[0].numpy()

        save(phase="rewrite")
        swapped = 0
        if a.conv3d_as_2d:
            for name, mod in list(vae.named_modules()):
                if isinstance(mod, torch.nn.Conv3d):
                    parent, _, attr = name.rpartition(".")
                    setattr(vae.get_submodule(parent), attr, Conv3dAs2d(mod))
                    swapped += 1
        save(phase="convert", conv3d_rewritten_as_2d=swapped)
        t = time.perf_counter()
        # trace shape differs from the check shape and from every benchmark shape, so a baked-in dim would show up
        example = torch.randn(1, channels, 3, 5, 8)
        with torch.no_grad():
            m = ov.convert_model(DecoderWrapper(vae), example_input=example, input=[ov.PartialShape([1, channels, -1, -1, -1])])
        m.outputs[0].get_tensor().set_names({"sample"})
        ov.save_model(m, a.ir, compress_to_fp16=False)
        convert_s = time.perf_counter() - t
        np.savez(a.ir + ".check.npz", z=z, ref=ref)  # for the verify job (own process: keeps peak memory low)
        save(phase="done", convert_s=round(convert_s, 2), ir_bytes=os.path.getsize(a.ir[:-4] + ".bin"))

    elif a.mode == "tiny":  # smoke only: same architecture and layer counts, 1/8 of the channels, random weights
        import torch
        from diffusers import AutoencoderKLLTXVideo

        save(phase="build")
        torch.manual_seed(0)
        ch = (16, 32, 64, 64)
        AutoencoderKLLTXVideo(block_out_channels=ch, decoder_block_out_channels=ch).save_pretrained(a.vae_dir)
        save(phase="done")

    elif a.mode == "verify":  # IR vs the PyTorch reference on a tiny shape + which conv kernels the CPU plugin picked
        import openvino as ov

        save(phase="verify")
        check = np.load(a.ir + ".check.npz")
        z, ref = check["z"], check["ref"]
        core = ov.Core()
        cm = core.compile_model(a.ir, "CPU", {**ov_config(), "PERF_COUNT": "YES"})
        req = cm.create_infer_request()
        save(phase="check-decode")
        req.infer({0: z})
        out = req.get_output_tensor(0).data
        conv = [p for p in req.get_profiling_info() if p.node_type == "Convolution"]
        save(
            phase="done",
            decoder_in=str(cm.input(0).partial_shape), decoder_out=str(cm.output(0).partial_shape),
            check_latent_shape=list(z.shape), check_output_shape=list(out.shape),
            check_shape_ok=list(out.shape) == list(ref.shape),
            check_max_abs_diff_vs_pytorch=float(np.abs(out - ref).max()) if out.shape == ref.shape else None,
            conv_layers=len(conv), conv_exec_types=sorted({p.exec_type for p in conv}),
            ov_cpu_name=str(core.get_property("CPU", "FULL_DEVICE_NAME")),
            ov_cpu_capabilities=[str(c) for c in core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")],
            inference_precision=str(cm.get_property("INFERENCE_PRECISION_HINT")),
        )

    elif a.mode == "ov":
        import openvino as ov

        save(phase="compile")
        t = time.perf_counter()
        cm = ov.Core().compile_model(a.ir, "CPU", ov_config())
        load_s = time.perf_counter() - t
        req = cm.create_infer_request()
        z = make_latents(a.frames, a.height, a.width)
        save(phase="decode", compile_or_load_s=round(load_s, 2), rss_after_load_gb=rss_gb(), latent_shape=list(z.shape),
             threads=int(cm.get_property("INFERENCE_NUM_THREADS")))
        t = time.perf_counter()
        req.infer({0: z})  # single decode, includes the shape-specific setup a real pipeline also pays
        decode_s = time.perf_counter() - t
        save(phase="done", decode_s=round(decode_s, 2), output_shape=list(req.get_output_tensor(0).shape))

    else:  # pt
        import torch

        save(phase="load")
        t = time.perf_counter()
        vae = load_vae()
        load_s = time.perf_counter() - t
        if a.tiling != "none":
            kw = {}
            if a.tile_min:  # smoke only: tiny tiles so tiny shapes still take the tiled path
                kw = dict(tile_sample_min_height=a.tile_min, tile_sample_min_width=a.tile_min,
                          tile_sample_stride_height=a.tile_min * 3 // 4, tile_sample_stride_width=a.tile_min * 3 // 4)
            vae.enable_tiling(**kw)  # spatial tiles only
            if a.tiling == "spatial+temporal":
                vae.use_framewise_decoding = True  # temporal tiles; enable_tiling() does not switch this on
        z = torch.from_numpy(make_latents(a.frames, a.height, a.width))
        save(phase="decode", compile_or_load_s=round(load_s, 2), rss_after_load_gb=rss_gb(), latent_shape=list(z.shape),
             threads=torch.get_num_threads(),
             tile_settings=None if a.tiling == "none" else {
                 "min_hw": [vae.tile_sample_min_height, vae.tile_sample_min_width],
                 "stride_hw": [vae.tile_sample_stride_height, vae.tile_sample_stride_width],
                 "min_frames": vae.tile_sample_min_num_frames, "stride_frames": vae.tile_sample_stride_num_frames,
                 "framewise": bool(vae.use_framewise_decoding)})
        t = time.perf_counter()
        with torch.no_grad():
            out = vae.decode(z, None, return_dict=False)[0]
        decode_s = time.perf_counter() - t
        save(phase="done", decode_s=round(decode_s, 2), output_shape=list(out.shape))


# The guard matters: on macOS/Windows a helper process spawned during conversion re-imports this file,
# and without it that helper would start a second full conversion.
if __name__ == "__main__":
    main()
'''

T0 = time.time()


def log(msg):
    el = int(time.time() - T0)
    print(f"[s2 {time.strftime('%H:%M:%S')} +{el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d}] {msg}", flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"not available ({e!r})"


def pip(*pkgs):
    log("pip install " + " ".join(pkgs))
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs]).returncode == 0


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def mem_limit_bytes(psutil):
    """Usable RAM: physical total, or the container's cgroup limit if that is lower."""
    limit = psutil.virtual_memory().total
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            limit = min(limit, int(Path(p).read_text().strip()))
        except (OSError, ValueError):  # missing file or "max"
            pass
    return limit


def vm_hwm(pid):
    """Linux: the kernel's peak-RSS high-water mark for pid (bytes), else 0."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def run_child(psutil, worker, args, out_json, log_path, timeout_s, kill_rss, threads):
    """Run one worker job in a fresh process, sampling its RSS every 50 ms. Returns the result row fields."""
    Path(out_json).unlink(missing_ok=True)
    env = {**os.environ, "USE_TF": "0", "USE_FLAX": "0"}  # keep transformers from importing TensorFlow / JAX
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    swap0 = psutil.swap_memory().used
    peak = peak_swap = 0
    min_avail = psutil.virtual_memory().available
    killed = None
    t = time.perf_counter()
    with open(log_path, "w") as lf:
        proc = subprocess.Popen([sys.executable, str(worker), *map(str, args), "--out", str(out_json)],
                                stdout=lf, stderr=subprocess.STDOUT, env=env)
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss
            except psutil.Error:
                rss = 0
            peak = max(peak, rss, vm_hwm(proc.pid))
            peak_swap = max(peak_swap, psutil.swap_memory().used - swap0)
            min_avail = min(min_avail, psutil.virtual_memory().available)
            if rss > kill_rss or psutil.virtual_memory().available < MIN_AVAIL_GB * GB:
                killed = "oom"
            elif time.perf_counter() - t > timeout_s:
                killed = "timeout"
            if killed:
                proc.kill()
                proc.wait()
                break
            time.sleep(SAMPLE_S)
    wall = time.perf_counter() - t
    try:
        res = json.loads(Path(out_json).read_text())
    except (OSError, ValueError):
        res = {}
    tail = Path(log_path).read_text(errors="replace").strip().splitlines()[-15:]
    phase = res.get("phase", "start")
    if killed == "oom":
        status, detail = "oom", (f"watchdog kill during {phase}: RSS {rss / GB:.1f} GB (limit {kill_rss / GB:.1f} GB), "
                                 f"system available {min_avail / GB:.1f} GB")
    elif killed == "timeout":
        status, detail = "timeout", f"killed after {timeout_s:.0f} s during {phase}"
    elif proc.returncode == 0 and phase == "done":
        status, detail = "ok", ""
    elif proc.returncode == -9:
        status, detail = "oom", f"SIGKILL during {phase} (kernel OOM killer)"
    elif any(k in line for line in tail for k in ("bad_alloc", "MemoryError", "allocate", "not enough memory")):
        status, detail = "oom", f"allocation failure during {phase}: {tail[-1][:200]}"
    else:
        status, detail = "error", f"exit code {proc.returncode} during {phase}: {(tail or ['no output'])[-1][:200]}"
    res.pop("phase", None)
    return {"status": status, "detail": detail, "peak_rss_gb": round(peak / GB, 3), "wall_s": round(wall, 2),
            "peak_swap_delta_gb": round(peak_swap / GB, 3), "min_sys_avail_gb": round(min_avail / GB, 2), **res}


COLUMNS = ["backend", "tiling", "width", "height", "frames", "latent_frames", "status", "peak_rss_gb",
           "rss_after_load_gb", "wall_s", "compile_or_load_s", "decode_s", "peak_swap_delta_gb", "min_sys_avail_gb",
           "threads", "latent_shape", "output_shape", "detail", "finished_at"]


def save_results(out, rows, meta):
    import pandas as pd

    pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out / "s2_results.csv", index=False)
    (out / "s2_results.json").write_text(json.dumps({**meta, "results": rows}, indent=2))


def series_of(rows):
    """{(backend, tiling, width, height): [rows sorted by frames]} in first-seen order."""
    s = {}
    for r in rows:
        s.setdefault((r["backend"], r["tiling"], r["width"], r["height"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["frames"]) for k, v in s.items()}


def series_label(key):
    backend, tiling, w, h = key
    name = "OpenVINO untiled" if backend == "openvino" else {
        "none": "PyTorch untiled", "spatial": "PyTorch enable_tiling() (spatial)",
        "spatial+temporal": "PyTorch enable_tiling() + framewise (spatial+temporal)"}[tiling]
    return f"{name}, {w}x{h}"


def plot(out, rows, meta):
    """Peak RSS vs output frames. Colour = resolution, line style + marker = backend / tiling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#1f1f1e", "#6b6a63", "#fcfcfb"
    colors = dict(zip(sorted({(r["width"], r["height"]) for r in rows}, key=lambda wh: wh[0] * wh[1]),
                      ["#2a78d6", "#eb6834", "#1baf7a"]))
    styles = {("openvino", "none"): ("-", "o"), ("pytorch", "none"): ("--", "s"),
              ("pytorch", "spatial"): (":", "^"), ("pytorch", "spatial+temporal"): ("-.", "D")}
    fig, ax = plt.subplots(figsize=(10, 6.8), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    for key, srows in series_of(rows).items():
        ls, mk = styles[key[:2]]
        col = colors[key[2:]]
        ok = [r for r in srows if r["status"] == "ok"]
        if ok:
            ax.plot([r["frames"] for r in ok], [r["peak_rss_gb"] for r in ok], ls=ls, marker=mk, color=col, lw=2, ms=7,
                    mec=surface, mew=1.5, label=series_label(key))
            if key[0] == "openvino":  # direct label on the main series
                ax.annotate(f"{key[2]}x{key[3]}", (ok[-1]["frames"], ok[-1]["peak_rss_gb"]), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=9, color=ink)
        for r in srows:  # runs that did not finish: mark where memory stood when the run was stopped
            if r["status"] in ("oom", "timeout", "error"):
                ax.plot(r["frames"], r["peak_rss_gb"], marker="x", color=col, ms=10, mew=2.5, ls="none")
                ax.annotate({"oom": "OOM", "timeout": "timeout", "error": "error"}[r["status"]],
                            (r["frames"], r["peak_rss_gb"]), xytext=(0, 9), textcoords="offset points", ha="center",
                            fontsize=9, color=ink)
    ax.axhline(meta["mem_limit_gb"], color=muted, lw=1, ls=(0, (4, 3)))
    ax.annotate(f"RAM limit {meta['mem_limit_gb']:.1f} GB", (1, meta["mem_limit_gb"]), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=9, color=muted)
    frames = sorted({r["frames"] for r in rows})
    ax.set_xticks(frames)
    ax.set_xlim(0, max(frames) * 1.12)
    ax.set_ylim(0, max(meta["mem_limit_gb"], max(r.get("peak_rss_gb") or 0 for r in rows)) * 1.08)
    ax.set_xlabel("output frames", color=ink)
    ax.set_ylabel("peak process RSS (GB)", color=ink)
    ax.set_title("LTX-Video VAE decode: peak memory vs video length (CPU)\n"
                 "x = run stopped (OOM / timeout) at that RSS", color=ink, fontsize=11, loc="left")
    ax.grid(True, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c1")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.11), ncol=2, labelcolor=ink,
                  handlelength=3.5)  # below the axes: data can reach the RAM limit line
    fig.tight_layout()
    fig.savefig(out / "s2_memory_vs_frames.png", facecolor=surface)
    plt.close(fig)


def cell(r):
    if r is None:
        return "-"
    if r["status"] == "ok":
        return f"{r['peak_rss_gb']:.2f} GB / {r['decode_s']:.1f} s"
    if r["status"] == "skipped":
        return "skipped"
    return f"**{r['status'].upper()}** (RSS {r['peak_rss_gb']:.2f} GB when stopped, {r['wall_s']:.0f} s)"


def write_report(out, rows, meta):
    """s2_report.md from measured values only; no hand-typed numbers."""
    S = series_of(rows)
    cv = meta.get("convert", {})
    L = [f"# Step 2 report: LTX-Video VAE decode memory vs video length{' (SMOKE TEST, tiny shapes)' if meta['smoke'] else ''}",
         "",
         f"Generated by `s2_memory_bench.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}; total runtime "
         f"{(time.time() - T0) / 60:.1f} min. Every number below was measured in this run (raw: `s2_results.csv/json`).",
         ""]
    if meta["smoke"]:
        L += ["**This is a local smoke test of the code path. Its numbers say nothing about the real problem.**", ""]
    L += ["## Setup",
          f"- Machine: {meta['cpu_model']}, {meta['nproc']} logical CPUs, RAM limit {meta['mem_limit_gb']:.1f} GB, "
          f"swap {meta['swap_total_gb']:.1f} GB. Full details: `s2_env.txt`.",
          f"- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()) + ".",
          "- Model: " + ("TINY RANDOM-WEIGHT VAE of the same architecture (smoke test only)" if meta.get("tiny_vae")
                         else f"`{REPO}` `vae/` only") + f", decoder converted to OpenVINO IR (FP32, dynamic frames/H/W), "
          + ("every Conv3d rewritten as Conv2d (`--conv3d-as-2d`, ARM workaround)." if meta["conv3d_as_2d"]
             else "native Conv3d."),
          ]
    if cv.get("status") == "ok":
        diff = cv.get("check_max_abs_diff_vs_pytorch")
        L += [f"- Conversion: {cv.get('convert_s')} s, peak RSS {cv['peak_rss_gb']:.2f} GB, IR weights "
              f"{cv.get('ir_bytes', 0) / GB:.2f} GB, signature `{cv.get('decoder_in')}` -> `{cv.get('decoder_out')}`.",
              f"- IR check on latents {cv.get('check_latent_shape')}: output {cv.get('check_output_shape')}, max abs diff vs "
              f"PyTorch {'n/a (shape mismatch)' if diff is None else f'{diff:.3e}'}. "
              f"{cv.get('conv_layers')} Convolution layers ran with exec type(s) {cv.get('conv_exec_types')} "
              f"(OpenVINO CPU: {cv.get('ov_cpu_name')}; inference precision {cv.get('inference_precision')})."]
    elif cv:
        L += [f"- Conversion: {'reused existing IR' if cv.get('status') == 'reused' else 'FAILED: ' + str(cv.get('detail'))}."]
    L += ["",
          "## Method",
          f"- Each config runs in a fresh subprocess: load/compile the decoder, then ONE decode of random latents "
          f"(random normal x real per-channel `latents_std` + `latents_mean`). Decode memory and time depend on the tensor "
          f"shapes, not on the latent values: the decoder runs the same convolutions on the same shapes for any input.",
          f"- Peak RSS = max of the child's RSS sampled every {SAMPLE_S * 1000:.0f} ms and (Linux) the kernel high-water "
          f"mark VmHWM. It includes the runtime, the weights and the decoded output tensor.",
          f"- `OOM` = kernel SIGKILL, an allocation error, or the watchdog stopping the child when its RSS passed "
          f"{meta['kill_rss_gb']:.1f} GB (default: RAM limit - {WATCHDOG_HEADROOM_GB} GB) or system available memory fell "
          f"under {MIN_AVAIL_GB} GB. For OOM / timeout rows the RSS shown is "
          f"where the run was stopped, a lower bound on what it needed.",
          f"- `timeout` = still running after {meta['run_timeout_s'] / 60:.0f} min. After the first OOM / timeout in a series, "
          f"longer videos are `skipped`. `skipped` also covers configs dropped by the {meta['budget_s'] / 3600:.1f} h budget "
          f"(see `detail` in the CSV).",
          f"- Latent frames = (frames - 1) / 8 + 1; latent H, W = H / 32, W / 32.",
          ""]

    def table(keys, title):
        if not keys:
            return
        frames = sorted({r["frames"] for k in keys for r in S[k]})
        L.extend([f"## {title}", "Cell = peak RSS / decode time.", "",
                  "| output frames | latent frames | " + " | ".join(series_label(k) for k in keys) + " |",
                  "|---|---|" + "---|" * len(keys)])
        for f in frames:
            by = [next((r for r in S[k] if r["frames"] == f), None) for k in keys]
            L.append(f"| {f} | {(f - 1) // 8 + 1} | " + " | ".join(cell(r) for r in by) + " |")
        L.append("")

    ov_keys = [k for k in S if k[0] == "openvino"]
    pt_keys = [k for k in S if k[0] == "pytorch"]
    table(ov_keys, "OpenVINO, untiled decode")
    table([k for k in ov_keys if k[2:] in {p[2:] for p in pt_keys}] + pt_keys, "Reference: diffusers PyTorch, untiled vs tiled")

    L += ["## What the numbers say", ""]
    for k in ov_keys + pt_keys:
        ok = [r for r in S[k] if r["status"] == "ok"]
        bad = next((r for r in S[k] if r["status"] in ("oom", "timeout", "error")), None)
        line = f"- **{series_label(k)}**: "
        if ok:
            a, b = ok[0], ok[-1]
            line += f"longest video decoded: {b['frames']} frames at {b['peak_rss_gb']:.2f} GB peak, {b['decode_s']:.1f} s"
            if b is not a:
                line += (f"; from {a['frames']} to {b['frames']} frames peak RSS went {a['peak_rss_gb']:.2f} -> "
                         f"{b['peak_rss_gb']:.2f} GB ({b['peak_rss_gb'] / a['peak_rss_gb']:.1f}x), "
                         f"{(b['peak_rss_gb'] - a['peak_rss_gb']) / (b['frames'] - a['frames']) * 8:.2f} GB per 8 extra frames on average")
            line += ". "
        else:
            line += "no config completed. "
        if bad:
            line += f"First failure: {bad['frames']} frames, {bad['status'].upper()} ({bad['detail']})."
        elif not any(r["status"] == "skipped" for r in S[k]):
            line += "No failure in the tested range."
        L.append(line)
    # tiling effect where both the untiled and a tiled PyTorch run finished
    for k in pt_keys:
        if k[1] == "none":
            continue
        base = S.get(("pytorch", "none", k[2], k[3]), [])
        for r in S[k]:
            b = next((x for x in base if x["frames"] == r["frames"] and x["status"] == "ok"), None)
            if b and r["status"] == "ok":
                L.append(f"- Tiling ({k[1]}) at {k[2]}x{k[3]}, {r['frames']} frames: peak RSS {b['peak_rss_gb']:.2f} -> "
                         f"{r['peak_rss_gb']:.2f} GB ({(r['peak_rss_gb'] / b['peak_rss_gb'] - 1) * 100:+.0f}%), decode "
                         f"{b['decode_s']:.1f} -> {r['decode_s']:.1f} s.")
    bad_shape = [r for r in rows if r["status"] == "ok" and r.get("output_shape") != [1, 3, r["frames"], r["height"], r["width"]]]
    L += ["",
          "## Notes",
          "- Single run per config, no repeats: treat times as indicative, memory as the result.",
          "- PyTorch runs load only the decoder half of the diffusers VAE, so both backends hold the same model.",
          "- diffusers `enable_tiling()` only turns on spatial tiles; temporal tiling is the separate "
          "`use_framewise_decoding` flag, so both variants were measured.",
          "- The OpenVINO runs decode in one shot, as a plain compiled decoder IR does. Whether OpenVINO GenAI's own "
          "pipeline adds any tiling was not tested in this step.",
          "- Output shape check: " + ("all completed runs returned [1, 3, frames, H, W]." if not bad_shape else "MISMATCH in "
                                      + "; ".join(f"{r['backend']} {r['tiling']} {r['width']}x{r['height']} {r['frames']} frames -> "
                                                  f"{r.get('output_shape')}" for r in bad_shape)),
          f"- Peak swap growth during any run: {max((r.get('peak_swap_delta_gb') or 0) for r in rows) if rows else 0:.2f} GB.",
          "",
          "![peak RSS vs frames](s2_memory_vs_frames.png)",
          ""]
    (out / "s2_report.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny local run of the same code path")
    ap.add_argument("--conv3d-as-2d", action="store_true", help="export Conv3d as Conv2d (ARM workaround, off by default)")
    ap.add_argument("--vae-dir", help="existing vae/ folder (skips the download)")
    ap.add_argument("--ir", help="existing decoder IR .xml with latents_stats.npz next to it (skips conversion)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--threads", type=int, help="inference threads (0 = all)")
    ap.add_argument("--kill-rss-gb", type=float, help="watchdog: kill a run whose RSS passes this")
    a, _ = ap.parse_known_args()  # Kaggle runs scripts through a kernel launcher with its own argv
    smoke = a.smoke or os.environ.get("S2_SMOKE") == "1"
    on_kaggle = Path("/kaggle/working").is_dir()
    if not smoke and not on_kaggle:
        sys.exit("The full benchmark only runs on Kaggle. Use --smoke locally.")

    here = Path(globals().get("__file__", "s2_memory_bench.py")).resolve().parent
    if smoke:
        root = here.parent.parent
        out = Path(a.out) if a.out else root / "results" / "s2_smoke"
        work = root / "models" / "s2_smoke"
        vae_dir = Path(a.vae_dir) if a.vae_dir else None
        threads = a.threads if a.threads is not None else 6
        budget_s, run_timeout_s, convert_timeout_s = 9 * 60, 150, 300
    else:
        out = Path(a.out) if a.out else Path("/kaggle/working")
        work = Path("/kaggle/temp/s2_work") if Path("/kaggle/temp").is_dir() else Path("/tmp/s2_work")
        vae_dir = Path(a.vae_dir) if a.vae_dir else None
        threads = a.threads or 0
        budget_s, run_timeout_s, convert_timeout_s = 5 * 3600, 25 * 60, 30 * 60
    for d in (out, out / "logs", work):
        d.mkdir(parents=True, exist_ok=True)

    # 1. environment ----------------------------------------------------------------------------
    if on_kaggle and not smoke:
        if not pip(*PINS):
            log("pinned OpenVINO not installable, falling back to latest stable")
            pip("openvino", "openvino-genai")
        pip("diffusers==0.39.0", "huggingface_hub", "safetensors", "psutil", "pandas", "matplotlib")
        if importlib.util.find_spec("torch") is None:  # Kaggle ships torch; only fetch the CPU wheel if it is missing
            pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    import psutil

    limit = mem_limit_bytes(psutil)
    kill_rss = (a.kill_rss_gb * GB) if a.kill_rss_gb else (3.7 * GB if smoke else limit - WATCHDOG_HEADROOM_GB * GB)
    lscpu = sh(["lscpu"])
    cpu_model = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")),
                     platform.processor() or platform.machine())
    meta = {
        "smoke": smoke, "conv3d_as_2d": a.conv3d_as_2d, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_model": cpu_model, "nproc": os.cpu_count(), "threads_requested": threads,
        "ram_total_gb": round(psutil.virtual_memory().total / GB, 2), "mem_limit_gb": round(limit / GB, 2),
        "swap_total_gb": round(psutil.swap_memory().total / GB, 2), "kill_rss_gb": round(kill_rss / GB, 2),
        "run_timeout_s": run_timeout_s, "budget_s": budget_s, "sample_interval_s": SAMPLE_S,
        "versions": {"python": platform.python_version(), **{p: version(p) for p in (
            "openvino", "openvino-genai", "torch", "diffusers", "numpy", "psutil", "pandas", "matplotlib")}},
    }
    env_txt = "\n".join([
        f"date: {meta['started']}", f"platform: {platform.platform()}", f"python: {sys.version}",
        *(f"{k}: {v}" for k, v in meta["versions"].items() if k != "python"),
        f"nproc: {sh(['nproc'])} (os.cpu_count {os.cpu_count()})",
        f"RAM total: {meta['ram_total_gb']} GB; usable limit (min of RAM and cgroup): {meta['mem_limit_gb']} GB; "
        f"available now: {psutil.virtual_memory().available / GB:.2f} GB; swap total: {meta['swap_total_gb']} GB",
        f"watchdog kill RSS: {meta['kill_rss_gb']} GB; inference threads requested: {threads or 'all'}",
        "", "--- lscpu ---", lscpu, ""])
    (out / "s2_env.txt").write_text(env_txt)
    print(env_txt, flush=True)

    worker = work / "s2_worker.py"
    worker.write_text(WORKER_SRC)
    rows = []

    def child(name, args, timeout_s):
        return run_child(psutil, worker, args, work / f"{name}.json", out / "logs" / f"{name}.log", timeout_s, kill_rss, threads)

    try:
        # 2. VAE download + decoder conversion ----------------------------------------------------
        if a.ir:
            ir, stats = Path(a.ir), Path(a.ir).parent / "latents_stats.npz"
            meta["convert"] = {"status": "reused", "ir": str(ir)}
            log(f"reusing IR {ir}")
        else:
            if vae_dir is None and smoke:
                # The real decoder needs more than the 4 GB local limit in OpenVINO on ARM even for 9 frames at
                # 256x160, so the smoke test runs the same code on a tiny random-weight VAE of the same architecture.
                vae_dir, meta["tiny_vae"] = work / "tiny_vae", True
                tiny = child("tiny_vae", ["tiny", "--vae-dir", vae_dir], convert_timeout_s)
                if tiny["status"] != "ok":
                    raise RuntimeError(f"tiny VAE build failed: {tiny['detail']} (see logs/tiny_vae.log)")
            elif vae_dir is None:
                from huggingface_hub import snapshot_download

                log(f"downloading {REPO} vae/ ...")
                vae_dir = Path(snapshot_download(REPO, allow_patterns=["vae/*"], local_dir=work / "ltx_vae_pt")) / "vae"
            ir, stats = work / "vae_decoder.xml", work / "latents_stats.npz"
            log(f"converting decoder to OpenVINO IR ({'Conv3d as Conv2d' if a.conv3d_as_2d else 'native Conv3d'}) ...")
            cv = child("convert", ["convert", "--vae-dir", vae_dir, "--ir", ir, "--stats", stats, "--threads", threads,
                                   *(["--conv3d-as-2d"] if a.conv3d_as_2d else [])], convert_timeout_s)
            if cv["status"] == "ok":
                ver = child("verify", ["verify", "--ir", ir, "--threads", threads], convert_timeout_s)
                cv = {**cv, **{k: v for k, v in ver.items() if k.startswith(("check_", "conv_", "ov_cpu", "inference_", "decoder_"))},
                      "status": ver["status"], "detail": ver["detail"] and f"IR check: {ver['detail']}"}
            meta["convert"] = cv
            log(f"convert: {cv['status']} {cv['detail']} | {cv.get('convert_s')} s, peak RSS {cv['peak_rss_gb']} GB, "
                f"max abs diff vs PyTorch {cv.get('check_max_abs_diff_vs_pytorch')}, conv exec {cv.get('conv_exec_types')}")
            if cv["status"] != "ok":
                raise RuntimeError(f"decoder conversion failed: {cv['detail']} (see logs/convert.log, logs/verify.log)")
            if not cv["check_shape_ok"] or cv["check_max_abs_diff_vs_pytorch"] > 1e-3:
                log("WARNING: IR check vs PyTorch is off; memory numbers still measured, see report")

        # 3. the grid -------------------------------------------------------------------------------
        if smoke:
            plan = [("openvino", "none", 256, 160, f) for f in (9, 17)]
            plan += [("pytorch", t, 256, 160, 17) for t in PT_TILINGS]
        else:  # the two smaller OpenVINO resolutions, then the PyTorch reference, then the largest resolution
            plan = [("openvino", "none", w, h, f) for (w, h) in RESOLUTIONS[:2] for f in FRAMES]
            plan += [("pytorch", t, *PT_RES, f) for t in PT_TILINGS for f in PT_FRAMES]
            plan += [("openvino", "none", *RESOLUTIONS[2], f) for f in FRAMES]
        stopped = {}  # series -> reason; larger frame counts are skipped
        for i, (backend, tiling, w, h, f) in enumerate(plan, 1):
            key = (backend, tiling, w, h)
            row = {"backend": backend, "tiling": tiling, "width": w, "height": h, "frames": f, "latent_frames": (f - 1) // 8 + 1}
            name = f"{backend}_{tiling.replace('+', '_')}_{w}x{h}_{f}f"
            left = budget_s - (time.time() - T0) - 120  # keep 2 min for the plot and report
            if key in stopped:
                row.update(status="skipped", detail=stopped[key])
            elif left < 60:
                row.update(status="skipped", detail="time budget used up")
            else:
                log(f"[{i}/{len(plan)}] {name} ...")
                common = ["--stats", stats, "--frames", f, "--height", h, "--width", w, "--threads", threads]
                if backend == "openvino":
                    args = ["ov", "--ir", ir, *common]
                else:
                    if vae_dir is None:
                        raise RuntimeError("PyTorch baseline needs --vae-dir when --ir is used")
                    args = ["pt", "--vae-dir", vae_dir, "--tiling", tiling, *common, *(["--tile-min", 128] if smoke else [])]
                row.update(child(name, args, min(run_timeout_s, left)))
                if row["status"] in ("oom", "timeout"):
                    stopped[key] = f"{row['status']} at {f} frames"
            row["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            rows.append(row)
            log(f"[{i}/{len(plan)}] {name}: {row['status']} {row.get('detail', '')} | peak RSS {row.get('peak_rss_gb')} GB, "
                f"load {row.get('compile_or_load_s')} s, decode {row.get('decode_s')} s, wall {row.get('wall_s')} s")
            save_results(out, rows, meta)
    finally:
        save_results(out, rows, meta)
        if rows:
            plot(out, rows, meta)
        write_report(out, rows, meta)
        log(f"outputs in {out}: " + ", ".join(sorted(p.name for p in out.iterdir() if p.is_file())))
    return rows


if __name__ == "__main__":
    main()
