"""Step 3 of LongVid-OV: tiled (memory-bounded) decode of the LTX-Video VAE on OpenVINO: memory and quality.

Kaggle "script" kernel (CPU, internet on). Single file: the three modules of src/longvid_ov (tiled_decode.py =
the algorithm, metrics.py, bench_worker.py) are embedded at the bottom and written to disk at start, because
every measurement runs in a FRESH subprocess. In the repo the files in src/longvid_ov are used directly
(scripts/s3_embed.py refreshes the embedded copies, tests/test_tiled_decode.py checks they are in sync).

What the full run does (Kaggle, 5 h budget)
  1. pip installs pinned deps, records versions / lscpu / RAM                              -> s3_env.txt
  2. downloads ONLY vae/ of Lightricks/LTX-Video, converts the decoder to OpenVINO IR (FP32, NATIVE Conv3d)
  3. real latents: encodes the diffusers "hiker" sample clip (49 frames, extended by ping-pong) at each resolution
  4. grid 512x320 and 768x512 x 49/97/161/257 frames x modes none / spatial / temporal / spatial+temporal:
     one decode per fresh subprocess, peak RSS sampled every 50 ms, output kept on disk for the metrics
  5. quality of every tiled decode vs the untiled one where the untiled one fits: PSNR, SSIM, seams, drift
  6. port check: our tiled decode on OpenVINO vs diffusers' own tiled decode in PyTorch, same settings
  7. writes s3_results.json / .csv (after every run), s3_env.txt, s3_report.md to /kaggle/working

Local runs (never the full grid):
  --smoke   tiny random-weight VAE of the same architecture, 256x160, under a minute
  --local   the step 3 comparison on the Mac (4 threads, every process watched and killed above 5 GB):
            real VAE through PyTorch, real VAE through a native-Conv3d IR at a tiny shape, tiny VAE through
            OpenVINO for the memory curve. Why three parts: see "What broke" in results/s3_report.md.
  OMP_NUM_THREADS=4 .venv/bin/python kaggle/s3_tiling_bench/s3_tiling_bench.py --smoke
"""
import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = "Lightricks/LTX-Video"
CLIP_REPO, CLIP_FILE = "huggingface/documentation-images", "diffusers/hiker.mp4"
PINS = ["openvino==2026.4.1", "openvino-genai==2026.4.1.0"]
MODES = ["none", "spatial", "temporal", "spatial+temporal"]
TILE_T = (16, 8)  # diffusers defaults: chunk of 16 frames (+1 causal), stride 8
SAMPLE_S = 0.05
WATCHDOG_HEADROOM_GB = 1.5
MIN_AVAIL_GB = 1.0
PORT_TOLERANCE = 1e-3  # max abs diff (raw output, range ~[-1,1]) for "our tiling == diffusers tiling"
GB = 2**30
T0 = time.time()


def groups_for(profile):
    """What gets measured. One group = one VAE + one runtime + one resolution; `port` = (frames, modes) for the
    check against diffusers' own tiling. Spatial tiles: diffusers default 512/448 where the video is big enough
    to be tiled by it, otherwise a smaller tile with the same kind of overlap."""
    G = lambda **kw: {"modes": MODES, "port": None, **kw}
    if profile == "full":
        frames = [49, 97, 161, 257]
        return [
            G(key="ov_512x320", title="real VAE, OpenVINO (native Conv3d IR), 512x320", vae="real", backend="ov",
              ir="native", width=512, height=320, frames=frames, tile_hw=(256, 192), port=(49, ["spatial+temporal"])),
            G(key="ov_768x512", title="real VAE, OpenVINO (native Conv3d IR), 768x512", vae="real", backend="ov",
              ir="native", width=768, height=512, frames=frames, tile_hw=(512, 448)),
        ]
    tiny = G(key="tiny_ov", title="TINY random-weight VAE, OpenVINO (Conv3d-as-Conv2d IR), 256x160", vae="tiny",
             backend="ov", ir="as2d", width=256, height=160, frames=[17, 33, 49], tile_hw=(128, 96),
             port=(49, MODES[1:]))
    if profile == "smoke":
        return [{**tiny, "frames": [17, 33], "port": (33, ["spatial+temporal"])}]
    return [  # local
        G(key="real_pt", title="real VAE, PyTorch runtime, 256x160", vae="real", backend="pt", ir=None,
          width=256, height=160, frames=[17, 33, 49], tile_hw=(128, 96), port=(49, MODES[1:])),
        G(key="real_ov", title="real VAE, OpenVINO (native Conv3d IR), 128x96", vae="real", backend="ov", ir="native",
          width=128, height=96, frames=[17], tile_hw=(96, 64), modes=["none"], port=(17, ["spatial+temporal"])),
        tiny,
    ]


def log(msg):
    el = int(time.time() - T0)
    print(f"[s3 {time.strftime('%H:%M:%S')} +{el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d}] {msg}", flush=True)


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
    """Run one worker job in a fresh process, sampling its RSS every 50 ms (step 2 code). Returns the result fields."""
    Path(out_json).unlink(missing_ok=True)
    env = {**os.environ, "USE_TF": "0", "USE_FLAX": "0"}  # keep transformers from importing TensorFlow / JAX
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    peak = rss = 0
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
    return {"status": status, "detail": detail, "peak_rss_gb": round(peak / GB, 3), "wall_s": round(wall, 2), **res}


COLUMNS = ["group", "backend", "mode", "rule", "role", "width", "height", "frames", "status", "peak_rss_gb",
           "rss_after_load_gb", "rss_after_decode_gb", "decode_s", "load_s", "wall_s", "decoder_calls", "threads",
           "latent_shape", "output_shape", "detail", "finished_at"]


def save_results(out, rows, comps, meta):
    with open(out / "s3_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "s3_results.json").write_text(json.dumps({**meta, "runs": rows, "comparisons": comps}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny local run of the same code path")
    ap.add_argument("--local", action="store_true", help="the step 3 local comparison (small shapes, 5 GB cap)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--report-only", metavar="DIR", help="rewrite s3_report.md from DIR/s3_results.json, run nothing")
    a, _ = ap.parse_known_args()  # Kaggle runs scripts through a kernel launcher with its own argv
    if a.report_only:
        saved = json.loads((Path(a.report_only) / "s3_results.json").read_text())
        return write_report(Path(a.report_only), saved.pop("runs"), saved.pop("comparisons"), saved)
    profile = "smoke" if a.smoke or os.environ.get("S3_SMOKE") == "1" else "local" if a.local else "full"
    on_kaggle = Path("/kaggle/working").is_dir()
    if profile == "full" and not on_kaggle:
        sys.exit("The full benchmark only runs on Kaggle. Use --smoke (or --local) on your own machine.")

    here = Path(globals().get("__file__", "s3_tiling_bench.py")).resolve().parent
    root = here.parent.parent
    if profile == "full":
        out = Path(a.out) if a.out else Path("/kaggle/working")
        work = Path("/kaggle/temp/s3_work") if Path("/kaggle/temp").is_dir() else Path("/tmp/s3_work")
        logs, threads = out / "logs", 0
        budget_s, run_timeout_s, setup_timeout_s = 5 * 3600, 40 * 60, 30 * 60
    else:
        out = Path(a.out) if a.out else root / "results" / ("s3_smoke" if profile == "smoke" else "")
        work = root / "models" / f"s3_{profile}"
        logs, threads = out / ("logs" if profile == "smoke" else "s3_logs"), 4
        budget_s, run_timeout_s, setup_timeout_s = 20 * 60, 300, 300
    for d in (out, logs, work):
        d.mkdir(parents=True, exist_ok=True)

    # 1. environment ------------------------------------------------------------------------------------
    if profile == "full":
        if not pip(*PINS):
            log("pinned OpenVINO not installable, falling back to latest stable")
            pip("openvino", "openvino-genai")
        pip("diffusers==0.39.0", "huggingface_hub", "safetensors", "psutil", "scikit-image", "imageio", "imageio-ffmpeg")
        if importlib.util.find_spec("torch") is None:  # Kaggle ships torch; only fetch the CPU wheel if it is missing
            pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    import psutil

    limit = mem_limit_bytes(psutil)
    kill_rss = limit - WATCHDOG_HEADROOM_GB * GB if profile == "full" else 5.0 * GB
    lscpu = sh(["lscpu"])
    cpu_model = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")),
                     platform.processor() or platform.machine())
    groups = groups_for(profile)
    meta = {
        "profile": profile, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu_model": cpu_model,
        "nproc": os.cpu_count(), "threads": threads, "mem_limit_gb": round(limit / GB, 2),
        "kill_rss_gb": round(kill_rss / GB, 2), "run_timeout_s": run_timeout_s, "budget_s": budget_s,
        "tile_t": TILE_T, "port_tolerance": PORT_TOLERANCE, "groups": groups, "setup": {},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in (
            "openvino", "torch", "diffusers", "numpy", "scikit-image", "psutil")}},
    }
    env_txt = "\n".join([
        f"date: {meta['started']}", f"platform: {platform.platform()}", f"python: {sys.version}",
        *(f"{k}: {v}" for k, v in meta["versions"].items() if k != "python"),
        f"cpus: {os.cpu_count()}; inference threads: {threads or 'all'}",
        f"RAM limit (min of RAM and cgroup): {meta['mem_limit_gb']} GB; available now: "
        f"{psutil.virtual_memory().available / GB:.2f} GB; watchdog kill RSS: {meta['kill_rss_gb']} GB",
        "", "--- lscpu ---", lscpu, ""])
    (out / "s3_env.txt").write_text(env_txt)
    print(env_txt, flush=True)

    # The worker and the two modules it imports: straight from the repo, or the embedded copies on Kaggle.
    pkg = root / "src" / "longvid_ov"
    if not (pkg / "bench_worker.py").exists():
        pkg = work / "longvid_ov"
        pkg.mkdir(exist_ok=True)
        for name, src in EMBEDDED.items():
            (pkg / name).write_text(src)
    worker = pkg / "bench_worker.py"
    rows, comps = [], []

    def child(name, args, timeout_s):
        return run_child(psutil, worker, [*args, "--threads", threads], work / f"{name}.json", logs / f"{name}.log",
                         timeout_s, kill_rss, threads)

    def setup(name, args, reuse=None):
        """A preparation job (tiny VAE, IR conversion, encoding). Raises if it fails; result kept in meta."""
        if reuse and Path(reuse).exists():
            meta["setup"][name] = {"status": "reused", "path": str(reuse)}
        else:
            log(f"setup {name} ...")
            meta["setup"][name] = r = child(name, args, setup_timeout_s)
            log(f"setup {name}: {r['status']} {r['detail']} | peak RSS {r['peak_rss_gb']} GB, {r['wall_s']} s")
        return meta["setup"][name]

    try:
        # 2. clip + VAEs ----------------------------------------------------------------------------------
        clip = root / "data" / "clip" / CLIP_FILE
        if not clip.exists():
            from huggingface_hub import hf_hub_download

            clip = Path(hf_hub_download(CLIP_REPO, CLIP_FILE, repo_type="dataset", local_dir=work / "clip"))
        vae_dirs = {}
        if any(g["vae"] == "real" for g in groups):
            vae_dirs["real"] = root / "models" / "ltx_vae_pt" / "vae"
            if not vae_dirs["real"].exists():
                from huggingface_hub import snapshot_download

                log(f"downloading {REPO} vae/ ...")
                vae_dirs["real"] = Path(snapshot_download(REPO, allow_patterns=["vae/*"], local_dir=work / "ltx_vae_pt")) / "vae"
        if any(g["vae"] == "tiny" for g in groups):
            vae_dirs["tiny"] = work / "tiny_vae"
            if setup("tiny_vae", ["tiny", "--vae-dir", vae_dirs["tiny"]], vae_dirs["tiny"] / "config.json")["status"] not in ("ok", "reused"):
                raise RuntimeError("tiny VAE build failed (see logs/tiny_vae.log)")

        # 3. the groups -----------------------------------------------------------------------------------
        stopped = {}  # series -> reason; longer videos of the same series are skipped
        for g in groups:
            gk, w, h = g["key"], g["width"], g["height"]
            try:
                # real latents for the longest video of the group; shorter ones use a prefix (the encoder is causal)
                latents = work / f"latents_{g['vae']}_{w}x{h}_{max(g['frames'])}f.npy"
                enc_args = ["encode", "--vae-dir", vae_dirs[g["vae"]], "--clip", clip, "--frames", max(g["frames"]),
                            "--height", h, "--width", w, "--save", latents]
                enc = setup(f"encode_{gk}", enc_args, latents)
                if enc["status"] == "oom":  # one-shot encode does not fit: diffusers' tiled + framewise encode
                    enc = setup(f"encode_{gk}_tiled", [*enc_args, "--tiled-encode"])
                if enc["status"] not in ("ok", "reused"):
                    raise RuntimeError(f"encode failed: {enc['detail']}")
                ir = None
                if g["ir"]:
                    ir = work / f"{g['vae']}_{g['ir']}" / "vae_decoder.xml"
                    ir.parent.mkdir(exist_ok=True)
                    cv = setup(f"convert_{g['vae']}_{g['ir']}", ["convert", "--vae-dir", vae_dirs[g["vae"]], "--ir", ir,
                                                               *(["--conv3d-as-2d"] if g["ir"] == "as2d" else [])], ir)
                    if cv["status"] not in ("ok", "reused"):
                        raise RuntimeError(f"IR conversion failed: {cv['detail']}")
            except RuntimeError as e:
                log(f"group {gk} skipped: {e}")
                meta["setup"][f"group_{gk}"] = {"status": "skipped", "detail": str(e)}
                continue

            for f in g["frames"]:
                # (backend, mode, first-frame rule, role)
                plan = [(g["backend"], m, "drop_first", "grid") for m in g["modes"]]
                port = g["port"][1] if g["port"] and g["port"][0] == f else []
                for m in port:  # the port check needs our decoder with diffusers' first-frame rule + diffusers itself
                    ours = (g["backend"], m, "diffusers" if "temporal" in m else "drop_first")
                    if not any(p[:3] == ours for p in plan):
                        plan.append((*ours, "port-test"))
                    plan.append(("diffusers", m, "diffusers", "port-ref"))
                done = {}  # (backend, mode, rule) -> (row, output file)
                for backend, mode, rule, role in plan:
                    name = f"{gk}_{backend}_{mode.replace('+', '_')}_{rule}_{f}f"
                    row = {"group": gk, "backend": backend, "mode": mode, "rule": rule, "role": role, "width": w,
                           "height": h, "frames": f, "name": name}
                    series = (gk, backend, mode, rule)
                    left = budget_s - (time.time() - T0) - 180  # keep 3 min for the report
                    if series in stopped:
                        row.update(status="skipped", detail=stopped[series])
                    elif left < 60:
                        row.update(status="skipped", detail="time budget used up")
                    else:
                        log(f"{name} ...")
                        save_to = work / f"{name}.npy"
                        row.update(child(name, [
                            "decode", "--backend", backend, "--mode", mode, "--rule", rule, "--latents", latents,
                            "--frames", f, "--tile-hw", *g["tile_hw"], "--tile-t", *TILE_T, "--save", save_to,
                            *(["--ir", ir] if backend == "ov" else ["--vae-dir", vae_dirs[g["vae"]]])],
                            min(run_timeout_s, left)))
                        if row["status"] in ("oom", "timeout"):
                            stopped[series] = f"{row['status']} at {f} frames"
                        done[(backend, mode, rule)] = (row, save_to)
                    row["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    rows.append(row)
                    log(f"{name}: {row['status']} {row.get('detail', '')} | peak RSS {row.get('peak_rss_gb')} GB, "
                        f"decode {row.get('decode_s')} s, {row.get('decoder_calls')} decoder calls")
                    save_results(out, rows, comps, meta)

                # quality: every tiled decode of ours vs our untiled decode; port: ours vs diffusers
                def compare(kind, test, ref):
                    (trow, tfile), (rrow, rfile) = done[test], done[ref]
                    if trow["status"] != "ok" or rrow["status"] != "ok":
                        return
                    name = f"metrics_{kind}_{trow['name']}"
                    r = child(name, ["metrics", "--ref", rfile, "--test", tfile,
                                     *(["--layout", json.dumps(trow["layout"])] if kind == "vs_untiled" else [])], run_timeout_s)
                    comps.append({"kind": kind, "group": gk, "frames": f, "mode": trow["mode"], "rule": trow["rule"],
                                  "test": trow["name"], "ref": rrow["name"], **r})
                    log(f"{name}: {r['status']} {r['detail']} | PSNR {r.get('psnr_db')} dB, max abs diff {r.get('max_abs_diff_raw')}")

                untiled = (g["backend"], "none", "drop_first")
                for key in done:
                    if key[0] == g["backend"] and key[1] != "none" and untiled in done:
                        compare("vs_untiled", key, untiled)
                for m in port:
                    compare("port", (g["backend"], m, "diffusers" if "temporal" in m else "drop_first"),
                            ("diffusers", m, "diffusers"))
                for _, save_to in done.values():  # the decoded videos are only needed for the metrics
                    save_to.unlink(missing_ok=True)
                save_results(out, rows, comps, meta)
    finally:
        meta["runtime_min"] = round((time.time() - T0) / 60, 1)
        save_results(out, rows, comps, meta)
        write_report(out, rows, comps, meta)
        log(f"outputs in {out}: " + ", ".join(sorted(p.name for p in out.iterdir() if p.is_file() and p.name.startswith("s3_"))))


# ---- report (generated from the measured values only) -----------------------------------------------------
RULE_LABEL = {"drop_first": "drop first (ours)", "diffusers": "drop last (diffusers 0.39)"}


def db(x):
    return "-" if x is None else "identical" if x == float("inf") else f"{x:.2f}"


def write_report(out, rows, comps, meta):
    groups = {g["key"]: g for g in meta["groups"]}
    title = {"smoke": " (SMOKE TEST: tiny random VAE, numbers mean nothing)", "local": " (local run, small shapes)", "full": ""}[meta["profile"]]
    ok = lambda c: c["status"] == "ok"
    L = [f"# Step 3 report: tiled decode of the LTX-Video VAE on OpenVINO{title}", "",
         f"Generated by `s3_tiling_bench.py --{meta['profile']}` from the run started {meta['started']}; runtime "
         f"{meta.get('runtime_min')} min. Every number below was measured in this run (raw: `s3_results.json`, `s3_results.csv`)"
         + (", except the block quoted from `s3_arm_memory_probe.txt`." if meta["profile"] == "local" else "."), "",
         "## Setup",
         f"- Machine: {meta['cpu_model']}, {meta['nproc']} logical CPUs, {meta['threads'] or 'all'} inference threads, RAM limit "
         f"{meta['mem_limit_gb']} GB. Each run is a fresh subprocess; the parent samples its RSS every {SAMPLE_S * 1000:.0f} ms "
         f"and kills it above {meta['kill_rss_gb']} GB (`OOM`) or after {meta['run_timeout_s'] / 60:.0f} min (`timeout`).",
         "- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()) + ".",
         f"- Latents: real, from encoding the diffusers `hiker.mp4` sample clip (center-cropped, Lanczos-resized) with the "
         f"PyTorch encoder (posterior mean). Shorter videos decode a prefix of the same latents (the VAE config has "
         f"`encoder_causal=True`, so a prefix should be the latents of the shorter clip; not re-checked here).",
         "- Groups:"]
    for g in meta["groups"]:
        enc = meta["setup"].get(f"encode_{g['key']}_tiled") or meta["setup"].get(f"encode_{g['key']}", {})
        note = meta["setup"].get(f"group_{g['key']}")
        extra = ""
        if "clip_frames_in_file" in enc:
            extra = (f"; the clip file has {enc['clip_frames_in_file']} frames"
                     + (", extended to the longest video by ping-pong (forward, backward, ...)" if enc["clip_extended_by_ping_pong"] else ", no looping needed")
                     + ("; encoded with diffusers tiled + framewise encoding (one-shot encode did not fit)" if enc.get("tiled_encode") else ""))
        L.append(f"  - `{g['key']}`: {g['title']}; frames {g['frames']}; spatial tile {g['tile_hw'][0]} px, stride "
                 f"{g['tile_hw'][1]} px (overlap {g['tile_hw'][0] - g['tile_hw'][1]} px); temporal chunk {meta['tile_t'][0]} frames, "
                 f"stride {meta['tile_t'][1]} (overlap {meta['tile_t'][0] - meta['tile_t'][1]} frames){extra}"
                 + (f". **SKIPPED: {note['detail']}**" if note else "."))
    for k, s in meta["setup"].items():
        if k.startswith("convert_") and s["status"] == "ok":
            L.append(f"- IR conversion `{k}`: {s.get('convert_s')} s, peak RSS {s['peak_rss_gb']:.2f} GB, weights "
                     f"{s.get('ir_bytes', 0) / GB:.2f} GB, {s.get('conv3d_rewritten_as_2d')} Conv3d rewritten as Conv2d.")
    L += ["",
          "## What each mode does (`src/longvid_ov/tiled_decode.py`)",
          "The decoder IR is called once per tile; tiles overlap and the overlap is cross-faded with a linear ramp "
          "(`previous * (1 - k/n) + new * k/n`, weights sum to 1). Tile sizes, strides, overlap, blend formula and blend "
          "order are those of diffusers 0.39 `AutoencoderKLLTXVideo` (`tiled_decode`, `_temporal_tiled_decode`, `blend_v/h/t`).",
          "",
          "| mode | what is decoded per call | memory bounded in | diffusers equivalent |",
          "|---|---|---|---|",
          "| `none` | the whole latent tensor | nothing | default |",
          "| `spatial` | one H/W tile of all frames | height, width | `enable_tiling()` |",
          "| `temporal` | one chunk of latent frames (chunk/8 + 1), full H/W | frames | `use_framewise_decoding = True` |",
          "| `spatial+temporal` | one H/W tile of one chunk | frames, height, width | both |",
          "",
          "Like diffusers, a mode falls back to a plain decode when the input is not larger than one tile / one chunk. "
          "Unlike diffusers, tiles are blended and written to the output as they are decoded (only the previous chunk / "
          "previous row of tiles is kept), and a trailing chunk that would contribute no frames is not decoded.",
          "",
          "**First-frame rule.** The decoder turns the first latent frame of whatever it is given into ONE video frame "
          "and every other latent frame into 8. A chunk starting at latent frame i > 0 therefore begins with an extra "
          "frame (video frame 8i) that the previous chunk already produced. `drop_first` (ours) drops that frame, so the "
          "chunk starts at frame 8i+1, where it is pasted. diffusers 0.39 drops the chunk's LAST frame instead, so the "
          "chunk starts at frame 8i but is still pasted at 8i+1 (`first_frame_rule=\"diffusers\"` reproduces that for "
          "the port check).",
          ""]

    # port check
    port = [c for c in comps if c["kind"] == "port"]
    L += ["## Port check: our tiled decode vs diffusers' tiled decode (same latents, same settings)",
          "Ours runs with `first_frame_rule=\"diffusers\"` here, on the runtime of the group; diffusers runs its own tiling "
          f"in PyTorch. `match` = max abs diff of the raw output (range ~[-1,1]) below {meta['port_tolerance']}.", "",
          "| group | mode | frames | max abs diff (raw) | PSNR dB | SSIM | match |", "|---|---|---|---|---|---|---|"]
    for c in port:
        L.append(f"| `{c['group']}` | {c['mode']} | {c['frames']} | " + (
            f"{c['max_abs_diff_raw']:.3e} | {db(c['psnr_db'])} | {c['ssim']:.6f} | {'yes' if c['max_abs_diff_raw'] < meta['port_tolerance'] else '**NO**'} |"
            if ok(c) else f"{c['status']}: {c['detail']} | | | |"))
    if not port:
        L.append("| (no port check completed) | | | | | | |")

    # quality vs untiled
    q = [c for c in comps if c["kind"] == "vs_untiled"]
    L += ["", "## Quality: tiled vs untiled decode of the same latents (same runtime)",
          "PSNR / SSIM per frame on displayed values ([0,1], clipped); `PSNR` is from the MSE pooled over all frames.", "",
          "| group | frames | mode | first-frame rule | PSNR dB | worst frame PSNR | SSIM | worst frame SSIM | max abs diff (raw) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in q:
        rule = RULE_LABEL[c["rule"]] if "temporal" in c["mode"] else "n/a"
        L.append(f"| `{c['group']}` | {c['frames']} | {c['mode']} | {rule} | " + (
            f"{db(c['psnr_db'])} | {db(c['psnr_min_db'])} | {c['ssim']:.4f} | {c['ssim_min']:.4f} | {c['max_abs_diff_raw']:.3f} |"
            if ok(c) else f"{c['status']}: {c['detail']} | | | | |"))
    small = sorted({g["tile_hw"][0] for g in meta["groups"] if g["tile_hw"][0] < 512})
    if small:
        L += ["", f"Spatial tiles of {' / '.join(map(str, small))} px are used where the video is too small for diffusers' default "
              "512 px tile (448 px stride, 64 px overlap). A small tile cuts away much more of the context the decoder would "
              "normally see around each pixel, so the `spatial` rows of those groups are a hard case for spatial tiling, not "
              "a measurement of the default tile."]
    missing = [r for r in rows if r["role"] == "grid" and r["mode"] != "none" and r["status"] == "ok"
               and not any(c["test"] == r["name"] for c in q)]
    if missing:
        L += ["", "No untiled reference (it did not fit / did not finish), so no quality numbers for: "
              + ", ".join(f"`{r['name']}`" for r in missing) + "."]

    # first-frame rule evidence
    shifted = [c for c in q if ok(c) and "temporal" in c["mode"] and "psnr_frames_2plus_db" in c]
    if shifted:
        L += ["", "## First-frame rule: is the tiled video one frame late?",
              "Tiled frame i compared with untiled frame i, and with untiled frame i-1 (frames 2..end in both columns). "
              "If the second number is the higher one, the tiled video shows the previous frame.", "",
              "| group | frames | mode | first-frame rule | PSNR vs same frame | PSNR vs previous frame |", "|---|---|---|---|---|---|"]
        for c in shifted:
            L.append(f"| `{c['group']}` | {c['frames']} | {c['mode']} | {RULE_LABEL[c['rule']]} | {db(c['psnr_frames_2plus_db'])} | "
                     f"{db(c['psnr_frames_2plus_vs_previous_ref_frame_db'])} |")

    # seams
    L += ["", "## Seams",
          "Reference-free scores, computed at the tile boundaries for the tiled video and at the SAME positions for the "
          "untiled video (so content that happens to sit there cancels out). 1.0 = a boundary looks like any other place.",
          "- temporal: mean frame-to-frame change at the chunk boundaries / the clip's median frame-to-frame change",
          "- spatial: mean gradient across the tile borders (both edges of each blend band) / median gradient over all rows, "
          "same for columns, averaged", "",
          "| group | frames | mode | first-frame rule | temporal seam tiled | untiled | spatial seam tiled | untiled |", "|---|---|---|---|---|---|---|---|"]
    sc = lambda s: "-" if not s else f"{s['score']:.3f}"
    for c in q:
        if ok(c):
            t, s = c["temporal_seam"], c["spatial_seam"]
            rule = RULE_LABEL[c["rule"]] if "temporal" in c["mode"] else "n/a"
            L.append(f"| `{c['group']}` | {c['frames']} | {c['mode']} | {rule} | {sc(t['tiled'])} | {sc(t['untiled'])} | "
                     f"{sc(s['tiled'])} | {sc(s['untiled'])} |")

    # drift
    L += ["", "## Drift: does quality get worse later in the clip?",
          "PSNR vs untiled over the first third of the frames and over the last third.", "",
          "| group | frames | mode | first-frame rule | first third dB | last third dB | drop dB |", "|---|---|---|---|---|---|---|"]
    for c in q:
        if ok(c):
            d = c["drift"]
            rule = RULE_LABEL[c["rule"]] if "temporal" in c["mode"] else "n/a"
            L.append(f"| `{c['group']}` | {c['frames']} | {c['mode']} | {rule} | {db(d['first_third_psnr_db'])} | "
                     f"{db(d['last_third_psnr_db'])} | {d['drop_db']:+.2f} |")

    by_len = {}
    for c in q:
        if ok(c) and c["rule"] == "drop_first":
            by_len.setdefault((c["group"], c["mode"]), []).append(c)
    L += ["", "Last-third PSNR by clip length (if tiling drifted, it would fall as the clip gets longer):"]
    L += [f"- `{gk}`, {m}: " + ", ".join(f"{c['frames']} frames {db(c['drift']['last_third_psnr_db'])} dB" for c in cs)
          for (gk, m), cs in by_len.items() if len(cs) > 1]
    L += ["", "PSNR vs untiled per block of 8 frames (frame 0, then 1-8, 9-16, ...; pooled MSE), longest clip of each group:"]
    for (gk, m), cs in by_len.items():
        mse = [10 ** (-p / 10) for p in cs[-1]["psnr_per_frame"]]
        blocks = [mse[:1]] + [mse[i : i + 8] for i in range(1, len(mse), 8)]
        L.append(f"- `{gk}`, {m}, {cs[-1]['frames']} frames: " + " ".join(
            "identical" if sum(b) == 0 else f"{-10 * math.log10(sum(b) / len(b)):.1f}" for b in blocks))

    # memory
    L += ["", "## Memory: peak RSS per mode (fresh subprocess per cell)",
          "Cell = peak RSS / decode time / decoder calls. Peak RSS includes the runtime, the weights and the full decoded "
          "video (float32), which every mode has to hold.", ""]
    cell = lambda r: ("-" if r is None else f"{r['peak_rss_gb']:.2f} GB / {r['decode_s']:.1f} s / {r['decoder_calls']}" if r["status"] == "ok"
                      else "skipped" if r["status"] == "skipped" else f"**{r['status'].upper()}** at {r['peak_rss_gb']:.2f} GB")
    for gk, g in groups.items():
        # our decoder's runs of this group; where a mode ran with both first-frame rules, the grid one (ours) is shown
        mine = sorted((r for r in rows if r["group"] == gk and r["backend"] == g["backend"]), key=lambda r: r["role"] != "grid")
        modes = [m for m in MODES if any(r["mode"] == m for r in mine)]
        if not mine:
            continue
        L += [f"**`{gk}`: {g['title']}**", "", "| frames | latent frames | " + " | ".join(modes) + " |",
              "|---|---|" + "---|" * len(modes)]
        for f in g["frames"]:
            L.append(f"| {f} | {(f - 1) // 8 + 1} | " + " | ".join(
                cell(next((r for r in mine if r["frames"] == f and r["mode"] == m), None)) for m in modes) + " |")
        for m in modes:  # growth from the shortest to the longest video that finished
            done = [r for r in mine if r["mode"] == m and r["role"] == "grid" and r["status"] == "ok"]
            if len(done) > 1:
                a, b = done[0], done[-1]
                L.append(f"- `{m}`: {a['frames']} -> {b['frames']} frames: peak RSS {a['peak_rss_gb']:.2f} -> {b['peak_rss_gb']:.2f} GB "
                         f"({b['peak_rss_gb'] - a['peak_rss_gb']:+.2f} GB, {b['peak_rss_gb'] / a['peak_rss_gb']:.2f}x).")
        bad = [r for r in mine if r["status"] in ("oom", "timeout", "error")]
        L += [f"- `{r['mode']}`, {r['frames']} frames: {r['status'].upper()}: {r['detail']}" for r in bad] + [""]

    L += ["## What broke and how it was fixed", ""] + notes(out, meta, comps)
    (out / "s3_report.md").write_text("\n".join(L) + "\n")


def notes(out, meta, comps):
    """Hand-written observations from developing this step. Numbers are only quoted from this run's results or
    from the saved probe output."""
    N = ["1. **diffusers' temporal tiling shows every frame after the first chunk one frame late.** "
         "`_temporal_tiled_decode` trims each chunk with `decoded[:, :, :-1]`, but the surplus frame is the chunk's FIRST "
         "one (the causal frame); its encode twin `_temporal_tiled_encode` does drop the first (`tile[:, :, 1:]`). "
         "Found by decoding with a toy decoder that just repeats each latent frame (tiled must then equal untiled "
         "exactly; it did not), confirmed on the real VAE in the first-frame-rule table above. Fix: `first_frame_rule="
         "\"drop_first\"` (the default); the diffusers rule is kept only so the port can be proven identical to diffusers."]
    pairs = {}
    for c in comps:
        if c["kind"] == "vs_untiled" and c["status"] == "ok" and "temporal" in c["mode"]:
            pairs.setdefault((c["group"], c["frames"], c["mode"]), {})[c["rule"]] = c
    for (gk, f, m), p in pairs.items():
        if len(p) == 2:
            N.append(f"   - `{gk}`, {f} frames, {m}: PSNR vs untiled {db(p['diffusers']['psnr_db'])} dB with the diffusers rule, "
                     f"{db(p['drop_first']['psnr_db'])} dB with ours.")
    N += ["2. **diffusers decodes all tiles first and blends afterwards**, so it holds every decoded tile at once. "
          "The port blends each tile as soon as it is decoded and keeps only what the blend reads (previous chunk, "
          "previous row of tiles); the port-check table shows the output is unchanged.",
          "3. **diffusers runs the decoder on a last chunk of one latent frame and discards the result** (it decodes to "
          "the single causal frame, which is then trimmed). The port skips that call.",
          "4. **The infer request reuses its output buffer.** The blend writes into the decoded tile in place, and the "
          "previous tile is read again when the next one is blended, so `OVDecoder` returns a copy of the output tensor."]
    if meta["profile"] == "local":
        probe = out / "s3_arm_memory_probe.txt"
        N += ["5. **The real decoder cannot run in OpenVINO on this Mac under the 5 GB per-process limit**, with the "
              "step 1 Conv3d-as-Conv2d IR (the only fast one on ARM). The first inference passes 5 GB within seconds even "
              "for 9 frames at 256x160, at f32 and f16, and a second inference of the same shape grows further. The "
              "largest activation at that shape is about 12 MB (128 channels x 9 frames x 40 x 64, float32), so the video itself "
              "is not what fills the memory; the likely cause is what the ARM convolution kernels allocate per layer "
              "(copies of the weights, work buffers), which was not investigated further. Probe output (`s3_arm_memory_probe.txt`, "
              "`scripts/s3_arm_memory_probe.py`):", "",
              "```", probe.read_text().rstrip() if probe.exists() else "(probe output file missing)", "```", "",
              "   How the local comparison was split to stay inside the limit (the Kaggle job runs the real thing on x86 "
              "with native Conv3d):",
              "   - `real_pt`: real VAE and real latents, our tiled decoder driving the PyTorch decoder. Quality, seams, drift "
              "and memory-vs-frames of the ALGORITHM on the real model. (Step 1 measured OpenVINO vs PyTorch decode at "
              "2.5e-05 max abs diff, so the per-tile results are the same to that precision.)",
              "   - `real_ov`: real VAE on OpenVINO through a native-Conv3d IR (reference kernel on ARM: low memory but "
              "slow), so only one tiny shape: the OpenVINO-vs-diffusers port check on real weights.",
              "   - `tiny_ov`: random-weight VAE of the same architecture (1/8 of the channels) on OpenVINO: all modes, "
              "port check, and the memory curve on the OpenVINO runtime. Its picture quality numbers mean nothing "
              "(random weights); they are listed only for completeness."]
    return N


# --- BEGIN EMBEDDED src/longvid_ov (generated by scripts/s3_embed.py; do not edit by hand) ---
EMBEDDED = {
    "tiled_decode.py": r'''"""Memory-bounded decode for the LTX-Video VAE decoder: run the decoder on small pieces, cross-fade the overlaps.

The decoder itself is a black box here: any callable `latents [1,C,f,h,w] -> video [1,3,F,H,W]` (numpy, float32).
`OVDecoder` wraps the OpenVINO IR; the tests also plug in the PyTorch decoder and a toy decoder.

Shapes (LTX-Video VAE): 1 latent pixel = 32x32 video pixels; in time the decoder is "causal":
    latent frame 0  -> video frame 0          (1 frame)
    latent frame k  -> video frames 8k-7..8k  (8 frames)        so  F = 8 * (f - 1) + 1.

Modes
    "none"              one decoder call on the whole latent tensor (memory grows with frames * H * W)
    "spatial"           overlapping H/W tiles, blended with a linear ramp          (bounds memory in H, W)
    "temporal"          overlapping chunks of latent frames, blended in time       (bounds memory in frames)
    "spatial+temporal"  temporal chunks, each decoded with spatial tiles           (bounds both)

This is a port of diffusers 0.39 `AutoencoderKLLTXVideo` (`tiled_decode`, `_temporal_tiled_decode`, `blend_v/h/t`):
same tile sizes, strides, overlaps, blend formula and the same "is the input big enough to tile" tests.
Two deliberate differences:
  1. Streaming. diffusers decodes every tile first and blends afterwards; here each tile is blended and written
     into the output as soon as it is decoded. Only the previous chunk (time) / previous row of tiles (space) is
     kept, because that is all the blend ever reads. The result is identical, the working memory is bounded.
  2. The causal first-frame rule, `first_frame_rule`. A chunk that starts at latent frame i > 0 still decodes its
     first latent frame to ONE video frame (the decoder cannot know it is not the start of the video). That frame
     is video frame 8i, which the previous chunk already produced, so it must be dropped:
        "drop_first" (default): drop it. The chunk then starts at video frame 8i+1, exactly where it is pasted.
        "diffusers":            diffusers 0.39 drops the LAST frame of the chunk instead, so the chunk starts at
                                video frame 8i but is pasted at 8i+1: everything after the first chunk is shown one
                                frame late. Kept only to reproduce diffusers bit-for-bit in the port check.
"""
from dataclasses import dataclass

import numpy as np

MODES = ("none", "spatial", "temporal", "spatial+temporal")


@dataclass
class TilingConfig:
    """Sizes in video pixels / video frames, same names and defaults as diffusers `AutoencoderKLLTXVideo`."""

    tile_sample_min_height: int = 512  # tile size; inputs not larger than this are not tiled
    tile_sample_min_width: int = 512
    tile_sample_stride_height: int = 448  # distance between tile origins; overlap = min - stride = 64 px
    tile_sample_stride_width: int = 448
    tile_sample_min_num_frames: int = 16  # chunk length (+1 causal frame); overlap = min - stride = 8 frames
    tile_sample_stride_num_frames: int = 8
    spatial_compression_ratio: int = 32
    temporal_compression_ratio: int = 8
    first_frame_rule: str = "drop_first"  # or "diffusers", see module docstring

    def __post_init__(self):
        s, t = self.spatial_compression_ratio, self.temporal_compression_ratio
        spatial = (self.tile_sample_min_height, self.tile_sample_min_width,
                   self.tile_sample_stride_height, self.tile_sample_stride_width)
        temporal = (self.tile_sample_min_num_frames, self.tile_sample_stride_num_frames)
        # Tiles are cut on the latent grid, so sizes must land on it (otherwise the blend would be misaligned).
        assert all(v > 0 and v % s == 0 for v in spatial), f"spatial tile sizes must be multiples of {s}"
        assert all(v > 0 and v % t == 0 for v in temporal), f"temporal tile sizes must be multiples of {t}"
        assert self.tile_sample_stride_height <= self.tile_sample_min_height
        assert self.tile_sample_stride_width <= self.tile_sample_min_width
        assert self.tile_sample_stride_num_frames <= self.tile_sample_min_num_frames
        assert self.first_frame_rule in ("drop_first", "diffusers")


def blend_weights(extent):
    """Linear cross-fade over `extent` positions: (weight of the previous tile, weight of the new tile).

    Position k gets  previous * (1 - k/extent) + new * (k/extent).  The two always sum to 1, so a constant
    image stays constant. At k = 0 the previous tile wins completely; the new tile never quite reaches 1
    inside the overlap, it takes over fully on the first position after it.
    """
    w_new = np.arange(extent, dtype=np.float64) / extent
    return (1.0 - w_new).astype(np.float32), w_new.astype(np.float32)


def blend(prev, cur, extent, axis):
    """Cross-fade the LAST `extent` slices of `prev` into the FIRST `extent` slices of `cur` along `axis`.

    Modifies and returns `cur`. Same as diffusers blend_t (axis 2), blend_v (axis 3), blend_h (axis 4),
    including the clamp of `extent` to what both tiles actually have (edge tiles are smaller).
    """
    extent = min(prev.shape[axis], cur.shape[axis], extent)
    if extent == 0:
        return cur
    w_prev, w_cur = (w.reshape([-1 if a == axis else 1 for a in range(cur.ndim)]) for w in blend_weights(extent))
    head = tuple(slice(0, extent) if a == axis else slice(None) for a in range(cur.ndim))
    tail = tuple(slice(-extent, None) if a == axis else slice(None) for a in range(cur.ndim))
    cur[head] = prev[tail] * w_prev + cur[head] * w_cur
    return cur


class TiledDecoder:
    """decode(latents, mode) on top of any `decoder(latents) -> video` callable (see module docstring)."""

    def __init__(self, decoder, config=None):
        self.decoder = decoder
        self.cfg = config or TilingConfig()
        self.calls = 0  # decoder calls made by the last decode()

    # ---- the public entry point -------------------------------------------------------------------------
    def decode(self, latents, mode="spatial+temporal"):
        """latents [1,C,f,h,w] float32 -> video [1,3,8(f-1)+1,32h,32w] float32 (range ~[-1,1])."""
        assert mode in MODES, f"mode must be one of {MODES}"
        assert latents.ndim == 5 and latents.shape[0] == 1, "expected latents [1,C,f,h,w]"
        self.calls = 0
        spatial = mode in ("spatial", "spatial+temporal")
        if mode in ("temporal", "spatial+temporal") and self._needs_temporal(latents.shape):
            return self._decode_temporal(latents, spatial)
        return self._decode_frames(latents, spatial)

    # ---- "is it big enough to tile" (same tests as diffusers `_decode`) ----------------------------------
    def _needs_spatial(self, shape):
        c = self.cfg
        return (shape[4] > c.tile_sample_min_width // c.spatial_compression_ratio
                or shape[3] > c.tile_sample_min_height // c.spatial_compression_ratio)

    def _needs_temporal(self, shape):
        c = self.cfg
        return shape[2] > c.tile_sample_min_num_frames // c.temporal_compression_ratio

    def _decode_one(self, latents):
        self.calls += 1
        return self.decoder(np.ascontiguousarray(latents, dtype=np.float32))

    def _decode_frames(self, latents, spatial):
        """Decode one temporal chunk (or the whole video): spatially tiled if asked for and needed."""
        if spatial and self._needs_spatial(latents.shape):
            return self._decode_spatial(latents)
        return self._decode_one(latents)

    # ---- spatial tiling (diffusers `tiled_decode`) -------------------------------------------------------
    def _decode_spatial(self, z):
        c, ratio = self.cfg, self.cfg.spatial_compression_ratio
        lat_h, lat_w = z.shape[3], z.shape[4]
        tile_h, tile_w = c.tile_sample_min_height // ratio, c.tile_sample_min_width // ratio  # in latent pixels
        step_h, step_w = c.tile_sample_stride_height // ratio, c.tile_sample_stride_width // ratio
        keep_h, keep_w = c.tile_sample_stride_height, c.tile_sample_stride_width  # video pixels kept per tile
        blend_h, blend_w = c.tile_sample_min_height - keep_h, c.tile_sample_min_width - keep_w  # overlap, video px

        out = None
        row_above = []  # decoded + blended tiles of the previous row
        for i in range(0, lat_h, step_h):  # tile origins, top to bottom
            row = []
            for j in range(0, lat_w, step_w):  # left to right
                tile = self._decode_one(z[:, :, :, i : i + tile_h, j : j + tile_w])  # edge tiles come out smaller
                # Fade in from the neighbours that are already final: first the tile above, then the tile on the
                # left (same order as diffusers; it matters in the corners where both overlaps meet).
                if row_above:
                    blend(row_above[len(row)], tile, blend_h, axis=3)
                if row:
                    blend(row[-1], tile, blend_w, axis=4)
                row.append(tile)
                # Each tile owns its top-left keep_h x keep_w pixels; the rest is overlap that only feeds the
                # blend of the next tile / next row.
                if out is None:
                    out = np.empty((*tile.shape[:3], lat_h * ratio, lat_w * ratio), dtype=tile.dtype)
                piece = tile[:, :, :, :keep_h, :keep_w]
                y, x = i * ratio, j * ratio
                out[:, :, :, y : y + piece.shape[3], x : x + piece.shape[4]] = piece
            row_above = row
        return out

    # ---- temporal tiling (diffusers `_temporal_tiled_decode`) --------------------------------------------
    def _decode_temporal(self, z, spatial):
        c, ratio = self.cfg, self.cfg.temporal_compression_ratio
        lat_f = z.shape[2]
        num_frames = (lat_f - 1) * ratio + 1
        chunk = c.tile_sample_min_num_frames // ratio  # in latent frames (a chunk reads chunk + 1 of them)
        step = c.tile_sample_stride_num_frames // ratio
        keep = c.tile_sample_stride_num_frames  # video frames kept per chunk
        blend_t = c.tile_sample_min_num_frames - keep  # overlap, video frames

        out = None
        prev = None  # previous decoded chunk: its tail is cross-faded into the head of the next one
        pos = 0  # next video frame to write
        for i in range(0, lat_f, step):
            piece = z[:, :, i : i + chunk + 1]  # +1: the extra latent frame that plays the "causal first frame"
            if i > 0 and piece.shape[2] == 1:
                # A lone latent frame decodes to just the causal frame, which is dropped below: nothing to add.
                # (diffusers still runs the decoder here and throws the result away.)
                break
            video = self._decode_frames(piece, spatial)
            if i == 0:
                frames = video[:, :, : keep + 1]  # the true first frame + `keep` frames
                out = np.empty((*video.shape[:2], num_frames, *video.shape[3:]), dtype=video.dtype)
            else:
                # The causal first-frame rule: make the chunk 8 * (latent frames - 1) long again.
                if c.first_frame_rule == "drop_first":
                    video = video[:, :, 1:]  # starts at video frame 8i+1 = `pos`
                else:
                    video = video[:, :, :-1]  # diffusers 0.39: starts at video frame 8i, one frame early
                blend(prev, video, blend_t, axis=2)
                frames = video[:, :, :keep]
            frames = frames[:, :, : num_frames - pos]
            out[:, :, pos : pos + frames.shape[2]] = frames
            pos += frames.shape[2]
            prev = video
        assert pos == num_frames, (pos, num_frames)
        return out

    # ---- where the seams are (for the quality metrics) ---------------------------------------------------
    def seam_layout(self, latent_shape, mode):
        """Where tiles meet in the output of decode(latents, mode), as "transition" indices: index t means the
        step from frame/row/column t to t+1.

        frames / rows / cols: both edges of every blend band: the step into the band (the first position a new
        chunk / tile contributes to) and the step out of it (where the previous one stops contributing).
        Empty lists where the mode does not tile (or the input is too small to be tiled).
        """
        c = self.cfg
        _, _, lat_f, lat_h, lat_w = latent_shape
        layout = {"frames": [], "rows": [], "cols": []}
        if mode in ("temporal", "spatial+temporal") and self._needs_temporal(latent_shape):
            keep = c.tile_sample_stride_num_frames
            last = (lat_f - 1) * c.temporal_compression_ratio  # index of the last video frame
            edges = set()
            for start in range(keep, last, keep):  # chunk i owns frames start+1 .. start+keep
                edges.update((start, start + c.tile_sample_min_num_frames - keep))
            layout["frames"] = sorted(e for e in edges if e < last)
        if mode in ("spatial", "spatial+temporal") and self._needs_spatial(latent_shape):
            for key, size, stride, tile in (
                ("rows", lat_h * c.spatial_compression_ratio, c.tile_sample_stride_height, c.tile_sample_min_height),
                ("cols", lat_w * c.spatial_compression_ratio, c.tile_sample_stride_width, c.tile_sample_min_width),
            ):
                edges = set()
                for start in range(stride, size, stride):
                    edges.update((start - 1, start + (tile - stride) - 1))
                layout[key] = sorted(e for e in edges if 0 <= e < size - 1)
        return layout


class OVDecoder:
    """The OpenVINO decoder IR as a `latents -> video` callable (CPU, float32, dynamic frames/H/W)."""

    def __init__(self, ir_path, threads=0, device="CPU"):
        import openvino as ov

        config = {"INFERENCE_PRECISION_HINT": "f32"}  # f32 everywhere (ARM defaults to f16, some x86 to bf16)
        if threads:
            config["INFERENCE_NUM_THREADS"] = threads
        self.compiled = ov.Core().compile_model(str(ir_path), device, config)
        self.request = self.compiled.create_infer_request()

    def __call__(self, latents):
        self.request.infer({0: latents})
        # copy: the request reuses its output buffer on the next call, and the blend writes into the tile
        return self.request.get_output_tensor(0).data.copy()


def decode(latents, mode, decoder, config=None):
    """One-shot helper: TiledDecoder(decoder, config).decode(latents, mode)."""
    return TiledDecoder(decoder, config).decode(latents, mode)
''',
    "metrics.py": r'''"""Quality metrics for tiled VAE decoding. The functions take videos as [F,H,W,3] views of the raw decoder
output (`as_frames`), convert one frame at a time to displayed [0,1] values, so memory-mapped arrays are fine.

PSNR / SSIM   tiled vs untiled decode of the same latents, per frame.
SEAM          does the tiling leave a visible line? Reference-free, so it is computed for the tiled AND the
              untiled video at the same positions; the tiled/untiled ratio separates artefact from content.
                temporal: frame-to-frame change at the chunk boundaries / the clip's median frame-to-frame change
                spatial:  gradient across the tile borders / the median gradient across all rows (columns)
              1.0 = the boundaries look like any other place; > 1 = something happens exactly there.
DRIFT         PSNR vs untiled over the first part of the clip minus over the last part: > 0 means the quality
              gets worse later in the clip.
"""
import numpy as np
from skimage.metrics import structural_similarity


def as_frames(video):
    """Raw decoder output [1,3,F,H,W] in ~[-1,1] -> a [F,H,W,3] view (no copy; works on a np.memmap)."""
    return video[0].transpose(1, 2, 3, 0)


def _frame(video, i):
    """Frame i as float32 in [0,1] (clipped the way it would be shown)."""
    return np.clip(np.asarray(video[i], dtype=np.float32) / 2 + 0.5, 0, 1)


def psnr_from_mse(mse):
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def mse_per_frame(ref, test, shift=0):
    """MSE of test[i] against ref[i - shift] for every frame where both exist (shift=1: test is one frame late)."""
    idx = range(max(shift, 0), min(len(test), len(ref) + shift))
    return np.array([np.mean((_frame(test, i).astype(np.float64) - _frame(ref, i - shift)) ** 2) for i in idx])


def psnr_per_frame(ref, test):
    return [psnr_from_mse(m) for m in mse_per_frame(ref, test)]


def ssim_per_frame(ref, test):
    return [float(structural_similarity(_frame(ref, i), _frame(test, i), channel_axis=-1, data_range=1.0))
            for i in range(len(ref))]


def temporal_seam(video, boundaries):
    """boundaries: transition indices t (the step from frame t to t+1), from TiledDecoder.seam_layout()["frames"]."""
    if not boundaries:
        return None
    diff = np.array([np.mean(np.abs(_frame(video, t + 1) - _frame(video, t))) for t in range(len(video) - 1)])
    median = float(np.median(diff))
    at = float(np.mean(diff[boundaries]))
    return {"score": at / median if median else float("nan"), "boundary_diff": at, "median_diff": median,
            "boundaries": len(boundaries)}


def spatial_seam(video, rows, cols):
    """rows / cols: transition indices (step from row y to y+1 / column x to x+1), from seam_layout()."""
    if not rows and not cols:
        return None
    grad_y, grad_x = 0.0, 0.0  # mean |difference| across each row / column step, averaged over frames
    for i in range(len(video)):
        f = _frame(video, i)
        grad_y = grad_y + np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)) / len(video)
        grad_x = grad_x + np.abs(np.diff(f, axis=1)).mean(axis=(0, 2)) / len(video)
    out, scores = {}, []
    for name, grad, lines in (("rows", grad_y, rows), ("cols", grad_x, cols)):
        if lines:
            median = float(np.median(grad))
            out[name] = {"score": float(np.mean(grad[lines])) / median if median else float("nan"),
                         "border_grad": float(np.mean(grad[lines])), "median_grad": median, "borders": len(lines)}
            scores.append(out[name]["score"])
    out["score"] = float(np.mean(scores))
    return out


def drift(mse_frames):
    """PSNR (from the pooled MSE) of the first third vs the last third of the clip."""
    third = max(1, len(mse_frames) // 3)
    first, last = psnr_from_mse(float(np.mean(mse_frames[:third]))), psnr_from_mse(float(np.mean(mse_frames[-third:])))
    return {"first_third_psnr_db": first, "last_third_psnr_db": last,
            "drop_db": first - last if np.isfinite(first) and np.isfinite(last) else 0.0, "frames_per_part": third}


def compare(ref, test, layout=None):
    """Everything above for one tiled decode (`test`) against the untiled decode (`ref`) of the same latents.

    Inputs are raw decoder outputs [1,3,F,H,W]. `layout` = TiledDecoder.seam_layout(...) of the tiled mode.
    """
    max_abs = max(float(np.abs(np.asarray(ref[0, :, i], np.float32) - test[0, :, i]).max())
                  for i in range(ref.shape[2]))  # on the raw output, frame by frame
    ref, test = as_frames(ref), as_frames(test)
    mse = mse_per_frame(ref, test)
    psnr, ssim = [psnr_from_mse(m) for m in mse], ssim_per_frame(ref, test)
    res = {
        "frames": len(mse), "max_abs_diff_raw": max_abs,
        "psnr_db": psnr_from_mse(float(mse.mean())), "psnr_min_db": min(psnr), "psnr_per_frame": psnr,
        "ssim": float(np.mean(ssim)), "ssim_min": min(ssim), "ssim_per_frame": ssim,
        "drift": drift(mse),
    }
    if len(mse) > 2:
        # Is the tiled video the untiled one shown a frame late? Compare frame i with reference frame i-1,
        # over the same frames (2..end) as the unshifted number next to it.
        res["psnr_frames_2plus_db"] = psnr_from_mse(float(mse[2:].mean()))
        res["psnr_frames_2plus_vs_previous_ref_frame_db"] = psnr_from_mse(float(mse_per_frame(ref, test, shift=1)[1:].mean()))
    if layout:
        res["temporal_seam"] = {"tiled": temporal_seam(test, layout["frames"]), "untiled": temporal_seam(ref, layout["frames"])}
        res["spatial_seam"] = {"tiled": spatial_seam(test, layout["rows"], layout["cols"]),
                               "untiled": spatial_seam(ref, layout["rows"], layout["cols"])}
    return res
''',
    "bench_worker.py": r'''"""s3 worker: one job per process, so every measurement starts from a clean heap.

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
''',
}
# --- END EMBEDDED ---

if __name__ == "__main__":
    main()
