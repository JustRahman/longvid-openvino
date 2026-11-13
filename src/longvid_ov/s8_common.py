"""Shared by the two step 8 Kaggle kernels (s8a_long_and_tuning, s8b_memory_and_int8): the step 2 / step 3
benchmark plumbing as one class. Install block, env / lscpu logging, VAE download, native-Conv3d IR conversion,
fresh-subprocess runs with 50 ms RSS sampling and a kill watchdog, per-run timeouts, save-after-every-run.

Imports nothing heavy: every measurement happens in a child process (bench_worker.py / s8_worker.py).
"""
import csv
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
CLIP_REPO, CLIP_FILE = "huggingface/documentation-images", "diffusers/hiker.mp4"
PINS = ["openvino==2026.4.1", "openvino-genai==2026.4.1.0"]
SAMPLE_S = 0.05
WATCHDOG_HEADROOM_GB = 1.5
WATCHDOG_MAX_GB = 28.5
MIN_AVAIL_GB = 1.0
REPORT_RESERVE_S = 300
GB = 2**30
T0 = time.time()


def log(msg):
    el = int(time.time() - T0)
    print(f"[s8 {time.strftime('%H:%M:%S')} +{el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d}] {msg}", flush=True)


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


class Bench:
    """One kernel run: folders, installs, env log, the watchdog, the list of result rows."""

    def __init__(self, tag, pkg, smoke, out_arg=None, extra_pip=(), budget_h=9.5):
        self.tag, self.pkg, self.smoke = tag, Path(pkg), smoke
        self.bw, self.s8w = self.pkg / "bench_worker.py", self.pkg / "s8_worker.py"
        root = self.pkg.parent.parent
        self.root = root if (root / "kaggle").is_dir() else None  # the repo, when run from it
        if not smoke and not Path("/kaggle/working").is_dir():
            sys.exit("The full run only works on Kaggle. Use --smoke on your own machine.")
        if smoke:
            self.out = Path(out_arg) if out_arg else (self.root or Path.cwd()) / "results" / f"{tag}_smoke"
            self.work = (self.root / "models" if self.root else Path("/tmp")) / f"{tag}_smoke"
            self.threads, self.budget_s, self.run_timeout_s, self.setup_timeout_s = 4, 20 * 60, 300, 300
        else:
            self.out = Path(out_arg) if out_arg else Path("/kaggle/working")
            self.work = Path(f"/kaggle/temp/{tag}_work") if Path("/kaggle/temp").is_dir() else Path(f"/tmp/{tag}_work")
            self.threads, self.budget_s, self.run_timeout_s, self.setup_timeout_s = 0, budget_h * 3600, 45 * 60, 40 * 60
        self.logs = self.out / "logs"
        for d in (self.out, self.logs, self.work):
            d.mkdir(parents=True, exist_ok=True)

        if not smoke:
            if not pip(*PINS):
                log("pinned OpenVINO not installable, falling back to latest stable")
                pip("openvino", "openvino-genai")
            pip("diffusers==0.39.0", "huggingface_hub", "safetensors", "psutil", "scikit-image", "imageio",
                "imageio-ffmpeg", "matplotlib", *extra_pip)
            if importlib.util.find_spec("torch") is None:  # Kaggle ships torch; only fetch the CPU wheel if missing
                pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        import psutil

        self.psutil = psutil
        limit = mem_limit_bytes(psutil)
        self.kill_rss = 5.0 * GB if smoke else min(limit - WATCHDOG_HEADROOM_GB * GB, WATCHDOG_MAX_GB * GB)
        lscpu = sh(["lscpu"])
        cpu_model = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")),
                         platform.processor() or platform.machine())
        self.meta = {
            "kernel": tag, "profile": "smoke" if smoke else "full", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_model": cpu_model, "nproc": os.cpu_count(), "threads": self.threads,
            "mem_limit_gb": round(limit / GB, 2), "swap_gb": round(psutil.swap_memory().total / GB, 2),
            "kill_rss_gb": round(self.kill_rss / GB, 2), "run_timeout_s": self.run_timeout_s,
            "budget_s": self.budget_s, "sample_s": SAMPLE_S, "setup": {},
            "versions": {"python": platform.python_version(), **{p: version(p) for p in (
                "openvino", "nncf", "torch", "diffusers", "numpy", "scikit-image", "psutil", "matplotlib")}},
        }
        env_txt = "\n".join([
            f"date: {self.meta['started']}", f"platform: {platform.platform()}", f"python: {sys.version}",
            *(f"{k}: {v}" for k, v in self.meta["versions"].items() if k != "python"),
            f"cpus: {os.cpu_count()}; inference threads: {self.threads or 'all'}",
            f"RAM limit (min of RAM and cgroup): {self.meta['mem_limit_gb']} GB; swap: {self.meta['swap_gb']} GB; "
            f"available now: {psutil.virtual_memory().available / GB:.2f} GB; watchdog kill RSS: {self.meta['kill_rss_gb']} GB",
            "", "--- lscpu ---", lscpu, "", "--- free -h ---", sh(["free", "-h"]), ""])
        (self.out / f"{tag}_env.txt").write_text(env_txt)
        print(env_txt, flush=True)
        self.rows, self.comps, self.stopped = [], [], {}

    # ---- time budget ----------------------------------------------------------------------------------
    def left(self):
        """Seconds of budget left for runs (the report reserve is already taken off)."""
        return self.budget_s - (time.time() - T0) - REPORT_RESERVE_S

    # ---- children -------------------------------------------------------------------------------------
    def child(self, name, worker, args, timeout_s):
        return run_child(self.psutil, worker, [*args, "--threads", self.threads], self.work / f"{name}.json",
                         self.logs / f"{name}.log", timeout_s, self.kill_rss, self.threads)

    def setup(self, name, worker, args, reuse=None, timeout_s=None):
        """A preparation job (IR conversion, encoding, ...). Result kept in meta["setup"]; the caller checks status."""
        if reuse and Path(reuse).exists():
            self.meta["setup"][name] = {"status": "reused", "path": str(reuse)}
        else:
            log(f"setup {name} ...")
            self.meta["setup"][name] = r = self.child(name, worker, args, timeout_s or self.setup_timeout_s)
            log(f"setup {name}: {r['status']} {r['detail']} | peak RSS {r['peak_rss_gb']} GB, {r['wall_s']} s")
            self.save()
        return self.meta["setup"][name]

    def need(self, name, worker, args, reuse=None, timeout_s=None):
        r = self.setup(name, worker, args, reuse, timeout_s)
        if r["status"] not in ("ok", "reused"):
            raise RuntimeError(f"setup {name} failed: {r['status']} {r.get('detail', '')}")
        return r

    def run(self, row, worker, args, timeout_s=None, series=None, min_left_s=60):
        """One measured run. `row` (must have "name") gets the result fields, is appended and saved.
        A series that hit oom / timeout is not tried again (longer videos of it are skipped)."""
        left = self.left()
        if series is not None and series in self.stopped:
            row.update(status="skipped", detail=self.stopped[series])
        elif left < min_left_s:
            row.update(status="skipped", detail="time budget used up")
        else:
            log(f"{row['name']} ...")
            row.update(self.child(row["name"], worker, args, min(timeout_s or self.run_timeout_s, left)))
            if series is not None and row["status"] in ("oom", "timeout"):
                self.stopped[series] = f"{row['status']} at {row['name']}"
        row["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.rows.append(row)
        log(f"{row['name']}: {row['status']} {row.get('detail', '')} | peak RSS {row.get('peak_rss_gb')} GB, "
            f"decode {row.get('decode_s')} s")
        self.save()
        return row

    def metrics(self, kind, test_row, test_file, ref_row, ref_file, layout=False, **tags):
        """PSNR / SSIM (+ seams with layout=True) of one saved decode against another, in a child (bench_worker)."""
        if test_row.get("status") != "ok" or ref_row.get("status") != "ok":
            return None
        name = f"metrics_{kind}_{test_row['name']}"
        r = self.child(name, self.bw, ["metrics", "--ref", ref_file, "--test", test_file,
                                       *(["--layout", json.dumps(test_row["layout"])] if layout else [])],
                       self.run_timeout_s)
        c = {"kind": kind, "test": test_row["name"], "ref": ref_row["name"], **tags, **r}
        self.comps.append(c)
        log(f"{name}: {r['status']} {r['detail']} | PSNR {r.get('psnr_db')} dB, SSIM {r.get('ssim')}")
        self.save()
        return c

    # ---- model + data ---------------------------------------------------------------------------------
    def prepare_model(self):
        """Clip, VAE folder (real on Kaggle, tiny random one for --smoke), FP32 decoder IR, latent statistics."""
        clip = self.root / "data" / "clip" / CLIP_FILE if self.root else None
        if not clip or not clip.exists():
            from huggingface_hub import hf_hub_download

            clip = Path(hf_hub_download(CLIP_REPO, CLIP_FILE, repo_type="dataset", local_dir=self.work / "clip"))
        self.clip = clip
        if self.smoke:
            self.vae_dir = self.work / "tiny_vae"
            self.need("tiny_vae", self.bw, ["tiny", "--vae-dir", self.vae_dir], self.vae_dir / "config.json")
            ir_kind, extra = "tiny_as2d", ["--conv3d-as-2d"]  # as2d: the only fast IR on an ARM Mac (step 1)
        else:
            from huggingface_hub import snapshot_download

            log(f"downloading {REPO} vae/ ...")
            self.vae_dir = Path(snapshot_download(REPO, allow_patterns=["vae/*"], local_dir=self.work / "ltx_vae_pt")) / "vae"
            ir_kind, extra = "real_native", []
        self.ir = self.work / ir_kind / "vae_decoder.xml"
        self.ir.parent.mkdir(exist_ok=True)
        self.need(f"convert_{ir_kind}", self.bw, ["convert", "--vae-dir", self.vae_dir, "--ir", self.ir, *extra], self.ir)
        self.stats = self.work / "latents_stats.npz"
        self.need("latent_stats", self.s8w, ["stats", "--vae-dir", self.vae_dir, "--save", self.stats], self.stats)

    def encode(self, width, height, frames):
        """Real latents of the hiker clip (ping-pong looped to `frames`). Falls back to diffusers' tiled encode."""
        latents = self.work / f"latents_{width}x{height}_{frames}f.npy"
        args = ["encode", "--vae-dir", self.vae_dir, "--clip", self.clip, "--frames", frames, "--height", height,
                "--width", width, "--save", latents]
        name = f"encode_{width}x{height}_{frames}f"
        enc = self.setup(name, self.bw, args, latents)
        if enc["status"] == "oom":  # one-shot encode does not fit
            enc = self.setup(name + "_tiled", self.bw, [*args, "--tiled-encode"])
        if enc["status"] not in ("ok", "reused"):
            raise RuntimeError(f"encode {width}x{height} {frames}f failed: {enc.get('detail')}")
        return latents

    # ---- results --------------------------------------------------------------------------------------
    def save(self):
        self.meta["runtime_min"] = round((time.time() - T0) / 60, 1)
        scalar = lambda v: v is None or isinstance(v, (str, int, float, bool))
        cols = []
        for r in self.rows:
            cols += [k for k, v in r.items() if scalar(v) and k not in cols]
        with open(self.out / f"{self.tag}_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
        (self.out / f"{self.tag}_results.json").write_text(
            json.dumps({**self.meta, "runs": self.rows, "comparisons": self.comps}, indent=2, default=str))


def load_saved(folder, tag):
    """(rows, comps, meta) from <folder>/<tag>_results.json, for --report-only."""
    saved = json.loads((Path(folder) / f"{tag}_results.json").read_text())
    return saved.pop("runs"), saved.pop("comparisons"), saved


def setup_lines(meta, title):
    """The "Setup" section both reports start with (measured values of this run only)."""
    smoke = " (SMOKE TEST: tiny random-weight VAE, numbers mean nothing)" if meta["profile"] == "smoke" else ""
    return [
        f"# {title}{smoke}", "",
        f"Generated from the run started {meta['started']}; runtime {meta.get('runtime_min')} min. Every number below "
        f"was measured in this run (raw: `{meta['kernel']}_results.json`, `.csv`). Single runs, not averaged. "
        "Timings are only comparable inside this run.", "",
        "## Setup",
        f"- Machine: {meta['cpu_model']}, {meta['nproc']} logical CPUs, {meta['threads'] or 'all'} inference threads, RAM "
        f"limit {meta['mem_limit_gb']} GB, swap {meta['swap_gb']} GB. Each run is a fresh subprocess; the parent samples "
        f"its RSS every {meta['sample_s'] * 1000:.0f} ms (plus the kernel's VmHWM) and kills it above {meta['kill_rss_gb']} GB "
        "or when system-available RAM drops under 1 GB (`OOM`), or at its timeout.",
        "- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()) + ".",
        *(f"- IR conversion `{k}`: {s.get('convert_s')} s, peak RSS {s['peak_rss_gb']:.2f} GB, weights "
          f"{s.get('ir_bytes', 0) / GB:.2f} GB, FP32, {s.get('conv3d_rewritten_as_2d')} Conv3d rewritten as Conv2d."
          for k, s in meta["setup"].items() if k.startswith("convert_") and s["status"] == "ok"),
        *(f"- Setup job `{k}`: **{s['status']}** {s.get('detail', '')}"
          for k, s in meta["setup"].items() if s["status"] not in ("ok", "reused")),
    ]


def db(x):
    return "-" if x is None else "identical" if x == float("inf") else f"{x:.2f}"


def num(x, fmt=".2f", none="-"):
    return none if x is None else format(x, fmt)
