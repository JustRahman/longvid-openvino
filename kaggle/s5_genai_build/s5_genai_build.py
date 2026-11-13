"""Step 5 of LongVid-OV: build OpenVINO GenAI with the temporal VAE tiling patch, test it, check it against the reference.

Kaggle "script" kernel (CPU, internet on, 6 h budget). Single file: the patch (patches/0001-ltx-temporal-vae-tiling.patch)
and the Python reference (src/longvid_ov/tiled_decode.py, bench_worker.py) are embedded at the bottom by
scripts/s5_embed_patch.py. It ONLY runs on Kaggle (it builds C++; never run it on the Mac).

Stages (results in /kaggle/working/s5_results.json after every stage, logs in /kaggle/working/s5_logs)
  1. clone   openvino.genai at BASE_COMMIT (the HEAD of third_party/openvino.genai the patch was written against), submodules
  2. patch   git apply the embedded patch
  3. openvino  install the OpenVINO + tokenizers nightly that pyproject.toml of that commit asks for
  4. build   pip wheel . (Ninja, 4 cores), wheel kept in /kaggle/working/wheels; time logged
  5. install the wheel, import check
  6. tests   tests/python_tests/test_video_generation.py (existing tests + the new tiling tests) and test_ltx2_rope.py
  7. parity  real LTX-Video VAE, 512x320, 49 and 97 frames, same latents, one decode per fresh process (peak RSS):
               cpp_untiled / cpp_tiled        AutoencoderKLLTXVideo.decode() of the built wheel (uint8 output)
               py_untiled / py_tiled          tiled_decode.py on the raw decoder IR, blended in float, then -> uint8
               py_tiling_on_cpp_decoder       tiled_decode.py driving the wheel's UNTILED decode() chunk by chunk
             cpp_tiled vs py_tiling_on_cpp_decoder isolates the tiling logic (same decoder, same uint8 chunks).
  8. report  s5_report.md
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

# False = a quick re-run: build + only the new tests (~30 min)
RUN_FULL_SUITE = False  # all of test_video_generation.py + test_ltx2_rope.py (stage tests_video_generation)
RUN_PARITY = False  # stage 7: parity on the real LTX-Video VAE

GENAI_REPO = "https://github.com/openvinotoolkit/openvino.genai.git"
OV_INDEXES = ["--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/pre-release",
              "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly"]
LTX_REPO = "Lightricks/LTX-Video"
CLIP_REPO, CLIP_FILE = "huggingface/documentation-images", "diffusers/hiker.mp4"
WIDTH, HEIGHT, FRAMES = 512, 320, [49, 97]
TILE_T = (16, 8)  # chunk of 16 video frames (+1 causal), stride 8: diffusers defaults
NEW_TESTS = "TestAutoEncoderKLLTXVideoTemporalTiling"
# (name, worker job, tiled)
VARIANTS = [("cpp_untiled", "cpp", False), ("cpp_tiled", "cpp", True), ("py_untiled", "pyref", False),
            ("py_tiled", "pyref", True), ("py_tiling_on_cpp_decoder", "pyref_on_cpp", True)]
# (test, reference, what it shows, max abs diff in uint8 levels that counts as a pass or None = informational)
COMPARISONS = [
    ("cpp_tiled", "py_tiling_on_cpp_decoder", "tiling logic: C++ vs Python reference on the same decoder", 1),
    ("cpp_tiled", "py_tiled", "C++ (blend after clamp, uint8) vs Python reference (blend raw float, then clamp)", None),
    ("cpp_untiled", "py_untiled", "no tiling: GenAI post-processing vs NumPy post-processing", None),
    ("cpp_tiled", "cpp_untiled", "cost of tiling, C++", None),
    ("py_tiled", "py_untiled", "cost of tiling, Python reference", None),
]
BUDGET_S = 6 * 3600 - 15 * 60  # leave 15 min to write the report
SAMPLE_S = 0.05
GB = 2**30
T0 = time.time()
OUT = Path("/kaggle/working")
LOGS = OUT / "s5_logs"
WORK = Path("/kaggle/temp/s5_work") if Path("/kaggle/temp").is_dir() else Path("/tmp/s5_work")
RES = {"stages": {}}

# The measured jobs of stage 7. Written next to the embedded tiled_decode.py and run as: s5_worker.py <job> ... --out x.json
WORKER = r'''
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiled_decode import OVDecoder, TiledDecoder, TilingConfig  # noqa: E402


def save(out, **kw):
    with open(out + ".tmp", "w") as f:
        json.dump(kw, f)
    os.replace(out + ".tmp", out)


def to_uint8(raw):
    """[1,3,F,H,W] float32 in ~[-1,1] -> [1,F,H,W,3] uint8: the post-processing GenAI merges into the decoder model."""
    return np.rint(np.clip(raw * np.float32(0.5) + np.float32(0.5), 0.0, 1.0) * np.float32(255.0)).astype(np.uint8).transpose(0, 2, 3, 4, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["cpp", "pyref", "pyref_on_cpp"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", required=True)  # GenAI layout: vae_decoder/openvino_model.xml + config.json, transformer/config.json
    ap.add_argument("--latents", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--tile-t", type=int, nargs=2, default=[0, 8], metavar=("MIN", "STRIDE"))  # MIN 0 = no tiling
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    try:  # make this process the kernel OOM killer's first choice, not the parent
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass

    z = np.ascontiguousarray(np.load(a.latents)[:, :, : (a.frames - 1) // 8 + 1], dtype=np.float32)
    decoder_dir = os.path.join(a.model_dir, "vae_decoder")
    tile_min, tile_stride = a.tile_t
    save(a.out, phase="load")
    t = time.perf_counter()
    calls = None
    if a.job in ("cpp", "pyref_on_cpp"):
        import openvino as ov
        import openvino_genai as ov_genai

        vae = ov_genai.AutoencoderKLLTXVideo(decoder_dir, "CPU", INFERENCE_PRECISION_HINT="f32")
    if a.job == "cpp":
        # np.array: decode() may return the output buffer of the infer request
        run = lambda: np.array(vae.decode(ov.Tensor(z), tile_min, tile_stride).data)
    else:
        if a.job == "pyref":
            decoder = OVDecoder(os.path.join(decoder_dir, "openvino_model.xml"))
            finish = to_uint8
        else:  # the reference algorithm on top of the wheel's untiled decode: uint8 [1,F,H,W,3] -> float32 [1,3,F,H,W]
            decoder = lambda lat: np.array(vae.decode(ov.Tensor(lat)).data).transpose(0, 4, 1, 2, 3).astype(np.float32)
            finish = lambda video: np.rint(video).astype(np.uint8).transpose(0, 2, 3, 4, 1)
        cfg = TilingConfig(tile_sample_min_num_frames=tile_min, tile_sample_stride_num_frames=tile_stride) if tile_min else None
        tiled = TiledDecoder(decoder, cfg)
        run = lambda: finish(tiled.decode(z, "temporal" if tile_min else "none"))
    save(a.out, phase="decode", load_s=round(time.perf_counter() - t, 2))
    t = time.perf_counter()
    video = run()
    decode_s = round(time.perf_counter() - t, 2)
    if a.job != "cpp":
        calls = tiled.calls
    assert video.dtype == np.uint8 and video.shape[1] == a.frames, (video.dtype, video.shape)
    np.save(a.save, video)
    save(a.out, phase="done", decode_s=decode_s, decoder_calls=calls, latent_shape=list(z.shape), output_shape=list(video.shape))


if __name__ == "__main__":
    main()
'''


def log(msg):
    el = int(time.time() - T0)
    print(f"[s5 {time.strftime('%H:%M:%S')} +{el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d}] {msg}", flush=True)


def left():
    return BUDGET_S - (time.time() - T0)


def save_results():
    tmp = OUT / "s5_results.json.tmp"
    tmp.write_text(json.dumps(RES, indent=1))
    tmp.replace(OUT / "s5_results.json")


def stage(name, **kw):
    RES["stages"][name] = {**RES["stages"].get(name, {}), **kw, "finished_at": time.strftime("%H:%M:%S")}
    save_results()
    log(f"stage {name}: {json.dumps(kw)[:600]}")


def tail(path, n=40):
    try:
        return "\n".join(Path(path).read_text(errors="replace").strip().splitlines()[-n:])
    except OSError:
        return ""


def run(name, cmd, cwd=None, env=None, timeout=3600):
    """Run cmd with its output in s5_logs/<name>.log. Returns (ok, seconds). Prints the end of the log on failure."""
    path = LOGS / f"{name}.log"
    cmd = [str(c) for c in cmd]
    log(f"$ {' '.join(cmd)}    -> s5_logs/{path.name}")
    t = time.time()
    beat = t + 300
    killed = False
    with open(path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            time.sleep(1)
            if time.time() - t > timeout:
                killed = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            elif time.time() > beat:
                log(f"  ... {name} running for {int(time.time() - t)} s")
                beat += 300
    seconds = round(time.time() - t, 1)
    ok = proc.returncode == 0 and not killed
    if not ok:
        log(f"{name} FAILED ({'timeout' if killed else 'exit code ' + str(proc.returncode)}) after {seconds} s. Log tail:")
        print(tail(path), flush=True)
    return ok, seconds


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"not available ({e!r})"


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def pip(name, *args, timeout=1800):
    return run(name, [sys.executable, "-m", "pip", "install", *args], timeout=timeout)[0]


def mem_limit_bytes(psutil):
    limit = psutil.virtual_memory().total
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            limit = min(limit, int(Path(p).read_text().strip()))
        except (OSError, ValueError):
            pass
    return limit


def vm_hwm(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def run_child(psutil, name, script, args, timeout_s, kill_rss):
    """One job in a fresh process, RSS sampled every 50 ms (the step 2 / step 3 harness). Returns its result fields."""
    out_json, log_path = WORK / f"{name}.json", LOGS / f"{name}.log"
    out_json.unlink(missing_ok=True)
    env = {**os.environ, "USE_TF": "0", "USE_FLAX": "0"}
    peak = rss = 0
    killed = None
    t = time.perf_counter()
    with open(log_path, "w") as lf:
        proc = subprocess.Popen([sys.executable, str(script), *map(str, args), "--out", str(out_json)],
                                stdout=lf, stderr=subprocess.STDOUT, env=env)
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss
            except psutil.Error:
                rss = 0
            peak = max(peak, rss, vm_hwm(proc.pid))
            if rss > kill_rss or psutil.virtual_memory().available < 1.0 * GB:
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
        res = json.loads(out_json.read_text())
    except (OSError, ValueError):
        res = {}
    phase = res.pop("phase", "start")
    last = (tail(log_path, 1) or "no output")[:300]
    if killed == "oom":
        status, detail = "oom", f"watchdog kill during {phase}: RSS {rss / GB:.1f} GB (limit {kill_rss / GB:.1f} GB)"
    elif killed == "timeout":
        status, detail = "timeout", f"killed after {timeout_s:.0f} s during {phase}"
    elif proc.returncode == 0 and phase == "done":
        status, detail = "ok", ""
    elif proc.returncode == -9:
        status, detail = "oom", f"SIGKILL during {phase} (kernel OOM killer)"
    else:
        status, detail = "error", f"exit code {proc.returncode} during {phase}: {last}"
    r = {"status": status, "detail": detail, "peak_rss_gb": round(peak / GB, 3), "wall_s": round(wall, 1), **res}
    log(f"  {name}: {status} {detail} | peak RSS {r['peak_rss_gb']} GB, {r['wall_s']} s")
    return r


def patch_bytes():
    data = zlib.decompress(base64.b64decode(PATCH_B64))
    assert hashlib.sha256(data).hexdigest() == PATCH_SHA256, "embedded patch is corrupted: run scripts/s5_embed_patch.py"
    return data


# ---- stages 1-5: clone, patch, OpenVINO nightly, build, install ---------------------------------------------------
def build(src):
    src.mkdir(parents=True, exist_ok=True)
    git = ["git", "-C", src]
    ok = all(run(f"clone_{i}", c, timeout=1800)[0] for i, c in enumerate([
        [*git, "init", "-q"],
        [*git, "remote", "add", "origin", GENAI_REPO],
        [*git, "fetch", "-q", "--depth", "1", "origin", BASE_COMMIT],
        [*git, "checkout", "-q", "FETCH_HEAD"],
        [*git, "submodule", "update", "--init", "--recursive", "--depth", "1"],
    ]))
    head = sh([*git, "rev-parse", "HEAD"])
    log(f"openvino.genai HEAD: {head}   (expected {BASE_COMMIT})")
    stage("clone", ok=ok and head == BASE_COMMIT, head=head, expected=BASE_COMMIT,
          subject=sh([*git, "log", "-1", "--format=%s (%cd)", "--date=short"]),
          submodules=sh([*git, "submodule", "status", "--recursive"]))
    if not (ok and head == BASE_COMMIT):
        return None

    patch = WORK / "0001-ltx-temporal-vae-tiling.patch"
    patch.write_bytes(patch_bytes())
    shutil.copy(patch, OUT / patch.name)
    ok = run("patch_check", [*git, "apply", "--check", "-v", patch])[0] and run("patch_apply", [*git, "apply", "-v", patch])[0]
    stage("patch", ok=ok, sha256=PATCH_SHA256, diffstat=sh([*git, "diff", "--stat"]))
    if not ok:
        return None
    # committed (locally, in the throw-away clone) so that `git diff` after the build shows only what the build changed
    sh([*git, "add", "-A"])
    sh([*git, "-c", "user.name=s5", "-c", "user.email=s5@localhost", "commit", "-q", "-m", "ltx temporal vae tiling"])

    # The OpenVINO the wheel is built against must be the one it runs with: install it first, then pin it for the build.
    pyproject = (src / "pyproject.toml").read_text()
    specs = sorted(set(re.findall(r'"(openvino(?:_tokenizers)?~=[^"]+)"', pyproject)))
    log(f"OpenVINO requirements of this commit: {specs}")
    ok = bool(specs) and pip("pip_openvino", "--pre", "-U", *specs, *OV_INDEXES)
    ov_version, tok_version = version("openvino"), version("openvino-tokenizers")
    constraints = WORK / "constraints.txt"
    constraints.write_text(f"openvino=={ov_version}\nopenvino-tokenizers=={tok_version}\n")
    stage("openvino", ok=ok, requirements=specs, openvino=ov_version, openvino_tokenizers=tok_version)
    if not ok:
        return None

    pip("pip_ninja", "ninja")
    wheels = OUT / "wheels"
    wheels.mkdir(exist_ok=True)
    # Ninja: pyproject.toml builds with a bare `--parallel`, which means `make -j` (unlimited) with Makefiles.
    # No CI variables: with CI set the build fails if the committed .pyi stub differs from the generated one;
    # without it the generated stub is copied into the source tree, and the difference is saved below.
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "TF_BUILD", "JENKINS_URL")}
    env.update(CMAKE_GENERATOR="Ninja", CMAKE_BUILD_PARALLEL_LEVEL=str(os.cpu_count() or 4), PIP_CONSTRAINT=str(constraints))
    ok, seconds = run("build_wheel", [sys.executable, "-m", "pip", "wheel", "-v", "--no-deps", "--wheel-dir", wheels,
                                      *OV_INDEXES, src], env=env, timeout=min(3 * 3600, max(600, left() - 3600)))
    built = sorted(wheels.glob("openvino_genai-*.whl"))
    stub_diff = sh([*git, "diff", "--", "src/python/openvino_genai"])
    (OUT / "s5_stub_diff.patch").write_text(stub_diff + "\n")
    stage("build", ok=ok and bool(built), seconds=seconds, minutes=round(seconds / 60, 1), cores=os.cpu_count(),
          wheel=built[-1].name if built else None, wheel_mb=round(built[-1].stat().st_size / 2**20, 1) if built else None,
          compiler=sh(["c++", "--version"]).splitlines()[:1], cmake_generator="Ninja",
          stub_matches_generated=not stub_diff.strip(), build_changed_files=sh([*git, "status", "--short"])[:500],
          stub_diff_lines=len(stub_diff.splitlines()))
    if not (ok and built):
        return None

    ok = pip("pip_wheel_install", "--force-reinstall", "--no-deps", built[-1])
    check = ("import openvino, openvino_genai as g; c = g.VideoGenerationConfig(); "
             "print(openvino.__version__, g.__version__, c.vae_tile_sample_min_num_frames, c.vae_tile_sample_stride_num_frames)")
    ok = ok and run("import_check", [sys.executable, "-c", check], cwd=WORK)[0]
    stage("install", ok=ok, import_check=tail(LOGS / "import_check.log", 3))
    return constraints if ok else None


# ---- stage 6: the repo's tests ------------------------------------------------------------------------------------
def junit(path):
    try:  # the file is written by our own pytest run a moment earlier
        cases = list(ET.parse(path).getroot().iter("testcase"))
    except (OSError, ET.ParseError):
        return None
    kind = lambda c: next((k for k in ("failure", "error", "skipped") if c.find(k) is not None), "passed")
    summary = {k: 0 for k in ("passed", "failure", "error", "skipped")}
    bad = []
    for c in cases:
        summary[kind(c)] += 1
        if kind(c) in ("failure", "error"):
            node = c.find(kind(c))
            bad.append({"test": f"{c.get('classname')}::{c.get('name')}", "new_test": NEW_TESTS in c.get("classname", ""),
                        "message": (node.get("message") or node.text or "")[:400]})
    return {**summary, "total": len(cases), "failed_tests": bad}


def tests(src, constraints):
    reqs = (src / "tests/python_tests/requirements.txt").read_text().splitlines()
    wanted = [ln.strip() for ln in reqs if re.match(r"(diffusers|optimum-intel|pytest|transformers|hf_transfer|tqdm)\b", ln.strip())]
    ok = pip("pip_test_deps", "-c", constraints, "--extra-index-url", "https://download.pytorch.org/whl/cpu", *OV_INDEXES,
             *wanted, "torch", "torchvision", "huggingface_hub", "safetensors", "psutil", "imageio", "imageio-ffmpeg", "pillow")
    after = {"openvino": version("openvino"), "openvino_tokenizers": version("openvino-tokenizers"),
             "openvino_genai": version("openvino-genai")}
    stage("test_deps", ok=ok, requirements=wanted, **after, torch=version("torch"), transformers=version("transformers"),
          diffusers=version("diffusers"), optimum_intel=version("optimum-intel"))
    env = {**os.environ, "OV_CACHE": str(WORK / "ov_cache"), "USE_TF": "0", "USE_FLAX": "0"}
    pytest = [sys.executable, "-m", "pytest", "-v", "-rA", "-p", "no:cacheprovider"]
    files = ["tests/python_tests/test_video_generation.py", "tests/python_tests/test_ltx2_rope.py"]
    for name, extra, timeout in (("tests_new", [files[0], "-k", NEW_TESTS], 40 * 60), ("tests_video_generation", files, 120 * 60)):
        if name == "tests_video_generation" and not RUN_FULL_SUITE:
            stage(name, ok=False, skipped="skipped (RUN_FULL_SUITE = False)")
            continue
        if left() < 90 * 60:
            stage(name, ok=False, skipped="not enough time left")
            continue
        xml = OUT / f"s5_{name}.xml"
        ok, seconds = run(name, [*pytest, f"--junitxml={xml}", *extra], cwd=src, env=env, timeout=min(timeout, left() - 60 * 60))
        stage(name, ok=ok, seconds=seconds, summary_line=tail(LOGS / f"{name}.log", 1), junit=junit(xml))


# ---- stage 7: parity on the real VAE ------------------------------------------------------------------------------
def compare(test, ref):
    """uint8 [1,F,H,W,3] vs the same: max abs diff (levels), PSNR (peak 255), share of values that differ, worst frame."""
    import numpy as np

    worst, sq, diff, count = 0, [], 0, 0
    for f in range(test.shape[1]):
        d = test[0, f].astype(np.int16) - ref[0, f]
        worst = max(worst, int(np.abs(d).max()))
        sq.append(float(np.mean(d.astype(np.float32) ** 2)))
        diff += int(np.count_nonzero(d))
        count += d.size
    psnr = lambda mse: None if mse == 0 else round(10 * math.log10(255.0**2 / mse), 2)  # None = identical
    return {"max_abs_diff": worst, "psnr_db": psnr(sum(sq) / len(sq)), "worst_frame_psnr_db": psnr(max(sq)),
            "worst_frame": sq.index(max(sq)), "differing_values_share": round(diff / count, 6)}


def parity():
    import numpy as np
    import psutil
    from huggingface_hub import hf_hub_download, snapshot_download

    kill_rss = mem_limit_bytes(psutil) - 1.5 * GB
    pkg = WORK / "longvid_ov"
    pkg.mkdir(exist_ok=True)
    for name, text in EMBEDDED.items():
        (pkg / name).write_text(text)
    (pkg / "s5_worker.py").write_text(WORKER)
    bench, worker = pkg / "bench_worker.py", pkg / "s5_worker.py"
    P = RES["parity"] = {"width": WIDTH, "height": HEIGHT, "tile_t": TILE_T, "setup": {}, "runs": [], "comparisons": []}

    ltx = Path(snapshot_download(LTX_REPO, allow_patterns=["vae/*", "transformer/config.json"], local_dir=WORK / "ltx"))
    clip = hf_hub_download(CLIP_REPO, CLIP_FILE, repo_type="dataset", local_dir=WORK / "clip")
    # The layout AutoencoderKLLTXVideo(vae_decoder_path) reads: the decoder IR with the VAE config next to it,
    # and ../transformer/config.json (patch sizes).
    model = WORK / "genai_model"
    (model / "vae_decoder").mkdir(parents=True, exist_ok=True)
    (model / "transformer").mkdir(exist_ok=True)
    shutil.copy(ltx / "vae" / "config.json", model / "vae_decoder" / "config.json")
    shutil.copy(ltx / "transformer" / "config.json", model / "transformer" / "config.json")
    ir = model / "vae_decoder" / "openvino_model.xml"
    P["setup"]["convert"] = r = run_child(psutil, "convert", bench, ["convert", "--vae-dir", ltx / "vae", "--ir", ir], 1800, kill_rss)
    if r["status"] != "ok":
        return save_results()
    latents = WORK / f"latents_{WIDTH}x{HEIGHT}_{max(FRAMES)}f.npy"
    enc = ["encode", "--vae-dir", ltx / "vae", "--clip", clip, "--frames", max(FRAMES), "--height", HEIGHT, "--width", WIDTH,
           "--save", latents]
    P["setup"]["encode"] = r = run_child(psutil, "encode", bench, enc, 1800, kill_rss)
    if r["status"] == "oom":  # the one-shot PyTorch encode does not fit: diffusers' tiled + framewise encode
        P["setup"]["encode_tiled"] = r = run_child(psutil, "encode_tiled", bench, [*enc, "--tiled-encode"], 1800, kill_rss)
    save_results()
    if r["status"] != "ok":
        return

    for frames in FRAMES:
        saved = {}
        for name, job, tiled in VARIANTS:
            if left() < 20 * 60:
                P["runs"].append({"frames": frames, "variant": name, "status": "skipped", "detail": "time budget"})
                continue
            npy = WORK / f"{name}_{frames}f.npy"
            r = run_child(psutil, f"{name}_{frames}f", worker,
                          [job, "--model-dir", model, "--latents", latents, "--frames", frames, "--save", npy,
                           *(["--tile-t", *TILE_T] if tiled else [])], min(40 * 60, left() - 10 * 60), kill_rss)
            P["runs"].append({"frames": frames, "variant": name, **r})
            if r["status"] == "ok":
                saved[name] = npy
            save_results()
        for test, ref, what, tolerance in COMPARISONS:
            if test in saved and ref in saved:
                c = compare(np.load(saved[test], mmap_mode="r"), np.load(saved[ref], mmap_mode="r"))
                c.update(frames=frames, test=test, ref=ref, what=what, tolerance=tolerance,
                         passed=None if tolerance is None else c["max_abs_diff"] <= tolerance)
                P["comparisons"].append(c)
                log(f"  {frames} frames, {test} vs {ref}: {json.dumps(c)}")
        save_results()
        for npy in saved.values():
            npy.unlink(missing_ok=True)


# ---- stage 8: report ----------------------------------------------------------------------------------------------
def write_report():
    S = RES["stages"]
    g = lambda name, key, default="n/a": S.get(name, {}).get(key, default)
    L = ["# Step 5 report: OpenVINO GenAI with temporal VAE tiling for LTX-Video (Kaggle build)", "",
         f"Run started {RES['started']}, runtime {(time.time() - T0) / 60:.0f} min. {RES['machine']}", "",
         "## Build",
         f"- openvino.genai commit: `{g('clone', 'head')}` {g('clone', 'subject')}; expected `{BASE_COMMIT}`: "
         f"{'match' if g('clone', 'ok') is True else 'MISMATCH / clone failed'}",
         f"- patch `0001-ltx-temporal-vae-tiling.patch` (sha256 `{PATCH_SHA256[:16]}...`): "
         f"{'applied cleanly' if g('patch', 'ok') is True else 'DID NOT APPLY' if 'patch' in S else 'not reached'}",
         f"- OpenVINO: {g('openvino', 'openvino')}, tokenizers {g('openvino', 'openvino_tokenizers')} (from {g('openvino', 'requirements')})",
         f"- wheel build: {'ok' if g('build', 'ok') is True else 'FAILED' if 'build' in S else 'not reached'}, "
         f"{g('build', 'minutes')} min on {g('build', 'cores')} cores ({g('build', 'compiler')}), wheel `{g('build', 'wheel')}` "
         f"({g('build', 'wheel_mb')} MB) in `wheels/`",
         f"- committed `.pyi` stub equals the generated one: {g('build', 'stub_matches_generated')} "
         f"(difference: `s5_stub_diff.patch`, {g('build', 'stub_diff_lines')} lines)",
         f"- install + import check: {g('install', 'ok')} `{g('install', 'import_check')}`",
         f"- versions after installing the test requirements: openvino {g('test_deps', 'openvino')}, torch {g('test_deps', 'torch')}, "
         f"transformers {g('test_deps', 'transformers')}, diffusers {g('test_deps', 'diffusers')}, optimum-intel {g('test_deps', 'optimum_intel')}",
         "", "## Tests", "", "| run | result | passed | failed | errors | skipped | time s |", "|---|---|---|---|---|---|---|"]
    for name in ("tests_new", "tests_video_generation"):
        s, j = S.get(name), (S.get(name) or {}).get("junit") or {}
        if s:
            L.append(f"| `{name}` | {'ok' if s.get('ok') else s.get('skipped') or 'FAILED'} | {j.get('passed', '-')} | {j.get('failure', '-')} | "
                     f"{j.get('error', '-')} | {j.get('skipped', '-')} | {s.get('seconds', '-')} |")
    L += ["", f"`tests_new` = `-k {NEW_TESTS}`; `tests_video_generation` = all of `test_video_generation.py` + `test_ltx2_rope.py`.", ""]
    for name in ("tests_new", "tests_video_generation"):
        for b in ((S.get(name) or {}).get("junit") or {}).get("failed_tests", []):
            L.append(f"- `{name}` FAILED `{b['test']}` ({'new test' if b['new_test'] else 'existing test'}): {b['message']!r}")
    P = RES.get("parity")
    L += ["", f"## Parity on the real LTX-Video VAE, {WIDTH}x{HEIGHT}, chunk {TILE_T[0]} / stride {TILE_T[1]} frames", ""]
    if not P:
        L.append("Not reached." if RUN_PARITY else "Skipped (RUN_PARITY = False).")
    else:
        L += [f"Setup: {json.dumps({k: (v['status'], v['detail'], v['peak_rss_gb'], v['wall_s']) for k, v in P['setup'].items()})}", "",
              "| frames | variant | status | peak RSS GB | decode s | decoder calls |", "|---|---|---|---|---|---|"]
        L += [f"| {r['frames']} | `{r['variant']}` | {r['status']} {r.get('detail', '')} | {r.get('peak_rss_gb', '-')} | "
              f"{r.get('decode_s', '-')} | {r.get('decoder_calls') or '-'} |" for r in P["runs"]]
        L += ["", "Differences are in uint8 levels (0..255) of the final video; PSNR with peak 255, `identical` = no value differs.", "",
              "| frames | test | reference | what | max abs diff | PSNR dB | worst frame PSNR dB | values that differ | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        db = lambda v: "identical" if v is None else v
        L += [f"| {c['frames']} | `{c['test']}` | `{c['ref']}` | {c['what']} | {c['max_abs_diff']} | {db(c['psnr_db'])} | "
              f"{db(c['worst_frame_psnr_db'])} | {100 * c['differing_values_share']:.4f} % | "
              f"{'-' if c['passed'] is None else 'PASS' if c['passed'] else 'FAIL'} (tolerance {c['tolerance']}) |" for c in P["comparisons"]]
    (OUT / "s5_report.md").write_text("\n".join(L) + "\n")
    log("wrote s5_report.md")


def selfcheck():
    """No build, no heavy imports: the embedded patch decodes, matches its checksum and the embedded sources compile."""
    data = patch_bytes()
    for name, text in {**EMBEDDED, "s5_worker.py": WORKER}.items():
        compile(text, name, "exec")
    print(f"ok: base {BASE_COMMIT}, patch {len(data)} bytes sha256 {PATCH_SHA256}, embedded {sorted(EMBEDDED)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="verify the embedded patch / sources, run nothing")
    a, _ = ap.parse_known_args()  # Kaggle runs scripts through a kernel launcher with its own argv
    if a.selfcheck:
        return selfcheck()
    if not OUT.is_dir():
        sys.exit("This job builds OpenVINO GenAI from source and only runs on Kaggle. Locally: --selfcheck.")
    for d in (LOGS, WORK):
        d.mkdir(parents=True, exist_ok=True)
    lscpu = sh(["lscpu"])
    cpu = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")), platform.machine())
    RES.update(started=time.strftime("%Y-%m-%d %H:%M:%S"), base_commit=BASE_COMMIT, patch_sha256=PATCH_SHA256,
               machine=f"{cpu}, {os.cpu_count()} CPUs, python {platform.python_version()}, {platform.platform()}.")
    (OUT / "s5_env.txt").write_text("\n".join([RES["started"], RES["machine"], sh(["free", "-g"]), sh(["df", "-h", "/kaggle"]),
                                               sh(["git", "--version"]), sh(["c++", "--version"]), "", lscpu, ""]))
    print((OUT / "s5_env.txt").read_text(), flush=True)
    src = WORK / "openvino.genai"
    try:
        constraints = build(src)
        if constraints:
            tests(src, constraints)
            if RUN_PARITY and left() > 30 * 60:
                parity()
    except Exception as e:  # keep what was measured
        import traceback

        RES["crash"] = traceback.format_exc()
        log(f"CRASH: {e!r}\n{RES['crash']}")
    save_results()
    write_report()
    print((OUT / "s5_report.md").read_text(), flush=True)


# --- BEGIN GENERATED (scripts/s5_embed_patch.py; do not edit by hand) ---
BASE_COMMIT = "1e573f46bc7143d280558fb21bdadfce65c841b9"
PATCH_SHA256 = "4b1f9cf084f5a03a7b896ac51b3481ebfc2c5cdc9e370b8826d211b75ae0a590"
PATCH_B64 = (
    "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"
    "cIbivlaz1e+T1sg8GZqDVuu/AQJy4gY="
)
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
# --- END GENERATED ---

if __name__ == "__main__":
    main()
