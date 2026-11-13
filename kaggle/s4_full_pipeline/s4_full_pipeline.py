"""Step 4 of LongVid-OV: full LTX-Video on OpenVINO (INT8) + OpenVINO GenAI, and a long video from chained windows.

Kaggle "script" kernel (CPU, 4 vCPU, ~30 GB RAM, internet on). Single file: the per-job worker (WORKER_SRC) and
src/longvid_ov/tiled_decode.py (TILED_SRC, embedded copy) are written to disk at start, because every heavy job
runs in a FRESH subprocess that the parent watches (RSS of the whole process tree every 50 ms, kill at ~28 GB).

Stages (results are saved after every job, so a timeout still leaves s4_results.json / .csv / s4_report.md)
  A. pip installs (OpenVINO + tokenizers + GenAI nightly, optimum-intel, nncf, diffusers), versions / lscpu / RAM / df
  B. an INT8 OpenVINO LTX-Video folder, first route that works:
       1a  optimum-cli export openvino --task text-to-video --weight-format int8      (everything loaded in fp32)
       1b  optimum main_export, pipeline loaded in bf16, weights >1B params -> int8, VAEs compressed afterwards
       2   component-wise export (text encoder bf16, transformer, VAE), nncf INT8 per component, assemble layout
       3   community FP32 IR MX69Moi/LTX_Video_i2v_ov, nncf INT8 per component, convert tokenizer, assemble
     then load it in openvino_genai.Text2VideoPipeline / Image2VideoPipeline on CPU
  C. Text2Video 512x320: 25 frames / 8 steps (memory, time) and 49 frames / 20 steps (quality clip)
  D. long video: window 1 = Text2Video, windows 2..N = Image2Video on the last frame of the previous window;
     seam score at every window boundary, drift across windows, contact sheet, long mp4
  E. if time remains: VAE-encode the long video, decode its latents untiled vs temporal-tiled (our tiled_decode),
     also at 2x / 4x the length (latents extended by ping-pong), peak RSS per decode
  F. s4_report.md, model manifest, the INT8 folder itself if it is under 12 GB

Nothing here is a hard-coded result: every number in the outputs is measured in the run that wrote it.

  python s4_full_pipeline.py --dry-run     print the plan and check that the packages can be found; runs nothing
The full run refuses to start outside Kaggle (the model does not fit a laptop).
"""
import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "Lightricks/LTX-Video"
REPO_PATTERNS = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"]
COMMUNITY_REPO = "MX69Moi/LTX_Video_i2v_ov"  # FP32 IR, has vae_encoder, no openvino_tokenizer
NIGHTLY_INDEX = "https://storage.openvinotoolkit.org/simple/wheels/nightly"
OPTIMUM_INTEL_PIN = "optimum-intel==2.2.0"  # LTX export: task "text-to-video" -> LTXPipeline, model types
# "ltx-video-transformer" / "ltx-vae-encoder" / "ltx-vae-decoder" / "t5-encoder-model" (read from its source);
# it requires openvino>=2026.0, transformers>=4.51,<5.6, nncf>=3.3.0
BASE_PKGS = [OPTIMUM_INTEL_PIN, "nncf>=3.3.0", "diffusers==0.39.0", "transformers>=4.51,<5.6", "accelerate",
             "safetensors", "sentencepiece", "protobuf", "psutil", "scikit-image", "imageio", "imageio-ffmpeg", "pillow"]
OV_PKGS = ["openvino", "openvino-tokenizers", "openvino-genai"]
COMPONENTS = ["text_encoder", "transformer", "vae_encoder", "vae_decoder"]

WIDTH, HEIGHT, FPS, GUIDANCE = 512, 320, 25, 3.0
FAST = {"frames": 25, "steps": 8}
QUALITY = {"frames": 49, "steps": 20}
SEED = 42
NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"
PROMPT_FAST = ("A red fox walks through a snowy forest at dawn, soft golden light between the pine trees, snow "
               "falling gently, the camera follows the fox from the side, realistic wildlife footage")
PROMPT_LONG = ("A small wooden sailboat glides across a calm turquoise sea toward a rocky island with a white "
               "lighthouse, late afternoon sun, gentle waves, seagulls in the distance, the camera slowly follows the "
               "boat from behind, cinematic, realistic, steady shot")  # quality clip and every window of the long video
MIN_WINDOWS, MAX_WINDOWS = 3, 8
TILED_REPEATS = [1, 2, 4]  # stage E: length multipliers of the long video's latent sequence

BUDGET_S = 9.5 * 3600
RESERVE_S = 20 * 60  # kept for the report and copying the model
KILL_RSS_GB = 28.0
SAMPLE_S = 0.05
MIN_AVAIL_GB = 1.0  # watchdog also fires when the whole system has less than this available
MIN_DISK_GB = 1.0  # ... or when the scratch disk has less than this free
MAX_MODEL_COPY_GB = 12.0
GB = 2**30
T0 = time.time()

WORKER_SRC = r'''
"""s4 worker: one job per process. Writes progress/results as JSON to --out (phase "done" = finished)."""
import argparse, gc, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import psutil

a = None
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


def ov_config():
    cfg = {"INFERENCE_PRECISION_HINT": "f32"}
    if a.threads:
        cfg["INFERENCE_NUM_THREADS"] = a.threads
    return cfg


# ---------------------------------------------------------------- B: getting the model
def job_export_api():
    """Route 1b: the optimum export through its Python API with the pipeline loaded in bf16 (half the RAM).

    With ov_config=None optimum compresses every submodel above 1B parameters to int8_asym, one at a time
    (_apply_model_size_based_quantization); the two VAEs are compressed afterwards by the `compress` job.
    """
    import torch
    from optimum.exporters.openvino import main_export

    save(phase="export")
    main_export(model_name_or_path=a.src, output=a.dst, task="text-to-video", library_name="diffusers",
                convert_tokenizer=True, model_loading_kwargs={"torch_dtype": torch.bfloat16})
    save(phase="done")


def export_config(model, model_type, task, runtime_options=True):
    """The export config optimum-intel 2.2.0 builds in get_ltx_video_models_for_export, for one component."""
    import optimum.exporters.openvino.model_configs  # noqa: F401  (registers the OpenVINO configs)
    from optimum.exporters.tasks import TasksManager

    ctor = TasksManager.get_exporter_config_constructor(model=model, exporter="openvino", library_name="diffusers",
                                                        task=task, model_type=model_type)
    cfg = ctor(model.config, int_dtype="int64", float_dtype="fp32")
    if runtime_options:
        cfg.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    return cfg


def job_export_component():
    """Route 2: one component -> <dst>/<name>/openvino_model.xml + config.json, same calls as optimum's exporter."""
    import copy

    import torch
    from optimum.exporters.openvino.convert import export_models

    src, dst = Path(a.src), Path(a.dst)
    save(phase="load")
    models, is16 = {}, False
    if a.component == "text_encoder":
        from transformers import T5EncoderModel

        te = T5EncoderModel.from_pretrained(src / "text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if te.dtype != torch.bfloat16:
            te = te.to(torch.bfloat16)
        is16 = True
        models["text_encoder"] = (te.eval(), export_config(te, "t5-encoder-model", "feature-extraction"))
    else:
        from diffusers import AutoencoderKLLTXVideo

        vae = AutoencoderKLLTXVideo.from_pretrained(src / "vae", torch_dtype=torch.float32).eval()
        if a.component == "transformer":
            from diffusers import LTXVideoTransformer3DModel

            tr = LTXVideoTransformer3DModel.from_pretrained(src / "transformer", torch_dtype=torch.float32).eval()
            tr.config.vae_temporal_compression_ratio = vae.temporal_compression_ratio
            tr.config.vae_spatial_compression_ratio = vae.spatial_compression_ratio
            del vae
            models["transformer"] = (tr, export_config(tr, "ltx-video-transformer", "semantic-segmentation", False))
        else:  # vae: encoder and decoder
            enc = copy.deepcopy(vae)
            enc.forward = lambda sample: {"latent_parameters": enc.encode(x=sample)["latent_dist"].parameters}
            models["vae_encoder"] = (enc, export_config(enc, "ltx-vae-encoder", "semantic-segmentation"))
            dec = vae
            dec.register_to_config(latents_mean_data=dec.latents_mean.tolist(), latents_std_data=dec.latents_std.tolist())
            dec.forward = lambda latent_sample, timestep=None: dec.decode(z=latent_sample, temb=timestep)
            models["vae_decoder"] = (dec, export_config(dec, "ltx-vae-decoder", "semantic-segmentation"))
    for name, (model, _) in models.items():  # what export_from_model does for diffusers subcomponents
        if hasattr(model, "save_config"):
            model.save_config(dst / name)
        else:
            model.config.save_pretrained(dst / name)
    save(phase="export", loaded_dtype="bf16" if is16 else "fp32", rss_after_load_gb=rss_gb())
    t = time.perf_counter()
    export_models(models_and_export_configs=models, output_dir=dst,
                  output_names=[os.path.join(n, "openvino_model.xml") for n in models], stateful=False,
                  patch_16bit_model=is16, library_name="diffusers")
    save(phase="done", export_s=round(time.perf_counter() - t, 1))


def constant_census(model):
    """Bytes of Constant nodes per element type: shows whether the weights really are int8."""
    out = {}
    for op in model.get_ops():
        if op.get_type_name() == "Constant":
            n = 1
            for d in op.get_output_shape(0):
                n *= d
            et = op.get_output_element_type(0)
            out[et.get_type_name()] = out.get(et.get_type_name(), 0) + n * et.size
    return {k: round(v / 2**30, 3) for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v >= 2**20}


def job_compress():
    """nncf weight compression (INT8_ASYM, what optimum's --weight-format int8 uses) of one IR: --src xml -> --dst xml."""
    import nncf
    import openvino as ov

    save(phase="read")
    model = ov.Core().read_model(a.src)
    before = constant_census(model)
    save(phase="compress", constants_before_gb=before)
    t = time.perf_counter()
    model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    save(phase="save", compress_s=round(time.perf_counter() - t, 1))
    ov.save_model(model, a.dst, compress_to_fp16=False)  # rt_info (runtime_options) travels with the model
    save(phase="done", constants_after_gb=constant_census(model), bin_bytes=os.path.getsize(a.dst[:-4] + ".bin"))


def job_assemble():
    """model_index.json, scheduler/, tokenizer/ (+ openvino_tokenizer.xml) the way optimum's export writes them."""
    import openvino as ov
    from diffusers import FlowMatchEulerDiscreteScheduler
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoTokenizer

    src, dst = Path(a.src), Path(a.dst)
    save(phase="configs")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "model_index.json", dst / "model_index.json")
    FlowMatchEulerDiscreteScheduler.from_pretrained(src / "scheduler").save_pretrained(dst / "scheduler")
    save(phase="tokenizer")
    tok = AutoTokenizer.from_pretrained(src / "tokenizer")
    tok.save_pretrained(dst / "tokenizer")
    try:
        converted = convert_tokenizer(tok, with_detokenizer=True)
    except Exception as e:  # same fallback as optimum's export_tokenizer
        save(phase="tokenizer", detokenizer_error=repr(e)[:300])
        converted = convert_tokenizer(tok, with_detokenizer=False)
    if not isinstance(converted, tuple):
        converted = (converted,)
    for model, name in zip(converted, ("openvino_tokenizer.xml", "openvino_detokenizer.xml")):
        ov.save_model(model, dst / "tokenizer" / name)
    save(phase="done", tokenizer_class=type(tok).__name__, tokenizer_files=sorted(p.name for p in (dst / "tokenizer").iterdir()))


def job_validate():
    """Does the folder load in GenAI on CPU? Text2VideoPipeline is required, Image2VideoPipeline is recorded."""
    import openvino as ov
    import openvino_genai as genai

    save(phase="census")
    census = {}
    for comp in ("text_encoder", "transformer", "vae_encoder", "vae_decoder"):
        xml = Path(a.model) / comp / "openvino_model.xml"
        if xml.exists():
            census[comp] = constant_census(ov.Core().read_model(xml))
    save(phase="load-t2v", constants_gb=census, has_i2v_class=hasattr(genai, "Image2VideoPipeline"))
    t = time.perf_counter()
    pipe = genai.Text2VideoPipeline(a.model, "CPU")
    save(phase="load-i2v", t2v_load_s=round(time.perf_counter() - t, 1), rss_after_t2v_load_gb=rss_gb())
    del pipe
    gc.collect()
    i2v_ok, i2v_error = False, ""
    try:
        t = time.perf_counter()
        pipe = genai.Image2VideoPipeline(a.model, "CPU")
        i2v_ok = True
        R["i2v_load_s"] = round(time.perf_counter() - t, 1)
    except Exception as e:
        i2v_error = repr(e)[:400]
    save(phase="done", i2v_ok=i2v_ok, i2v_error=i2v_error)


# ---------------------------------------------------------------- C / D: generation
def write_mp4(path, frames, fps):
    import imageio.v2 as imageio

    imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)


def job_gen():
    """One GenAI generate call. --mode t2v | i2v (i2v: --image = npy of one [H,W,3] uint8 frame)."""
    import openvino as ov
    import openvino_genai as genai

    save(phase="load")
    t = time.perf_counter()
    cls = genai.Text2VideoPipeline if a.mode == "t2v" else genai.Image2VideoPipeline
    pipe = cls(a.model, "CPU")
    load_s = time.perf_counter() - t
    stamps = []

    def callback(*args):  # (step, num_steps, latent)
        stamps.append(time.perf_counter())
        return False

    kw = dict(negative_prompt=a.neg, height=a.height, width=a.width, num_frames=a.frames,
              num_inference_steps=a.steps, num_videos_per_prompt=1, frame_rate=a.fps, guidance_scale=a.guidance,
              generator=genai.CppStdGenerator(a.seed), callback=callback)
    save(phase="generate", load_s=round(load_s, 2), rss_after_load_gb=rss_gb())
    t0 = time.perf_counter()
    if a.mode == "t2v":
        out = pipe.generate(a.prompt, **kw)
    else:
        cond = np.load(a.image)
        out = pipe.generate(ov.Tensor(np.ascontiguousarray(cond)), a.prompt, **kw)
    generate_s = time.perf_counter() - t0
    video = np.array(out.video.data)  # [B, F, H, W, 3] uint8
    if video.ndim != 5 or video.shape[-1] != 3:
        raise RuntimeError(f"unexpected video tensor shape {video.shape}")
    video = video[0]
    np.save(a.npy, video)
    write_mp4(a.mp4, video, a.fps)
    steps = np.diff(stamps) if len(stamps) > 1 else np.array([])  # callback to callback = one denoising step
    res = dict(generate_s=round(generate_s, 2), frames_out=int(video.shape[0]), video_shape=list(video.shape),
               callbacks=len(stamps),
               first_callback_s=round(stamps[0] - t0, 2) if stamps else None,  # text encoding + first step
               step_s_median=round(float(np.median(steps)), 3) if steps.size else None,
               step_s_mean=round(float(steps.mean()), 3) if steps.size else None,
               after_last_callback_s=round(t0 + generate_s - stamps[-1], 2) if stamps else None)  # ~ VAE decode
    try:  # GenAI's own counters (milliseconds)
        pm = out.perf_metrics
        res["genai_generate_ms"] = round(float(pm.get_generate_duration()), 1)
        res["genai_transformer_step_ms_mean"] = round(float(pm.get_transformer_infer_duration().mean), 1)
        res["genai_vae_decoder_ms"] = round(float(pm.get_vae_decoder_infer_duration()), 1)
    except Exception as e:
        res["genai_perf_metrics_error"] = repr(e)[:200]
    save(phase="done", **res)


# ---------------------------------------------------------------- E: tiled decode at the long length
def job_encode():
    """Frames npy [F,H,W,3] uint8 -> VAE latents (posterior mean) npy [1,C,f,h,w], with the folder's vae_encoder IR.

    --chunk 0: one call on the whole video. --chunk 49: 49-frame pieces that share one frame; the first latent
    frame of every later piece is dropped (it re-encodes the shared frame), so the length matches the one-call result.
    """
    import openvino as ov

    save(phase="compile")
    frames = np.load(a.frames_npy)
    cm = ov.Core().compile_model(str(Path(a.model) / "vae_encoder" / "openvino_model.xml"), "CPU", ov_config())
    req = cm.create_infer_request()
    save(phase="encode", video_frames=int(frames.shape[0]))

    def enc(fr):
        x = (fr.astype(np.float32) / 127.5 - 1.0).transpose(3, 0, 1, 2)[None]
        req.infer({0: np.ascontiguousarray(x)})
        return req.get_output_tensor(0).data.copy()

    t = time.perf_counter()
    if a.chunk:
        step = a.chunk - 1
        parts = [enc(frames[i : i + a.chunk]) for i in range(0, len(frames) - 1, step)]
        lat = np.concatenate([parts[0], *(p[:, :, 1:] for p in parts[1:])], axis=2)
    else:
        lat = enc(frames)
    encode_s = time.perf_counter() - t
    if lat.shape[1] % 2 == 0 and lat.shape[1] > a.latent_channels:  # mean + logvar -> mean
        lat = lat[:, : lat.shape[1] // 2]
    np.save(a.npy, lat.astype(np.float32))
    save(phase="done", encode_s=round(encode_s, 2), latent_shape=list(lat.shape), chunk=a.chunk)


def job_decode():
    """Decode a latent sequence with our TiledDecoder: --tiling none | temporal. --repeat k: k times the length."""
    sys.path.insert(0, a.pkg)
    from tiled_decode import OVDecoder, TiledDecoder, TilingConfig

    save(phase="compile")
    lat = np.load(a.lat)
    if a.repeat > 1:  # ping-pong in time: real latents, continuous motion, memory depends only on the shape
        n = lat.shape[2]
        cycle = list(range(n)) + list(range(n - 2, 0, -1))
        lat = np.ascontiguousarray(lat[:, :, [cycle[i % len(cycle)] for i in range((n - 1) * a.repeat + 1)]])
    calls = [0]
    ovd = OVDecoder(str(Path(a.model) / "vae_decoder" / "openvino_model.xml"), a.threads)

    def decoder(z):
        calls[0] += 1
        return ovd(z)

    save(phase="decode", latent_shape=list(lat.shape), rss_after_load_gb=rss_gb())
    t = time.perf_counter()
    video = TiledDecoder(decoder, TilingConfig()).decode(lat, a.tiling)
    decode_s = time.perf_counter() - t
    res = dict(decode_s=round(decode_s, 2), decoder_calls=calls[0], output_shape=list(video.shape))
    if a.npy:
        save(phase="write", **res)
        np.save(a.npy, video)
        u8 = (np.clip(video[0].transpose(1, 2, 3, 0) / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)
        write_mp4(a.mp4, u8, a.fps)
    save(phase="done", **res)


JOBS = {"export_api": job_export_api, "export_component": job_export_component, "compress": job_compress,
        "assemble": job_assemble, "validate": job_validate, "gen": job_gen, "encode": job_encode, "decode": job_decode}


def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=sorted(JOBS))
    ap.add_argument("--out")
    for name in ("src", "dst", "model", "component", "mode", "prompt", "neg", "image", "npy", "mp4", "frames-npy",
                 "lat", "pkg"):
        ap.add_argument("--" + name)
    for name in ("width", "height", "frames", "steps", "seed", "fps", "threads", "chunk"):
        ap.add_argument("--" + name, type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--latent-channels", type=int, default=128)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--tiling", default="none")
    a = ap.parse_args()
    try:  # Linux: make this process the kernel OOM killer's first choice, not the parent
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    JOBS[a.job]()


# The guard matters: helper processes spawned during conversion may re-import this file.
if __name__ == "__main__":
    main()
'''


def log(msg):
    el = int(time.time() - T0)
    print(f"[s4 {time.strftime('%H:%M:%S')} +{el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d}] {msg}", flush=True)


def remaining():
    return BUDGET_S - (time.time() - T0)


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


def free_gb(path):
    return shutil.disk_usage(path).free / GB


def dir_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def run_child(psutil, cmd, out_json, log_path, timeout_s, kill_rss, env_extra, disk_path):
    """Run one job in a fresh process, sampling the RSS of its process tree every 50 ms. Returns the result fields.

    out_json=None: a plain command (optimum-cli); exit code 0 = ok. Otherwise the worker's JSON must reach phase "done".
    """
    if out_json:
        Path(out_json).unlink(missing_ok=True)
        cmd = [*cmd, "--out", str(out_json)]
    env = {**os.environ, "USE_TF": "0", "USE_FLAX": "0", "TOKENIZERS_PARALLELISM": "false", **env_extra}
    swap0 = psutil.swap_memory().used
    peak = peak_swap = rss = 0
    min_avail = psutil.virtual_memory().available
    killed = None
    t = time.perf_counter()
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(list(map(str, cmd)), stdout=lf, stderr=subprocess.STDOUT, env=env)
        ps = psutil.Process(proc.pid)
        n = 0
        while proc.poll() is None:
            tree = [ps]
            try:
                tree += ps.children(recursive=True)
            except psutil.Error:
                pass
            rss = 0
            for p in tree:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss, vm_hwm(proc.pid))
            peak_swap = max(peak_swap, psutil.swap_memory().used - swap0)
            avail = psutil.virtual_memory().available
            min_avail = min(min_avail, avail)
            n += 1
            if rss > kill_rss or avail < MIN_AVAIL_GB * GB:
                killed = "oom"
            elif time.perf_counter() - t > timeout_s:
                killed = "timeout"
            elif n % 100 == 0 and free_gb(disk_path) < MIN_DISK_GB:
                killed = "disk_full"
            if killed:
                for p in reversed(tree):
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                proc.wait()
                break
            time.sleep(SAMPLE_S)
    wall = time.perf_counter() - t
    res = {}
    if out_json:
        try:
            res = json.loads(Path(out_json).read_text())
        except (OSError, ValueError):
            pass
    tail = Path(log_path).read_text(errors="replace").strip().splitlines()[-15:]
    phase = res.get("phase", "start") if out_json else "run"
    finished = proc.returncode == 0 and (phase == "done" or not out_json)
    if killed == "oom":
        status, detail = "oom", (f"watchdog kill during {phase}: tree RSS {rss / GB:.1f} GB (limit {kill_rss / GB:.1f} GB), "
                                 f"system available {min_avail / GB:.1f} GB")
    elif killed == "timeout":
        status, detail = "timeout", f"killed after {timeout_s:.0f} s during {phase}"
    elif killed == "disk_full":
        status, detail = "disk_full", f"scratch disk below {MIN_DISK_GB} GB free during {phase}"
    elif finished:
        status, detail = "ok", ""
    elif proc.returncode == -9:
        status, detail = "oom", f"SIGKILL during {phase} (kernel OOM killer)"
    elif any(k in line for line in tail for k in ("bad_alloc", "MemoryError", "allocate", "not enough memory")):
        status, detail = "oom", f"allocation failure during {phase}: {tail[-1][:200]}"
    elif any("No space left" in line for line in tail):
        status, detail = "disk_full", f"no space left during {phase}"
    else:
        status, detail = "error", f"exit code {proc.returncode} during {phase}: {(tail or ['no output'])[-1][:300]}"
    res.pop("phase", None)
    return {"status": status, "detail": detail, "peak_rss_gb": round(peak / GB, 3), "wall_s": round(wall, 2),
            "peak_swap_delta_gb": round(peak_swap / GB, 3), "min_sys_avail_gb": round(min_avail / GB, 2), **res}


def save_results(out, meta, rows):
    (out / "s4_results.json").write_text(json.dumps({**meta, "jobs": rows}, indent=2, default=str))
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(out / "s4_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(r[k]) if isinstance(r.get(k), (dict, list)) else r.get(k, "") for k in cols])
    try:
        write_report(out, meta, rows)
    except Exception as e:  # the report must never stop the run
        log(f"report not written: {e!r}")


# ---------------------------------------------------------------------------------------------- video metrics
def frame_diffs(video):
    """Mean |frame[i+1] - frame[i]| per transition, on [0,1] values. video: uint8 [F,H,W,3]."""
    import numpy as np

    return np.array([np.mean(np.abs(video[i + 1].astype(np.int16) - video[i])) / 255.0 for i in range(len(video) - 1)])


def small_gray(frame, size=(256, 160)):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.fromarray(frame).convert("L").resize(size, Image.LANCZOS), dtype=np.float32) / 255.0


def window_stats(win):
    """Per-window appearance numbers on [0,1] values: mean RGB, brightness (luma), sharpness (mean |gradient|)."""
    import numpy as np

    v = win.astype(np.float32) / 255.0
    luma = v @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sharp = float(np.mean(np.abs(np.diff(luma, axis=1))) + np.mean(np.abs(np.diff(luma, axis=2)))) / 2
    return {"mean_rgb": [round(float(c), 4) for c in v.mean(axis=(0, 1, 2))], "brightness": round(float(luma.mean()), 4),
            "sharpness": round(sharp, 5)}


def long_video_metrics(windows, conds):
    """windows: list of uint8 [F,H,W,3] as generated (windows 2.. start with their copy of the conditioning frame).

    SEAM   transition from the last frame of window k to the first NEW frame of window k+1, divided by the median
           frame-to-frame change inside the windows. 1.0 = the boundary moves like any other frame step.
    DRIFT  per window vs window 1: brightness and mean-RGB shift, sharpness ratio, SSIM of the middle frames
           (grayscale, 256x160). Content legitimately changes over time, so SSIM falling is expected; the colour,
           brightness and sharpness trends are the drift signal.
    """
    import numpy as np
    from skimage.metrics import structural_similarity

    inside = np.concatenate([frame_diffs(w if k == 0 else w[1:]) for k, w in enumerate(windows)])
    median_inside = float(np.median(inside))
    seams = []
    for k in range(1, len(windows)):
        prev_last, w = windows[k - 1][-1], windows[k]
        jump = float(np.mean(np.abs(w[1].astype(np.int16) - prev_last)) / 255.0)
        seams.append({
            "boundary": f"{k}->{k + 1}", "jump": round(jump, 5),
            "seam_score": round(jump / median_inside, 3) if median_inside else None,
            # how faithfully Image2Video reproduces its conditioning frame as its own frame 0
            "cond_vs_frame0": round(float(np.mean(np.abs(w[0].astype(np.int16) - conds[k])) / 255.0), 5),
            "frame0_to_frame1": round(float(np.mean(np.abs(w[1].astype(np.int16) - w[0])) / 255.0), 5)})
    stats = [window_stats(w) for w in windows]
    ref_mid = small_gray(windows[0][len(windows[0]) // 2])
    drift = []
    for k, (w, s) in enumerate(zip(windows, stats)):
        drift.append({
            "window": k + 1, **s,
            "brightness_shift": round(s["brightness"] - stats[0]["brightness"], 4),
            "rgb_shift": round(float(np.mean(np.abs(np.array(s["mean_rgb"]) - np.array(stats[0]["mean_rgb"])))), 4),
            "sharpness_ratio": round(s["sharpness"] / stats[0]["sharpness"], 3) if stats[0]["sharpness"] else None,
            "ssim_mid_vs_window1": round(float(structural_similarity(ref_mid, small_gray(w[len(w) // 2]), data_range=1.0)), 4),
            "median_frame_diff": round(float(np.median(frame_diffs(w))), 5)})
    return {"median_frame_diff_inside_windows": round(median_inside, 5), "seams": seams, "drift": drift}


def contact_sheet(path, clips, cols=6, labels=None):
    """One row per clip, `cols` evenly spaced frames per row."""
    import numpy as np
    from PIL import Image, ImageDraw

    h, w = clips[0].shape[1:3]
    sheet = Image.new("RGB", (cols * w, len(clips) * h), "black")
    draw = ImageDraw.Draw(sheet)
    for r, clip in enumerate(clips):
        for c, i in enumerate(np.linspace(0, len(clip) - 1, cols).round().astype(int)):
            sheet.paste(Image.fromarray(clip[i]), (c * w, r * h))
            draw.text((c * w + 6, r * h + 4), f"{labels[r] if labels else ''} f{i}", fill="white")
    sheet.save(path)


def psnr_frames(ref_path, test_path):
    """PSNR of two raw decoder outputs [1,3,F,H,W] (displayed values, clipped), frame by frame via memmap."""
    import numpy as np

    ref, test = np.load(ref_path, mmap_mode="r"), np.load(test_path, mmap_mode="r")
    if ref.shape != test.shape:
        return {"error": f"shape mismatch {ref.shape} vs {test.shape}"}
    mse = np.array([np.mean((np.clip(np.asarray(ref[0, :, i], np.float64) / 2 + 0.5, 0, 1)
                             - np.clip(np.asarray(test[0, :, i], np.float64) / 2 + 0.5, 0, 1)) ** 2)
                    for i in range(ref.shape[2])])
    db = lambda m: None if m == 0 else round(float(10 * np.log10(1.0 / m)), 2)  # noqa: E731  (None = identical)
    return {"psnr_db": db(float(mse.mean())), "psnr_worst_frame_db": db(float(mse.max())), "frames": int(len(mse))}


# ---------------------------------------------------------------------------------------------- report
def table(rows, cols):
    def cell(r, c):
        v = r.get(c, "")
        return json.dumps(v) if isinstance(v, (dict, list)) else "" if v is None else str(v)

    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
                      *("| " + " | ".join(cell(r, c) for c in cols) + " |" for r in rows)])


def write_report(out, meta, rows):
    by = lambda s: [r for r in rows if r["stage"] == s]  # noqa: E731
    el = time.time() - T0
    L = [f"# Step 4 report: full LTX-Video on OpenVINO INT8 + GenAI, long video from chained windows",
         "",
         f"Generated by `s4_full_pipeline.py`, run started {meta['started']}; elapsed when this was written "
         f"{el / 3600:.2f} h of a {BUDGET_S / 3600:.1f} h budget. Every number below was measured in this run "
         f"(raw: `s4_results.json`, `s4_results.csv`). Stages finished so far: {', '.join(meta['stages_done']) or 'none'}.",
         "",
         "## Setup",
         f"- Machine: {meta['cpu_model']}, {meta['nproc']} logical CPUs, RAM limit {meta['mem_limit_gb']} GB. Every job is a "
         f"fresh subprocess; the parent samples the RSS of its process tree every {int(SAMPLE_S * 1000)} ms and kills it "
         f"above {meta['kill_rss_gb']} GB (`oom`).",
         "- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()) + ".",
         f"- Install: {meta.get('install', {})}",
         f"- Scratch dir `{meta['scratch']}`; free disk at start: {meta['disk_free_start_gb']}.",
         ""]
    b = by("B")
    if b:
        L += ["## B. Getting the INT8 model",
              f"Route that produced the model used below: **{meta.get('model_route') or 'none (all routes failed)'}**.",
              "", "Attempts (peak RSS = max over the attempt's jobs, time = sum):", "",
              table(meta.get("attempts", []), ["route", "status", "peak_rss_gb", "wall_s", "detail"]),
              "", "Jobs:", "",
              table(b, ["name", "status", "peak_rss_gb", "wall_s", "detail"]), ""]
        if meta.get("downloads"):
            L += ["Downloads: " + "; ".join(f"`{d['repo']}` {d['gb']} GB in {d['seconds']} s" for d in meta["downloads"]), ""]
        if meta.get("model_manifest"):
            m = meta["model_manifest"]
            L += [f"Final folder: {m['total_gb']} GB. " + ", ".join(f"`{k}` {v} GB" for k, v in m["component_bin_gb"].items()) + ".",
                  f"Weight constants per component (GB by element type): `{json.dumps(meta.get('constants_gb', {}))}`.",
                  f"GenAI load check: Text2VideoPipeline {meta.get('t2v_load', 'n/a')}; Image2VideoPipeline "
                  f"{meta.get('i2v_load', 'n/a')}. Folder copied to the output: {m.get('copied')}.", ""]
    c = by("C") + by("D")
    if c:
        L += ["## C / D. Generation runs (GenAI, CPU)",
              f"{WIDTH}x{HEIGHT}, guidance {GUIDANCE}, {FPS} fps, negative prompt `{NEGATIVE}`. `step_s_median` = median time "
              "between two step callbacks; `first_callback_s` includes text encoding; `after_last_callback_s` is mostly "
              "the VAE decode.", "",
              table(c, ["name", "mode", "frames", "steps", "seed", "status", "peak_rss_gb", "rss_after_load_gb", "load_s",
                        "generate_s", "step_s_median", "first_callback_s", "after_last_callback_s", "wall_s", "detail"]),
              "", f"- fast prompt: {PROMPT_FAST}", f"- quality / long-video prompt: {PROMPT_LONG}", ""]
    lv = meta.get("long_video")
    if lv:
        L += ["## D. Long video from chained windows",
              f"{lv['windows']} windows of {lv['frames_per_window']} frames, {lv['steps']} steps each; window 1 is Text2Video, "
              "every later window is Image2Video conditioned on the last frame of the previous one (same prompt, seed + "
              "window index). The first frame of windows 2.. (the copy of the conditioning frame) is dropped.",
              f"Result: **{lv['total_frames']} frames = {lv['seconds']} s at {FPS} fps**, generated in {lv['generate_wall_s']} s "
              f"of process time (`{lv['mp4']}`, contact sheet `{lv['contact_sheet']}`). Peak RSS over the windows: "
              f"{lv['peak_rss_gb']} GB.",
              "" if lv["windows"] >= MIN_WINDOWS else f"**Fewer than {MIN_WINDOWS} windows: {lv.get('stopped', '')}**", "",
              f"Seams. Median frame-to-frame change inside windows: {lv['median_frame_diff_inside_windows']}. "
              "`seam_score` = jump at the boundary / that median (1.0 = like any other frame step).", "",
              table(lv["seams"], ["boundary", "jump", "seam_score", "cond_vs_frame0", "frame0_to_frame1"]), "",
              "Drift vs window 1 (values in [0,1]; SSIM of the middle frames, grayscale 256x160):", "",
              table(lv["drift"], ["window", "brightness", "brightness_shift", "rgb_shift", "sharpness", "sharpness_ratio",
                                  "ssim_mid_vs_window1", "median_frame_diff"]), ""]
    e = by("E")
    if e:
        L += ["## E. Decoding the long video's latents: untiled vs temporal tiling (our `tiled_decode.py`)",
              "Latents = the generated long video re-encoded with the folder's VAE encoder IR (GenAI does not hand out the "
              "final latents in a documented way); `repeat` k = the same latents extended k times in time by ping-pong. "
              "Decoder = the folder's `vae_decoder` IR (INT8 weights), f32 inference.", "",
              table(e, ["name", "tiling", "repeat", "latent_shape", "output_shape", "status", "peak_rss_gb", "decode_s",
                        "decoder_calls", "wall_s", "detail"]), ""]
        if meta.get("tiled_vs_untiled"):
            L += [f"Temporal-tiled vs untiled decode of the same latents (repeat 1): `{json.dumps(meta['tiled_vs_untiled'])}`.", ""]
    L += ["## Notes", *(f"- {n}" for n in meta.get("notes", [])), "",
          "## Files", *(f"- `{p.name}` ({p.stat().st_size / 2**20:.1f} MB)" for p in sorted(out.iterdir()) if p.is_file())]
    (out / "s4_report.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------- main
def plan_text():
    return "\n".join([
        "LongVid-OV step 4 plan",
        f"  A. install {OPTIMUM_INTEL_PIN}, nncf, diffusers==0.39.0, then --pre {' '.join(OV_PKGS)} from {NIGHTLY_INDEX}",
        f"  B. INT8 model from {REPO}: 1a optimum-cli -> 1b optimum API bf16 -> 2 component-wise -> 3 {COMMUNITY_REPO}; "
        f"watchdog kill at {KILL_RSS_GB} GB",
        f"  C. Text2Video {WIDTH}x{HEIGHT}: {FAST['frames']} frames / {FAST['steps']} steps, then {QUALITY['frames']} frames / "
        f"{QUALITY['steps']} steps (seed {SEED})",
        f"  D. long video: {MIN_WINDOWS}..{MAX_WINDOWS} chained windows (T2V, then I2V on the last frame), seams + drift",
        f"  E. tiled vs untiled decode of the long latents, length x{TILED_REPEATS}",
        f"  F. report, manifest, model folder if < {MAX_MODEL_COPY_GB} GB",
        f"  budget {BUDGET_S / 3600:.1f} h, {RESERVE_S // 60} min reserved for F"])


def dry_run():
    print(plan_text())
    mods = ["torch", "openvino", "openvino_genai", "openvino_tokenizers", "optimum", "nncf", "diffusers", "transformers",
            "huggingface_hub", "psutil", "skimage", "imageio", "PIL", "numpy"]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    for m in mods:
        print(f"  {'MISSING' if m in missing else 'found  '} {m}")
    for p in ("openvino", "openvino-genai", "openvino-tokenizers", "optimum-intel", "nncf", "diffusers", "transformers", "torch"):
        print(f"  {p} {version(p)}")
    print("dry run: nothing was installed, downloaded or executed." + (f" Missing: {missing}" if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan, check that packages can be found, run nothing")
    ap.add_argument("--out", help="output folder (default /kaggle/working)")
    a, _ = ap.parse_known_args()  # Kaggle runs scripts through a kernel launcher with its own argv
    if a.dry_run:
        return dry_run()
    if not Path("/kaggle/working").is_dir():
        sys.exit("The full pipeline only runs on Kaggle (needs ~30 GB RAM). Use --dry-run elsewhere.")

    out = Path(a.out) if a.out else Path("/kaggle/working")
    cands = [p for p in (Path("/kaggle/tmp"), Path("/kaggle/temp"), Path("/tmp")) if p.is_dir()] or [Path("/tmp")]
    scratch = max(cands, key=free_gb) / "s4_work"
    logs = out / "logs"
    for d in (out, logs, scratch, scratch / "tmp", scratch / "hf"):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(scratch / "hf")
    env_extra = {"HF_HOME": str(scratch / "hf"), "TMPDIR": str(scratch / "tmp")}  # optimum-cli exports FP32 to TMPDIR first
    print(plan_text(), flush=True)

    # A. environment --------------------------------------------------------------------------------------
    install = {"base": pip(*BASE_PKGS)}
    if not install["base"]:
        log("pinned install failed, retrying with optimum-intel unpinned")
        install["base_unpinned"] = pip("optimum-intel", *BASE_PKGS[1:])
    if importlib.util.find_spec("torch") is None:  # Kaggle ships torch; only fetch the CPU wheel if it is missing
        install["torch"] = pip("torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")
    install["openvino_nightly"] = pip("--pre", "-U", *OV_PKGS, "--extra-index-url", NIGHTLY_INDEX)
    if not install["openvino_nightly"]:
        log("nightly OpenVINO not installable, falling back to the latest stable (Image2VideoPipeline may be missing)")
        install["openvino_stable"] = pip("-U", *OV_PKGS)
    import psutil

    limit = mem_limit_bytes(psutil)
    kill_rss = min(KILL_RSS_GB * GB, limit - 1.5 * GB)
    lscpu = sh(["lscpu"])
    cpu_model = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")),
                     platform.processor() or platform.machine())
    meta = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu_model": cpu_model, "nproc": os.cpu_count(),
        "mem_limit_gb": round(limit / GB, 2), "kill_rss_gb": round(kill_rss / GB, 2), "budget_s": BUDGET_S,
        "scratch": str(scratch), "install": install, "stages_done": [], "attempts": [], "downloads": [], "notes": [],
        "disk_free_start_gb": {str(p): round(free_gb(p), 1) for p in {out, scratch}},
        "settings": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "guidance": GUIDANCE, "fast": FAST, "quality": QUALITY,
                     "seed": SEED, "negative": NEGATIVE, "prompt_fast": PROMPT_FAST, "prompt_long": PROMPT_LONG},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in (
            "openvino", "openvino-genai", "openvino-tokenizers", "optimum-intel", "optimum", "nncf", "diffusers",
            "transformers", "torch", "huggingface_hub", "numpy", "scikit-image", "psutil")}},
    }
    env_txt = "\n".join([
        f"date: {meta['started']}", f"platform: {platform.platform()}", f"python: {sys.version}",
        *(f"{k}: {v}" for k, v in meta["versions"].items() if k != "python"), f"install steps ok: {install}",
        f"cpus: {os.cpu_count()}", f"RAM limit (min of RAM and cgroup): {meta['mem_limit_gb']} GB; available now: "
        f"{psutil.virtual_memory().available / GB:.2f} GB; swap: {psutil.swap_memory().total / GB:.2f} GB; "
        f"watchdog kill RSS: {meta['kill_rss_gb']} GB",
        f"scratch: {scratch}", "", "--- lscpu ---", lscpu, "", "--- free -h ---", sh(["free", "-h"]),
        "", "--- df -h ---", sh(["df", "-h"]), "", "--- pip freeze (selected) ---",
        "\n".join(ln for ln in sh([sys.executable, "-m", "pip", "freeze"]).splitlines()
                  if any(k in ln.lower() for k in ("openvino", "optimum", "nncf", "diffusers", "transformers", "torch",
                                                   "tokenizers", "huggingface", "numpy", "safetensors", "accelerate"))), ""])
    (out / "s4_env.txt").write_text(env_txt)
    print(env_txt, flush=True)

    worker = scratch / "s4_worker.py"
    worker.write_text(WORKER_SRC)
    here = Path(globals().get("__file__", "s4_full_pipeline.py")).resolve().parent
    pkg = here.parent.parent / "src" / "longvid_ov"  # the repo's module if the repo is there, else the embedded copy
    if not (pkg / "tiled_decode.py").exists():
        pkg = scratch / "longvid_ov"
        pkg.mkdir(exist_ok=True)
        (pkg / "tiled_decode.py").write_text(TILED_SRC)
    rows = []
    meta["stages_done"].append("A")
    save_results(out, meta, rows)

    def job(stage, name, args, timeout_s, cmd=None, **extra):
        """Run a worker job (or a plain command), record it as a row, save everything."""
        timeout_s = max(60, min(timeout_s, remaining() - RESERVE_S))
        log(f"[{stage}] {name} ... (timeout {timeout_s / 60:.0f} min, scratch free {free_gb(scratch):.1f} GB)")
        r = run_child(psutil, cmd or [sys.executable, worker, *args], None if cmd else scratch / f"{name}.json",
                      logs / f"{name}.log", timeout_s, kill_rss, env_extra, scratch)
        row = {"stage": stage, "name": name, **extra, **r, "finished_at": time.strftime("%H:%M:%S")}
        rows.append(row)
        log(f"[{stage}] {name}: {r['status']} {r['detail']} | peak RSS {r['peak_rss_gb']} GB, {r['wall_s']} s")
        save_results(out, meta, rows)
        return row

    def download(repo, patterns, dest):
        from huggingface_hub import snapshot_download

        if (dest / "model_index.json").exists() and meta.get("downloaded_" + repo):
            return dest
        log(f"downloading {repo} {patterns or ''} -> {dest} (scratch free {free_gb(scratch):.1f} GB)")
        t = time.perf_counter()
        snapshot_download(repo, allow_patterns=patterns, local_dir=dest)
        meta["downloads"].append({"repo": repo, "gb": round(dir_bytes(dest) / GB, 2), "seconds": round(time.perf_counter() - t)})
        meta["downloaded_" + repo] = True
        return dest

    class RouteFailed(Exception):
        pass

    def need(row):
        if row["status"] != "ok":
            raise RouteFailed(f"{row['name']}: {row['status']} {row['detail']}")
        return row

    def copy_small_files(src_dir, dst_dir):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in Path(src_dir).iterdir():
            if p.is_file() and not p.name.startswith("openvino_model"):
                shutil.copy(p, dst_dir / p.name)

    def compress_into(tag, src_root, dst_root, comps, delete_src=False):
        for comp in comps:
            copy_small_files(src_root / comp, dst_root / comp)
            need(job("B", f"{tag}_compress_{comp}", ["compress", "--src", src_root / comp / "openvino_model.xml",
                                                      "--dst", dst_root / comp / "openvino_model.xml"], 90 * 60))
            if delete_src:  # free the disk as we go
                shutil.rmtree(src_root / comp, ignore_errors=True)

    pt_dir, ov_root = scratch / "ltx_pt", scratch / "ov"

    def route_1a():
        dst = ov_root / "r1a_int8"
        need(job("B", "r1a_optimum_cli", None, 120 * 60, cmd=[
            sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino", "--model", download(REPO, REPO_PATTERNS, pt_dir),
            "--task", "text-to-video", "--weight-format", "int8", dst]))
        return dst

    def route_1b():
        dst = ov_root / "r1b_int8"
        need(job("B", "r1b_optimum_api_bf16", ["export_api", "--src", download(REPO, REPO_PATTERNS, pt_dir), "--dst", dst], 120 * 60))
        for comp in ("vae_encoder", "vae_decoder"):  # below optimum's 1B-parameter auto-compression threshold
            tmp = scratch / "r1b_vae" / comp
            need(job("B", f"r1b_compress_{comp}", ["compress", "--src", dst / comp / "openvino_model.xml",
                                                   "--dst", tmp / "openvino_model.xml"], 30 * 60))
            for ext in (".xml", ".bin"):
                shutil.move(tmp / f"openvino_model{ext}", dst / comp / f"openvino_model{ext}")
        return dst

    def route_2():
        src, fp, dst = download(REPO, REPO_PATTERNS, pt_dir), ov_root / "r2_fp", ov_root / "r2_int8"
        for comp, comps in (("vae", ["vae_encoder", "vae_decoder"]), ("transformer", ["transformer"]),
                            ("text_encoder", ["text_encoder"])):  # cheapest first: a broken export shows up early
            need(job("B", f"r2_export_{comp}", ["export_component", "--component", comp, "--src", src, "--dst", fp], 90 * 60))
            compress_into("r2", fp, dst, comps, delete_src=True)
        need(job("B", "r2_assemble", ["assemble", "--src", src, "--dst", dst], 20 * 60))
        return dst

    def route_3():
        for p in pt_dir.rglob("*.safetensors"):  # route 3 does not need the PyTorch weights; make room for 28 GB of IR
            p.unlink()
        src, dst = download(COMMUNITY_REPO, None, scratch / "community_fp32"), ov_root / "r3_int8"
        compress_into("r3", src, dst, COMPONENTS, delete_src=True)
        row = job("B", "r3_assemble", ["assemble", "--src", src, "--dst", dst], 20 * 60)
        if row["status"] != "ok" and (pt_dir / "tokenizer").is_dir():  # community tokenizer unusable: the original one
            log("retrying assemble with the tokenizer / scheduler / model_index of " + REPO)
            row = job("B", "r3_assemble_original_tokenizer", ["assemble", "--src", pt_dir, "--dst", dst], 20 * 60)
        need(row)
        return dst

    def validate(tag, folder):
        row = need(job("B", f"{tag}_validate", ["validate", "--model", folder], 30 * 60))
        if not row.get("i2v_ok") and row.get("has_i2v_class"):
            # same files, model_index.json naming the image-to-video pipeline class (what optimum writes for that task)
            alt = folder.parent / (folder.name + "_i2v_index")
            alt.mkdir(exist_ok=True)
            for p in folder.iterdir():
                if p.name != "model_index.json" and not (alt / p.name).exists():
                    os.symlink(p, alt / p.name)
            idx = json.loads((folder / "model_index.json").read_text())
            idx["_class_name"] = "LTXImageToVideoPipeline"
            (alt / "model_index.json").write_text(json.dumps(idx, indent=2))
            row2 = job("B", f"{tag}_validate_i2v_index", ["validate", "--model", alt], 30 * 60)
            if row2["status"] == "ok" and row2.get("i2v_ok"):
                meta["notes"].append("Image2VideoPipeline only loaded with model_index.json `_class_name` = "
                                     f"LTXImageToVideoPipeline (first error: {row.get('i2v_error')}); used a symlinked copy for I2V.")
                return row, alt
        return row, folder

    model_dir = i2v_dir = None
    try:
        # B. the model ------------------------------------------------------------------------------------
        for tag, route, label in (("r1a", route_1a, "1a optimum-cli export (fp32 load) --weight-format int8"),
                                  ("r1b", route_1b, "1b optimum main_export, bf16 load, auto int8 + VAE compress"),
                                  ("r2", route_2, "2 component-wise export + nncf INT8 + assembled layout"),
                                  ("r3", route_3, f"3 community FP32 IR {COMMUNITY_REPO} + nncf INT8")):
            if remaining() < 3 * 3600:
                meta["notes"].append(f"Route {label} not started: less than 3 h of budget left.")
                break
            n0, status, detail = len(rows), "ok", ""
            try:
                folder = route()
                vrow, i2v_folder = validate(tag, folder)
            except RouteFailed as e:
                status, detail = "failed", str(e)[:400]
            except Exception as e:  # download errors etc.
                status, detail = "failed", repr(e)[:400]
            mine = rows[n0:]
            meta["attempts"].append({"route": label, "status": status, "detail": detail,
                                     "peak_rss_gb": max([r["peak_rss_gb"] for r in mine], default=0),
                                     "wall_s": round(sum(r["wall_s"] for r in mine), 1)})
            if status == "ok":
                model_dir, i2v_dir = folder, i2v_folder
                meta.update(model_route=label, constants_gb=vrow.get("constants_gb"),
                            t2v_load=f"ok in {vrow.get('t2v_load_s')} s (RSS after load {vrow.get('rss_after_t2v_load_gb')} GB)",
                            i2v_load="ok" if i2v_folder != folder or vrow.get("i2v_ok") else f"FAILED: {vrow.get('i2v_error')}",
                            i2v_ok=bool(i2v_folder != folder or vrow.get("i2v_ok")))
                break
            log(f"route {label} failed: {detail}")
            for d in (ov_root, scratch / "tmp"):  # leftovers of the failed attempt
                shutil.rmtree(d, ignore_errors=True)
            (scratch / "tmp").mkdir(exist_ok=True)
            save_results(out, meta, rows)
        if model_dir is None:
            raise RuntimeError("no route produced a model that loads in GenAI; see the attempts table")
        shutil.rmtree(pt_dir, ignore_errors=True)
        shutil.rmtree(scratch / "hf", ignore_errors=True)
        files = {str(p.relative_to(model_dir)): p.stat().st_size for p in sorted(model_dir.rglob("*")) if p.is_file()}
        meta["model_manifest"] = {
            "total_gb": round(sum(files.values()) / GB, 2), "copied": False,
            "component_bin_gb": {c: round(files.get(f"{c}/openvino_model.bin", 0) / GB, 2) for c in COMPONENTS}, "files": files}
        (out / "s4_model_manifest.json").write_text(json.dumps(meta["model_manifest"], indent=2))
        meta["stages_done"].append("B")
        save_results(out, meta, rows)

        # C. generation runs ------------------------------------------------------------------------------
        import numpy as np

        def gen(stage, name, mode, cfg, prompt, seed, timeout_s, image=None):
            args = ["gen", "--model", i2v_dir if mode == "i2v" else model_dir, "--mode", mode, "--prompt", prompt,
                    "--neg", NEGATIVE, "--width", WIDTH, "--height", HEIGHT, "--frames", cfg["frames"], "--steps", cfg["steps"],
                    "--seed", seed, "--fps", FPS, "--guidance", GUIDANCE, "--npy", scratch / f"{name}.npy",
                    "--mp4", out / f"s4_{name}.mp4"]
            if image is not None:
                np.save(scratch / f"{name}_cond.npy", image)
                args += ["--image", scratch / f"{name}_cond.npy"]
            return job(stage, name, args, timeout_s, mode=mode, frames=cfg["frames"], steps=cfg["steps"], seed=seed)

        fast = gen("C", "t2v_fast", "t2v", FAST, PROMPT_FAST, SEED, 60 * 60)
        if fast["status"] == "ok":
            contact_sheet(out / "s4_t2v_fast_contact.png", [np.load(scratch / "t2v_fast.npy")], labels=["fast"])
        qual = gen("C", "t2v_quality", "t2v", QUALITY, PROMPT_LONG, SEED, 2 * 3600)
        if qual["status"] == "ok":
            contact_sheet(out / "s4_t2v_quality_contact.png", [np.load(scratch / "t2v_quality.npy")], labels=["quality"])
        meta["stages_done"].append("C")
        save_results(out, meta, rows)

        # D. long video -----------------------------------------------------------------------------------
        # Window settings: the quality settings if that clip worked, else the fast ones. Window 1 of the chain is the
        # quality clip itself when it exists (same prompt, same seed), so it is not generated twice.
        wcfg, first = (QUALITY, qual) if qual["status"] == "ok" else (FAST, None)
        if first is None and fast["status"] != "ok":
            raise RuntimeError("both Text2Video runs failed; no long video")
        windows, conds, wrows, stopped = [], [None], [], ""
        if first is not None:
            windows.append(np.load(scratch / "t2v_quality.npy"))
            wrows.append(first)
        else:
            r = gen("D", "window_01", "t2v", wcfg, PROMPT_LONG, SEED, 2 * 3600)
            if r["status"] == "ok":
                windows.append(np.load(scratch / "window_01.npy"))
                wrows.append(r)
            else:
                stopped = f"window 1 {r['status']}: {r['detail']}"
        if windows and not meta.get("i2v_ok"):
            stopped = f"Image2VideoPipeline did not load: {meta.get('i2v_load')}"
        while windows and not stopped and len(windows) < MAX_WINDOWS:
            k = len(windows) + 1
            est = max(r["wall_s"] for r in wrows) * 1.15  # measured in this run
            if remaining() - RESERVE_S < est:
                stopped = f"budget: {remaining() / 60:.0f} min left, a window takes about {est / 60:.0f} min"
                break
            cond = windows[-1][-1]
            r = gen("D", f"window_{k:02d}", "i2v", wcfg, PROMPT_LONG, SEED + k - 1, max(est * 3, 30 * 60), image=cond)
            if r["status"] != "ok":
                stopped = f"window {k} {r['status']}: {r['detail']}"
                break
            windows.append(np.load(scratch / f"window_{k:02d}.npy"))
            conds.append(cond)
            wrows.append(r)
        if stopped:
            meta["notes"].append(f"Long video stopped at {len(windows)} windows: {stopped}")
        long_npy = None
        if windows:
            long = np.concatenate([windows[0], *(w[1:] for w in windows[1:])])
            long_npy = scratch / "long_video.npy"
            np.save(long_npy, long)
            import imageio.v2 as imageio

            imageio.mimwrite(out / "s4_long_video.mp4", list(long), fps=FPS, quality=8, macro_block_size=1)
            contact_sheet(out / "s4_long_contact.png", windows, labels=[f"w{k + 1}" for k in range(len(windows))])
            meta["long_video"] = {
                "windows": len(windows), "frames_per_window": wcfg["frames"], "steps": wcfg["steps"],
                "total_frames": int(len(long)), "seconds": round(len(long) / FPS, 2),
                "generate_wall_s": round(sum(r["wall_s"] for r in wrows), 1),
                "per_window_wall_s": [r["wall_s"] for r in wrows], "per_window_peak_rss_gb": [r["peak_rss_gb"] for r in wrows],
                "peak_rss_gb": max(r["peak_rss_gb"] for r in wrows), "mp4": "s4_long_video.mp4",
                "contact_sheet": "s4_long_contact.png", "stopped": stopped,
                **long_video_metrics(windows, conds)}
        meta["stages_done"].append("D")
        save_results(out, meta, rows)

        # E. tiled vs untiled decode of the long latents -----------------------------------------------------
        if long_npy is not None and len(windows) > 1 and remaining() - RESERVE_S > 20 * 60:
            lat = scratch / "long_latents.npy"
            enc = job("E", "encode_long", ["encode", "--model", model_dir, "--frames-npy", long_npy, "--npy", lat], 40 * 60)
            if enc["status"] != "ok":  # whole video at once did not fit / work: window-sized pieces
                enc = job("E", "encode_long_chunked", ["encode", "--model", model_dir, "--frames-npy", long_npy, "--npy", lat,
                                                       "--chunk", wcfg["frames"]], 40 * 60)
            dead = set()
            for rep in TILED_REPEATS if enc["status"] == "ok" else []:
                for tiling in ("none", "temporal"):
                    if tiling in dead:
                        continue
                    if remaining() - RESERVE_S < 15 * 60:
                        meta["notes"].append(f"Stage E stopped before repeat {rep} / {tiling}: budget.")
                        dead.update(("none", "temporal"))
                        continue
                    name = f"decode_x{rep}_{tiling}"
                    args = ["decode", "--model", model_dir, "--lat", lat, "--pkg", pkg, "--tiling", tiling, "--repeat", rep, "--fps", FPS]
                    if rep == 1:  # keep the outputs of the real length for the quality comparison
                        args += ["--npy", scratch / f"{name}.npy", "--mp4", out / f"s4_long_{name}.mp4"]
                    r = job("E", name, args, 60 * 60, tiling=tiling, repeat=rep)
                    if r["status"] != "ok":
                        dead.add(tiling)  # longer videos of the same mode would fail the same way
            a_, b_ = scratch / "decode_x1_none.npy", scratch / "decode_x1_temporal.npy"
            if a_.exists() and b_.exists():
                meta["tiled_vs_untiled"] = psnr_frames(a_, b_)
            meta["stages_done"].append("E")
        else:
            meta["notes"].append("Stage E skipped (no long video, or not enough budget left).")
    except Exception as e:
        log(f"stopped: {e!r}")
        meta["notes"].append(f"Run stopped early: {e!r}")
    finally:
        # F. outputs --------------------------------------------------------------------------------------
        try:
            m = meta.get("model_manifest")
            if m and model_dir and m["total_gb"] < MAX_MODEL_COPY_GB and free_gb(out) > m["total_gb"] + 1:
                shutil.copytree(model_dir, out / "ltx_video_ov_int8", dirs_exist_ok=True)
                m["copied"] = True
            elif m:
                meta["notes"].append(f"Model folder not copied to the output ({m['total_gb']} GB, limit {MAX_MODEL_COPY_GB} GB, "
                                     f"{free_gb(out):.1f} GB free): manifest only.")
        except Exception as e:
            meta["notes"].append(f"Copying the model folder failed: {e!r}")
        meta["stages_done"].append("F")
        meta["elapsed_s"] = round(time.time() - T0)
        meta["disk_free_end_gb"] = {str(p): round(free_gb(p), 1) for p in {out, scratch}}
        save_results(out, meta, rows)
        log(f"done in {meta['elapsed_s'] / 3600:.2f} h; outputs in {out}")


# --- BEGIN EMBEDDED (copy of src/longvid_ov/tiled_decode.py) ---
TILED_SRC = r'''"""Memory-bounded decode for the LTX-Video VAE decoder: run the decoder on small pieces, cross-fade the overlaps.

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
'''
# --- END EMBEDDED ---

if __name__ == "__main__":
    main()
