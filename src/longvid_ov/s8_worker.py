"""s8 worker: the step 8 jobs that bench_worker.py (step 3) does not have. One job per process.

    python s8_worker.py stats | long | pt | ovprobe | irstats | variant  ... --out result.json

    stats     per-channel latents_mean / latents_std of the VAE -> .npz (for random latents like the pipeline's)
    long      A: OpenVINO decode of a long video, untiled or temporal-tiled; tiled output either kept in RAM
              (float32, as in step 3) or streamed to disk as uint8 frames, chunk by chunk
    pt        B: PyTorch eager untiled decode (torch.inference_mode or torch.no_grad)
    ovprobe   B: OpenVINO untiled decode with one set of properties; RSS logged after every phase
    irstats   B: sizes of the tensors in the IR after reshaping it to a static shape
    variant   D: build an IR variant (fp16 / int8 weight-only / int8 post-training quantized) and describe it

Plain decodes, encodes, IR conversion and the metrics are bench_worker.py jobs.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_worker as bw  # noqa: E402
from bench_worker import load_vae, rss_gb, save  # noqa: E402
from tiled_decode import OVDecoder, TiledDecoder, TilingConfig, blend  # noqa: E402

GB = 2**30


def proc_mem():
    """Linux: the kernel's own numbers for this process, in GB: peak RSS so far (hwm) and the RSS split into
    anonymous memory (heap: activations, copies of weights) and file-backed memory (mmap'ed IR weights, libraries)."""
    keys = {"VmHWM": "hwm", "RssAnon": "anon", "RssFile": "file", "RssShmem": "shmem"}
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k = line.split(":")[0]
                if k in keys:
                    out[keys[k]] = round(int(line.split()[1]) * 1024 / GB, 3)
    except (OSError, ValueError, IndexError):
        pass
    return out


def mark(tag):
    """RSS now + the kernel numbers, under one key of the result."""
    return {f"mem_{tag}": {"rss": rss_gb(), **proc_mem()}}


def latent_frames(frames):
    return (frames - 1) // 8 + 1


def make_latents(stats, frames, height, width):
    """Random normal latents de-normalised like LTXPipeline does: z * std / scaling_factor + mean (step 2 code)."""
    s = np.load(stats)
    mean, std = s["mean"].astype(np.float32), s["std"].astype(np.float32)
    shape = (1, mean.size, latent_frames(frames), height // 32, width // 32)
    z = np.random.default_rng(0).standard_normal(shape, dtype=np.float32)
    return z * (std / float(s["scaling_factor"])).reshape(1, -1, 1, 1, 1) + mean.reshape(1, -1, 1, 1, 1)


def get_latents(a):
    if a.latents:
        return np.load(a.latents)[:, :, : latent_frames(a.frames)], "real"
    return make_latents(a.stats, a.frames, a.height, a.width), "random"


def to_u8(video):
    """Raw decoder output [1,3,F,H,W] in ~[-1,1] -> displayed frames [F,H,W,3] uint8."""
    return np.rint(np.clip(video[0].transpose(1, 2, 3, 0) / 2 + 0.5, 0, 1) * 255).astype(np.uint8)


def decode_temporal_stream(dec, z, sink):
    """TiledDecoder._decode_temporal (first_frame_rule "drop_first") with one change: finished frames go to
    `sink(frames [1,3,n,H,W])` as soon as they are final, instead of into a full-length output array. Only the
    previous decoded chunk is kept, so the working memory does not depend on the video length at all."""
    c, ratio = dec.cfg, dec.cfg.temporal_compression_ratio
    assert c.first_frame_rule == "drop_first"
    lat_f = z.shape[2]
    num_frames = (lat_f - 1) * ratio + 1
    chunk = c.tile_sample_min_num_frames // ratio
    step = c.tile_sample_stride_num_frames // ratio
    keep = c.tile_sample_stride_num_frames
    blend_t = c.tile_sample_min_num_frames - keep
    dec.calls = 0
    prev, pos = None, 0
    for i in range(0, lat_f, step):
        piece = z[:, :, i : i + chunk + 1]
        if i > 0 and piece.shape[2] == 1:
            break
        video = dec._decode_frames(piece, False)
        if i == 0:
            frames = video[:, :, : keep + 1]
        else:
            video = video[:, :, 1:]  # drop the causal first frame: the chunk starts at video frame 8i+1 = pos
            blend(prev, video, blend_t, axis=2)
            frames = video[:, :, :keep]
        frames = frames[:, :, : num_frames - pos]
        sink(frames)
        pos += frames.shape[2]
        prev = video
    assert pos == num_frames, (pos, num_frames)
    return num_frames


# ---- jobs ---------------------------------------------------------------------------------------------
def job_stats(a):
    from safetensors import safe_open

    save(phase="stats")
    with safe_open(os.path.join(a.vae_dir, "diffusion_pytorch_model.safetensors"), "np") as f:
        mean, std = f.get_tensor("latents_mean"), f.get_tensor("latents_std")
    with open(os.path.join(a.vae_dir, "config.json")) as f:
        scaling = float(json.load(f).get("scaling_factor", 1.0))
    np.savez(a.save, mean=mean, std=std, scaling_factor=scaling)
    save(phase="done", channels=int(mean.size), scaling_factor=scaling,
         latents_mean_range=[float(mean.min()), float(mean.max())],
         latents_std_range=[float(std.min()), float(std.max())])


def job_long(a):
    """A. mode none: one decoder call, output in RAM. mode temporal: sink "ram" = TiledDecoder.decode (the full
    float32 video is held, as in step 3); sink "stream" = frames are written to --save as uint8 while decoding.
    --check: also decode in RAM and compare with the streamed file (the streaming loop must not change a pixel)."""
    z, kind = get_latents(a)
    cfg = TilingConfig(tile_sample_min_num_frames=a.tile_t[0], tile_sample_stride_num_frames=a.tile_t[1])
    f_out, h, w = (z.shape[2] - 1) * 8 + 1, z.shape[3] * 32, z.shape[4] * 32
    save(phase="load", latents=kind, latent_shape=list(z.shape), frames_out=f_out,
         video_s_at_24fps=round(f_out / 24, 2), output_float32_gb=round(3 * f_out * h * w * 4 / GB, 3),
         output_uint8_gb=round(3 * f_out * h * w / GB, 3))
    t = time.perf_counter()
    ov_dec = OVDecoder(a.ir, a.threads)
    dec = TiledDecoder(ov_dec, cfg)
    tiled = a.mode == "temporal" and dec._needs_temporal(z.shape)
    save(phase="decode", load_s=round(time.perf_counter() - t, 2), rss_after_load_gb=rss_gb(),
         threads=int(ov_dec.compiled.get_property("INFERENCE_NUM_THREADS")), tiled=tiled)
    out = None
    t = time.perf_counter()
    if tiled and a.sink == "stream":
        with open(a.save, "wb") as f:
            decode_temporal_stream(dec, z, lambda frames: f.write(to_u8(frames).tobytes()))
    else:
        out = dec.decode(z, a.mode)
    decode_s = time.perf_counter() - t
    save(phase="check", decode_s=round(decode_s, 2), s_per_frame=round(decode_s / f_out, 4),
         decoder_calls=dec.calls, rss_after_decode_gb=rss_gb(), **mark("after_decode"))
    if a.check:
        ram = to_u8(dec.decode(z, a.mode))
        disk = np.fromfile(a.save, dtype=np.uint8).reshape(ram.shape)
        save(stream_equals_ram=bool(np.array_equal(ram, disk)),
             stream_vs_ram_max_abs_diff_u8=int(np.abs(ram.astype(np.int16) - disk).max()))
    if a.save and os.path.exists(a.save):
        save(stream_file_gb=round(os.path.getsize(a.save) / GB, 3))
        os.remove(a.save)  # only its size (and the --check) matters
    del out
    save(phase="done")


def job_pt(a):
    """B. PyTorch eager, untiled, the call diffusers makes (vae.decode(z, None))."""
    import torch

    z, kind = get_latents(a)
    save(phase="load", latents=kind, latent_shape=list(z.shape), grad_mode=a.grad, **mark("after_import"))
    t = time.perf_counter()
    vae = load_vae(a.vae_dir, "decoder", a.threads)
    save(phase="decode", load_s=round(time.perf_counter() - t, 2), threads=torch.get_num_threads(),
         rss_after_load_gb=rss_gb(), **mark("after_load"))
    t = time.perf_counter()
    with torch.inference_mode() if a.grad == "inference_mode" else torch.no_grad():
        out = vae.decode(torch.from_numpy(z), None, return_dict=False)[0]
    decode_s = time.perf_counter() - t
    save(phase="done", decode_s=round(decode_s, 2), output_shape=list(out.shape),
         output_gb=round(out.numel() * 4 / GB, 3), rss_after_decode_gb=rss_gb(), **mark("after_infer"))


def job_ovprobe(a):
    """B. OpenVINO untiled decode with one configuration. Phases: import, read, compile, infer(s), copy of the output."""
    import openvino as ov

    z, kind = get_latents(a)
    props, core_props = json.loads(a.props), json.loads(a.core_props)
    save(phase="read", latents=kind, latent_shape=list(z.shape), props=props, core_props=core_props,
         static=a.static, from_path=a.from_path, precision_hint=not a.no_precision_hint, **mark("after_import"))
    core = ov.Core()
    if core_props:
        core.set_property(core_props)
    cfg = {} if a.no_precision_hint else {"INFERENCE_PRECISION_HINT": "f32"}
    if a.threads:
        cfg["INFERENCE_NUM_THREADS"] = a.threads
    cfg.update(props)
    t = time.perf_counter()
    if a.from_path:  # exactly what OVDecoder (steps 2-3) does
        compiled = core.compile_model(str(a.ir), "CPU", cfg)
    else:
        model = core.read_model(str(a.ir))
        if a.static:
            model.reshape(list(z.shape))
        save(phase="compile", **mark("after_read"))
        compiled = core.compile_model(model, "CPU", cfg)
        del model
    request = compiled.create_infer_request()
    effective = {}
    for k in ("INFERENCE_PRECISION_HINT", "NUM_STREAMS", "PERFORMANCE_HINT", "INFERENCE_NUM_THREADS",
              "CPU_RUNTIME_CACHE_CAPACITY", "OPTIMAL_NUMBER_OF_INFER_REQUESTS"):
        try:
            effective[k] = str(compiled.get_property(k))
        except Exception as e:  # property not readable in this build
            effective[k] = f"n/a ({type(e).__name__})"
    try:
        effective["ENABLE_MMAP"] = str(core.get_property("ENABLE_MMAP"))
    except Exception as e:
        effective["ENABLE_MMAP"] = f"n/a ({type(e).__name__})"
    save(phase="infer", compile_s=round(time.perf_counter() - t, 2), effective_properties=effective,
         rss_after_load_gb=rss_gb(), **mark("after_compile"))
    infer_s, after = [], []
    for i in range(a.infers):
        t = time.perf_counter()
        request.infer({0: z})
        infer_s.append(round(time.perf_counter() - t, 2))
        after.append({"rss": rss_gb(), **proc_mem()})
        save(decode_s=infer_s[0], infer_s=infer_s, mem_after_each_infer=after, mem_after_infer=after[0])
    save(phase="copy")
    out = request.get_output_tensor(0).data.copy()  # what OVDecoder returns
    save(phase="done", output_shape=list(out.shape), output_gb=round(out.nbytes / GB, 3),
         rss_after_decode_gb=rss_gb(), **mark("after_copy"))


def job_irstats(a):
    """B. Reshape the IR to the static shape of one video and list the largest tensors it would have to hold."""
    import openvino as ov

    save(phase="irstats")
    model = ov.Core().read_model(str(a.ir))
    channels = model.inputs[0].get_partial_shape()[1].get_length()
    shape = [1, channels, latent_frames(a.frames), a.height // 32, a.width // 32]
    model.reshape(shape)

    def nbytes(port):
        ps = port.get_partial_shape()
        if ps.is_dynamic:
            return None
        return int(np.prod(list(ps.to_shape()), dtype=np.int64)) * port.get_element_type().size

    ops, const_bytes, act_bytes, dynamic, types = [], 0, 0, 0, collections.Counter()
    for op in model.get_ordered_ops():
        kind = op.get_type_name()
        types[kind] += 1
        outs = [nbytes(o) for o in op.outputs()]
        if any(b is None for b in outs):
            dynamic += 1
            continue
        if kind == "Constant":
            const_bytes += sum(outs)
            continue
        if kind in ("Parameter", "Result") or not outs:
            continue
        ins = 0
        for i in op.inputs():
            src = i.get_source_output()
            if src.get_node().get_type_name() != "Constant":
                ins += nbytes(src) or 0
        act_bytes += sum(outs)
        ops.append({"name": op.get_friendly_name(), "type": kind, "shape": list(op.output(0).get_partial_shape().to_shape()),
                    "dtype": str(op.output(0).get_element_type()), "out_gb": round(sum(outs) / GB, 4),
                    "in_plus_out_gb": round((ins + sum(outs)) / GB, 4)})
    result_bytes = sum(nbytes(o) or 0 for o in model.outputs)
    top = sorted(ops, key=lambda o: -o["out_gb"])[: a.top]
    worst = max(ops, key=lambda o: o["in_plus_out_gb"])
    save(phase="done", static_input_shape=shape, output_shape=list(model.outputs[0].get_partial_shape().to_shape()),
         output_tensor_gb=round(result_bytes / GB, 4), largest_tensors=top, largest_op_in_plus_out=worst,
         sum_of_all_activations_gb=round(act_bytes / GB, 3), constants_gb=round(const_bytes / GB, 3),
         ops=len(ops), ops_with_dynamic_output=dynamic, op_types=dict(types.most_common(12)))


def calibration_samples(latents, limit):
    """PTQ calibration set from real latents [1,C,f,h,w]: every window of 3 latent frames (= one default tiled
    chunk, 17 video frames) x square spatial crops (side = the smaller latent side) at 3 positions."""
    z = np.load(latents)
    _, _, f, h, w = z.shape
    side = min(h, w)
    crops = sorted({(y, x) for y in {0, (h - side) // 2, h - side} for x in {0, (w - side) // 2, w - side}})
    samples = [np.ascontiguousarray(z[:, :, t : t + 3, y : y + side, x : x + side], dtype=np.float32)
               for t in range(f - 2) for y, x in crops]
    if len(samples) > limit:
        samples = [samples[i] for i in np.linspace(0, len(samples) - 1, limit).round().astype(int)]
    return samples


def job_variant(a):
    """D. fp32: describe the baseline IR. fp16: ov.save_model(compress_to_fp16=True). int8w: nncf.compress_weights
    (INT8_ASYM, weights only, data-free). ptq: nncf.quantize (weights + activations, needs calibration latents)."""
    import openvino as ov

    core = ov.Core()
    save(phase="build", kind=a.kind)
    t = time.perf_counter()
    if a.kind == "fp32":
        a.save = a.ir
    else:
        model = core.read_model(str(a.ir))
        if a.kind == "fp16":
            ov.save_model(model, a.save, compress_to_fp16=True)
        else:
            import nncf

            if a.kind == "int8w":
                model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
            else:
                samples = calibration_samples(a.calib, a.calib_n)
                save(calibration_samples=len(samples), calibration_sample_shape=list(samples[0].shape))
                model = nncf.quantize(model, nncf.Dataset(samples), subset_size=len(samples))
            ov.save_model(model, a.save, compress_to_fp16=False)
        del model
    build_s = time.perf_counter() - t
    save(phase="describe", build_s=round(build_s, 2))
    saved = core.read_model(str(a.save))  # what is really in the file
    const_gb, types = collections.Counter(), collections.Counter()
    for op in saved.get_ordered_ops():
        types[op.get_type_name()] += 1
        if op.get_type_name() == "Constant":
            et = op.output(0).get_element_type()
            const_gb[str(et)] += int(np.prod(list(op.output(0).get_shape()), dtype=np.int64)) * et.size / GB
    xml, bin_ = os.path.getsize(a.save), os.path.getsize(str(a.save)[:-4] + ".bin")
    save(phase="done", ir=str(a.save), xml_bytes=xml, bin_bytes=bin_, ir_gb=round((xml + bin_) / GB, 4),
         constants_gb_by_type={k: round(v, 4) for k, v in const_gb.most_common()},
         fake_quantize_ops=types.get("FakeQuantize", 0), convolution_ops=types.get("Convolution", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["stats", "long", "pt", "ovprobe", "irstats", "variant"])
    ap.add_argument("--out")
    ap.add_argument("--vae-dir")
    ap.add_argument("--ir")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save")
    ap.add_argument("--stats")
    ap.add_argument("--latents", help="real latents .npy (a prefix is used); without it: random latents from --stats")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--mode", choices=["none", "temporal"], default="none")
    ap.add_argument("--sink", choices=["ram", "stream"], default="ram")
    ap.add_argument("--tile-t", type=int, nargs=2, default=[16, 8], metavar=("MIN", "STRIDE"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--grad", choices=["inference_mode", "no_grad"], default="inference_mode")
    ap.add_argument("--props", default="{}", help="JSON: compile_model properties")
    ap.add_argument("--core-props", default="{}", help="JSON: Core properties set before read_model")
    ap.add_argument("--static", action="store_true", help="model.reshape to the static latent shape")
    ap.add_argument("--from-path", action="store_true", help="compile_model(path) like OVDecoder")
    ap.add_argument("--no-precision-hint", action="store_true")
    ap.add_argument("--infers", type=int, default=1)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--kind", choices=["fp32", "fp16", "int8w", "ptq"])
    ap.add_argument("--calib")
    ap.add_argument("--calib-n", type=int, default=80)
    a = ap.parse_args()
    bw.OUT = a.out

    try:  # Linux: make this process the kernel OOM killer's first choice, not the parent
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    {"stats": job_stats, "long": job_long, "pt": job_pt, "ovprobe": job_ovprobe, "irstats": job_irstats,
     "variant": job_variant}[a.job](a)


if __name__ == "__main__":
    main()
