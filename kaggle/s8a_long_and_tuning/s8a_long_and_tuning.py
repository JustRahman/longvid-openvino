"""Step 8a of LongVid-OV: (A) long video in flat memory, (C) tuning the temporal tile size for speed.

Kaggle "script" kernel (CPU, internet on). Single file: the modules of src/longvid_ov it needs (tiled_decode.py,
metrics.py, bench_worker.py from step 3, plus s8_common.py and s8_worker.py) are embedded at the bottom and written
to disk at start, because every measurement runs in a FRESH subprocess. In the repo the files in src/longvid_ov
are used directly (scripts/s8_embed.py refreshes the embedded copies).

What the full run does (Kaggle, 9.5 h budget, most important first)
  1. pip installs pinned deps, records versions / lscpu / RAM                               -> s8a_env.txt
  2. downloads ONLY vae/ of Lightricks/LTX-Video, converts the decoder to OpenVINO IR (FP32, NATIVE Conv3d)
  3. A: OpenVINO temporal tiled decode (FP32, chunk 16 / stride 8, first-frame rule "drop_first") of 257, 513,
     1025 frames at 512x320 and 257, 513 frames at 768x512, frames streamed to disk as uint8 while decoding.
     RANDOM latents with the VAE's real per-channel mean / std (memory and time depend on shapes, not values).
  4. C: 768x512, 97 frames, REAL latents (hiker.mp4, ping-pong looped to 97 frames): untiled reference, then
     chunk 16/24/32/48/64 frames x overlap 8/16 frames: peak RSS, time, PSNR / SSIM / seam score vs untiled.
  5. A: untiled decode of 513 frames at both resolutions (expected to hit the watchdog; recorded either way)
  6. A: the same tiled decodes with the whole float32 video kept in RAM (what step 3 measured), 257 and 513
     frames, only if the budget still covers the last run
  7. A: 1025 frames at 768x512, streamed (last, it is the longest run)
  8. s8a_results.json / .csv (after every run), s8a_report.md, s8a_long_memory.png, s8a_tuning_pareto.png

Local: --smoke only (tiny random-weight VAE, tiny shapes, 5 GB cap). The full run refuses to start outside Kaggle.
"""
import argparse
import json
import os
import sys
from pathlib import Path

TAG = "s8a"
FPS = 24
DEFAULT_T = (16, 8)  # chunk, stride in video frames (diffusers defaults): overlap 8
CHUNKS, OVERLAPS = (16, 24, 32, 48, 64), (8, 16)
SETTINGS = [(c, o) for c in CHUNKS for o in OVERLAPS if o < c]  # (chunk, overlap); stride = chunk - overlap
PSNR_GUARD_DB = 1.0  # "best setting" rule: PSNR within this of the best measured PSNR


def plan_for(smoke):
    if smoke:
        return {"long": [(256, 160, [33, 65]), (192, 128, [33, 65])], "check_frames": 33, "untiled_frames": 65,
                "ram_frames": [33], "last": (192, 128, 65), "tune": (192, 128, 97), "tile_hw": (128, 96)}
    return {"long": [(512, 320, [257, 513, 1025]), (768, 512, [257, 513, 1025])], "check_frames": 49,
            "untiled_frames": 513, "ram_frames": [257, 513], "last": (768, 512, 1025), "tune": (768, 512, 97),
            "tile_hw": (512, 448)}


def package_dir():
    """src/longvid_ov of the repo, or the embedded copies written to a scratch folder (Kaggle)."""
    here = Path(globals().get("__file__", f"{TAG}.py")).resolve().parent
    pkg = here.parent.parent / "src" / "longvid_ov"
    if not (pkg / "s8_worker.py").exists():
        pkg = Path("/kaggle/temp" if Path("/kaggle/temp").is_dir() else "/tmp") / f"{TAG}_pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        for name, src in EMBEDDED.items():
            (pkg / name).write_text(src)
    sys.path.insert(0, str(pkg))
    return pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny local run of the same code path")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--report-only", metavar="DIR", help="rewrite the report from DIR/s8a_results.json, run nothing")
    a, _ = ap.parse_known_args()  # Kaggle runs scripts through a kernel launcher with its own argv
    pkg = package_dir()
    import s8_common as c

    if a.report_only:
        return write_report(Path(a.report_only), *c.load_saved(a.report_only, TAG))
    smoke = a.smoke or os.environ.get("S8_SMOKE") == "1"
    b = c.Bench(TAG, pkg, smoke, a.out)
    P = plan_for(smoke)
    long_timeout = 300 if smoke else 4 * 3600
    b.meta.update(fps=FPS, default_tile_t=list(DEFAULT_T), settings=SETTINGS, plan=P, psnr_guard_db=PSNR_GUARD_DB)

    def long_run(w, h, f, mode, sink, check=False, timeout_s=None, min_left_s=60):
        name = f"long_{w}x{h}_{f}f_{mode}" + (f"_{sink}" if mode == "temporal" else "") + ("_check" if check else "")
        row = {"section": "A", "name": name, "width": w, "height": h, "frames": f, "mode": mode,
               "sink": sink if mode == "temporal" else "ram", "role": "check" if check else "grid"}
        args = ["long", "--ir", b.ir, "--stats", b.stats, "--frames", f, "--height", h, "--width", w, "--mode", mode,
                "--sink", sink, "--tile-t", *DEFAULT_T]
        if sink == "stream":
            args += ["--save", b.work / f"{name}.u8"]
        return b.run(row, b.s8w, args + (["--check"] if check else []), timeout_s or long_timeout,
                     series=(w, h, mode, sink), min_left_s=min_left_s)

    try:
        b.prepare_model()

        # A.0: the streaming loop must give the same pixels as the step 3 decoder (short clip, both in one process)
        w0, h0, _ = P["long"][0]
        long_run(w0, h0, P["check_frames"], "temporal", "stream", check=True)

        # A.1: long videos, frames streamed to disk. Everything except the very last (longest) run.
        s_per_frame = {}
        for w, h, frames in P["long"]:
            for f in frames:
                if (w, h, f) != P["last"]:
                    r = long_run(w, h, f, "temporal", "stream")
                    if r["status"] == "ok":
                        s_per_frame[(w, h)] = r["s_per_frame"]

        # C: tile size tuning on real latents
        try:
            tune(b, P)
        except RuntimeError as e:
            c.log(f"section C skipped: {e}")
            b.meta["setup"]["section_C"] = {"status": "skipped", "detail": str(e)}

        # A.2: untiled at the length where it is expected not to fit
        for w, h, _ in P["long"]:
            long_run(w, h, P["untiled_frames"], "none", "ram", timeout_s=300 if smoke else 3600)

        # A.3: tiled with the full float32 video held in RAM (step 3 behaviour), if the last run still fits after it
        lw, lh, lf = P["last"]
        est_last = s_per_frame.get((lw, lh), 0) * lf * 1.25
        b.meta["estimated_last_run_s"] = round(est_last)
        for w, h, _ in P["long"]:
            for f in P["ram_frames"]:
                est = s_per_frame.get((w, h), 0) * f * 1.25
                long_run(w, h, f, "temporal", "ram", min_left_s=60 + est_last + est)

        # A.4: the longest run, last
        long_run(lw, lh, lf, "temporal", "stream", timeout_s=max(b.left(), 60))
    finally:
        b.save()
        write_report(b.out, b.rows, b.comps, b.meta)
        c.log(f"outputs in {b.out}: " + ", ".join(sorted(p.name for p in b.out.iterdir() if p.is_file())))


def tune(b, P):
    """C: untiled reference, then every (chunk, overlap); quality of each against the untiled decode."""
    w, h, f = P["tune"]
    latents = b.encode(w, h, f)
    base = ["decode", "--backend", "ov", "--rule", "drop_first", "--latents", latents, "--frames", f, "--ir", b.ir,
            "--tile-hw", *P["tile_hw"]]

    def decode(name, mode, chunk, overlap):
        save_to = b.work / f"{name}.npy"
        row = {"section": "C", "name": name, "width": w, "height": h, "frames": f, "mode": mode, "chunk": chunk,
               "overlap": overlap, "stride": chunk - overlap if chunk else None}
        tile_t = (chunk, chunk - overlap) if chunk else DEFAULT_T
        b.run(row, b.bw, [*base, "--mode", mode, "--tile-t", *tile_t, "--save", save_to])
        return row, save_to

    ref, ref_file = decode(f"tune_{w}x{h}_{f}f_untiled", "none", None, None)
    for chunk, overlap in SETTINGS:
        row, file = decode(f"tune_{w}x{h}_{f}f_chunk{chunk}_overlap{overlap}", "temporal", chunk, overlap)
        b.metrics("tune", row, file, ref, ref_file, layout=True, chunk=chunk, overlap=overlap)
        file.unlink(missing_ok=True)
    ref_file.unlink(missing_ok=True)


# ---- analysis + report (generated from the measured values only) ----------------------------------------
def tuning_table(rows, comps):
    """One dict per finished tiled setting: memory, time, ratios vs untiled, quality, Pareto flag; + the untiled row."""
    ref = next((r for r in rows if r["section"] == "C" and r["mode"] == "none" and r["status"] == "ok"), None)
    T = []
    for r in rows:
        if r["section"] == "C" and r["mode"] == "temporal" and r["status"] == "ok":
            q = next((x for x in comps if x["test"] == r["name"] and x["status"] == "ok"), None)
            seam = (q or {}).get("temporal_seam") or {}
            T.append({
                "chunk": r["chunk"], "overlap": r["overlap"], "rss": r["peak_rss_gb"], "time": r["decode_s"],
                "calls": r.get("decoder_calls"),
                "time_ratio": r["decode_s"] / ref["decode_s"] if ref else None,
                "rss_ratio": r["peak_rss_gb"] / ref["peak_rss_gb"] if ref else None,
                "psnr": q["psnr_db"] if q else None, "psnr_min": q["psnr_min_db"] if q else None,
                "ssim": q["ssim"] if q else None,
                "seam": (seam.get("tiled") or {}).get("score"), "seam_untiled": (seam.get("untiled") or {}).get("score"),
            })
    for t in T:  # Pareto front on (peak RSS, decode time), lower is better for both
        t["pareto"] = not any(o["rss"] <= t["rss"] and o["time"] <= t["time"] and (o["rss"] < t["rss"] or o["time"] < t["time"])
                              for o in T)
    return ref, T


def best_setting(T, guard_db):
    """The rule, fixed before the run: on the memory/time Pareto front, PSNR within `guard_db` of the best PSNR
    measured, smallest (peak RSS x decode time)."""
    with_q = [t for t in T if t["psnr"] is not None]
    if not with_q:
        return None
    top = max(t["psnr"] for t in with_q)
    cand = [t for t in with_q if t["pareto"] and t["psnr"] >= top - guard_db] or [t for t in with_q if t["pareto"]]
    return min(cand, key=lambda t: t["rss"] * t["time"]) if cand else None


def plots(out, rows, comps, meta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = [r for r in rows if r["section"] == "A" and r.get("role") == "grid" and r["status"] in ("ok", "oom")]
    if A:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        label = {("temporal", "stream"): "temporal tiled, frames streamed to disk",
                 ("temporal", "ram"): "temporal tiled, float32 video kept in RAM", ("none", "ram"): "untiled"}
        for key in sorted({(r["width"], r["height"], r["mode"], r["sink"]) for r in A}):
            pts = sorted((r for r in A if (r["width"], r["height"], r["mode"], r["sink"]) == key), key=lambda r: r["frames"])
            line, = ax.plot([r["frames"] for r in pts], [r["peak_rss_gb"] for r in pts], marker="o",
                            linestyle="-" if key[2:] == ("temporal", "stream") else "--" if key[2] == "temporal" else ":",
                            label=f"{key[0]}x{key[1]} {label[key[2:]]}")
            for r in pts:
                if r["status"] == "oom":
                    ax.annotate("OOM (killed)", (r["frames"], r["peak_rss_gb"]), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=8, color=line.get_color())
        ax.axhline(meta["kill_rss_gb"], color="gray", linewidth=0.8)
        ax.text(ax.get_xlim()[0], meta["kill_rss_gb"], " watchdog limit", va="bottom", fontsize=8, color="gray")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["frames"] for r in A}), [str(f) for f in sorted({r["frames"] for r in A})])
        ax.set_xlabel(f"video frames ({meta['fps']} fps)")
        ax.set_ylabel("peak RSS, GB")
        ax.set_ylim(bottom=0)
        ax.set_title("LTX-Video VAE decode on OpenVINO: peak memory vs video length")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "s8a_long_memory.png")
        plt.close(fig)

    ref, T = tuning_table(rows, comps)
    if T:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        q = [t for t in T if t["psnr"] is not None]
        if q:
            sc = ax.scatter([t["rss"] for t in q], [t["time"] for t in q], c=[t["psnr"] for t in q], cmap="viridis",
                            s=70, edgecolors="black", linewidths=0.5, zorder=3)
            fig.colorbar(sc, ax=ax, label="PSNR vs untiled, dB")
        noq = [t for t in T if t["psnr"] is None]
        ax.scatter([t["rss"] for t in noq], [t["time"] for t in noq], c="lightgray", s=70, edgecolors="black",
                   linewidths=0.5, zorder=3)
        for t in T:
            ax.annotate(f"{t['chunk']}/{t['overlap']}", (t["rss"], t["time"]), textcoords="offset points",
                        xytext=(6, 5), fontsize=8)
        front = sorted((t for t in T if t["pareto"]), key=lambda t: t["rss"])
        ax.plot([t["rss"] for t in front], [t["time"] for t in front], color="gray", linewidth=1, zorder=2,
                label="Pareto front (tiled settings)")
        if ref:
            ax.scatter([ref["peak_rss_gb"]], [ref["decode_s"]], marker="*", s=200, color="black", zorder=3, label="untiled")
        ax.set_xlabel("peak RSS, GB")
        ax.set_ylabel("decode time, s")
        g = next(r for r in rows if r["section"] == "C")
        ax.set_title(f"Temporal tile tuning, {g['width']}x{g['height']}, {g['frames']} frames (label = chunk/overlap, frames)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "s8a_tuning_pareto.png")
        plt.close(fig)


def write_report(out, rows, comps, meta):
    import s8_common as c

    try:
        plots(out, rows, comps, meta)
    except Exception as e:  # a missing plot must not cost the report
        c.log(f"plots failed: {e!r}")
    ok = lambda r: r["status"] == "ok"
    num = c.num
    L = c.setup_lines(meta, "Step 8a report: long video in flat memory, and tile size tuning (OpenVINO, FP32)")
    chunk, stride = meta["default_tile_t"]

    # ---- A ----
    A = [r for r in rows if r["section"] == "A"]
    L += ["", "## A. Long video in flat memory",
          f"OpenVINO decode of the LTX-Video VAE (native Conv3d IR, FP32). `temporal` = `tiled_decode.py`, chunk {chunk} "
          f"frames, stride {stride} (overlap {chunk - stride}), first-frame rule `drop_first` (the correct one from step 3). "
          "Latents here are RANDOM normal values scaled with the VAE's real per-channel `latents_mean` / `latents_std`: "
          "memory and time depend on the tensor shapes, not on the values; picture quality is not measured in this section.",
          "",
          "- `streamed`: each chunk's finished frames are converted to uint8 and written to disk at once; only the previous "
          "decoded chunk stays in memory.",
          "- `in RAM`: the whole decoded video is held as float32 (what step 3 measured); its size is the last column.",
          f"- video length = frames / {meta['fps']} fps.", "",
          "| resolution | frames | length s | mode | output | status | peak RSS GB | decode s | s / frame | decoder calls | float32 video GB |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    kind = lambda r: "untiled" if r["mode"] == "none" else "temporal"
    sink = lambda r: "streamed" if r["sink"] == "stream" else "in RAM"
    for r in sorted((r for r in A if r.get("role") == "grid"), key=lambda r: (r["width"], r["mode"] == "none", r["sink"] != "stream", r["frames"])):
        st = "ok" if ok(r) else "skipped" if r["status"] == "skipped" else f"**{r['status'].upper()}**"
        L.append(f"| {r['width']}x{r['height']} | {r['frames']} | {r['frames'] / meta['fps']:.1f} | {kind(r)} | {sink(r)} | {st} | "
                 f"{num(r.get('peak_rss_gb'))} | {num(r.get('decode_s'), '.1f')} | {num(r.get('s_per_frame'), '.3f')} | "
                 f"{num(r.get('decoder_calls'), 'd')} | {num(r.get('output_float32_gb'))} |")
    L.append("")
    for key in sorted({(r["width"], r["height"], r["sink"]) for r in A if r["mode"] == "temporal" and r.get("role") == "grid"}):
        done = sorted((r for r in A if r.get("role") == "grid" and r["mode"] == "temporal" and ok(r)
                       and (r["width"], r["height"], r["sink"]) == key), key=lambda r: r["frames"])
        if len(done) > 1:
            x, y = done[0], done[-1]
            L.append(f"- {key[0]}x{key[1]}, temporal, {'streamed' if key[2] == 'stream' else 'in RAM'}: {x['frames']} -> {y['frames']} "
                     f"frames ({y['frames'] / x['frames']:.1f}x longer): peak RSS {x['peak_rss_gb']:.2f} -> {y['peak_rss_gb']:.2f} GB "
                     f"({y['peak_rss_gb'] - x['peak_rss_gb']:+.2f} GB, {y['peak_rss_gb'] / x['peak_rss_gb']:.2f}x); "
                     f"{x['s_per_frame']:.3f} -> {y['s_per_frame']:.3f} s per frame.")
    for r in A:
        if r.get("role") == "grid" and r["status"] in ("oom", "timeout", "error", "skipped"):
            L.append(f"- {r['width']}x{r['height']}, {r['frames']} frames, {kind(r)} ({sink(r)}): {r['status'].upper()}: {r.get('detail', '')}")
        elif r.get("role") == "grid" and r["mode"] == "none" and ok(r):
            L.append(f"- {r['width']}x{r['height']}, {r['frames']} frames, untiled: it FIT (was expected not to): peak RSS {r['peak_rss_gb']:.2f} GB.")
    for r in A:
        if r.get("role") == "check":
            L.append(f"- Streaming check ({r['width']}x{r['height']}, {r['frames']} frames, same process): " + (
                f"streamed frames equal the in-RAM tiled decode after uint8 rounding: **{r.get('stream_equals_ram')}** "
                f"(max difference {r.get('stream_vs_ram_max_abs_diff_u8')} / 255)." if ok(r) else f"{r['status']}: {r.get('detail', '')}"))
    L += ["", "Plot: `s8a_long_memory.png`."]

    # ---- C ----
    ref, T = tuning_table(rows, comps)
    C = [r for r in rows if r["section"] == "C"]
    L += ["", "## C. Tile size tuning"]
    note = meta["setup"].get("section_C")
    if note:
        L.append(f"**Skipped: {note['detail']}**")
    if C:
        g = C[0]
        enc = next((s for k, s in meta["setup"].items() if k.startswith(f"encode_{g['width']}x{g['height']}") and "clip_frames_in_file" in s), {})
        L += [f"{g['width']}x{g['height']}, {g['frames']} frames, REAL latents: the diffusers `hiker.mp4` sample clip"
              + (f" ({enc['clip_frames_in_file']} frames in the file" + (", extended by ping-pong: forward, backward, ..." if enc.get("clip_extended_by_ping_pong") else "")
                 + (", encoded with diffusers' tiled encoder because the one-shot encode did not fit" if enc.get("tiled_encode") else "") + ")" if enc else "")
              + ", encoded with the PyTorch encoder (posterior mean). The whole float32 video is kept in RAM in every run of "
              "this section (the metrics need it), so peak RSS includes it. Quality = tiled vs untiled decode of the same "
              "latents; seam = frame-to-frame change at the chunk boundaries / the clip's median change (1.0 = no seam), "
              "next to the same score of the untiled video at the same positions.", ""]
        if ref:
            L.append(f"Untiled reference: peak RSS {ref['peak_rss_gb']:.2f} GB, decode {ref['decode_s']:.1f} s.")
        else:
            u = next((r for r in C if r["mode"] == "none"), None)
            L.append(f"**No untiled reference** ({u['status']}: {u.get('detail', '')}): no ratios and no quality numbers." if u else "")
        L += ["", "| chunk | overlap | stride | decoder calls | peak RSS GB | RSS / untiled | decode s | time / untiled | PSNR dB | worst frame dB | SSIM | seam tiled | seam untiled | Pareto |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in sorted(T, key=lambda t: (t["chunk"], t["overlap"])):
            L.append(f"| {t['chunk']} | {t['overlap']} | {t['chunk'] - t['overlap']} | {num(t['calls'], 'd')} | {t['rss']:.2f} | {num(t['rss_ratio'])} | "
                     f"{t['time']:.1f} | {num(t['time_ratio'])} | {c.db(t['psnr'])} | {c.db(t['psnr_min'])} | {num(t['ssim'], '.4f')} | "
                     f"{num(t['seam'], '.3f')} | {num(t['seam_untiled'], '.3f')} | {'yes' if t['pareto'] else ''} |")
        for r in C:
            if r["mode"] == "temporal" and not ok(r):
                L.append(f"| {r['chunk']} | {r['overlap']} | {r['stride']} | **{r['status'].upper()}**: {r.get('detail', '')} | | | | | | | | | | |")
        L += ["", "Pareto = no other tiled setting is at least as good in both peak RSS and decode time. Plot: `s8a_tuning_pareto.png` "
              "(x = peak RSS, y = decode time, colour = PSNR)."]
        best = best_setting(T, meta["psnr_guard_db"])
        if best:
            pick = lambda key, f: f(T, key=lambda t: t[key])
            fast, lean = pick("time", min), pick("rss", min)
            name = lambda t: f"chunk {t['chunk']} / overlap {t['overlap']}"
            L += ["", f"**Best setting: {name(best)}** (peak RSS {best['rss']:.2f} GB, {best['time']:.1f} s"
                  + (f", {best['time_ratio']:.2f}x the untiled time, {best['rss_ratio']:.2f}x the untiled memory" if ref else "")
                  + f", PSNR {c.db(best['psnr'])} dB). Rule, fixed before the run: on the memory/time Pareto front, PSNR within "
                  f"{meta['psnr_guard_db']} dB of the best PSNR measured, smallest peak RSS x decode time.",
                  f"- fastest: {name(fast)} ({fast['time']:.1f} s, {fast['rss']:.2f} GB); lowest memory: {name(lean)} "
                  f"({lean['rss']:.2f} GB, {lean['time']:.1f} s)"
                  + (f"; highest PSNR: {name(max((t for t in T if t['psnr'] is not None), key=lambda t: t['psnr']))}." if any(t["psnr"] is not None for t in T) else "."),
                  f"- default (chunk {chunk} / overlap {chunk - stride}): " + next((
                      f"{t['rss']:.2f} GB, {t['time']:.1f} s, PSNR {c.db(t['psnr'])} dB." for t in T
                      if (t["chunk"], t["overlap"]) == (chunk, chunk - stride)), "not measured.")]
        elif T:
            L += ["", "No best setting named: no tiled setting has quality numbers."]
    L += ["", "## Limits",
          "- Single runs on a shared Kaggle CPU; small differences between settings can be noise.",
          "- Section A uses random latents (no picture to look at); section C uses one real clip, looped.",
          "- Streamed output is uint8 on disk; the in-RAM rows hold float32, so the two are not the same output format."]
    (out / "s8a_report.md").write_text("\n".join(L) + "\n")


# --- BEGIN EMBEDDED src/longvid_ov (generated by scripts/s8_embed.py; do not edit by hand) ---
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
    "s8_common.py": r'''"""Shared by the two step 8 Kaggle kernels (s8a_long_and_tuning, s8b_memory_and_int8): the step 2 / step 3
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
''',
    "s8_worker.py": r'''"""s8 worker: the step 8 jobs that bench_worker.py (step 3) does not have. One job per process.

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
''',
}
# --- END EMBEDDED ---

if __name__ == "__main__":
    main()
