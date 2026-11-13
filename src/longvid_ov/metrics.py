"""Quality metrics for tiled VAE decoding. The functions take videos as [F,H,W,3] views of the raw decoder
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
