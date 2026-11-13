# Step 8c report: temporal tile size tuning (OpenVINO, FP32)

Generated from the run started 2026-11-12 07:22:24; runtime 70.0 min. Every number below was measured in this run (raw: `s8c_results.json`, `.csv`). Single runs, not averaged. Timings are only comparable inside this run.

## Setup
- Machine: Intel(R) Xeon(R) CPU @ 2.20GHz, 4 logical CPUs, all inference threads, RAM limit 30.0 GB, swap 0.0 GB. Each run is a fresh subprocess; the parent samples its RSS every 50 ms (plus the kernel's VmHWM) and kills it above 28.5 GB or when system-available RAM drops under 1 GB (`OOM`), or at its timeout.
- Versions: python 3.13.18, openvino 2026.4.1, nncf not installed, torch 2.11.0+cpu, diffusers 0.39.0, numpy 2.1.3, scikit-image 0.25.2, psutil 5.9.5, matplotlib 3.10.0.
- IR conversion `convert_real_native`: 36.0 s, peak RSS 1.77 GB, weights 0.89 GB, FP32, 0 Conv3d rewritten as Conv2d.

## C. Tile size tuning
OpenVINO decode of the LTX-Video VAE (native Conv3d IR, FP32). `temporal` = `tiled_decode.py`, first-frame rule `drop_first`. Budget of this kernel: 75 min; a run that did not fit is listed as SKIPPED.
768x512, 97 frames, REAL latents: the diffusers `hiker.mp4` sample clip (49 frames in the file, extended by ping-pong: forward, backward, ...), encoded with the PyTorch encoder (posterior mean). The whole float32 video is kept in RAM in every run of this section (the metrics need it), so peak RSS includes it. Quality = tiled vs untiled decode of the same latents; seam = frame-to-frame change at the chunk boundaries / the clip's median change (1.0 = no seam), next to the same score of the untiled video at the same positions.

Untiled reference: peak RSS 9.69 GB, decode 345.2 s.

| chunk | overlap | stride | decoder calls | peak RSS GB | RSS / untiled | decode s | time / untiled | PSNR dB | worst frame dB | SSIM | seam tiled | seam untiled | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 8 | 8 | 12 | 3.89 | 0.40 | 702.0 | 2.03 | 43.57 | 39.03 | 0.9949 | 1.065 | 1.008 | yes |
| 24 | 8 | 16 | 6 | 4.69 | 0.48 | 515.9 | 1.49 | 46.40 | 39.20 | 0.9971 | 1.058 | 1.024 | yes |
| 24 | 16 | 8 | 12 | 4.79 | 0.49 | 996.1 | 2.89 | 47.45 | 39.04 | 0.9978 | 1.045 | 1.008 |  |
| 32 | 8 | 24 | 4 | 5.17 | 0.53 | 446.0 | 1.29 | 47.98 | 39.51 | 0.9981 | 1.089 | 1.052 | yes |
| 32 | 16 | 16 | **TIMEOUT**: killed after 641 s during decode | | | | | | | | | | |
| 48 | 8 | 40 | **SKIPPED**: time budget used up | | | | | | | | | | |
| 48 | 16 | 32 | **SKIPPED**: time budget used up | | | | | | | | | | |
| 64 | 8 | 56 | **SKIPPED**: time budget used up | | | | | | | | | | |
| 64 | 16 | 48 | **SKIPPED**: time budget used up | | | | | | | | | | |

Pareto = no other tiled setting is at least as good in both peak RSS and decode time. Plot: `s8c_tuning_pareto.png` (x = peak RSS, y = decode time, colour = PSNR).

**Best setting: chunk 32 / overlap 8** (peak RSS 5.17 GB, 446.0 s, 1.29x the untiled time, 0.53x the untiled memory, PSNR 47.98 dB). Rule, fixed before the run: on the memory/time Pareto front, PSNR within 1.0 dB of the best PSNR measured, smallest peak RSS x decode time.
- fastest: chunk 32 / overlap 8 (446.0 s, 5.17 GB); lowest memory: chunk 16 / overlap 8 (3.89 GB, 702.0 s); highest PSNR: chunk 32 / overlap 8.
- default (chunk 16 / overlap 8): 3.89 GB, 702.0 s, PSNR 43.57 dB.

## Limits
- Single runs on a shared Kaggle CPU; small differences between settings can be noise.
- One real clip (hiker.mp4), looped; one resolution and one length.
