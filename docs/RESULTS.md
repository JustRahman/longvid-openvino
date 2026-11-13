# LongVid-OV: final results

One table per experiment. Every table names its source file. All runs are single runs, not averaged benchmarks:
memory is the result, times are indicative.

Machines:
- **Mac**: Apple ARM, 12 threads, 18 GB RAM (Steps 1 and 3 local).
- **Xeon**: Kaggle, Intel Xeon @ 2.20 GHz, 4 vCPU, 30 GB RAM, no swap (everything else).

Peak RSS = the highest resident memory of a fresh subprocess, sampled every 50 ms plus the kernel's `VmHWM` on Linux.
It includes the runtime, the weights and (unless said otherwise) the decoded float32 video.
`OOM` = a watchdog stopped the run when it passed 28.5 GB (28.0 GB in Step 4) on the 30 GB machine.

---

## 1. VAE in OpenVINO: correctness (Mac, FP32)

Source: `results/s1_report.md`, `results/s1_metrics.json`. Clip `hiker.mp4`, 25 frames, 512x320, real latents.

| what | value |
|---|---|
| latents for 25 frames 512x320 | `[1, 128, 4, 10, 16]` (150x fewer values) |
| OV vs PyTorch decode, max abs diff (raw) | 2.538e-05 |
| OV vs PyTorch decode, PSNR | 125.95 dB |
| OV vs PyTorch, max diff after uint8 rounding | 1 / 255 |
| second shape (9 frames 256x192), OV vs PT | 2.193e-05 max abs, 125.49 dB |
| reconstruction vs original clip (both runtimes) | 35.92 dB / SSIM 0.9518 |
| decoder IR size, FP32 | 0.89 GB |

Conv3d speed (latents `[1, 128, 2, 6, 8]`). Source: `results/s1_report.md`, `results/s1_conv3d_baseline.json`, `results/s2_kaggle/s2_report.md`.

| decoder | where | time / kernel |
|---|---|---|
| native Conv3d IR | Mac ARM | 102.25 s, 100.74 s in 45 conv layers on `ref_any_f32` |
| PyTorch | Mac ARM | 3.25 s |
| Conv3d-as-Conv2d IR | Mac ARM | 2.72 s |
| native Conv3d IR | Xeon | 45 conv layers on `brgconv_avx2_f32` / `brgconv_avx2_1x1_f32`; IR vs PyTorch max abs diff 8.285e-06 |

## 2. Untiled decode: memory vs frames (Xeon, OpenVINO 2026.4.1, FP32, native Conv3d)

Source: `results/s2_kaggle/s2_report.md` (raw `s2_results.csv/json`). Random latents scaled with the real latent mean and std.
Cell = peak RSS / decode time.

| frames | latent frames | 512x320 | 768x512 | 1216x704 |
|---|---|---|---|---|
| 9 | 2 | 2.25 GB / 12.3 s | 2.72 GB / 28.4 s | 3.67 GB / 67.7 s |
| 25 | 4 | 2.77 GB / 32.7 s | 3.99 GB / 76.0 s | 6.43 GB / 180.0 s |
| 49 | 7 | 3.57 GB / 62.7 s | 5.89 GB / 167.9 s | 10.57 GB / 340.3 s |
| 97 | 13 | 5.16 GB / 123.7 s | 9.69 GB / 294.6 s | 18.84 GB / 669.1 s |
| 161 | 21 | 7.27 GB / 210.4 s | 14.76 GB / 482.4 s | **OOM** at 28.57 GB (1121 s) |
| 257 | 33 | 10.43 GB / 319.9 s | 22.36 GB / 823.6 s | not run (after OOM) |
| growth per 8 frames | | 0.26 GB | 0.63 GB | 1.38 GB |

PyTorch (diffusers) reference, 768x512, same source:

| frames | OpenVINO untiled | PyTorch untiled | PyTorch spatial (`enable_tiling`) | PyTorch spatial + temporal |
|---|---|---|---|---|
| 25 | 3.99 GB / 76.0 s | 3.17 GB / 72.7 s | 2.57 GB / 88.4 s | 2.41 GB / 170.2 s |
| 49 | 5.89 GB / 167.9 s | 4.55 GB / 138.0 s | 3.54 GB / 168.7 s | 2.68 GB / 355.8 s |
| 97 | 9.69 GB / 294.6 s | 7.49 GB / 287.5 s | 5.52 GB / 395.7 s | 3.10 GB / 729.7 s |

## 3. Tiled vs untiled per mode and size (Xeon, OpenVINO, real latents)

Source: `results/s3_kaggle/s3_report.md` (raw `s3_results.json/csv`). Temporal chunk 16 / stride 8, first-frame rule `drop_first`.
Spatial tile 256 px at 512x320 (stride 192), 512 px at 768x512 (stride 448). Cell = peak RSS / decode time / decoder calls.

512x320:

| frames | none | spatial | temporal | spatial+temporal |
|---|---|---|---|---|
| 49 | 3.55 GB / 67.0 s / 1 | 3.55 GB / 97.3 s / 6 | 2.65 GB / 126.7 s / 6 | 3.18 GB / 185.3 s / 36 |
| 97 | 5.14 GB / 135.1 s / 1 | 4.38 GB / 191.2 s / 6 | 2.74 GB / 267.4 s / 12 | 3.27 GB / 391.2 s / 72 |
| 161 | 7.25 GB / 217.1 s / 1 | 5.49 GB / 319.0 s / 6 | 2.85 GB / 445.6 s / 20 | 3.38 GB / 651.5 s / 120 |
| 257 | 10.41 GB / 344.6 s / 1 | 7.14 GB / 522.3 s / 6 | 3.03 GB / 730.0 s / 32 | 3.56 GB / 1078.0 s / 192 |
| growth 49 -> 257 | +6.86 GB | +3.59 GB | +0.38 GB | +0.38 GB |

768x512:

| frames | none | spatial | temporal | spatial+temporal |
|---|---|---|---|---|
| 49 | 5.88 GB / 162.7 s / 1 | 5.34 GB / 191.1 s / 4 | 3.67 GB / 309.6 s / 6 | 3.57 GB / 371.3 s / 24 |
| 97 | 9.68 GB / 313.5 s / 1 | 8.24 GB / 386.5 s / 4 | 3.88 GB / 652.6 s / 12 | 3.78 GB / 773.9 s / 48 |
| 161 | 14.74 GB / 526.1 s / 1 | 12.11 GB / 649.8 s / 4 | 4.16 GB / 1101.6 s / 20 | 4.07 GB / 1316.5 s / 80 |
| 257 | 22.34 GB / 879.3 s / 1 | 17.89 GB / 1062.3 s / 4 | timeout (40 min) at 4.26 GB | not run (after timeout) |

Quality, tiled vs untiled decode of the same latents (same source, PSNR pooled over all frames):

| size | frames | mode | PSNR | worst frame | SSIM | temporal seam tiled / untiled |
|---|---|---|---|---|---|---|
| 512x320 | 257 | spatial | 43.02 dB | 38.83 dB | 0.9966 | - |
| 512x320 | 257 | temporal | 42.79 dB | 38.67 dB | 0.9938 | 1.032 / 0.999 |
| 512x320 | 257 | spatial+temporal | 39.66 dB | 36.18 dB | 0.9903 | 1.043 / 0.999 |
| 768x512 | 161 | spatial | 46.91 dB | 45.86 dB | 0.9980 | - |
| 768x512 | 161 | temporal | 43.53 dB | 39.11 dB | 0.9949 | 1.041 / 1.003 |
| 768x512 | 161 | spatial+temporal | 41.76 dB | 38.34 dB | 0.9928 | 1.040 / 1.003 |

Port check on the Xeon: our tiling with the diffusers rule vs diffusers' own tiling, 512x320, 49 frames, spatial+temporal:
max abs diff 1.751e-05, 129.19 dB, match (same source).

## 4. The diffusers first-frame bug (one frame late)

Sources: `results/s3_report.md` (Mac, PyTorch runtime, 256x160) and `results/s3_kaggle/s3_report.md` (Xeon, OpenVINO, 512x320).

| run | mode | diffusers rule (drop last) | our rule (drop first) | gain |
|---|---|---|---|---|
| Mac, real VAE, PyTorch, 49 frames 256x160 | temporal | 30.70 dB | 42.29 dB | +11.59 dB |
| Mac, real VAE, PyTorch, 49 frames 256x160 | spatial+temporal | 29.07 dB | 32.53 dB | +3.46 dB |
| Xeon, real VAE, OpenVINO, 49 frames 512x320 | spatial+temporal | 29.31 dB | 40.01 dB | +10.70 dB |

One-frame-shift check (tiled frame i vs untiled frame i and i-1):

| run | rule | vs same frame | vs previous frame | late? |
|---|---|---|---|---|
| Mac, temporal, 49 frames | diffusers | 30.52 dB | 37.63 dB | yes |
| Mac, temporal, 49 frames | ours | 42.11 dB | 31.13 dB | no |
| Xeon, spatial+temporal, 49 frames | diffusers | 29.14 dB | 35.20 dB | yes |
| Xeon, spatial+temporal, 49 frames | ours | 39.88 dB | 29.22 dB | no |

Port check on the Mac (ours with the diffusers rule vs diffusers): max abs diff exactly 0 for spatial, temporal and both (`results/s3_report.md`).

## 5. Flat memory up to 1,025 frames (Xeon, S8a)

Source: `results/s8a_kaggle/s8a_results.csv` (raw `s8a_results.json`, log `longvid-ov-s8a-long-video-and-tuning.log`).
The run was stopped after these rows.
Temporal tiling 16 / 8, 512x320, random latents scaled with the real mean and std, finished frames streamed to disk as uint8
(so the float32 video is not held in RAM).

| frames | latent frames | peak RSS | decode time |
|---|---|---|---|
| 257 | 33 | 2.591 GB | 792.62 s |
| 513 | 65 | 2.597 GB | 1542.24 s |
| 1,025 | 129 | 2.603 GB | 3136.69 s |

Comparison with untiled at the same size (Section 2): 257 frames untiled = 10.43 GB. So 4x the frames in 0.25x the memory.
The 257-frame streamed row (2.59 GB) vs the in-RAM tiled row of Section 3 (3.03 GB): the difference is close to the
float32 video itself (3 x 257 x 320 x 512 x 4 bytes = 0.51 GB, my arithmetic).

## 6. C++ temporal tiling in OpenVINO GenAI (Xeon)

Patch: `patches/0001-ltx-temporal-vae-tiling.patch` (11 files, +377 / -7), on GenAI commit `1e573f4`, OpenVINO 2026.5.1 nightly.

Build and tests. Sources: `results/s5_kaggle/s5_report.md` (first run), `results/s5b_kaggle/s5_report.md` (after the test fix).

| run | what | result |
|---|---|---|
| s5 | wheel build | ok, 23.3 min on 4 cores; `.pyi` stub equals the generated one |
| s5 | new tiling tests | 4 passed, 8 failed: test fixture bug (`AutoencoderKLLTXVideo(path, "CPU")` picked the wrong pybind constructor) |
| s5 | full video-generation suite | 119 passed, 41 errors (setup: `optimum-cli` export of `tiny-random-ltx2.3` failed, unrelated), 8 failed (the same 8 new tests) |
| s5b | wheel build | ok, 20.9 min |
| s5b | new tiling tests | **12 passed, 0 failed**, 124.0 s |

The fix between s5 and s5b changed only the test fixture (construct with the decoder path, then `compile("CPU")`).
The C++ code was the same in both runs, so the s5 parity numbers below hold for the s5b build.

Parity on the real LTX-Video VAE, 512x320, chunk 16 / stride 8. Source: `results/s5_kaggle/s5_report.md`. Differences in uint8 levels of the final video.

| frames | test | reference | max abs diff | values that differ | result |
|---|---|---|---|---|---|
| 49 | C++ tiled | Python tiling on the C++ decoder | 0 | 0.0000 % | identical |
| 97 | C++ tiled | Python tiling on the C++ decoder | 0 | 0.0000 % | identical |
| 49 | C++ untiled | Python untiled post-processing | 0 | 0.0000 % | identical |
| 97 | C++ untiled | Python untiled post-processing | 0 | 0.0000 % | identical |
| 49 | C++ tiled | Python tiled (blends raw float, C++ blends after uint8 clamp) | 21 | 9.49 % | 58.34 dB |
| 97 | C++ tiled | Python tiled (same note) | 21 | 11.57 % | 57.48 dB |
| 49 | C++ tiled | C++ untiled (cost of tiling) | 115 | 31.99 % | 43.03 dB |
| 97 | C++ tiled | C++ untiled (cost of tiling) | 132 | 37.64 % | 42.62 dB |

Memory, same source:

| frames | C++ untiled | C++ tiled | Python untiled | Python tiled |
|---|---|---|---|---|
| 49 | 3.446 GB / 74.23 s | 2.515 GB / 145.31 s | 3.743 GB / 74.57 s | 2.745 GB / 151.34 s |
| 97 | 4.901 GB / 149.26 s | 2.567 GB / 308.43 s | 5.507 GB / 149.02 s | 3.01 GB / 309.48 s |

## 7. Chunk-size tuning (Xeon, S8c, 768x512, 97 frames, real latents)

Source: `results/s8c_kaggle/s8c_report.md`. Untiled reference: 9.69 GB, 345.2 s. Best-setting rule fixed before the run.

| chunk | overlap | stride | calls | peak RSS | RSS / untiled | decode | time / untiled | PSNR | worst frame | seam tiled / untiled |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 (default) | 8 | 8 | 12 | 3.89 GB | 0.40 | 702.0 s | 2.03x | 43.57 dB | 39.03 dB | 1.065 / 1.008 |
| 24 | 8 | 16 | 6 | 4.69 GB | 0.48 | 515.9 s | 1.49x | 46.40 dB | 39.20 dB | 1.058 / 1.024 |
| 24 | 16 | 8 | 12 | 4.79 GB | 0.49 | 996.1 s | 2.89x | 47.45 dB | 39.04 dB | 1.045 / 1.008 |
| **32** | **8** | **24** | **4** | **5.17 GB** | **0.53** | **446.0 s** | **1.29x** | **47.98 dB** | **39.51 dB** | **1.089 / 1.052** |
| 32 | 16 | 16 | - | timeout after 641 s | | | | | | |

Chunks 48 and 64 were not run (time budget used up).

## 8. Dynamic vs static shapes, and other OpenVINO settings (Xeon, S8b)

Source: `results/s8b_kaggle/s8b_report.md`. One untiled decode per fresh process, 768x512.

| configuration | 49 frames peak | vs `ov_dynamic` | 25 frames peak | vs `ov_dynamic` |
|---|---|---|---|---|
| PyTorch, `inference_mode` | 4.59 GB | -1.30 GB | 3.13 GB | -0.86 GB |
| OpenVINO dynamic-shape IR (baseline) | 5.89 GB | 0 | 3.99 GB | 0 |
| OpenVINO `model.reshape` to a static shape | **4.70 GB** | **-1.19 GB** | 3.37 GB | -0.62 GB |
| static + 1 stream + runtime cache 0 | 4.70 GB | -1.19 GB | 3.37 GB | -0.62 GB |
| `NUM_STREAMS=1`, `LATENCY`, mmap on / off, cache 0, no precision hint | 5.89 GB | 0.00 | 3.99 GB | 0.00 |
| `PERFORMANCE_HINT=THROUGHPUT` | 5.92 GB | +0.03 GB | 4.02 GB | +0.03 GB |

Where the memory sits at 49 frames (same source):

| | before inference | added during inference | anon / file after |
|---|---|---|---|
| OpenVINO dynamic | 1.02 GB | 4.87 GB | 4.71 / 0.97 GB |
| OpenVINO static | 1.93 GB | 2.78 GB | 3.52 / 0.96 GB |
| PyTorch | 0.53 GB | 4.05 GB | 0.69 / 1.05 GB |

Tensor sizes in the IR at 49 frames: output video 0.215 GB, weights 0.89 GB, largest single op (`Add` in `up_blocks.3`) 1.72 GB,
nine `Concat` tensors of `[1, 128, 51, 128, 192]` at 0.598 GB each.

## 9. FP16 and INT8 decoders (Xeon, S8b)

Source: `results/s8b_kaggle/s8b_report.md`. CPU plugin with `INFERENCE_PRECISION_HINT=f32` for all, so FP16 = FP16 weights on disk.
PTQ = `nncf.quantize` (weights + activations), calibrated on 57 small crops of the same hiker clip, nncf defaults.
Pass bar fixed before the run: PSNR vs FP32 in the same mode of at least 40 dB in every run.

| variant | IR size | mean peak RSS (4 runs) | mean time / FP32 | lowest PSNR vs FP32 | bar |
|---|---|---|---|---|---|
| FP32 | 0.890 GB | 3.95 GB | 1.00 | reference | pass |
| FP16 weights | 0.446 GB | 3.51 GB | 1.03 | 86.32 dB | pass |
| INT8 weights only | 0.223 GB | 4.01 GB | 1.03 | 52.84 dB | pass |
| INT8 weights + activations (PTQ) | 0.224 GB | 2.23 GB | 0.64 | 37.05 dB | **fail** |

Per run, 49 frames (same source):

| video | mode | FP32 | FP16 | INT8 weights | PTQ |
|---|---|---|---|---|---|
| 512x320 | untiled | 3.56 GB / 59.5 s | 3.12 GB / 60.4 s | 3.64 GB / 62.8 s | 1.91 GB / 38.8 s, 37.34 dB |
| 512x320 | temporal | 2.66 GB / 115.0 s | 2.22 GB / 117.6 s | 2.72 GB / 116.7 s | 1.23 GB / 73.9 s, 38.11 dB |
| 768x512 | untiled | 5.89 GB / 146.4 s | 5.44 GB / 153.7 s | 5.91 GB / 148.1 s | 3.71 GB / 95.0 s, 37.05 dB |
| 768x512 | temporal | 3.68 GB / 280.0 s | 3.24 GB / 293.6 s | 3.77 GB / 286.3 s | 2.07 GB / 176.0 s, 37.83 dB |

## 10. Full LTX-Video INT8 model: export routes (Xeon, S4)

Source: `results/s4_kaggle/s4_report.md`.

| route | status | peak RSS | time |
|---|---|---|---|
| 1a `optimum-cli export openvino --weight-format int8` (FP32 load) | OOM | 28.02 GB | 139.5 s |
| 1b optimum `main_export`, bf16 load, INT8 + VAE compress | OOM | 31.0 GB | 120.7 s |
| 2 component-wise export + `nncf` INT8 + assembled folder | **ok** | 20.03 GB (text encoder export) | 1298.4 s |

Final folder 6.64 GB: text encoder 4.44 GB, transformer 1.80 GB, VAE encoder 0.17 GB, VAE decoder 0.22 GB.
GenAI `Text2VideoPipeline` and `Image2VideoPipeline` both load it (Text2Video in 13.3 s, RSS 1.32 GB after load).

Generation, 512x320, 25 fps, GenAI on CPU (same source):

| run | frames | steps | peak RSS | generate | median step |
|---|---|---|---|---|---|
| Text2Video fast | 25 | 8 | 15.98 GB | 575.6 s | 59.0 s |
| Text2Video quality | 49 | 20 | 16.26 GB | 2226.7 s | 105.5 s |
| Image2Video windows 2-8 | 49 each | 20 | 16.50 - 18.09 GB | 2237 - 2277 s each | 104.4 - 106.5 s |

## 11. Long video from 8 chained windows (Xeon, S4)

Source: `results/s4_kaggle/s4_report.md`. Window 1 = Text2Video, windows 2-8 = Image2Video on the last frame of the previous window.
Result: **385 frames = 15.4 s at 25 fps**, 18199.5 s of process time, peak RSS 18.09 GB.

Seams (jump at the boundary / median frame-to-frame change inside windows, 1.0 = normal):

| boundary | 1->2 | 2->3 | 3->4 | 4->5 | 5->6 | 6->7 | 7->8 |
|---|---|---|---|---|---|---|---|
| seam score | 1.585 | 1.223 | 1.039 | 0.983 | 0.807 | 1.068 | 1.307 |

Drift vs window 1:

| window | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| brightness shift | 0.0 | +0.0075 | -0.0026 | 0.0 | -0.012 | -0.0033 | +0.0129 | -0.0274 |
| RGB shift | 0.0 | 0.0121 | 0.0142 | 0.0214 | 0.0353 | 0.0337 | 0.0247 | 0.0589 |
| sharpness ratio | 1.0 | 0.876 | 0.707 | 0.584 | 0.504 | 0.533 | 0.542 | 0.635 |
| SSIM of middle frame vs window 1 | 1.0 | 0.574 | 0.596 | 0.570 | 0.577 | 0.570 | 0.568 | 0.468 |

## 12. Decoding the generated long video's latents (Xeon, S4)

Source: `results/s4_kaggle/s4_report.md`. Latents = the 385-frame video re-encoded with the folder's VAE encoder; repeat k = extended k times
by ping-pong. Decoder = the folder's INT8-weight VAE decoder, f32 inference. Temporal tiling = our `tiled_decode.py`, 16 / 8.

| frames | latent frames | untiled | temporal tiled |
|---|---|---|---|
| 385 | 49 | 16.06 GB / 561.8 s | 4.61 GB / 1144.1 s (48 calls) |
| 769 | 97 | **27.32 GB** / 1130.7 s | **4.00 GB** / 2283.1 s (96 calls) |
| 1,537 | 193 | **OOM** at 28.03 GB | timeout after 3600 s, at 4.82 GB |

Tiled vs untiled, 385 frames: 46.23 dB, worst frame 39.2 dB.

## 13. ARM memory probe (Mac)

Source: `results/s3_arm_memory_probe.txt`. Real decoder, Conv3d-as-Conv2d IR, 9 frames 256x160: RSS passes 5 GB within seconds at f32 and f16,
while the largest activation at that shape is about 12 MB. Not investigated further; the real curves were run on the Xeon.
