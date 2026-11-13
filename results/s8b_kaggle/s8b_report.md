# Step 8b report: OpenVINO vs PyTorch decode memory, and FP16 / INT8 decoders

Generated from the run started 2026-11-12 04:33:21; runtime 119.6 min. Every number below was measured in this run (raw: `s8b_results.json`, `.csv`). Single runs, not averaged. Timings are only comparable inside this run.

## Setup
- Machine: Intel(R) Xeon(R) CPU @ 2.20GHz, 4 logical CPUs, all inference threads, RAM limit 30.0 GB, swap 0.0 GB. Each run is a fresh subprocess; the parent samples its RSS every 50 ms (plus the kernel's VmHWM) and kills it above 28.5 GB or when system-available RAM drops under 1 GB (`OOM`), or at its timeout.
- Versions: python 3.13.18, openvino 2026.4.1, nncf 3.4.0, torch 2.11.0+cpu, diffusers 0.39.0, numpy 2.1.3, scikit-image 0.25.2, psutil 5.9.5, matplotlib 3.10.0.
- IR conversion `convert_real_native`: 29.6 s, peak RSS 1.76 GB, weights 0.89 GB, FP32, 0 Conv3d rewritten as Conv2d.
- Latents: real, from encoding the diffusers `hiker.mp4` sample clip with the PyTorch encoder (posterior mean); shorter videos decode a prefix of the same latents.

## B. Why does untiled OpenVINO decode use more memory than PyTorch?
One untiled decode per fresh process. `peak` = the parent's 50 ms sampler (with the kernel's VmHWM). `before inference` = RSS after the model is loaded (PyTorch) or compiled (OpenVINO); `added` = peak minus that. `anon` / `file` = the kernel's split of the RSS after the inference into anonymous memory (heap) and file-backed pages (memory-mapped weights, libraries). Delta = peak minus the `ov_dynamic` peak of the same video.
Differences under 0.1 GB are not called differences (single runs).

### 768x512, 49 frames

| configuration | status | peak GB | delta GB | before inference GB | added GB | anon GB | file GB | decode s |
|---|---|---|---|---|---|---|---|---|
| `pt_inference_mode`: PyTorch eager, torch.inference_mode | ok | 4.59 | -1.30 | 0.53 | 4.05 | 0.69 | 1.05 | 137.2 |
| `ov_dynamic`: OpenVINO, dynamic-shape IR, compile_model(path) (steps 2-3, the baseline) | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 152.8 |
| `ov_static`: OpenVINO, read_model + model.reshape(static shape) | ok | 4.70 | -1.19 | 1.93 | 2.78 | 3.52 | 0.96 | 142.8 |
| `ov_dynamic_read`: OpenVINO, read_model + compile_model, dynamic (control for the static rows) | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 161.6 |
| `ov_streams1`: OpenVINO, NUM_STREAMS=1 | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 147.9 |
| `ov_latency`: OpenVINO, PERFORMANCE_HINT=LATENCY | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 147.3 |
| `ov_throughput`: OpenVINO, PERFORMANCE_HINT=THROUGHPUT | ok | 5.92 | +0.03 | 1.04 | 4.88 | 4.74 | 0.97 | 295.8 |
| `ov_mmap_off`: OpenVINO, ENABLE_MMAP=False | ok | 5.89 | -0.00 | 1.02 | 4.87 | 5.60 | 0.07 | 150.0 |
| `ov_mmap_on`: OpenVINO, ENABLE_MMAP=True (explicit) | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 153.5 |
| `ov_cache0`: OpenVINO, CPU_RUNTIME_CACHE_CAPACITY=0 | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 149.1 |
| `ov_no_precision_hint`: OpenVINO, no INFERENCE_PRECISION_HINT (plugin default) | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 148.4 |
| `ov_static_streams1_cache0`: OpenVINO, static + NUM_STREAMS=1 + CPU_RUNTIME_CACHE_CAPACITY=0 | ok | 4.70 | -1.19 | 1.93 | 2.78 | 3.52 | 0.97 | 146.2 |
| `ov_dynamic_2infers`: OpenVINO, baseline, two inferences on the same request | ok | 5.89 | +0.00 | 1.02 | 4.87 | 4.71 | 0.97 | 148.3 |
| `pt_no_grad`: PyTorch eager, torch.no_grad (what step 2 used) | ok | 4.58 | -1.31 | 0.53 | 4.05 | 0.69 | 1.05 | 151.2 |

**The gap.** OpenVINO (`ov_dynamic`) peaks at 5.89 GB, PyTorch (`pt_inference_mode`) at 4.59 GB: +1.30 GB.
- Before the first inference: OpenVINO holds 1.02 GB after compile, PyTorch 0.53 GB after loading the decoder: +0.49 GB of the gap.
- During the inference: OpenVINO adds 4.87 GB, PyTorch adds 4.05 GB: +0.82 GB of the gap.
- So in this run the larger part of the gap is memory OpenVINO allocates while it decodes (activations and work buffers).
- Kernel split after the inference: OpenVINO 4.71 GB anonymous + 0.97 GB file-backed; PyTorch 0.69 GB anonymous + 1.05 GB file-backed.
- PyTorch `no_grad` (step 2) vs `inference_mode`: 4.58 vs 4.59 GB.
- **Settings that reduce the peak** (vs `ov_dynamic`): `ov_static` -1.19 GB; `ov_static_streams1_cache0` -1.19 GB. The best one, `ov_static`, leaves +0.12 GB to PyTorch.
- Effective properties of the baseline, as reported by OpenVINO: INFERENCE_PRECISION_HINT=<Type: 'float32'>, NUM_STREAMS=1, PERFORMANCE_HINT=LATENCY, INFERENCE_NUM_THREADS=2, CPU_RUNTIME_CACHE_CAPACITY=n/a (RuntimeError), OPTIMAL_NUMBER_OF_INFER_REQUESTS=1, ENABLE_MMAP=True.
- Second inference on the same request: RSS 5.67 GB after the first, 5.67 GB after the second; 148.27 s then 144.16 s.

**Tensor sizes in the IR** (reshaped to latents `[1, 128, 7, 16, 24]`; sizes follow from the shapes, they are not measured memory):
- output video `[1, 3, 49, 512, 768]` float32: 0.215 GB; weights (constants): 0.89 GB; all 664 intermediate tensors added up (never all alive at once): 64.3 GB.
- the single operation that needs the most at once (inputs + output): `Add` `__module.vae.decoder.up_blocks.3.resnets.0/aten::add/Add`, 1.72 GB. A runtime cannot go below this plus the weights. Measured `added` during inference: OpenVINO 4.87 GB = 2.8x that, PyTorch 4.05 GB = 2.4x.

| largest intermediate tensors | op type | shape | GB |
|---|---|---|---|
| `__module.vae.decoder.up_blocks.3.resnets.0.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.resnets.0.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.resnets.1.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.resnets.1.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.resnets.2.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.resnets.2.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.resnets.3.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.resnets.3.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.conv_out/aten::concatenate/Concat` | Concat | `[1, 128, 51, 128, 192]` <Type: 'float32'> | 0.598 |
| `__module.vae.decoder.up_blocks.3.upsamplers.0.conv.conv/aten::_convolution/Convolution` | Convolution | `[1, 1024, 25, 64, 96]` <Type: 'float32'> | 0.586 |
| `__module.vae.decoder.up_blocks.3.upsamplers.0.conv.conv/aten::_convolution/Add` | Add | `[1, 1024, 25, 64, 96]` <Type: 'float32'> | 0.586 |
| `__module.vae.decoder.up_blocks.3.upsamplers.0/aten::reshape/Reshape` | Reshape | `[1, 128, 2, 2, 2, 25, 64, 96]` <Type: 'float32'> | 0.586 |

### 768x512, 25 frames

| configuration | status | peak GB | delta GB | before inference GB | added GB | anon GB | file GB | decode s |
|---|---|---|---|---|---|---|---|---|
| `pt_inference_mode`: PyTorch eager, torch.inference_mode | ok | 3.13 | -0.86 | 0.53 | 2.60 | 0.60 | 1.05 | 72.1 |
| `ov_dynamic`: OpenVINO, dynamic-shape IR, compile_model(path) (steps 2-3, the baseline) | ok | 3.99 | +0.00 | 1.02 | 2.97 | 2.91 | 0.97 | 77.5 |
| `ov_static`: OpenVINO, read_model + model.reshape(static shape) | ok | 3.37 | -0.62 | 1.93 | 1.44 | 2.29 | 0.97 | 72.5 |
| `ov_dynamic_read`: OpenVINO, read_model + compile_model, dynamic (control for the static rows) | ok | 3.99 | +0.00 | 1.02 | 2.97 | 2.91 | 0.97 | 75.8 |
| `ov_streams1`: OpenVINO, NUM_STREAMS=1 | ok | 3.99 | -0.00 | 1.02 | 2.97 | 2.91 | 0.96 | 75.8 |
| `ov_latency`: OpenVINO, PERFORMANCE_HINT=LATENCY | ok | 3.99 | -0.00 | 1.02 | 2.97 | 2.91 | 0.96 | 75.8 |
| `ov_throughput`: OpenVINO, PERFORMANCE_HINT=THROUGHPUT | ok | 4.02 | +0.03 | 1.04 | 2.98 | 2.94 | 0.97 | 162.1 |
| `ov_mmap_off`: OpenVINO, ENABLE_MMAP=False | ok | 3.99 | +0.00 | 1.02 | 2.97 | 3.80 | 0.08 | 76.0 |
| `ov_mmap_on`: OpenVINO, ENABLE_MMAP=True (explicit) | ok | 3.99 | +0.00 | 1.02 | 2.97 | 2.91 | 0.97 | 75.6 |
| `ov_cache0`: OpenVINO, CPU_RUNTIME_CACHE_CAPACITY=0 | ok | 3.99 | +0.00 | 1.02 | 2.97 | 2.91 | 0.97 | 75.9 |
| `ov_no_precision_hint`: OpenVINO, no INFERENCE_PRECISION_HINT (plugin default) | ok | 3.99 | -0.00 | 1.02 | 2.97 | 2.91 | 0.96 | 75.2 |
| `ov_static_streams1_cache0`: OpenVINO, static + NUM_STREAMS=1 + CPU_RUNTIME_CACHE_CAPACITY=0 | ok | 3.37 | -0.62 | 1.93 | 1.44 | 2.29 | 0.97 | 72.7 |
| `ov_dynamic_2infers`: OpenVINO, baseline, two inferences on the same request | ok | 3.99 | -0.00 | 1.02 | 2.97 | 2.91 | 0.96 | 77.7 |
| `pt_no_grad`: PyTorch eager, torch.no_grad (what step 2 used) | ok | 3.07 | -0.92 | 0.53 | 2.54 | 0.54 | 1.05 | 71.0 |

**The gap.** OpenVINO (`ov_dynamic`) peaks at 3.99 GB, PyTorch (`pt_inference_mode`) at 3.13 GB: +0.86 GB.
- Before the first inference: OpenVINO holds 1.02 GB after compile, PyTorch 0.53 GB after loading the decoder: +0.49 GB of the gap.
- During the inference: OpenVINO adds 2.97 GB, PyTorch adds 2.60 GB: +0.37 GB of the gap.
- So in this run the larger part of the gap is memory OpenVINO holds for the compiled model before any video is decoded.
- Kernel split after the inference: OpenVINO 2.91 GB anonymous + 0.97 GB file-backed; PyTorch 0.60 GB anonymous + 1.05 GB file-backed.
- PyTorch `no_grad` (step 2) vs `inference_mode`: 3.07 vs 3.13 GB.
- **Settings that reduce the peak** (vs `ov_dynamic`): `ov_static` -0.62 GB; `ov_static_streams1_cache0` -0.62 GB. The best one, `ov_static`, leaves +0.24 GB to PyTorch.
- Effective properties of the baseline, as reported by OpenVINO: INFERENCE_PRECISION_HINT=<Type: 'float32'>, NUM_STREAMS=1, PERFORMANCE_HINT=LATENCY, INFERENCE_NUM_THREADS=2, CPU_RUNTIME_CACHE_CAPACITY=n/a (RuntimeError), OPTIMAL_NUMBER_OF_INFER_REQUESTS=1, ENABLE_MMAP=True.
- Second inference on the same request: RSS 3.88 GB after the first, 3.88 GB after the second; 77.69 s then 74.36 s.

**Tensor sizes in the IR** (reshaped to latents `[1, 128, 4, 16, 24]`; sizes follow from the shapes, they are not measured memory):
- output video `[1, 3, 25, 512, 768]` float32: 0.110 GB; weights (constants): 0.89 GB; all 664 intermediate tensors added up (never all alive at once): 33.5 GB.
- the single operation that needs the most at once (inputs + output): `Add` `__module.vae.decoder.up_blocks.3.resnets.0/aten::add/Add`, 0.88 GB. A runtime cannot go below this plus the weights. Measured `added` during inference: OpenVINO 2.97 GB = 3.4x that, PyTorch 2.60 GB = 3.0x.

| largest intermediate tensors | op type | shape | GB |
|---|---|---|---|
| `__module.vae.decoder.up_blocks.3.resnets.0.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.resnets.0.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.resnets.1.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.resnets.1.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.resnets.2.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.resnets.2.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.resnets.3.conv1/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.resnets.3.conv2/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.conv_out/aten::concatenate/Concat` | Concat | `[1, 128, 27, 128, 192]` <Type: 'float32'> | 0.316 |
| `__module.vae.decoder.up_blocks.3.upsamplers.0.conv.conv/aten::_convolution/Convolution` | Convolution | `[1, 1024, 13, 64, 96]` <Type: 'float32'> | 0.305 |
| `__module.vae.decoder.up_blocks.3.upsamplers.0.conv.conv/aten::_convolution/Add` | Add | `[1, 1024, 13, 64, 96]` <Type: 'float32'> | 0.305 |
| `__module.vae.decoder.up_blocks.3.upsamplers.0/aten::reshape/Reshape` | Reshape | `[1, 128, 2, 2, 2, 13, 64, 96]` <Type: 'float32'> | 0.305 |

Plot: `s8b_memory_variants.png`. The explanation above is what the RSS numbers of this run show; it does not look inside the allocator, so it says where the memory appears (before or during inference, heap or file-backed), not which internal buffer it is.

## D. FP16 / INT8 decoder
All variants run on the CPU plugin with `INFERENCE_PRECISION_HINT=f32`, so FP16 here means FP16 weights on disk (expanded to float32 at load), not FP16 arithmetic.

| variant | status | IR on disk GB | vs FP32 | weights by type (GB) | FakeQuantize ops | build s | build peak RSS GB |
|---|---|---|---|---|---|---|---|
| FP32 (baseline) | ok  | 0.890 | 1.00x | <Type: 'float32'> 0.890, <Type: 'int64_t'> 0.000, <Type: 'int32_t'> 0.000 | 0 | 0.0 | 0.08 |
| FP16 weights (`ov.save_model(compress_to_fp16=True)`) | ok  | 0.446 | 0.50x | <Type: 'float16'> 0.445, <Type: 'int64_t'> 0.000, <Type: 'int32_t'> 0.000, <Type: 'float32'> 0.000 | 0 | 1.0 | 1.09 |
| INT8 weights only (`nncf.compress_weights`, INT8_ASYM) | ok  | 0.223 | 0.25x | <Type: 'uint8_t'> 0.222, <Type: 'float32'> 0.000, <Type: 'float16'> 0.000, <Type: 'int64_t'> 0.000, <Type: 'int32_t'> 0.000 | 0 | 11.1 | 1.46 |
| INT8 weights + activations (`nncf.quantize`), EXPERIMENTAL | ok  | 0.224 | 0.25x | <Type: 'int8_t'> 0.222, <Type: 'float32'> 0.000, <Type: 'int64_t'> 0.000, <Type: 'int32_t'> 0.000 | 308 | 614.3 | 5.12 |

EXPERIMENTAL post-training quantization: calibrated on 57 latent samples of shape `[1, 128, 3, 8, 8]` (windows of 3 latent frames x spatial crops) from the SAME hiker clip encoded at 384x256, 161 frames (ping-pong looped). The test clip is therefore not independent of the calibration data; nncf defaults, no accuracy control.

Decode, 49 frames (real latents). Quality is against the FP32 decode in the same mode (isolates the weight format) and against the FP32 untiled decode (what the viewer loses in total, tiling included). Temporal tiled = chunk 16 / stride 8, `drop_first`.

| video | mode | variant | status | peak RSS GB | load s | decode s | time / FP32 | PSNR dB vs FP32 same mode | SSIM | PSNR dB vs FP32 untiled | SSIM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512x320 49f | untiled | fp32 | ok | 3.56 | 0.9 | 59.5 | 1.00 | - | - | - | - |
| 512x320 49f | untiled | fp16 | ok | 3.12 | 0.9 | 60.4 | 1.02 | 86.38 | 1.0000 | 86.38 | 1.0000 |
| 512x320 49f | untiled | int8w | ok | 3.64 | 4.7 | 62.8 | 1.06 | 52.84 | 0.9985 | 52.84 | 0.9985 |
| 512x320 49f | untiled | ptq | ok | 1.91 | 2.4 | 38.8 | 0.65 | 37.34 | 0.9695 | 37.34 | 0.9695 |
| 512x320 49f | temporal tiled | fp32 | ok | 2.66 | 0.9 | 115.0 | 1.00 | - | - | 43.24 | 0.9948 |
| 512x320 49f | temporal tiled | fp16 | ok | 2.22 | 0.9 | 117.6 | 1.02 | 86.39 | 1.0000 | 43.25 | 0.9948 |
| 512x320 49f | temporal tiled | int8w | ok | 2.72 | 4.6 | 116.7 | 1.01 | 53.05 | 0.9985 | 42.98 | 0.9934 |
| 512x320 49f | temporal tiled | ptq | ok | 1.23 | 2.4 | 73.9 | 0.64 | 38.11 | 0.9748 | 36.76 | 0.9694 |
| 768x512 49f | untiled | fp32 | ok | 5.89 | 1.0 | 146.4 | 1.00 | - | - | - | - |
| 768x512 49f | untiled | fp16 | ok | 5.44 | 0.9 | 153.7 | 1.05 | 86.32 | 1.0000 | 86.32 | 1.0000 |
| 768x512 49f | untiled | int8w | ok | 5.91 | 4.6 | 148.1 | 1.01 | 53.21 | 0.9985 | 53.21 | 0.9985 |
| 768x512 49f | untiled | ptq | ok | 3.71 | 2.4 | 95.0 | 0.65 | 37.05 | 0.9701 | 37.05 | 0.9701 |
| 768x512 49f | temporal tiled | fp32 | ok | 3.68 | 0.9 | 280.0 | 1.00 | - | - | 43.98 | 0.9957 |
| 768x512 49f | temporal tiled | fp16 | ok | 3.24 | 1.0 | 293.6 | 1.05 | 86.37 | 1.0000 | 43.98 | 0.9957 |
| 768x512 49f | temporal tiled | int8w | ok | 3.77 | 4.6 | 286.3 | 1.02 | 53.49 | 0.9985 | 43.70 | 0.9943 |
| 768x512 49f | temporal tiled | ptq | ok | 2.07 | 2.4 | 176.0 | 0.63 | 37.83 | 0.9751 | 36.65 | 0.9703 |

Per variant, over its finished runs (only variants whose every run finished):

| variant | runs | mean peak RSS GB | mean time / FP32 | lowest PSNR dB vs FP32 same mode | IR GB |
|---|---|---|---|---|---|
| fp32 | 4 | 3.95 | 1.00 | reference | 0.890 |
| fp16 | 4 | 3.51 | 1.03 | 86.32 | 0.446 |
| int8w | 4 | 4.01 | 1.03 | 52.84 | 0.223 |
| ptq | 4 | 2.23 | 0.64 | 37.05 | 0.224 |

**Verdict** (rule fixed before the run: a variant counts only if its PSNR vs the FP32 decode in the same mode is at least 40 dB in every run). Passing: fp32, fp16, int8w. Failing the bar: ptq.
- lowest memory: **fp16** (mean peak RSS 3.51 GB; FP32 3.95 GB)
- fastest: **fp32** (1.00x the FP32 decode time on average)
- smallest file: **int8w** (0.223 GB)

Plot: `s8b_int8_variants.png`.

## Limits
- Single runs on a shared Kaggle CPU; small differences can be noise.
- One real clip (hiker), 49 frames; quality numbers are for this clip only.
- PSNR / SSIM against the FP32 decode measure how much a variant changes the decoder's output, not how good the video looks.
