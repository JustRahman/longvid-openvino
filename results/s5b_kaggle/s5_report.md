# Step 5 report: OpenVINO GenAI with temporal VAE tiling for LTX-Video (Kaggle build)

Run started 2026-11-12 06:35:53, runtime 25 min. Intel(R) Xeon(R) CPU @ 2.20GHz, 4 CPUs, python 3.13.18, Linux-6.18.55+-x86_64-with-glibc2.39.

## Build
- openvino.genai commit: `1e573f46bc7143d280558fb21bdadfce65c841b9` [ASR][Samples] Add forced aligner timestamp generation sample (#4652) (2026-11-11); expected `1e573f46bc7143d280558fb21bdadfce65c841b9`: match
- patch `0001-ltx-temporal-vae-tiling.patch` (sha256 `4b1f9cf084f5a03a...`): applied cleanly
- OpenVINO: 2026.5.1.dev20261111, tokenizers 2026.5.1.0.dev20261111 (from ['openvino_tokenizers~=2026.5.1.0.dev', 'openvino~=2026.5.1.0.dev'])
- wheel build: ok, 20.9 min on 4 cores (['c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0']), wheel `openvino_genai-2026.5.1.0-cp313-cp313-linux_x86_64.whl` (7.2 MB) in `wheels/`
- committed `.pyi` stub equals the generated one: True (difference: `s5_stub_diff.patch`, 0 lines)
- install + import check: True `2026.5.1-23845-452ed9862db 2026.5.1.0-2-f731f0dead2 0 8`
- versions after installing the test requirements: openvino 2026.5.1.dev20261111, torch 2.11.0+cpu, transformers 5.5.4, diffusers 0.39.0, optimum-intel 2.2.0.dev0+2039025

## Tests

| run | result | passed | failed | errors | skipped | time s |
|---|---|---|---|---|---|---|
| `tests_new` | ok | 12 | 0 | 0 | 0 | 124.0 |
| `tests_video_generation` | skipped (RUN_FULL_SUITE = False) | - | - | - | - | - |

`tests_new` = `-k TestAutoEncoderKLLTXVideoTemporalTiling`; `tests_video_generation` = all of `test_video_generation.py` + `test_ltx2_rope.py`.


## Parity on the real LTX-Video VAE, 512x320, chunk 16 / stride 8 frames

Skipped (RUN_PARITY = False).
