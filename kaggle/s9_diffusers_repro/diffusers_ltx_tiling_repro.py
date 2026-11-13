# Minimal repro: AutoencoderKLLTXVideo temporal tiled decode (use_framewise_decoding) shifts frames by one.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.39.0", "imageio[ffmpeg]"], check=True)
import torch, numpy as np, imageio.v3 as iio, diffusers
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKLLTXVideo
print("diffusers", diffusers.__version__, "torch", torch.__version__)
torch.manual_seed(0)
vae = AutoencoderKLLTXVideo.from_pretrained("Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.float32).eval()
# real clip: 49 frames at 256x160
path = hf_hub_download("huggingface/documentation-images", "diffusers/hiker.mp4", repo_type="dataset")
frames = iio.imread(path)[:49]
import torch.nn.functional as F
x = torch.from_numpy(frames).float().div(127.5).sub(1).permute(3, 0, 1, 2)[None]
x = F.interpolate(x[0].permute(1, 0, 2, 3), size=(160, 256), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)[None]
with torch.inference_mode():
    z = vae.encode(x).latent_dist.mode()
    vae.use_framewise_decoding = False
    ref = vae.decode(z).sample
    vae.use_framewise_decoding = True
    til = vae.decode(z).sample
def psnr(a, b):
    a = a.clamp(-1, 1); b = b.clamp(-1, 1)
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(4.0 / mse)
T = ref.shape[2]
print("frames", T, "tile_sample_min_num_frames", vae.tile_sample_min_num_frames, "stride", vae.tile_sample_stride_num_frames)
same = psnr(til[:, :, 1:], ref[:, :, 1:])
prev = psnr(til[:, :, 1:], ref[:, :, :-1])
print(f"tiled vs untiled, same frame:      {same:.2f} dB")
print(f"tiled vs untiled, previous frame:  {prev:.2f} dB")
print("per-frame best match (offset 0 or -1), frames 9..48:",
      [0 if psnr(til[:, :, t], ref[:, :, t]) >= psnr(til[:, :, t], ref[:, :, t-1]) else -1 for t in range(9, T)])
# proposed fix: drop the first frame of each later chunk instead of the last
import types
src = AutoencoderKLLTXVideo._temporal_tiled_decode
def fixed(self, z, temb, return_dict=True):
    import inspect
    code = inspect.getsource(src).replace("decoded = decoded[:, :, :-1, :, :]", "decoded = decoded[:, :, 1:, :, :]")
    code = "\n".join(l[4:] if l.startswith("    ") else l for l in code.splitlines())
    ns = {}
    exec(code, vars(sys.modules[AutoencoderKLLTXVideo.__module__]), ns)
    return ns["_temporal_tiled_decode"](self, z, temb, return_dict)
vae._temporal_tiled_decode = types.MethodType(fixed, vae)
with torch.inference_mode():
    fix = vae.decode(z).sample
print(f"with decoded[:, :, 1:] fix, same frame: {psnr(fix[:, :, 1:], ref[:, :, 1:]):.2f} dB")
