import openvino as ov, openvino_genai as og, nncf, torch, diffusers, transformers, psutil, numpy as np
print("openvino", ov.__version__); print("genai", og.__version__, "| Text2Video:", hasattr(og,"Text2VideoPipeline"), "| Image2Video:", hasattr(og,"Image2VideoPipeline"))
print("nncf", nncf.__version__, "| torch", torch.__version__, "| diffusers", diffusers.__version__, "| transformers", transformers.__version__)
from diffusers import AutoencoderKLLTXVideo; print("diffusers LTX VAE class OK")
p=ov.opset13.parameter([2],np.float32); m=ov.Core().compile_model(ov.Model([ov.opset13.relu(p)],[p]),"CPU"); print("CPU infer:", m(np.array([-1,2],np.float32))[0])
print("devices", ov.Core().available_devices, "| RAM GB", round(psutil.virtual_memory().total/2**30,1))
