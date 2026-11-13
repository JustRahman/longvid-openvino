"""Step 3 diagnostic: RSS of the real LTX VAE decoder IR (Conv3d-as-Conv2d, step 1) in OpenVINO on the ARM Mac,
phase by phase, sampled every 10 ms. Exits as soon as RSS passes the cap. Output of the runs done during step 3 is
saved in results/s3_arm_memory_probe.txt. Needs models/s3_local/latents_real_256x160_49f.npy (from --local).

    OMP_NUM_THREADS=4 .venv/bin/python scripts/s3_arm_memory_probe.py <frames> '<ov config json>' <cap GB>
"""
import sys, time, threading, psutil, numpy as np, os, json
import openvino as ov
frames, cfg = int(sys.argv[1]), json.loads(sys.argv[2]); CAP = float(sys.argv[3])
p = psutil.Process(); t0 = time.time(); log = []; stop = False; marks = []
def report():
    prev = 0
    for n, t, r in marks + [("end", time.time() - t0, p.memory_info().rss / 2**30)]:
        pk = max([r2 for t2, r2 in log if prev <= t2 <= t] or [0]); print(f"{n:14s} t={t:6.2f}s rss_now={r:.2f} peak_in_phase={pk:.2f}", flush=True); prev = t
def samp():
    while not stop:
        r = p.memory_info().rss / 2**30; log.append((time.time() - t0, r))
        if r > CAP: print("OVER CAP, exit", flush=True); report(); os._exit(1)
        time.sleep(0.01)
threading.Thread(target=samp, daemon=True).start()
def mark(n): marks.append((n, time.time() - t0, p.memory_info().rss / 2**30))
z = np.load("models/s3_local/latents_real_256x160_49f.npy")[:, :, : (frames - 1) // 8 + 1]
cm = ov.Core().compile_model("models/ltx_vae_ov/vae_decoder.xml", "CPU", {"INFERENCE_NUM_THREADS": 4, **cfg}); mark("loaded")
req = cm.create_infer_request()
req.infer({0: z}); mark("decode1")
req.infer({0: z}); mark("decode2")
z2 = np.ascontiguousarray(z[:, :, :, :4, :4]); req.infer({0: z2}); mark("other shape")
req.infer({0: z}); mark("decode3")
stop = True; report()
