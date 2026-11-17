"""Start ONE vLLM server with exactly the flags of plearn.models.sampler.VLLMSampler (the PL_* environment switches included) and keep it up
until this process is stopped (2-node run: the second machine's server; the trainer's rank 0 reaches it with PL_VLLM_REMOTE).
usage: python -m plearn.models.serve_vllm MODEL GPUS PORT LOG      (VLLM_MEM sets --gpu-memory-utilization)"""
import os, signal, sys, time
from plearn.models.sampler import VLLMSampler

model, gpus, port, log = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
s = VLLMSampler(model, gpus, gpu_memory_utilization=float(os.environ.get("VLLM_MEM", "0.45")), port=port, log=log)
s.wait(timeout=1800); print(f"vLLM up on port {port}, GPUs {gpus}", flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
while s.proc.poll() is None: time.sleep(10)
print("vLLM exited", s.proc.returncode, flush=True); sys.exit(1)
