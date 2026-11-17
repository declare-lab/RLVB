"""Models: a weights folder, or a registered model (configs/models/<name>.yaml: 'path', or 'local_dir' under
$PL_MODELS; when that folder is not there, the first folder of 'local_dir_alt' under $PL_MODELS that is there), served by the picture server of plearn/models/mm_server.py (vLLM, tensor parallel over the given GPUs) with an
optional LoRA adapter, and the processor that renders a prompt with pictures."""
from __future__ import annotations
import json, os

from plearn.api.registry import registry

# the pictures at their native size: the largest picture of the benchmarks (1152 x 1152) is not made smaller
NATIVE_SIZE = json.dumps({"shortest_edge": 32 * 32, "longest_edge": 1152 * 1152})
# the context of a server for thinking with no cap (max_think < 0), when PL_MAX_MODEL_LEN is not set
NO_CAP_CONTEXT = 81920


def resolve(model):
    """(weights folder, registry entry or {})."""
    if os.path.isdir(model): return model, {}
    e = registry().get("models", model)
    root = os.environ.get("PL_MODELS", ""); path = e.get("path") or os.path.join(root, e.get("local_dir", model))
    if not e.get("path") and not os.path.isdir(path):                # another host's folder name (local_dir_alt), in its order
        path = next((p for p in (os.path.join(root, d) for d in e.get("local_dir_alt") or []) if os.path.isdir(p)), path)
    if not os.path.isdir(path): raise FileNotFoundError(f"model {model!r}: no weights at {path} (set PL_MODELS or 'path')")
    return path, e


def processor(path, mm_size=NATIVE_SIZE):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path); size = json.loads(mm_size)
    if hasattr(proc, "image_processor"):
        proc.image_processor.size = size
        for k, v in (("min_pixels", size["shortest_edge"]), ("max_pixels", size["longest_edge"])):
            if hasattr(proc.image_processor, k): setattr(proc.image_processor, k, v)
    return proc


def serve(path, gpus, mem=0.85, log=None, max_think=8192, max_images=128, mm_size=NATIVE_SIZE, adapter=None):
    """The picture server over the GPUs (one tensor-parallel group), with the adapter loaded when given."""
    os.environ.setdefault("PL_MM_SIZE", mm_size); os.environ.setdefault("PL_MAX_IMAGES", str(max_images))
    os.environ.setdefault("PL_MAX_MODEL_LEN", str(max_think + 40960 if max_think >= 0 else NO_CAP_CONTEXT))
    from plearn.models.sampler import MMSampler
    srv = MMSampler(path, gpus, gpu_memory_utilization=mem, log=log); srv.wait(timeout=3600)
    if adapter: srv.load(adapter)
    return srv


FLASH = ("flash_attention_3", "flash_attention_2")


def attn_implementation(name=None, device=None):
    """The trainer's attention implementation (flash attention when it is installed: the fastest setting).
    name (the caller passes PL_ATTN first, then the run file's 'attn'), else PL_ATTN, else "auto" = flash_attention_3 when
    flash_attn_interface imports (FlashAttention 3: Hopper only, H100/H200), else flash_attention_2 when flash_attn imports
    (FlashAttention 2: Ampere and Hopper, e.g. A100), else sdpa; or one of those names, used as given. A100 hosts: FA3 does not run
    there; install flash_attn (FA2) and set PL_ATTN=flash_attention_2 (it overrides a run file's flash_attention_3). The flash packages are not
    in requirements: install the wheel, or put the folder of a prebuilt wheel on PYTHONPATH (for example flash_attn_3
    3.0.0+cu130torch2.13; docs/install.md)."""
    import importlib.util
    name = name or os.environ.get("PL_ATTN") or "auto"
    if name != "auto":
        return name
    if (device is not None and str(device).startswith("cpu")) or not _cuda():
        return "sdpa"                                     # flash attention has no CPU path (a CPU test, a merge on a CPU node)
    if importlib.util.find_spec("flash_attn_interface") is not None:
        return "flash_attention_3"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
