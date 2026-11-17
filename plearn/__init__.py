"""plearn: dataset registration, evaluation and training for RLVB (EM) and RLVB (GRPO) (the structure of lmms-eval).

plearn/api (the registry, the task, the episode contract), plearn/tasks/<dataset>/ (YAML task files, the episode, the teacher),
plearn/method (the trainer), plearn/models (the model server), the evaluator and the CLI; models and training runs in configs/,
the benchmarks' own code in benchmarks/. python -m plearn tasks | show | eval | train. See plearn/README.md.
"""
import sys as _sys
from pathlib import Path as _Path


def raise_open_files(limit=1 << 20):
    """The soft limit of open files up to the hard one (at most limit), for this process and every process it starts. An ssh
    session can start with a soft limit of 1,024; the RLVB (EM) actor's sampling threads (HTTP connections to the vLLM servers) can pass
    it in a long run (OSError [Errno 24] Too many open files). Returns the soft limit."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = limit if hard == resource.RLIM_INFINITY else min(hard, limit)
        if soft != resource.RLIM_INFINITY and soft < want: resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except (ImportError, ValueError, OSError):
        return None


raise_open_files()


class _CudnnSdpOff:
    """A meta path finder that runs once: after the first 'import torch' of the process, cuDNN SDPA is off."""
    def find_spec(self, name, path=None, target=None):
        if name != "torch": return None
        import importlib.util
        _sys.meta_path.remove(self)
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.loader is None: return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module); _sdp_off()
        spec.loader.exec_module = exec_module
        return spec


def _sdp_off():
    import importlib
    importlib.import_module("torch.backends.cuda").enable_cudnn_sdp(False)


def cudnn_sdp_off():
    """PL_NO_CUDNN_SDP=1: turn off the cuDNN SDPA backend of torch (only that backend) in this process. Use it when the learner
    stops with a cuDNN SDPA error in backward. torch already imported:
    at once; else after the first 'import torch' (plearn does not import torch here). Not set: no change. Returns True when set."""
    import os
    if os.environ.get("PL_NO_CUDNN_SDP") != "1": return False
    if "torch" in _sys.modules: _sdp_off()
    elif not any(isinstance(f, _CudnnSdpOff) for f in _sys.meta_path): _sys.meta_path.insert(0, _CudnnSdpOff())
    return True


cudnn_sdp_off()

REPO = _Path(__file__).resolve().parents[1]                   # the repository
BENCHMARKS = REPO / "benchmarks"              # the benchmarks' own code: proworlds, roboquest, vagen, textarena
for _name in ("proworlds", "vagen"):                          # benchmarks/proworlds; `import vagen` (VAGEN's FrozenLake and Sokoban code)
    if str(BENCHMARKS / _name) not in _sys.path: _sys.path.insert(0, str(BENCHMARKS / _name))
