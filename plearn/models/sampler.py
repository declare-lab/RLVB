"""Sampling through a vLLM server with the trainer's LoRA hot-loaded after every update.

The trainer exports its LoRA weights in PEFT layout, the server loads them under a fresh adapter name
(VLLM_ALLOW_RUNTIME_LORA_UPDATING), and completions are requested with that name. Prompts are raw strings
(the chat template is applied by the caller), so the sampler sees exactly what the trainer scores.
"""
from __future__ import annotations
import itertools, json, os, re, socket, subprocess, sys, time
import torch
import urllib.error, urllib.request


def check_spec(spec):
    """MTP with a rule per batch size (num_speculative_tokens_per_batch_size) needs vLLM 0.32.0 or
    later. vLLM 0.29-0.31 stop at the start in the drafter's CUDA-graph capture ('AssertionError: 0 < num_reqs <= num_tokens',
    vLLM issue #58692; the fix of PR #48329 is in 0.32.0), and their drafter ignores the rule (issue #51510; PR #57053, in 0.32.0)."""
    if not spec or "num_speculative_tokens_per_batch_size" not in spec: return
    from importlib.metadata import version
    v = version("vllm")
    if tuple(int(x) for x in re.findall(r"\d+", v)[:2]) < (0, 32):
        raise RuntimeError(f"PL_VLLM_SPEC has an MTP rule per batch size, which needs vLLM 0.32.0 or later (this is vLLM {v}): "
                           "pip install vllm==0.32.0 (torch stays 2.13.0), or set PL_VLLM_SPEC= (no MTP)")


def gen_timeout():
    """The HTTP client timeout of one /generate request in seconds (MMSampler._generate): PL_GEN_TIMEOUT, default 7200."""
    return float(os.environ.get("PL_GEN_TIMEOUT") or 7200)


def is_timeout(ex):
    """Whether an exception of urlopen is a client timeout (the read: TimeoutError / socket.timeout; the connect: URLError with it)."""
    if isinstance(ex, (TimeoutError, socket.timeout)): return True
    return isinstance(ex, urllib.error.URLError) and isinstance(getattr(ex, "reason", None), (TimeoutError, socket.timeout))


def free_port():
    """A free TCP port below the kernel's ephemeral range (Linux: 32768 and up). A vLLM server binds its port about a minute after
    the port is chosen; an ephemeral port can go to another socket in that time ('Address already in use')."""
    import random
    for _ in range(200):
        p = random.SystemRandom().randint(20000, 32000)              # not the trainer's seeded random: two processes never share a choice
        with socket.socket() as s:
            try: s.bind(("0.0.0.0", p)); return p
            except OSError: continue
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def export_lora(model, path, base_model, r, alpha=32):
    """Write the LoRALinear modules of `model` as a PEFT adapter directory that vLLM can load."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True); tensors, targets = {}, set()
    for name, mod in model.named_modules():
        if hasattr(mod, "A") and hasattr(mod, "B") and hasattr(mod, "base"):
            tensors[f"base_model.model.{name}.lora_A.weight"] = mod.A.detach().to(torch.bfloat16).cpu().contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = mod.B.detach().to(torch.bfloat16).cpu().contiguous()
            targets.add(name.rsplit(".", 1)[-1])
    save_file(tensors, os.path.join(path, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "base_model_name_or_path": base_model, "r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
               "bias": "none", "target_modules": sorted(targets), "task_type": "CAUSAL_LM", "inference_mode": True},
              open(os.path.join(path, "adapter_config.json"), "w"), indent=1)
    return path


class VLLMSampler:
    def __init__(self, model, gpu, *, max_model_len=8192, max_lora_rank=16, gpu_memory_utilization=0.45, max_num_seqs=64, log=None, port=None):
        max_model_len = int(os.environ.get("PL_MAX_MODEL_LEN", max_model_len)); self.max_model_len = max_model_len   # uncapped thinking
        max_num_seqs = int(os.environ.get("PL_MAX_NUM_SEQS", max_num_seqs))       # speed
        self.model, self.port = model, port or free_port(); self.base = f"http://127.0.0.1:{self.port}"
        argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", "base",
                "--port", str(self.port), "--dtype", "bfloat16", "--max-model-len", str(max_model_len), "--enable-lora",
                "--max-lora-rank", str(max_lora_rank), "--max-loras", "2", "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-num-seqs", str(max_num_seqs), "--enable-prefix-caching", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
        tp = len([g for g in str(gpu).split(",") if g.strip()])       # PL_VLLM_TP=1: one server over all the GPUs (tensor parallel)
        if tp > 1:                                                     # custom all-reduce + sleep mode fail in CUDA-graph capture:
            argv += ["--tensor-parallel-size", str(tp)] + (["--disable-custom-all-reduce"] if os.environ.get("PL_VLLM_SLEEP") == "1" else [])   # off only with sleep
        if os.environ.get("PL_VLLM_SPEC"):                             # e.g. {"method": "mtp", "num_speculative_tokens": 3}
            check_spec(os.environ["PL_VLLM_SPEC"]); argv += ["--speculative-config", os.environ["PL_VLLM_SPEC"]]
        if os.environ.get("PL_MAX_BATCHED"): argv += ["--max-num-batched-tokens", os.environ["PL_MAX_BATCHED"]]   # speed: the E-step is prefill-bound at the default 8,192
        if os.environ.get("PL_PRIORITY") == "1": argv += ["--scheduling-policy", "priority"]   # episode turns before the E-step's samples
        if os.environ.get("PL_VLLM_ARGS"):
            import shlex; argv += shlex.split(os.environ["PL_VLLM_ARGS"])   # any other vLLM flag (speed tests)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)    # the trainer's expandable_segments (cuMem) breaks vLLM's custom all-reduce IPC
        if tp > 1:                                                     # the trainer's torchrun variables must not reach vLLM's own workers
            for k in [k for k in env if k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                                              "GROUP_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT") or k.startswith("TORCHELASTIC_")]: env.pop(k)
        self.sleep_mode = os.environ.get("PL_VLLM_SLEEP") == "1"     # speed + memory: the server sleeps (weights to CPU, KV freed) during the M-step
        if self.sleep_mode: argv.append("--enable-sleep-mode"); env["VLLM_SERVER_DEV_MODE"] = "1"
        if log: os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
        self.log = open(log, "a") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen(argv, env=env, stdout=self.log, stderr=subprocess.STDOUT)   # same session: dies with the trainer
        self.adapter = None; self.n_loaded = 0
        import atexit; atexit.register(self.stop)

    def _post(self, route, payload, timeout=600):
        req = urllib.request.Request(self.base + route, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r: body = r.read()
        try: return json.loads(body)
        except ValueError: return body.decode(errors="replace")          # the adapter routes answer with plain text

    def wait(self, timeout=None):
        timeout = float(os.environ.get("PL_SERVER_START_TIMEOUT") or 900) if timeout is None else timeout   # a cold load of a large model can take more than 700 s
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None: raise RuntimeError("vLLM server exited")
            try:
                urllib.request.urlopen(self.base + "/health", timeout=5); return
            except Exception: time.sleep(3)
        raise RuntimeError("vLLM server did not come up")

    def score(self, prefix, conts, *, n_prefix, use_adapter=True):
        """The summed log-probability under the server of each continuation after the prefix (prompt_logprobs of prefix +
        continuation; the server caches the prefix). n_prefix: the prefix's token count under the model's tokenizer."""
        from concurrent.futures import ThreadPoolExecutor
        def chunk(cs):                                                   # one request per 32 continuations: the server batches them
            payload = {"model": self.adapter if (use_adapter and self.adapter) else "base", "prompt": [prefix + c for c in cs], "max_tokens": 1,
                       "temperature": 0.0, "prompt_logprobs": 0, "skip_special_tokens": True}
            ch = sorted(self._post("/v1/completions", payload)["choices"], key=lambda c: c["index"])
            return [float(sum(next(iter(t.values()))["logprob"] for t in c["prompt_logprobs"][n_prefix:] if t)) for c in ch]
        with ThreadPoolExecutor(max_workers=8) as ex: return [x for part in ex.map(chunk, [conts[i:i + 32] for i in range(0, len(conts), 32)]) for x in part]

    def load(self, adapter_dir):
        """Hot-load a new adapter directory; the previous one is unloaded."""
        self.adapter_dir = adapter_dir
        name = f"lora{self.n_loaded}"; self.n_loaded += 1
        self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": os.path.abspath(adapter_dir)})
        if self.adapter: self._post("/v1/unload_lora_adapter", {"lora_name": self.adapter})
        self.adapter = name

    def complete(self, prompts, *, n=1, max_tokens=1024, temperature=1.0, top_p=0.95, top_k=20, stop=None, logprobs=False, use_adapter=True,
                 priority=None):
        """Returns, per prompt, a list of n dicts: text, finish_reason, and token logprobs when requested."""
        from concurrent.futures import ThreadPoolExecutor
        def one(p):
            mt = max_tokens
            if mt < 0:          # uncapped thinking: the rest of the context, less PL_TAIL_RESERVE tokens for the belief line and action
                mt = self.max_model_len - self._post("/tokenize", {"model": "base", "prompt": p})["count"] - int(os.environ.get("PL_TAIL_RESERVE", 512))
            elif len(p.encode()) + mt > self.max_model_len:   # a byte-level BPE token is >= 1 byte: only a prompt this long can overflow the context
                mt = min(mt, self.max_model_len - self._post("/tokenize", {"model": "base", "prompt": p})["count"])
            if mt <= 0:         # the prompt fills the context (else vLLM gives HTTP 400 in the belief-line request):
                return [{"text": "", "finish": "length", "logprobs": [] if logprobs else None} for _ in range(n)]   # a cut, empty completion
            # (mt was clipped above: a completion that reaches the context end comes back with finish "length", the same as a cut thought)
            payload = {"model": self.adapter if (use_adapter and self.adapter) else "base", "prompt": p, "n": n, "max_tokens": mt,
                       "temperature": temperature, "top_p": top_p, "top_k": top_k, "stop": stop or [], "skip_special_tokens": True}
            if os.environ.get("PL_PRESENCE_PENALTY"): payload["presence_penalty"] = float(os.environ["PL_PRESENCE_PENALTY"])
            if logprobs: payload["logprobs"] = 0
            if priority is not None and os.environ.get("PL_PRIORITY") == "1": payload["priority"] = priority
            if n > 1 and os.environ.get("PL_PREFIX_WARM") == "1":    # speed: vLLM runs the n samples as n requests in one step, each with its own
                self._post("/v1/completions", {**payload, "n": 1, "max_tokens": 1, "temperature": 0.0, "stop": [], "logprobs": None},   # prefill;
                           timeout=int(os.environ.get("PL_HTTP_TIMEOUT", 3600)))       # one 1-token request first puts the prompt in the prefix cache
            try: r = self._post("/v1/completions", payload, timeout=int(os.environ.get("PL_HTTP_TIMEOUT", 3600)))   # uncapped thoughts can run > 600 s
            except urllib.error.HTTPError as ex:   # the server's reason goes to the log
                print(f"sampler: HTTP {ex.code} max_tokens {mt} prompt bytes {len(p.encode())}: {ex.read()[:2000]!r}", file=sys.stderr, flush=True)
                raise
            return [{"text": c["text"], "finish": c["finish_reason"], "logprobs": (c.get("logprobs") or {}).get("token_logprobs")} for c in r["choices"]]
        with ThreadPoolExecutor(max_workers=int(os.environ.get("PL_HTTP_WORKERS", 16))) as ex: return list(ex.map(one, prompts))   # one TP server: more requests at once

    def sleep(self):
        if self.sleep_mode: self._post("/sleep?level=1", {})

    def wake(self):
        """Wake from sleep, then load the same adapter again (its GPU copy is not kept over a sleep)."""
        if not self.sleep_mode: return
        self._post("/wake_up", {})
        if self.adapter:
            old = self.adapter; name = f"lora{self.n_loaded}"; self.n_loaded += 1
            self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": os.path.abspath(self.adapter_dir)}); self.adapter = name
            try: self._post("/v1/unload_lora_adapter", {"lora_name": old})
            except Exception: pass

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(60)
            except Exception: self.proc.kill()


class RemoteVLLMSampler(VLLMSampler):
    """A vLLM server that another process started (on another machine of a 2-node run, for example 8 GPUs, 4 on each
    of two machines): the same requests over HTTP; adapters are loaded from paths on the shared disk. Start it with
    plearn/models/serve_vllm.py, which uses the same flags as VLLMSampler."""
    def __init__(self, url, max_model_len=8192):
        self.base = url.rstrip("/"); self.port = int(self.base.rsplit(":", 1)[1]); self.proc = None
        self.max_model_len = int(os.environ.get("PL_MAX_MODEL_LEN", max_model_len)); self.sleep_mode = os.environ.get("PL_VLLM_SLEEP") == "1"
        self.adapter = None; self.n_loaded = 0; self.log = subprocess.DEVNULL

    def wait(self, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                urllib.request.urlopen(self.base + "/health", timeout=5); return
            except Exception: time.sleep(5)
        raise RuntimeError(f"remote vLLM server {self.base} did not come up")

    def stop(self):
        pass                                                    # its own launcher stops it


# --------------------------------------------------------------------- prompts with images
IMAGES = {}            # prompt key -> the images of that prompt, in order; filled by the trainer when it builds a prompt
MM_SIZE = '{"shortest_edge": 16384, "longest_edge": 147456}'   # the lab's pixel limits (a 128x128 core is 16 image tokens):
                                                               # a domain without a size keeps the model's own
_TAGS = itertools.count()


def tag_prompt(text, images):
    """Keep the images of an image prompt under a key of its own. Two histories can have the same text (the same actions)
    and other images, so the key is a tag in front of the text, which untag removes before the model reads it."""
    key = f"\x00{next(_TAGS)}\x00"; IMAGES[key] = images; return key + text


def prompt_key(text):
    """The key of a prompt's images: its tag, else everything before the assistant turn."""
    if text.startswith("\x00"): return text[:text.index("\x00", 1) + 1]
    return text.split("<|im_start|>assistant\n", 1)[0]


def untag(text):
    """The prompt as the model reads it."""
    return text[text.index("\x00", 1) + 1:] if text.startswith("\x00") else text


def png64(img):
    """The image as base64 PNG, encoded once per image object (the sampler sends every image of a prompt with each request)."""
    import base64, io
    if getattr(img, "_png64", None) is None:
        buf = io.BytesIO(); img.save(buf, format="PNG"); img._png64 = base64.b64encode(buf.getvalue()).decode()
    return img._png64


def img_id(img):
    """A content id of the image (the hash of its PNG), the key of the image server's cache."""
    import hashlib
    if getattr(img, "_id", None) is None: img._id = hashlib.sha1(png64(img).encode()).hexdigest()
    return img._id


class MMSampler(VLLMSampler):
    """The same interface over plearn/models/mm_server.py, a vLLM engine that accepts raw prompts with images."""
    def __init__(self, model, gpu, *, max_model_len=16384, max_lora_rank=16, gpu_memory_utilization=0.45, max_images=24, log=None, port=None):
        self.model, self.port = model, port or free_port(); self.base = f"http://127.0.0.1:{self.port}"
        max_model_len = int(os.environ.get("PL_MAX_MODEL_LEN", max_model_len)); self.max_model_len = max_model_len
        max_images = int(os.environ.get("PL_MAX_IMAGES", max_images)); tp = len([g for g in str(gpu).split(",") if g.strip()])
        argv = [sys.executable, "-m", "plearn.models.mm_server", "--model", model, "--port", str(self.port), "--mem", str(gpu_memory_utilization),
                "--max_model_len", str(max_model_len), "--max_lora_rank", str(max_lora_rank), "--max_images", str(max_images), "--tp", str(tp),
                "--max_num_seqs", os.environ.get("PL_MAX_NUM_SEQS", "64")]
        if os.environ.get("PL_PRESENCE_PENALTY"): argv += ["--presence_penalty", os.environ["PL_PRESENCE_PENALTY"]]
        if os.environ.get("PL_MM_SIZE"): argv += ["--mm_size", os.environ["PL_MM_SIZE"]]   # the domain's picture size (the trainer sets it)
        if os.environ.get("PL_MAX_BATCHED"): argv += ["--max_batched", os.environ["PL_MAX_BATCHED"]]
        if os.environ.get("PL_VLLM_SPEC"): check_spec(os.environ["PL_VLLM_SPEC"]); argv += ["--spec", os.environ["PL_VLLM_SPEC"]]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)                       # as VLLMSampler: the trainer's allocator setting breaks vLLM's IPC
        if tp > 1:                                                     # the trainer's torchrun variables must not reach vLLM's own workers
            for k in [k for k in env if k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                                              "GROUP_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT") or k.startswith("TORCHELASTIC_")]: env.pop(k)
        if log: os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
        self.log = open(log, "a") if log else subprocess.DEVNULL
        # its own process group, so that stopping it also stops vLLM's engine-core child
        self.proc = subprocess.Popen(argv, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.adapter = None; self.n_loaded = 0
        import atexit; atexit.register(self.stop)

    def stop(self):
        """Stop the server and every process of its group: vLLM's engine and workers can outlive the server process for some
        seconds, with their GPU memory (else the next user of the GPUs can run out of memory)."""
        import signal
        if self.proc.poll() is None:
            try: os.killpg(self.proc.pid, signal.SIGTERM); self.proc.wait(60)
            except Exception:
                try: os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception: pass
        t_ = time.time()
        while time.time() - t_ < 120:
            try: os.killpg(self.proc.pid, 0)
            except ProcessLookupError: return
            time.sleep(1)
        try: os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception: pass

    def load(self, adapter_dir):
        self._post("/load", {"path": os.path.abspath(adapter_dir)}); self.adapter = adapter_dir

    def _generate(self, payload):
        """POST /generate with the client timeout gen_timeout() (PL_GEN_TIMEOUT). A timeout sends the request once more
        (logged); a second timeout fails, and the actor stops."""
        t = gen_timeout()
        for k in range(2):
            try: return self._post("/generate", payload, timeout=t)
            except Exception as ex:
                if k or not is_timeout(ex): raise
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} /generate at {self.base}: no answer in {t:.0f} s (PL_GEN_TIMEOUT); "
                      f"the request goes again, once ({len(payload.get('requests', []))} prompts)", flush=True)

    def complete(self, prompts, *, n=1, max_tokens=1024, temperature=1.0, top_p=0.95, top_k=20, stop=None, logprobs=False, use_adapter=True,
                 priority=None):
        """Images go by content id; the bytes of an image go once, the first time the server sees it (it keeps a cache), and
        again if the server asks for them (its cache was emptied)."""
        known = getattr(self, "known", None)
        if known is None: known = self.known = set()
        reqs = [{"prompt": untag(p), "images": [img_id(im) for im in IMAGES.get(prompt_key(p), [])]} for p in prompts]
        ims = {img_id(im): im for p in prompts for im in IMAGES.get(prompt_key(p), [])}
        body = {"requests": reqs, "n": n, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "stop": stop or [], "use_adapter": use_adapter, "priority": 1 if priority is None else priority}
        for attempt in range(2):
            r = self._generate({**body, "new": {k: png64(im) for k, im in ims.items() if k not in known or attempt}})
            if isinstance(r, dict) and r.get("missing"): known.clear(); continue
            known.update(ims); break
        return [[{"text": c["text"], "finish": c["finish"], "logprobs": None} for c in cs] for cs in r]


class RemoteMMSampler(MMSampler):
    """A picture server on the other machine of a 2-node run (plearn/models/mm_server.py with --host 0.0.0.0): the MMSampler requests
    over HTTP; adapters are loaded from paths there (the trainer copies them when the machines share no disk: remote_copy). With
    PL_REMOTE_SERVE (JSON {url: {"host", "ssh", "repo", "env", "model", "gpus", "log"}}; two nodes, every GPU sampling
    and then every GPU training, as verl's colocated mode) this process starts the server over ssh with the local server's settings
    (launch) and stops it (stop), so that under --colocate the learner gets that machine's GPUs too; else its own launcher does."""
    def __init__(self, url, max_model_len=16384):
        self.base = url.rstrip("/"); self.port = int(self.base.rsplit(":", 1)[1]); self.proc = None; self.model = None
        self.max_model_len = int(os.environ.get("PL_MAX_MODEL_LEN", max_model_len)); self.adapter = None; self.n_loaded = 0
        self.log = subprocess.DEVNULL; self.ctl = json.loads(os.environ.get("PL_REMOTE_SERVE") or "{}").get(self.base)

    def remote_argv(self, mem):
        """The server's command on the other machine: the local MMSampler's settings, its model path, port and GPUs."""
        c = self.ctl; tp = len([g for g in str(c["gpus"]).split(",") if g.strip()])
        argv = ["python", "-m", "plearn.models.mm_server", "--model", c["model"], "--port", str(self.port), "--host", "0.0.0.0", "--mem", str(mem),
                "--max_model_len", str(self.max_model_len), "--max_lora_rank", str(c.get("max_lora_rank", 16)), "--max_images",
                os.environ.get("PL_MAX_IMAGES", "24"), "--tp", str(tp), "--max_num_seqs", os.environ.get("PL_MAX_NUM_SEQS", "64")]
        if os.environ.get("PL_PRESENCE_PENALTY"): argv += ["--presence_penalty", os.environ["PL_PRESENCE_PENALTY"]]
        if os.environ.get("PL_MM_SIZE"): argv += ["--mm_size", os.environ["PL_MM_SIZE"]]
        if os.environ.get("PL_MAX_BATCHED"): argv += ["--max_batched", os.environ["PL_MAX_BATCHED"]]
        return argv

    def _ssh(self, cmd, timeout):
        import shlex
        return subprocess.run(shlex.split(self.ctl.get("ssh", "ssh")) + [self.ctl["host"], cmd], timeout=timeout, capture_output=True, text=True)

    def launch(self, mem):
        """Start the server there in a session of its own (only the setsid command goes to the background, so ssh returns at once)."""
        import shlex
        c = self.ctl; log = c["log"]; q = shlex.quote
        used = self._ssh(f"nvidia-smi -i {c['gpus']} --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits", 60).stdout
        big = [l.strip() for l in used.splitlines() if l.count(",") >= 2 and float(l.split(",")[1]) > 2000]
        if big:                                                       # another process holds those GPUs (more than a CUDA context): a server left
            raise RuntimeError(f"the GPUs {c['gpus']} on {c['host']} are in use (pid, MiB, process): " + "; ".join(sorted(set(big))[:8]))   # from an earlier start, or another user
        cmd = (f"cd {q(c['repo'])}; source {q(c['env'])} >/dev/null 2>&1; mkdir -p {q(os.path.dirname(log))}; "
               f"CUDA_VISIBLE_DEVICES={c['gpus']} VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup {shlex.join(self.remote_argv(mem))} >> {q(log)} 2>&1 "
               f"< /dev/null & echo $! > {q(log + '.pid')}")
        r = self._ssh(cmd, 120)
        if r.returncode: raise RuntimeError(f"the picture server on {c['host']} did not start: {r.stderr[-500:]}")

    def wait(self, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                urllib.request.urlopen(self.base + "/health", timeout=5); return
            except Exception: time.sleep(5)
        raise RuntimeError(f"remote picture server {self.base} did not come up")

    def stop(self):
        """With PL_REMOTE_SERVE: stop the server's process group there, then wait until those GPUs are free (vLLM's workers keep their
        memory for some seconds after the server ends; then SIGKILL). Without it, its own launcher stops it."""
        if not self.ctl: return
        import shlex
        c = self.ctl; pid = shlex.quote(c["log"] + ".pid"); g = c["gpus"]
        free = f'[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {g} | sort -n | tail -1)" -lt 2000 ]'
        cmd = (f'p=$(cat {pid} 2>/dev/null); if [ -n "$p" ]; then kill -TERM -- -$p 2>/dev/null; '
               f'for i in $(seq 1 90); do {free} && break; sleep 2; done; {free} || kill -KILL -- -$p 2>/dev/null; sleep 3; fi; rm -f {pid}')
        self._ssh(cmd, 600)

