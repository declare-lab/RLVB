"""A vLLM engine for prompts that carry images, behind a small HTTP endpoint.
Prompts are raw strings with the chat template already applied by the trainer (one <|vision_start|><|image_pad|><|vision_end|>
per image), so the sampler sees exactly what the trainer scores; images are named by content id in prompt order, and their
bytes (base64 PNG) come once, in "new", then stay in a cache here (a request that names an image the cache no longer holds
is answered with {"missing": true} and the client sends the bytes again). The LoRA adapter is loaded by path and used for
every request until the next /load.
The engine is vLLM's async engine: the requests of every HTTP call run at the same time and share the engine's batch with the
requests of other calls (the old offline engine ran one call at a time under a lock); tensor parallel over all visible GPUs
(--tp), prefix caching, and the image size of the trainer's processor (--mm_size, the lab's 16 image tokens for a 128x128 core).
usage: python -m plearn.models.mm_server --model DIR --port P [--mem 0.3] [--tp 2] [--mm_size JSON]
"""
# no "from __future__ import annotations": FastAPI must see the Request annotations of the handlers as classes
import argparse, asyncio, base64, io, json, os, uuid
from PIL import Image

CACHE, CACHE_MAX = {}, 20000      # images by content id (the client sends an image's bytes once); 128x128 cores are small
state = {"lora": None, "n": 0}


def image(b64): return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def too_long(ex):
    """Whether a vLLM error is a prompt over the context (for example VLLMValidationError "The decoder
    prompt (length 81962) is longer than the maximum model length of 81920"). One such prompt failed the whole /generate (asyncio.gather)."""
    m = str(ex); return "longer than the maximum model length" in m or "maximum context length" in m


def images_of(ids, new):
    """The request's images from the cache, after the new ones are decoded into it; None when one is unknown."""
    for k, b in new.items():
        if k not in CACHE:
            while len(CACHE) >= CACHE_MAX: CACHE.pop(next(iter(CACHE)))
            CACHE[k] = image(b)
    return None if any(k not in CACHE for k in ids) else [CACHE[k] for k in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0: the other machine of a 2-node run uses this server")
    ap.add_argument("--mem", type=float, default=0.3); ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_lora_rank", type=int, default=16); ap.add_argument("--max_images", type=int, default=24)
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--max_batched", type=int, default=0, help="--max-num-batched-tokens (0: vLLM's default)")
    ap.add_argument("--mm_size", default="", help='the image processor size, e.g. {"shortest_edge": 16384, "longest_edge": 147456}')
    ap.add_argument("--spec", default="", help="a speculative-decoding config (JSON), as PL_VLLM_SPEC")
    ap.add_argument("--presence_penalty", type=float, default=0.0,
                    help="vLLM's presence penalty on every sampled request (the model card's thinking setting is 1.5, "
                         "against endless repetition; PL_PRESENCE_PENALTY); 0: none, as before")
    args = ap.parse_args()

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    kw = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, enable_lora=True, max_lora_rank=args.max_lora_rank,
              max_loras=2, gpu_memory_utilization=args.mem, enable_prefix_caching=True, tensor_parallel_size=args.tp,
              max_num_seqs=args.max_num_seqs, limit_mm_per_prompt={"image": args.max_images, "video": 0},
              scheduling_policy="priority", disable_log_stats=False)   # a request's priority orders the queue (0 first: the episodes' turns)
    if args.max_batched: kw["max_num_batched_tokens"] = args.max_batched
    if os.environ.get("PL_VLLM_SEED"): kw["seed"] = int(os.environ["PL_VLLM_SEED"])   # the engine's seed (one per test seed); unset: vLLM's default seed 0, as before
    if args.mm_size: kw["mm_processor_kwargs"] = {"size": json.loads(args.mm_size)}
    if args.spec:                                               # node 1's server starts here directly: the same vLLM version check
        from plearn.models.sampler import check_spec; check_spec(args.spec); kw["speculative_config"] = json.loads(args.spec)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
    app = FastAPI()

    @app.on_event("startup")
    async def stats():                                          # vLLM's load line every 10 s (running, waiting, cache use, throughput)
        async def loop():
            while True:
                await asyncio.sleep(10)
                try: await engine.do_log_stats()
                except Exception: pass
        asyncio.get_running_loop().create_task(loop())

    @app.get("/health")
    async def health(): return {"ok": True}

    @app.post("/load")
    async def load(req: Request):
        body = await req.json(); state["n"] += 1
        state["lora"] = LoRARequest(f"lora{state['n']}", state["n"], body["path"])
        return {"ok": True, "adapter": state["n"]}

    async def one(x, sp, lora, prio, plen=False):
        final = None
        try:
            async for out in engine.generate(x, sp, request_id=uuid.uuid4().hex, lora_request=lora, priority=prio): final = out
        except Exception as ex:            # a prompt over the context: a cut, empty completion (finish "length"), as the text sampler;
            if not too_long(ex): raise     # the other requests of the call go on
            print(f"mm_server: prompt over the context, cut: {str(ex)[:200]}", flush=True)
            return args.max_model_len if plen else [{"text": "", "finish": "length"} for _ in range(sp.n)]
        if plen: return len(final.prompt_token_ids)                      # the prompt's tokens with its images (uncapped thinking)
        return [{"text": c.text, "finish": c.finish_reason} for c in final.outputs]

    @app.post("/generate")
    async def generate(req: Request):
        body = await req.json(); inputs = []
        for r in body["requests"]:
            x = {"prompt": r["prompt"]}
            if r.get("images"):
                ims = images_of(r["images"], body.get("new", {}))
                if ims is None: return JSONResponse({"missing": True})          # the client sends the bytes again
                x["multi_modal_data"] = {"image": ims}
            inputs.append(x)
        mt = int(body.get("max_tokens", 1024))
        def params(m):
            return SamplingParams(n=body.get("n", 1), max_tokens=m, temperature=body.get("temperature", 1.0),
                                  top_p=body.get("top_p", 1.0), top_k=body.get("top_k", -1), stop=body.get("stop") or None,
                                  presence_penalty=args.presence_penalty, skip_special_tokens=True, output_kind=RequestOutputKind.FINAL_ONLY)
        sp = params(max(1, mt))
        lora = state["lora"] if body.get("use_adapter", True) else None; prio = int(body.get("priority", 1))
        if sp.n > 1 or mt < 0:             # vLLM starts the n samples as n requests in one step, each with its own prefill of the prompt
            warm = SamplingParams(n=1, max_tokens=1, temperature=0.0, output_kind=RequestOutputKind.FINAL_ONLY)   # and its images:
            lens = await asyncio.gather(*(one(x, warm, lora, prio, plen=True) for x in inputs))   # one 1-token request first puts them in the prefix cache
        if mt < 0:                         # uncapped thinking (as the text sampler): the rest of the context, less
            sps = [params(max(1, args.max_model_len - n_ - int(os.environ.get("PL_TAIL_RESERVE", 512)))) for n_ in lens]   # PL_TAIL_RESERVE
        else: sps = [sp] * len(inputs)     # tokens for the belief line and the action
        return JSONResponse(await asyncio.gather(*(one(x, s, lora, prio) for x, s in zip(inputs, sps))))

    print(f"picture server up on port {args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
