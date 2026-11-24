"""The full-benchmark test at the decision level: the eval50 instances of the ten RoboQuest tasks (bench/), in the
text version or the simulator version, scored by each task's own success check and by the benchmark's progress P (its own
code, roboquest/scoring/progress.py, on the private files each episode keeps in <out>/<tag>/episodes/). The protocol and how
it differs from the benchmark's: README.md beside this file.

The agent's turn is the trainer's: the thought, the belief line and the action are sampled from the policy, the model under
its sampler (temperature 0.6, top-p 0.95, top-k 20, the same as in training), with the trainer's system prompt; when the belief line holds the ACTION line instead, that line is the
action. The prompt holds the benchmark's goal, the adapter's intro and action
grammar, and every earlier action with its status and its reading (text version) or its status and, for the two newest
observations, their three camera images (simulator version; the benchmark's own memory, roboquest/harness/agent.py: the images
of the most recent two observations, the start view counted as the first). An evidence action uses one
of the task's T, an action the helpers cannot do also uses one; after T only SUBMIT is taken, and an episode without SUBMIT
after T + 2 decisions fails. Episodes run in parallel over several simulator bridges (plearn/tasks/roboquest/bench_sim.py).

    python -m plearn.tasks.roboquest.bench_eval --version text|sim --tag NAME [--init_lora PATH] [--tasks a,b] [--limit N] [--repeats R] [--gpu G]
"""
from __future__ import annotations
import argparse, base64, io, json, os, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from plearn.method import estimator as E
from plearn.method import trainer as TR
from plearn.tasks.roboquest.bench import TASKS
from plearn.tasks.roboquest.roboquest import SYSTEM, ANCHOR_HINT
from plearn.models.sampler import tag_prompt

RT = os.environ.get("ROBOQUEST_RUNTIME", "")                 # the RoboQuest runtime folder (benchmarks/roboquest/README.md); no default
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository
EVAL50 = os.path.join(REPO, "benchmarks", "roboquest", "suite", "v1", "eval50")
READING_RULE = ("\nHow to read the balance: the balance has a black pointer fixed to the beam. The pointer hangs below the pivot, in front of a small dial with five marks; the red mark in the middle means level. "
                "The pointer turns with the beam, so its lower end moves the opposite way from the lower pan: lower end to the left of the red mark means the RIGHT pan is lower; lower end to the right of the red mark means the LEFT pan is lower; "
                "lower end on the red mark means level (the pointer can then hide the red mark). The camera looks from an angle, so a level beam can look slanted and the two pans can look at different heights in the picture. "
                "Do not judge from the beam or the pans. Judge only from the pointer and the dial, shown in the zoom.")   # RQ_DIAL_ZOOM=1, Odd Parcel only: the reading rule and the angle warning, right after the task's own text
VIEWS = ("left camera", "right camera", "wrist camera", "zoom on the balance dial")   # the 4th picture only with RQ_DIAL_ZOOM=1 (Odd Parcel); zip stops at the pictures there are
KEEP_OBSERVATIONS = 2                                 # the benchmark keeps the images of the most recent two observations


class Bridge:
    """One simulator process; its sessions are worked one request at a time."""
    def __init__(self, gpu, log):
        if not RT: raise RuntimeError("set ROBOQUEST_RUNTIME to the RoboQuest runtime folder (benchmarks/roboquest/README.md)")
        env = {**os.environ, "MUJOCO_GL": "egl", "RQ_GPU": str(gpu), "OMP_NUM_THREADS": "1", "ROBOQUEST_RUNTIME": RT,
               "NUMBA_CACHE_DIR": RT + "/cache/numba"}
        self.p = subprocess.Popen([RT + "/envs/robocasa/bin/python", "-B", os.path.join(REPO, "plearn", "tasks", "roboquest", "bench_sim.py")],
                                  cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "a"),
                                  text=True, bufsize=1)
        self.lock = threading.Lock()

    def call(self, **req):
        with self.lock:
            self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
            r = json.loads(self.p.stdout.readline())
        if not r["ok"]:
            raise RuntimeError(r["error"])
        return r["result"]


def decode(images):
    return [Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB") for x in images] if images else None


class Episode:
    # final_submit (off by default; --task_args roboquest.final_submit=1): an episode that has used its T + 2 decisions
    # without a SUBMIT gets one more turn, whose prompt says that only SUBMIT is left (last_call). Without a SUBMIT in that turn
    # the episode ends as before (no_answer). Nothing changes for an episode that submits within T + 2 decisions.
    final_submit, last_call = False, False

    def __init__(self, task, iid, sid, bridge, rep=0, ep_dir=None):
        self.task, self.iid, self.sid, self.bridge, self.rep, self.ep_dir = task, iid, sid, bridge, rep, ep_dir
        self.log, self.turns, self.t, self.decisions, self.result = [], [], 0, 0, None

    def open(self, render):
        r = self.bridge.call(op="open", sid=self.sid, task=self.task, iid=self.iid, render=render, ep_dir=self.ep_dir)
        self.goal, self.intro, self.help, self.T, self.start = r["goal"], r["intro"], r["help"], r["T"], decode(r["images"])

    def content(self, version):
        """The user turn: text, and in the simulator version the kept images in order. Returns (parts, images)."""
        head = f"Task: {self.goal}\n{self.intro}\nActions:\n{self.help}"
        so_far = "\nEvidence so far:" if self.log else "\nEvidence so far: none yet"
        tail = f"\nEvidence actions used: {self.t} of {self.T}. Decide now."
        if self.last_call:                                             # final_submit: the one extra turn
            tail = (f"\nEvidence actions used: {self.t} of {self.T}. No evidence action is left and this is the last turn: "
                    "reply with ACTION: SUBMIT <answer>.")
        if version == "text":
            lines = [f"\n[{k}] {x['action']}: {' '.join(s for s in (x['status'], x['reading']) if s)}" for k, x in enumerate(self.log)]
            return head + so_far + "".join(lines) + tail, []
        seen = ([-1] if self.start else []) + [k for k, x in enumerate(self.log) if x["images"]]   # -1: the start view
        keep = set(seen[-KEEP_OBSERVATIONS:])                 # the observations that keep their images, as in the benchmark
        parts, images = [{"type": "text", "text": head}], []
        if self.start:
            parts.append({"type": "text", "text": "\nStart view:"})
            if -1 in keep:
                for cam, im in zip(VIEWS, self.start):
                    parts += [{"type": "text", "text": f" {cam}"}, {"type": "image"}]; images.append(im)
            else:
                parts.append({"type": "text", "text": " (images no longer shown)"})
        parts.append({"type": "text", "text": so_far})
        for k, x in enumerate(self.log):
            parts.append({"type": "text", "text": f"\n[{k}] {x['action']}: {x['status']}"})
            if k in keep:
                for cam, im in zip(VIEWS, x["images"]):
                    parts += [{"type": "text", "text": f" {cam}"}, {"type": "image"}]; images.append(im)
            elif x["images"]:
                parts.append({"type": "text", "text": " (images no longer shown)"})
        parts.append({"type": "text", "text": tail})
        return parts, images

    def step(self, action):
        """Carry out the agent's action line; returns True when the episode has ended."""
        self.decisions += 1
        t = action.strip()
        body = t[len("ACTION:"):].strip() if t.upper().startswith("ACTION:") else t
        if body.upper().startswith("SUBMIT"):
            r = self.bridge.call(op="submit", sid=self.sid, answer=body[len("SUBMIT"):].strip())
            self.result = (dict(self.bridge.call(op="finish", sid=self.sid), outcome="invalid_answer", errors=[r["invalid"]])
                           if "invalid" in r else r)
            return True
        if self.t >= self.T:
            self.log.append(dict(action=body, status="No evidence actions are left; only SUBMIT is possible.", reading="", images=None))
        else:
            self.t += 1
            r = self.bridge.call(op="act", sid=self.sid, line=body)
            if "invalid" in r:
                self.log.append(dict(action=body, status=f"The helpers cannot do this ({r['invalid']}).", reading="", images=None))
            else:
                self.log.append(dict(action=body, status=r["status"], reading=r["reading"], images=decode(r["images"])))
        if self.decisions >= self.T + 2:
            if self.final_submit and not self.last_call:
                self.last_call = True; return False                    # one more turn, for the SUBMIT alone
            self.result = dict(self.bridge.call(op="finish", sid=self.sid), outcome="no_answer",
                               errors=["no SUBMIT within T + 2 decisions"])
            return True
        return False

    def record(self, version, tag):
        return dict(task=self.task, iid=self.iid, rep=self.rep, version=version, tag=tag, success=bool(self.result["success"]),
                    P=self.result.get("P"), stages=self.result.get("stages"), deduction=self.result.get("deduction"),
                    outcome=self.result.get("outcome"), errors=self.result.get("errors", []), t=self.t, T=self.T,
                    decisions=self.decisions, actions=[dict(action=x["action"], status=x["status"], reading=x["reading"],
                                                            images=len(x["images"] or [])) for x in self.log],
                    turns=self.turns, intro=self.intro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=("text", "sim"), required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=os.environ.get("EVIDENCE_MODEL", "Qwen/Qwen3.5-4B"))
    ap.add_argument("--init_lora", default=""); ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=1, help="trials per instance (the thought is sampled, so trials differ)")
    ap.add_argument("--gpu", default="0", help="the GPU of the model and the sampler"); ap.add_argument("--render_gpu", default="")
    ap.add_argument("--bridges", type=int, default=4); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vllm_mem", type=float, default=0.6); ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95); ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_think", type=int, default=1024); ap.add_argument("--max_anchor", type=int, default=160)
    ap.add_argument("--max_action", type=int, default=200); ap.add_argument("--out", default="runs/evidence/bench")
    args = ap.parse_args()
    out = os.path.join(args.out, args.tag); os.makedirs(out, exist_ok=True)
    images = args.version == "sim"
    tok = AutoTokenizer.from_pretrained(args.model); proc = AutoProcessor.from_pretrained(args.model) if images else None
    # the policy: the base model with a LoRA (zero for the base model, or a trained one), loaded into the sampler
    TR.SAMPLER.update(T=args.temperature, top_p=args.top_p, top_k=args.top_k)
    sampler = TR.ServerSampler(args.model, args.gpu, 1.0, out, args.lora_r, args.vllm_mem, images=images)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    E.add_lora(model, r=args.lora_r)
    if args.init_lora:
        res = model.load_state_dict(torch.load(args.init_lora, map_location="cpu"), strict=False); assert not res.unexpected_keys
    sampler.sync(model); del model
    system = SYSTEM + ANCHOR_HINT

    def prompt(e):
        if images:
            parts, ims = e.content("sim")
            if e.task == "odd_parcel" and os.environ.get("RQ_DIAL_ZOOM") == "1": parts[0]["text"] += READING_RULE
            text = proc.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": parts}],
                                            tokenize=False, add_generation_prompt=True)
            return tag_prompt(text, ims)
        text, _ = e.content("text")
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)

    path = os.path.join(out, "episodes.jsonl")
    done = {(r["task"], r["iid"], r.get("rep", 0)) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    todo = [(t, i, k) for k in range(args.repeats) for t in args.tasks.split(",")
            for i in open(os.path.join(EVAL50, f"{t}.txt")).read().split()[:args.limit] if (t, i, k) not in done]
    bridges = [Bridge(args.render_gpu or args.gpu, os.path.join(out, "bridge.log")) for _ in range(args.bridges)]
    pool = ThreadPoolExecutor(len(bridges)); sink = open(path, "a"); active, sid, t0 = [], 0, time.time()
    print(f"{len(todo)} episodes to run ({len(done)} done before)", flush=True)
    while todo or active:
        new = []
        while todo and len(active) + len(new) < args.batch:
            task, iid, rep = todo.pop(0); sid += 1
            new.append(Episode(task, iid, sid, bridges[sid % len(bridges)], rep, os.path.join(out, "episodes", f"{task}-{iid}-r{rep}")))
        for e, err in zip(new, pool.map(lambda e: _try(e.open, images), new)):
            if err:
                print(f"OPEN FAILED {e.task} {e.iid}: {err[-300:]}", flush=True)
                sink.write(json.dumps(dict(task=e.task, iid=e.iid, version=args.version, tag=args.tag, success=False,
                                           outcome="build_failed", errors=[err[-500:]])) + "\n"); sink.flush()
            else:
                active.append(e)
        if not active:
            continue
        prompts = [prompt(e) for e in active]
        zs = [r[0] for r in sampler.sample(prompts, 1, args.max_think, stop_think=True)]
        pres = [p + z + TR.CLOSE for p, (z, _) in zip(prompts, zs)]
        anc = [TR.first_line(t) for (t, _), in sampler.sample(pres, 1, args.max_anchor, stop_think=False)]
        acts = [TR.first_line(t) for (t, _), in sampler.sample([p + a + TR.TERM for p, a in zip(pres, anc)], 1, args.max_action,
                                                                stop_think=False)]
        for k, (e, (z, stopped), a, s) in enumerate(zip(active, zs, anc, acts)):
            e.turns.append(dict(thought=z, stopped=bool(stopped), belief=a, action=s))
            if not s.upper().startswith("ACTION:") and a.upper().startswith("ACTION:"):
                acts[k] = a                                   # the two lines in the wrong order: take the ACTION line
        ended = list(pool.map(lambda x: _try(x[0].step, x[1]), zip(active, acts)))
        still = []
        for e, r in zip(active, ended):
            if isinstance(r, str):                            # the bridge failed on this action
                e.result = dict(success=False, outcome="bridge_error", errors=[r[-500:]]); r = True
            if r is True:
                sink.write(json.dumps(e.record(args.version, args.tag)) + "\n"); sink.flush()
                _try(lambda: e.bridge.call(op="close", sid=e.sid))
                print(f"{e.task} {e.iid} success={e.result['success']} t={e.t} ({time.time() - t0:.0f} s)", flush=True)
            else:
                still.append(e)
        active = still
    rows = [json.loads(x) for x in open(path)]

    def mean_p(rs):
        ps = [r["P"] for r in rs if r.get("P") is not None]
        return sum(ps) / len(ps) if ps else None
    summary = {t: {"n": len(rs), "success": sum(r["success"] for r in rs) / len(rs), "P": mean_p(rs),
                   "mean_t": sum(r.get("t", 0) for r in rs) / len(rs)} for t in args.tasks.split(",")
               for rs in [[r for r in rows if r["task"] == t]] if rs}
    summary["all"] = {"n": len(rows), "success": sum(r["success"] for r in rows) / max(1, len(rows)), "P": mean_p(rows)}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1); print("SUMMARY", json.dumps(summary), flush=True)
    sampler.stop()
    for b in bridges:
        b.p.terminate()


def _try(fn, *a):
    """fn(*a), or the error text when it fails."""
    try:
        return fn(*a)
    except Exception:
        import traceback; return traceback.format_exc()


if __name__ == "__main__":
    main()
