"""A training run from its file (configs/train/<name>.yaml, or any path): the model, the training tasks, the tasks of the
evaluation after every phase, the GPUs, the method's settings and W&B. It starts the trainer (plearn/method/trainer.py) as an actor
(the picture server: sample sets, E-step, evaluations) and a learner (the M-step on its ranks), and stops both when one fails.

    train: NAME
    model: qwen3-vl-8b-thinking        a registered model or a weights folder
    train_tasks: train_suite           tasks and groups with a teacher (plearn/tasks/mixture.py)
    eval_tasks: [train_suite]          tasks and groups evaluated at update 0 and after every phase, on method.eval_split (validation)
    eval_n: 0                          episodes of each evaluated task (0: the task's own number)
    watch: {proworlds: 4, ...}         at every evaluation, the first n episodes of each task's own test list (the benchmarks, a
                                       sample by task), to watch the direction of training; never reported
    output: ${PL_RUNS}/NAME            the run's folder: logs/, queue-<tag>/, case-<tag>/ (rollouts, evaluations), config.json
    env_sh: ${PL_ENV_SH:-/dev/null}    a file to source first (for example a conda environment)
    gpus: {actor: "1,4,6,7", learner: "1,4,6,7", colocate: true}
    method: {...}                      the trainer's flags (name: value; true is a bare flag)
    domain_args: {...}                 more arguments for the teacher (e.g. ref_pictures)
    server: {...}                      the picture server's settings (PL_* variables)
    wandb: {project, entity, group, name}  name: default NAME-MMDD-HHMM, the start's day and time (runs must be
                                       told apart); a resumed run adds -rR
    nodes: [{host, ssh, addr, repo, env_sh, model, gpus, runs, server_port}]   another machine (for example 2 nodes x 8 GPUs, every
                                       GPU sampling and then every GPU training, as verl's colocated mode): its learner ranks join the
                                       learner (torchrun over the machines), and its picture server, which the actor starts and stops over
                                       ssh, takes requests beside the local one; adapters are copied there (no shared disk)
    master: {addr, port}               this machine's address for the learner's rendezvous (with nodes)
    ifname: eth0                       the network interface of NCCL and gloo between the machines (with nodes)
"""
from __future__ import annotations
import json, os, shlex, signal, subprocess, sys, time
from pathlib import Path

from plearn.api.registry import REPO_ROOT, expand, load_yaml, registry

METHOD = {"objective": "critic", "proposal": "base", "phase": 25, "alpha": 1, "rho": "fidelity", "line_floor": 50,
          "line_score": "decision", "line_mix": 0.1,   # the belief line scored by the act it implies, plus 0.1 of its log score
          "carry": True, "episodes": 1, "worlds": 4, "G": 1, "excite": 0.5, "rho_rb": 1, "B": 32, "em_k": 8, "em_m": 4,
          "em_j": 32,   # the updates are fast next to the sample sets, so the M-step fits q more completely. On two measured
                        # sample sets, 8 draws fit about 6 turns per history at a total variation of 0.25-0.38 from q,
                        # 32 draws 10-15 turns at 0.06-0.09 (q's effective size is 9-15 turns)
          "em_more": 1, "m_step": "blocks", "think_chunk": 1, "phase_set": 300, "set_draw": "passes", "set_prio": 0.5,
          "set_keep": 200,   # dynamic sampling (as DAPO's): the batches only from the first 200 histories of the set that
                             # give a gradient, an own turn with an accepted action; the set draws 1.5 times that (the 4B study: 48 for 32),
                             # as 80 percent of the measured dumped histories had an accepted own turn
          "accept_only": 1, "em_more_joint": 2.0, "leave_out": 0,   # weights only on turns with an accepted action; a history without one
                                         # that has a line within 2 nats of the exact line gets one more round (16 reasonings in all), then its
                                         # best turns are fitted: no history is left out
          "dec_term": "own",   # fix A: the decision target is the turn as the model wrote it (trainer turn_item)
          "max_think": 16384, "eval_max_think": 8192,   # a larger reasoning cap in training; the evaluation keeps the tests' 8,192
          "think_cost": "0",   # the tie-break among alike samples is part of the length reward now
          "length_reward": "4",   # correct reasoning within the test's budget; the shorter correct thought gets more, a
                                  # wrong turn no length reward (a long one a penalty), from the first phase. On two measured sample
                                  # sets, C = 4 moves the E-step weight on correct turns that end within 8,192 tokens (em_q_budget)
                                  # from 0.34 to 0.42 and from 0.17 to 0.22; the weight on correct turns falls by 0.1-0.2 points
          "steps": 100, "eval_every": 25, "held_n": 40, "eval_episodes": 0, "eval_split": "validation", "policy_metrics": 0, "seed": 0,
          "value": "log", "tau": 0.1,   # the teacher's value: the average log accuracy, with its temperature in nats
          "readout": 0, "dump_em_steps": "1,25,50,100",
          "lr": 1e-5, "weight_decay": 0.01,   # one LoRA adapter for every method, at the usual rate of RL:
                                              # LoRA at 10x the full-parameter rate of RL fine-tuning (1e-6: DeepSeekMath's GRPO, DAPO), as verl's
                                              # LoRA guide (docs/advance/ppo_lora.rst) and LoRA Without Regret advise (TRL's LoRA runs: 1e-5); AdamW
                                              # weight decay 0.01 (verl's optimizer default)
          "kl_stop": 0.015, "kl_probe": 32}   # a phase stops when it has moved 0.015 nats per token from its prior (PPO's
                                              # early stop of its epochs at 1.5 x its target KL 0.01, as OpenAI's Spinning Up)
SERVER = {"PL_VLLM_TP": 1, "PL_ASYNC_EPISODES": 1, "PL_PLAN_STREAM": 1, "PL_CHUNKED_HEAD": 1, "PL_MAX_NUM_SEQS": 512, "PL_MAX_BATCHED": 65536,
          "PL_HTTP_WORKERS": 64, "PL_MAX_IMAGES": 128}


def load(config):
    """A run's settings: a registered training run or a YAML path, environment placeholders expanded."""
    if os.path.exists(config): c = load_yaml(config)
    else: c = registry().get("train", config)
    return expand(c)


def flags(d):
    out = []
    for k, v in d.items():
        if v is True: out.append(f"--{k}")
        elif v is False or v is None: continue
        else: out += [f"--{k}", str(v) if not isinstance(v, (list, dict)) else json.dumps(v)]
    return out


def commands(c, resume_step=0, stamp=None):
    """The learner's and the actor's commands, their environment and the run's paths. stamp: the start's MMDD-HHMM in the W&B
    run name (default: now)."""
    from plearn import models
    name = c["train"]; out = Path(c.get("output") or os.path.join(os.environ.get("PL_RUNS", "runs"), name)); tag = c.get("tag", name)
    queue = out / f"queue-{tag}"; model, _ = models.resolve(c["model"])
    g = c.get("gpus") or {}; actor_gpus, learner_gpus = str(g.get("actor", "0")), str(g.get("learner", g.get("actor", "0")))
    colocate = bool(g.get("colocate", actor_gpus == learner_gpus)); n_learner = len(learner_gpus.split(","))
    method = {**METHOD, **(c.get("method") or {})}
    nodes = c.get("nodes") or []; nn = 1 + len(nodes)                  # another machine: learner ranks and a picture server
    assert len(nodes) <= 1, "one other machine at most (the adapter copy and the server control know one)"
    tasks = registry().tasks(c["train_tasks"]); evals = c.get("eval_tasks") or []
    from plearn.tasks.mixture import teachers_of
    teachers = teachers_of(tasks)                                    # one teacher class, or several (a mixed run)
    dargs = {"tasks": tasks, "images": int(any(getattr(t, "images", False) for t, _ in teachers)), **(c.get("domain_args") or {})}
    common = ["--domain", "plearn.tasks.mixture", "--domain_args", json.dumps(dargs), "--model", model, "--tag", tag, "--out", str(out),
              "--queue", str(queue), "--colocate", str(int(colocate)), "--resume_step", str(resume_step),
              "--eval_tasks", ",".join(evals) if isinstance(evals, list) else str(evals), "--eval_n", str(c.get("eval_n", 0)),
              "--watch", json.dumps(c.get("watch") or {}) if c.get("watch") else "",
              "--dump_em", str(out / f"case-{tag}" / "em"), "--dump_rollouts", str(out / f"case-{tag}" / "rollouts")] + flags(method)
    w = c.get("wandb") or {}
    wname = w.get("name") or f"{tag}-{stamp or time.strftime('%m%d-%H%M')}" + (f"-r{resume_step}" if resume_step else "")
    common += flags({"wandb_project": w.get("project"), "wandb_entity": w.get("entity"), "wandb_group": w.get("group"), "wandb_name": wname})
    env = {k: str(v) for k, v in {**SERVER, **(c.get("server") or {})}.items()}
    think = int(method["max_think"])                                 # unlimited reasoning (max_think < 0): the context alone bounds it, the
    env.setdefault("PL_MAX_MODEL_LEN", str(think + 40960 if think > 0 else 81920))   # verl runs' prompt and response lengths together
    if nodes:                                                        # the actor's view of the other machine, and the network between them
        k_ = int(env.get("PL_GPUS_PER_SERVER") or 0); urls = {}             # speed: k GPUs a server on the other machine too
        for nd in nodes:
            gs = [g.strip() for g in str(nd["gpus"]).split(",") if g.strip()]
            for i, b in enumerate(range(0, len(gs), k_ or len(gs))):
                urls[f"http://{nd['addr']}:{int(nd['server_port']) + i}"] = {**nd, "gpus": ",".join(gs[b:b + (k_ or len(gs))]), "log_name": f"mm_server-{i}.log" if k_ else "mm_server.log"}
        env["PL_VLLM_REMOTE"] = ",".join(urls)
        env["PL_REMOTE_SERVE"] = json.dumps({u: {"host": nd["host"], "ssh": nd["ssh"], "repo": nd["repo"], "env": nd["env_sh"], "model": nd["model"],
                                                  "gpus": str(nd["gpus"]), "log": f"{nd['runs']}/{name}/logs/{nd['log_name']}"} for u, nd in urls.items()})
        env["PL_REMOTE_SYNC"] = f"{nodes[0]['host']}:{nodes[0]['runs']}/{name}/adapters"; env["PL_REMOTE_SSH"] = nodes[0]["ssh"]
        env.update({"NCCL_SOCKET_IFNAME": str(c.get("ifname", "eth0")), "GLOO_SOCKET_IFNAME": str(c.get("ifname", "eth0")), "NCCL_IB_DISABLE": "1"})
    if nodes:                                                        # the learner over the machines: rank 0 here, rendezvous here
        mst = c["master"]; rdzv = ["--nnodes", str(nn), "--master_addr", str(mst["addr"]), "--master_port", str(mst["port"])]
        lead = rdzv + ["--node_rank", "0"]
    else: lead = ["--standalone"]
    learner = [sys.executable, "-m", "torch.distributed.run"] + lead + ["--nproc_per_node", str(n_learner), "-m", "plearn.method.trainer",
               "--role", "learner"] + common
    remote = []
    for i, nd in enumerate(nodes):                                   # the other machine's ranks: the same flags, its own model path
        argv = ["python", "-m", "torch.distributed.run"] + rdzv + ["--node_rank", str(i + 1), "--nproc_per_node", str(len(str(nd["gpus"]).split(","))),
                "-m", "plearn.method.trainer", "--role", "learner"] + with_flag(common, "--model", nd["model"])
        remote.append({**nd, "argv": argv, "log": f"{nd['runs']}/{name}/logs/{tag}-node{i + 1}.log"})
    mem = (c.get("server") or {}).get("vllm_mem", 0.85 if colocate or not set(actor_gpus.split(",")) & set(learner_gpus.split(",")) else 0.3)
    actor = [sys.executable, "-m", "plearn.method.trainer", "--role", "actor"] + common + ["--vllm_gpu", actor_gpus, "--vllm_mem", str(mem),
                                                                                      "--ahead", "3", "--gen_depth", "2"]
    return {"out": out, "tag": tag, "queue": queue, "learner": learner, "actor": actor, "env": env, "actor_gpus": actor_gpus,
            "learner_gpus": learner_gpus, "colocate": colocate, "n_learner": n_learner, "remote": remote}


def with_flag(argv, flag, value):
    """argv with the value after flag replaced."""
    out = list(argv); out[out.index(flag) + 1] = value; return out


def remote_cmd(nd, env, argv, log):
    """The shell command that starts argv on the other machine in a session of its own (only the setsid command goes to the
    background, so ssh returns at once), with its environment file, the run's variables and its pid beside the log."""
    q = shlex.quote; envs = " ".join(f"{k}={q(str(v))}" for k, v in env.items())
    return (f"cd {q(nd['repo'])}; source {q(nd['env_sh'])} >/dev/null 2>&1; mkdir -p {q(os.path.dirname(log))}; "
            f"{envs} setsid nohup {shlex.join(argv)} > {q(log)} 2>&1 < /dev/null & echo $! > {q(log + '.pid')}")


def remote_alive(nd):
    """Whether the other machine's learner still runs (its process group)."""
    q = shlex.quote
    r = subprocess.run(shlex.split(nd["ssh"]) + [nd["host"], f"kill -0 -- -$(cat {q(nd['log'] + '.pid')}) 2>/dev/null"], capture_output=True, timeout=60)
    return r.returncode == 0


def remote_stop(nd, env):
    """Stop the other machine's learner (its process group) and its picture server (the actor stops the server itself; this is
    for a failure)."""
    q = shlex.quote
    subprocess.run(shlex.split(nd["ssh"]) + [nd["host"], f"p=$(cat {q(nd['log'] + '.pid')} 2>/dev/null); [ -n \"$p\" ] && kill -TERM -- -$p 2>/dev/null; true"],
                   capture_output=True, timeout=120)
    os.environ["PL_REMOTE_SERVE"] = env.get("PL_REMOTE_SERVE", "{}")
    from plearn.models.sampler import RemoteMMSampler
    for u in [x for x in env.get("PL_VLLM_REMOTE", "").split(",") if x]: RemoteMMSampler(u).stop()


def prepare_queue(r, resume_step, phase):
    """A fresh run: a new queue. A resumed run (after the phase end resume_step): the batches stay, the stopped processes' flags go,
    and a set whose batches the learner had begun is made again."""
    q = r["queue"]
    if resume_step:
        for f in list(q.glob("learner_parked_*")) + [q / "actor_failed", q / "learner_failed", q / "actor_done"]:
            if f.exists(): f.unlink()
        if not all((q / f"batch_{u}.pkl").exists() for u in range(resume_step + 1, resume_step + phase + 1)):
            (q / f"gen_done_{resume_step}").unlink(missing_ok=True)
        (q / "learner_step").write_text(str(resume_step))
    else:
        import shutil
        shutil.rmtree(q, ignore_errors=True)
    q.mkdir(parents=True, exist_ok=True)


def run(config, resume_step=0, dry_run=False):
    c = load(config)
    r = commands(c, resume_step); out = r["out"]; (out / "logs").mkdir(parents=True, exist_ok=True)
    logt = r["tag"] + (f"-r{resume_step}" if resume_step else "")
    json.dump({"config": c, "learner": r["learner"], "actor": r["actor"], "env": r["env"]}, open(out / f"config-{logt}.json", "w"), indent=1)
    if dry_run:
        print("learner:", shlex.join(r["learner"])); print("actor:", shlex.join(r["actor"])); print("env:", r["env"]); return 0
    prepare_queue(r, resume_step, int({**METHOD, **(c.get("method") or {})}["phase"]))
    env = {**os.environ, **r["env"]}
    if c.get("env_sh"):                                              # the server's profile (conda env, paths) under this process's env
        sh = str(c["env_sh"]) if os.path.isabs(str(c["env_sh"])) else str(REPO_ROOT / c["env_sh"])
        dump = subprocess.run(["bash", "-c", f"source {shlex.quote(sh)} >/dev/null 2>&1; env -0"], capture_output=True).stdout
        env.update(dict(l.split("=", 1) for l in dump.decode().split("\0") if "=" in l)); env.update(r["env"])
    py = env.get("EVIDENCE_PYTHON") or shutil_which("python", env) or sys.executable
    r["learner"][0] = r["actor"][0] = py
    print(f"{time.strftime('%c')} {c['train']}: model {c['model']}, train {c['train_tasks']}, eval {c.get('eval_tasks')} -> {out}/logs/{logt}.log", flush=True)
    lf = open(out / "logs" / f"{logt}.log", "a"); af = open(out / "logs" / f"{logt}-actor.log", "a")
    for nd in r["remote"]:                                            # the other machine's learner ranks first: the rendezvous waits for them
        cmd = remote_cmd(nd, {**r["env"], "CUDA_VISIBLE_DEVICES": str(nd["gpus"]), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, nd["argv"], nd["log"])
        subprocess.run(shlex.split(nd["ssh"]) + [nd["host"], cmd], check=True, timeout=120)
        print(f"{time.strftime('%c')} learner ranks on {nd['host']} ({nd['gpus']}) -> {nd['log']}", flush=True)
    learner = subprocess.Popen(r["learner"], cwd=REPO_ROOT, env={**env, "CUDA_VISIBLE_DEVICES": r["learner_gpus"],
                               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if set(r["actor_gpus"].split(",")) & set(r["learner_gpus"].split(",")):   # shared GPUs: the learner's models first
        for _ in range(360):
            if (out / "logs" / f"{logt}.log").read_text(errors="ignore").count("lora layers") >= r["n_learner"]: break
            if learner.poll() is not None: print("the learner stopped before its models were loaded", flush=True); return 4
            time.sleep(5)
    actor = subprocess.Popen(r["actor"], cwd=REPO_ROOT, env={**env, "CUDA_VISIBLE_DEVICES": r["actor_gpus"].split(",")[0]},
                             stdout=af, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    remote_down, k = False, 0
    while learner.poll() is None and actor.poll() is None:
        time.sleep(10); k += 1
        if r["remote"] and k % 6 == 0 and not all(remote_alive(nd) for nd in r["remote"]):   # the other machine's ranks ended: NCCL would
            remote_down = True; break                                                      # wait for them for hours
    first = learner if learner.poll() is not None else actor
    if first.returncode or remote_down:                              # one side failed: stop the other (it would wait for ever)
        print(f"{time.strftime('%c')} " + ("the other machine's learner ranks ended" if remote_down else f"a process failed (exit {first.returncode})")
              + "; stopping the rest", flush=True)
        for p in (learner, actor):
            if p.poll() is None: p.send_signal(signal.SIGTERM)
    for p in (learner, actor): p.wait()
    for nd in r["remote"]: remote_stop(nd, r["env"])                  # the other machine's ranks and picture server are not left behind
    print(f"{time.strftime('%c')} done", flush=True)
    return first.returncode or (5 if remote_down else 0)


def shutil_which(cmd, env):
    import shutil
    return shutil.which(cmd, path=env.get("PATH"))
