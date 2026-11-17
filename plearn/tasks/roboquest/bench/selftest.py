"""Self-test of a task adapter on eval50 instances (run with the RoboQuest runtime interpreter):

    python -m plearn.tasks.roboquest.bench.selftest TASK [N] [--render] [--out DIR] [--ids id1,id2]

For each instance: print the intro and the action grammar, run the adapter's privileged oracle_lines() and print every status
and reading, then submit oracle_answer() (the score must be a success, with the benchmark's progress P = 1); on a second
build of the same instance, submit oracle_answer(wrong=True) (the score must not be a success). Both episodes keep the
benchmark runner's private files in DIR/<instance>-oracle and DIR/<instance>-wrong (bench/base.py record()), from which P is
scored. With --render the images of every action are saved in DIR.
The oracle methods read hidden facts; they exist only for this test and are never shown to an agent.
"""
import argparse, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]; ROOT = REPO / "benchmarks" / "roboquest"
sys.path[:0] = [str(ROOT), str(REPO)]
from roboquest.harness.egl import configure_egl
EGL = configure_egl(int(os.environ.get("RQ_GPU", "0")))
from PIL import Image
from roboquest import registry
from roboquest.kitchen import make_env
from roboquest.manifest import task_class
from plearn.tasks.roboquest.bench import adapter
from plearn.tasks.roboquest.bench.base import Invalid, SIZE


def build(task, iid, render):
    inst = registry.load_instance(f"{ROOT / 'suite' / 'v1' / task / 'registry'}#{iid}")
    env = make_env(task_class(task), inst, image_size=SIZE, gpu=EGL["egl_index"], render=render, camera_obs=False)
    env.reset()
    return adapter(task)(env, render)


def save(imgs, out, stem):
    for j, im in enumerate(imgs or []):
        Image.fromarray(im).save(os.path.join(out, f"{stem}_cam{j}.jpg"), quality=85)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("n", type=int, nargs="?", default=3)
    ap.add_argument("--render", action="store_true"); ap.add_argument("--out", default="/tmp/bench_selftest")
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    ids = args.ids.split(",") if args.ids else open(ROOT / "suite" / "v1" / "eval50" / f"{args.task}.txt").read().split()[:args.n]
    os.makedirs(args.out, exist_ok=True); ok = bad = full = 0
    for iid in ids:
        t0 = time.time(); a = build(args.task, iid, args.render); a.record(os.path.join(args.out, f"{iid}-oracle"), args.task, iid)
        print(f"===== {args.task} {iid}  (build {time.time() - t0:.0f} s)")
        print("GOAL:", a.goal()); print("INTRO:", a.intro()); print("HELP:\n" + a.help())
        save(a.views(), args.out, f"{iid}_start")
        for k, line in enumerate(a.oracle_lines()):
            try:
                r = a.act(line)
            except Invalid as ex:
                print(f"  [{k}] {line} -> INVALID {ex}"); continue
            print(f"  [{k}] {line} -> {r['status']} | {r['reading']}"); save(r.get("images"), args.out, f"{iid}_a{k}")
        ans = a.oracle_answer()
        s = a.submit(ans); print(f"  SUBMIT {ans} -> {s}"); ok += bool(s["success"]); full += s.get("P") == 1
        a.env.close()
        b = build(args.task, iid, False); b.record(os.path.join(args.out, f"{iid}-wrong"), args.task, iid)
        for line in b.oracle_lines():
            try:
                b.act(line)
            except Invalid:
                pass
        wrong = b.oracle_answer(wrong=True); w = b.submit(wrong); print(f"  SUBMIT {wrong} (wrong) -> {w}")
        bad += not w["success"]; b.env.close()
    print(f"RESULT {args.task}: correct answers scored success {ok}/{len(ids)} (P = 1: {full}/{len(ids)}); "
          f"wrong answers scored failure {bad}/{len(ids)}")


if __name__ == "__main__":
    main()
