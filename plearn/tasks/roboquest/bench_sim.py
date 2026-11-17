"""Full-benchmark bridge (runs in the RoboQuest runtime interpreter, JSON lines on stdin/stdout).

Each session is one eval instance of one task in its own environment (no environment is shared between sessions), worked at
the decision level by the task's adapter in plearn/tasks/roboquest/bench. With render false the scene is built for physics only (text
version); with render true the adapter also returns the benchmark's three policy-camera images.
ops:  open {sid, task, iid, render, ep_dir?}  ->  {goal, intro, help, T, images}   (images: the start view, simulator version only)
      act {sid, line}                ->  {status, reading, images} | {invalid}
      submit {sid, answer}           ->  {success, outcome, errors, P, stages, deduction, flags} | {invalid}
      finish {sid}                   ->  {success: false, P, stages, deduction, flags}   (an episode that ended without SUBMIT)
      close {sid}
With ep_dir the episode keeps the benchmark runner's private files there and P is the benchmark's progress (bench/base.py).
For a demo scene (the test never sends it), open takes src "demos".
"""
from __future__ import annotations
import base64, io, json, os, sys
from pathlib import Path
PROTO, sys.stdout = sys.stdout, sys.stderr      # the simulator libraries print to stdout; replies keep the real one

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "benchmarks" / "roboquest"
sys.path[:0] = [str(ROOT), str(REPO)]
from roboquest.harness.egl import configure_egl
EGL = configure_egl(int(os.environ.get("RQ_GPU", "0")))           # before MuJoCo is imported
from PIL import Image
from roboquest import registry
from roboquest.kitchen import make_env
from roboquest.manifest import task_class
from plearn.tasks.roboquest.bench import adapter
from plearn.tasks.roboquest.bench.base import Invalid, SIZE

def _own_gl_first():
    """robosuite frees a scene's MuJoCo render context while ANOTHER scene's GL context is current, so it deletes the buffers of the
    scene that rendered last: that scene then returns the same stale picture for all 3 cameras until it ends (same 3 cameras,
    upside-down views). Any close during play in a process with more than one open scene triggers it. Make the context's own GL
    context current before each free."""
    from robosuite.utils import binding_utils as BU
    C = BU.MjRenderContext; old_del, old_size = C.__del__, C.update_offscreen_size
    def __del__(self):
        try: self.gl_ctx.make_current()
        except Exception: pass
        old_del(self)
    def update_offscreen_size(self, width, height):
        self.gl_ctx.make_current(); old_size(self, width, height)
    C.__del__, C.update_offscreen_size = __del__, update_offscreen_size
_own_gl_first()

SESS = {}                     # sid -> adapter


def png64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def images(arrs):
    return [png64(a) for a in arrs] if arrs is not None else None


def handle(req):
    op = req["op"]
    if op == "open":
        task, render = req["task"], bool(req.get("render"))
        inst = registry.load_instance(f"{ROOT / 'suite' / 'v1' / task / req.get('src', 'registry')}#{req['iid']}")
        env = make_env(task_class(task), inst, image_size=SIZE, gpu=EGL["egl_index"], render=render, camera_obs=False)
        env.reset()
        a = adapter(task)(env, render); SESS[req["sid"]] = a
        if req.get("ep_dir"):
            a.record(req["ep_dir"], task, req["iid"])
        return {"goal": a.goal(), "intro": a.intro(), "help": a.help(), "T": a.T, "images": images(a.views())}
    if op == "act":
        try:
            r = SESS[req["sid"]].act(req["line"])
        except Invalid as ex:
            return {"invalid": str(ex)}
        return dict(r, images=images(r.get("images")))
    if op == "submit":
        try:
            return SESS[req["sid"]].submit(req["answer"])
        except Invalid as ex:
            return {"invalid": str(ex)}
    if op == "finish":
        return SESS[req["sid"]].finish()
    if op == "close":
        a = SESS.pop(req["sid"], None)
        if a is not None:
            a.env.close()
        return True
    raise ValueError("unknown op " + op)


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            out = {"ok": True, "result": handle(json.loads(line))}
        except Exception:                             # report and keep serving
            import traceback; out = {"ok": False, "error": traceback.format_exc()[-3000:]}
        PROTO.write(json.dumps(out) + "\n"); PROTO.flush()
