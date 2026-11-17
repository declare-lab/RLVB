"""The pictures of the tabletop environment (teacher.py): ManiSkill3's table scene (SAPIEN, rendered off-screen with Vulkan), with
no robot. Each family has one shared scene in the process with a fixed set of object slots (lamps, buttons, cups, the objects
under them, blocks); a picture of an episode's state sets the slots' poses, colours and glow and renders the two cameras, so
nothing is built per episode. The slots an episode does not use stand far off the table. A lock keeps one picture at a time.
The renderer uses the process's GPU: the first of CUDA_VISIBLE_DEVICES, or in a torchrun rank the rank's own (LOCAL_RANK).

    render(family, objects) -> (left view, right view)    objects: {slot: (x, y, z, rgba, emission rgba or None)}
"""
from __future__ import annotations
import os, threading

SIZE = 512                                            # each camera's picture, as RoboQuest's cameras at 512 px
AWAY = (5.0, 5.0, -1.0)                               # where an unused slot stands
EYES = {"left": ([0.52, -0.30, 0.42], [-0.06, 0.0, 0.03]), "right": ([0.52, 0.30, 0.42], [-0.06, 0.0, 0.03])}
SLOTS = {   # family: {slot: (shape, size)}; box size = half extents, sphere = radius, cylinder = (radius, half length)
    "lamps": {**{f"bulb{i}": ("sphere", 0.032) for i in range(5)}, **{f"stand{i}": ("cylinder", (0.010, 0.050)) for i in range(5)},
              **{f"button{i}": ("box", (0.020, 0.020, 0.010)) for i in range(5)}, "panel": ("box", (0.040, 0.21, 0.004))},
    "cups": {**{f"cup{i}": ("cylinder", (0.042, 0.048)) for i in range(5)},
             **{f"obj{i}_{s}": (s, z) for i in range(5) for s, z in (("box", (0.018, 0.018, 0.018)), ("sphere", 0.020),
                                                                       ("cylinder", (0.018, 0.020)))}},
    "blocks": {**{f"block{i}": ("box", (0.025, 0.025, 0.025)) for i in range(4)},
               **{f"mark{j}": ("box", (0.002, 0.26, 0.0006)) for j in range(6)}},
}
_ENVS, _LOCK = {}, threading.Lock()


def _env(family):
    """The family's scene (built once in the process)."""
    if family in _ENVS: return _ENVS[family]
    import gymnasium as gym
    import numpy as np
    import sapien
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    import mani_skill.envs  # noqa: F401

    name = f"PLTabletop{family.capitalize()}-v1"

    class Scene(BaseEnv):
        SUPPORTED_ROBOTS = ["none"]
        def __init__(self, *a, **k): super().__init__(*a, robot_uids="none", **k)
        @property
        def _default_sensor_configs(self): return []
        @property
        def _default_human_render_camera_configs(self):
            return [CameraConfig(cam, sapien_utils.look_at(eye, at), SIZE, SIZE, 0.95, 0.01, 10) for cam, (eye, at) in EYES.items()]
        def _load_agent(self, options): super()._load_agent(options, sapien.Pose())
        def _load_scene(self, options):
            TableSceneBuilder(self).build(); self.slots = {}
            for slot, (shape, size) in SLOTS[family].items():
                b = self.scene.create_actor_builder(); mat = sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1.0], roughness=0.6)
                if shape == "box": b.add_box_visual(half_size=list(size), material=mat)
                elif shape == "sphere": b.add_sphere_visual(radius=size, material=mat)
                else: b.add_cylinder_visual(radius=size[0], half_length=size[1], material=mat)
                b.initial_pose = sapien.Pose(p=list(AWAY)); self.slots[slot] = (b.build_kinematic(name=slot), shape)
        def _initialize_episode(self, env_idx, options): pass
        def evaluate(self): return {}
        def _get_obs_extra(self, info): return {}
        def get_state_dict(self): return self.scene.get_sim_state()              # no robot: no controller state
        def set_state_dict(self, state, env_idx=None): self.scene.set_sim_state(state, env_idx)

    register_env(name, max_episode_steps=10 ** 9)(Scene)
    device = f"cuda:{os.environ['LOCAL_RANK']}" if "LOCAL_RANK" in os.environ else "gpu"
    env = gym.make(name, obs_mode="none", render_mode="all", num_envs=1, render_backend=device); env.reset(seed=0); u = env.unwrapped
    mats = {}
    for slot, (actor, _) in u.slots.items():
        mats[slot] = [shape.material for obj in actor._objs
                      for shape in obj.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes]
    _ENVS[family] = (env, u, mats, sapien, np)
    return _ENVS[family]


def render(family, objects):
    """The two camera pictures (PIL, left then right) of the family's scene with objects {slot: (x, y, z, rgba, emission)}; a
    cylinder stands upright (its axis vertical)."""
    from PIL import Image
    with _LOCK:
        env, u, mats, sapien, np = _env(family)
        upright = [0.7071068, 0.0, -0.7071068, 0.0]                  # SAPIEN's cylinders lie along x: a quarter turn about y
        for slot, (actor, shape) in u.slots.items():
            x, y, z, rgba, glow = objects.get(slot, (*AWAY, (0.5, 0.5, 0.5, 1.0), None))
            actor.set_pose(sapien.Pose(p=[x, y, z], q=upright if shape == "cylinder" else [1.0, 0.0, 0.0, 0.0]))
            for m in mats[slot]: m.set_base_color(list(rgba)); m.set_emission(list(glow) if glow else [0.0, 0.0, 0.0, 0.0])
        u.scene.update_render()
        pics = env.unwrapped.render_all() if hasattr(env.unwrapped, "render_all") else env.render()
        a = pics[0].cpu().numpy() if hasattr(pics, "cpu") else np.asarray(pics)[0]
        a = a.astype(np.uint8)
        return Image.fromarray(a[:, :SIZE]), Image.fromarray(a[:, SIZE:2 * SIZE])
