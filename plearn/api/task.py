"""A registered task: its environment, its episodes for a split, and its metrics.

The task file (plearn/tasks/<dataset>/<task>.yaml) holds:
    task: NAME                     the registry name
    dataset: NAME                  the dataset the task belongs to (one folder per dataset)
    env: module:Class              the Episode class (see plearn/episode.py), env_args: its keyword arguments
    test_split, train_split        the splits for evaluation and for training (the environment's own names)
    num_episodes, seed             episodes of one evaluation (--limit overrides), the seed of their specs
    generation_kwargs              max_think, temperature, top_p, top_k, line_temperature, max_belief, max_action, carry
                                   (PL_TEST_T set: temperature and line_temperature of EVERY task = PL_TEST_T; test_temperature)
    metric_list                    [{metric, aggregation: mean|sum, higher_is_better}]: the values come from Episode.row(),
                                   or from process_results (module:function, row -> {metric: value})
    teacher: module:Class          optional: the training domain with the exact teacher (plearn/method/trainer.py's contract),
                                   teacher_args: its keyword arguments
    requires                       optional: {paths: [...], modules: [...]} that must exist before a run
    metadata                       {version: ...}
"""
from __future__ import annotations
import importlib.util, inspect, math, os
from dataclasses import dataclass, field

from plearn.api.registry import RegistryError, registry, resolve

GEN_DEFAULTS = {"max_think": 8192, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "line_temperature": 0.6, "max_belief": 160,
                "max_action": 40, "carry": True}
def test_temperature(g):
    """The generation settings with the test temperature of the environment: PL_TEST_T set (for example 1.0) gives temperature and
    line_temperature = PL_TEST_T for every task, over the task file's values. PL_TEST_T not set or empty: the settings do not change."""
    t = os.environ.get("PL_TEST_T", "").strip()
    return {**g, "temperature": float(t), "line_temperature": float(t)} if t else g


FIELDS = {"task", "dataset", "description", "env", "env_args", "test_split", "train_split", "num_episodes", "seed", "generation_kwargs",
          "metric_list", "process_results", "teacher", "teacher_args", "requires", "metadata", "_source"}


@dataclass
class Task:
    name: str
    dataset: str
    env: str
    source: str
    env_args: dict = field(default_factory=dict)
    test_split: str = "test"
    train_split: str = "train"
    num_episodes: int = 8
    seed: int = 0
    generation_kwargs: dict = field(default_factory=dict)
    metric_list: list = field(default_factory=list)
    process_results: str | None = None
    teacher: str | None = None
    teacher_args: dict = field(default_factory=dict)
    requires: dict = field(default_factory=dict)
    description: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name, reg=None):
        e = (reg or registry()).get("tasks", name)
        if "group" in e: raise RegistryError(f"{name!r} is a group, not a task")
        unknown = set(e) - FIELDS
        if unknown: raise RegistryError(f"{e['_source']}: unknown fields {sorted(unknown)}")
        for k in ("dataset", "env"):
            if k not in e: raise RegistryError(f"{e['_source']}: '{k}' is missing")
        return cls(name=name, dataset=e["dataset"], env=e["env"], source=e["_source"], env_args=e.get("env_args") or {},
                   test_split=e.get("test_split", "test"), train_split=e.get("train_split", "train"),
                   num_episodes=int(e.get("num_episodes", 8)), seed=int(e.get("seed", 0)),
                   generation_kwargs=test_temperature({**GEN_DEFAULTS, **(e.get("generation_kwargs") or {})}), metric_list=e.get("metric_list") or [],
                   process_results=e.get("process_results"), teacher=e.get("teacher"), teacher_args=e.get("teacher_args") or {},
                   requires=e.get("requires") or {}, description=e.get("description", ""), metadata=e.get("metadata") or {})

    # the environment
    def env_class(self): return resolve(self.env, self.source)
    def check(self):
        """The task's requirements (paths, Python modules) are there."""
        missing = [p for p in self.requires.get("paths", []) if not os.path.exists(p)]
        missing += [m for m in self.requires.get("modules", []) if importlib.util.find_spec(m) is None]
        if missing: raise RegistryError(f"task {self.name}: missing {missing}")
    def specs(self, split=None, n=None):
        """The episode specs of a split (default the test split): the first n of the environment's own order."""
        return list(self.env_class().specs(split or self.test_split, n or self.num_episodes, self.seed, **self.env_args))
    def episode(self, spec, out_dir=None):
        """One episode; an env that takes out_dir (files it keeps per episode) gets the evaluation's output folder."""
        cls = self.env_class()
        if out_dir and "out_dir" in inspect.signature(cls).parameters: return cls(spec, out_dir=out_dir, **self.env_args)
        return cls(spec, **self.env_args)

    # the scores
    def metric_values(self, row):
        if self.process_results: return resolve(self.process_results, self.source)(row)
        return {m["metric"]: row.get(m["metric"]) for m in self.metric_list}
    def score(self, rows):
        """Each metric's aggregation over the episodes; an episode with 'error' (a failure of the server or the environment, not
        of the agent) is left out and counted."""
        ok = [r for r in rows if "error" not in r]; vals = [self.metric_values(r) for r in ok]
        out = {"episodes": len(rows), "errors": len(rows) - len(ok)}
        for m in self.metric_list:
            xs = [v[m["metric"]] for v in vals if v.get(m["metric"]) is not None]
            agg = m.get("aggregation", "mean")
            if agg == "mean": out[m["metric"]] = sum(map(float, xs)) / len(xs) if xs else None
            elif agg == "sum": out[m["metric"]] = sum(map(float, xs))
            else: raise RegistryError(f"{self.source}: unknown aggregation {agg!r}")
            if agg == "mean" and xs and all(float(x) in (0.0, 1.0) for x in xs):      # a share: its standard error
                p = out[m["metric"]]; out[m["metric"] + "_stderr"] = math.sqrt(p * (1 - p) / len(xs))
        return out
