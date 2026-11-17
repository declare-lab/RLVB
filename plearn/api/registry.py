"""The registry: every task, group, model and training run is a YAML file (the style of lmms-eval).

    plearn/tasks/<dataset>/*.yaml      one task (task: NAME) or one group (group: NAME, task: [members]) per file
    configs/models/*.yaml       one model (model: NAME) per file
    configs/train/*.yaml        one training run (train: NAME) per file

A file may 'include' one or more templates (paths relative to the file); its own keys win and nested mappings are merged.
Files whose name starts with '_' are templates and are never registered. Strings may use ${NAME} or ${NAME:-default} for
environment variables; ${REPO} is the repository root. A task names its Python parts as 'module:attribute' with the module
relative to its own folder (for example 'expansion_env:Episode' in plearn/tasks/proworlds/ is plearn.tasks.proworlds.expansion_env.Episode), or
as a full dotted path. More roots (each with tasks/, models/ or train/) come from PLEARN_INCLUDE_PATH (os.pathsep-separated).
"""
from __future__ import annotations
import copy, importlib, os, re
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]                      # plearn/
REPO_ROOT = PKG_ROOT.parent
ROOTS = {"tasks": PKG_ROOT / "tasks", "models": REPO_ROOT / "configs" / "models", "train": REPO_ROOT / "configs" / "train"}
KEYS = {"tasks": "task", "models": "model", "train": "train"}
# ${NAME} or ${NAME:-default}; a default may itself hold one level of braces, e.g. ${DATA:-${REPO}/data}
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-((?:[^{}]|\{[^{}]*\})*))?\}")


class RegistryError(ValueError):
    pass


def merge(base, override):
    """Deep-merge two mappings; the values of override win."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def expand(value, env=None):
    """${NAME} and ${NAME:-default} in every string of value."""
    env = os.environ if env is None else env
    def sub(m):
        name, default = m.group(1), m.group(2)
        if name == "REPO": return str(REPO_ROOT)
        if env.get(name): return env[name]
        return expand(default, env) if default is not None else m.group(0)
    if isinstance(value, str): return _PLACEHOLDER.sub(sub, value)
    if isinstance(value, dict): return {k: expand(v, env) for k, v in value.items()}
    if isinstance(value, list): return [expand(v, env) for v in value]
    return value


def load_yaml(path, _stack=()):
    """One YAML file with its 'include' templates applied."""
    path = Path(path).resolve()
    if path in _stack: raise RegistryError(f"include cycle: {' -> '.join(map(str, _stack + (path,)))}")
    try: data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e: raise RegistryError(f"{path}: invalid YAML: {e}") from None
    if not isinstance(data, dict): raise RegistryError(f"{path}: the top level must be a mapping")
    inc = data.pop("include", None)
    if inc is None: return data
    base = {}
    for f in [inc] if isinstance(inc, str) else inc: base = merge(base, load_yaml(path.parent / f, _stack + (path,)))
    return merge(base, data)


def resolve(ref, source):
    """The Python object a task names: 'module:attr' (module relative to the task file's folder, or a full dotted path)."""
    mod, _, attr = ref.partition(":")
    if not attr: raise RegistryError(f"{source}: {ref!r} is not 'module:attribute'")
    folder = Path(source).resolve().parent; candidates = []
    while True:                                                       # the file's folder, then each folder above it to the repo
        try: rel = folder.relative_to(REPO_ROOT)
        except ValueError: break
        if not rel.parts: break
        candidates.append(".".join(rel.parts + (mod,))); folder = folder.parent
    if "." in mod: candidates.append(mod)
    for name in candidates:
        try: return getattr(importlib.import_module(name), attr)
        except ModuleNotFoundError as e:
            if e.name and not name.startswith(e.name): raise           # a missing dependency inside the module, not the module
    raise RegistryError(f"{source}: cannot import {ref!r} (tried {candidates})")


class Registry:
    def __init__(self, include_paths=()):
        roots = {k: [v] for k, v in ROOTS.items()}
        extra = [Path(p) for p in include_paths] + [Path(p) for p in os.environ.get("PLEARN_INCLUDE_PATH", "").split(os.pathsep) if p]
        for r in extra:
            for k in roots:
                if (r / k).is_dir(): roots[k].append(r / k)
        self.entries = {k: {} for k in roots}
        for kind, folders in roots.items():
            for folder in folders:
                for path in sorted(folder.rglob("*.yaml")) if folder.is_dir() else []:
                    if not path.name.startswith("_"): self._add(kind, path)

    def _add(self, kind, path):
        data = load_yaml(path)
        name = data.get("group") if kind == "tasks" and "group" in data else data.get(KEYS[kind])
        if not isinstance(name, str) or not name: raise RegistryError(f"{path}: no '{KEYS[kind]}' (or 'group') name")
        if name in self.entries[kind]:
            raise RegistryError(f"{path}: {name!r} is already defined in {self.entries[kind][name]['_source']}")
        data["_source"] = str(path); self.entries[kind][name] = data

    def names(self, kind="tasks", groups=None):
        out = list(self.entries[kind])
        if kind == "tasks" and groups is not None: out = [n for n in out if self.is_group(n) == groups]
        return out

    def is_group(self, name): return "group" in self.entries["tasks"].get(name, {})

    def get(self, kind, name):
        """One entry, with its environment placeholders expanded."""
        try: e = self.entries[kind][name]
        except KeyError:
            raise RegistryError(f"unknown {KEYS[kind]} {name!r}; registered: {', '.join(sorted(self.entries[kind])) or 'none'}") from None
        return expand(copy.deepcopy(e))

    def tasks(self, spec):
        """The task names of a comma list (or list) of tasks and groups, groups expanded in order, each task once."""
        names = [s.strip() for s in spec.split(",")] if isinstance(spec, str) else list(spec)
        out, seen = [], set()
        def visit(n, stack):
            if n in stack: raise RegistryError(f"group cycle: {' -> '.join(stack + (n,))}")
            e = self.get("tasks", n)
            if "group" in e:
                for m in e["task"]: visit(m, stack + (n,))
            elif n not in seen: seen.add(n); out.append(n)
        for n in filter(None, names): visit(n, ())
        if not out: raise RegistryError("no tasks selected")
        return out

    def groups_of(self, task):
        """Every group that holds the task, directly or through another group."""
        out = []
        for g in self.names("tasks", groups=True):
            try:
                if task in self.tasks(g): out.append(g)
            except RegistryError:
                pass
        return out


_DEFAULT = None


def registry():
    global _DEFAULT
    if _DEFAULT is None: _DEFAULT = Registry()
    return _DEFAULT
