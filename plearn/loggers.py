"""W&B for evaluation and training. The project, entity, run name and group come from the command line or the training file,
else from WANDB_PROJECT / WANDB_ENTITY; the key comes from the environment (WANDB_API_KEY or an earlier 'wandb login') and is
never written into the repository."""
from __future__ import annotations
import os


class WandbLogger:
    def __init__(self, project=None, entity=None, name=None, group=None, config=None, job_type="eval", run_id=None):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project or os.environ.get("WANDB_PROJECT"), entity=entity or os.environ.get("WANDB_ENTITY"),
                              name=name, group=group, config=config or {}, job_type=job_type, id=run_id, resume="allow" if run_id else None)

    def log_results(self, res, step=None):
        """Every task's and group's metrics as 'eval/<name>/<metric>', and one table of all of them."""
        flat, rows = {}, []
        for kind in ("groups", "results"):
            for name, r in res.get(kind, {}).items():
                for k, v in r.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool): flat[f"eval/{name}/{k}"] = v
                rows.append([("group" if kind == "groups" else "task"), name, r.get("episodes"), r.get("success"), r.get("errors")])
        flat["eval/table"] = self.wandb.Table(columns=["kind", "name", "episodes", "success", "errors"], data=rows)
        self.run.log(flat, step=step)

    def finish(self): self.run.finish()


def from_args(args, config=None, job_type="eval"):
    """A logger when --wandb_project (or WANDB_PROJECT) is set, else None."""
    project = getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT")
    if not project: return None
    return WandbLogger(project=project, entity=getattr(args, "wandb_entity", None), name=getattr(args, "wandb_name", None),
                       group=getattr(args, "wandb_group", None), config=config, job_type=job_type)
