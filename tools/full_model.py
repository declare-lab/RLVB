"""A full model from a base model and a LoRA adapter, and its Hugging Face upload, in the form of the published full models
(posterior-learning/grpo-qwen3-vl-8b-thinking, posterior-learning/pl-qwen3-vl-8b-thinking, ...): W <- W + (lora_alpha / r) * B @ A for every LoRA module,
computed in fp32 and stored in the base dtype; every other tensor byte-identical to the base; the non-weight files (config,
tokenizer, chat template, processor and generation configs) the base files unchanged, the base README as README.base.md; a model
card README.md and MERGE-REPORT.json with three checks:
  check_a  3 random merged modules within bf16 rounding of W + s B A (max |diff| / half an ulp at that size < 1)
  check_b  every other tensor byte-identical to the base
  check_c  total tensor bytes = the base's (and the same tensor names, shapes and dtypes)
The adapter: PEFT layout (plearn's export_lora, verl's merger lora_adapter/): base_model.model.<module>.lora_A.weight [r, in] and
.lora_B.weight [out, r]; <module>.weight must be a base tensor of shape [out, in] (no other mapping is tried: a miss stops).

    python tools/full_model.py merge --base DIR --adapter DIR --out DIR --repo ORG/NAME --base_id Qwen/Qwen3.5-4B --about TEXT [--dry]
    python tools/full_model.py upload --dir DIR --repo ORG/NAME [--private] [--yes]

merge --dry: the key map, the shapes and the card text only (no tensor is merged, nothing is written). upload without --yes: the
file list and sizes only. upload --yes: HfApi.upload_folder with HF_HUB_DISABLE_XET=1 (hf_xet uploads can hang), then the file
list on the Hub is compared with the folder (names and sizes). Qwen3.5 (qwen3_5): the LoRA modules are the language model's 200
projections (attention, MLP, gated-delta-net in_proj_qkv / in_proj_z / out_proj); the vision tower and the MTP head (mtp.*) are
copied as they are; the embeddings are tied (no lm_head tensor), and LoRA does not touch them."""
from __future__ import annotations
import argparse, hashlib, json, os, random, shutil, sys, time

PREFIX = "base_model.model."


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def base_index(base):
    """{tensor name: shard file} of the base (one file or an index)."""
    idx = os.path.join(base, "model.safetensors.index.json")
    if os.path.exists(idx): return json.load(open(idx))["weight_map"]
    from safetensors import safe_open
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as f: return {k: "model.safetensors" for k in f.keys()}


def adapter_map(adapter, wmap):
    """(r, alpha, {base tensor: (A key, B key)}) of a PEFT adapter; stops on a key that is not a base tensor."""
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json"))); r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    from safetensors import safe_open
    with safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt") as f: keys = list(f.keys())
    mods, bad = {}, []
    for k in keys:
        if not (k.startswith(PREFIX) and k.endswith((".lora_A.weight", ".lora_B.weight"))): bad.append(k); continue
        m = k[len(PREFIX):].rsplit(".lora_", 1)[0]; mods.setdefault(m + ".weight", {})[k.rsplit(".", 2)[-2]] = k
    bad += [w for w, ab in mods.items() if w not in wmap or set(ab) != {"lora_A", "lora_B"}]
    if bad: raise SystemExit(f"{len(bad)} adapter keys or modules do not map to the base (first: {bad[:3]})")
    return r, alpha, {w: (ab["lora_A"], ab["lora_B"]) for w, ab in sorted(mods.items())}


def card(repo, base_id, about, r, alpha, n_merged, n_other, total, adapter_sha):
    name = repo.split("/")[-1]
    return f"""---
base_model: {base_id}
library_name: transformers
tags: [merged-lora, posterior-learning]
---
# {name}

Full model = base **[{base_id}](https://huggingface.co/{base_id})** with the LoRA adapter merged in.

- Adapter: the run's own LoRA; sha256 `{adapter_sha}`
- {about}
- LoRA r {r}, lora_alpha {alpha:g}; merge rule for every LoRA module: `W <- W + (lora_alpha / r) * B @ A` (scale {alpha / r}), computed in fp32, stored in bf16.
- Merged modules: {n_merged}; all other tensors are byte-identical to the base. Non-weight files (config, tokenizer, chat template, processor configs, generation config) are the base files, unchanged; the base README is `README.base.md`.
- Merged in bf16: outputs can differ slightly from base + adapter at run time.

Checks before upload (MERGE-REPORT.json): 3 random modules within bf16 rounding of W + s*B@A; {n_other} other tensors byte-identical; total tensor bytes = base ({total}).
"""


def half_ulp(x):
    """Half the bf16 spacing at |x| (elementwise, fp32)."""
    import torch
    e = torch.floor(torch.log2(x.abs().clamp_min(torch.finfo(torch.float32).tiny)))
    return torch.pow(2.0, e - 7) / 2 * 1.0001


def merge(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    t0 = time.time(); wmap = base_index(a.base); r, alpha, mods = adapter_map(a.adapter, wmap); s = alpha / r
    ad = safe_open(os.path.join(a.adapter, "adapter_model.safetensors"), "pt")
    shapes = {}
    for shard in sorted(set(wmap.values())):
        with safe_open(os.path.join(a.base, shard), "pt") as f:
            for k in f.keys(): shapes[k] = tuple(f.get_slice(k).get_shape())
    bad = [w for w, (ka, kb) in mods.items()
           if tuple(ad.get_slice(ka).get_shape())[0] != r or (tuple(ad.get_slice(kb).get_shape())[0], tuple(ad.get_slice(ka).get_shape())[1]) != shapes[w]]
    if bad: raise SystemExit(f"{len(bad)} modules: adapter shapes do not fit the base (first: {bad[:3]})")
    other = [k for k in wmap if k not in mods]
    print(json.dumps({"base": a.base, "adapter": a.adapter, "r": r, "lora_alpha": alpha, "scale": s, "merged_modules": len(mods),
                      "other_tensors": len(other), "shards": sorted(set(wmap.values())), "out": a.out}, indent=1))
    asha = sha256(os.path.join(a.adapter, "adapter_model.safetensors"))
    if a.dry:
        print(card(a.repo, a.base_id, a.about, r, alpha, len(mods), len(other), "<the base's total tensor bytes>", asha))
        print("DRY: nothing merged or written"); return 0
    if os.path.exists(a.out): raise SystemExit(f"{a.out} exists (a new folder only)")
    os.makedirs(a.out)
    for fn in sorted(os.listdir(a.base)):                       # the non-weight files, unchanged
        p = os.path.join(a.base, fn)
        if not os.path.isfile(p) or fn.endswith(".safetensors"): continue
        shutil.copy2(p, os.path.join(a.out, "README.base.md" if fn == "README.md" else fn))
    base_bytes = out_bytes = 0
    for shard in sorted(set(wmap.values())):
        out = {}
        with safe_open(os.path.join(a.base, shard), "pt") as f:
            meta = f.metadata() or {"format": "pt"}
            for k in f.keys():
                w = f.get_tensor(k); base_bytes += w.numel() * w.element_size()
                if k in mods:
                    A = ad.get_tensor(mods[k][0]).float(); B = ad.get_tensor(mods[k][1]).float()
                    w = (w.float() + s * (B @ A)).to(w.dtype)
                out[k] = w.contiguous(); out_bytes += out[k].numel() * out[k].element_size()
        save_file(out, os.path.join(a.out, shard), metadata=meta); del out
    rng = random.Random(0); pick = rng.sample(sorted(mods), 3); rep = {"base": a.base, "adapter": a.adapter, "adapter_sha256": asha,
                                                                    "scale": s, "r": r, "lora_alpha": alpha, "merged_modules": len(mods), "other_tensors": len(other)}
    chk_a = {}
    for k in pick:                                              # check a: W + s B A in fp32 against the stored bf16 tensor
        with safe_open(os.path.join(a.base, wmap[k]), "pt") as f: w = f.get_tensor(k).float()
        with safe_open(os.path.join(a.out, wmap[k]), "pt") as f: o = f.get_tensor(k).float()
        want = w + s * (ad.get_tensor(mods[k][1]).float() @ ad.get_tensor(mods[k][0]).float())
        d = (o - want).abs(); ratio = float((d / half_ulp(want)).max())
        chk_a[k] = {"max_abs_diff": float(d.max()), "max_diff_over_halfulp": ratio, "pass": ratio < 1.0}
    mism = 0                                                    # check b: every other tensor byte-identical
    by_shard = {}
    for k in other: by_shard.setdefault(wmap[k], []).append(k)
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(a.base, shard), "pt") as f, safe_open(os.path.join(a.out, shard), "pt") as g:
            for k in ks:
                x, y = f.get_tensor(k), g.get_tensor(k)
                if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x.view(-1).view(torch.uint8), y.view(-1).view(torch.uint8)): mism += 1
    idx = os.path.join(a.base, "model.safetensors.index.json")
    if os.path.exists(idx): shutil.copy2(idx, os.path.join(a.out, "model.safetensors.index.json"))
    rep.update({"check_a": chk_a, "check_a_pass": all(v["pass"] for v in chk_a.values()), "check_b_tensors": len(other), "check_b_mismatch": mism,
                "check_c_base_tensor_bytes": base_bytes, "check_c_out_tensor_bytes": out_bytes, "check_c_pass": base_bytes == out_bytes,
                "seconds": round(time.time() - t0, 1)})
    rep["ALL_PASS"] = rep["check_a_pass"] and mism == 0 and rep["check_c_pass"]
    json.dump(rep, open(os.path.join(a.out, "MERGE-REPORT.json"), "w"), indent=1)
    open(os.path.join(a.out, "README.md"), "w").write(card(a.repo, a.base_id, a.about, r, alpha, len(mods), len(other), base_bytes, asha))
    print(json.dumps({k: v for k, v in rep.items() if k != "check_a"}, indent=1)); return 0 if rep["ALL_PASS"] else 1


def upload(a):
    files = sorted(os.path.relpath(os.path.join(d, f), a.dir) for d, _, fs in os.walk(a.dir) for f in fs)
    rep = json.load(open(os.path.join(a.dir, "MERGE-REPORT.json")))
    if not rep.get("ALL_PASS"): raise SystemExit(f"{a.dir}/MERGE-REPORT.json: ALL_PASS is not true; no upload")
    for f in files: print(f"{os.path.getsize(os.path.join(a.dir, f)):>14,}  {f}")
    print(f"repo {a.repo} ({'private' if a.private else 'public'}), {len(files)} files")
    if not a.yes: print("no --yes: nothing uploaded"); return 0
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(); api.create_repo(a.repo, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=a.dir, repo_id=a.repo, commit_message=f"full model ({rep['merged_modules']} LoRA modules merged)")
    info = api.model_info(a.repo, files_metadata=True); hub = {s.rfilename: s.size for s in info.siblings}
    miss = [f for f in files if hub.get(f) != os.path.getsize(os.path.join(a.dir, f))]
    print(f"on the Hub: {len(hub)} files, sha {info.sha}; missing or other size: {miss or 'none'}"); return 1 if miss else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python tools/full_model.py"); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge"); m.add_argument("--base", required=True); m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True); m.add_argument("--repo", required=True); m.add_argument("--base_id", required=True)
    m.add_argument("--about", default=""); m.add_argument("--dry", action="store_true")
    u = sub.add_parser("upload"); u.add_argument("--dir", required=True); u.add_argument("--repo", required=True)
    u.add_argument("--private", action="store_true"); u.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    return merge(a) if a.cmd == "merge" else upload(a)


if __name__ == "__main__":
    sys.exit(main())
