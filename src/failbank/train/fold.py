"""Fold a strict-LoRA adapter into its base checkpoint -> a standard OpenPI checkpoint.

    failbank-fold --adapter <out>/offset_0 --base <ckpt>/pi05_vla_arena_finetuned \
                  --output <folded>

Every LoRA'd weight in these models uses LoRAConfig(axes=(-2,-1), scaling_value=1.0), so the
fold is one batched matmul over the rank axis, ``w_fold = w + lora_a @ lora_b``, for both
attention (``.../<name>/lora_a``) and MLP (``.../<name>_lora_a``) weights. The output is a
non-LoRA tree with the base structure, which the evaluation config loads unchanged; the base
``assets/`` (norm stats) are copied alongside.

Runs on CPU (set JAX_PLATFORMS=cpu); peak memory is ~53 GB for pi0.5.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import jax
import numpy as np
import orbax.checkpoint as ocp

SEP = "\x1f"


def flatten(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, pre + SEP + str(k) if pre else str(k)))
    else:
        out[pre] = d
    return out


def unflatten(flat):
    root = {}
    for path, v in flat.items():
        parts = path.split(SEP)
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return root


def restore_numpy(path):
    """Restore to host numpy arrays, ignoring the saved sharding."""
    path = pathlib.Path(path).resolve()
    pck = ocp.PyTreeCheckpointer()
    meta = pck.metadata(path)
    restore_args = jax.tree.map(lambda _m: ocp.ArrayRestoreArgs(restore_type=np.ndarray), meta)
    return pck.restore(path, args=ocp.args.PyTreeRestore(restore_args=restore_args))


def fold(adapter_dir: pathlib.Path, base_dir: pathlib.Path, out_dir: pathlib.Path,
         ck=None, verbose: bool = False) -> int:
    """Fold ``adapter_dir/params`` into ``base_dir/params``; write ``out_dir/{params,assets}``."""
    ck = ck or ocp.StandardCheckpointer()
    base = restore_numpy(base_dir / "params")
    adap = restore_numpy(adapter_dir / "params")
    bf = flatten(base)   # params<SEP>PaliGemma<SEP>...<SEP>w
    af = flatten(adap)   # PaliGemma<SEP>...<SEP>lora_a   (no "params" prefix)

    folded = dict(bf)
    n_folded = 0
    for ak in list(af.keys()):
        leaf = ak.split(SEP)[-1]
        if leaf == "lora_a":                      # attn:  .../<name>/lora_a
            parent = ak.rsplit(SEP, 1)[0]
            bkey = "params" + SEP + parent + SEP + "w"
            lb = parent + SEP + "lora_b"
        elif leaf.endswith("_lora_a"):            # mlp:   .../<name>_lora_a
            parent = ak.rsplit(SEP, 1)[0]
            name = leaf[: -len("_lora_a")]
            bkey = "params" + SEP + parent + SEP + name
            lb = parent + SEP + name + "_lora_b"
        else:
            continue
        bkey_val = bkey + SEP + "value" if (bkey + SEP + "value") in bf else bkey
        if bkey_val not in bf:
            raise KeyError(f"base weight not found for {ak}: {bkey_val}")
        if lb not in af:
            raise KeyError(f"lora_b not found for {ak}: {lb}")
        w = np.asarray(bf[bkey_val])
        a = np.asarray(af[ak], dtype=np.float32)
        b = np.asarray(af[lb], dtype=np.float32)
        delta = np.matmul(a, b)                    # [..., A, L] @ [..., L, B] -> [..., A, B]
        if delta.shape != w.shape:
            raise ValueError(f"{ak}: delta {delta.shape} != base {w.shape}")
        folded[bkey_val] = (w.astype(np.float32) + delta).astype(w.dtype)
        n_folded += 1
        if verbose:
            print(f"  folded {bkey_val}  w{w.shape} += a{a.shape}@b{b.shape}  |delta|={np.abs(delta).mean():.3e}")
    assert not any("lora" in k.lower() for k in folded), "lora key leaked into folded tree"
    assert n_folded == len(af) // 2, f"folded {n_folded} but adapter has {len(af)} lora leaves"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    ck.save((out_dir / "params").resolve(), unflatten(folded))
    ck.wait_until_finished()
    if (base_dir / "assets").exists():
        shutil.copytree(base_dir / "assets", out_dir / "assets")
    return n_folded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", type=pathlib.Path, required=True,
                    help="adapter directory holding params/ (e.g. <output-root>/offset_0)")
    ap.add_argument("--base", type=pathlib.Path, required=True, help="base checkpoint the adapter was trained from")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    adapter, base, out = a.adapter.resolve(), a.base.resolve(), a.output.resolve()
    if out == base or out.is_relative_to(base):
        raise SystemExit("refusing to write the folded checkpoint into the base checkpoint")
    n = fold(adapter, base, out, verbose=a.verbose)
    print(f"FOLD_OK folded={n} -> {out}")


if __name__ == "__main__":
    main()
