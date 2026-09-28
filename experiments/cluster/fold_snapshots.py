#!/usr/bin/env python3
"""Fold every sweep snapshot into an evaluable merged checkpoint.

fold_lora_adapter.fold_one() reads FOLDS/offset_<N>/params and writes OUT/offset_<N>,
so each step_XXXX snapshot is exposed under an offset_<step> symlink and the module
constants are pointed at the sweep tree. No change to the folding code itself.
"""
import argparse, pathlib, sys, os, shutil, json

OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))
import fold_lora_adapter as F                      # noqa: E402
import orbax.checkpoint as ocp                     # noqa: E402

SWEEP = pathlib.Path("${WORK_ROOT}/lora_dagger/phase2_sweep")
MERGED = pathlib.Path("${WORK_ROOT}/lora_dagger/phase2_sweep_merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, required=True)
    a = ap.parse_args()

    fold_dir = SWEEP / f"offset_{a.offset}"
    out_dir = MERGED / f"offset_{a.offset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(int(p.name.split("_")[1]) for p in fold_dir.glob("step_*") if p.is_dir())
    if not steps:
        print(f"no snapshots under {fold_dir}")
        return
    print(f"offset {a.offset}: snapshots {steps}")

    # expose step_XXXX as offset_XXXX so fold_one's naming works unchanged
    for s in steps:
        link = fold_dir / f"offset_{s}"
        if not link.exists():
            os.symlink(fold_dir / f"step_{s:04d}", link)

    F.FOLDS = fold_dir
    F.OUT = out_dir
    ck = ocp.StandardCheckpointer()
    done = []
    for s in steps:
        tgt = out_dir / f"offset_{s}"
        if (tgt / "params").exists():
            print(f"  step {s}: already folded")
            done.append(s)
            continue
        try:
            n = F.fold_one(s, ck)
            print(f"  step {s}: folded {n} weights -> {tgt}", flush=True)
            done.append(s)
        except Exception as exc:
            print(f"  step {s}: FOLD FAILED {exc}", flush=True)
        # eval infra needs assets alongside params
        assets = tgt / "assets"
        if not assets.exists():
            src = pathlib.Path(F.BASE) / "assets"
            if src.exists():
                try:
                    os.symlink(src, assets)
                except Exception:
                    shutil.copytree(src, assets)
    (out_dir / "folded_steps.json").write_text(json.dumps(sorted(done)) + "\n")
    print(f"FOLD_SNAPSHOTS_DONE offset={a.offset} steps={sorted(done)}")


if __name__ == "__main__":
    main()
