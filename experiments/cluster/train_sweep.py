#!/usr/bin/env python3
"""Retrain one LOSO fold, saving an adapter at EVERY validation.

Why: deployment is bimodal (4/9 offsets succeed shield-free, 5/9 keep the base
failure geometry) and ACROSS folds a larger held-out triggered-loss improvement
went with worse deployment. That cross-fold correlation is confounded by offset
difficulty and is not significant at n=9 (p~0.14). The decisive test is WITHIN a
fold: keep every checkpoint along one training trajectory and check whether the
triggered-loss ordering matches the closed-loop ordering.

Early stopping is disabled so the whole curve is observed, and the stock trainer
is reused unchanged apart from the additive snapshot_callback hook.
"""
import argparse, json, pathlib, sys, logging

OP = pathlib.Path("${SE_VLA_ROOT}/external/VLA-Arena/vla_arena/models/openpi")
sys.path.insert(0, str(OP / "scripts"))

import train as _train                      # noqa: E402
import train_phase2_lora as P               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--sweep-root", type=pathlib.Path, required=True)
    ap.add_argument("--validation-interval", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save-all", action="store_true",
                    help="also save guard-REJECTED checkpoints (diagnostic only; "
                         "such checkpoints are not deployable under the guard)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    out = a.sweep_root / f"offset_{a.offset}"
    out.mkdir(parents=True, exist_ok=True)

    # Build the stock trainer's args exactly as run_phase2_full.sh does.
    sys.argv = ["train_phase2_lora.py",
                "--offset", str(a.offset),
                "--batch-size", "2",
                "--quiet-weight", "0.0",
                "--output-root", str(out / "_best")]
    args = P.parse_args()
    args.records_root = args.records_root.resolve()
    args.derived_root = args.derived_root.resolve()
    args.output_root = args.output_root.resolve()
    args.assets_dir = args.assets_dir.resolve()
    P.assert_paths(base_checkpoint=P.BASE_CHECKPOINT, output_path=args.output_root)

    log = []
    cfg_holder = {}
    orig = _train.main_phase2

    def snapshot(step, state, m):
        rec = {"step": step, **m}
        print(f"[snapshot] step={step} triggered={m['triggered_loss']:.6f} "
              f"ratio={m['quiet_flow_ratio']:.4f} drift={m['quiet_action_drift']:.5f} "
              f"accepted={m['accepted']}", flush=True)
        if m["accepted"] or a.save_all:
            try:
                P.save_adapter_only(state, cfg_holder["config"], out / f"step_{step:04d}")
                rec["saved"] = True
            except Exception as exc:      # a failed snapshot must never kill training
                rec["saved"] = False
                rec["save_error"] = str(exc)
                print(f"[snapshot] save FAILED at {step}: {exc}", flush=True)
        else:
            rec["saved"] = False
        log.append(rec)
        (out / "snapshots.json").write_text(json.dumps(log, indent=2) + "\n")

    def wrapped(config, **kw):
        cfg_holder["config"] = config
        kw["snapshot_callback"] = snapshot
        kw["patience"] = 10 ** 6                      # observe the entire curve
        kw["validation_interval"] = a.validation_interval
        return orig(config, **kw)

    _train.main_phase2 = wrapped
    try:
        metrics = P.train_fold(args, a.offset)
        (out / "fold_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    finally:
        _train.main_phase2 = orig
    print(f"SWEEP_DONE offset={a.offset} snapshots={len([r for r in log if r.get('saved')])}")


if __name__ == "__main__":
    main()
