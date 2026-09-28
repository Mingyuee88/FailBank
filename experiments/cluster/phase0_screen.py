#!/usr/bin/env python3
"""Phase0 base-scan manifest generator for LoRA-DAgger offset qualification.

Pure base / pass-through runs (aegis OFF, SRD telemetry ON) over a candidate
offset range x screening seeds {7,11,13} on task2 (safety_static_obstacles L1
task-id 2). No observation dump. Downstream qualification is done by
phase0_qualify.py using the authoritative policy-induced-cost decomposition.

Screening seeds are frozen (7/11/13). Do NOT reuse collection seeds (17/19) or
evaluation seeds (23/29/31) here -- that would leak selection into the eval.
"""
import json, sys
from pathlib import Path

sys.path[:0] = ['.', 'src']
try:
    from roundG.pi05_matrix.matrix import code_checksum
except Exception:                                   # binding point on cluster
    def code_checksum() -> str:
        return "UNBOUND_code_checksum"

OUT = Path("${WORK_ROOT}/lora_dagger/phase0_base_scan/batch")

# Full first batch per converged protocol: scan 0-47, exclude off9 (selection
# exposed). No early stop at target count -- the whole batch runs.
SCREEN_SEEDS = [7, 11, 13]
OFFSETS = [o for o in range(0, 48) if o != 9]        # off9 kept as regression gate only

TASK_SUITE = "safety_static_obstacles"
TASK_LEVEL = 1
TASK_ID = 2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = code_checksum()
    cells = []
    idx = 0
    # NOTE: manifest fields match the PROVEN base-scan launcher
    # (stage2_srd/p0_variance/run.sh -> roundG/pi05_static_sweep/run.py).
    # Pure base scan sets NO aegis/SRD env; SRD telemetry (step_cost +
    # pre_step_features.cost_predicate_values) is default-on in static_sweep,
    # verified present in p0_variance/batch/p0_off9_seed11/shield_steps.jsonl.
    for off in OFFSETS:
        for s in SCREEN_SEEDS:
            idx += 1
            nm = f"p0_base_task2_off{off}_seed{s}"
            cells.append(dict(
                name=nm, output_dir=str(OUT / nm),
                task_suite_name=TASK_SUITE, task_level=TASK_LEVEL, task_id=TASK_ID,
                offset=off, seed=s, replan_steps=1, trials=1,
                arm="base", code_checksum=d))
            (OUT / f"manifest_{idx:04d}.json").write_text(json.dumps(cells[-1]) + "\n")
    (OUT / "batch.json").write_text(json.dumps(
        {"n": len(cells), "offsets": OFFSETS, "seeds": SCREEN_SEEDS,
         "task_suite": TASK_SUITE, "task_level": TASK_LEVEL, "task_id": TASK_ID,
         "checksum": d}, indent=1))
    print(f"cells={len(cells)} offsets={len(OFFSETS)} seeds={SCREEN_SEEDS} checksum={d[:12]}")
    print(f"set SGE array to 1-{len(cells)} in run_phase0_screen.sh")


if __name__ == "__main__":
    main()
