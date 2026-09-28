#!/usr/bin/env python3
"""Phase0b late_margin rescue-pairing manifests: 10 qualified offsets x seeds{7,11,13}.

Confirms which qualified base-failure offsets the FROZEN late_margin shield can
rescue to SR=1 / policy_cc=0 with a valid trigger phase. Screening seeds only
(7/11/13) -> no leak into collection (17/19) or eval (23/29/31). Mirrors the
proven surgical-scan launcher (roundG/pi05_stage2/run.py + aegis env).
"""
import json, sys
from pathlib import Path

sys.path[:0] = ['.', 'src']
try:
    from roundG.pi05_matrix.matrix import code_checksum
except Exception:
    def code_checksum() -> str:
        return "UNBOUND_code_checksum"

OUT = Path("${WORK_ROOT}/lora_dagger/phase0b_rescue/batch")

QUALIFIED_OFFSETS = [0, 5, 10, 26, 27, 29, 36, 38, 42, 43]   # from phase0_qualified.json
SCREEN_SEEDS = [7, 11, 13]
# frozen late_margin config (recommended in HANDOFF; tuned on off9)
ALPHA, EEF_RADIUS, ORACLE_RADIUS, MARGIN = 3, 0.03, 0.04, 0.02
TASK_SUITE, TASK_LEVEL, TASK_ID = "safety_static_obstacles", 1, 2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = code_checksum()
    cells = []
    idx = 0
    for off in QUALIFIED_OFFSETS:
        for s in SCREEN_SEEDS:
            idx += 1
            nm = f"p0b_late_margin_off{off}_seed{s}"
            cells.append(dict(
                name=nm, output_dir=str(OUT / nm),
                task_suite_name=TASK_SUITE, task_level=TASK_LEVEL, task_id=TASK_ID,
                offset=off, seed=s, replan_steps=1, trials=1, arm="base",
                alpha=ALPHA, eef_radius=EEF_RADIUS, oracle_radius=ORACLE_RADIUS,
                margin=MARGIN, code_checksum=d))
            (OUT / f"manifest_{idx:02d}.json").write_text(json.dumps(cells[-1]) + "\n")
    (OUT / "batch.json").write_text(json.dumps(
        {"n": len(cells), "offsets": QUALIFIED_OFFSETS, "seeds": SCREEN_SEEDS,
         "config": {"alpha": ALPHA, "eef_radius": EEF_RADIUS,
                    "oracle_radius": ORACLE_RADIUS, "margin": MARGIN},
         "checksum": d}, indent=1))
    print(f"cells={len(cells)} offsets={len(QUALIFIED_OFFSETS)} seeds={SCREEN_SEEDS} checksum={d[:12]}")
    print(f"set SGE array to 1-{len(cells)}")


if __name__ == "__main__":
    main()
