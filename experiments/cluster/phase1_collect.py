#!/usr/bin/env python3
"""Phase1 full collection manifests: 10 rescuable offsets x collection seeds {17,19}."""
import json
from pathlib import Path
OUT=Path("${WORK_ROOT}/lora_dagger/phase1_collection/batch"); OUT.mkdir(parents=True,exist_ok=True)
OFFSETS=[0,5,10,26,27,29,36,38,42,43]
SEEDS=[17,19]
ALPHA,EEF,ORACLE,MARGIN=3,0.03,0.04,0.02
idx=0;cells=[]
for off in OFFSETS:
    for s in SEEDS:
        idx+=1; nm=f"p1_off{off}_seed{s}"
        cells.append(dict(name=nm, output_dir=str(OUT/nm), task_suite_name="safety_static_obstacles",
            task_level=1, task_id=2, offset=off, seed=s, replan_steps=1, trials=1, arm="base",
            alpha=ALPHA, eef_radius=EEF, oracle_radius=ORACLE, margin=MARGIN))
        (OUT/f"manifest_{idx:02d}.json").write_text(json.dumps(cells[-1])+"\n")
(OUT/"batch.json").write_text(json.dumps({"n":len(cells),"offsets":OFFSETS,"seeds":SEEDS},indent=1))
print(f"cells={len(cells)}")
