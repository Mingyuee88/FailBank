#!/usr/bin/env python3
"""Rebuild the merged teacher root with the SUCCESS offsets split in half.

The first merge put every base-success offset into training, which would have made
the within-task retention number meaningless: retention is measured on base-success
offsets, and they would all have been seen during training. The original LoRA line
did not have this problem because it trained only on failure offsets, leaving all
37 successes as a clean retention set -- the merge is what broke it.

Split (fixed here, before any training, and not revisited):
  negative-train : every other success offset, contributes "do nothing" states
  held-out       : the rest, never seen in training, measures within-task retention

The final judgement is still the cross-task transfer on L1t1/L1t3/L1t4, where
nothing has been seen at all. The within-task number is a diagnostic, not the claim
-- geo5 scored 4/6 on the development domain and then destroyed half the successes
on transfer, so a good L1t2 number proves nothing on its own.
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

LR = Path("${WORK_ROOT}/lora_dagger")
POS = LR / "phase1_collection/records"
NEG = LR / "v16_negcollect/records"
OUT = LR / "v18_split/records"

def offset_of(ep_dir: Path):
    ej = json.loads((ep_dir / "episode.json").read_text())
    rp = (ej.get("result_ref") or {}).get("path")
    if not rp or not Path(rp).exists():
        return None, None
    r = json.loads(Path(rp).read_text())
    e = (r.get("episodes") or [{}])[0]
    return str(r.get("init_state_offset")), bool(e.get("success"))

# episodes/ is two-level prefixed: episodes/<xx>/<full_hash>/
neg_eps = {}
for pre in sorted((NEG / "episodes").iterdir()):
    if not pre.is_dir():
        continue
    for d in sorted(pre.iterdir()):
        if not (d / "episode.json").exists():
            continue
        off, ok = offset_of(d)
        if off is not None:
            neg_eps.setdefault(off, []).append((d, ok))

offs = sorted(neg_eps, key=int)
train_offs = offs[0::2]
heldout_offs = offs[1::2]
print(f"negative collection offsets ({len(offs)}): {offs}")
print(f"  -> negative-TRAIN ({len(train_offs)}): {train_offs}")
print(f"  -> HELD-OUT       ({len(heldout_offs)}): {heldout_offs}")
(LR / "v18_split").mkdir(parents=True, exist_ok=True)
(LR / "v18_split/split.json").write_text(json.dumps(
    {"negative_train_offsets": train_offs, "heldout_success_offsets": heldout_offs,
     "note": "fixed before training; final judgement is cross-task transfer"}, indent=2))

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "blobs").mkdir(parents=True)
(OUT / "episodes").mkdir(parents=True)

def link_dir(src: Path, dst: Path):
    subprocess.run(["cp", "-rs", f"{src}/.", f"{dst}/"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# blobs: both halves, content-addressed so duplicates are the same bytes
link_dir(POS / "blobs", OUT / "blobs")
link_dir(NEG / "blobs", OUT / "blobs")
# episodes: all of the positive half, only the training slice of the negative half
link_dir(POS / "episodes", OUT / "episodes")
kept = 0
for off in train_offs:
    for d, _ in neg_eps[off]:
        dst = OUT / "episodes" / d.parent.name / d.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(d, dst)
            kept += 1
n_all = sum(1 for pre in (OUT / "episodes").iterdir() if pre.is_dir()
            for _ in pre.iterdir())
print(f"episodes linked: {n_all} (negative-train slice contributed {kept})")

r = subprocess.run(
    ["external/VLA-Arena/envs/openpi/.venv/bin/python", "-m",
     "vlsa_arena.learning_records.build_derived", "--records-root", str(OUT)],
    cwd="${SE_VLA_ROOT}",
    env={**os.environ, "PYTHONPATH": "src:external/VLA-Arena:."},
    capture_output=True, text=True)
print(r.stdout[-2000:])
if r.returncode:
    print("BUILD FAILED", r.stderr[-1500:]); sys.exit(1)
