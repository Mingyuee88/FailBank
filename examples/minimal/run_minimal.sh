#!/usr/bin/env bash
# FailBank end to end at toy scale on one GPU (pi0.5, VLA-Arena safety_static_obstacles L1-T2):
#   1. collect N_OFFSETS observe-only episodes with the base policy, writing learning records
#   2. derive, accumulate (one round), gate
#   3. one guarded LoRA update (fold 0 = offset 0 held out), fold it into the base
#   4. evaluate the updated policy on the held-out offset 0
# Needs: VLA-Arena 2ddcb00 + patches/vla-arena-2ddcb00.patch on PYTHONPATH, FailBank installed
# (or src/ on PYTHONPATH), $FAILBANK_CHECKPOINTS/pi05_vla_arena_finetuned, one 48 GB GPU and
# ~60 GB host RAM for the fold. No AEGIS checkout and no VLM service.
set -euo pipefail
: "${FAILBANK_CHECKPOINTS:?export FAILBANK_CHECKPOINTS (see README)}"
PY=${PYTHON:-python}
OUT=${OUT:-runs/minimal}
N_OFFSETS=${N_OFFSETS:-6}
STEPS=${STEPS:-800}
# Validate every VAL_EVERY steps and keep the best candidate the guard accepts. Five short
# episodes overfit long before 800 steps (at step 800 the held-out flow ratio is ~1.28, above
# the 1.10 limit), so a single end-of-run validation would reject the update.
VAL_EVERY=${VAL_EVERY:-100}
PORT_BASE=${PORT_BASE:-38000}
BASE=$FAILBANK_CHECKPOINTS/pi05_vla_arena_finetuned
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WANDB_MODE=disabled
mkdir -p "$OUT"
echo "=== FailBank minimal example -> $OUT (offsets 0..$((N_OFFSETS-1)), $STEPS update steps) on $(hostname)"

# 1. Stage 1: observe-only collection (the environment executes the policy's own actions)
for off in $(seq 0 $((N_OFFSETS - 1))); do
  cell=$OUT/collect/off$off
  if [ -s "$cell/records_commit.json" ] && grep -q '"committed": true' "$cell/records_commit.json"; then
    echo "--- collect off$off: done"; continue
  fi
  echo "--- collect off$off"
  $PY -m failbank.runtime.rollout --output "$cell/result.json" --task-suite safety_static_obstacles \
      --task-level 1 --task-id 2 --offset "$off" --record-root "$OUT/records" \
      --port-base $((PORT_BASE + 20 * off))
done

# 2. Stages 2-3: derive (fold 0), accumulate, gate
$PY -m failbank.pipeline.build_derived --records-root "$OUT/records" --folds 0
$PY -m failbank.pipeline.merge_bank --round r1="$OUT/records" --out "$OUT/bank" --fold offset_0
$PY -m failbank.pipeline.build_round_records --src-root "$OUT/bank" --dst-root "$OUT/bank_s1s2" \
    --fold offset_0

# 3. Stage 4: guarded update from the base checkpoint, then fold (CPU)
if [ ! -s "$OUT/adapter/offset_0/metrics.json" ]; then
  rm -rf "$OUT/adapter" "$OUT/ckpt"
  $PY -m failbank.train.lora_update --records "$OUT/bank_s1s2" --fold 0 --base-checkpoint "$BASE" \
      --output-root "$OUT/adapter" --steps "$STEPS" --batch-size 32 --quiet-weight 0.0 --data-seed 1 \
      --validation-interval "$VAL_EVERY"
fi
if [ ! -d "$OUT/ckpt/params" ]; then
  JAX_PLATFORMS=cpu $PY -m failbank.train.fold --adapter "$OUT/adapter/offset_0" --base "$BASE" \
      --output "$OUT/ckpt"
fi

# 4. evaluate base and updated policy on the held-out initial state (offset 0), same GPU
port=$((PORT_BASE + 500))
for arm in base updated; do
  ck=(); [ "$arm" = updated ] && ck=(--policy-checkpoint "$OUT/ckpt")
  $PY -m failbank.runtime.rollout --output "$OUT/eval/$arm/off0/result.json" \
      --task-suite safety_static_obstacles --task-level 1 --task-id 2 --offset 0 \
      --port-base "$port" ${ck[@]+"${ck[@]}"}
  port=$((port + 20))
done

$PY - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
m = json.load(open(out / "adapter/offset_0/metrics.json"))
print(f"update: accepted={m['accepted']} flow_ratio={m['best_quiet_flow_ratio']:.4f} "
      f"drift={m['best_quiet_action_drift']:.4f} records={sum(1 for _ in open(m['train_manifest']))}")
for arm in ("base", "updated"):
    r = json.load(open(out / f"eval/{arm}/off0/result.json"))
    md = r["metric_decomposition"]
    print(f"{arm:8s} held-out offset 0: success={r['successes']} official_cc={md['official_cc']} "
          f"policy_induced_cc={md['policy_induced_cc']}")
print("(single-episode demo: it shows the pipeline runs end to end, not how well the method works)")
PY
echo "MINIMAL_EXAMPLE_DONE"
