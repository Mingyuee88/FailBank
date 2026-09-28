#!/usr/bin/env bash
# Build the merged teacher-record root and run the repo's own derived builder on it.
#
# Nothing is copied and nothing existing is modified: the merged root is a tree of
# symlinks into the two collections, so build_derived.py sees one records root
# spanning both halves while phase1_collection stays byte-identical.
#
#   positive half  phase1_collection/records   10 base-FAILURE offsets x seeds{17,19}
#   negative half  v16_negcollect/records      36 base-SUCCESS offsets, seed 17
#
# The point of the merge is the QUIET POOL. build_derived.py already emits a record
# for every step with `quiet = not triggered` and teacher weight 0 for untriggered
# steps, but until now every quiet record came from an episode the base policy was
# going to fail. The `quiet_weight` knob therefore reweighted the wrong
# distribution, which is why its four-arm sweep found nothing. After the merge the
# quiet pool contains states visited during SUCCESSFUL grasps for the first time.
#
# Blobs are content-addressed, so the two halves share files. `cp -rs` aborts on
# those (and under set -e killed the previous attempt before episodes were linked);
# ln -sf is used instead -- overwriting a link to identical content is harmless.
set -euo pipefail
LR=${WORK_ROOT}/lora_dagger
POS=$LR/phase1_collection/records
NEG=$LR/v16_negcollect/records
M=$LR/v17_merged/records

# Blobs are content-addressed: a hash present in both halves is the same bytes, so
# cp's "File exists" on the second pass is the correct outcome, not an error. It is
# tolerated rather than avoided -- the earlier ln -sf loop was correct but did one
# mkdir+ln per file and was far too slow for ~10^5 blobs.
rm -rf "$LR/v17_merged"
mkdir -p "$M/blobs" "$M/episodes"
for half in "$POS" "$NEG"; do
  cp -rs "$half/blobs/." "$M/blobs/" 2>/dev/null || true
  cp -rs "$half/episodes/." "$M/episodes/" 2>/dev/null || true
done

echo "merged episodes: $(ls "$M/episodes" | wc -l)  (pos $(ls "$POS/episodes" | wc -l) + neg $(ls "$NEG/episodes" | wc -l))"
echo "merged blobs   : $(find "$M/blobs" -type l | wc -l)"

cd ${SE_VLA_ROOT}
PYTHONPATH=src:external/VLA-Arena:. external/VLA-Arena/envs/openpi/.venv/bin/python \
  -m vlsa_arena.learning_records.build_derived --records-root "$M" 2>&1 | tail -8
