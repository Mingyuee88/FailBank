# Release verification

This package is a refactor of the research code that produced the paper. Before release, the
refactored code was run side by side with the original code on the same machine (one 4×L40S
node; every cell pinned to it) and compared with the archived paper runs. Rule applied: a change
is accepted only if what the process actually did matches, not if the source reads the same.

## Runtime (Stage 1 and evaluation)

Setup: a clean VLA-Arena checkout at `2ddcb00` with only `patches/vla-arena-2ddcb00.patch`
applied (unpatched `train.py`, no project code on the path), against the original research
checkout. Each cell is one episode (`--trials 1`).

| cell | policy | compared | result |
|---|---|---|---|
| π0.5 L1-T2 (Mango), offset 0, sampler repeat 0 | base | release vs original vs archived | identical |
| π0.5 L1-T2, offset 0, sampler repeat 1 | base | release vs original vs archived | identical |
| π0.5 L1-T2, offset 0, repeat 0 | FailBank adapter `ncs1` (paper) | release vs original vs archived | identical |
| π0 L1-T0 (Apple), offset 0 | base | release vs original vs archived | identical |

"Identical" means: `successes`, `cost`, `episodes` (success, official and policy-induced cost),
`metric_decomposition` and `task_identity` in `result.json`; and at every control step the
executed action (float-exact), the step cost, `done`, the truth values of all cost predicates, and
the teacher's trigger, correction norm and barrier value. Teacher triggers per cell: 54, 76, 38
and 33 of 300, 300, 97 and 98 steps.

## Stage-1 records

One collection cell (π0.5 base, L1-T2, offset 0, observe-only, records on), original vs release
on the same host: all 300 `raw_steps.jsonl` rows equal in every field except the random episode
id; 3,300 blob files byte-identical; `episode.json` equal except run metadata (ids, timestamps,
the task description, which the original stored as the run label).

## Stages 2 and 3 (CPU)

On a subset of the paper's two collection rounds (offsets 0–2 of each round, 6 episodes), the
original `build_derived_c1.py`, `merge_bank.py` and `build_round_records.py --mode s1s2` against
`failbank-build-derived`, `failbank-merge-bank` and `failbank-build-round`: every output file
(train/validation folds, candidate files and manifests of both rounds, the merged bank, the gated
training set) byte-identical. Teacher projection (`oracle_geometry.project_translation`) against
the original on 20,000 random configurations (0–3 obstacles): identical translation, status,
barrier and active-constraint count.

## Stage 4

Pending at the time of writing: the paper's `ncs1` adapter (two-round bank, 6,535 gated records,
data seed 1, 800 steps, batch 32, λ_q = 0) retrained with the original trainer and with
`failbank-train`, compared with each other and with the archived guard metrics.

## Minimal example

Pending at the time of writing: `examples/minimal/run_minimal.sh` run unmodified against the
clean VLA-Arena clone.

## What changed on purpose

Behaviour-neutral for the paper's runs (verified above), but different from the research code:

* Research branches that the paper does not use were removed (see `experiments/README.md`).
  The telemetry-only collision-course logger (`SE_VLA_SRD_*`) was replaced by `teacher.jsonl`.
* Configuration moved from `SE_VLA_*` environment variables to command-line flags whose defaults
  are the paper's values.
* The step seam in the VLA-Arena evaluator defaults to the upstream `env.step(action.tolist())`
  instead of importing project code.
* The weighted objective and held-out guard moved out of VLA-Arena's `train.py` into
  `failbank/train/guard.py`; `train.py` is no longer patched.
* A crashed cell no longer commits its learning records (the original committed them, and the
  record builder would then have labelled the episode a failure).
* Record validation accepts teacher-specific candidate lists (it previously required exactly the
  five candidates of the built-in teacher, which the built-in teacher still produces).

Files of the research checkout not carried over: backups (`*.bak*`, `*.pre_*`), caches, the
π0-FAST evaluation config (π0-FAST is not supported by the update), an unused OpenVLA evaluator
hook, and VLA-Arena's `openpi.yaml` edit (replaced by `src/failbank/configs/`).
