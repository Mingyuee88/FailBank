# FailBank — detailed notes (install details, reproduction boundary, data traps)

**Learning from runtime feedback through failure-bank self-evolution for vision-language-action models.**

[Mingyue Cui](https://github.com/Mingyuee88), [Zheyuan Liu](https://franciscoliu.github.io/), [Yihan Zhu](https://yihan226.github.io/), [Zheyuan Zhang](https://jasonzhangzy1757.github.io/)<br>
University of Notre Dame · contact: mcui3@nd.edu

<!-- TODO: fill in the arXiv id and the GitHub Pages URL -->
[Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX) · [Project page](https://mingyuee88.github.io/FailBank/)

FailBank turns runtime feedback into persistent policy improvement. While the VLA policy
controls the robot, an **observe-only** control-barrier teacher labels every nominal action
with a barrier-compliant proposal but never executes it. Outcome-aware selection turns those
labels into learning records, which accumulate across rounds in a **failure bank**; a LoRA
update guarded by a held-out drift check then produces the next policy. The shield is used
as a source of supervision, not as a permanent action filter.

This repository contains the method (all four stages), the evaluation harness used for the
paper, and a minimal end-to-end example. It is evaluated on the
[VLA-Arena](https://github.com/PKU-Alignment/VLA-Arena) static-obstacle safety suite (Levels 1
and 2, five tasks each) with the Arena-finetuned π0.5 and π0 checkpoints.

```
rollout (policy acts) ──> teacher labels a_t  ──> learning records ──> failure bank ──> guarded LoRA update ──> next policy
     Stage 1: observe-only          Stage 2: admit + weight      Stage 3: accumulate       Stage 4: update + guard
```

## Contents

| Stage | What it does | Code | CLI |
|---|---|---|---|
| 1 | Roll out the policy in VLA-Arena; the teacher proposes ~a_t and a trigger z_t at every step; the environment executes a_t; each step is written as a learning record | `failbank/runtime/rollout.py`, `failbank/teacher/`, `failbank/records/` | `failbank-rollout` |
| 2a | Outcome windows, teacher weight (`teacher_quality`), risk staging, episode-wise held-out folds | `failbank/pipeline/build_derived.py` | `failbank-build-derived` |
| 3 | Add a round's records to the bank (never substitute) | `failbank/pipeline/merge_bank.py` | `failbank-merge-bank` |
| 2b | Admission gate (`keep_record`) on the bank's training split | `failbank/pipeline/build_round_records.py` | `failbank-build-round` |
| 4 | Record-weighted first-action LoRA update; keep a candidate only if the held-out guard accepts it | `failbank/train/` | `failbank-train`, `failbank-fold` |
| eval | Same rollout, no records: SR, official CC, policy-induced CC | `failbank/runtime/` | `failbank-rollout` |

Other directories: `patches/` (the VLA-Arena patch), `extras/aegis_baseline/` (published AEGIS as an
optional in-loop baseline), `examples/minimal/`, `experiments/` (the original SGE launchers,
kept for provenance), `tests/` (CPU unit tests), `docs/` (project page; serve it with GitHub
Pages from the `docs/` folder).

## Install

FailBank runs inside VLA-Arena's OpenPI environment; it does not vendor or re-pin it.

```bash
# 1. VLA-Arena at the commit this release was verified against, plus our patch
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena && git checkout 2ddcb00
git apply /path/to/failbank/patches/vla-arena-2ddcb00.patch
# 2. its OpenPI environment (follow VLA-Arena's instructions; we used envs/openpi with uv, Python 3.11)
# 3. FailBank into that environment
pip install -e /path/to/failbank            # core: reproduces FailBank
pip install -e "/path/to/failbank[aegis]"   # optional: the AEGIS baseline (see extras/aegis_baseline)
export PYTHONPATH=/path/to/VLA-Arena:$PYTHONPATH
```

What the patch changes (`git apply --stat` shows 11 files, +83/−2), and nothing else:

* `models/openpi/evaluator.py`: a step seam `runtime_env_step(env, obs, task, action, cfg, t)` that
  `run_episode` calls instead of `env.step(action.tolist())` (its default is exactly that call);
  a websocket readiness handshake replacing the open-port check (an open port is not a ready
  policy server); optional backview + depth cameras when `FAILBANK_AEGIS_CAMERAS=1` (AEGIS only).
* ten scene XMLs: one `backview` camera each, copied from vlsa-aegis, used only by the AEGIS
  baseline. Cameras do not affect physics, objects or task semantics.

Checkpoints (params + assets; the optimizer state is not needed):

| backbone | Hugging Face | directory name expected under `$FAILBANK_CHECKPOINTS` |
|---|---|---|
| π0.5 | `VLA-Arena/pi05-vla-arena-finetuned` (we used revision `31bbab4`) | `pi05_vla_arena_finetuned` |
| π0 | `VLA-Arena/pi0-vla-arena-fintuned` (revision `9fb1694`) | `pi0_vla_arena_finetuned` |

```bash
export FAILBANK_CHECKPOINTS=/path/to/checkpoints
python -m pytest tests          # CPU-only checks of the teacher and the record pipeline
```

## Quick start

`examples/minimal/` runs the whole loop at toy scale on one GPU: collect a few observe-only
episodes, build and gate the records, one 800-step guarded update, fold, and evaluate one cell.
No AEGIS checkout and no VLM service are involved. See `examples/minimal/README.md`.

## Running the method

All commands take `--help`. Settings shown are the paper's.

**Stage 1 — collect** one cell (one task, one initial-state offset) with the observe-only
teacher. Repeat over offsets; one process per cell.

```bash
failbank-rollout --output runs/r1/off0/result.json --task-suite safety_static_obstacles \
    --task-level 1 --task-id 2 --offset 0 --record-root runs/r1/records \
    [--policy-checkpoint <folded checkpoint of the previous round's policy>]
```

**Stage 2a / 3 / 2b — build, accumulate, gate**

```bash
failbank-build-derived --records-root runs/r1/records --folds 0
failbank-build-derived --records-root runs/r2/records --folds 0
failbank-merge-bank --round r1=runs/r1/records --round r2=runs/r2/records --out runs/bank --fold offset_0
failbank-build-round --src-root runs/bank --dst-root runs/bank_s1s2 --mode s1s2 --fold offset_0
```

**Stage 4 — update, fold**

```bash
failbank-train --records runs/bank_s1s2 --fold 0 \
    --base-checkpoint $FAILBANK_CHECKPOINTS/pi05_vla_arena_finetuned --output-root runs/adapter_s1 \
    --steps 800 --batch-size 32 --quiet-weight 0.0 --data-seed 1
JAX_PLATFORMS=cpu failbank-fold --adapter runs/adapter_s1/offset_0 \
    --base $FAILBANK_CHECKPOINTS/pi05_vla_arena_finetuned --output runs/ckpt_s1
```

The paper's method is three such adapters, `--data-seed 1/2/3`, trained from the base
checkpoint on the two-round bank with `--quiet-weight 0.0`. For π0 add
`--lora-config pi0_vla_arena_low_mem_finetune` and the π0 base checkpoint.

**Evaluate** a cell (same command without `--record-root`):

```bash
failbank-rollout --output runs/eval/ncs1/L1t2/off0_r0/result.json --task-level 1 --task-id 2 \
    --offset 0 --policy-checkpoint runs/ckpt_s1 [--sampler-advance 1] [--openpi-config pi0]
```

Protocol used in the paper: every initial-state offset of a task, two sampler repeats per
offset (`--sampler-advance 0` and `1`), all cells of a comparison on **one host**; the
restored checkpoint is checked per cell against `server_port*.log`. Each `result.json` reports
`successes`, and under `metric_decomposition` the official CC and the policy-induced CC.

## Plugging in another shield

Any object with `name`, `reset(ctx)` and `propose(ctx, nominal) -> TeacherProposal` is a
teacher (`failbank/teacher/base.py`). Select it with
`--teacher your_package.module:YourShield`; `--execute teacher` runs it in the loop instead of
observe-only. The built-in teacher (`oracle_geometry`) projects the nominal translation with a
CBF onto spheres placed at the simulator's ground-truth hazard positions, so it only exists in
simulation. **A different teacher produces different labels and will not reproduce the
paper's numbers; that is expected.**

`extras/aegis_baseline/` wraps published AEGIS (GLM-4.5V perception + CBF-QP) behind the
same interface; it is the in-loop baseline of the paper and needs a vlsa-aegis checkout and a
VLM endpoint. The core never imports it.

## What this release can and cannot reproduce

| can | cannot |
|---|---|
| all four stages, the admission gate, the held-out guard and the evaluation protocol | the trained adapters of the paper (not included; retrain them) |
| record building and bank accumulation from your own collections | the paper's raw rollouts and record bank (3.1 GB; not included) |
| the paper's evaluation numbers from the paper's adapters, bit-for-bit on the same GPU model | bit-identical numbers on a different GPU model |
| a guarded update that is bit-reproducible on one GPU model when run with `XLA_FLAGS=--xla_gpu_deterministic_ops=true` | bit-identical copies of the paper's adapters (they were trained without that flag) |

* **Evaluation: the GPU model is the only source of run-to-run variation.** With the
  deterministic server flags set by `failbank-rollout`, a cell is bit-reproducible on one GPU
  model; the same weights on a different model give different trajectories (L1-T3 base SR moved
  from 65.0 to 72.0 between two machines). Pin every cell of a comparison to one host; `--seed`
  does not change the simulation.
* **Training is not deterministic by default.** Without
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, two identical updates on the same GPU differ in
  the last digits of their losses (and so in their adapters); with it they are bit-identical.
  The paper's adapters were trained without the flag.
* **The default teacher uses privileged simulator geometry.** It is a supervision source for
  simulation, not a deployable shield.

## Traps and naming

Kept for compatibility with existing record banks; read these before touching the data.

1. **`executed_action` in records is not the executed action.** It is the teacher's proposal
   ~a_t (the learning target). During observe-only collection the environment executed
   `nominal_action`. `runtime_info.shield_observe_only` says which regime a record came from.
2. **"quiet" means three things:** the record label `quiet = not triggered`; the loss weight of
   quiet records (`--quiet-weight`, λ_q in the paper); and the guard metrics `quiet_flow_ratio`
   / `quiet_action_drift`, which are computed on the whole fixed validation batch (triggered and
   quiet records together), not on quiet records only.
3. **The admission gate applies to `train.jsonl` only.** `validation.jsonl` is copied unchanged,
   so the held-out set the guard measures on still contains the post-contact and emergency
   records training never sees (in the paper's bank, 470 of the 600 validation records are
   `POST_crossing`).
4. **The guard's reference is the checkpoint the run started from** (`base_state` in
   `train/guard.py`), not "the original base". They coincide in the paper only because every
   update starts from the base checkpoint.
5. **Risk stages are shifted by 10 steps.** `steps_to_first_risk` is computed as
   (row index of the first risk crossing, counted from 0) − `step_index`, but `step_index`
   starts at 10 because the evaluator's 10 settle steps do not pass through the step seam.
   Every stage boundary is therefore 10 steps earlier in time than its nominal value. In true
   steps before the crossing: `S1_early` ≥ 40, `S2_mid` 25–39, `S3_emergency` 10–24, and
   `POST_crossing` also contains the 10 steps just before the crossing. The released code keeps
   this behaviour so that it reproduces the paper's records.
6. **`runtime_info.cost_pair_min_distance` is read after `env.step`**, i.e. it describes the
   post-step state; risk staging uses it.
7. **The formal candidate is called `late_margin`.** Records store the teacher's own proposal as
   the projection candidate named `late_margin` (`teacher.FORMAL_CANDIDATE`); `teacher_quality`
   looks it up by that name. The name is historical (a CBF parameter set).
8. **Design-time constants.** The risk threshold 0.1087 (a collision-warning line chosen on one
   task, L1-T2) and the teacher-weight buckets (0 / 0.25 / 0.60 / 1.00 at score edges
   0.25 / 0.45 / 0.70) were set at design time, not fitted, and were not re-calibrated per task.
9. **In VLA-Arena every `done` counts as success** when `cfg.safety` is False (the default). Any
   runtime change that terminates an episode early turns a timeout into a "success".
10. **Bank `episodes/` entries are symlinks** (`<bank>/episodes/<round tag>` → that round's
    `episodes/`); follow them when tracing a record back to its rollout.
11. **Raw camera images are stored as the simulator returns them** (rows and columns reversed
    relative to the view the policy gets; the evaluator rotates them by 180°). The model inputs
    under `model_input_refs` are the rotated 224×224 images the policy actually saw.
12. **`experiments/` scripts use the original code layout** (`roundG/pi05_stage2/run.py` driven by
    `SE_VLA_*` environment variables); in them the arm named `ours` in `run_l2_task.sh` resolves to
    a curriculum checkpoint, while the paper's method is always `nocurr` (`ncs1/2/3`). Read
    checkpoint paths, not arm names. `experiments/README.md` maps the old variables to the new flags.

## Release verification

Before release, the refactored code was run against the original research code on the same host
(and against the archived cells of the paper), see `VERIFICATION.md`: identical per-step executed
actions, costs, predicates and teacher verdicts for π0.5 and π0 evaluation cells (including a
FailBank adapter and a sampler repeat), identical Stage-1 records and blobs for a collection cell,
byte-identical derived folds, bank and gated training set, and the teacher projection on 20,000
random inputs.

## Limitations

* The quiet-anchor gain at λ_q = 0.2 does not survive retraining (ten-task ΔSR −5.73,
  ΔCC_policy +9.68), and on the collection task its direction flips under a different training
  order on the same host. The paper's method uses λ_q = 0.
* Most of the multi-round gain arrives in the first round; SR differences between adjacent
  rounds are not significant.
* Evaluated only on the static-obstacle suite; the other Safety suites do not fit the obstacle
  model for structural reasons (see the paper's appendix).

## License and acknowledgements

FailBank is released under the Apache License 2.0 (`LICENSE`). The CBF projection in
`failbank/teacher/oracle_geometry.py` is adapted from
[vlsa-aegis](https://github.com/THU-RCSCT/vlsa-aegis) (MIT, © 2023 Lifelong Robot Learning); the
training and data-loading code in `failbank/train/` is derived from the OpenPI code distributed
with VLA-Arena (Apache-2.0). See `THIRD_PARTY_NOTICES.md` and `licenses/`.

```bibtex
@article{cui2026failbank,
  title   = {Learning from Runtime Feedback through Failure-Bank Self-Evolution for Vision-Language-Action Models},
  author  = {Cui, Mingyue and Liu, Zheyuan and Zhu, Yihan and Zhang, Zheyuan},
  journal = {arXiv preprint},
  year    = {2026}
}
```
