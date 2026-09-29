# Notes

Reference details that the [README](README.md) leaves out.

## Reproducibility

- **Evaluation** is bit-reproducible on a fixed GPU model; `failbank-rollout` sets the
  deterministic policy-server flags. Different GPU models give different trajectories, so run
  all cells of a comparison on one machine. `--seed` does not change the simulation; use
  `--sampler-advance 0/1` for independent repeats of a cell.
- **Training** is bit-reproducible only with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.
  Without it, repeated updates differ slightly.
- **Paper settings**: `failbank-train --steps 800 --batch-size 32 --quiet-weight 0.0`, data
  seeds 1/2/3; evaluation on every initial state of each task, sampler advance 0 and 1.
  For π0 add `--lora-config pi0_vla_arena_low_mem_finetune` and `--openpi-config pi0`.

## Record format

| Field / name | Meaning |
|---|---|
| `nominal_action` | the policy's action; what the environment executed during observe-only collection |
| `executed_action` | the **teacher's proposal** ã<sub>t</sub> (training target), not the executed action |
| `formal_projection.triggered` | the teacher's trigger z<sub>t</sub> |
| `quiet` | `not triggered`; weighted by `--quiet-weight` (λ<sub>q</sub>) |
| `late_margin` candidate | the teacher's own proposal inside `projection_candidates` |
| `runtime_info.cost_pair_min_distance` | object–hazard distance, read after the step |
| `steps_to_first_risk` | lead time to the first risk crossing, minus 10 (the evaluator's 10 settle steps bypass the step seam); kept as in the paper |

Stage boundaries in true steps before the first crossing: `S1_early` ≥ 40, `S2_mid` 25–39,
`S3_emergency` 10–24, `POST_crossing` < 10 (including after contact).

## Behaviour worth knowing

- The admission gate filters `train.jsonl` only; the held-out `validation.jsonl` is copied
  unchanged.
- The update guard compares against the checkpoint training started from.
- `quiet_flow_ratio` / `quiet_action_drift` are computed on the whole validation batch.
- In VLA-Arena, any `done` counts as success when `cfg.safety` is False (the default); do not
  end episodes early.
- Raw camera images are stored as the simulator returns them (rotated 180° relative to the
  policy input); `model_input_refs` hold the images the policy saw.
- A bank's `episodes/<round>` entries are symlinks to each round's episodes.
- The risk threshold 0.1087 and the teacher-weight buckets are fixed design constants.

## Using another shield

Implement `name`, `reset(ctx)` and `propose(ctx, nominal) -> TeacherProposal`
(see `src/failbank/teacher/base.py`) and pass `--teacher your_pkg.module:YourShield`.
The built-in teacher reads simulator geometry, so it is for simulation only.
