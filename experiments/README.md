# Original experiment launchers (SGE cluster only, provenance)

`cluster/` holds the launchers and read-out scripts that produced the paper's runs, kept so that
every number can be traced to the command that produced it. **They do not run against this
package.** They drive the original research layout (`roundG/pi05_stage2/run.py`, configured by
`SE_VLA_*` environment variables) on an SGE cluster (`#$ -q ...`, `qsub -t` arrays). Use the
`failbank-*` commands for new work; the table below maps one onto the other.

Placeholders: `${SE_VLA_ROOT}` (research checkout), `${WORK_ROOT}` (outputs and checkpoints),
`${PAPER_ROOT}`, `${GROUP_ROOT}`; `HOST_A`/`HOST_B` (4×L40S nodes), `HOST_C`/`HOST_D`/`HOST_E`
(3×A6000 nodes), `@@GROUP` (the group's queue). Evaluation cells were pinned to one host per
comparison; training was not.

## Where the paper's runs come from

Read checkpoint paths, not arm names: in `run_l2_task.sh` the arm called `ours` resolves to a
curriculum checkpoint, while the paper's method is always `nocurr` / `ncs1`, `ncs2`, `ncs3`
(`nocurr_s{1,2,3}/ckpt/R2_q00`, data seeds 1, 2, 3). Every evaluated cell records the checkpoint it
restored in `server_port*.log` ("Finished restoring checkpoint ... from ...").

| Paper | Entry points in `cluster/` |
|---|---|
| Stage 1, round 1 (L1-T2, observe-only) | `run_C1_collect.sh` |
| Stage 1, round 2 | `run_r2_collect.sh`; round 3 (round curves): `run_r3_collect.sh`, `run_bank_chain.sh`, `run_r5_chain.sh` |
| Stages 2–3 | `build_derived_c1.py`, `merge_bank.py`, `build_round_records.py` (`--mode s1s2`) |
| Stage 4, the method (λ_q = 0, 800 steps, batch 32) | `run_nocurr_train.sh` (`DATA_SEED`, `OUT_ROOT` for `nocurr_s1..3`); review re-runs: `review0913/train.sh`, `review0913/fold.sh` |
| Main table, π0.5 Level 1 | `run_multitask.sh` (arms `base`, `aegis`, `ncs1`–`ncs3`; `TAG`=t0…t4) |
| Main table, π0.5 Level 2 | `run_l2_task.sh` |
| Main table, π0 | `run_pi0_col.sh` (collection), `review0913/train.sh` (update), `run_pi0_eval2.sh`, `run_pi0_eval2_remaining.sh` |
| Unified single-cell evaluator used for the review experiments | `review0913/eval.sh` |
| AEGIS baseline | `aegis` arms above; VLM server: `run_glm_serve.sh`, `run_glm_serve_a6k.sh` |
| Collection-interface ablation (shield in the loop) | `run_inloop_collect.sh` |
| Review experiments E1–E17 and their read-outs | `review0913/submit_*.sh`, `review0913/e*_readout.py`, `review0913/analyze_review.py` |
| Paired statistics and tables | `review0913/analyze_review.py`, `review0913/safe_metrics.py`, `review0913/occ_table.py`, `analyze_multi.py`, `paired_effect.py` |

Qualitative frames for the paper figure were dumped with a debugging switch that is not part of
this release (`run_qual_frames.sh` is therefore not included).

## Old variables → new flags

| original (`run.py`, `SE_VLA_*`) | release |
|---|---|
| `SE_VLA_AEGIS_SOURCE=1` and `SE_VLA_SRD_ENABLED=1` (teacher computed) | `--teacher oracle_geometry` (default) |
| `SE_VLA_SHIELD_OBSERVE_ONLY=1` | `--execute nominal` (default) |
| `SE_VLA_SHIELD_IMPL=local` without observe-only (labelling teacher executed) | `--execute teacher` |
| `SE_VLA_SHIELD_IMPL=vlsa_aegis`, `SE_VLA_VLSA_DOF` | `--teacher failbank_aegis:VlsaAegisTeacher --execute teacher`, `AEGIS_DOF` |
| `SE_VLA_GLM_BASE_URL` | `AEGIS_GLM_BASE_URL` |
| `SE_VLA_AEGIS_{ALPHA,EEF_RADIUS,ORACLE_RADIUS,MARGIN,MAX_TRANSLATION,OVERRIDE_EPS}` | `--teacher-{alpha,eef-radius,oracle-radius,margin,max-translation,override-eps}`; defaults are the paper's values (3, 0.03, 0.04, 0.02, 1.0, 1e-6), not the old code defaults |
| `SE_VLA_PHASE1_RECORD=1`, `SE_VLA_PHASE1_CANDIDATES=1`, `SE_VLA_PHASE1_ROOT` | `--record-root` |
| `SE_VLA_SAMPLER_ADVANCE` | `--sampler-advance` |
| `SE_VLA_POLICY_CHECKPOINT_DIR` | `--policy-checkpoint` |
| `SE_VLA_OPENPI_YAML=.../openpi_pi0.yaml`, `SE_VLA_CONFIG_RESOLUTION` | `--openpi-config pi0` |
| `SE_VLA_EPISODE_ID`, `SE_VLA_SEED`, `SE_VLA_RESULT_PATH` | `--episode-id`, `--seed`, `--output` |
| `SE_VLA_ARENA_SUITE` | taken from `--task-suite` |
| `PI05_SERVER_TIMEOUT_SEC`, `PI05_SERVER_LOG_DIR` | `--server-timeout`, `--server-log-dir` |
| `run.py --arm base` | always base; the arm flag is gone |
| `shield_steps.jsonl` / `srd.jsonl` (`aegis_projection` rows) | `steps.jsonl` / `teacher.jsonl` (`teacher_step` rows) |
| `train_phase2_lora.py --records-root X/blobs --derived-root X/derived/folds` | `failbank-train --records X` |
| `--smoke-steps N --validation-interval N --patience 99` | `--steps N` (validation once at the end and patience 99 are the defaults) |
| `SE_VLA_TRAIN_BASE_CHECKPOINT`, `SE_VLA_TRAIN_LORA_CONFIG`, `SE_VLA_SUPERVISE_FIRST_K` | `--base-checkpoint`, `--lora-config`, `--supervise-first-k` |
| `fold_lora_adapter.fold_one` with `F.FOLDS`/`F.OUT`/`F.BASE` | `failbank-fold --adapter --base --output` |
| `build_derived_c1.py` | `failbank-build-derived` |
| `merge_bank.py --r1 A --r2 B --tag1 r1 --tag2 r2` | `failbank-merge-bank --round r1=A --round r2=B` |

Telemetry-only and removed: `SE_VLA_SRD_{WRITES_ENABLED,NEAR_DISTANCE_M,RELEASE_DISTANCE_M,MIN_CLOSING_M_PER_STEP,TELEMETRY_PATH}`
(a collision-course logger from an abandoned relation-memory method; with writes disabled it
never influenced actions or records), `SE_VLA_RELATION_MEMORY_CREDIT_ENABLED`,
`SE_VLA_HAZARD_LOOKAHEAD_S` (0 was the only value used in the reported runs).

## Removed research branches

These switches belonged to abandoned lines of work and are not in the release; scripts that set
them were not copied: best-of-N selection and its critic/state gate (`BEST_OF_N`, `COST_CRITIC`,
`STATE_GATE`), oracle branching and action replay (`BRANCH_*`, `REPLAY_ACTIONS`), contact
budget (`CONTACT_BUDGET`, `BUDGET_LIFT`), contact abort and retreat (`ABORT_*`, `RETREAT_*`,
`DAMP_FACTOR`), learned distance/direction/visual heads (`DIST_HEAD`, `DIR_*`, `VHEAD*`),
hazard-zone teacher for `safety_hazard_avoidance` (`HAZARD_ZONE`, `HAZARD_LIFT_M`,
`HAZARD_PLACE_EXEMPT`), barrier variants (`AEGIS_BARRIER_CENTER`, `AEGIS_GRASP_GATE_M`,
`AEGIS_GATE_DISTANCE_M`, `AEGIS_SINGLE_OBSTACLE`, `CBF_JOINT_ITERS`), sampling-dispersion probe
(`VLSA_DISPERSION`), residual-split logging (`VLSA_RESIDUAL`), frame dumps (`FRAME_DUMP_*`) and the
phase-1 regression trace (`PHASE1_TRACE`). Also not copied: the `patch_*.py` scripts that
introduced those branches into `run.py`, and the trainers of the removed heads. `run_multitask.sh`
is kept although its `d021` arm uses the removed retreat switches.
