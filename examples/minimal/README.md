# Minimal end-to-end example

`run_minimal.sh` runs the whole FailBank loop once, at toy scale, on one GPU:

1. **Stage 1** — six observe-only episodes of the Arena π0.5 base policy on
   `safety_static_obstacles` Level 1, task 2 (offsets 0–5), writing learning records;
2. **Stages 2–3** — derived records with fold 0 (offset 0 held out), a one-round bank, the
   admission gate (`--mode s1s2`);
3. **Stage 4** — one guarded 800-step LoRA update from the base checkpoint (batch 32,
   λ_q = 0, data seed 1), validated every 100 steps; the best candidate the guard accepts is
   folded into a standard checkpoint;
4. **evaluation** — base and updated policy on the held-out offset 0.

```bash
export FAILBANK_CHECKPOINTS=/path/to/checkpoints        # contains pi05_vla_arena_finetuned/
export PYTHONPATH=/path/to/VLA-Arena:$PYTHONPATH        # patched, see the top-level README
bash examples/minimal/run_minimal.sh                    # OUT=runs/minimal N_OFFSETS=6 STEPS=800
```

Requirements: one GPU with 48 GB (the update uses batch 32), ~60 GB of host RAM for the fold,
roughly one hour on an L40S. Each stage is skipped on re-run if its output exists. Outputs:

```
runs/minimal/collect/off*/      result.json, steps.jsonl, teacher.jsonl, records_commit.json
runs/minimal/records/           episodes/ + blobs/ (Stage 1) and derived/folds/offset_0/ (Stage 2a)
runs/minimal/bank/              the one-round bank (Stage 3)
runs/minimal/bank_s1s2/         the gated training set (Stage 2b)
runs/minimal/adapter/offset_0/  LoRA params + metrics.json (guard verdict)
runs/minimal/ckpt/              folded checkpoint
runs/minimal/eval/{base,updated}/off0/result.json
```

The update is trained on five episodes of one task and overfits quickly: at step 800 the
held-out flow ratio is about 1.28, above the guard's 1.10 limit, which is why the example
validates every 100 steps (`VAL_EVERY`) and keeps the best accepted point. If no point passes,
the run stops with `metrics_rejected.json` and a non-zero exit; that is the guard working. A single
held-out episode illustrates the pipeline; it is not evidence about the method. The paper's
adapters are trained on a two-round bank of 6,535 gated records.
