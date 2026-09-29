# Minimal end-to-end example

Runs the whole FailBank loop once at toy scale, to check that your setup works.

| Step | What happens |
|---|---|
| Collect | 6 observe-only episodes of the π0.5 base policy (Level 1 Mango, offsets 0–5) |
| Build | learning records → one-round bank → admission gate (offset 0 held out) |
| Update | guarded LoRA update (800 steps, validated every 100), folded into a checkpoint |
| Evaluate | base and updated policy on the held-out offset 0 |

```bash
export FAILBANK_CHECKPOINTS=/path/to/checkpoints     # contains pi05_vla_arena_finetuned/
export PYTHONPATH=/path/to/VLA-Arena:$PYTHONPATH     # patched, see the top-level README
bash examples/minimal/run_minimal.sh
```

Needs one 48 GB GPU and ~60 GB host RAM, and takes about one hour on an L40S. Finished steps are
skipped on re-run. Results go to `runs/minimal/`.

> [!NOTE]
> The final evaluation is a single episode. It shows that the pipeline runs end to end, and it is
> not a performance result. The paper's numbers come from 6,535 training records and full
> evaluation over every initial state.
