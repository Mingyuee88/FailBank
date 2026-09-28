# Third-party notices

FailBank is licensed under the Apache License 2.0 (`LICENSE`). It contains code adapted from the
projects below; their license texts are in `licenses/`. The repository does not redistribute
any of these projects: VLA-Arena and vlsa-aegis are cloned by the user and modified only by the
patches in `patches/` and `extras/aegis_baseline/patches/`.

## vlsa-aegis — MIT

* Source: https://github.com/THU-RCSCT/vlsa-aegis (verified against commit `57b1aef`)
* Copyright (c) 2023 Lifelong Robot Learning
* License: MIT, `licenses/vlsa-aegis-MIT.txt`
* Used in:
  * `src/failbank/teacher/oracle_geometry.py` — the sequential affine CBF projection
    (`project_translation`, `Ellipsoid`) is adapted from `vlsa_aegis.cbf_qp`; the file header
    carries the attribution.
  * `extras/aegis_baseline/failbank_aegis/vlsa_port.py` — glue that calls the published
    functions (`obstacle_detection`, `get_point_cloud`, `filtering_points`, `fit_ellipse`,
    `compute_h_coeffs_3d`) from the user's vlsa-aegis checkout and re-implements the
    per-step QP of `main_aegis_translational.py` / `main_aegis.py` with their constants.
  * `extras/aegis_baseline/patches/vlsa-aegis-57b1aef.patch` — changes to `main/utils.py` and
    `main/main_aegis_translational.py` (API key and endpoint from the environment, a call-site
    fix, an unshielded switch).
* The vlsa-aegis repository also vendors GroundingDINO, openpi and model checkpoints under their
  own licenses. FailBank uses none of those files directly; the AEGIS baseline loads
  GroundingDINO from the user's checkout.

## VLA-Arena — Apache-2.0

* Source: https://github.com/PKU-Alignment/VLA-Arena (commit `2ddcb00`)
* Copyright 2025 The VLA-Arena Authors
* License: Apache-2.0, `licenses/VLA-Arena-Apache-2.0.txt`
* Used in:
  * `patches/vla-arena-2ddcb00.patch` — modifies `vla_arena/models/openpi/evaluator.py` and ten
    scene XMLs (see README, "Install").
  * `src/failbank/configs/openpi_*.yaml` — derived from `vla_arena/configs/evaluation/openpi.yaml`.
  * `src/failbank/runtime/rollout.py` drives VLA-Arena's evaluator functions unchanged.

## OpenPI (as distributed with VLA-Arena) — Apache-2.0

* Source: `vla_arena/models/openpi` in VLA-Arena, derived from https://github.com/Physical-Intelligence/openpi
* License: Apache-2.0, `licenses/openpi-Apache-2.0.txt`
* Used in:
  * `src/failbank/train/guard.py` — `weighted_train_step` is OpenPI's `train_step` with the
    record-weighted objective; the guarded loop reuses OpenPI's `init_train_state`, sharding and
    optimizer utilities.
  * `src/failbank/train/derived_lora_data.py`, `lora_update.py` — use OpenPI's data transforms,
    configs, weight loaders and checkpointing.
