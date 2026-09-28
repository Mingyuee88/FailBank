# Published AEGIS as an in-loop baseline (optional)

The paper compares FailBank against AEGIS (vlsa-aegis): GLM-4.5V names the most likely
obstruction once per episode, GroundingDINO + RGB-D build its point cloud, an ellipsoid is
fitted, and a CBF-QP filters every action. Here it runs behind FailBank's teacher interface
and **in the loop** (`--execute teacher`): the environment executes the shield's action.

FailBank itself never needs any of this.

## Setup

```bash
pip install -e "/path/to/failbank[aegis]"        # cvxpy, scipy, httpx; installs failbank_aegis
git clone https://github.com/THU-RCSCT/vlsa-aegis.git && cd vlsa-aegis && git checkout 57b1aef
git apply /path/to/failbank/extras/aegis_baseline/patches/vlsa-aegis-57b1aef.patch
# install GroundingDINO and its SwinT-OGC weights as described in vlsa-aegis' README, so that
#   GroundingDINO/GroundingDINO_SwinT_OGC.py and GroundingDINO/groundingdino_swint_ogc.pth exist
export AEGIS_ROOT=/path/to/vlsa-aegis
# a GLM-4.5V endpoint: either a local OpenAI-compatible server (we served GLM-4.5V with vLLM)
export AEGIS_GLM_BASE_URL=http://<host>:<port>/v1
# or the hosted API
export ZHIPUAI_API_KEY=...
```

The patch reads the API key from the environment (upstream asks for it to be pasted into
`utils.py`), adds the local-endpoint path, strips a leaked reasoning block from the answer,
fixes a call-site bug in `main_aegis_translational.py` (`filtering_points` called with one
argument) and adds `AEGIS_DISABLE_SHIELD=1` for an unshielded run of their own harness.

## Run a cell

```bash
failbank-rollout --output runs/aegis/L1t2/off0_r0/result.json --task-level 1 --task-id 2 --offset 0 \
    --teacher failbank_aegis:VlsaAegisTeacher --execute teacher
```

The teacher sets `FAILBANK_AEGIS_CAMERAS=1`, which makes the patched evaluator add the
`backview` camera and depth planes that AEGIS perception needs. `teacher.jsonl` ends with a
`teacher_audit` row (obstacle name, point counts, QP solves); check it before reading a null
result as a property of the method. `AEGIS_DOF=6` selects their full 6-DoF formulation; the
paper reports the translational variant (`AEGIS_DOF=3`, the default). Only suites with a
measured crop box are accepted (`safety_static_obstacles`, `safety_state_preservation`,
`safety_hazard_avoidance`).

Status: this wrapper is a direct port of the adapter that produced the paper's AEGIS columns;
unlike the core it was not re-run end to end in the release layout (it needs the VLM service).
