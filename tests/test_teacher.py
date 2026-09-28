import numpy as np
import pytest

from failbank.teacher import FORMAL_CANDIDATE, TeacherContext, load_teacher
from failbank.teacher.oracle_geometry import Ellipsoid, project_translation

KW = dict(alpha=3.0, max_translation=1.0, eef_radius=0.03, obstacle_margin=0.02)


class _Data:
    def __init__(self, xpos):
        self.body_xpos = xpos


class _Sim:
    def __init__(self, xpos):
        self.data = _Data(xpos)


class FakeEnv:
    """Minimal stand-in for an Arena env: one hazard ("box_1") and one task object."""

    def __init__(self, hazard=(0.0, 0.0, 0.0), obj=(0.5, 0.5, 0.5)):
        self.sim = _Sim(np.array([hazard, obj], dtype=float))
        self.obj_body_id = {"box_1": 0, "mango_1": 1}
        self.parsed_problem = {"cost_state": [["incontact", "mango_1", "box_1"], ["fall", "box_1"]],
                               "obj_of_interest": ["mango_1"]}


def ctx(env, eef):
    return TeacherContext(env=env, obs={}, features={"eef_position": list(eef)}, step=10)


def test_far_from_hazard_is_identity_and_quiet():
    t = load_teacher("oracle_geometry")
    nominal = [0.1, -0.2, 0.3, 0.01, 0.02, 0.03, -1.0]
    p = t.propose(ctx(FakeEnv(), [1.0, 1.0, 1.0]), nominal)
    assert p.action == nominal and not p.triggered and p.projection["status"] == "nominal_safe"


def test_moving_into_hazard_triggers_and_only_edits_translation():
    t = load_teacher("oracle_geometry")
    nominal = [-1.0, 0.0, 0.0, 0.01, 0.02, 0.03, -1.0]   # towards the hazard at the origin
    p = t.propose(ctx(FakeEnv(), [0.12, 0.0, 0.0]), nominal)
    assert p.triggered and p.projection["status"] == "projected"
    assert p.action[3:] == nominal[3:]                   # rotation and gripper untouched
    assert p.action[0] > nominal[0]                      # pushed away from the hazard
    assert p.projection["min_barrier"] == pytest.approx(0.12 - (0.03 + 0.04 + 0.02))


def test_task_objects_are_not_obstacles():
    t = load_teacher("oracle_geometry")
    # eef right next to the task object only: no constraint
    p = t.propose(ctx(FakeEnv(hazard=(5, 5, 5), obj=(0, 0, 0)), [0.05, 0, 0]), [-1.0, 0, 0, 0, 0, 0, 1])
    assert not p.triggered


def test_clipping_is_not_a_trigger():
    t = load_teacher("oracle_geometry")
    p = t.propose(ctx(FakeEnv(), [1.0, 1.0, 1.0]), [2.0, 0.0, 0.0, 0, 0, 0, 1])
    assert p.action[0] == 1.0 and not p.triggered


def test_candidates_carry_formal_proposal_second():
    t = load_teacher("oracle_geometry")
    nominal = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    p = t.propose(ctx(FakeEnv(), [0.12, 0.0, 0.0]), nominal)
    names = [c.name for c in t.candidates(nominal, p)]
    assert names[:2] == ["nominal", FORMAL_CANDIDATE]
    assert t.candidates(nominal, p)[1].full_action == p.action


def test_projection_satisfies_the_barrier_when_feasible():
    rng = np.random.default_rng(1)
    for _ in range(500):
        c = rng.normal(0, 0.2, 3)
        eef = c + rng.normal(0, 0.3, 3)
        obs = [Ellipsoid.sphere(c, 0.04)]
        r = project_translation(rng.normal(0, 0.5, 3), eef, obs, **KW)
        away = eef - c
        n = away / np.linalg.norm(away)
        barrier = np.linalg.norm(away) - (0.03 + 0.04 + 0.02)
        assert n @ r.translation >= -3.0 * barrier - 1e-9 or np.any(np.abs(r.translation) >= 1.0 - 1e-12)


def test_custom_teacher_spec_is_validated():
    with pytest.raises(ValueError):
        load_teacher("not_a_module_spec")
    assert load_teacher("none") is None
