import json
import math

from failbank.pipeline import build_derived as B
from failbank.pipeline import build_round_records as R
from failbank.pipeline import merge_bank as M
from failbank.runtime.metrics import decompose_episode


def raw_step(i, *, triggered=False, dist=0.3, cost=0.0, barrier=0.1, delta=0.0, progress=-0.001):
    nominal = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    executed = list(nominal)
    executed[0] += delta
    return {
        "step_index": 10 + i, "infer_call_index": i,
        "nominal_action": nominal, "executed_action": executed,
        "formal_projection": {"translation": executed[:3], "min_barrier": barrier,
                              "active_constraints": int(triggered),
                              "status": "projected" if triggered else "nominal_safe",
                              "triggered": triggered},
        "projection_candidates": [
            {"name": "nominal", "full_action": nominal, "action_delta_norm": 0.0, "min_barrier": None, "status": "identity/no_qp"},
            {"name": "late_margin", "full_action": executed, "action_delta_norm": abs(delta), "min_barrier": barrier, "status": "projected" if triggered else "nominal_safe"},
        ],
        "progress_delta": progress, "model_input_refs": {}, "raw_observation_refs": {},
        "nominal_action_chunk_ref": {"path": "x.npy"},
        "runtime_info": {"cost": cost, "cost_pair_min_distance": dist},
    }


def write_episode(root, eid, offset, success, steps):
    ed = root / "episodes" / eid[:2] / eid
    ed.mkdir(parents=True)
    result = root / f"result_{eid}.json"
    result.write_text(json.dumps({"init_state_offset": offset,
                                  "episodes": [{"success": success, "policy_induced_cost": 0.0}]}))
    (ed / "episode.json").write_text(json.dumps({"result_ref": {"path": str(result)}}))
    (ed / "raw_steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    (ed / "COMPLETE").write_text("{}")
    return ed


def test_stage_lead_is_offset_by_the_ten_settle_steps(tmp_path):
    # first risk crossing at row 50, i.e. step_index 60. The record at step_index 20 is 40
    # steps before it but is labelled 30 -- the documented, paper-preserving 10-step shift.
    steps = [raw_step(i, dist=0.05 if i >= 50 else 0.3) for i in range(60)]
    ed = write_episode(tmp_path, "aa" + "0" * 22, 0, False, steps)
    assert B.first_risk_step(ed) == 50
    recs, kept, _ = B.build_all(tmp_path, B.Thresholds())
    by_idx = {r["step_index"]: r for r in recs}
    assert kept == 1
    assert by_idx[20]["steps_to_first_risk"] == 30 and by_idx[20]["curriculum_stage"] == "S1_early"
    assert by_idx[21]["curriculum_stage"] == "S2_mid"          # 39 steps before the crossing
    assert by_idx[35]["curriculum_stage"] == "S2_mid"          # 25 steps before
    assert by_idx[36]["curriculum_stage"] == "S3_emergency"    # 24 steps before
    assert by_idx[51]["curriculum_stage"] == "POST_crossing"   # 9 steps BEFORE the crossing
    assert by_idx[60]["steps_to_first_risk"] == -10


def test_teacher_quality_rejects_untriggered_and_weights_triggered(tmp_path):
    steps = [raw_step(i, triggered=(i == 5), delta=0.2 if i == 5 else 0.0) for i in range(20)]
    ed = write_episode(tmp_path, "bb" + "0" * 22, 3, True, steps)
    recs = B.build_episode(ed, "3", True, B.Thresholds())
    assert recs[5]["triggered"] and recs[5]["teacher"]["gate"] and recs[5]["teacher"]["weight"] > 0
    assert all(r["teacher"]["weight"] == 0.0 for i, r in enumerate(recs) if i != 5)
    assert all(r["quiet"] == (not r["triggered"]) for r in recs)


def test_keep_record_rules():
    base = {"nominal_action": [0.1, 0, 0, 0, 0, 0, -1], "executed_action": [0.3, 0, 0, 0, 0, 0, -1],
            "formal_projection": {"triggered": True}}
    assert not R.keep_record({**base, "curriculum_stage": "POST_crossing", "eventual_success": True})
    assert not R.keep_record({**base, "curriculum_stage": "S3_emergency", "eventual_success": True})
    assert R.keep_record({**base, "curriculum_stage": "S2_mid", "eventual_success": True})
    # failure trajectory: only a real S1 correction survives
    assert R.keep_record({**base, "curriculum_stage": "S1_early", "eventual_success": False})
    assert not R.keep_record({**base, "curriculum_stage": "S2_mid", "eventual_success": False})
    assert not R.keep_record({**base, "curriculum_stage": "S1_early", "eventual_success": False,
                              "executed_action": [0.1, 0, 0, 0, 0, 0, -1]})


def test_merge_bank_accumulates_and_dedupes(tmp_path):
    for tag, ids in (("r1", ["a", "b"]), ("r2", ["b", "c"])):
        fold = tmp_path / tag / "derived" / "folds" / "offset_0"
        fold.mkdir(parents=True)
        (tmp_path / tag / "blobs").mkdir()
        (tmp_path / tag / "episodes").mkdir()
        rows = [{"record_id": i, "formal_projection": {"triggered": True}, "curriculum_stage": "S1_early"} for i in ids]
        (fold / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (fold / "validation.jsonl").write_text("")
    out = tmp_path / "bank"
    M.main(["--round", f"r1={tmp_path / 'r1'}", "--round", f"r2={tmp_path / 'r2'}", "--out", str(out)])
    rows = [json.loads(l) for l in (out / "derived/folds/offset_0/train.jsonl").read_text().splitlines()]
    assert [(r["record_id"], r["bank_round"]) for r in rows] == [("a", "r1"), ("b", "r1"), ("c", "r2")]
    assert (out / "episodes" / "r1").is_symlink()


def test_policy_induced_cost_excludes_only_persistent_terminal_settlement():
    v = lambda on: {"cost_predicate_values": [{"predicate": "fall:box_1", "value": on}]}
    # box fallen from the first step and still fallen at the terminal settlement: initial cost
    rows = [{"step_idx": i, "step_cost": 0.0, "done": False, "pre_step_features": v(True)} for i in range(5)]
    rows[-1].update(step_cost=10.0, done=True)
    d = decompose_episode(rows)
    assert d["official_cost"] == 10.0 and d["policy_induced_cost"] == 0.0
    # the same cost mid-episode is policy-induced
    rows[2]["step_cost"] = 3.0
    assert decompose_episode(rows)["policy_induced_cost"] == 3.0
    assert math.isclose(decompose_episode(rows)["official_cost"], 13.0)
