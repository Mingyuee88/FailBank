#!/usr/bin/env python3
"""E14 stage 0: hazard geometry and task feasibility on safety_hazard_avoidance (CPU physics, no policy).
For every level-1/2 task and all 50 init states: parse the cost predicates, measure the hazard's world
extent (contact geoms, rbound-conservative), the surface distances object/container/gripper -> hazard at
reset with the SAME env.check_distance the cost uses, and whether candidate hazard ellipsoids would
contain the object or the container (which would make the teacher block the task)."""
import os, sys, json, statistics
import numpy as np
sys.path.insert(0, "${SE_VLA_ROOT}")
sys.path.insert(0, "${SE_VLA_ROOT}/external/VLA-Arena")
from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv
OUT = sys.argv[1]
SU = "safety_hazard_avoidance"
suite = benchmark.get_benchmark_dict()[SU]()
report = {}

def flat(x):
    if isinstance(x, (list, tuple)):
        for y in x: yield from flat(y)
    else: yield x

def geoms_of(base, name):
    obj = base.get_object(name)
    names = list(getattr(obj, "contact_geoms", []) or [])
    ids = []
    for g in names:
        try: ids.append(base.sim.model.geom_name2id(g))
        except Exception: pass
    return names, ids

def aabb(base, ids):
    m, d = base.sim.model, base.sim.data
    c = np.array([d.geom_xpos[i] for i in ids]); r = np.array([m.geom_rbound[i] for i in ids])
    return (c - r[:, None]).min(0), (c + r[:, None]).max(0)

for lv in (1, 2):
    for tid in range(5):
        task = suite.get_task_by_level_id(lv, tid)
        states = suite.get_task_init_states(lv, tid)
        bddl = os.path.join(get_vla_arena_path("bddl_files"), task.problem_folder, f"level_{task.level}", task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        env.seed(17); env.reset()
        base = env.env if hasattr(env, "env") else env
        pp = base.parsed_problem
        cost = pp.get("cost_state", [])
        task_objs = list(pp.get("obj_of_interest", []))
        preds = []
        for st in cost:
            s = [str(x).lower() if i == 0 else x for i, x in enumerate(flat(st))] if isinstance(st, (list, tuple)) else [st]
            preds.append(s)
        dist_preds = [p for p in preds if p and p[0] in ("checkdistance", "checkgripperdistance")]
        hazards = sorted({p[2] if p[0] == "checkdistance" else p[1] for p in dist_preds})
        obj = [p[1] for p in dist_preds if p[0] == "checkdistance"]
        obj = obj[0] if obj else None
        container = [o for o in task_objs if o != obj]
        container = container[0] if container else None
        d_obj = [float(p[3]) for p in dist_preds if p[0] == "checkdistance"]
        d_grip = [float(p[2]) for p in dist_preds if p[0] == "checkgripperdistance"]
        rows = []
        for k, s0 in enumerate(states[:50]):
            env.reset(); env.set_init_state(s0)
            for h in hazards:
                hn, hid = geoms_of(base, h)
                lo, hi = aabb(base, hid)
                on, oid = geoms_of(base, obj) if obj else ([], [])
                cn, cid = geoms_of(base, container) if container else ([], [])
                d_o = base.check_distance(on, hn) if on else None
                d_c = base.check_distance(cn, hn) if cn else None
                d_g = base.check_gripper_distance(hn)
                r_obj = float(max(base.sim.model.geom_rbound[i] for i in oid)) if oid else 0.0
                oc = np.array(base.sim.data.body_xpos[base.obj_body_id[obj]]) if obj else None
                cc = np.array(base.sim.data.body_xpos[base.obj_body_id[container]]) if container else None
                ctr, half = (lo + hi) / 2, (hi - lo) / 2
                dd = max(d_obj) if d_obj else 0.0
                cand = {"C1_box_plus_d": half + dd, "C2_box_plus_d_plus_robj": half + dd + r_obj}
                inside = {}
                for nm, ax in cand.items():
                    inside[nm] = {"obj": bool(oc is not None and np.sum(((oc - ctr) / ax) ** 2) <= 1),
                                  "container": bool(cc is not None and np.sum(((cc - ctr) / ax) ** 2) <= 1)}
                rows.append(dict(state=k, hazard=h, lo=lo.round(3).tolist(), hi=hi.round(3).tolist(), d_obj_surface=d_o,
                                 d_container_surface=d_c, d_gripper=d_g, r_obj=r_obj,
                                 obj_center=None if oc is None else oc.round(3).tolist(),
                                 container_center=None if cc is None else cc.round(3).tolist(), inside=inside))
        env.close()
        key = f"L{lv}T{tid}"
        def col(f): return [f(r) for r in rows if f(r) is not None]
        summ = dict(bddl=task.bddl_file, hazards=hazards, object=obj, container=container, d_obj_thresh=d_obj, d_grip_thresh=d_grip,
                    preds=[p[:4] for p in preds],
                    hazard_extent_xyz=np.array([np.array(r["hi"]) - np.array(r["lo"]) for r in rows]).mean(0).round(3).tolist(),
                    d_obj_surface_min_med=[round(min(col(lambda r: r["d_obj_surface"])), 3), round(statistics.median(col(lambda r: r["d_obj_surface"])), 3)] if obj else None,
                    d_container_surface_min_med=[round(min(col(lambda r: r["d_container_surface"])), 3), round(statistics.median(col(lambda r: r["d_container_surface"])), 3)] if container else None,
                    init_obj_in_cost_zone=sum(1 for r in rows if d_obj and r["d_obj_surface"] is not None and r["d_obj_surface"] <= max(d_obj)),
                    container_in_cost_zone=sum(1 for r in rows if d_obj and r["d_container_surface"] is not None and r["d_container_surface"] <= max(d_obj)),
                    init_gripper_in_zone=sum(1 for r in rows if d_grip and r["d_gripper"] <= max(d_grip)),
                    C1_contains_obj=sum(r["inside"]["C1_box_plus_d"]["obj"] for r in rows), C1_contains_container=sum(r["inside"]["C1_box_plus_d"]["container"] for r in rows),
                    C2_contains_obj=sum(r["inside"]["C2_box_plus_d_plus_robj"]["obj"] for r in rows), C2_contains_container=sum(r["inside"]["C2_box_plus_d_plus_robj"]["container"] for r in rows),
                    r_obj=round(rows[0]["r_obj"], 3) if rows else None, n=len(rows))
        report[key] = dict(summary=summ, rows=rows)
        print(key, json.dumps(summ), flush=True)
json.dump(report, open(OUT, "w"), indent=1, default=float)
print("E14_GEOM_PROBE_DONE", OUT)
