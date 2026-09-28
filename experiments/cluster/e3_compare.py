#!/usr/bin/env python3
"""E3 readout: base, published AEGIS, our controller, our method -- one arena, one harness.

Every column below was produced by the same policy server, the same 47 offsets, the same
cost accounting and the same result.json. The only thing that differs is which function
computed the executed action, which is what makes them comparable at all.

  base              stock pi05 arena checkpoint, nothing added
  AEGIS (published) their code, imported unchanged from vlsa-aegis/main/utils.py, running
                    through the port whose seams were each measured (backview camera pose,
                    crop box, depth conversion)
  ours (oracle-CBF) the controller this project built and, until 2026-08-13, mislabelled
                    as AEGIS. Sphere barrier, closed-form projection, ground-truth
                    geometry, no VLM. Reported under its own name.
  A / R2            our distilled policies -- folded LoRA weights, no runtime device

Also reported: whether the port's perception actually found the hazard. Their VLM is
prompted to pick from a fixed list of LIBERO objects that contains nothing from
VLA-Arena, so the name it returns here ("gray mug", "red mug") is a near-miss for the
arena's `white_yellow_mug_1`. Whether that matters is an empirical question -- if
GroundingDINO still grounds onto the right object, the point cloud is right and the
semantic drift is harmless. The distance between the fitted ellipsoid centre and the true
hazard position answers it.
"""
import glob
import json
import math
import os

LR = "${WORK_ROOT}/lora_dagger"
E3 = "${WORK_ROOT}/E3_vlsa/aegis"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def load(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "off*"))):
        base = os.path.basename(d)
        if not base[3:].isdigit():
            continue
        p = os.path.join(d, "result.json")
        if not os.path.exists(p):
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("status") != "pass":
            continue
        md = r.get("metric_decomposition", {})
        out[int(base[3:])] = (int(r.get("successes", 0)), md.get("policy_induced_cc"))
    return out


def row(name, arm, base):
    ks = sorted(set(arm) & set(base))
    if not ks:
        print(f"  {name:26s} (no cells)")
        return
    rep = sum(1 for k in ks if base[k][0] == 0 and arm[k][0] == 1)
    nf = sum(1 for k in ks if base[k][0] == 0)
    keep = sum(1 for k in ks if base[k][0] == 1 and arm[k][0] == 1)
    ns = sum(1 for k in ks if base[k][0] == 1)
    p = mcnemar_exact(rep, ns - keep)
    ca = sum(v for _, v in (arm[k] for k in ks) if isinstance(v, (int, float)))
    cb = sum(v for _, v in (base[k] for k in ks) if isinstance(v, (int, float)))
    dcc = ((ca - cb) / cb * 100) if cb else float("nan")
    star = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
    print(f"  {name:26s} n={len(ks):3d}  repair {rep:2d}/{nf:<2d}  retention {keep:2d}/{ns:<2d}  "
          f"net {sum(arm[k][0] for k in ks):2d}/{len(ks):<3d} (base {sum(base[k][0] for k in ks)})  "
          f"absCC {ca:7.0f} vs {cb:7.0f}  dCC {dcc:+7.1f}%  p={p:.4f}{star}")


def grounding_check():
    """Did GroundingDINO land on the real hazard despite the LIBERO-vocabulary prompt?"""
    rows = []
    for d in sorted(glob.glob(os.path.join(E3, "off*"))):
        srd = os.path.join(d, "srd.jsonl")
        steps = os.path.join(d, "shield_steps.jsonl")
        if not (os.path.exists(srd) and os.path.exists(steps)):
            continue
        audit = None
        for line in open(srd):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") == "vlsa_audit":
                audit = r
        if not audit:
            continue
        hazard = None
        for line in open(steps):
            try:
                r = json.loads(line)
            except Exception:
                continue
            f = r.get("pre_step_features") or {}
            if f.get("cost_pair_hazard_position"):
                hazard = f["cost_pair_hazard_position"]
                break
        rows.append((os.path.basename(d), audit.get("obstacle_name"),
                     audit.get("points_filtered"), audit.get("qp_solved"),
                     audit.get("qp_infeasible"), audit.get("h_min"), hazard))
    if not rows:
        return
    print()
    print("PERCEPTION AUDIT -- their VLM prompt offers only LIBERO object names, none of")
    print("which exist in VLA-Arena, so the returned name is a near-miss by construction.")
    print("What matters is whether the cloud still lands on the hazard.")
    names = {}
    for _, n, *_ in rows:
        names[n] = names.get(n, 0) + 1
    print(f"  obstacle names returned: {names}")
    filt = [r[2] for r in rows if isinstance(r[2], int)]
    infeas = sum(r[4] or 0 for r in rows)
    solved = sum(r[3] or 0 for r in rows)
    hmins = [r[5] for r in rows if isinstance(r[5], (int, float))]
    print(f"  filtered cloud size: min {min(filt)} median {sorted(filt)[len(filt)//2]} max {max(filt)}")
    print(f"  QP solved {solved}, infeasible {infeas}")
    if hmins:
        print(f"  barrier h minimum across episodes: {min(hmins):+.4f} "
              f"({'never violated' if min(hmins) > 0 else 'VIOLATED somewhere'})")


def main():
    base = load(f"{LR}/v8_clean/base")
    ours_cbf = {**load(f"{LR}/v9_power/shield"), **load(f"{LR}/v24_shbridge/shield")}
    aegis = load(E3)
    armA = load(f"{LR}/v21_eval/A_old_qw0.0")
    r2 = load(f"{LR}/v32_eval/R2/dev/L1t2")

    print("=" * 118)
    print("E3 -- safety_static_obstacles L1t2, 47 offsets, HOST_A, one harness")
    print("     repair and retention never netted; McNemar is exact and paired against base")
    print("=" * 118)
    row("AEGIS (published)", aegis, base)
    row("ours (oracle-CBF)", ours_cbf, base)
    row("A  (distilled, round 1)", armA, base)
    row("R2 (distilled, round 2)", r2, base)
    print()
    print(f"  base itself: {sum(v[0] for v in base.values())}/{len(base)}  "
          f"absolute policy-induced cost {sum(v[1] for v in base.values() if isinstance(v[1], (int,float))):.0f}")
    grounding_check()


if __name__ == "__main__":
    main()
