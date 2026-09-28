"""Why did retreating make crashes worse? Look at what happens after the trigger."""
import json, glob, pathlib, statistics as st
R = pathlib.Path("${WORK_ROOT}/lora_dagger/t3_safe/N8_G0.6")
fired = []
for d in sorted(glob.glob(str(R / "off*_r*"))):
    p = pathlib.Path(d)
    srd = p / "srd.jsonl"
    if not srd.exists(): continue
    at = None; nogeo = 0
    dists = []
    for line in srd.open():
        try: x = json.loads(line)
        except Exception: continue
        if x.get("type") == "contact_abort" and at is None: at = x.get("step")
        pre = x.get("pre_step_features") or {}
        d_ = pre.get("cost_pair_min_distance")
        if d_ is not None: dists.append(d_)
    if at is None: continue
    r = json.load(open(p / "result.json")); md = r.get("metric_decomposition") or {}
    fired.append((p.name, at, int((r.get("successes") or 0) > 0),
                  float(md.get("policy_induced_cc") or 0), dists))
print("cells where the retreat fired: %d\n" % len(fired))
print("%-14s %8s %5s %9s %28s" % ("cell", "fired@", "sr", "polCC", "cost_pair dist: at fire -> end"))
for name, at, sr, cc, dists in fired[:12]:
    if not dists: continue
    a = dists[min(at, len(dists)-1)] if at < len(dists) else dists[-1]
    tail = dists[min(at, len(dists)-1):]
    print("%-14s %8d %5d %9.0f    %.4f -> %.4f (min %.4f over %d steps)"
          % (name, at, sr, cc, a, dists[-1], min(tail) if tail else float("nan"), len(tail)))
if fired:
    after = [len(d) - min(a, len(d)) for _, a, _, _, d in fired if d]
    print("\nsteps remaining after the trigger: median %.0f (the retreat keeps running to the end)"
          % st.median(after))
    closed = 0; opened = 0
    for _, a, _, _, d in fired:
        if not d or a >= len(d): continue
        start = d[min(a, len(d)-1)]; end = d[-1]
        if end > start: opened += 1
        else: closed += 1
    print("cells where distance INCREASED after retreat (escaped): %d" % opened)
    print("cells where distance did NOT increase (still pressed):  %d" % closed)
