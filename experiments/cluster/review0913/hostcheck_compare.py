#!/usr/bin/env python3
"""E13 host check: are pi0 L2 cells bit-identical on HOST_A vs HOST_B?
Compares the 12 re-run base cells against the E11 originals (same eval.sh, same ARM/episode id)."""
import json, os, sys
O = "${WORK_ROOT}/lora_dagger/review0913"
def key(p):
    r = json.load(open(p)); md = r.get("metric_decomposition") or {}; e = (r.get("episodes") or [{}])[0]
    return dict(status=r.get("status"), s=r.get("successes"), occ=md.get("official_cc"), pcc=md.get("policy_induced_cc"),
                steps=e.get("steps", e.get("episode_length", e.get("num_steps"))))
done = ok = 0; lines = []
for tag in ("apple", "lemon"):
    for off in (0, 1):
        for a in (0, 1, 2):
            new = f"{O}/e13/hostcheck/{tag}/base/off{off}_r{a}"
            ref = f"{O}/e11/pi0_l2_{tag}/base/off{off}_r{a}"
            if not os.path.exists(new + "/result.json"):
                lines.append(f"{tag} off{off}_r{a}: PENDING"); continue
            h = open(new + "/host.txt").read().strip()
            hr = open(ref + "/host.txt").read().strip()
            kn, kr = key(new + "/result.json"), key(ref + "/result.json")
            same = kn == kr and h.startswith("HOST_A") and hr.startswith("HOST_B")
            done += 1; ok += same
            lines.append(f"{tag} off{off}_r{a}: {'SAME' if same else 'DIFF'} new({h.split('.')[0]})={kn} ref({hr.split('.')[0]})={kr}")
print("\n".join(lines))
print(f"HOSTCHECK done={done}/12 identical={ok}")
if done == 12:
    print("HOSTCHECK_PASS" if ok == 12 else "HOSTCHECK_FAIL")
