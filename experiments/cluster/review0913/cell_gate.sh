#!/usr/bin/env bash
#$ -N RvGate
#$ -cwd
#$ -pe smp 1
#$ -j y
# Review 2026-09-15, E11: gate the full AEGIS-on-pi0 arrays on smoke cells.
# CELLS: comma-separated cell directories. Each must have status pass, architecture_family pi0,
# and a vlsa_audit row with perception_ok true. Exit 100 (dependents stay held) otherwise.
set -uo pipefail
: "${CELLS:?set CELLS}"
python3 - "$CELLS" <<'PY'
import json, os, sys
bad = []
for d in sys.argv[1].split(","):
    f = os.path.join(d, "result.json")
    if not os.path.exists(f):
        bad.append((d, "no result.json")); continue
    r = json.load(open(f))
    fam = (r.get("architecture") or {}).get("architecture_family")
    ok = None
    s = os.path.join(d, "srd.jsonl")
    if os.path.exists(s):
        for line in open(s, errors="ignore"):
            if "vlsa_audit" in line:
                try: ok = json.loads(line).get("perception_ok")
                except Exception: ok = False
                break
    print("GATE_CELL", d, "status=%s family=%s perception_ok=%s successes=%s" % (r.get("status"), fam, ok, r.get("successes")))
    if r.get("status") != "pass" or fam != "pi0" or ok is not True:
        bad.append((d, "status=%s family=%s perception_ok=%s" % (r.get("status"), fam, ok)))
if bad:
    print("GATE_FAIL", bad); sys.exit(100)
print("GATE_PASS")
PY
rc=$?
exit $rc
