#!/usr/bin/env python3
"""Verify a trained checkpoint's ACTUAL architecture family by reading the parameter
tree, not `params/_METADATA`.

`_METADATA` lists only the PaliGemma vision tower (51 keys, identical for pi0 and
pi05, containing Dense_0/time_mlp and lacking state_proj/action_time_mlp). Any check
based on it passes pi05 unconditionally and fails pi0 unconditionally, regardless of
what the weights actually are -- which is how a pi0 run was trained as pi05 unnoticed.
"""
import sys, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
import orbax.checkpoint as ocp

# Markers must be anchored: "time_mlp" is a SUBSTRING of "action_time_mlp", so a bare
# "time_mlp" marker matches pi0 checkpoints too and makes every tree look ambiguous.
# Match on the path segment, not on a substring of it.
PI0_ONLY = ("/state_proj/", "/action_time_mlp_in/", "/action_time_mlp_out/")
PI05_ONLY = ("final_norm_1/Dense_0", "/time_mlp_in/", "/time_mlp_out/")

def leaves(params_dir):
    meta = ocp.PyTreeCheckpointer().metadata(Path(params_dir))
    out = []
    def walk(d, pref=()):
        if hasattr(d, "items"):
            for k, v in d.items():
                walk(v, pref + (k,))
        else:
            out.append("/".join(map(str, pref)))
    walk(meta)
    return ["/"+l+"/" for l in out]

def family_of(params_dir):
    ls = leaves(params_dir)
    pi0 = sum(any(p in l for p in PI0_ONLY) for l in ls)
    pi05 = sum(any(p in l for p in PI05_ONLY) for l in ls)
    if pi0 and not pi05:
        return "pi0", pi0, pi05, len(ls)
    if pi05 and not pi0:
        return "pi05", pi0, pi05, len(ls)
    return "ambiguous", pi0, pi05, len(ls)

if __name__ == "__main__":
    d = sys.argv[1]
    expect = sys.argv[2] if len(sys.argv) > 2 else None
    fam, n0, n05, n = family_of(d)
    print(f"VERIFY_FAMILY dir={d} leaves={n} pi0_markers={n0} pi05_markers={n05} family={fam}")
    if expect and fam != expect:
        print(f"FAMILY_MISMATCH expected={expect} got={fam}")
        sys.exit(1)
