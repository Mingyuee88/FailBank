#!/usr/bin/env python3
"""Fetch params/ + assets/ of VLA-Arena/pi0-vla-arena-fintuned (skip train_state/, inference-only).

Mirrors how the pi05 checkpoint was obtained: optimizer train_state is intentionally omitted.
"""
import json, os, sys, time, urllib.request, urllib.parse

REPO = "VLA-Arena/pi0-vla-arena-fintuned"
DEST = "${SE_VLA_ROOT}/checkpoints/pi0_vla_arena_finetuned"
KEEP = ("params/", "assets/")

def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    return json.load(urllib.request.urlopen(req, timeout=60))

info = api(f"https://huggingface.co/api/models/{urllib.parse.quote(REPO)}?blobs=true")
sha = info.get("sha", "main")
sib = info.get("siblings") or []
want = [s for s in sib if s.get("rfilename", "").startswith(KEEP)]
skipped = [s for s in sib if s not in want]
total = sum((s.get("size") or 0) for s in want)
print(f"repo={REPO} sha={sha[:12]}", flush=True)
print(f"keep={len(want)} files {total/1e9:.2f}GB | skip={len(skipped)} files "
      f"{sum((s.get('size') or 0) for s in skipped)/1e9:.2f}GB (train_state etc)", flush=True)

done = 0
for i, s in enumerate(sorted(want, key=lambda x: x["rfilename"]), 1):
    rel, size = s["rfilename"], (s.get("size") or 0)
    out = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == size:
        done += size
        print(f"[{i}/{len(want)}] SKIP {rel}", flush=True)
        continue
    url = (f"https://huggingface.co/{REPO}/resolve/{sha}/"
           f"{urllib.parse.quote(rel)}?download=true")
    tmp = out + ".part"
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)
        got = os.path.getsize(tmp)
        if size and got != size:
            print(f"[{i}] SIZE MISMATCH {rel}: got {got} want {size}", flush=True)
            os.remove(tmp); sys.exit(2)
        os.rename(tmp, out)
        done += got
        print(f"[{i}/{len(want)}] OK {rel} {got/1e6:.1f}MB in {time.time()-t0:.1f}s "
              f"({done/total*100:.1f}%)", flush=True)
    except Exception as e:
        print(f"[{i}] FAIL {rel}: {type(e).__name__} {e}", flush=True)
        sys.exit(3)

print(f"HF_DOWNLOAD_COMPLETE bytes={done}", flush=True)
