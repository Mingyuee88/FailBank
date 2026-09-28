#!/usr/bin/env python3
"""Download gs://openpi-assets/checkpoints/pi0_base via public HTTPS."""
import json, os, sys, urllib.request, urllib.parse, time

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi0_base/"
DEST = "${SE_VLA_ROOT}/checkpoints/pi0_base"

def listall(prefix):
    items, tok = [], None
    while True:
        u = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
             f"?prefix={urllib.parse.quote(prefix)}&maxResults=1000"
             f"&fields=items(name,size),nextPageToken")
        if tok:
            u += "&pageToken=" + tok
        d = json.load(urllib.request.urlopen(u, timeout=60))
        items += d.get("items", [])
        tok = d.get("nextPageToken")
        if not tok:
            break
    return items

items = listall(PREFIX)
total = sum(int(x["size"]) for x in items)
print(f"objects={len(items)} total={total/1e9:.2f}GB dest={DEST}", flush=True)

done_bytes = 0
for i, it in enumerate(items, 1):
    name, size = it["name"], int(it["size"])
    rel = name[len(PREFIX):]
    if not rel:
        continue
    out = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == size:
        done_bytes += size
        print(f"[{i}/{len(items)}] SKIP {rel} ({size/1e6:.1f}MB)", flush=True)
        continue
    url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}"
    t0 = time.time()
    tmp = out + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)
        got = os.path.getsize(tmp)
        if got != size:
            print(f"[{i}] SIZE MISMATCH {rel}: got {got} want {size}", flush=True)
            os.remove(tmp)
            sys.exit(2)
        os.rename(tmp, out)
        done_bytes += size
        dt = time.time() - t0
        print(f"[{i}/{len(items)}] OK {rel} {size/1e6:.1f}MB in {dt:.1f}s "
              f"({done_bytes/total*100:.1f}% total)", flush=True)
    except Exception as e:
        print(f"[{i}] FAIL {rel}: {e}", flush=True)
        sys.exit(3)

print(f"DOWNLOAD_COMPLETE bytes={done_bytes}", flush=True)
