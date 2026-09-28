#!/usr/bin/env python3
"""Can the CBF correction be predicted from observations? Leave-one-task-out.

Direction matters here in a way it did not for the distance head: the mean unit correction
has |x| 0.398, |y| 0.475, |z| 0.530, so a fixed +z lift cannot express most of it. That is
why the hand-built retreat helped on L1-T1 and hurt on L1-T4.

Unlike the distance, the CBF correction is ACTION-CONDITIONED -- it is the minimal edit to
the action the policy proposed -- so nominal_action is an input, not just the observation.

Three baselines under one protocol, judged on what actually matters:
    gate accuracy  -- does it know WHEN a correction is needed (62% of steps need none)
    cosine         -- on steps that do need one, is the DIRECTION right
    magnitude MAE  -- and is the size right
MSE is not reported as the headline: 62% of targets are zero, so predicting zero everywhere
already scores well on it while being useless.
"""
import json, pathlib, numpy as np, torch, torch.nn as nn

LR = pathlib.Path("${WORK_ROOT}/lora_dagger")
rows = [json.loads(l) for l in (LR/"cbf_clone"/"index.jsonl").open()]
z = np.load(LR/"dist_head_multi"/"cache_96.npz", allow_pickle=True)
img, wri, stt = z["img"], z["wri"], z["state"]
assert len(rows) == len(stt), "cache and cbf index differ in length"

# the two extractions walked the same sorted banks; verify alignment rather than assume it
roots = {}
for probe in (0, len(rows)//2, len(rows)-1):
    r = rows[probe]
    root = roots.setdefault(r["root"], pathlib.Path(r["root"]))
    s = np.load(root/"blobs"/r["state"])
    assert np.allclose(s, stt[probe], atol=1e-6), "cache row %d does not match cbf index" % probe
print("alignment verified on 3 probes")

task = np.array([r["task"] for r in rows])
nominal = np.asarray([r["nominal"] for r in rows], dtype=np.float32)
delta = np.asarray([r["delta"] for r in rows], dtype=np.float32)
nrm = np.linalg.norm(delta, axis=1)
needs = (nrm >= 1e-9)
print("steps %d   need correction %.1f%%   mean|delta| on those %.4f"
      % (len(rows), 100*needs.mean(), nrm[needs].mean()))
TASKS = sorted(set(task.tolist()))
dev = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        f = 8 + 3   # state(8) + nominal translation(3), see extract_cbf.py
        if vision:
            def enc():
                return nn.Sequential(
                    nn.Conv2d(3,16,5,2,2), nn.ReLU(), nn.BatchNorm2d(16),
                    nn.Conv2d(16,32,3,2,1), nn.ReLU(), nn.BatchNorm2d(32),
                    nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.BatchNorm2d(64),
                    nn.Conv2d(64,64,3,2,1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.e1, self.e2 = enc(), enc(); f += 128
        self.trunk = nn.Sequential(nn.Linear(f,192), nn.ReLU(), nn.Linear(192,128), nn.ReLU())
        self.dhead = nn.Linear(128, 3)     # correction vector
        self.ghead = nn.Linear(128, 1)     # does this step need one at all
    def forward(self, a, b, s, u):
        parts = [s, u]
        if self.vision: parts = [self.e1(a), self.e2(b)] + parts
        h = self.trunk(torch.cat(parts, 1))
        return self.dhead(h), self.ghead(h).squeeze(1)

def run(kind, tr, te, epochs=10):
    if kind == "zero":
        return np.zeros((te.sum(), 3), np.float32), np.zeros(te.sum(), np.float32)
    m = Net(vision=(kind == "visual")).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xi = torch.from_numpy(img[tr]).permute(0,3,1,2).float().div_(255)
    Xw = torch.from_numpy(wri[tr]).permute(0,3,1,2).float().div_(255)
    Xs = torch.from_numpy(stt[tr]); Xu = torch.from_numpy(nominal[tr])
    Yd = torch.from_numpy(delta[tr]); Yg = torch.from_numpy(needs[tr].astype(np.float32))
    n, bs = len(Yd), 128
    for _ in range(epochs):
        perm = torch.randperm(n); m.train()
        for k in range(0, n, bs):
            i = perm[k:k+bs]
            pd, pg = m(Xi[i].to(dev), Xw[i].to(dev), Xs[i].to(dev), Xu[i].to(dev))
            loss = (nn.functional.smooth_l1_loss(pd, Yd[i].to(dev), beta=0.05)
                    + 0.3*nn.functional.binary_cross_entropy_with_logits(pg, Yg[i].to(dev)))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); D=[]; G=[]
    Ti = torch.from_numpy(img[te]).permute(0,3,1,2).float().div_(255)
    Tw = torch.from_numpy(wri[te]).permute(0,3,1,2).float().div_(255)
    Ts = torch.from_numpy(stt[te]); Tu = torch.from_numpy(nominal[te])
    with torch.no_grad():
        for k in range(0, len(Ts), 256):
            pd, pg = m(Ti[k:k+256].to(dev), Tw[k:k+256].to(dev), Ts[k:k+256].to(dev), Tu[k:k+256].to(dev))
            D.append(pd.cpu().numpy()); G.append(torch.sigmoid(pg).cpu().numpy())
    return np.concatenate(D), np.concatenate(G)

print("\n=== leave-one-task-out ===")
print("%-7s %-7s %10s %9s %11s" % ("held", "model", "gate acc", "cosine", "|mag| MAE"))
summ = {}
for ho in TASKS:
    te = (task == ho); tr = ~te
    yd, yn = delta[te], needs[te]
    for kind in ("zero", "state", "visual"):
        pd, pg = run(kind, tr, te)
        gate = ((pg > 0.5) == yn).mean() if kind != "zero" else (~yn).mean()
        sel = yn & (np.linalg.norm(pd, axis=1) > 1e-9)
        if sel.any():
            cos = float((pd[sel]*yd[sel]).sum(1).mean() /
                        max(1e-9, (np.linalg.norm(pd[sel],axis=1)*np.linalg.norm(yd[sel],axis=1)).mean()))
        else:
            cos = 0.0
        mag = float(np.abs(np.linalg.norm(pd,axis=1) - np.linalg.norm(yd,axis=1))[yn].mean()) if yn.any() else 0.0
        summ.setdefault(kind, []).append((gate, cos, mag))
        print("%-7s %-7s %10.3f %9.3f %11.4f" % (ho, kind, gate, cos, mag), flush=True)
    print("", flush=True)
print("=== averages over held-out tasks ===")
for k, v in summ.items():
    print("  %-7s gate %.3f   cosine %.3f   |mag| MAE %.4f"
          % (k, np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), np.mean([x[2] for x in v])))
print()
print("a fixed +z lift, for reference, has cosine = mean z-component of the unit target = 0.53")
