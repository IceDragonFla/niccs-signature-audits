# hint_perrow.py — per-row label-scale fitting + escalating k-error finisher.
#
# The hint-channel operator's signal scale is row-dependent (the real-data
# effective slope varies by row: the tier-1 T statistic gave kappa_eff
# ~ 1.07/1.10/0.86 for rows 0/1/2), so each row gets its own clustering fit.
# With per-row scales the solve residual drops to ~0.18 R-units and the
# direct misclassification count falls to 1/4/2 per row; the finisher then
# runs an escalating k-error search per row (k=1..5) over the worst-margin
# coordinates, accepting ONLY a zero-nbad correction set.
import itertools
import json
import sys
import numpy as np

D, Q, TAU, G2 = 256, 26881, 30, 682
NSIG = 1_000_000
M = NSIG * TAU
DATA = sys.argv[1] if len(sys.argv) > 1 else "real"

g16 = np.fromfile("tmp/bit128_1m/ground.bin", dtype="<i2").astype(np.int64)
s0_true = g16[:3 * D].reshape(3, D)
b1_true = g16[6 * D:9 * D].reshape(3, D)
A0 = g16[12 * D:].reshape(3, 3, D)

def negacyc(a, s):
    conv = np.convolve(a, s)
    conv = np.concatenate([conv, [0]])
    return (conv[:D] - conv[D:2 * D]) % Q

def center(x):
    return ((x % Q) + Q // 2) % Q - Q // 2

tv = np.zeros((3, D), np.int64)
for i in range(3):
    acc = np.zeros(D, np.int64)
    for p in range(3):
        acc = (acc + negacyc(A0[i, p], s0_true[p])) % Q
    tv[i] = acc
R_true = np.stack([center(tv[i] - 16 * (b1_true[i] % Q)) for i in range(3)])

Fop = np.load("tmp/attack2/Fop_150000.npy").astype(np.float64)
Phi = np.load(f"tmp/attack2/phi_hint_{DATA}_1000000.npy").astype(np.float64)
import os
import json as _json
if len(sys.argv) > 2:                      # extra same-key dataset (e.g. sim3m)
    _tag = sys.argv[2]
    _meta = _json.load(open(f"tmp/attack2/phi_hint_{_tag}_meta.json"))
    Phi = Phi + np.load(f"tmp/attack2/phi_hint_{_tag}.npy").astype(np.float64)
    NSIG += int(_meta["nsig"])
    M = NSIG * TAU                          # keep the signal scale consistent
    print(f"[data] combined with {_tag} (n={_meta['nsig']}): N={NSIG}, M={M}",
          flush=True)
m_axis = np.arange(D)
N_F = NSIG * Fop
b1f = (16 * (b1_true % Q)).reshape(3 * D)

# ---------------- A0 inverse
r_ = np.arange(D)
Mx = np.zeros((3 * D, 3 * D), np.int64)
for i in range(3):
    for p in range(3):
        idxm = (r_[:, None] - r_[None, :]) % D
        sgn = np.where(r_[None, :] > r_[:, None], -1, 1)
        Mx[i * D:(i + 1) * D, p * D:(p + 1) * D] = A0[i, p][idxm] * sgn
n = 3 * D
Aaug = np.concatenate([Mx % Q, np.eye(n, dtype=np.int64)], axis=1).astype(np.int64)
for k in range(n):
    piv = k + int(np.argmax(Aaug[k:, k] != 0))
    Aaug[[k, piv]] = Aaug[[piv, k]]
    inv = pow(int(Aaug[k, k]), Q - 2, Q)
    Aaug[k] = (Aaug[k] * inv) % Q
    col = Aaug[:, k].copy()
    col[k] = 0
    Aaug = (Aaug - col[:, None] * Aaug[k][None, :]) % Q
Ainv = Aaug[:, n:]

def nbad_of(Rv):
    c = center((Ainv @ ((b1f + Rv.reshape(3 * D)) % Q)) % Q)
    return int(np.sum((c < -1) | (c > 1)))

# ---------------- per-row scale fit
Rint = np.zeros((3, D), np.int64)
Rhat = np.zeros((3, D))
s_row = []
for i in range(3):
    phi = Phi[0, i].copy()
    if i == 0:
        phi = phi + (0.924 * 0.5 * M * (1.0 - 2.0 * m_axis / D)) / 682.0  # b-term scales with N and s
    rhs = -G2 * phi
    best = None
    for s in np.arange(0.88, 1.001, 0.001):
        Rh = np.linalg.solve(s * M * np.eye(D) + N_F, rhs)
        o = float(np.cos(2 * np.pi * Rh).mean())
        if best is None or o > best[0]:
            best = (o, float(s), Rh)
    o, s, Rh = best
    s_row.append(s)
    Rhat[i] = Rh
    Rint[i] = np.clip(np.rint(Rh), -9, 8).astype(np.int64)
    wr = int((Rint[i] != R_true[i]).sum())
    print(f"[fit] row{i}: s={s:.3f} obj={o:.4f} direct wrong={wr}/256 "
          f"resid={float((Rh - R_true[i]).std()):.3f}", flush=True)

# ---------------- escalating per-row k-error finisher
Rcur = Rint.copy()
nbad = nbad_of(Rcur)
print(f"[finisher] start nbad={nbad}", flush=True)
stage_log = []
for i in range(3):
    if nbad == 0:
        break
    margins = np.abs(Rhat[i] - Rcur[i])
    W = 10
    worst = list(np.argsort(margins)[::-1][:W])
    plans = [(1, 9), (2, 3), (3, 2), (4, 2), (5, 2)]
    done_row = False
    for k, dmax in plans:
        if nbad == 0:
            break
        best = None
        base = Rcur[i].copy()
        for combo in itertools.combinations(worst, k):
            for dset in itertools.product(range(-dmax, dmax + 1), repeat=k):
                if all(d == 0 for d in dset):
                    continue
                Rtry = Rcur.copy()
                for w, d in zip(combo, dset):
                    nv = int(base[w]) + d
                    if nv < -9 or nv > 8:
                        break
                    Rtry[i, w] = nv
                else:
                    nb = nbad_of(Rtry)
                    if best is None or nb < best[0]:
                        best = (nb, i, combo, dset)
                        if nb == 0:
                            break
            if best is not None and best[0] == 0:
                break
        if best is not None:
            nb, ii, combo, dset = best
            if nb < nbad:
                if nb == 0:
                    for w, d in zip(combo, dset):
                        Rcur[i, w] = int(Rcur[i, w]) + d
                    nbad = 0
                    stage_log.append({"row": i, "k": k, "nb": nb,
                                      "fixes": [[int(w), int(d)] for w, d in zip(combo, dset)]})
                    print(f"[finisher] row{i} k={k}: CLOSED "
                          f"({[[int(w), int(d)] for w, d in zip(combo, dset)]})", flush=True)
                    done_row = True
                    break
                else:
                    print(f"[finisher] row{i} k={k}: best nb={nb} (not applied)",
                          flush=True)
                    stage_log.append({"row": i, "k": k, "nb": nb})
    if not done_row and nbad > 0:
        print(f"[finisher] row{i}: no zero-nbad fix found", flush=True)

print(f"[finisher] final nbad={nbad}", flush=True)
s0v = (Ainv @ ((b1f + Rcur.reshape(3 * D)) % Q)) % Q
c = center(s0v).reshape(3, D)
n_match = int((c == s0_true).sum())
exact = bool(np.array_equal(c, s0_true))
print(f"[final] s0 match {n_match}/768 exact={exact}", flush=True)

out = {"data": DATA, "s_row": s_row, "stage_log": stage_log,
       "nbad_final": nbad, "s0_match": n_match, "s0_exact": exact}
if exact:
    Rcur_i = Rcur
    ehat = np.where(Rcur_i >= 8, -1, np.where(Rcur_i <= -9, 1, 0)).astype(np.int64)
    b0hat = Rcur_i + ehat
    assert np.all((b0hat >= -8) & (b0hat < 8)) and np.abs(ehat).max() <= 1
    with open(f"tmp/attack2/key_hat_h_{DATA}.bin", "wb") as fh:
        for p in range(3):
            fh.write(c[p].astype(np.int16).tobytes())
        for i in range(3):
            fh.write(ehat[i].astype(np.int16).tobytes())
        for i in range(3):
            fh.write(b0hat[i].astype(np.int16).tobytes())
    out["key_written"] = True
    out["e_hat_nonzero"] = int((ehat != 0).sum())
    print(f"[final] equivalent key written (e_hat nonzero at "
          f"{int((ehat != 0).sum())} coords)", flush=True)
json.dump(out, open(f"tmp/attack2/hint_perrow_{DATA}.json", "w"), indent=1)
print("[done]", flush=True)
