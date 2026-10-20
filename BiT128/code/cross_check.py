# cross_check.py — independent cross-validation battery for the hint-channel
# attack conclusions. Every check uses an INDEPENDENT code path or dataset
# combination from the main pipeline:
#   X1  cross-round key identity: stage-E oracle key == hint-channel key
#   X2  N=2M replication (real + sim1m): per-row fit, global staged finisher,
#       recovered key must equal key_hat_h_real.bin byte-for-byte
#   X4  fresh FFT-based algebra check: A0 s0 + e_hat == 16 b1 + b0_hat (mod q)
#       with e_hat in {-1,0,1}, b0_hat in [-8,8)
#   X5  blocking independence: fast accumulator with two block sizes agrees
#   X6  cross-dataset agreement: per-dataset R-hat estimates (real-1M and
#       sim3m-3M, fitted independently) agree with each other and with the
#       combined estimate
# Scoring (R_true, s0_true) is used ONLY for reporting, never for decisions.
import hashlib
import itertools
import json
import numpy as np

D, Q, TAU, G2 = 256, 26881, 30, 682
OUT = {"checks": {}}

def center(x):
    return ((x % Q) + Q // 2) % Q - Q // 2

def negacyc_fft(a, s):
    """Independent negacyclic product via FFT with [x, -x] extension."""
    af = np.fft.rfft(a, 2 * D)
    se = np.concatenate([s, -s])
    conv = np.fft.irfft(af * np.fft.rfft(se), 2 * D)
    return np.rint(conv[:D]).astype(np.int64) % Q

g16 = np.fromfile("tmp/bit128_1m/ground.bin", dtype="<i2").astype(np.int64)
s0_true = g16[:3 * D].reshape(3, D)
e_true = g16[3 * D:6 * D].reshape(3, D)
b1_true = g16[6 * D:9 * D].reshape(3, D)
b0_true = g16[9 * D:12 * D].reshape(3, D)
A0 = g16[12 * D:].reshape(3, 3, D)

tv = np.zeros((3, D), np.int64)
for i in range(3):
    acc = np.zeros(D, np.int64)
    for p in range(3):
        acc = (acc + negacyc_fft(A0[i, p], s0_true[p])) % Q
    tv[i] = acc
R_true = np.stack([center(tv[i] - 16 * (b1_true[i] % Q)) for i in range(3)])
print("[base] FFT-derived R == b0-e:", bool(np.array_equal(R_true, b0_true - e_true)),
      flush=True)
OUT["checks"]["fft_R_equals_b0_minus_e"] = bool(np.array_equal(R_true, b0_true - e_true))

# ---------------------------------------------------------------- X1
h_orc = hashlib.sha256(open("tmp/attack2/key_hat_oracle.bin", "rb").read()).hexdigest()
h_stagee = hashlib.sha256(open("tmp/attack2/key_hat_h_oracle.bin", "rb").read()).hexdigest()
h_att = hashlib.sha256(open("tmp/attack2/key_hat_h_real.bin", "rb").read()).hexdigest()
x1 = {"stageE_oracle_sha": h_orc, "hint_oracle_sha": h_stagee, "hint_attack_sha": h_att,
      "stageE_equals_hint_oracle": h_orc == h_stagee,
      "hint_oracle_equals_hint_attack": h_stagee == h_att}
OUT["checks"]["X1_key_identity"] = x1
print(f"[X1] stageE==hint_oracle: {h_orc == h_stagee}, "
      f"hint_oracle==hint_attack: {h_stagee == h_att}", flush=True)

# ---------------------------------------------------------------- helpers
Fop = np.load("tmp/attack2/Fop_150000.npy").astype(np.float64)
m_axis = np.arange(D)
b1f = (16 * (b1_true % Q)).reshape(3 * D)

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

def fit_and_attack(Phi, N, tag):
    """Per-row scale fit, global staged finisher; returns recovered key bytes."""
    M = N * TAU
    N_F = N * Fop
    Rint = np.zeros((3, D), np.int64)
    Rhat = np.zeros((3, D))
    s_row = []
    for i in range(3):
        phi = Phi[0, i].copy()
        if i == 0:
            phi = phi + (0.924 * 0.5 * M * (1.0 - 2.0 * m_axis / D)) / 682.0
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
    nbad = nbad_of(Rint)
    direct_wrong = {i: int((Rint[i] != R_true[i]).sum()) for i in range(3)}
    print(f"[{tag}] N={N} s_row={[round(v,3) for v in s_row]} "
          f"direct wrong={list(direct_wrong.values())} nbad={nbad}", flush=True)

    def scan(k, dmax, W):
        margins = np.abs(Rhat - Rint).reshape(3 * D)
        worst = list(np.argsort(margins)[::-1][:W])
        best = None
        for combo in itertools.combinations(worst, k):
            for dset in itertools.product(range(-dmax, dmax + 1), repeat=k):
                if all(d == 0 for d in dset):
                    continue
                Rtry = Rint.reshape(3 * D).copy()
                okflag = True
                for w, d in zip(combo, dset):
                    nv = int(Rtry[w]) + d
                    if nv < -9 or nv > 8:
                        okflag = False
                        break
                    Rtry[w] = nv
                if not okflag:
                    continue
                nb = nbad_of(Rtry)
                if best is None or nb < best[0]:
                    best = (nb, list(zip(combo, dset)))
                    if nb == 0:
                        return best
        return best

    Rcur = Rint.reshape(3 * D).copy()
    stages = []
    for k, dmax, W in ((1, 9, 3 * D), (2, 3, 12), (3, 2, 12)):
        if nbad == 0:
            break
        res = scan(k, dmax, W)
        if res is not None and res[0] == 0:
            nb, fixes = res
            for w, d in fixes:
                Rcur[w] += d
            nbad = 0
            stages.append({"k": k, "nb": nb})
            print(f"[{tag}] stage k={k}: CLOSED", flush=True)
            break
        stages.append({"k": k, "nb": res[0] if res else None})
        print(f"[{tag}] stage k={k}: best nb={res[0] if res else None} (not applied)",
              flush=True)
    s0v = (Ainv @ ((b1f + Rcur % Q) % Q)) % Q
    c = center(s0v)
    exact = bool(np.array_equal(c.reshape(3, D), s0_true))
    keybytes = b"".join((c.astype(np.int16)).tobytes()
                        for _ in (0,))  # placeholder, replaced below
    keybytes = c.astype(np.int16).tobytes()
    eh = np.where(Rcur.reshape(3, D) >= 8, -1,
                  np.where(Rcur.reshape(3, D) <= -9, 1, 0)).astype(np.int64)
    b0h = Rcur.reshape(3, D) + eh
    keybytes += eh.astype(np.int16).tobytes() + b0h.astype(np.int16).tobytes()
    return {"tag": tag, "N": N, "s_row": s_row, "direct_wrong": direct_wrong,
            "stages": stages, "nbad_final": nbad, "s0_exact": exact,
            "key_sha": hashlib.sha256(keybytes).hexdigest()}

# ---------------------------------------------------------------- X2
Pr = np.load("tmp/attack2/phi_hint_real_1000000.npy").astype(np.float64)
Ps1 = np.load("tmp/attack2/phi_hint_sim_1000000.npy").astype(np.float64)
OUT["checks"]["X2_replication_2M"] = fit_and_attack(Pr + Ps1, 2_000_000, "real+sim1m")
OUT["checks"]["X2_key_equals_main"] = (
    OUT["checks"]["X2_replication_2M"]["key_sha"] == h_att)
print(f"[X2] recovered key == key_hat_h_real: "
      f"{OUT['checks']['X2_key_equals_main']}", flush=True)

# ---------------------------------------------------------------- X4
arr = np.fromfile("tmp/attack2/key_hat_h_real.bin", dtype="<i2").astype(np.int64)
s0r, eh, b0h = arr[:768].reshape(3, D), arr[768:1536].reshape(3, D), arr[1536:].reshape(3, D)
tvk = np.zeros((3, D), np.int64)
for i in range(3):
    acc = np.zeros(D, np.int64)
    for p in range(3):
        acc = (acc + negacyc_fft(A0[i, p], s0r[p])) % Q
    tvk[i] = acc
lhs = center(tvk[i] if False else 0) if False else None
resid = center(tvk - 16 * (b1_true % Q) + eh - b0h)
x4 = {"algebra_identity_holds": bool(np.all(resid == 0)),
      "e_hat_in_range": bool(np.all(np.abs(eh) <= 1)),
      "b0_hat_in_range": bool(np.all((b0h >= -8) & (b0h < 8))),
      "s0_matches_truth": bool(np.array_equal(s0r, s0_true)),
      "e_hat_nonzero": int((eh != 0).sum())}
OUT["checks"]["X4_algebra"] = x4
print(f"[X4] A0*s0+eh-16b1-b0h == 0 (FFT path): {x4['algebra_identity_holds']}; "
      f"s0==truth: {x4['s0_matches_truth']}; e_hat nz: {x4['e_hat_nonzero']}",
      flush=True)

# ---------------------------------------------------------------- X5
# Blocking independence: accumulate the first 100k of the sim1m dump with two
# different block sizes and compare the Phi matrices element-wise.
m_range = np.arange(D)
raw = np.memmap("tmp/sim1m/dump.bin", dtype=np.uint8, mode="r", shape=(100_000, 4609))
LUT = np.zeros((15, 2))
zg = np.arange(-7, 8)
f0v = lambda x: np.where(np.abs(x) <= 7, (8 - np.abs(x)) / 64.0, 0.0)
with np.errstate(divide="ignore", invalid="ignore"):
    el = np.log(f0v(zg - 1)) - np.log(f0v(zg + 1))
LUT[:, 1] = np.clip(np.where(np.isfinite(el), el, np.sign(el) * 40.0), -40.0, 40.0)

def accum(block):
    Ph = np.zeros((3, 3, D), np.int64)
    rng = np.random.default_rng(777)
    for st in range(0, 100_000, block):
        b = min(block, 100_000 - st)
        cc = raw[st:st+b, 0:512].copy().view("<i2").reshape(b, D).astype(np.int64)
        eps_o = (1 - 2 * raw[st:st+b, 512].astype(np.int64))
        h = raw[st:st+b, 3073:4609].copy().view("<i2").reshape(b, 3, D)
        idx = np.nonzero(cc)[1].reshape(b, TAU)
        psi_ext = np.concatenate([h, -h], axis=2)
        psiT = np.ascontiguousarray(psi_ext.transpose(2, 0, 1))
        rows_rep = np.repeat(np.arange(b), TAU)
        idx_flat = idx.reshape(-1)
        g = psiT[idx_flat[:, None] + m_range[None, :], rows_rep[:, None], :]
        g = np.ascontiguousarray(g.reshape(b, TAU, D, 3).transpose(0, 1, 3, 2))
        eps_h = np.where(LUT[raw[st:st+b, 1025:1537].copy().view("<i2")
                             .reshape(b, D).astype(np.int64) + 7, cc].sum(axis=1) > 0,
                         1, -1).astype(np.int64)
        ep = eps_o[rng.permutation(b)]
        Ph[0] += (g * eps_h[:, None, None, None]).sum(axis=(0, 1))
        Ph[1] += (g * eps_o[:, None, None, None]).sum(axis=(0, 1))
        Ph[2] += (g * ep[:, None, None, None]).sum(axis=(0, 1))
    return Ph

Pa = accum(25_000)
Pb = accum(50_000)
x5 = {"block_independence_deterministic_arms":
      bool(np.array_equal(Pa[:2], Pb[:2])),
      "perm_arm_differs_expected": not bool(np.array_equal(Pa[2], Pb[2]))}
OUT["checks"]["X5_blocking"] = x5
print(f"[X5] block-size independence (deterministic arms): "
      f"{x5['block_independence_deterministic_arms']} "
      f"(perm differs as expected: {x5['perm_arm_differs_expected']})", flush=True)

# ---------------------------------------------------------------- X6
def solve_Rhat(Phi, N):
    M = N * TAU
    N_F = N * Fop
    out = np.zeros((3, D))
    ss = []
    for i in range(3):
        phi = Phi[0, i].copy()
        if i == 0:
            phi = phi + (0.924 * 0.5 * M * (1.0 - 2.0 * m_axis / D)) / 682.0
        rhs = -G2 * phi
        best = None
        for s in np.arange(0.88, 1.001, 0.002):
            Rh = np.linalg.solve(s * M * np.eye(D) + N_F, rhs)
            o = float(np.cos(2 * np.pi * Rh).mean())
            if best is None or o > best[0]:
                best = (o, float(s), Rh)
        out[i] = best[2]
        ss.append(round(best[1], 3))
    return out, ss

Rh_r, ss_r = solve_Rhat(Pr, 1_000_000)
Rh_s, ss_s = solve_Rhat(Ps3 := np.load("tmp/attack2/phi_hint_sim3m.npy")
                        .astype(np.float64), 3_000_000)
agree = int((np.clip(np.rint(Rh_r), -9, 8) == np.clip(np.rint(Rh_s), -9, 8)).sum())
corrs = [float(np.corrcoef(Rh_r[i], Rh_s[i])[0, 1]) for i in range(3)]
x6 = {"real_scales": ss_r, "sim3m_scales": ss_s,
      "rounded_agreement": f"{agree}/768", "per_row_corr": corrs}
OUT["checks"]["X6_cross_dataset"] = x6
print(f"[X6] rounded agreement {agree}/768, per-row corr {corrs}", flush=True)

json.dump(OUT, open("tmp/attack2/cross_check.json", "w"), indent=1)
print("[done] wrote tmp/attack2/cross_check.json", flush=True)
