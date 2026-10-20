# hint_solve.py — operator-based R recovery from the saved hint-channel Phi.
#
# Exact linear model derived from the alignment combinatorics (iid-uniform
# challenge positions, negacyclic wrap signs; validated per-position against
# the measured E[eps*h] field at corr 0.99):
#   E[Phi[m]] = -(1/682) * [ M*R[m] + beta*(K R)[m] ]  (+ row-0 b-term offset)
#   K[m,v]    = d - 2*|m-v|          (tent kernel, LINEAR distance)
#   beta      = N*tau*(tau-1)/d^2
#   b-term    = -P0*M*(1-2m/d)/682 on row 0  => rhs correction -P0*M*(1-2m/d)
# The field is a KNOWN linear functional of R, so R is recovered by solving
#   (M*I + beta*K) R = -682*Phi + b0corr
# per K-row (256x256), with NO fitted scale and NO oracle calibration.
import json
import sys
import numpy as np

D, Q, TAU = 256, 26881, 30
G2 = 682
DATA = sys.argv[1] if len(sys.argv) > 1 else "real"
SUF = sys.argv[2] if len(sys.argv) > 2 else "1000000"
NSIG = int(SUF)

g16 = np.fromfile("tmp/bit128_1m/ground.bin", dtype="<i2").astype(np.int64)
s0_true = g16[:3 * D].reshape(3, D)
b1_true = g16[6 * D:9 * D].reshape(3, D)
A0 = g16[12 * D:].reshape(3, 3, D)
e_true = g16[3 * D:6 * D].reshape(3, D)
b0_true = g16[9 * D:12 * D].reshape(3, D)

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

Phi = np.load(f"tmp/attack2/phi_hint_{DATA}_{SUF}.npy").astype(np.float64)  # (3,3,256)
M = NSIG * TAU
beta = NSIG * TAU * (TAU - 1) / D ** 2

mv = np.abs(np.arange(D)[:, None] - np.arange(D)[None, :]).astype(np.float64)
K = D - 2.0 * mv                     # tent kernel
Amat = M * np.eye(D) + beta * K      # same for every row: factor once
m_axis = np.arange(D)
b0corr = 0.5 * M * (1.0 - 2.0 * m_axis / D)   # P0 = 0.5 by design (uniform bit)

out = {"data": DATA, "nsig": NSIG, "M": M, "beta": beta, "modes": {}}
Rhat_by_mode = {}
for mi, mode in enumerate(("attack", "oracle", "perm")):
    Rhat = np.zeros((3, D))
    field_pred = np.zeros((3, D))
    for i in range(3):
        phi = Phi[mi, i].copy()
        if i == 0:
            phi = phi + b0corr / 682.0       # remove the row-0 b-term offset
        # field prediction (scoring only): -(beta/682)*K R_true
        field_pred[i] = -(beta / 682.0) * (K @ R_true[i])
        rhs = -G2 * phi
        Rhat[i] = np.linalg.solve(Amat, rhs)
    if mode == "perm":
        res = {"Rhat_std": float(Rhat.std()), "note": "control"}
    else:
        Rint = np.clip(np.rint(Rhat), -9, 8).astype(np.int64)
        acc = float((Rint == R_true).mean())
        sig = -(M / 682.0) * R_true          # diagonal signal part (scoring only)
        res = {
            "exact_acc": acc,
            "per_row_acc": [float((Rint[i] == R_true[i]).mean()) for i in range(3)],
            "n_wrong": int((Rint != R_true).sum()),
            "resid_std": float((Rhat - R_true).std()),
            "field_corr": [float(np.corrcoef(Phi[mi, i] - sig[i], field_pred[i])[0, 1])
                           for i in range(3)],
            "field_slope": [float(np.polyfit(field_pred[i], Phi[mi, i] - sig[i], 1)[0])
                            for i in range(3)],
        }
        Rhat_by_mode[mode] = Rint
    out["modes"][mode] = res
    print(f"[{mode}] {res}", flush=True)

# ---- solve s0 from the attack-mode R (and oracle mode for the mechanism tier)
def negacyc_matrix(a):
    r = np.arange(D)
    idxm = (r[:, None] - r[None, :]) % D
    sign = np.where(r[None, :] > r[:, None], -1, 1)
    return a[idxm] * sign

def solve_modq(Mx, rhs):
    A = np.concatenate([Mx % Q, (rhs % Q)[:, None]], axis=1).astype(np.int64)
    n = A.shape[0]
    for k in range(n):
        piv = k + int(np.argmax(A[k:, k] != 0))
        if A[piv, k] == 0:
            raise RuntimeError("singular")
        if piv != k:
            A[[k, piv]] = A[[piv, k]]
        inv = pow(int(A[k, k]), Q - 2, Q)
        col = (A[k + 1:, k] * inv) % Q
        A[k + 1:, k:] = (A[k + 1:, k:] - col[:, None] * A[k, k:][None, :]) % Q
    x = np.zeros(n, np.int64)
    for k in range(n - 1, -1, -1):
        x[k] = ((A[k, n] - (A[k, k + 1:n] * x[k + 1:]).sum()) % Q) \
            * pow(int(A[k, k]), Q - 2, Q) % Q
    return x

Mx = np.zeros((3 * D, 3 * D), np.int64)
for i in range(3):
    for p in range(3):
        Mx[i * D:(i + 1) * D, p * D:(p + 1) * D] = negacyc_matrix(A0[i, p])

sol = {}
for mode in ("attack", "oracle"):
    if mode not in Rhat_by_mode:
        continue
    Rint = Rhat_by_mode[mode]
    rhs = (16 * (b1_true % Q).reshape(3 * D) + Rint.reshape(3 * D)) % Q
    x = solve_modq(Mx, rhs)
    s0_hat = center(x).reshape(3, D)
    n_match = int((s0_hat == s0_true).sum())
    tvh = np.zeros((3, D), np.int64)
    for i in range(3):
        acc2 = np.zeros(D, np.int64)
        for p in range(3):
            acc2 = (acc2 + negacyc(A0[i, p], s0_hat[p])) % Q
        tvh[i] = acc2
    R_impl = np.stack([center(tvh[i] - 16 * (b1_true[i] % Q)) for i in range(3)])
    sol[mode] = {
        "s0_coords_matching_truth": n_match,
        "s0_exact": n_match == 3 * D,
        "R_implied_equals_Rhat": bool(np.array_equal(R_impl, Rint)),
    }
    print(f"[solve:{mode}] s0 match {n_match}/768 exact={n_match == 3 * D} "
          f"R_impl==Rhat={sol[mode]['R_implied_equals_Rhat']}", flush=True)
    if n_match == 3 * D:
        ehat = np.where(R_impl >= 8, -1, np.where(R_impl <= -9, 1, 0)).astype(np.int64)
        b0hat = R_impl + ehat
        assert np.all((b0hat >= -8) & (b0hat < 8)) and np.abs(ehat).max() <= 1
        with open(f"tmp/attack2/key_hat_h_{mode}.bin", "wb") as f:
            for p in range(3):
                f.write(s0_hat[p].astype(np.int16).tobytes())
            for i in range(3):
                f.write(ehat[i].astype(np.int16).tobytes())
            for i in range(3):
                f.write(b0hat[i].astype(np.int16).tobytes())
        sol[mode]["key_written"] = True
        sol[mode]["e_hat_nonzero_coords"] = int((ehat != 0).sum())
        print(f"[solve:{mode}] equivalent key written "
              f"(e_hat nonzero at {int((ehat != 0).sum())} coords)", flush=True)

out["solve"] = sol
with open(f"tmp/attack2/hint_solve_{DATA}_{SUF}.json", "w") as f:
    json.dump(out, f, indent=1)
print("[done] wrote tmp/attack2/hint_solve_" + DATA + "_" + SUF + ".json", flush=True)
