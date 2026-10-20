#!/usr/bin/env python3
"""two_stage.py — numerically clean two-stage estimator (numpy).

Stage 1: min-norm solution of the identity equations A s = t:
    t = center_q(qhat*beta - e_obs),  sid = A^T (A A^T)^{-1} t
    (the ~2-4% wrong e_obs entries project through A^T with tiny gain)
Stage 2: add the kernel-space component of the z1 estimate:
    s = sid + K (x_z1 - sid),  K = I - A^T (A A^T)^{-1} A
Theoretically the error drops from rms(x_z1) to rms(x_z1)*sqrt(1/2).

usage: two_stage.py <pk_cols.bin> <raw_estimates.bin> <truth.bin> [out]
Scoring uses the truth file (homework context).
"""
import sys
import numpy as np

Q = 130817
N = 512
HALF_Q = (Q + 1) // 2

def center_q(v):
    v = np.mod(v, Q)
    v[v > Q // 2] -= Q
    return v

def main():
    colsp, rawp, truthp = sys.argv[1], sys.argv[2], sys.argv[3]
    outp = sys.argv[4] if len(sys.argv) > 4 else "recovered_ts.bin"

    cols = np.fromfile(colsp, dtype=np.int64).reshape(2 * N, N)   # rows = columns
    raw = np.fromfile(rawp, dtype=np.float64)
    off = 3 * N + 5
    a0, a1, ae = raw[3*N], raw[3*N+1], raw[3*N+2]
    xz0 = raw[off:off+N] / a0            # descaled z1-route estimates
    xz1 = raw[off+N:off+2*N] / a1
    xe = raw[off+2*N:off+3*N] / ae       # descaled hint-route e estimate

    # e_obs (ternary rounding of the hint estimate)
    e_obs = np.clip(np.rint(xe), -1, 1).astype(np.int64)

    # t = center_q(qhat*beta - e_obs)
    t = np.zeros(N, dtype=np.int64)
    t[0] = HALF_Q
    t = center_q(t - e_obs)

    # Stage 1: min-norm sid = A^T R^{-1} t,  R = A A^T (512x512), A = cols.T
    A_mat = cols.T.astype(np.float64)                 # (512, 1024)
    R = A_mat @ A_mat.T
    w = np.linalg.solve(R, t.astype(np.float64))
    sid = A_mat.T @ w

    # Stage 1: P s = A^T R^{-1} t  (row-space component, EXACT up to e_obs noise)
    P_s = A_mat.T @ w

    # Stage 2: kernel component from the DESCALED full z1 estimate.
    # CRITICAL: the key itself is (nearly) a null vector of A (A s = qhat*beta - e
    # ~ 0), so the kernel carries almost ALL of s, and it must come from the
    # de-ALIASED estimate: K*(xz/alpha), NOT K*(xz - sid) in the raw frame.
    xz_descaled = np.concatenate([xz0, xz1])
    Kxz = xz_descaled - A_mat.T @ np.linalg.solve(R, A_mat @ xz_descaled)
    s_hat = P_s + Kxz

    truth = np.fromfile(truthp, dtype=np.int8).astype(np.float64)
    s_true = np.concatenate([truth[:N], truth[N:2*N]])
    err = s_hat - s_true
    rms = float(np.sqrt((err ** 2).mean()))
    wrong = int(np.sum(np.rint(s_hat) != s_true))
    print(f"two-stage: rms={rms:.4f}  wrong={wrong}/1024")
    # also plain-xz baseline
    err0 = xz_descaled - s_true
    print(f"plain xz : rms={np.sqrt((err0**2).mean()):.4f}  "
          f"wrong={int(np.sum(np.rint(xz_descaled)!=s_true))}/1024")

    e_rec = center_q(t - (A_mat @ np.rint(s_hat))).astype(np.int64)
    e_true = truth[2*N:3*N].astype(np.int64)
    print(f"e via identity: wrong={int(np.sum(e_rec != e_true))}/512, "
          f"non-ternary={int(np.sum(np.abs(e_rec)>1))}")
    out = np.concatenate([np.rint(s_hat[:N]).astype(np.int8),
                          np.rint(s_hat[N:]).astype(np.int8),
                          e_rec.astype(np.int8)])
    out.tofile(outp)
    print(f"wrote {outp}")

if __name__ == "__main__":
    main()
