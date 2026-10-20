#!/usr/bin/env python3
"""wrap_decode.py — wrap-coordinate-descent decoder for the DARTS128 (1-2b) attack.

The MN-LWE key identity holds as A*s = t + q*k (integers), where the wrap
vector k = (A*s - t)/q is unknown (~ +-dozens per coordinate). Any real-number
solver of A s = t without k is garbage; but k can be found by coordinate
descent: the correct k minimizes the ternary distance of the combined estimate
    s_hat(k) = A^T R^{-1} (t + q k)  +  K xz_alpha,
where K xz_alpha (kernel component of the descaled z1 estimate) is FIXED.
Each single-coordinate wrap change moves s_hat by a precomputed rank-1 delta,
so a full descent sweep is cheap. No exhaustive search.

usage: wrap_decode.py <cols.bin> <raw_estimates.bin> <truth.bin> [sweeps]
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

def tern_dist(v):
    """per-coordinate squared distance to {-1,0,1}"""
    a = np.abs(v)
    d = np.minimum(a, np.abs(a - 1.0))
    return d * d

def main():
    colsp, rawp, truthp = sys.argv[1], sys.argv[2], sys.argv[3]
    sweeps = int(sys.argv[4]) if len(sys.argv) > 4 else 12

    cols = np.fromfile(colsp, dtype=np.int64).reshape(2 * N, N)
    A = cols.T.astype(np.float64)                     # (512, 1024)
    raw = np.fromfile(rawp, dtype=np.float64)
    off = 3 * N + 5
    a0, a1, ae = raw[3*N], raw[3*N+1], raw[3*N+2]
    xz = np.concatenate([raw[off:off+N] / a0, raw[off+N:off+2*N] / a1])
    xe = raw[off+2*N:off+3*N] / ae
    truth = np.fromfile(truthp, dtype=np.int8).astype(np.int64)
    s_true = truth[:2*N].astype(np.float64)

    e_obs = np.clip(np.rint(xe), -1, 1).astype(np.int64)
    t = np.zeros(N, dtype=np.int64)
    t[0] = HALF_Q
    t = center_q(t - e_obs)

    R = A @ A.T
    # G = A^T R^{-1}  (solve once per column via factorization)
    Lu = np.linalg.lstsq(R, np.eye(N), rcond=None)[0] if False else None
    Rinv = np.linalg.inv(R)
    G = A.T @ Rinv                                    # (1024, 512)
    Mq = G * Q                                        # delta per unit wrap

    # kernel component of the descaled z1 estimate (fixed through the descent)
    Kxz = xz - A.T @ (Rinv @ (A @ xz))

    # initial wraps from the plain candidate (round((A xz - t)/q))
    k = np.rint((A @ xz - t) / Q).astype(np.int64)
    P_s = G @ (t + Q * k).astype(np.float64)
    s_hat = P_s + Kxz
    obj = tern_dist(s_hat).sum()
    print(f"init: k norm={np.abs(k).sum()} obj={obj:.2f} "
          f"wrong={int(np.sum(np.rint(s_hat) != s_true))}/1024")

    for sweep in range(sweeps):
        changed = 0
        for i in range(N):
            base = P_s.copy()
            best_dk, best_obj, best_ps = 0, obj, None
            for dk in (-2, -1, 1, 2):
                P_try = base + dk * Mq[:, i]
                o = tern_dist(P_try + Kxz).sum()
                if o < best_obj - 1e-9:
                    best_dk, best_obj, best_ps = dk, o, P_try
            if best_dk != 0:
                k[i] += best_dk
                P_s = best_ps
                s_hat = P_s + Kxz
                obj = best_obj
                changed += 1
        wrong = int(np.sum(np.rint(s_hat) != s_true))
        print(f"sweep {sweep}: changed={changed} obj={obj:.2f} "
              f"|k|_1={np.abs(k).sum()} wrong={wrong}/1024")
        if changed == 0:
            break

    s_hat_r = np.rint(s_hat).astype(np.int64)
    e_rec = center_q(t + Q * k - A @ s_hat_r)
    et = truth[2*N:3*N]
    print(f"final: s wrong={int(np.sum(s_hat_r != s_true))}/1024  "
          f"e wrong={int(np.sum(e_rec != et))}/512  e non-ternary={int(np.sum(np.abs(e_rec)>1))}")
    out = np.concatenate([s_hat_r[:N], s_hat_r[N:], e_rec]).astype(np.int8)
    out.tofile("recovered_wr.bin")
    print("wrote recovered_wr.bin")

if __name__ == "__main__":
    main()
