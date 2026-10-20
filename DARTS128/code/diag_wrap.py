import numpy as np
Q = 130817; N = 512; HALF_Q = (Q + 1) // 2
def center_q(v):
    v = np.mod(v, Q); v[v > Q // 2] -= Q
    return v
cols = np.fromfile('cols.bin', dtype=np.int64).reshape(2 * N, N).astype(np.float64)
A = cols.T                                    # (512, 1024)
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.int64)
s_true = truth[:2*N].astype(np.float64)
R = A @ A.T
off = 3*N + 5
ae = raw[3*N+2]
xe = raw[off+2*N:off+3*N]/ae
e_obs = np.clip(np.rint(xe), -1, 1).astype(np.int64)
t_red = np.zeros(N, dtype=np.int64); t_red[0] = HALF_Q
t_red = center_q(t_red - e_obs)
# true unreduced A s
As = A @ s_true                                # exact integers in float64
# P s_true via unreduced
w_true = np.linalg.solve(R, As)
P_s_true = A.T @ w_true
# P_s via reduced t (e_obs)
w_red = np.linalg.solve(R, t_red.astype(np.float64))
P_s = A.T @ w_red
err = P_s - P_s_true
print("rms(P_s - P s_true) =", float(np.sqrt((err**2).mean())), "  max =", float(np.abs(err).max()))
# the two-stage with the CLEAN row part:
Kxz = xz_kernel = None
off2 = off
a0, a1 = raw[3*N], raw[3*N+1]
xz = np.concatenate([raw[off2:off2+N]/a0, raw[off2+N:off2+2*N]/a1])
xz_d = xz
Kxz = xz_d - A.T @ np.linalg.solve(R, A @ xz_d)
s_hat = P_s_true + Kxz          # ideal row part + kernel from z1
err2 = s_hat - s_true
print("two-stage (ideal row): rms =", float(np.sqrt((err2**2).mean())),
      " wrong =", int(np.sum(np.rint(s_hat) != s_true)))
s_hat2 = P_s + Kxz              # real row part (e_obs + reduced t)
err3 = s_hat2 - s_true
print("two-stage (real  row): rms =", float(np.sqrt((err3**2).mean())),
      " wrong =", int(np.sum(np.rint(s_hat2) != s_true)))
