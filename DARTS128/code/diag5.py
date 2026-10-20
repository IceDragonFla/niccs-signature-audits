import numpy as np
Q = 130817; N = 512; HALF_Q = (Q + 1) // 2
def center_q(v):
    v = np.mod(v, Q); v[v > Q // 2] -= Q
    return v
cols = np.fromfile('cols.bin', dtype=np.int64).reshape(2 * N, N)
A = cols.T.astype(np.float64)
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.int64)
s_true = truth[:2*N].astype(np.float64)
R = A @ A.T
off = 3*N + 5
a0, a1, ae = raw[3*N], raw[3*N+1], raw[3*N+2]
xz0 = raw[off:off+N]/a0; xz1 = raw[off+N:off+2*N]/a1; xe = raw[off+2*N:off+3*N]/ae
xz = np.concatenate([xz0, xz1])
u = xz - s_true
print("rms(u) =", np.sqrt((u**2).mean()))
P_u = A.T @ np.linalg.solve(R, A @ u)
K_u = u - P_u
print("rms(P u) =", np.sqrt((P_u**2).mean()), " rms(K u) =", np.sqrt((K_u**2).mean()))
e_obs = np.clip(np.rint(xe), -1, 1).astype(np.int64)
t = np.zeros(N, dtype=np.int64); t[0] = HALF_Q
t = center_q(t - e_obs)
w = np.linalg.solve(R, t.astype(np.float64))
P_s = A.T @ w
s_hat = P_s + K_u
err = s_hat - s_true
print("rms(s_hat - s) =", np.sqrt((err**2).mean()), " wrong =", int(np.sum(np.rint(s_hat) != s_true)))
# per-coordinate: is the error concentrated where |u| is large?
big = np.abs(u) > 0.5
print("coords with |u|>0.5:", int(big.sum()), " of which wrong:", int(np.sum(np.rint(s_hat)[big] != s_true[big])))
print("err on |u|<=0.5 coords: rms =", np.sqrt((err[~big]**2).mean()))
