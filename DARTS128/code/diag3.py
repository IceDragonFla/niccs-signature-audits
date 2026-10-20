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
off = 3*N + 5
a0, a1, ae = raw[3*N], raw[3*N+1], raw[3*N+2]
xz = np.concatenate([raw[off:off+N]/a0, raw[off+N:off+2*N]/a1])
u = xz - s_true
print("rms(u)=rms(xz_d - s) =", np.sqrt((u**2).mean()))

R = A @ A.T
P_u = A.T @ np.linalg.solve(R, A @ u)
K_u = u - P_u
print("rms(P u) =", np.sqrt((P_u**2).mean()), " rms(K u) =", np.sqrt((K_u**2).mean()))

e_obs = np.clip(np.rint(raw[off+2*N:off+3*N]/ae), -1, 1).astype(np.int64)
t = np.zeros(N, dtype=np.int64); t[0] = HALF_Q
t = center_q(t - e_obs)
w = np.linalg.solve(R, t.astype(np.float64))
P_s = A.T @ w
print("rms(P_s - P s_true) =", np.sqrt(((P_s - (s_true - K_u*(0)+ (s_true - K_u*0) - (s_true - (s_true - P_u*0 - K_u*0))))**2).mean()) if False else
      np.sqrt(((P_s - (s_true - (u - P_u)))**2).mean()))
# NOTE: P s_true = s_true - K s_true; K s_true = ?
K_s = s_true - A.T @ np.linalg.solve(R, A @ s_true)
print("rms(K s_true) =", np.sqrt((K_s**2).mean()), " rms(P s_true) =", np.sqrt(((s_true - K_s)**2).mean()))
print("rms(P_s - P s_true) =", np.sqrt(((P_s - (s_true - K_s))**2).mean()))

s_hat = P_s + K_u
err = s_hat - s_true
print("rms(s_hat - s) =", np.sqrt((err**2).mean()), " wrong =", int(np.sum(np.rint(s_hat) != s_true)))
print("decomp: rms(P err) =", np.sqrt(((P_u)**2).mean()), "(should equal P-part of err negated)")
