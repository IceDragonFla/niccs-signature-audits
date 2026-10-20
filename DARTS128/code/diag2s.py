import numpy as np
Q = 130817; N = 512; HALF_Q = (Q + 1) // 2
def center_q(v):
    v = np.mod(v, Q); v[v > Q // 2] -= Q
    return v
cols = np.fromfile('cols.bin', dtype=np.int64).reshape(2 * N, N)
A = cols.T.astype(np.float64)
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.int64)
s_true = truth[:2*N]

# check 1: identity with TRUE e: A@s_true == qhat*beta - e_true
t_true = np.zeros(N, dtype=np.int64); t_true[0] = HALF_Q
t_true = center_q(t_true - truth[2*N:3*N])
r0 = center_q(t_true - A @ s_true)
print("check1 identity residual (true s, true e): max =", np.abs(r0).max())

# check 2: conditioning
R = A @ A.T
ev = np.linalg.eigvalsh(R)
print("check2 cond(R) =", ev.max()/ev.min())

# check 3: min-norm solution with TRUE e
w = np.linalg.solve(R, t_true.astype(np.float64))
sid = A.T @ w
err_sid = sid - s_true
print("check3 rms(sid - s_true) =", np.sqrt((err_sid**2).mean()),
      " (expect ~ sqrt(0.5)*rms(s) =", 0.707*np.sqrt(0.3), ")")

# check 4: kernel projector consistency: K @ sid == 0
Ksid = sid - A.T @ np.linalg.solve(R, A @ sid)
print("check4 ||K sid|| =", np.sqrt((Ksid**2).mean()))

# check 5: two-stage with TRUE x_z1 = s_true (sanity: should return s_true)
d0 = s_true.astype(np.float64) - sid
w2 = np.linalg.solve(R, A @ d0)
xcomb = sid + (d0 - A.T @ w2)
print("check5 rms(xcomb(s_true input) - s_true) =", np.sqrt(((xcomb - s_true)**2).mean()))

# check 6: two-stage with the real xz
off = 3*N + 5
xz = np.concatenate([raw[off:off+N]/raw[3*N], raw[off+N:off+2*N]/raw[3*N+1]])
d = xz - sid
w3 = np.linalg.solve(R, A @ d)
xcomb2 = sid + (d - A.T @ w3)
print("check6 rms(xcomb(real xz) - s_true) =", np.sqrt(((xcomb2 - s_true)**2).mean()),
      " wrong =", int(np.sum(np.rint(xcomb2) != s_true)))
