import numpy as np
Q = 130817; N = 512; HALF_Q = (Q + 1) // 2
def center_q(v):
    v = np.mod(v, Q); v[v > Q // 2] -= Q
    return v
cols = np.fromfile('cols.bin', dtype=np.int64).reshape(2 * N, N).astype(np.float64)
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.int64)
off = 3*N + 5
a0, a1, ae = raw[3*N], raw[3*N+1], raw[3*N+2]
s = np.clip(np.rint(np.concatenate([raw[off:off+N]/a0, raw[off+N:off+2*N]/a1])), -1, 1).astype(np.int64)
delta = s - truth[:2*N]
print("wrong coords:", int(np.sum(delta != 0)))
e_obs = np.clip(np.rint(raw[off+2*N:off+3*N]/ae), -1, 1).astype(np.int64)
t = np.zeros(N, dtype=np.int64); t[0] = HALF_Q
t = center_q(t - e_obs)
r = center_q(t - cols.T @ s.astype(np.float64))
cb = int(np.sum(np.abs(r) > 1))
print("count_bad =", cb, "  r nnz =", int(np.sum(r != 0)))
