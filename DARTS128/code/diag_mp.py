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
s_init = np.clip(np.rint(np.concatenate([raw[off:off+N]/a0, raw[off+N:off+2*N]/a1])), -1, 1).astype(np.int64)
delta = s_init - truth[:2*N]
print("true error count:", int(np.sum(delta != 0)))
xe = raw[off+2*N:off+3*N]/ae
e_obs = np.clip(np.rint(xe), -1, 1).astype(np.int64)
t = np.zeros(N, dtype=np.int64); t[0] = HALF_Q
t = center_q(t - e_obs)
A = cols.T  # (512, 1024)
r = center_q(t - A @ s_init)
print("r: nnz =", int(np.sum(r != 0)), " max|r| =", int(np.abs(r).max()), " rms =", float(np.sqrt((r.astype(float)**2).mean())))
rho = (cols @ r.astype(np.float64)) / (cols**2).sum(axis=1)
# correlation of rho with -delta
d = delta.astype(np.float64)
c = np.corrcoef(rho, -d)[0,1]
print("corr(rho, -delta) =", round(c, 4))
err_idx = np.flatnonzero(delta != 0)
print("rho at true errors:", np.round(rho[err_idx][:20], 3))
print("max |rho| at NON-error coords:", float(np.abs(rho[delta == 0]).max()))
top = np.argsort(-np.abs(rho))[:10]
print("top-10 |rho|:", [(int(j), round(float(rho[j]),3), "err" if delta[j] else "clean") for j in top])
