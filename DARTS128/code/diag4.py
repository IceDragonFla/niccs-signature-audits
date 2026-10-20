import numpy as np
Q = 130817; N = 512; HALF_Q = (Q + 1) // 2
def center_q(v):
    v = np.mod(v, Q); v[v > Q // 2] -= Q
    return v
cols = np.fromfile('cols.bin', dtype=np.int64).reshape(2 * N, N)
A = cols.T.astype(np.float64)
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.int64)
R = A @ A.T
off = 3*N + 5
ae = raw[3*N+2]
xe = raw[off+2*N:off+3*N]
e_obs = np.clip(np.rint(xe/ae), -1, 1).astype(np.int64)
eta = e_obs - truth[2*N:3*N]
print("eta: nnz =", int(np.sum(eta != 0)), " max|.| =", int(np.abs(eta).max()), " norm =", float(np.sqrt((eta**2).sum())))
# how big is xe/ae actually?
print("xe/ae: min=%.2f max=%.2f  rms=%.3f" % ((xe/ae).min(), (xe/ae).max(), np.sqrt(((xe/ae)**2).mean())))
w = np.linalg.solve(R, (-eta).astype(np.float64))
Ap_eta = A.T @ w
print("rms(A+ eta) =", np.sqrt((Ap_eta**2).mean()))
# singular values of A
sv = np.linalg.svd(A, compute_uv=False)
print("sigma_min(A) = %.4g  sigma_max(A) = %.4g  cond = %.4g" % (sv[-1], sv[0], sv[0]/sv[-1]))
