"""linalg.py — public linear algebra shared by decoder and validator.

build_relation_matrix: turn the (K, L, d) kernel blocks A0 into the flat
(K*d, L*d) integer matrix Mx under the spec's ring convention, so that
    A0 * s  ==  Mx @ s          (mod q)
with the ONE convolution convention of spec.RingStructure:
    (a * s)[k] = sum_t a_t * s[(k - p_t) mod d] * flip(k - p_t).

invert_modq: Gauss-Jordan elimination over GF(q) (q prime).
"""
import numpy as np


def kernel_matrix(spec, a):
    """(d, d) signed-circulant of one kernel vector under the ring
    convention: (a*s)[k] = sum_m a[(k-m)%d] * flip * s[m]."""
    d = len(a)
    r = np.arange(d)
    idxm = (r[:, None] - r[None, :]) % d
    if spec.ring.kind == "cyclic":
        sgn = np.ones((d, d))
    else:
        sgn = np.where(r[None, :] > r[:, None], -1.0, 1.0)
    return (np.asarray(a)[idxm] * sgn).astype(np.int64)


def build_relation_matrix(spec, A0):
    """A0: (K, L, d) kernel blocks (or (n, n) to pass through)."""
    if A0.ndim == 2:
        return A0.astype(np.int64)
    d, K, L = spec.d, spec.K, spec.L
    Mx = np.zeros((K * d, L * d), np.int64)
    for i in range(K):
        for p in range(L):
            Mx[i * d:(i + 1) * d, p * d:(p + 1) * d] = kernel_matrix(spec,
                                                                    A0[i, p])
    return Mx


def invert_modq(M, q):
    """Inverse of M over GF(q), q prime. Returns None if singular."""
    M = np.asarray(M, dtype=np.int64) % q
    n = M.shape[0]
    assert M.shape[0] == M.shape[1]
    Aug = np.concatenate([M, np.eye(n, dtype=np.int64)], axis=1) % q
    for k in range(n):
        piv = k + int(np.argmax(Aug[k:, k] != 0))
        if Aug[piv, k] == 0:
            return None
        Aug[[k, piv]] = Aug[[piv, k]]
        iv = pow(int(Aug[k, k]), q - 2, q)
        Aug[k] = (Aug[k] * iv) % q
        col = Aug[:, k].copy()
        col[k] = 0
        Aug = (Aug - col[:, None] * Aug[k][None, :]) % q
    return Aug[:, n:]


def center_modq(x, q):
    """Centered representative in (-q/2, q/2]."""
    return ((np.asarray(x) % q) + q // 2) % q - q // 2
