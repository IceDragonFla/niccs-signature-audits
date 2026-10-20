import numpy as np, time, math, sys, os, pickle
N = 256

def rmul(a, b):
    c = np.convolve(a.astype(np.int64), b.astype(np.int64))
    out = c[:N].copy()
    out[:len(c) - N] -= c[N:]
    return out

def rconj(v):
    o = np.zeros(N, dtype=np.int64); o[0] = v[0]; o[1:] = -v[N:0:-1]; return o

def obj_rmul(a, b):
    c = np.convolve(a.astype(object), b.astype(object))
    out = c[:N].copy(); out[:len(c) - N] -= c[N:]
    return out

def mrmul(a, b, mod):
    a = np.asarray(a, dtype=object) % mod
    b = np.asarray(b, dtype=object) % mod
    c = np.convolve(a, b) % mod
    out = c[:N].copy(); out[:len(c) - N] -= c[N:]
    return out % mod

def mpow(a, e, mod):
    r = np.zeros(N, dtype=object); r[0] = 1
    a = np.asarray(a, dtype=object) % mod
    e = int(e)
    while e:
        if e & 1: r = mrmul(r, a, mod)
        a = mrmul(a, a, mod); e >>= 1
    return r

s = np.fromfile('s_true_1m.bin', dtype=np.int32).reshape(3, N).astype(np.int64)
v = s[0].copy()
gam = rmul(v, rconj(v))

# ---------- load chain ----------
st = pickle.load(open('gs_chain_state.pkl', 'rb'))
Rs = st['Rs']
print('chain levels available: %d (R_0..R_%d)' % (len(Rs), len(Rs) - 1), flush=True)

# ---------- reconstruct gamma_j for verification ----------
gams = [None] * (len(Rs) + 1)
gams[0] = gam
for j in range(len(Rs)):
    rho_j = rmul(Rs[j], rconj(Rs[j]))
    if j == 0:
        c = obj_rmul(rho_j, np.ones(N, dtype=object))
        # gamma_1 = rho_0 / gamma
        from flint import fmpz_mat, fmpq_mat
        def solve1(g, cvec):
            Mg = fmpz_mat(N, N)
            for i in range(N):
                e = np.zeros(N, dtype=np.int64); e[i] = 1
                col = rmul(g, e)
                for k in range(N): Mg[k, i] = int(col[k])
            rhs = fmpq_mat(N, 1)
            for k in range(N): rhs[k, 0] = int(cvec[k])
            U = fmpq_mat(Mg).solve(rhs)
            num, den = U.numer_denom()
            return np.array([int(num[k, 0]) for k in range(N)], dtype=object), int(den)
        a1, d1 = solve1(gam, rho_j)
    else:
        rho_prev = rmul(Rs[j-1], rconj(Rs[j-1]))
        rp2 = rmul(rho_prev, rho_prev)
        num = obj_rmul(rho_j, rmul(gams[j], gams[j]))
        a1, d1 = solve1(rp2, num.astype(np.int64))
    assert d1 == 1, 'gamma_%d non-integral' % (j+1)
    gams[j+1] = a1.astype(np.int64)
print('gammas reconstructed', flush=True)

def inv_mod_prime(a, mod):
    """ring inverse of a mod (mod), mod prime: solve a*x=1 via flint nmod or lin algebra"""
    from flint import nmod_mat
    A = nmod_mat(N, N, [int(x) % mod for _ in [0] for x in [0]], mod) if False else None
    M = np.zeros((N, N), dtype=object)
    for i in range(N):
        e = np.zeros(N, dtype=np.int64); e[i] = 1
        col = mrmul(a, e, mod)
        for k in range(N): M[k, i] = int(col[k])
    # gaussian elimination mod `mod` (prime)
    M = np.concatenate([M, np.eye(N, dtype=object)], axis=1) % mod
    inv2 = pow(2, mod - 2, mod)
    for col in range(N):
        piv = None
        for r in range(col, N):
            if int(M[r, col]) % mod != 0: piv = r; break
        assert piv is not None, 'not invertible mod %d' % mod
        M[[col, piv]] = M[[piv, piv]] + (M[[col, col]] - M[[col, col]]) if False else None
        if piv != col:
            t = M[col].copy(); M[col] = M[piv]; M[piv] = t
        pval = int(M[col, col]) % mod
        pinv = pow(pval, mod - 2, mod)
        M[col] = M[col] * pinv % mod
        for r in range(N):
            if r != col and int(M[r, col]) % mod != 0:
                f = int(M[r, col]) % mod
                M[r] = (M[r] - f * M[col]) % mod
    return np.array([int(M[i, N + i]) % mod for i in range(N)], dtype=object)

def eval_chain(mod, k_target, Rs):
    """compute y = f^k * W mod `mod`, where W = prod v_l^e tracked symbolically.
    returns (y, Wspec). Uses: R_j = f^{2^j} v_{j+1};  v_{m+1} = v_m^2 * R_m / R_{m-1}^2."""
    bits = [i for i in range(len(Rs) - 1, -1, -1) if (k_target >> i) & 1]
    assert bits, 'k=0'
    top = bits[0]
    y = np.asarray(Rs[top], dtype=object) % mod      # f^{2^top} v_{top+1}
    W = [(top + 1, 1)]
    cur = top
    for b in bits[1:]:
        for _ in range(cur - b):
            # square: y = f^{2k} * prod v_l^{2e}; normalize each v_l^{2e} -> v_{l+1}^e
            y = mrmul(y, y, mod)
            Wn = []
            corr = np.zeros(N, dtype=object); corr[0] = 1
            for (l, e) in W:
                # v_l^{2e} = v_{l+1}^e * (R_l / R_{l-1}^2)^{-e}
                y = mrmul(y, mpow(mrmul(Rs[l] % mod, mpow(inv_cache(mod, Rs[l-1]), 2, mod), mod), e % (mod-1) if False else e, mod), mod)
                Wn.append((l + 1, e))
            W = Wn
            cur += 1
        # multiply by R_b : k += 2^b, W += v_{b+1}
        y = mrmul(y, Rs[b] % mod, mod)
        W.append((b + 1, 1))
    assert (1 << cur) | 0
    return y, W

_inv_cache = {}
def inv_cache(mod, elem):
    key = (mod, elem.tobytes() if isinstance(elem, np.ndarray) else None)
    if key in _inv_cache: return _inv_cache[key]
    r = inv_mod_prime(np.asarray(elem, dtype=object) % mod, mod)
    _inv_cache[key] = r
    return r

if __name__ == '__main__':
    mod0 = 71
    # quick ground-truth sanity of the y-chain for small exponents: y should equal f^k * W mod mod0
    for k in [3, 5, 9, 17, 2**6 + 2**3 + 1]:
        if k >= (1 << len(Rs)): continue
        y, W = eval_chain(mod0, k, Rs)
        fk = v.copy()
        for _ in range(20):
            if (k >> _) == 0: break
        # compute f^k mod
        fk = mpow(v % mod0, k, mod0)
        # compute W exactly: product of v_l^e
        Wv = np.zeros(N, dtype=object); Wv[0] = 1
        for (l, e) in W:
            tv = true_vj(l) if l <= 10 else None
            if tv is None: continue
            Wv = mrmul(Wv, mpow(tv % mod0, e, mod0), mod0)
        ok = np.array_equal(y % mod0, mrmul(fk, Wv, mod0) % mod0)
        print('k=%d: y == f^k*W mod %d: %s (Wspec=%s)' % (k, mod0, ok, W), flush=True)
