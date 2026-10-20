import numpy as np, time, math, sys, pickle
from sympy import isprime, nextprime, Poly, symbols, ZZ, factor_list
N = 256
x = symbols('x')

def rmul(a, b):
    c = np.convolve(a.astype(np.int64), b.astype(np.int64))
    out = c[:N].copy()
    out[:len(c) - N] -= c[N:]
    return out

def rconj(v):
    o = np.zeros(N, dtype=np.int64); o[0] = v[0]; o[1:] = -v[N:0:-1]; return o

def obj_rmul(a, b):
    c = np.convolve(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
    out = c[:N].copy(); out[:len(c) - N] -= c[N:]
    return out

def rconj_obj(vv):
    vv = np.asarray(vv, dtype=object)
    o = np.zeros(N, dtype=object); o[0] = vv[0]; o[1:] = -vv[N:0:-1]
    return o

def mrmul(a, b, m):
    a = np.asarray(a, dtype=object) % m
    b = np.asarray(b, dtype=object) % m
    c = np.convolve(a, b) % m
    out = c[:N].copy(); out[:len(c) - N] -= c[N:]
    return out % m

def mpow(a, e, m):
    r = np.zeros(N, dtype=object); r[0] = 1
    a = np.asarray(a, dtype=object) % m
    e = int(e)
    while e:
        if e & 1: r = mrmul(r, a, m)
        a = mrmul(a, a, m); e >>= 1
    return r

def minv(a, m):
    """ring inverse mod prime m via sympy half_gcdex over Z[x]"""
    a = np.asarray(a, dtype=object)
    p1 = Poly(list(a), x, domain=ZZ)
    p2 = Poly([1] + [0] * (N - 1) + [1], x, domain=ZZ)
    try:
        s, h = p1.half_gcdex(p2)
    except Exception:
        return None
    if h.degree() != 0:
        return None
    res = int(h.LC())
    if math.gcd(res, m) != 1:
        return None
    sv = [int(c) for c in s.all_coeffs()][::-1]
    sv = sv + [0] * (N - len(sv))
    return np.array(sv, dtype=object) * pow(res, m - 2, m) % m

def centerlift_vec(vv, m):
    v = np.asarray(vv, dtype=object) % m
    return np.where(v > m // 2, v - m, v)

def main():
    st = pickle.load(open('gs_chain_state.pkl', 'rb'))
    Rs = st['Rs']
    nlev = len(Rs)
    print('levels: %d (R_0..R_%d)' % (nlev, nlev - 1), flush=True)
    s = np.fromfile('s_true_1m.bin', dtype=np.int32).reshape(3, N).astype(np.int64)
    v = s[0].copy()
    gam = rmul(v, rconj(v))

    cands = [(a, 512 * (2 ** a + 1) + 1) for a in range(1, nlev - 9)]
    cands = [(a, P) for a, P in cands if isprime(P)]
    print('prime cands (a,p):', cands, flush=True)
    p_main = p_second = None
    for a, P in cands:
        if a % 2 == 1 and p_main is None: p_main = (a, P)
    for a, P in cands:
        if a % 2 == 0 and p_second is None and math.gcd(p_main[1] - 1, P - 1) == 512:
            p_second = (a, P)
    a1, p1 = p_main; a2, p2 = p_second
    print('p1=%d (a1=%d) p2=%d (a2=%d)' % (p1, a1, p2, a2), flush=True)

    # Stage A: y = R_{a+9} * R_9 mod P  (= f^{P-1} * W), W = v_{a+10} * v_10
    t0 = time.time()
    y1 = mrmul(Rs[a1 + 9] % p1, Rs[9] % p1, p1)
    W1 = centerlift_vec(y1, p1)
    y2 = mrmul(Rs[a2 + 9] % p2, Rs[9] % p2, p2)
    W2 = centerlift_vec(y2, p2)
    print('W1 cmax %d (win %d), W2 cmax %d (win %d) [%.0fs]' % (
        abs(W1).max(), p1 // 2, abs(W2).max(), p2 // 2, time.time() - t0), flush=True)

    # ground-truth-free structural check: y1 * conj(y1) mod p1 should be
    # (gamma_{a1+10} * gamma_10) mod p1 — recompute gammas from Rs
    gams = [gam]
    for j in range(nlev):
        rho = obj_rmul(Rs[j], rconj_obj(Rs[j]))
        if j == 0:
            num = rho
            den = gam
        else:
            rho_prev = obj_rmul(Rs[j - 1], rconj_obj(Rs[j - 1]))
            den = obj_rmul(rho_prev, rho_prev)
            num = obj_rmul(rho, obj_rmul(gams[j], gams[j]))
        from flint import fmpz_mat, fmpq_mat
        den64 = np.asarray(den, dtype=object)
        A = fmpz_mat(N, N)
        for i in range(N):
            e = np.zeros(N, dtype=np.int64); e[i] = 1
            col = np.convolve(den64.astype(object), e.astype(object))
            o = col[:N].copy(); o[:len(col) - N] -= col[N:]
            for k in range(N): A[k, i] = int(o[k])
        rhs = fmpq_mat(N, 1)
        for k in range(N): rhs[k, 0] = int(num[k])
        U = fmpq_mat(A).solve(rhs)
        nm, dn = U.numer_denom()
        assert dn == 1, 'gamma_%d non-integral' % (j + 1)
        gams.append(np.array([int(nm[k, 0]) for k in range(N)], dtype=object))
    Wtrue = obj_rmul(gams[a1 + 10], gams[10])
    okW = np.array_equal(obj_rmul(W1, rconj_obj(W1)) % p1, Wtrue % p1)
    print('W1 structural check mod p1:', okW, flush=True)
    okW2 = np.array_equal(obj_rmul(W2, rconj_obj(W2)) % p2,
                          obj_rmul(gams[a2 + 10], gams[10]) % p2)
    print('W2 structural check mod p2:', okW2, flush=True)
    if not (okW and okW2):
        print('STRUCTURAL CHECK FAILED — chain or window broken'); return

    # Stage B: f^512 exact via Bezout + CRT
    from sympy import gcdex
    g0 = gcdex(p1 - 1, p2 - 1)
    alpha = int(g0[0]); beta = int(g0[1])
    assert alpha * (p1 - 1) + beta * (p2 - 1) == 512
    Qs = []
    q = nextprime(2 ** 63)
    while len(Qs) < 34:
        ok = True
        for P in (p1, p2):
            Wi = minv(W1 % q if P == p1 else W2 % q, q)
            if Wi is None: ok = False; break
        if ok: Qs.append(int(q))
        q = nextprime(q)
    print('CRT primes: %d' % len(Qs), flush=True)
    coeffs = []
    for Q in Qs:
        yq1 = mrmul(Rs[a1 + 9] % Q, Rs[9] % Q, Q)
        yq2 = mrmul(Rs[a2 + 9] % Q, Rs[9] % Q, Q)
        f1Q = mrmul(yq1, minv(W1 % Q, Q), Q)
        f2Q = mrmul(yq2, minv(W2 % Q, Q), Q)
        g = mpow(f1Q, abs(alpha), Q) if alpha >= 0 else mpow(minv(f1Q, Q), -alpha, Q)
        h = mpow(f2Q, abs(beta), Q) if beta >= 0 else mpow(minv(f2Q, Q), -beta, Q)
        coeffs.append((Q, mrmul(g, h, Q)))
    M = 1
    for Q, _ in coeffs: M *= Q
    fpow = np.zeros(N, dtype=object)
    for i in range(N):
        r = 0
        for Q, g in coeffs:
            Mi = M // Q
            r += int(g[i]) * Mi * pow(Mi % Q, Q - 2, Q)
        fpow[i] = centerlift_vec(np.array([r % M], dtype=object), M)[0]
    print('f^512 cmax ~ %.3g' % float(abs(fpow).max()), flush=True)
    gamK = gam.astype(object)
    for _ in range(9): gamK = obj_rmul(gamK, gamK)
    ok = np.array_equal(obj_rmul(fpow, rconj_obj(fpow)), gamK)
    print('VERIFY f^512*conj == gamma^512:', ok, flush=True)
    if not ok: return
    np.save('gs_fpow512.npy', fpow.astype(object))

    # Stage C: nonsplit root extraction
    def ord512(P):
        if P % 2 == 0: return 0
        o, c = 1, P % 512
        while c != 1:
            c = c * P % 512; o += 1
        return o
    pn = nextprime(2 ** 16)
    while ord512(pn) != 128: pn = nextprime(pn)
    print('nonsplit pn = %d' % pn, flush=True)
    fac = factor_list(Poly([1] + [0] * 255 + [1], x, modulus=pn))
    fs = [f for f, m in fac[1]]
    assert len(fs) == 2 and fs[0].degree() == 128, [f.degree() for f in fs]
    f1 = fs[0]; f2 = fs[1]
    f1c = [int(c) % pn for c in f1.all_coeffs()][::-1]
    f2c = [int(c) % pn for c in f2.all_coeffs()][::-1]

    def qred(c):
        c = [int(t) % pn for t in np.asarray(c, dtype=object)] + [0] * 256
        c = c[:384]
        for i in range(383, 127, -1):
            co = c[i]
            if co:
                for j in range(128):
                    c[i - 128 + j] = (c[i - 128 + j] - co * f1c[j]) % pn
                c[i] = 0
        return np.array(c[:128], dtype=object)

    def qmul(a, b):
        return qred(np.convolve(np.asarray(a, dtype=object) % pn, np.asarray(b, dtype=object) % pn))

    def qpow(a, e):
        r = np.zeros(128, dtype=object); r[0] = 1
        a = qred(a)
        while e:
            if e & 1: r = qmul(r, a)
            a = qmul(a, a); e >>= 1
        return r

    c1 = qred(fpow % pn)
    qm1 = pn ** 128 - 1
    assert qm1 % 512 == 0 and qm1 % 1024 != 0
    odd = qm1 // 512
    d = pow(512, -1, odd)
    r1 = qpow(c1, d)
    print('r1^512 == c1:', np.array_equal(qpow(r1, 512), c1), flush=True)
    # second component: c2 = f^512 mod f2, and its root r2 in F_pn[y]/f2
    def q2red(c):
        c = [int(t) % pn for t in np.asarray(c, dtype=object)] + [0] * 256
        c = c[:384]
        for i in range(383, 127, -1):
            co = c[i]
            if co:
                for j in range(128):
                    c[i - 128 + j] = (c[i - 128 + j] - co * f2c[j]) % pn
                c[i] = 0
        return np.array(c[:128], dtype=object)
    def q2mul(a, b):
        return q2red(np.convolve(np.asarray(a, dtype=object) % pn, np.asarray(b, dtype=object) % pn))
    def q2pow(a, e):
        r = np.zeros(128, dtype=object); r[0] = 1
        a = q2red(a)
        while e:
            if e & 1: r = q2mul(r, a)
            a = q2mul(a, a); e >>= 1
        return r
    c2 = q2red(fpow % pn)
    r2 = q2pow(c2, d)
    print('r2^512 == c2:', np.array_equal(q2pow(r2, 512), c2), flush=True)

    # CRT matrices: v = v1 + f1 * s, s = (v2 - v1) * inv(f1 mod f2) in F_pn[y]/f2
    # => v = P1 v1 + P2 v2 where P1 = (I - f1*L), P2 = f1*L, L = mult-by-inv(f1 mod f2) on f2-side
    inv_f1 = Poly(f1c[::-1], x, modulus=pn).invert(Poly(f2c[::-1], x, modulus=pn))
    invc = [int(c) % pn for c in inv_f1.all_coeffs()][::-1] + [0] * 128
    invc = invc[:128]
    # L matrix (128x128): columns = q2mul(invc, e_j)
    Lm = np.zeros((128, 128), dtype=np.int64)
    for j in range(128):
        e = np.zeros(128, dtype=object); e[j] = 1
        col = q2mul(invc, e)
        Lm[:, j] = col.astype(np.int64)
    # f1-multiplication matrix as poly (deg 255): f1 * s: convolve then NO reduction (deg ≤ 255 fits)
    F1M = np.zeros((256, 128), dtype=np.int64)
    for j in range(128):
        F1M[j:j + 129, j] += np.array(f1c, dtype=np.int64)
    # pad Lm result to 256
    def pad128(y): return np.concatenate([y, np.zeros(128, dtype=np.int64)])
    # v = v1(pad) + f1*( L*(v2 - v1) )
    #    = pad(v1) - F1M@Lm@pad(v1)^T-ish: build M1 (256x128) and M2 (256x128) over Z:
    # v = pad(v1) + F1M @ (Lm @ v2 - Lm @ v1) = (Pad128 - F1M Lm) v1 + (F1M Lm) v2
    P2M = (F1M.astype(np.int64) @ Lm) % pn
    P1M = (np.eye(256, dtype=np.int64)[:, :128] - P2M) % pn
    t_ = np.zeros(128, dtype=object); t_[1] = 1
    # t in f1-field is class of x; t2 in f2-field = class of y: candidate powers
    tpow1 = [np.zeros(128, dtype=np.int64); ]
    print('enumeration...', flush=True)
    # precompute t^j in field1 and y^j in field2 (y = class of x mod f2)
    def powlist(base, mul, cnt=512):
        acc = np.zeros(128, dtype=object); acc[0] = 1
        out = [acc.astype(np.int64)]
        for i in range(1, cnt):
            acc = mul(acc, out[0] * 0 + np.eye(1, dtype=object)[0, 0] * 0 + acc)  # placeholder
        return out
    # (rewritten below properly)
    def build_pows(mul):
        acc = np.zeros(128, dtype=object); acc[0] = 1
        res = [acc.astype(np.int64)]
        cur = base.copy()
        # cur = x
        xv = np.zeros(128, dtype=object); xv[1] = 1
        cur = xv
        res.append(cur.astype(np.int64))
        for i in range(2, 512):
            cur = mul(cur, xv)
            res.append(cur.astype(np.int64))
        return res
    T1 = build_pows(qmul)     # t^j in F_pn[x]/f1
    T2 = build_pows(q2mul)    # y^j in F_pn[y]/f2
    gamP = gam % pn
    fpP = fpow % pn
    hit = None
    t0 = time.time()
    for j1 in range(512):
        v1 = (T1[j1].astype(object) * r1) % pn
        # component-2 candidates
        for j2 in range(512):
            v2 = (T2[j2].astype(object) * r2) % pn
            vv = (P1M @ v1.astype(np.int64) + P2M @ v2.astype(np.int64)) % pn
            # cheap check: ||v||_inf small after centerlift
            w = vv.astype(np.int64)
            w = np.where(w > pn // 2, w - pn, w)
            if abs(w).max() <= 8:
                # exact check
                wl = w.astype(object)
                if np.array_equal(obj_rmul(wl, rconj_obj(wl)), obj_rmul(v.astype(object), rconj(v).astype(object))):
                    hit = wl
                    print('FOUND at j1=%d j2=%d: v cmax %d' % (j1, j2, abs(w).max()), flush=True)
                    break
        if hit is not None: break
    print('enum %.0fs, hit=%s' % (time.time() - t0, hit is not None), flush=True)
    if hit is not None:
        np.save('gs_recovered_s0.npy', hit.astype(np.int64))
        print('RECOVERED s0 (up to +-x^j): equal to truth x^j?', flush=True)

main()
