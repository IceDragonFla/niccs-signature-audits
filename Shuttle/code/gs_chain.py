import numpy as np, time, math, sys, os, pickle
from flint import fmpz_mat, fmpq_mat, fmpz_poly
N = 256

def rmul(a, b):
    c = np.convolve(a.astype(np.int64), b.astype(np.int64))
    out = c[:N].copy()
    out[:len(c) - N] -= c[N:]
    return out

def rconj(v):
    o = np.zeros(N, dtype=np.int64); o[0] = v[0]; o[1:] = -v[N:0:-1]; return o

def rconj_obj(vv):
    vv = np.asarray(vv, dtype=object)
    o = np.zeros(N, dtype=object); o[0] = vv[0]; o[1:] = -vv[N:0:-1]
    return o

def obj_rmul(a, b):
    c = np.convolve(a.astype(object), b.astype(object))
    out = c[:N].copy(); out[:len(c) - N] -= c[N:]
    return out

def to_poly(v):
    # flint fmpz_poly takes coeffs low-to-high as list
    return fmpz_poly([int(x) for x in v])

PHI_POLY = fmpz_poly([1] + [0] * (N - 1) + [1])

def res_phi(v):
    return abs(int(to_poly(v).resultant(PHI_POLY)))

def det_rows(rows):
    A = fmpz_mat(len(rows), N)
    for i, r in enumerate(rows):
        for j in range(N): A[i, j] = int(r[j])
    return abs(int(A.det()))

_J = np.arange(1, 512, 2)
_K = np.arange(N)
_W = np.exp(2j * np.pi * np.outer(_J, _K) / 512.0)

def embed(v):
    return _W @ v.astype(float)

def embed_many(rows):
    return np.array([_W @ r.astype(float) for r in rows])

_mulmat_cache = {}
def solve_div(g, cs):
    key = np.asarray(g, dtype=object).tobytes()
    if key not in _mulmat_cache:
        go = np.asarray(g, dtype=object)
        big = abs(go).max() >= 2**31
        Mg = fmpz_mat(N, N)
        for i in range(N):
            e = np.zeros(N, dtype=np.int64); e[i] = 1
            col = obj_rmul(go, e) if big else rmul(np.asarray(g, dtype=np.int64), e)
            for k in range(N): Mg[k, i] = int(col[k])
        _mulmat_cache[key] = fmpq_mat(Mg)
    Mq = _mulmat_cache[key]
    rhs = fmpq_mat(N, len(cs))
    for i, c in enumerate(cs):
        for k in range(N): rhs[k, i] = int(c[k])
    U = Mq.solve(rhs)
    num, den = U.numer_denom()
    out = []
    for i in range(len(cs)):
        row = np.array([int(num[k, i]) for k in range(N)], dtype=object)
        out.append((row, int(den)))
    return out

from fpylll import IntegerMatrix, LLL as FLLL, GSO, FPLLL
FPLLL.set_precision(256)

def twisted_lll_256(rows, gam):
    """rows: EXACTLY N rows spanning rank-N lattice. Returns reduced rows."""
    m = len(rows)
    sg = np.abs(embed(gam))
    E = embed_many(rows)
    E = E / np.sqrt(sg)[None, :]
    mx = np.abs(E).max()
    sc = (2.0 ** 30) / max(mx, 1e-30)
    M = np.empty((m, 2 * N), dtype=np.float64)
    M[:, :N] = E.real * sc
    M[:, N:] = E.imag * sc
    Mi = np.round(M).astype(np.int64)
    aug = np.zeros((m, 2 * N + m), dtype=np.int64)
    aug[:, :2 * N] = Mi
    for i in range(m): aug[i, 2 * N + i] = 1
    A = IntegerMatrix.from_matrix(aug.tolist())
    t0 = time.time()
    Mg = GSO.Mat(A, float_type="mpfr")
    red = FLLL.Reduction(Mg, delta=0.99)
    red()
    t1 = time.time()
    rowsT = [[int(A[i, j]) for j in range(2 * N + m)] for i in range(m)]
    out = []
    for r in rowsT:
        nrm = sum(x * x for x in r[:2 * N])
        if nrm > 10**10:
            out.append((nrm, r[2 * N:]))
    out.sort(key=lambda t: t[0])
    T = np.array([t[1] for t in out], dtype=object)
    R = np.array([r.astype(object) for r in rows])
    prod = T @ R
    res = []
    for i in range(len(T)):
        r = prod[i]
        res.append(r.astype(np.int64) if abs(r).max() < 2**62 else r)
    print('    lll %d rows: %.0fs' % (m, t1 - t0), flush=True)
    return res

def spanning_products(base, gj, tag, rng):
    """512 pair-products spanning the FULL (v_j^2) (coprime pair, generic);
    twisted-LLL handles the 512 rows directly (mpfr@256: ~30-300s)."""
    g0 = base[0]
    g1 = base[1]
    prods = [rmul(g0, b) for b in base] + [rmul(g1, b) for b in base]
    return prods

# ---------- ground truth ----------
s = np.fromfile('s_true_1m.bin', dtype=np.int32).reshape(3, N).astype(np.int64)
v = s[0].copy()
gam = rmul(v, rconj(v))
IB = np.load('gs_IB.npy')
rng = np.random.default_rng(20261019)

def true_fpow(j):
    fp = v.copy()
    for _ in range(j): fp = rmul(fp, fp)
    return fp

def main():
    R_LEV = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    statef = 'gs_chain_state.pkl'
    if os.path.exists(statef):
        st = pickle.load(open(statef, 'rb'))
        lev, NB, B, F, gj, rho_prev, Rs = st['lev'], st['NB'], st['B'], st['F'], st['gj'], st['rho_prev'], st['Rs']
        print('resumed at level %d' % lev, flush=True)
    else:
        lev, NB, F, gj, rho_prev, Rs = 0, list(np.load('gs_NB0.npy')), list(IB), gam, None, []
        B = NB
    t00 = time.time()
    for j in range(lev, R_LEV):
        t0 = time.time()
        base = NB if j == 0 else B
        gjo = np.asarray(gj, dtype=object)
        gamj2 = obj_rmul(gjo, rconj_obj(gjo))
        def rconj_obj_pre(x):
            o = np.zeros(N, dtype=object); o[0] = x[0]; o[1:] = -x[N:0:-1]; return o
        prods = spanning_products(base, gj, 'lev%d' % j, rng)
        red = twisted_lll_256(prods, gamj2)
        R_j = red[0] if isinstance(red[0], np.ndarray) else red[0]
        R_j = np.asarray(R_j)
        if R_j.dtype == object and abs(R_j).max() < 2**62:
            R_j = R_j.astype(np.int64)
        nR = np.linalg.norm(R_j.astype(float))
        Ro = np.asarray(R_j, dtype=object)
        rho_j = obj_rmul(Ro, rconj_obj(Ro))
        gj2v = obj_rmul(np.asarray(gj, dtype=object), np.asarray(gj, dtype=object))
        rp2 = None
        if j == 0:
            (a1, d1), = solve_div(gam, [rho_j])
        else:
            rp2 = obj_rmul(rho_prev, rho_prev)
            num = obj_rmul(rho_j, gj2v)
            (a1, d1), = solve_div(rp2, [num])
        assert d1 == 1, 'gamma non-integral lev %d' % j
        gnext = np.asarray(a1)
        ok = None
        if j <= 6:
            (vj1o, dv), = solve_div(true_fpow(j), [Ro])
            ok = (dv == 1)
            if ok:
                vj1 = vj1o.astype(np.int64)
                ok = (np.array_equal(obj_rmul(vj1, rconj_obj(vj1)), np.asarray(gnext, dtype=object))
                      and np.array_equal(obj_rmul(true_fpow(j), vj1), Ro))
        print('lev %d: ||R||=%.1f GT:%s (%.0fs)' % (j, nR, ok, time.time() - t0), flush=True)
        cF = [obj_rmul(Ro, rconj_obj(f)) for f in F[:256]]
        if j == 0:
            gens = solve_div(gam, cF)
        else:
            lifted = [obj_rmul(c, gj2v) for c in cF]
            gens = solve_div(rp2, lifted)
        Bg = []
        for arr, d in gens:
            assert d == 1, 'B gen non-int lev %d' % j
            if arr.dtype == object and abs(arr).max() < 2**62:
                arr = arr.astype(np.int64)
            Bg.append(arr)
        assert len(Bg) == N
        B = twisted_lll_256(Bg, gnext)
        B = [np.asarray(r) for r in B]
        B = [r.astype(np.int64) if r.dtype == object and abs(r).max() < 2**62 else r for r in B]
        assert len(B) == N, 'B rows %d' % len(B)
        cB = [obj_rmul(Ro, rconj_obj(b)) for b in B[:256]]
        gensF = solve_div(gnext, cB)
        F = []
        for arr, d in gensF:
            assert d == 1, 'F gen non-int lev %d' % j
            if arr.dtype == object and abs(arr).max() < 2**62:
                arr = arr.astype(np.int64)
            F.append(arr)
        print('   B cmax %d F cmax %d [tot %.0fs]' % (
            abs(B[0]).max(), abs(F[0]).max(), time.time() - t00), flush=True)
        gj = gnext
        rho_prev = rho_j
        Rs.append(R_j)
        pickle.dump({'lev': j + 1, 'NB': NB, 'B': B, 'F': F, 'gj': gj, 'rho_prev': rho_prev, 'Rs': Rs},
                    open(statef, 'wb'))
    print('chain of %d levels done %.0fs' % (R_LEV, time.time() - t00), flush=True)

main()
