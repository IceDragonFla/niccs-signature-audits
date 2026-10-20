import numpy as np, time, math, sys
from flint import fmpz_mat, fmpq_mat
N = 256
STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def rmul(a, b):
    c = np.convolve(a.astype(np.int64), b.astype(np.int64))
    out = c[:N].copy()
    out[:len(c) - N] -= c[N:]
    return out

def rconj(v):
    o = np.zeros(N, dtype=np.int64); o[0] = v[0]; o[1:] = -v[N:0:-1]; return o

def negsh(w, j):
    o = np.roll(w, j)
    if j: o[:j] = -o[:j]
    return o

def embed(v):
    j = np.arange(1, 512, 2)
    k = np.arange(N)
    W = np.exp(2j * np.pi * np.outer(j, k) / 512.0)
    return W @ v.astype(float)

# ---------- exact arithmetic helpers ----------
_mulmat_cache = {}
def mul_matrix(g):
    key = g.tobytes()
    if key in _mulmat_cache: return _mulmat_cache[key]
    Mg = fmpz_mat(N, N)
    for i in range(N):
        e = np.zeros(N, dtype=np.int64); e[i] = 1
        col = rmul(g, e)
        for k in range(N): Mg[k, i] = int(col[k])
    _mulmat_cache[key] = Mg
    return Mg

def solve_div(g, cs):
    """solve g*u = c_i (mod Phi) exactly over Q for a list of rhs vectors"""
    Mg = mul_matrix(g)
    Mq = fmpq_mat(Mg)
    rhs = fmpq_mat(len(cs), N)
    for i, c in enumerate(cs):
        for k in range(N): rhs[i, k] = int(c[k])
    U = Mq.solve(rhs.transpose())
    num, den = U.transpose().numer_denom()
    out = []
    for i in range(len(cs)):
        row = [int(num[i, k]) for k in range(N)]
        out.append((np.array(row, dtype=object), int(den)))
    return out

def hnf_rows(gens):
    A = fmpz_mat(len(gens), N)
    for i, g in enumerate(gens):
        for j in range(N): A[i, j] = int(g[j])
    H = A.hnf()
    rows = [[int(H[i, j]) for j in range(N)] for i in range(len(gens))]
    return [r for r in rows if any(r)]

# ---------- lattice part ----------
from fpylll import IntegerMatrix, LLL as FLLL

def twisted_lll(rows, gam):
    """reduce basis rows under twisted form t(xybar/gam). returns transform matrix T (len x len)."""
    m = len(rows)
    sg = np.abs(embed(gam))
    E = np.array([embed(r) for r in rows])          # m x 256 complex
    E = E / np.sqrt(sg)[None, :]
    scale = 2.0 ** 40
    M = np.zeros((m, 2 * N), dtype=object)
    M[:, :N] = np.round(E.real * scale).astype(object)
    M[:, N:] = np.round(E.imag * scale).astype(object)
    aug = np.zeros((m, 2 * N + m), dtype=object)
    aug[:, :2 * N] = M
    for i in range(m): aug[i, 2 * N + i] = 1
    A = IntegerMatrix.from_matrix([[int(x) for x in row] for row in aug])
    FLLL.reduction(A)
    rowsT = [[int(A[i, j]) for j in range(2 * N + m)] for i in range(m)]
    # collect nonzero transform rows: those with nonzero basis part, sorted by basis norm
    out = []
    for r in rowsT:
        tb = r[2 * N:]
        if any(r[:2 * N]):
            nrm = sum(x * x for x in r[:2 * N])
            out.append((nrm, tb))
    out.sort(key=lambda t: t[0])
    T = np.array([t[1] for t in out], dtype=object)
    return T

def apply_transform(T, rows):
    m = len(rows)
    new = []
    for i in range(len(T)):
        acc = np.zeros(N, dtype=object)
        for j in range(m):
            c = int(T[i, j])
            if c:
                acc = acc + c * rows[j].astype(object)
        new.append(acc)
    return [r.astype(np.int64) if abs(r).max() < 2**62 else r for r in new]

# ---------- ground truth ----------
s = np.fromfile('s_true_1m.bin', dtype=np.int32).reshape(3, N).astype(np.int64)
v = s[0].copy()
gam = rmul(v, rconj(v))
G1 = rmul(v, rconj(s[1])); G2 = rmul(v, rconj(s[2]))
IB = np.load('gs_IB.npy') if __import__('os').path.exists('gs_IB.npy') else None
if IB is None:
    IB = hnf_rows([negsh(G1, j) for j in range(N)] + [negsh(G2, j) for j in range(N)])
    IB = np.array(IB, dtype=np.int64)
print('J basis rows:', len(IB))

if STAGE == 1:
    t0 = time.time()
    T = twisted_lll(list(IB), gam)
    print('twisted LLL %.0fs, transform rows %d' % (time.time() - t0, len(T)))
    NB = apply_transform(T, list(IB))
    # verify with ground truth: row = v * a, a integral, norms
    res = solve_div(v, [r for r in NB[:6]])
    for i, (arr, den) in enumerate(res[:6]):
        nrm = np.sqrt(sum(float(x) * float(x) for x in arr))
        print('row %d: den=%d (integrality of a: %s) ||a||=%.1f' % (i, den, den == 1, nrm))
    np.save('gs_T0.npy', T.astype(np.int64))
    np.save('gs_NB0.npy', np.array([r.astype(np.int64) for r in NB]))
    print('STAGE1 done')

def twisted_lll_dyn(rows, gam):
    """twisted LLL with dynamic scale"""
    m = len(rows)
    sg = np.abs(embed(gam))
    E = np.array([embed(r) for r in rows])
    E = E / np.sqrt(sg)[None, :]
    mx = np.abs(E).max()
    sc = (2.0 ** 52) / max(mx, 1e-30)
    M = np.zeros((m, 2 * N), dtype=np.float64)
    M[:, :N] = E.real * sc
    M[:, N:] = E.imag * sc
    Mi = np.round(M).astype(np.int64)
    aug = np.zeros((m, 2 * N + m), dtype=np.int64)
    aug[:, :2 * N] = Mi
    for i in range(m): aug[i, 2 * N + i] = 1
    A = IntegerMatrix.from_matrix(aug.tolist())
    FLLL.reduction(A)
    rowsT = [[int(A[i, j]) for j in range(2 * N + m)] for i in range(m)]
    out = []
    for r in rowsT:
        if any(r[:2 * N]):
            out.append((sum(x * x for x in r[:2 * N]), r[2 * N:]))
    out.sort(key=lambda t: t[0])
    return [t[1] for t in out]

def mat_apply(Trows, rows):
    m = len(rows)
    new = []
    for tr in Trows:
        acc = np.zeros(N, dtype=object)
        for j in range(m):
            c = tr[j]
            if c: acc = acc + c * rows[j].astype(object)
        new.append(acc)
    return new

def obj_rmul(a, b):
    c = np.convolve(a.astype(object), b.astype(object))
    out = c[:N].copy(); out[:len(c) - N] -= c[N:]
    return out

if STAGE == 2:
    R_LEV = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    from sympy import isprime, nextprime
    p = nextprime(2**20)
    while p % 512 != 1: p = nextprime(p)
    print('aux p = %d  (p-1)/512=%d' % (p, (p-1)//512), flush=True)
    NB = list(np.load('gs_NB0.npy'))
    F = list(IB)
    gj = gam
    rho_prev = None
    Rs = []
    t00 = time.time()
    for j in range(R_LEV):
        t0 = time.time()
        if j == 0:
            base = NB
        else:
            base = B
        prods = [rmul(base[0], b) for b in base] + [rmul(base[1], b) for b in base[:64]]
        gamj2 = rmul(gj, rconj(gj))
        Tr = twisted_lll_dyn(prods, gamj2)
        red = mat_apply(Tr, prods)
        nz = [r for r in red if any(r)]
        assert len(nz) == N, 'lev %d: only %d indep rows' % (j, len(nz))
        R_j = nz[0].astype(np.int64)
        nR = np.linalg.norm(R_j.astype(float))
        rho_j = rmul(R_j, rconj(R_j))
        gj2v = rmul(gj, gj)
        if j == 0:
            (a1, d1), = solve_div(gam, [rho_j])
        else:
            rp2 = rmul(rho_prev, rho_prev)
            num = obj_rmul(rho_j, gj2v)
            (a1, d1), = solve_div(rp2, [num.astype(np.int64)])
        assert d1 == 1, 'gamma non-integral lev %d' % j
        gnext = a1.astype(np.int64)
        fpow = v.copy()
        for _ in range(j): fpow = rmul(fpow, fpow)
        (vj1o, dv), = solve_div(fpow, [R_j])
        ok = (dv == 1)
        if ok:
            vj1 = vj1o.astype(np.int64)
            ok = np.array_equal(rmul(vj1, rconj(vj1)), gnext)
        print('lev %d: ||R||=%.1f gamma-next ok, GT: R_j==v^2^j*v_{j+1}: %s (%.0fs)' % (
            j, nR, ok, time.time()-t0), flush=True)
        cF = [rmul(R_j, rconj(f)) for f in F[:300]]
        if j == 0:
            gens = solve_div(gam, cF)
        else:
            lifted = [obj_rmul(c, gj2v) for c in cF]
            gens = solve_div(rp2, [g.astype(np.int64) for g in lifted])
        Bg = []
        for arr, d in gens:
            assert d == 1, 'B gen non-int lev %d' % j
            Bg.append(arr.astype(np.int64))
        TrB = twisted_lll_dyn(Bg, gnext)
        B = [r for r in mat_apply(TrB, Bg) if any(r)]
        B = [r.astype(np.int64) for r in B]
        assert len(B) == N
        cB = [rmul(R_j, rconj(b)) for b in B[:300]]
        gensF = solve_div(gnext, cB)
        F = []
        for arr, d in gensF:
            assert d == 1, 'F gen non-int lev %d' % j
            F.append(arr.astype(np.int64))
        print('  B:%d rows cmax %d; F:%d gens cmax %d [tot %.0fs]' % (
            len(B), abs(B[0]).max(), len(F), abs(F[0]).max(), time.time()-t00), flush=True)
        gj = gnext
        rho_prev = rho_j
        Rs.append(R_j)
    np.save('gs_Rs.npy', np.array(Rs))
    print('STAGE2 %d levels done %.0fs' % (R_LEV, time.time()-t00))
