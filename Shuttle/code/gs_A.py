import numpy as np, time, math, sys
from flint import fmpz_mat, fmpq_mat
N = 256

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
    # sigma_j(x) = sum_k v_k * exp(2 pi i j k / 512), j odd 1..511 (256 embeddings)
    j = np.arange(1, 512, 2)
    k = np.arange(N)
    W = np.exp(2j * np.pi * np.outer(j, k) / 512.0)
    return W @ v.astype(float)

def solve_div(g, c):
    # solve g*u = c in Q[x]/(x^256+1): u = c/g (exact rational)
    Mg = fmpz_mat(N, N)
    for i in range(N):
        e = np.zeros(N, dtype=np.int64); e[i] = 1
        col = rmul(g, e)
        for k in range(N):
            Mg[k, i] = int(col[k])
    rhs = fmpq_mat(N, 1)
    for k in range(N):
        rhs[k, 0] = int(c[k])
    Mq = fmpq_mat(Mg)
    u = Mq.solve(rhs)
    num, den = u.numer_denom()
    arr = np.array([int(num[i, 0]) for i in range(N)], dtype=object)
    return arr, int(den)

t0 = time.time()
s = np.fromfile('s_true_1m.bin', dtype=np.int32).reshape(3, N).astype(np.int64)
v = s[0].copy()
gam = rmul(v, rconj(v))
G1 = rmul(v, rconj(s[1])); G2 = rmul(v, rconj(s[2]))
print('v range [%d,%d] gamma range [%d,%d]' % (v.min(), v.max(), gam.min(), gam.max()), flush=True)

# gamma embedding profile (float)
sg = np.abs(embed(gam))
lv = np.log(np.abs(embed(v)))
print('|sigma(gamma)| range: %.3f..%.3f  log|sigma(v)| range: %.3f..%.3f' % (sg.min(), sg.max(), lv.min(), lv.max()), flush=True)

# --- J basis via HNF + LLL (as in gs_validate2) ---
from fpylll import IntegerMatrix, LLL as FLLL

def hnf_basis(gens):
    A = fmpz_mat(len(gens), N)
    for i, g in enumerate(gens):
        for j in range(N):
            A[i, j] = int(g[j])
    H = A.hnf()
    rows = [[int(H[i, j]) for j in range(N)] for i in range(len(gens))]
    nz = [r for r in rows if any(r)]
    M = IntegerMatrix(len(nz), N)
    for i, r in enumerate(nz):
        for j in range(N):
            M[i, j] = int(r[j])
    FLLL.reduction(M)
    rows2 = [[int(M[i, j]) for j in range(N)] for i in range(len(nz))]
    nz2 = [np.array(r, dtype=np.int64) for r in rows2 if any(r)]
    return np.vstack(nz2)

t0 = time.time()
IB = hnf_basis([negsh(G1, j) for j in range(N)] + [negsh(G2, j) for j in range(N)])
np.save('gs_IB.npy', IB)
print('J basis: %d rows, %.0fs, row norms:' % (len(IB), time.time() - t0),
      np.round(np.sort(np.linalg.norm(IB.astype(float), axis=1))[:5], 1), flush=True)

# --- candidate x1 = min-norm LLL row: r = x1/v integral by construction; check defect k=N(r) ---
norms = np.linalg.norm(IB.astype(float), axis=1)
b = IB[int(np.argmin(norms))].copy()
print('x1 = min-norm row #%d, ||x1||=%.1f coeff range [%d,%d]' % (
    int(np.argmin(norms)), np.linalg.norm(b.astype(float)), b.min(), b.max()), flush=True)
t0 = time.time()
arr, den = solve_div(v, b)   # r = b / v
print('r = x1/v: denominator = %d  (%.0fs)' % (den, time.time() - t0), flush=True)
if den == 1:
    r = arr.astype(np.int64)
    print('r INTEGRAL (as theory predicts): coeff range [%d,%d], ||r||=%.1f' % (r.min(), r.max(), np.linalg.norm(r.astype(float))))
    lr = np.log(np.abs(embed(r)) + 1e-30)
    print('log|sigma(r)| spread: [%.3f, %.3f]' % (lr.min(), lr.max()))
    # norm of r = k (defect), via |N(v)| from ground truth
    sh = np.abs(embed(v)); Nv = int(np.round(np.prod(sh)))
    shb = np.abs(embed(b)); Nb = int(np.round(np.prod(shb)))
    k = abs(Nb) // abs(Nv)
    print('|N(v)|=%d |N(x1)|=%d -> defect k=|N(r)|=%d  (exact: %s)' % (
        abs(Nv), abs(Nb), k, abs(Nb) == k * abs(Nv)))
    try:
        from sympy import factorint
        print('k factorization (top):', dict(list(factorint(k).items())[:8]))
    except Exception as e:
        print('factor failed', e)
    # near-R rows: G_j = r_j * conj(r) — integral always; check size on a few rows
    for idx in [1, 2, 3]:
        row = IB[idx]
        rj, rden = solve_div(v, row)
        if rden != 1:
            print('row %d: row/v den=%d UNEXPECTED' % (idx, rden)); continue
        rj = rj.astype(np.int64)
        g_row = rmul(rj, rconj(r))
        print('near-R row %d: r_j*rbar coeff range [%d,%d] norm %.1f' % (
            idx, g_row.min(), g_row.max(), np.linalg.norm(g_row.astype(float))), flush=True)
else:
    print('r NOT integral — contradicts theory; investigate')

# --- prime selection ---
def isprime(n):
    from sympy import isprime as ip
    return bool(ip(n))

def ord512(p):
    # multiplicative order of p mod 512
    if p % 2 == 0: return 0
    o, c = 1, p % 512
    while c != 1:
        c = c * p % 512; o += 1
    return o

q = (1 << 24) + 1
while not (isprime(q) and q % 512 == 1):
    q += 512
p = (1 << 24) + 3
good = False
while not good:
    if isprime(p) and ord512(p) == 128:
        g = math.gcd(q - 1, (pow(p, 128, q - 1) - 1) % (q - 1))
        if g == 512:
            good = True; break
    p += 4
print('q = %d (split, q%%512=%d)  p = %d (ord128 mod 512)  gcd(q-1,p^128-1)=512' % (q, q % 512, p))
print('Part A done in %.0fs total' % (time.time() - t0))
