import numpy as np, time, pickle, math
N = 256

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
    while e:
        if e & 1: r = mrmul(r, a, m)
        a = mrmul(a, a, m); e >>= 1
    return r

def minv(a, m):
    from sympy import Poly, symbols, ZZ
    x = symbols('x')
    p1 = Poly(list(np.asarray(a, dtype=object)), x, domain=ZZ)
    p2 = Poly([1] + [0] * (N - 1) + [1], x, domain=ZZ)
    s, h = p1.half_gcdex(p2)
    if h.degree() != 0:
        return None
    res = int(h.LC())
    if math.gcd(res, m) != 1:
        return None
    sv = [int(c) for c in s.all_coeffs()][::-1]
    sv += [0] * (N - len(sv))
    return np.array(sv, dtype=object) * pow(res, m - 2, m) % m

s3 = np.fromfile('s_true_1m.bin', dtype=np.int32).reshape(3, N).astype(np.int64)
v = s3[0].copy()
gam = rmul(v, rconj(v))

st = pickle.load(open('gs_chain_state.pkl', 'rb'))
Rs = st['Rs']
print('levels done:', len(Rs), flush=True)

# --- mini endgame test 1: product-of-R identity ---
# R_j = f^{2^j} v_{j+1}; v_{j+1} = R_j / f^{2^j} (GT while materializable)
def fpow(j):
    r = v.copy()
    for _ in range(j): r = rmul(r, r)
    return r

from flint import fmpz_mat, fmpq_mat
def solve1(g, cvec):
    A = fmpz_mat(N, N)
    for i in range(N):
        e = np.zeros(N, dtype=np.int64); e[i] = 1
        col = np.convolve(np.asarray(g, dtype=object), e.astype(object))
        o = col[:N].copy(); o[:len(col) - N] -= col[N:]
        for k in range(N): A[k, i] = int(o[k])
    rhs = fmpq_mat(N, 1)
    for k in range(N): rhs[k, 0] = int(cvec[k])
    U = fmpq_mat(A).solve(rhs)
    nm, dn = U.numer_denom()
    return np.array([int(nm[k, 0]) for k in range(N)], dtype=object), int(dn)

vs = {}
for j in range(len(Rs)):
    if j <= 3:
        arr, d = solve1(fpow(j), Rs[j])
        assert d == 1, 'v_{j+1} non-integral at %d' % j
        vs[j + 1] = arr
        print('v_%d: cmax %d' % (j + 1, abs(arr).max()), flush=True)

# y = R_1 * R_0  ==  f^3 * v_2 * v_1 ?
for mod in (1049089, 10007):
    y = mrmul(Rs[1] % mod, Rs[0] % mod, mod)
    W = mrmul(vs[2] % mod, vs[1] % mod, mod)
    lhs = mrmul(mpow(v % mod, 3, mod), W, mod)
    print('mod %d: R1*R0 == f^3*v2*v1 : %s' % (mod, np.array_equal(y, lhs)), flush=True)

# --- mini endgame test 2: minv ---
a = Rs[0]
for m in (1049089, 1000003):
    ai = minv(a % m, m)
    if ai is None:
        print('minv mod %d: not invertible' % m)
        continue
    chk = mrmul(a % m, ai, m)
    one = np.zeros(N, dtype=object); one[0] = 1
    print('minv mod %d ok: %s' % (m, np.array_equal(chk % m, one)), flush=True)

# --- mini test 3: the v-ratio identity (endgame y-chain core) ---
# v_{j+1}/v_j^2 == R_j / R_{j-1}^2  (mod small prime), for j=1
for mod in (10007,):
    lhs = mrmul(vs[2] % mod, minv(obj_rmul(vs[1], vs[1]) % mod, mod), mod)
    rhs = mrmul(Rs[1] % mod, minv(obj_rmul(Rs[0], Rs[0]) % mod, mod), mod)
    print('v2/v1^2 == R1/R0^2 mod %d: %s' % (mod, np.array_equal(lhs, rhs)), flush=True)
print('mini endgame tests done', flush=True)
