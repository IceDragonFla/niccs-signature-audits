import numpy as np
rng = np.random.default_rng(20261016)
N = 256
def rmul(a, b):
    c = np.convolve(a.astype(np.int64), b.astype(np.int64))
    out = c[:N].copy()
    out[:len(c) - N] -= c[N:]
    return out
def rconj(v):
    o = np.zeros(N, dtype=np.int64); o[0] = v[0]; o[1:] = -v[N:0:-1]; return o

s = np.fromfile('s_true_1m.bin', dtype=np.int32).reshape(3, N).astype(np.int64)
v = s[0].copy()
gam = rmul(v, rconj(v))
G1 = rmul(v, rconj(s[1]))   # gamma_01 = s0 * s1bar
G2 = rmul(v, rconj(s[2]))   # gamma_02 = s0 * s2bar
print('v coeff range: [%d,%d]  gamma range: [%d,%d]' % (v.min(), v.max(), gam.min(), gam.max()))
assert np.array_equal(rmul(v, rconj(v)), gam)

# 12 generators of the ideal J=(gamma_01)+(gamma_02)=(s0):
#   rows 1-2 bare, then random small multiples
gens = [G1.copy(), G2.copy()]
for i in range(5):
    w = rng.integers(-2, 3, N)
    gens.append(rmul(G1, w))
for i in range(5):
    w = rng.integers(-2, 3, N)
    gens.append(rmul(G2, w))
gens = np.array(gens)
print('generator matrix:', gens.shape, 'coeff max', abs(gens).max())

# packing: 12 x 8 matrix over Z[x]/(x^32+1); A[i,j] = strided chunk j of gens[i]
F = 8; n = N // F
A = [[gens[i, j::F] for j in range(F)] for i in range(12)]

def pol(coeffs):
    nz = np.nonzero(coeffs)[0]
    if len(nz) == 0:
        return '0'
    return '+'.join(('(%d)*x^%d' % (int(coeffs[k]), k)) for k in nz[::-1])

with open('pari_input.gp', 'w') as f:
    f.write('GAM = %s;\n' % pol(gam))
    f.write('VTRUE = %s;\n' % pol(v))
    rows = ['[' + ','.join(pol(A[i][j]) for j in range(F)) + ']' for i in range(12)]
    f.write('AMAT = [' + ';'.join(rows) + '];\n')
print('pari_input.gp written')

# quick numeric self-check of the packing convention (match PARI polvec/vecpol)
def vecpol_pari(rows, n=32, fac=F):
    # PARI: multiP(a, n*fac, fac): coeff i of a -> x^(i*fac); then * x^(j-1) summed
    out = np.zeros(N, dtype=np.int64)
    for j in range(fac):
        a = rows[j]  # chunk j, deg < n
        for k in range(len(a)):
            out[k * fac + j] += a[k]
    return out
ok = all(np.array_equal(vecpol_pari(A[i]), gens[i]) for i in range(12))
print('packing self-check (coeff k*F+j convention):', ok)
