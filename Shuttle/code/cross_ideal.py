import numpy as np, time, sys
from fpylll import IntegerMatrix, LLL, BKZ
def rmul(a,b,n):
    c=np.convolve(a,b); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]
    return o
def negsh(v,j,n):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
def cp(M):
    R=IntegerMatrix(M.nrows,M.ncols)
    for i in range(M.nrows):
        for j in range(M.ncols): R[i,j]=M[i,j]
    return R
def load_key(fn,n):
    if '1m' in fn and 'mk' not in fn: return np.fromfile(fn,dtype=np.int32).reshape(3,n).astype(np.int64)
    if '4m' in fn: return np.fromfile(fn,dtype=np.int64).reshape(3,n)
    arr=np.frombuffer(open(fn,'rb').read(),dtype='<i4')
    return arr[3552//4:3552//4+3*n].reshape(3,n).astype(np.int64)
# test cross-only ideal on real keys at n=256: J=(gamma_01)+(gamma_02); check s0 in J
keys=[('1M','s_true_1m.bin'),('4M','s_true_4m.bin'),('mk1','mk1.bin_key.bin'),('mk2','mk2.bin_key.bin'),('mk3','mk3.bin_key.bin')]
for tag,fn in keys:
    s=load_key(fn,256)
    sb=[rconj(x,256) for x in s]
    g01=rmul(s[0],sb[1],256); g02=rmul(s[0],sb[2],256)
    G=np.array([negsh(g01,j,256) for j in range(256)]+[negsh(g02,j,256) for j in range(256)])
    M=IntegerMatrix(512,256)
    for i in range(512):
        for j in range(256): M[i,j]=int(G[i][j])
    t0=time.time(); LLL.reduction(M)
    rows=np.array([[M[i,j] for j in range(256)] for i in range(512)])
    nz=[r for r in rows if any(r)]
    B=np.vstack(nz[:256]).astype(np.int64)
    # membership of s0 in J
    Bf=B.astype(float)
    c=np.linalg.lstsq(Bf.T,s[0].astype(float),rcond=None)[0]
    ok=np.array_equal(np.round(c).astype(np.int64).astype(object)@B.astype(object),s[0].astype(object))
    print('%s: cross-only ideal, s0 in J: %s  (LLL %.0fs, nonzero rows %d)'%(tag,ok,time.time()-t0,len(nz)),flush=True)
