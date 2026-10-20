import numpy as np, time, sys
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
FPLLL.set_precision(212)
def rmul(a,b,n):
    c=np.convolve(a,b); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
def negsh(v,j,n):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
def run(n):
    rng=np.random.default_rng(7+n)
    s=[rng.choice([-2,-1,0,1,2],size=n,p=[.07,.17,.52,.17,.07]) for _ in range(3)]
    sb=[rconj(x,n) for x in s]
    gam={(k,kp):rmul(s[k],sb[kp],n) for k in range(3) for kp in range(3)}
    G=np.array([negsh(gam[(0,i)],j,n) for i in range(3) for j in range(n)])
    M=IntegerMatrix(3*n,n)
    for i in range(3*n):
        for j in range(n): M[i,j]=int(G[i][j])
    LLL.reduction(M)
    rows=np.array([[M[i,j] for j in range(n)] for i in range(3*n)])
    nz=[r for r in rows if any(r)]
    B=np.vstack(nz[:n]).astype(np.int64)
    print('n=%d: max|entry|=%d  ||b1||=%.1f  ||s0||=%.2f'%(n,int(np.abs(B).max()),float(np.min(np.linalg.norm(B.astype(float),axis=1))),float(np.linalg.norm(s[0].astype(float)))),flush=True)
    Mx=IntegerMatrix(n,n)
    for i in range(n):
        for j in range(n): Mx[i,j]=int(B[i][j])
    t1=time.time()
    try:
        BKZ.reduction(Mx,BKZ.Param(block_size=20,max_loops=5,float_type='mpfr'))
        B2=np.array([[Mx[i,j] for j in range(n)] for i in range(n)],dtype=np.int64)
        g00=gam[(0,0)]
        f=any(any(v) and np.array_equal(rmul(v.astype(np.int64),rconj(v.astype(np.int64),n),n),g00) for v in B2)
        print('  n=%d BKZ-20 clean %.1fs ||b1||=%.1f found=%s'%(n,time.time()-t1,float(np.min(np.linalg.norm(B2.astype(float),axis=1))),f),flush=True)
    except Exception as e:
        print('  n=%d BKZ-20 ABORT %.1fs %r'%(n,time.time()-t1,e),flush=True)
for n in [int(x) for x in sys.argv[1:]]:
    run(n)
