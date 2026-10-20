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
def cp(M):
    R=IntegerMatrix(M.nrows,M.ncols)
    for i in range(M.nrows):
        for j in range(M.ncols): R[i,j]=M[i,j]
    return R
def run(n,betas):
    rng=np.random.default_rng(7+n)
    s=[rng.choice([-2,-1,0,1,2],size=n,p=[.07,.17,.52,.17,.07]) for _ in range(3)]
    sb=[rconj(x,n) for x in s]
    gam={(k,kp):rmul(s[k],sb[kp],n) for k in range(3) for kp in range(3)}
    G=np.array([negsh(gam[(0,i)],j,n) for i in range(3) for j in range(n)])
    M=IntegerMatrix(3*n,n)
    for i in range(3*n):
        for j in range(n): M[i,j]=int(G[i][j])
    t0=time.time(); LLL.reduction(M)
    rows=np.array([[M[i,j] for j in range(n)] for i in range(3*n)])
    nz=[r for r in rows if any(r)]
    B=np.vstack(nz[:n]).astype(np.int64)
    g00=gam[(0,0)]
    def found_in(B):
        for v in B:
            if any(v) and np.array_equal(rmul(v.astype(np.int64),rconj(v.astype(np.int64),n),n),g00): return True
        return False
    n1=float(np.min([np.linalg.norm(b.astype(float)) for b in B]))
    print('n=%d LLL %.1fs ||b1||=%.1f ||s0||=%.2f found=%s'%(n,time.time()-t0,n1,np.linalg.norm(s[0].astype(float)),found_in(B)),flush=True)
    if found_in(B): return
    Mb=IntegerMatrix(n,n)
    for i in range(n):
        for j in range(n): Mb[i,j]=int(B[i][j])
    for beta in betas:
        t1=time.time()
        cur=cp(Mb)
        try:
            BKZ.reduction(cur,BKZ.Param(block_size=beta,max_duration=900,float_type='mpfr'))
        except Exception as e:
            print('  BKZ-%d interrupted %r'%(beta,e),flush=True)
        B2=np.array([[cur[i,j] for j in range(n)] for i in range(n)],dtype=np.int64)
        n1=float(np.min([np.linalg.norm(b.astype(float)) for b in B2]))
        f=found_in(B2)
        print('  n=%d BKZ-%d (%.0fs): ||b1||=%.1f found=%s'%(n,beta,time.time()-t1,n1,f),flush=True)
        Mb=cur
        if f: break
if __name__=='__main__':
    n=int(sys.argv[1]); betas=[int(x) for x in sys.argv[2:]]
    run(n,betas)
