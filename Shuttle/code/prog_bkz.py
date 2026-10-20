import numpy as np, time, sys, os
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
FPLLL.set_precision(212)
N=256
def ring_mul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:N].copy(); out[:len(c)-N]-=c[N:]
    return out
def ring_conj(s):
    o=np.zeros(N,dtype=np.int64)
    o[0]=s[0]; o[1:]=-s[N:0:-1]
    return o
def negshift(v,j):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
def fam_eq(a,b):
    for eps in (1,-1):
        for j in range(N):
            if np.array_equal(a,eps*negshift(b,j)): return (eps,j)
    return None
def load_key(fn):
    if '1m' in fn: return np.fromfile(fn,dtype=np.int32).reshape(3,N).astype(np.int64)
    if '4m' in fn: return np.fromfile(fn,dtype=np.int64).reshape(3,N)
    arr=np.frombuffer(open(fn,'rb').read(),dtype='<i4')
    return arr[3552//4:3552//4+768].reshape(3,N).astype(np.int64)
def imat(M2):
    M=IntegerMatrix(M2.shape[0],M2.shape[1])
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]): M[i,j]=int(M2[i,j])
    return M
def cp(M):
    R=IntegerMatrix(M.nrows,M.ncols)
    for i in range(M.nrows):
        for j in range(M.ncols): R[i,j]=M[i,j]
    return R
key=sys.argv[1]
s=load_key(key)
sb=[ring_conj(s[k]) for k in range(3)]
gam={(k,kp):ring_mul(s[k],sb[kp]) for k in range(3) for kp in range(3)}
g00=gam[(0,0)]
cache=os.path.basename(key)+'.b0.npy'
if os.path.exists(cache):
    B0=np.load(cache)
    print('loaded cached LLL basis',flush=True)
else:
    G=np.zeros((768,N),dtype=np.int64)
    r=0
    for i in range(3):
        for j in range(N):
            G[r]=negshift(gam[(0,i)],j); r+=1
    t0=time.time()
    M=imat(G); LLL.reduction(M)
    rows=np.array([[M[i,j] for j in range(N)] for i in range(768)],dtype=object)
    nz=[np.array(r_,dtype=np.int64) for r_ in rows if any(r_)]
    print('nonzero rows after LLL: %d'%len(nz),flush=True)
    B0=np.vstack(nz[:256])
    if len(nz)!=256: print('WARNING: nonzero row count != 256',flush=True)
    # exact det check via Bareiss vs multiplication matrix norm
    def bareiss_det(Ao):
        import copy
        A=[list(map(int,r)) for r in Ao]
        n=len(A); sign=1; prev=1
        for k in range(n-1):
            if A[k][k]==0:
                for i in range(k+1,n):
                    if A[i][k]!=0:
                        A[k],A[i]=A[i],A[k]; sign=-sign; break
                else:
                    return 0
            for i in range(k+1,n):
                for j in range(k+1,n):
                    A[i][j]=(A[i][j]*A[k][k]-A[i][k]*A[k][j])//prev
                A[i][k]=0
            prev=A[k][k]
        return sign*A[n-1][n-1]
    Mmul=np.zeros((N,N),dtype=np.int64)
    for j in range(N): Mmul[:,j]=negshift(s[0],j)
    import datetime
    tD=time.time()
    d1=bareiss_det(B0.tolist())
    d2=bareiss_det(Mmul.tolist())
    print('det(B0)=%d  det(Mmul)=%d  equal=%s  (bareiss %.0fs)'%(d1,d2,d1==d2,time.time()-tD),flush=True)
    np.save(cache,B0)
    print('LLL %.1fs ||b1||=%.1f ||s0||=%.2f'%(time.time()-t0,np.linalg.norm(B0[0].astype(float)),np.linalg.norm(s[0].astype(float))),flush=True)
def scan(B,topk=80):
    norms=np.array([np.linalg.norm(b.astype(float)) for b in B])
    for idx in np.argsort(norms)[:topk]:
        v=B[idx]
        if any(v)==False: continue
        if np.array_equal(ring_mul(v,ring_conj(v)),g00): return v
    return None
cur=imat(B0)
for beta in (20,30,40,50,60,70,80,90,100):
    t1=time.time()
    nxt=cp(cur)
    try:
        BKZ.reduction(nxt,BKZ.Param(block_size=beta,max_duration=1500,float_type='mpfr'))
    except Exception as e:
        print('BKZ-%d interrupted (%r), using current basis'%(beta,e),flush=True)
    B=np.array([[nxt[i,j] for j in range(N)] for i in range(256)],dtype=np.int64)
    n1=float(np.min([np.linalg.norm(b.astype(float)) for b in B]))
    v=scan(B)
    print('BKZ-%d (%.0fs): ||b1||=%.1f  found=%s'%(beta,time.time()-t1,n1,v is not None),flush=True)
    cur=nxt
    if v is not None:
        print('*** RECOVERED gauge=%s'%fam_eq(v,s[0]),flush=True)
        for i in (1,2):
            Mdiv=np.zeros((N,N),dtype=np.int64)
            for j in range(N): Mdiv[:,j]=negshift(s[0],j)
            c=np.linalg.lstsq(Mdiv.T.astype(float),gam[(0,i)].astype(float),rcond=None)[0]
            cr=np.round(c).astype(np.int64)
            ok=np.array_equal(cr.astype(object)@Mdiv.astype(object),gam[(0,i)].astype(object))
            print('  division block %d exact: %s'%(i,ok),flush=True)
        break
print('done',flush=True)
