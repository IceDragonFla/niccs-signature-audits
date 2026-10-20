import numpy as np, time
from fpylll import IntegerMatrix, LLL, BKZ, GSO, FPLLL
FPLLL.set_precision(212)
N=256
B0=np.load('s_true_1m.bin.b0.npy')
def imat(M2):
    M=IntegerMatrix(M2.shape[0],M2.shape[1])
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]): M[i,j]=int(M2[i,j])
    return M
def stats(M):
    G=GSO.Mat(M,float_type='mpfr')
    try:
        G.update_gso()
        r0=G.get_r(0,0)**0.5; rl=G.get_r(N-1,N-1)**0.5
    except Exception:
        rl=-1; r0=-1
    B2=np.array([[M[i,j] for j in range(N)] for i in range(N)],dtype=np.int64)
    return float(np.min(np.linalg.norm(B2.astype(float),axis=1))),r0,rl
print('orig: b1=%.2f r0=%.2f r_last=%.4f'%stats(imat(B0)),flush=True)
# (a) reversed rows + LLL mpfr delta .999
t0=time.time()
Ma=imat(B0[::-1].copy())
LLL.reduction(Ma,delta=0.999)
print('rev-LLL %.0fs: b1=%.2f r0=%.2f r_last=%.4f'%((time.time()-t0,)+stats(Ma)),flush=True)
np.save('b_rev.npy',np.array([[Ma[i,j] for j in range(N)] for i in range(N)],dtype=np.int64))
# (b) random unimodular restarts
rng=np.random.default_rng(5)
best=None
for t in range(3):
    U=np.eye(N,dtype=np.int64)
    for _ in range(3*N):
        i,j=rng.integers(N),rng.integers(N)
        if i!=j: U[i]+=int(rng.choice([-2,-1,1,2]))*U[j]
    Bx=(U@B0).astype(np.int64)
    Mx=imat(Bx)
    t1=time.time()
    LLL.reduction(Mx,delta=0.999)
    s=stats(Mx)
    print('unimod%d-LLL %.0fs: b1=%.2f r0=%.2f r_last=%.4f'%(t,time.time()-t1,*s),flush=True)
    if best is None or s[0]<best[0][0]: best=(s,Mx)
Bs,Mb=best
np.save('b_uni.npy',np.array([[Mb[i,j] for j in range(N)] for i in range(N)],dtype=np.int64))
