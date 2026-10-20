import numpy as np, time
from flint import fmpz_mat
N=256
def rmul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:N].copy(); out[:len(c)-N]-=c[N:]; return out
def rconj(v):
    o=np.zeros(N,dtype=np.int64); o[0]=v[0]; o[1:]=-v[N:0:-1]; return o
def sigma(v): return v*((-1)**np.arange(N))
def negsh(w,j):
    o=np.roll(w,j)
    if j: o[:j]=-o[:j]
    return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
v=s[0].copy()
gam=rmul(v,rconj(v))
G1=rmul(v,rconj(s[1])); G2=rmul(v,rconj(s[2]))
from fpylll import IntegerMatrix, LLL as FLLL
def hnf_basis(gens):
    A=fmpz_mat(len(gens),N)
    for i,g in enumerate(gens):
        for j in range(N): A[i,j]=int(g[j])
    H=A.hnf()
    rows=[[int(H[i,j]) for j in range(N)] for i in range(len(gens))]
    nz=[r for r in rows if any(r)]
    M=IntegerMatrix(len(nz),N)
    for i,r in enumerate(nz):
        for j in range(N): M[i,j]=int(r[j])
    FLLL.reduction(M)
    rows2=[[int(M[i,j]) for j in range(N)] for i in range(len(nz))]
    nz2=[np.array(r,dtype=np.int64) for r in rows2 if any(r)]
    return np.vstack(nz2)
t0=time.time()
IB=hnf_basis([negsh(G1,j) for j in range(N)]+[negsh(G2,j) for j in range(N)])
print('I basis (HNF): %d rows %.0fs'%(len(IB),time.time()-t0),flush=True)
def ideal_mul(B1,B2):
    gens=[rmul(a,b) for a in B1 for b in B2]
    return hnf_basis(gens)
t0=time.time()
I2=ideal_mul(IB,IB)
print('I^2 (HNF): %d rows %.0fs'%(len(I2),time.time()-t0),flush=True)
sgIB=np.array([sigma(b) for b in IB])
t0=time.time()
sI2=ideal_mul(sgIB,sgIB)
print('sigma(I)^2 %.0fs'%(time.time()-t0),flush=True)
Rb=np.eye(N,dtype=np.int64)
gamR=np.array([rmul(gam,e) for e in Rb])
ALL=np.vstack([I2,gamR,sI2])
t0=time.time()
LAM=hnf_basis(list(ALL))
print('Lambda(N=2) HNF: %d rows %.0fs'%(len(LAM),time.time()-t0),flush=True)
np.save('gs_Lam2.npy',LAM)
def in_lattice(LAM,x):
    Bf=LAM.astype(float)
    c=np.linalg.lstsq(Bf.T,x.astype(float),rcond=None)[0]
    cr=np.round(c).astype(np.int64)
    return np.array_equal(cr.astype(object)@LAM.astype(object),x.astype(object))
g0=rmul(v,v); g2=rmul(rconj(v),rconj(v))
print('v^2 in Lambda:',in_lattice(LAM,g0))
print('gamma in Lambda:',in_lattice(LAM,gam))
print('vbar^2 in Lambda:',in_lattice(LAM,g2))
def bareiss(Ao):
    A=[list(map(int,r)) for r in Ao]; n=len(A); sign=1; prev=1
    for k in range(n-1):
        if A[k][k]==0:
            for i in range(k+1,n):
                if A[i][k]!=0:
                    A[k],A[i]=A[i],A[k]; sign=-sign; break
            else: return 0
        for i in range(k+1,n):
            for j in range(k+1,n):
                A[i][j]=(A[i][j]*A[k][k]-A[i][k]*A[k][j])//prev
            A[i][k]=0
        prev=A[k][k]
    return sign*A[n-1][n-1]
import math
SH=np.fft.fft(v.astype(float)*np.exp(1j*np.pi*np.arange(N)/N))
NormV=int(np.round(np.prod(np.abs(SH))))
print('log10|Norm(v)|=%.2f'%math.log10(abs(NormV)))
dL=bareiss(LAM.tolist())
print('log10|det Lambda|=%.2f  vs 2*log10|Norm|=%.2f'%(math.log10(abs(dL)),2*math.log10(abs(NormV))))
norms=np.linalg.norm(LAM.astype(float),axis=1)
print('shortest 5 norms of Lambda:',np.round(np.sort(norms)[:5],1))
print('||v^2||=%.1f ||gamma||=%.1f ||vbar^2||=%.1f'%(np.linalg.norm(g0.astype(float)),np.linalg.norm(gam.astype(float)),np.linalg.norm(g2.astype(float))))
