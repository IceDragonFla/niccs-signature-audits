import numpy as np, time
from fpylll import IntegerMatrix, LLL
N=256
def rmul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:N].copy(); out[:len(c)-N]-=c[N:]; return out
def rconj(v):
    o=np.zeros(N,dtype=np.int64); o[0]=v[0]; o[1:]=-v[N:0:-1]; return o
def sigma(v): return v*((-1)**np.arange(N))
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
v=s[0].copy()
rmul(v,rconj(v))
G1=rmul(v,rconj(s[1])); G2=rmul(v,rconj(s[2]))
def negsh(w,j):
    o=np.roll(w,j)
    if j: o[:j]=-o[:j]
    return o
# ideal basis of I=(v) from (G1,G2) generators: x^j*G1, x^j*G2
GEN=np.array([negsh(G1,j) for j in range(N)]+[negsh(G2,j) for j in range(N)])
M=IntegerMatrix(2*N,N)
for i in range(2*N):
    for j in range(N): M[i,j]=int(GEN[i][j])
t0=time.time(); LLL.reduction(M)
rows=np.array([[M[i,j] for j in range(N)] for i in range(2*N)])
IB=np.vstack([np.array(r,dtype=np.int64) for r in rows if any(r)])  # I basis (256 rows)
print('I basis: %d rows, LLL %.0fs'%(len(IB),time.time()-t0))
# ideal product: generators b_i*b_j -> reduce
def ideal_mul(B1,B2,cap=600):
    gens=[]
    for a in B1:
        for b in B2[:cap]:
            gens.append(rmul(a,b))
    Gm=IntegerMatrix(len(gens),N)
    for i in range(len(gens)):
        for j in range(N): Gm[i,j]=int(gens[i][j])
    LLL.reduction(Gm)
    rows=np.array([[Gm[i,j] for j in range(N)] for i in range(len(gens))])
    nz=[np.array(r,dtype=np.int64) for r in rows if any(r)]
    return np.vstack(nz[:N])
t0=time.time()
I2=ideal_mul(IB,IB,cap=256)
print('I^2: %d rows (%.0fs)'%(len(I2),time.time()-t0))
sgIB=np.array([sigma(b) for b in IB])
sI2=ideal_mul(sgIB,sgIB,cap=256)
print('sigma(I)^2 done')
# N=4 lattice Lambda: L0=I^4? we only have I^2; use N=2 first: Lambda2 = I^2 + gamma*R + sigma(I)^2
Rb=np.eye(N,dtype=np.int64)
gam2=rmul(gam,gam)
Lparts=[I2, np.array([rmul(gam,e) for e in Rb]), sI2]
ALL=np.vstack(Lparts)
M2=IntegerMatrix(len(ALL),N)
for i in range(len(ALL)):
    for j in range(N): M2[i,j]=int(ALL[i][j])
t0=time.time(); LLL.reduction(M2)
rows=np.array([[M2[i,j] for j in range(N)] for i in range(len(ALL))])
nzB=[np.array(r,dtype=np.int64) for r in rows if any(r)]
LAM=np.vstack(nzB[:N])
print('Lambda(N=2): %d rows, LLL %.0fs'%(len(LAM),time.time()-t0))
# structural checks at N=2: g_0=v^2, g_1=gamma, g_2=vbar^2 must be IN Lambda
def in_lattice(LAM,x):
    Bf=LAM.astype(float)
    c=np.linalg.lstsq(Bf.T,x.astype(float),rcond=None)[0]
    cr=np.round(c).astype(np.int64)
    return np.array_equal(cr.astype(object)@LAM.astype(object),x.astype(object))
g0=rmul(v,v); g2=rmul(rconj(v),rconj(v))
print('v^2 in Lambda:',in_lattice(LAM,g0))
print('gamma in Lambda:',in_lattice(LAM,gam))
print('vbar^2 in Lambda:',in_lattice(LAM,g2))
norms=np.linalg.norm(LAM.astype(float),axis=1)
print('Lambda shortest 5 norms:',np.round(np.sort(norms)[:5],1))
print('||v^2||=%.1f ||gamma||=%.1f'%(np.linalg.norm(g0.astype(float)),np.linalg.norm(gam.astype(float))))
# det(Lambda) vs Norm(v)^2
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
dL=bareiss(LAM.tolist())
SH=np.fft.fft(v.astype(float)*np.exp(1j*np.pi*np.arange(N)/N))
NormV=round(np.prod(np.abs(SH).astype(object)))
print('log10|det Lambda| =%.1f ; log10|Norm(v)^2|=%.1f'%(math.log10(abs(dL)),2*math.log10(abs(NormV))))
np.save('gs_Lambda2.npy',LAM); np.save('gs_I_basis.npy',IB)
