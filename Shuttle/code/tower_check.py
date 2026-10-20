import numpy as np
N=256; H=128
def rmul(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
# sigma: x -> -x on R=Z[x]/(x^256+1);  sigma(a)(i) = (-1)^i a(i)
def sigma(a):
    return a*((-1)**np.arange(N))
# relative norm to R'=Z[y]/(y^128+1), y=x^2:  Nrel(a) = a*sigma(a) in R; even poly -> in R'
def Nrel(a):
    p=rmul(a,sigma(a),N)
    assert np.abs(p[1::2]).max()==0, 'not even!'
    return p[0::2].copy()   # 128 coeffs in y
n_s0=Nrel(s[0])
# check 1: sigma-norm form  s = A(y) + x B(y)
A=s[0][0::2].copy(); B=s[0][1::2].copy()
lhs=rmul(A,A,H)-np.concatenate(([0],rmul(B,B,H)[:H-1]))*0  # placeholder, do A^2 - y*B^2 mod y^128+1 properly
def ymul(a,b):  # multiply in Z[y]/(y^128+1)
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:H].copy()
    out[:len(c)-H]-=c[H:]
    return out
form=ymul(A,A)-np.roll(ymul(B,B),-1)  # y*B^2 = roll by -1 with fold... careful: y*P shifts, and y^128=-1 wrap
# redo properly: y*P(i) = P(i-1), with y^128=-1: (y*P)[127] = -P[127]... no: (y*P)[k]=P[k-1] for k>=1; [0]=0; plus fold -P[127] at [0]
def ytimes(P):
    o=np.zeros(H,dtype=np.int64)
    o[1:]=P[:H-1]
    o[0]=-P[H-1]
    return o
form=ymul(A,A)-ytimes(ymul(B,B))
print('check1 sigma-norm-form A^2 - yB^2 == Nrel(s0):', np.array_equal(form,n_s0))
# check 2: dim-128 ideal (n01,n02) = (N(s0)) membership
n01=Nrel(rmul(s[0],rconj(s[1],N),N))
n02=Nrel(rmul(s[0],rconj(s[2],N),N))
def negsh(v,j,n):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
G=np.array([negsh(n01,j,H) for j in range(H)]+[negsh(n02,j,H) for j in range(H)])
# rank/det compare with (N(s0)) lattice
from numpy.linalg import matrix_rank
print('check2: gen matrix rank:',matrix_rank(G.astype(float)))
M2=np.array([negsh(n_s0,j,H) for j in range(H)]).astype(float)
# membership: solve G c = n_s0 exactly (float then verify)
c=np.linalg.lstsq(G.astype(float).T,n_s0.astype(float),rcond=None)[0]
cr=np.round(c).astype(np.int64)
cr2=cr[:2*H]
ok=np.array_equal(cr2.astype(object)@G.astype(object),n_s0.astype(object))
    ok=np.array_equal(cr.astype(object)@G.astype(object),n_s0.astype(object))
print('check2: N(s0) in (n01,n02):',ok)
# det equality via bareiss (int)
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
# use LLL for basis then bareiss on reduced
try:
    from fpylll import IntegerMatrix, LLL
    Ml=IntegerMatrix(2*H,H)
    for i in range(2*H):
        for j in range(H): Ml[i,j]=int(G[i][j])
    LLL.reduction(Ml)
    rows=np.array([[Ml[i,j] for j in range(H)] for i in range(2*H)])
    nz=[r for r in rows if any(r)]
    B128=np.vstack(nz[:H])
    d1=bareiss(B128.tolist())
    Ml2=IntegerMatrix(H,H)
    for i in range(H):
        for j in range(H): Ml2[i,j]=int(negsh(n_s0,j,H)[j] if False else negsh(n_s0,j,H)[j])
    # (N(s0)) basis: x^j*n_s0 in y-ring
    M3=np.array([negsh(n_s0,j,H) for j in range(H)])
    d2=bareiss(M3.tolist())
    print('check2 dets: |J128|=|N(s0)|^? :',d1!=0 and d2!=0, 'd1/d2 =', (d1//d2 if d2!=0 and d1%d2==0 else (d1,d2)))
except ImportError:
    print('no fpylll locally; skipping det part')
# check 3: embedding moduli of s0 determined by N
i=np.arange(N); tw=np.exp(1j*np.pi*i/N)
SH=np.fft.fft(s[0].astype(float)*tw)
SHn=np.fft.fft(n_s0.astype(float)*np.exp(1j*np.pi*np.arange(H)/H))
# |sigma_j(N)| should equal |sigma_j(s)|*|sigma_j+128(s)|
print('check3: |sigma_j(s)|^2*|sigma_j+128(s)|^2 vs sigma_j(N): relerr=%.2e'%(
    np.abs(SH[0:H]**2*SH[H:N]**2/SHn-1).__abs__().max()))
# check 4: honest lift degree: N~(y) = A^2-yB^2 as honest poly (deg<=255 in y? A,B deg<=127 -> A^2 deg 254)
cA=np.convolve(A,A); cB=np.convolve(B,B)
print('check4: honest A^2-yB^2 degree bound:', len(cA)-1, '+1 ->', len(cA), 'coeffs vs R\'s 128: lift has', len(cA)-H, 'unknown high coeffs')
