import numpy as np
N=256; H=128
def rmul(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
def negsh(v,j,n):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
def sigma(a):
    return a*((-1)**np.arange(N))
def Nrel(a):
    p=rmul(a,sigma(a),N)
    assert np.abs(p[1::2]).max()==0
    return p[0::2].copy()
def ymul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:H].copy(); out[:len(c)-H]-=c[H:]
    return out
def ytimes(P):
    o=np.zeros(H,dtype=np.int64); o[1:]=P[:H-1]; o[0]=-P[H-1]
    return o
n_s0=Nrel(s[0])
A=s[0][0::2].copy(); B=s[0][1::2].copy()
form=ymul(A,A)-ytimes(ymul(B,B))
print('check1 A^2-yB^2==Nrel(s0):',np.array_equal(form,n_s0))
n01=Nrel(rmul(s[0],rconj(s[1],N),N))
n02=Nrel(rmul(s[0],rconj(s[2],N),N))
G=np.array([negsh(n01,j,H) for j in range(H)]+[negsh(n02,j,H) for j in range(H)])
from numpy.linalg import matrix_rank
print('check2 rank:',matrix_rank(G.astype(float)))
c=np.linalg.lstsq(G.astype(float).T,n_s0.astype(float),rcond=None)[0]
cr=np.round(c).astype(np.int64)
ok=np.array_equal(cr.astype(object)@G.astype(object),n_s0.astype(object))
print('check2 N(s0) in (n01,n02):',ok)
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
from fpylll import IntegerMatrix, LLL
Ml=IntegerMatrix(2*H,H)
for i in range(2*H):
    for j in range(H): Ml[i,j]=int(G[i][j])
LLL.reduction(Ml)
rows=np.array([[Ml[i,j] for j in range(H)] for i in range(2*H)])
nz=[r for r in rows if any(r)]
B128=np.vstack([np.array(r,dtype=np.int64) for r in nz[:H]])
d1=bareiss(B128.tolist())
M3=np.array([negsh(n_s0,j,H) for j in range(H)])
d2=bareiss(M3.tolist())
import math
print('check2 dets equal:',d1!=0 and d2!=0 and abs(d1)==abs(d2))
print('  log10|det| =',round(math.log10(abs(d2)),1) if d2 else None)
tw=np.exp(1j*np.pi*np.arange(N)/N)
SH=np.fft.fft(s[0].astype(float)*tw)
SHn=np.fft.fft(n_s0.astype(float)*np.exp(1j*np.pi*np.arange(H)/H))
pred=SH[0:H]*SH[H:N]
print('check3 sigma_j(N(s0)) == sigma_j(s)*sigma_j+128(s): relerr=%.2e'%float(np.abs(pred/SHn-1).max()))
print('check3b |sigma_j(s)| fully determined by N:',float(np.abs(np.abs(pred)**2/np.abs(SHn)**2-1).max()))
cA=np.convolve(A,A); cB=np.convolve(B,B)
print('check4 honest lift: A^2 deg',len(cA)-1,'; N~ has',len(cA),'y-coeffs; R-element N known only mod y^128+1 ->',len(cA)-H,'high coeffs unknown')
print('size: ||A||=%.1f ||B||=%.1f ||N(s0)||=%.1f'%(np.linalg.norm(A.astype(float)),np.linalg.norm(B.astype(float)),np.linalg.norm(n_s0.astype(float))))
d128=abs(d2)**(1.0/H)
import math as m
gh=m.sqrt(H/(2*m.pi*m.e))*d128
print('dim128 ideal: det^(1/128)=%.1f GH=%.1f  planted ||N(s0)||=%.1f'%(d128,gh,np.linalg.norm(n_s0.astype(float))))
