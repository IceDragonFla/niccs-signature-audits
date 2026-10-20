import numpy as np
import cvxpy as cp
N=256; H=128
def rmul_int(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj_int(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
gam00=rmul_int(s[0],rconj_int(s[0],N),N)
E0=gam00[0::2].astype(float); O0=gam00[1::2].astype(float)
def sigma(a): return a*((-1)**np.arange(N))
nrel=rmul_int(s[0],sigma(s[0]),N)[0::2].astype(float)
A=s[0][0::2].astype(float); B=s[0][1::2].astype(float)
xtrue=np.concatenate([A,B])
# ---- build W_k (R' mult coeff-extraction), T (involution), Ry (y-shift), Rinv (yinv)
W=[np.zeros((H,H)) for _ in range(H)]
for k in range(H):
    for i in range(H):
        j1=k-i
        if 0<=j1<H: W[k][i,j1]+=1.0
        j2=k+H-i
        if 0<=j2<H: W[k][i,j2]-=1.0
T=np.zeros((H,H))
for j in range(H):
    e=np.zeros(H); e[j]=1.0
    T[:,j]=rconj_int(np.r_[e.astype(np.int64),np.zeros(0)],H)*0+ (lambda v: np.r_[v[0], -v[:0:-1]])(e)
Ry=np.zeros((H,H))
for k in range(1,H): Ry[k,k-1]=1.0
Ry[0,H-1]=-1.0
Rinv=np.zeros((H,H))
for k in range(H-1): Rinv[k,k+1]=1.0
Rinv[H-1,0]=-1.0
def sym(M): return 0.5*(M+M.T)
# quadratic form check helper: value qf(Q, x) with Q full 256
def qf(Q,x): return float(x@Q@x)
# ---- assemble constraint forms
forms=[]; cvals=[]
for k in range(H):
    F1=sym(W[k]@T)
    Q=np.zeros((2*H,2*H)); Q[:H,:H]=F1; Q[H:,H:]=F1
    forms.append(Q); cvals.append(E0[k])
for k in range(H):
    C=(W[k]@T).T+sum(Rinv[k,m]*(W[m]@T) for m in range(H))
    Q=np.zeros((2*H,2*H)); Q[:H,H:]=0.5*C; Q[H:,:H]=0.5*C.T
    forms.append(Q); cvals.append(O0[k])
for k in range(H):
    Fb=sum(Ry[k,m]*W[m] for m in range(H))
    Q=np.zeros((2*H,2*H)); Q[:H,:H]=sym(W[k]); Q[H:,H:]=-sym(Fb)
    forms.append(Q); cvals.append(nrel[k])
# magnitude constraints: twisted embedding of s=A+xB
tw=np.exp(1j*np.pi*np.arange(N)/N)
def emb_row(j):
    # complex row: sigma_j(s) = sum_i s_i tw_i e^{-2 pi i j i / N}; s_{2m}=A_m, s_{2m+1}=B_m
    row=np.zeros(2*H,dtype=complex)
    for m in range(H):
        i1=2*m; i2=2*m+1
        ph1=tw[i1]*np.exp(-2j*np.pi*j*i1/N)
        ph2=tw[i2]*np.exp(-2j*np.pi*j*i2/N)
        row[m]=ph1; row[H+m]=ph2
    return row
G=np.fft.fft(gam00.astype(float)*tw)
mags2=np.abs(G)  # |sigma_j(s)|^2 (scale: |G| = N*|sigma|^2? check below)
# scale: sum_i mags2 = ... verify with truth
sc=np.zeros(N)
for j in range(N):
    r=emb_row(j)
    sc[j]=abs(r@xtrue)**2
print('scale ratio mags2/|sigma_j|^2 first3:',(mags2[:3]/sc[:3]))
forms_mag=[]
for j in range(N):
    r=emb_row(j)
    Rr=r.real; Ri=r.imag
    Q=np.outer(Rr,Rr)+np.outer(Ri,Ri)
    forms.append(Q); cvals.append(sc[j])
# verify all forms at truth
maxerr=0.0
for Q,c in zip(forms,cvals):
    maxerr=max(maxerr,abs(qf(Q,xtrue)-c))
print('form verification at truth: max |Q(x)-c| = %.3e  (nforms=%d)'%(maxerr,len(forms)))
np.save('pl_cvals.npy',np.array(cvals))
# ---- PhaseLift SDP
X=cp.Variable((2*H,2*H),PSD=True)
cons=[cp.trace(forms[i]@X)==cvals[i] for i in range(len(forms))]
obj=cp.Minimize(cp.trace(X))   # trace heuristic for rank-1
prob=cp.Problem(obj,cons)
print('solving SDP (dim %d, %d constraints)...'%(2*H,len(forms)),flush=True)
import time
t0=time.time()
try:
    prob.solve(solver=cp.SCS,verbose=False,max_iters=20000,eps=1e-9)
    print('solved %.0fs status=%s'%(time.time()-t0,prob.status),flush=True)
    Xv=X.value
    wv,vv=np.linalg.eigh(Xv)
    print('top eigs: %.3e %.3e %.3e  rank-1 ratio=%.3f'%(wv[-1],wv[-2],wv[-3],wv[-1]/max(wv[-2],1e-12)))
    v=vv[:,-1]
    v=v*np.sqrt(wv[-1])*(1 if v[:H].sum()>=0 else -1)/np.linalg.norm(v)
    Ab=v[:H]; Bb=v[H:]
    print('candidate norms: ||A||=%.2f ||B||=%.2f'%(np.linalg.norm(Ab),np.linalg.norm(Bb)))
    Ar=np.round(Ab); Br=np.round(Bb)
    ok=int((Ar==s[0][0::2]).sum()+(Br==s[0][1::2]).sum())
    print('rounded overlap: %d/256  maxdev=%.3f'%(ok,np.abs(np.r_[Ab-Ar,Bb-Br]).max()))
except Exception as e:
    print('SDP fail:',repr(e))
