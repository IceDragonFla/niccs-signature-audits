import numpy as np
N=256; H=128
def rmul(a,b,n):
    c=np.convolve(a.astype(np.float64),b.astype(np.float64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj(s,n):
    o=np.zeros(n); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
def rmul_int(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj_int(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
A=s[0][0::2].astype(np.float64); B=s[0][1::2].astype(np.float64)
def ymul(a,b):
    c=np.convolve(a,b); out=c[:H].copy(); out[:len(c)-H]-=c[H:]; return out
def ytimes(P):
    o=np.zeros(H); o[1:]=P[:H-1]; o[0]=-P[H-1]
    return o
def yconj(p):
    o=np.zeros(H); o[0]=p[0]; o[1:]=-p[H:0:-1]; return o
gam00=rmul_int(s[0],rconj_int(s[0],N),N)
E0=gam00[0::2].astype(np.float64)      # A*Ab + B*Bb
O0=(gam00[1::2]).astype(np.float64)    # coefficient of x^{2k+1} / x -> A*Bb + B*Ab
def sigma(a): return a*((-1)**np.arange(N))
def Nrel_int(a):
    p=rmul_int(a,sigma(a),N); return p[0::2].copy()
nrel=Nrel_int(s[0]).astype(np.float64)
# residual F(A,B) = [E0 - (AAb+BBb), O0 - (ABb+BAb), N - (A^2 - yB^2)]  (3H equations)
def resid(A,B):
    r1=E0-(ymul(A,yconj(A))+ymul(B,yconj(B)))
    r2=O0-(ymul(A,yconj(B))+ymul(B,yconj(A)))
    r3=nrel-(ymul(A,A)-ytimes(ymul(B,B)))
    return np.concatenate([r1,r2,r3])
# Jacobian by finite differences (3H x 2H) -- big but ok (384x256)
def jac_fd(A,B,h=1e-7):
    J=np.zeros((3*H,2*H))
    for i in range(H):
        Ap=A.copy(); Ap[i]+=h
        J[:,i]=(resid(Ap,B)-resid(A,B))/h
        Bp=B.copy(); Bp[i]+=h
        J[:,H+i]=(resid(A,Bp)-resid(A,B))/h
    return J
rng=np.random.default_rng(3)
# warm start: corrupt truth to ~46% correct? simulate attacker's estimate quality
def corrupt(t,frac_keep,rng):
    keep=rng.random(H)<frac_keep
    out=t.copy()
    out[~keep]=rng.choice([-2,-1,0,1,2],size=(~keep).sum())
    return out
for frac in (1.0,0.8,0.6,0.46):
    A0=corrupt(A,frac,rng); B0=corrupt(B,frac,rng)
    X=np.concatenate([A0,B0])
    errs=[]
    for it in range(30):
        Ac=X[:H]; Bc=X[H:]
        r=resid(Ac,Bc)
        J=jac_fd(Ac,Bc)
        dx=np.linalg.lstsq(J,-r,rcond=None)[0]
        X=X+dx
        errs.append(np.linalg.norm(resid(X[:H],X[H:])))
    Ac=X[:H]; Bc=X[H:]
    Ar=np.round(Ac); Br=np.round(Bc)
    ok=(Ar==s[0][0::2]).sum()+(Br==s[0][1::2]).sum()
    print('warmkeep=%.2f: final resid=%.3e  correct=%d/256  maxdev=%.3f'%(frac,errs[-1],ok,np.abs(np.concatenate([Ac-Ar,Bc-Br])).max()),flush=True)
