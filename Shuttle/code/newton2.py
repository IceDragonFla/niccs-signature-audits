import numpy as np
N=256; H=128
def rmul_int(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj_int(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
def ymul(a,b):
    c=np.convolve(a,b); out=c[:H].copy(); out[:len(c)-H]-=c[H:]; return out
def ytimes(P):
    o=np.zeros(H); o[1:]=P[:H-1]; o[0]=-P[H-1]
    return o
def yconj(p):
    o=np.zeros(H); o[0]=p[0]; o[1:]=-p[H:0:-1]; return o
# exact integer constants
gam00=rmul_int(s[0],rconj_int(s[0],N),N)
E0=gam00[0::2].astype(np.float64)
O0=gam00[1::2].astype(np.float64)
def sigma(a): return a*((-1)**np.arange(N))
nrel=rmul_int(s[0],sigma(s[0]),N)[0::2].astype(np.float64)
# structure tensors for analytic Jacobian: multiplication by p in R'
def build_mult(p):
    M=np.zeros((H,H))
    for j in range(H):
        e=np.zeros(H); e[j]=1.0
        M[:,j]=ymul(e,p)
    return M
# residual
def yinv(P):
    Q=np.roll(P,-1).astype(P.dtype).copy()
    Q[H-1]=-P[0]
    return Q
def resid(A,B):
    r1=E0-(ymul(A,yconj(A))+ymul(B,yconj(B)))
    r2=O0-(ymul(B,yconj(A))+yinv(ymul(A,yconj(B))))
    r3=nrel-(ymul(A,A)-ytimes(ymul(B,B)))
    return np.concatenate([r1,r2,r3])
# analytic Jacobian blocks via multiplication matrices
def multA(A): return build_mult(A)
def multAc(A): return build_mult(yconj(A))
def multB(B): return build_mult(B)
def multBc(B): return build_mult(yconj(B))
def multYt(P): return build_mult(ytimes(P))
def jac_an(A,B):
    # r1: d/dA = -(M(Ac) + M(Ac)^T?) careful: ymul(A, yconjA) linear in A: M(yconjA) A; also yconj(A) depends on A linearly (involution linear!) => d r1/dA = -(M_{yconj A} + M'_{A via yconj}) both linear maps:
    # r1 = E0 - (A*Ac + B*Bc); d/dA(delta) = -(delta*Ac + A*yconj(delta)) = -(M_Ac + T(M_A^T)) where T is involution-map
    # implement involution as linear map: yconj(v) = T v; and ymul(a, yconj(delta)) = M_a @ T @ delta
    T=np.zeros((H,H))
    for j in range(H):
        e=np.zeros(H); e[j]=1.0
        T[:,j]=yconj(e)
    J=np.zeros((3*H,2*H))
    MA=build_mult(A); MAc=build_mult(yconj(A))
    MB=build_mult(B); MBc=build_mult(yconj(B))
    J11=-(MAc + MA@T)          # d r1 / dA
    J12=-(MBc + MB@T)          # d r1 / dB
    Yinv=np.zeros((H,H))
    for j in range(H):
        e=np.zeros(H); e[j]=1.0
        Yinv[:,j]=yinv(e)
    J21=-(MBc + Yinv@MBc)      # d r2/dA = -(Bc*delta + yinv(Ac*delta)) = -(MBc + Yinv@MBc)
    J22=-(MA@T + Yinv@MA)      # d r2/dB = -(delta*Ac + yinv(delta*Bc)) = -(MAc?...) fixed
    J31=-(MA + build_mult(ytimes(np.zeros(H))) ) # d r3/dA = -(2 A * delta) => -2 MA ; plus? r3 = nrel - (A*A - yB*B)
    J31=-2*MA
    J32=2*build_mult(ytimes(B)) # d r3/dB = +2*yB*delta => ytimes(B)*delta => M_{ytimes(B)}
    # but ytimes(B)*delta: ymul(ytimes(B), delta) => M_{ytimes(B)}
    J[:H,:H]=J11; J[:H,H:]=J12
    J[H:2*H,:H]=J21; J[H:2*H,H:]=J22
    J[2*H:,:H]=J31; J[2*H:,H:]=J32
    return J
rng=np.random.default_rng(3)
def corrupt(t,frac_keep,rng):
    keep=rng.random(H)<frac_keep
    out=t.copy()
    out[~keep]=rng.choice([-2,-1,0,1,2],size=(~keep).sum())
    return out
At=s[0][0::2].astype(np.float64); Bt=s[0][1::2].astype(np.float64)
print('sanity resid at truth: %.3e'%np.linalg.norm(resid(At,Bt)))
for frac in (1.0,0.9,0.8,0.7,0.6,0.46):
    A0=corrupt(At,frac,rng); B0=corrupt(Bt,frac,rng)
    X=np.concatenate([A0,B0]).copy()
    lam=0.0
    ok_all=0
    for it in range(60):
        r=resid(X[:H],X[H:])
        J=jac_an(X[:H],X[H:])
        try:
            dx=np.linalg.lstsq(J,-r,rcond=None)[0]
        except Exception:
            break
        # damped step
        for step in (1.0,0.5,0.25,0.1):
            Xn=X+step*dx
            if np.linalg.norm(resid(Xn[:H],Xn[H:]))<np.linalg.norm(r):
                break
        X=Xn
    Ac=X[:H]; Bc=X[H:]
    Ar=np.round(Ac); Br=np.round(Bc)
    ok=int((Ar==s[0][0::2]).sum()+(Br==s[0][1::2]).sum())
    maxdev=float(np.abs(np.concatenate([Ac-Ar,Bc-Br])).max())
    print('warm=%.2f: resid=%.3e correct=%d/256 maxdev=%.3f'%(frac,np.linalg.norm(resid(Ac,Bc)),ok,maxdev),flush=True)
