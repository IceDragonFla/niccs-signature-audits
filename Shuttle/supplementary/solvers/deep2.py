import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N); perm=(1-idx)%N
s_true=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
d=open('corr1m','rb').read()
NSIG,=np.frombuffer(d,dtype=np.int64,count=1,offset=0)
Pre=np.frombuffer(d,dtype=np.float64,count=9*N,offset=8).reshape(3,3,N)
Pim=np.frombuffer(d,dtype=np.float64,count=9*N,offset=8+9*N*8).reshape(3,3,N)
P=(Pre+1j*Pim)/NSIG
bsum=np.frombuffer(d,dtype=np.float64,count=3,offset=8+18*N*8)
bsq=np.frombuffer(d,dtype=np.float64,count=3,offset=8+18*N*8+24)
var=bsq/(NSIG*N)-(bsum/(NSIG*N))**2
M=P.copy()
for k in range(3): M[k,k]-=N*var[k]
to_spec=lambda x: np.fft.fft(np.asarray(x,float)*tw)
from_spec=lambda sh: np.real(np.conj(tw)*np.fft.ifft(sh))
C3=62.95
xpairs=[(0,1),(0,2),(1,2)]
def xcost(SH):
    return float(sum(np.sum(np.abs(M[k,kp]-C3*SH[k]*np.conj(SH[kp]))**2) for k,kp in xpairs))
def full_cost(S):
    SH=np.array([to_spec(S[k]) for k in range(3)])
    return xcost(SH)
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
def fam_overlap(S,T):
    b=0
    for eps in (1,-1):
        for dd in range(N):
            b=max(b,int(sum((S[k]==eps*negshift(T[k],dd)).sum() for k in range(3))))
    return b
base=np.exp(-2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)
BASIS=[tw[i]*base[:,i] for i in range(N)]
# V with moduli (for theta-jump)
Wv=np.zeros((3,N),dtype=complex); lam=np.zeros(N)
for m in range(N):
    G=(M[:,:,m]+M[:,:,m].conj().T)/2
    l,v=np.linalg.eigh(G); Wv[:,m]=v[:,-1]; lam[m]=l[-1]
for m in range(N):
    mp=perm[m]
    if m<mp:
        a=0.5*(Wv[:,m]+np.conj(Wv[:,mp])); Wv[:,m]=a; Wv[:,mp]=np.conj(a)
V=Wv*np.sqrt(np.maximum(lam,0)/C3)
def theta_jump(S):
    SH=np.array([to_spec(S[k]) for k in range(3)])
    th=np.angle(sum(SH[k]*np.conj(V[k]) for k in range(3)))
    S2=np.clip(np.round(np.array([from_spec(np.exp(1j*th)*V[k]) for k in range(3)])),-9,9).astype(np.int64)
    return S2
def cd(S,sweeps=30):
    SH=np.array([to_spec(S[k]) for k in range(3)])
    for sw in range(sweeps):
        improved=False
        for t in range(768):
            k,i=divmod(t,N)
            bestv=S[k][i]; bestc=xcost(SH)
            for dv in (-2,-1,0,1,2):
                if dv==S[k][i]: continue
                SHk=SH[k]+(dv-S[k][i])*BASIS[i]
                c=0.0
                for (a,b) in xpairs:
                    Ga=SHk if a==k else SH[a]
                    Gb=SHk if b==k else SH[b]
                    c+=float(np.sum(np.abs(M[a,b]-C3*Ga*np.conj(Gb))**2))
                if c<bestc-1e-3: bestc=c; bestv=dv
            if bestv!=S[k][i]:
                S[k][i]=bestv; SH[k]=to_spec(S[k]); improved=True
        if not improved: break
    return S
S=np.load('Sr_thetaopt.npy')
print('start cost %.4e ov %d'%(full_cost(S),fam_overlap(S,s_true)))
cur=full_cost(S)
for it in range(8):
    S=cd(S.copy())
    c1=full_cost(S)
    S2=theta_jump(S)
    c2=full_cost(S2)
    if c2<c1: S,c1=S2,c2
    print('iter %d: cd %.4e -> jump %s %.4e  ov %d'%(it,c1,'taken' if c2<c1 else 'no',c2,fam_overlap(S,s_true)))
    if c1>=cur-1e-6 and it>0: 
        # SA kick
        rng=np.random.default_rng(it)
        S3=S.copy()
        for _ in range(6):
            k=rng.integers(3); i=rng.integers(N); S3[k][i]+=int(rng.choice([-1,1]))
        S3=cd(S3)
        c3v=full_cost(S3)
        if c3v<c1: S,c1=S3,c3v; print('  SA improved %.4e'%c1)
    cur=c1
np.save('Sr_deep2.npy',S)
print('FINAL cost %.4e ov %d/768'%(full_cost(S),fam_overlap(S,s_true)))
