import numpy as np
from scipy.optimize import minimize
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
W=np.zeros((3,N),dtype=complex); lam=np.zeros(N)
for m in range(N):
    G=(M[:,:,m]+M[:,:,m].conj().T)/2
    l,v=np.linalg.eigh(G); W[:,m]=v[:,-1]; lam[m]=l[-1]
for m in range(N):
    mp=perm[m]
    if m<mp:
        a=0.5*(W[:,m]+np.conj(W[:,mp])); W[:,m]=a; W[:,mp]=np.conj(a)
V=W*np.sqrt(np.maximum(lam,0)/62.95)
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
def fam_overlap(S,T):
    best=0
    for eps in (1,-1):
        for dd in range(N):
            best=max(best,int(sum((S[k]==eps*negshift(T[k],dd)).sum() for k in range(len(S)))))
    return best
half=np.array([m for m in range(N) if m<perm[m]]); nh=len(half)
def theta_full(ph):
    th=np.zeros(N)
    th[half]=ph; th[perm[half]]=-ph
    return th
def recon(ph):
    th=theta_full(ph)
    return np.array([from_spec(np.exp(1j*th)*V[k]) for k in range(3)])
def J(ph):
    S=recon(ph)
    return float(np.sum((S-np.round(S))**2))
def grad_ph(ph,eps=1e-6):
    g=np.zeros(nh)
    S=recon(ph)
    # d from_spec(e^{i th} V)/d th_m : imag part of (i e^{i th}V) ifft...
    for j in range(nh):
        pass
    return g
# analytic gradient: s_ki = Re[ conj(tw_i) (1/N) sum_m e^{i th_m} V_k(m) e^{2pi i m i/N} ]
# d/d th_j = Re[ conj(tw_i) (1/N) * ( i e^{i th_j} V_k(j) e^{2pi i j i/N} ) ] (j in half) ; for m=perm[j]: -that
E=np.exp(2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)  # E[i,m]
Vj=np.array([[V[k][j] for j in range(N)] for k in range(3)])
def Jgrad(ph):
    th=theta_full(ph)
    ev=np.exp(1j*th)
    g=np.zeros(nh); Jv=0.0
    for k in range(3):
        H=(V[k][None,:]*E)*ev[None,:]
        s=np.real(np.conj(tw)[:,None]*H).sum(axis=1)/N
        r=s-np.round(s); Jv+=float((r*r).sum())
        B=np.real(np.conj(tw)[:,None]*(1j*H))/N
        g+=2*r@B[:,half]-2*r@B[:,perm[half]]
    return Jv,g

# verify gradient numerically
ph0=np.random.default_rng(3).uniform(-0.3,0.3,nh)
Jv,g=Jgrad(ph0)
ph1=ph0.copy(); ph1[5]+=1e-6
print('grad check: analytic %.6f numeric %.6f'%(g[5],(J(ph1)-Jv)/1e-6))
# optimize from several anchors
anchors={}
Sf=np.load('S_final.npy')
anchors['Sfinal']=np.angle(sum(to_spec(Sf[k])*np.conj(V[k]) for k in range(3)))
Sr=np.load('Sr_trueu.npy')
anchors['Sr']=np.angle(sum(to_spec(Sr[k])*np.conj(V[k]) for k in range(3)))
ut=np.load('ut_true.npy'); anchors['true']=ut
rng=np.random.default_rng(5)
for r in range(4):
    t=rng.uniform(0,2*np.pi,N); anchors['rand%d'%r]=0.5*(t-t[perm])
def ph_of(th_full):
    return 0.5*(th_full[half]-th_full[perm[half]])
best=(1e18,None,None)
for tag,thf in anchors.items():
    ph=ph_of(thf)
    res=minimize(Jgrad,ph,method='L-BFGS-B',jac=True,options={'maxiter':4000,'maxfun':6000})
    ph2=res.x
    S=recon(ph2); Sr2=np.clip(np.round(S),-9,9).astype(np.int64)
    sl=np.abs(S-Sr2).mean()
    # per-block overlap
    po=[fam_overlap(Sr2[k:k+1],s_true[k:k+1]) for k in range(3)]
    print('%8s: J=%.4f slack=%.4f ov=%d/768 blocks=%s'%(tag,res.fun,sl,fam_overlap(Sr2,s_true),po))
    if res.fun<best[0]: best=(res.fun,Sr2.copy(),tag)
np.save('Sr_thetaopt.npy',best[1])
print('BEST',best[2],best[0])
