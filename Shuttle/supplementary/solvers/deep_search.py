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
pairs=[(k,kp) for k in range(3) for kp in range(k,3)]
def gram_SH(S):
    return np.array([to_spec(S[k]) for k in range(3)])
def cross_cost(S,c3=None):
    SH=gram_SH(S)
    if c3 is None:
        num=sum(np.real(np.sum(M[k,kp]*np.conj(SH[k]*np.conj(SH[kp])))) for k,kp in pairs)
        den=sum(np.sum(np.abs(SH[k]*np.conj(SH[kp]))**2) for k,kp in pairs)
        c3=num/den
    return float(np.mean([np.mean(np.abs(M[k,kp]-c3*SH[k]*np.conj(SH[kp]))**2) for k,kp in pairs])),c3
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
base=np.exp(-2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)  # fft kernel e^{-2pi i m i /N}
def cd_sweeps(S,c3,maxsweep=40):
    SH=gram_SH(S)
    improved=True; sweep=0
    while improved and sweep<maxsweep:
        improved=False; sweep+=1
        order=np.random.default_rng(sweep).permutation(768)
        for t in order:
            k,i=divmod(t,N)
            bi=tw[i]*base[:,i]   # spectrum basis vector for unit change at (k,i)
            bestv=S[k][i]; bestc=None
            for dv in (-2,-1,0,1,2):
                if dv==S[k][i]: continue
                dSH=(dv-S[k][i])*bi
                SHk=SH[k]+dSH
                c=0.0
                for (a,b) in pairs:
                    if a==k and b==k: G=SHk*np.conj(SHk)
                    elif a==k: G=SHk*np.conj(SH[b])
                    elif b==k: G=SH[a]*np.conj(SHk)
                    else: continue
                    c+=float(np.sum(np.abs(M[a,b]-c3*G)**2))
                if bestc is None or c<bestc-1e-6: bestc=c; bestv=dv
            if bestv!=S[k][i]:
                S[k][i]=bestv; SH[k]=to_spec(S[k]); improved=True
    return S,c3
def family_moves(S):
    outs=[]
    for eps in (1,-1):
        for dd in (0,1,7,63,97,128,191,255):
            outs.append(np.array([eps*negshift(S[k],dd) for k in range(3)]))
    return outs
rng=np.random.default_rng(2026)
cands=[np.load('Sr_thetaopt.npy')]
try: cands.append(np.load('S_final.npy'))
except: pass
bestS=None; bestc=1e18
for S0 in cands:
    S=S0.copy()
    c,c3=cross_cost(S)
    for rnd in range(6):
        S,c3=cd_sweeps(S,c3)
        c,c3=cross_cost(S)
        # theta realign jump
        Vm=np.array([to_spec(S[k]) for k in range(3)])
        num=sum(np.real(np.sum(M[k,kp]*np.conj(Vm[k]*np.conj(Vm[kp])))) for k,kp in pairs)
        den=sum(np.sum(np.abs(Vm[k]*np.conj(Vm[kp]))**2) for k,kp in pairs)
        c3=num/den
        if c<bestc: bestc=c; bestS=S.copy(); print('  new best cost %.4e ov %d'%(c,fam_overlap(S,s_true)))
        # SA kick: random 3-coordinate perturb
        for kick in range(200):
            S2=S.copy()
            for _ in range(3):
                k=rng.integers(3); i=rng.integers(N); S2[k][i]=int(rng.choice([-1,0,1]))
            c2,c32=cross_cost(S2,c3)
            if c2<c: S=S2; break
print('FINAL best cost %.4e overlap %d/768'%(bestc,fam_overlap(bestS,s_true)))
np.save('Sr_deepsearch.npy',bestS)
