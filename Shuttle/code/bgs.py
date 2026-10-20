import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
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
def xcost(S):
    SH=np.array([to_spec(S[k]) for k in range(3)])
    return float(sum(np.sum(np.abs(M[k,kp]-C3*SH[k]*np.conj(SH[kp]))**2) for k in range(3) for kp in range(k+1,3)))
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
def bgs(S,iters=60):
    S=S.copy()
    for it in range(iters):
        moved=False
        for k in range(3):
            oth=[j for j in range(3) if j!=k]
            # LS: SH_k(m) = C3 * sum_{k'!=k} M_{k,k'} conj-dagged... derive:
            # minimize sum_{k'} |M_kk' - C3 SH_k conj(SH_k')|^2
            # d/d SH_k^*: -C3 sum_k' (M_kk' - C3 SH_k conj(SH_k')) SH_k' = 0
            # SH_k = (sum_k' M_kk' SH_k') / (C3 sum_k' |SH_k'|^2)
            SH=np.array([to_spec(S[j]) for j in range(3)])
            num=M[k,oth[0]]*SH[oth[0]]+M[k,oth[1]]*SH[oth[1]]
            den=C3*(np.abs(SH[oth[0]])**2+np.abs(SH[oth[1]])**2)
            SHk=num/np.maximum(den,1e-9)
            srec=from_spec(SHk)
            Snew=np.clip(np.round(srec),-9,9).astype(np.int64)
            if not np.array_equal(Snew,S[k]): moved=True
            S[k]=Snew
        if not moved: break
    return S
rng=np.random.default_rng(42)
inits=[np.load('Sr_thetaopt.npy'),np.load('S_final.npy')]
for t in range(4):
    inits.append(rng.choice([-2,-1,0,1,2],size=(3,N),p=[.07,.16,.54,.16,.07]).astype(np.int64))
best=(1e18,None)
for ii,S0 in enumerate(inits):
    S=bgs(S0)
    c=xcost(S)
    ov=fam_overlap(S,s_true)
    tag='warm%d'%ii if ii<2 else 'rand%d'%ii
    print('%s: cost %.4e ov %d/768'%(tag,c,ov))
    if c<best[0]: best=(c,S)
print('BEST cost %.4e ov %d'%(best[0],fam_overlap(best[1],s_true)))
np.save('Sr_bgs.npy',best[1])
