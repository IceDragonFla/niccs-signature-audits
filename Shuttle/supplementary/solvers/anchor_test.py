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
Wv=np.zeros((3,N),dtype=complex); lam=np.zeros(N)
for m in range(N):
    G=(M[:,:,m]+M[:,:,m].conj().T)/2
    l,v=np.linalg.eigh(G); Wv[:,m]=v[:,-1]; lam[m]=l[-1]
for m in range(N):
    mp=perm[m]
    if m<mp:
        a=0.5*(Wv[:,m]+np.conj(Wv[:,mp])); Wv[:,m]=a; Wv[:,mp]=np.conj(a)
V=Wv*np.sqrt(np.maximum(lam,0)/62.95)
ut=np.load('ut_true.npy')
Sr=np.load('Sr_trueu.npy')
good=Sr==s_true
print('correct positions: %d/768'%good.sum())
def theta_from_partial(mask):
    # partial spectrum: only masked coefficients contribute
    part=np.zeros((3,N))
    part[mask.reshape(3,N)]=Sr.reshape(3,N)[mask.reshape(3,N)]
    SHp=np.array([to_spec(part[k]) for k in range(3)])
    th=np.angle(sum(SHp[k]*np.conj(V[k]) for k in range(3)))
    return th
def ang_err(th):
    return np.degrees(np.median(np.abs(np.angle(np.exp(1j*(th-ut))))))
th_good=theta_from_partial(good)
th_all=theta_from_partial(np.ones_like(good))
rng=np.random.default_rng(2)
msk=np.zeros_like(good)
sel=rng.permutation(768)[:good.sum()]
m2=msk.ravel(); m2[sel]=True; msk=m2.reshape(3,N)
th_rand=theta_from_partial(msk&False | (msk*0).astype(bool))
# control: random positions of TRUE s
part=np.zeros((3,N)); fl=s_true.ravel().copy(); fl[sel]=0; part=fl.reshape(3,N)
SHp=np.array([to_spec(part[k]) for k in range(3)])
th_ctrl=np.angle(sum(SHp[k]*np.conj(V[k]) for k in range(3)))
print('theta err: good-only %.1f deg | all-pos %.1f deg | true-random-subset %.1f deg'%(ang_err(th_good),ang_err(th_all),ang_err(th_ctrl)))
# reconstruct with th_good (using bad positions as zeros) then full round
S=np.array([from_spec(np.exp(1j*th_good)*V[k]) for k in range(3)])
Srg=np.clip(np.round(S),-9,9).astype(np.int64)
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
print('recon from good-anchor theta: overlap %d/768 slack %.3f'%(fam_overlap(Srg,s_true),np.abs(S-Srg).mean()))
