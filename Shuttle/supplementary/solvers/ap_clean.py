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
# rank-1 projected target (kills orthogonal systematic)
W=np.zeros((3,N),dtype=complex); L1=np.zeros(N)
for m in range(N):
    G=(M[:,:,m]+M[:,:,m].conj().T)/2
    lam,v=np.linalg.eigh(G)
    W[:,m]=v[:,-1]; L1[m]=lam[-1]
# normalize: enforce correct pairing m <-> (1-m) mod N
for m in range(N):
    mp=(1-m)%N
    if m<mp:
        a=W[:,m]; b=W[:,mp]
        # choose b = conj(a)*phase to make both consistent: average
        ph=np.angle(np.vdot(a,np.conj(b)))
        W[:,mp]=np.conj(a)*np.exp(1j*ph)*np.sqrt(abs(L1[mp])/max(L1[m],1e-12))
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
def overlap(S):
    return max(int(sum((S[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))
Sf=np.load('S_final.npy')
print('S_final overlap: %d/768'%overlap(Sf))
def run_AP(th,iters=500):
    for it in range(iters):
        S=np.clip(np.round(np.array([from_spec(np.exp(1j*th)*W[k]) for k in range(3)])),-9,9)
        acc=sum(to_spec(S[k])*np.conj(W[k]) for k in range(3))
        th=np.angle(acc)
        th=0.5*(th-th[(1-idx)%N])   # correct pairing antisymmetrization
    return th,S
rng=np.random.default_rng(7)
best=(1e9,None,0)
# anchor from partial solution
th0=np.angle(sum(to_spec(Sf[k])*np.conj(W[k]) for k in range(3)))
th0=0.5*(th0-th0[(1-idx)%N])
anchors=[('partial',th0)]+[('rand%d'%r,0.5*(lambda t:t-t[(1-idx)%N])(rng.uniform(0,2*np.pi,N))) for r in range(12)]
for tag,th in anchors:
    thf,S=run_AP(th.copy())
    S=np.clip(np.round(np.array([from_spec(np.exp(1j*thf)*W[k]) for k in range(3)])),-9,9).astype(np.int64)
    ov=overlap(S)
    if -ov<-best[2]: best=(-ov,S,ov)
    print('%8s -> overlap %d/768'%(tag,ov))
print('AP best: %d/768'%best[2])
np.save('Sr_apclean.npy',best[1])
