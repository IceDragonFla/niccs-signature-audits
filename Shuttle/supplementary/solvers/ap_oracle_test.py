import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
s_true=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
to_spec=lambda x: np.fft.fft(np.asarray(x,float)*tw)
from_spec=lambda sh: np.real(np.conj(tw)*np.fft.ifft(sh))
SH=np.array([to_spec(s_true[k]) for k in range(3)])
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
def overlap(S):
    return max(int(sum((S[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))
def run_AP(W,th,iters=400,perm=None):
    perm=perm if perm is not None else (1-idx)%N
    for it in range(iters):
        S=np.clip(np.round(np.array([from_spec(np.exp(1j*th)*W[k]) for k in range(3)])),-9,9)
        acc=sum(to_spec(S[k])*np.conj(W[k]) for k in range(3))
        th=np.angle(acc)
        th=0.5*(th-th[perm])
    return th,S
rng=np.random.default_rng(1)
for r in range(3):
    th0=rng.uniform(0,2*np.pi,N)
    th,S=run_AP(SH,th0.copy())
    S=np.clip(np.round(np.array([from_spec(np.exp(1j*th)*SH[k]) for k in range(3)])),-9,9).astype(np.int64)
    print('oracle W rand%d: overlap %d/768'%(r,overlap(S)))
# check SH pairing convention explicitly
m=17; mp=(1-m)%N
lhs=SH[:,mp]; rhs=np.conj(SH[:,m])
print('SH pairing (1-m): max|lhs-conj| =',np.abs(lhs-rhs).max(),'  |lhs|/|rhs| ratio',np.abs(lhs)/np.abs(rhs))
m=17; mp2=(N-1-m)%N
print('SH pairing (N-1-m): max diff =',np.abs(SH[:,mp2]-np.conj(SH[:,m])).max())
