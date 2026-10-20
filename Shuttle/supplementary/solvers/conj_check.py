import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N); perm=(1-idx)%N
s_true=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
to_spec=lambda x: np.fft.fft(np.asarray(x,float)*tw)
from_spec=lambda sh: np.real(np.conj(tw)*np.fft.ifft(sh))
W=np.load('W_eig.npy'); ut=np.load('ut_true.npy')
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
St=np.array([from_spec(ut*W[k]) for k in range(3)])
Sr=np.clip(np.round(St),-9,9).astype(np.int64)
# how close to integer per coefficient?
dev=np.abs(St-Sr)
print('dev: mean %.4f  frac>0.3 %.4f  frac>0.5(should be 0) %.4f  max %.3f'%(dev.mean(),(dev>0.3).mean(),(dev>0.5).mean(),dev.max()))
# ring conjugate family
sbar=np.array([np.round(from_spec(np.conj(to_spec(s_true[k])))).astype(np.int64) for k in range(3)])
print('sbar integrality: max dev %.4f ; sbar==s_true? %s'%(np.abs(np.array([from_spec(np.conj(to_spec(s_true[k]))) for k in range(3)])-sbar).max(),all((sbar[k]==s_true[k]).all() for k in range(3))))
def fam_overlap(S,T):
    best=(0,None)
    for eps in (1,-1):
        for dd in range(N):
            ov=int(sum((S[k]==eps*negshift(T[k],dd)).sum() for k in range(3)))
            if ov>best[0]: best=(ov,(eps,dd))
    return best
print('Sr vs s_true :',fam_overlap(Sr,s_true))
print('Sr vs sbar   :',fam_overlap(Sr,sbar))
np.save('Sr_trueu.npy',Sr); np.save('sbar_true.npy',sbar)
