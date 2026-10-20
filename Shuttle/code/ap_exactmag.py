import numpy as np
N=256
def rmul_int(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj_int(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
gam00=rmul_int(s[0],rconj_int(s[0],N),N)
tw=np.exp(1j*np.pi*np.arange(N)/N)
# exact magnitudes: |sigma_j(s)|^2 = twisted-DFT of gamma00 (times 1/N scaling consistent)
G=np.fft.fft(gam00.astype(float)*tw)
mags=np.sqrt(np.abs(G))   # |sigma_j(s)| up to scale
print('mag scale check: ||s||=%.2f  sqrt(sum mags^2/N)=%.2f'%(np.linalg.norm(s[0].astype(float)),np.sqrt((mags**2).sum()/N)))
def to_spec(x): return np.fft.fft(x.astype(float)*tw)
def from_spec(sh): return np.real(np.conj(tw)*np.fft.ifft(sh))
def negshift(v,j):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
def fam(S):
    for eps in (1,-1):
        for dd in range(N):
            yield eps*negshift(s[0],dd)
FAM=list(fam(s[0]))
def score(x):
    return max(int((x==f).sum()) for f in FAM)
# AP with EXACT magnitudes + integrality
rng=np.random.default_rng(11)
S_final=np.load('S_final.npy')
def run_AP(th,iters=200):
    for it in range(iters):
        S=np.clip(np.round(from_spec(mags*np.exp(1j*th))),-9,9)
        acc=to_spec(S)
        thn=np.angle(acc)
        th=0.5*(thn-thn[(1-np.arange(N))%N])  # pairing antisym (realness)
    return S
best=(0,None)
inits=[('Sfinal',None)]
for r in range(8):
    t=rng.uniform(0,2*np.pi,N)
    inits.append(('rand%d'%r,0.5*(t-t[(1-np.arange(N))%N])))
for tag,th0 in inits:
    th=th0 if th0 is not None else None
    if th is None:
        sh=to_spec(S_final[0].astype(float))
        th=0.5*(np.angle(sh)-np.angle(sh)[(1-np.arange(N))%N])
    S=run_AP(th.copy())
    sc=score(S)
    dev=np.abs(from_spec(mags*np.exp(1j*to_spec(S)))-S).mean()
    print('%8s: overlap=%d/256  dev=%.3f'%(tag,sc,dev),flush=True)
    if sc>best[0]: best=(sc,S)
print('BEST',best[0])
np.save('ap_exactmag_best.npy',best[1])
