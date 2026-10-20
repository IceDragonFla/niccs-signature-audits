import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
kb=np.fromfile('shipc_key.bin',dtype=np.uint8)
s200=np.frombuffer(kb[6624:6624+3*1024],dtype='<i4').reshape(3,N).astype(np.int64)
print('blocks@6624: density %.3f vals %s'%((s200!=0).mean(),sorted(set(s200.ravel().tolist()))))
Gt=np.array([[np.fft.fft(s200[k]*tw)*np.conj(np.fft.fft(s200[kp]*tw)) for kp in range(3)] for k in range(3)])
Ghi=np.load('strata_Ghi.npy'); Glo=np.load('strata_Glo.npy')
dG=Ghi-Glo
bad=~np.isfinite(dG[0,1])
print('non-finite bins:',int(bad.sum()),'freqs:',np.where(bad)[0][:10])
dG=np.where(np.isfinite(dG),dG,0)
print('dG rms per pair:',[round(float(np.sqrt(np.mean(np.abs(dG[k,kp])**2))),1) for k in range(3) for kp in range(k+1,3)])
print('pair  corr(dG,Gt)   kappa')
kaps=[]
for k in range(3):
    for kp in range(k+1,3):
        a=dG[k,kp]; b=Gt[k,kp]
        cc=np.corrcoef(np.r_[a.real,a.imag],np.r_[b.real,b.imag])[0,1]
        kap=(a*np.conj(b)).sum()/(np.abs(b)**2).sum()
        kaps.append(kap)
        print(' %d%d   %.4f   %+.4f%+.4fi'%(k,kp,cc,kap.real,kap.imag))
print('kappa spread:',np.abs(np.array(kaps)-np.mean(kaps)).max())
# per-frequency kappa consistency: for each m, fit kappa_m from 3 pairs jointly, check residual
resid_after=np.zeros(3)
for k in range(3):
    for kp in range(k+1,3):
        kap_m=(dG[k,kp]*np.conj(Gt[k,kp]))/np.maximum(np.abs(Gt[k,kp])**2,1e-9)
        resid_after[k*3+0 if False else 0]
kap_m=(sum(dG[k,kp]*np.conj(Gt[k,kp]) for k in range(3) for kp in range(k+1,3)))/sum(np.maximum(np.abs(Gt[k,kp])**2,1e-9) for k in range(3) for kp in range(k+1,3))
r_after=sum(np.sum(np.abs(dG[k,kp]-kap_m*Gt[k,kp])**2) for k in range(3) for kp in range(k+1,3))
r_before=sum(np.sum(np.abs(dG[k,kp])**2) for k in range(3) for kp in range(k+1,3))
print('shared per-freq kappa explains %.1f%% of dG energy'%(100*(1-r_after/r_before)))
np.save('kap_m.npy',kap_m)
