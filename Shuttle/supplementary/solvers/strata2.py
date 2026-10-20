import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
kb=np.fromfile('shipc_key.bin',dtype=np.uint8)
s200=np.frombuffer(kb[9696:9696+3*1024],dtype='<i4').reshape(3,N).astype(np.int64)
print('s/ep blocks: density %.3f vals %s'%((s200!=0).mean(),sorted(set(s200.ravel().tolist()))))
Gt=np.array([[np.fft.fft(s200[k]*tw)*np.conj(np.fft.fft(s200[kp]*tw)) for kp in range(3)] for k in range(3)])
Ghi=np.load('strata_Ghi.npy'); Glo=np.load('strata_Glo.npy')
dG=Ghi-Glo
print('dG rms per pair:',[float(np.sqrt(np.mean(np.abs(dG[k,kp])**2))) for k in range(3) for kp in range(k+1,3)])
# kappa(m) per pair: dG/Gt
print('pair  corr(dG, Gt)   kappa_rms   kappa_consistency')
kaps=[]
for k in range(3):
    for kp in range(k+1,3):
        a=dG[k,kp]; b=Gt[k,kp]
        cc=np.corrcoef(np.r_[a.real,a.imag],np.r_[b.real,b.imag])[0,1]
        kap=(a*np.conj(b)).sum()/(np.abs(b)**2).sum()   # complex scalar
        kaps.append(kap)
        print(' %d%d   %.3f  %+.3f%+.3fi'%(k,kp,cc,kap.real,kap.imag))
print('kappa values close to each other?',np.abs(np.array(kaps)-np.mean(kaps)).max())
# is dG explained by challenge-modulation difference? predicted: dG ~ kappa*Gt*(dA(m)) with dA(m)=E[A|hi]-E[A|lo]... need A stats
