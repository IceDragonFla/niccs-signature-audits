import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
NSIG=200000; REC=4096+32
# ground truth of 200k key: s/ep blocks after pk+skraw
kb=np.fromfile('shipc_key.bin',dtype=np.uint8)
print('key file len',len(kb))
s200=np.frombuffer(kb[3552:3552+3*1024],dtype='<i4').reshape(3,N).astype(np.int64)
print('blocks parsed: density',(s200!=0).mean(),'vals',sorted(set(s200.ravel().tolist())))
G200=np.load('Gtrue.npy')
def cross(x,y):
    fx=np.fft.fft(x*tw); fy=np.fft.fft(y*tw)
    return np.real(np.fft.ifft(fx*np.conj(fy))*256*np.conj(tw))
ok=all(np.abs(cross(s200[k],s200[k])/256.0-G200[k]).max()<0.5 for k in range(3))
print('Gtrue matches parsed blocks (scaled):',ok)
# stream signatures
z=np.memmap('shipc_z1.bin',dtype='<i4',mode='r',shape=(NSIG,REC//4))
Aall=np.zeros((NSIG,N),dtype=np.float32)
Ghi=np.zeros((3,3,N),dtype=complex); Glo=np.zeros((3,3,N),dtype=complex)
zhi=np.zeros(N); zlo=np.zeros(N)
CH=5000
Apow=[]
for st in range(0,NSIG,CH):
    en=min(st+CH,NSIG)
    blk=z[st:en]
    zz=blk[:,:1024].reshape(-1,4,N)[:,1:,:].astype(np.float64)   # blocks 1..3
    cb=blk[:,1024:].copy()
    ind=np.zeros((en-st,N))
    for r in range(en-st):
        bits=np.unpackbits(np.frombuffer(cb[r].tobytes(),dtype=np.uint8))
        ind[r]=bits[:N]
    F=np.fft.fft(ind,axis=1)
    A=np.abs(F)**2
    Aall[st:en]=A.astype(np.float32)
    zr=zz*np.exp(1j*np.pi*np.arange(N)/N)
    Zh=np.fft.fft(zr,axis=1)
    Apow.append((st,en))
    np.save('/dev/null',np.array([0])) if False else None
med=np.median(Aall,axis=0)
# second pass accumulate with strata
for st in range(0,NSIG,CH):
    en=min(st+CH,NSIG)
    blk=z[st:en]
    zz=blk[:,:1024].reshape(-1,4,N)[:,1:,:].astype(np.float64)
    zr=zz*np.exp(1j*np.pi*np.arange(N)/N)
    Zh=np.fft.fft(zr,axis=1)
    hi=Aall[st:en]>med[None,:]
    for k in range(3):
        for kp in range(k,3):
            prod=Zh[:,k]*np.conj(Zh[:,kp])
            Ghi[k,kp]+=np.sum(prod*hi,axis=0)
            Glo[k,kp]+=np.sum(prod*(~hi),axis=0)
nhi=(Aall>med[None,:]).sum(axis=0); nlo=NSIG-nhi
np.save('strata_Ghi.npy',Ghi/nhi[:,None].T)
np.save('strata_Glo.npy',Glo/nlo[:,None].T)
np.save('strata_med.npy',med)
print('strata accumulators saved; nhi mean',nhi.mean())
dA=(Aall>med[None,:]).astype(np.float64)
dAm=np.save('strata_dA_mean.npy',np.array([( (Aall[:,m]>med[m]).astype(np.float64)*0 ).mean() for m in range(0)])) if False else None
print('done')
