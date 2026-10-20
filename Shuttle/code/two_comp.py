import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
s_true=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
Sr=np.load('Sr_trueu.npy'); sbar=np.load('sbar_true.npy')
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
def gram(S):
    SH=[to_spec(S[k]) for k in range(3)]
    return np.array([[SH[k]*np.conj(SH[kp]) for kp in range(3)] for k in range(3)])
Gt=gram(s_true); Gb=gram(sbar); Gr=gram(Sr)
def fitres(basis):
    Bm=np.stack([b.ravel() for b in basis],axis=1)  # (2304, nb) complex
    y=M.ravel()
    co=np.linalg.lstsq(Bm,y,rcond=None)[0]
    r=y-Bm@co
    rms=np.sqrt(np.mean([np.mean(np.abs(r.reshape(3,3,N)[k,kp])**2) for k in range(3) for kp in range(k+1,3)]))
    return co,rms
for name,basis in [('Gt only',[Gt]),('Gt+Gb',[Gt,Gb]),('Gt+Gr',[Gt,Gr]),('Gt+Gb+Gr',[Gt,Gb,Gr])]:
    co,rms=fitres(basis)
    print('%10s: coefs %s  cross-resid rms=%.1f'%(name,np.round(co,2),rms))
print('noise level ~1758')
