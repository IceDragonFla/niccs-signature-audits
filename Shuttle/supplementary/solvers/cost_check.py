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
def gram(S):
    SH=[to_spec(S[k]) for k in range(3)]
    return np.array([[SH[k]*np.conj(SH[kp]) for kp in range(3)] for k in range(3)])
def cross_cost(S):
    c3=62.95
    G=gram(S)
    return float(np.mean([np.mean(np.abs(M[k,kp]-c3*G[k,kp])**2) for k in range(3) for kp in range(k+1,3)]))
print('cost(s_true)     = %.4e'%cross_cost(s_true))
for f in ['S_final.npy','Sr_trueu.npy','Sr_thetaopt.npy','Sr_gsmod.npy','Sr_apclean.npy']:
    try:
        S=np.load(f)
        # also try best negshift/sign alignment of S to truth for FAIR cost? cost is invariant to negshift family
        print('cost(%s) = %.4e'%(f,cross_cost(S)))
    except Exception as e: print(f,'skip',e)
rng=np.random.default_rng(9)
for t in range(3):
    S=rng.choice([-2,-1,0,1,2],size=(3,N),p=[.07,.16,.54,.16,.07]).astype(np.int64)
    print('cost(random%d)   = %.4e'%(t,cross_cost(S)))
