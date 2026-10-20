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
C3=62.95
xpairs=[(0,1),(0,2),(1,2)]
def xcost(S):
    SH=np.array([to_spec(S[k]) for k in range(3)])
    return float(sum(np.sum(np.abs(M[k,kp]-C3*SH[k]*np.conj(SH[kp]))**2) for k,kp in xpairs))
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
PHI=np.exp(-2j*np.pi*np.outer(idx,idx)/N)*tw[None,:]  # PHI[m,i]=tw_i e^{-2pi i m i/N}
def build_system(S0):
    SH=np.array([to_spec(S0[k]) for k in range(3)])
    rows=[]; rhs=[]
    for k,kp in xpairs:
        R=M[k,kp]-C3*SH[k]*np.conj(SH[kp])
        # d R/d delta_k(i) = C3 * PHI[:,i] * conj(SH_kp) ; d/d delta_kp(i) = C3 * conj(PHI[:,i]*SH_k)
        Ak=C3*np.conj(SH[kp])[:,None]*PHI     # (m,i) coeff of delta_k
        Akp=C3*np.conj(PHI*SH[k][:,None])     # conj over m and i
        Ap=np.zeros((512,768))
        Ap[:,k*256:(k+1)*256]=np.vstack([Ak.real,Ak.imag])
        Ap[:,kp*256:(kp+1)*256]+=np.vstack([Akp.real,Akp.imag])
        rows.append(Ap)
        rhs.append(np.r_[R.real,R.imag])
    A=np.vstack(rows); b=np.concatenate(rhs)
    return A,b
for tag,f in [('deep2','Sr_deep2.npy'),('thetaopt','Sr_thetaopt.npy'),('gsmod','Sr_gsmod.npy')]:
    try: S0=np.load(f).astype(np.int64)
    except: continue
    S=S0.copy()
    for it in range(8):
        A,b=build_system(S)
        delta,_,_,_=np.linalg.lstsq(A,b,rcond=None)
        dround=np.clip(np.round(delta),-5,5).astype(np.int64).ravel()
        step=0
        # line search over step scale to ensure cost decrease
        best=(xcost(S),S.copy(),True)
        for sc in (1.0,0.5,0.25):
            Sn=(S+(sc*dround).astype(np.int64).reshape(3,256))
            c=xcost(Sn)
            if c<best[0]: best=(c,Sn,False)
        c,S,stop=best
        print('%s it%d: |delta|_1=%d step_cost=%.4e ov=%d'%(tag,it,int(np.abs(dround).sum()),c,fam_overlap(S,s_true)))
        if stop: break
    np.save('Sr_gn_%s.npy'%tag,S)
