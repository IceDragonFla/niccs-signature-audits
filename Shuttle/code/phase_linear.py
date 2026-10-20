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
perm=(1-idx)%N
# enforce pairing on W exactly
W=np.zeros((3,N),dtype=complex)
for m in range(N):
    G=(M[:,:,m]+M[:,:,m].conj().T)/2
    lam,v=np.linalg.eigh(G); W[:,m]=v[:,-1]
for m in range(N):
    mp=perm[m]
    if m<mp:
        a=0.5*(W[:,m]+np.conj(W[:,mp]))
        W[:,m]=a; W[:,mp]=np.conj(a)
# unknown u with pairing: u_m = a_m + i b_m, a_{mp}=a_m, b_{mp}=-b_m
# constraint per (k,i): Im[ conj(tw_i) * (1/N) sum_m u_m W_k(m) e^{2pi i m i/N} ] = 0
# = Im[ conj(tw_i) * h_ki ] with h linear in (a,b)

E=np.exp(2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)  # E[i,m]
rows=[]; rhs=[]
# build coefficient matrices: h_ki(u) = (1/N) sum_m (a_m + i b_m) W_k(m) E[i,m]
# Im[conj(tw_i) h] = Re[conj(tw_i)] Im[h] - Im[conj(tw_i)] Re[h] ... do directly with linear ops
# Im[h(a,b)] = (1/N) sum_m (a_m Im[W E] + b_m Re[W E]); Re[h] = (1/N) sum (a Re[WE] - b Im[WE])
for k in range(3):
    WEm=W[k][None,:]*E            # (i,m)
    reW=WEm.real; imW=WEm.imag
    ct=np.conj(tw); cR=ct.real; cI=ct.imag
    # Im[ conj(tw)*h ] = cR*Im[h] - cI*Re[h]  (since Im[(cR+i cI)(hR+i hI)] = cR hI + cI hR; wait conj(tw)=cR+i cI)
    # (cR + i cI)(hR + i hI) -> imag = cR hI + cI hR
    Aa=(cR*imW + cI*reW)/N        # coeff of a_m
    Ab=(cR*reW - cI*imW)/N        # coeff of b_m  (i*b_m: Re part += -b Im[WE]; Im part += b Re[WE]) -> h = sum (a+i b)(WE): Re= a reW - b imW ; Im = a imW + b reW
    rows.append(np.hstack([Aa,Ab]))
A=np.vstack(rows)   # 768 x 512
u,s,vt=np.linalg.svd(A,full_matrices=False)
print('A shape',A.shape,'singular values tail:',np.round(s[-6:],4),'head:',np.round(s[:3],1))
print('effective rank (s>1e-6 s0):',int((s>s[0]*1e-6).sum()))
kdim=A.shape[1]-int((s>s[0]*1e-6).sum())
print('kernel dim:',kdim)
# also: does the TRUE u satisfy? compute true u from s_true vs W: u_m = |dot|... u_m^true = SH(m)/W direction
SH=np.array([to_spec(s_true[k]) for k in range(3)])
# solve LS with additional constraint from |u|=1 later; first inspect smallest singular vectors
for j in range(min(kdim,4) if kdim>0 else 0):
    v=vt[-1-j]
    a=v[:N]; b=v[N:]
    uu=(a+1j*b)
    # enforce pairing project
    ph=np.exp(1j*np.angle(uu))
    S=np.array([from_spec(ph*W[k]) for k in range(3)])
    Sr=np.clip(np.round(S),-9,9).astype(np.int64)
    slack=np.abs(S-Sr).mean()
    print('kernel vec %d: |Im-pairing viol| %.3f  slack %.3f'%(j,np.abs(uu[perm]-np.conj(uu)).mean(),slack))
