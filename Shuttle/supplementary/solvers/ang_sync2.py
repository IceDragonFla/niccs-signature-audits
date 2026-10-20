import numpy as np
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N); perm=(1-idx)%N
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
W=np.zeros((3,N),dtype=complex)
for m in range(N):
    G=(M[:,:,m]+M[:,:,m].conj().T)/2
    lam,v=np.linalg.eigh(G); W[:,m]=v[:,-1]
for m in range(N):
    mp=perm[m]
    if m<mp:
        a=0.5*(W[:,m]+np.conj(W[:,mp])); W[:,m]=a; W[:,mp]=np.conj(a)
E=np.exp(2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)
def build_rows(Wk):
    WEm=Wk[None,:]*E
    reW,imW=WEm.real,WEm.imag
    ct=np.conj(tw); cR,cI=ct.real,ct.imag
    return np.hstack([(cR[:,None]*imW+cI[:,None]*reW)/N,(cR[:,None]*reW-cI[:,None]*imW)/N])
A=np.vstack([build_rows(W[k]) for k in range(3)])
half=np.array([m for m in range(N) if m<perm[m]]); nh=len(half)
Hm=np.zeros((N,nh)); Hs=np.zeros((N,nh))
for j,m in enumerate(half): Hm[m,j]=1; Hm[perm[m],j]=1
for j,m in enumerate(half): Hs[m,j]=1; Hs[perm[m],j]=-1
Ah=np.hstack([A[:,:N]@Hm, A[:,N:]@Hs])
B=Ah.T@Ah
wv,vv=np.linalg.eigh(B)
u0red=vv[:,0]
u_full=Hm@u0red[:nh]+1j*(Hs@u0red[nh:])
sgn=np.sign(u_full[half]); sgn[sgn==0]=1
phi0=np.arctan2(np.imag(u_full[half]*sgn),np.maximum(np.real(u_full[half]*sgn),1e-9))
def uvec(sgn,phi):
    u=np.zeros(N,dtype=complex)
    u[half]=sgn*np.exp(1j*phi)
    u[perm[half]]=sgn*np.exp(-1j*phi)
    return u
def xvec(u): return np.concatenate([u[half].real,u[half].imag])
r0=Ah@xvec(uvec(sgn,phi0))
print('init ||r||=%.4f'%np.linalg.norm(r0))
eps=1e-6; phi=phi0.copy()
for it in range(100):
    u=uvec(sgn,phi); r=Ah@xvec(u)
    J=np.empty((len(r),nh))
    for j in range(nh):
        ph2=phi.copy(); ph2[j]+=eps
        J[:,j]=(Ah@xvec(uvec(sgn,ph2))-r)/eps
    dphi,_,_,_=np.linalg.lstsq(J,-r,rcond=None)
    phi+=0.8*dphi
    if np.linalg.norm(dphi)<1e-12: break
u=uvec(sgn,phi)
S=np.array([from_spec(u*W[k]) for k in range(3)])
Sr=np.clip(np.round(S),-9,9).astype(np.int64)
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
ov=max(int(sum((Sr[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))
print('GN: ||r||=%.4f slack=%.3f overlap=%d/768'%(np.linalg.norm(Ah@xvec(u)),np.abs(S-Sr).mean(),ov))
np.save('Sr_angsync.npy',Sr)
