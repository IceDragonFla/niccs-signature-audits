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
sv=np.linalg.svd(Ah,compute_uv=False)
print('smallest 12 sv:',np.array2string(sv[-12:],precision=6))
print('largest sv:',sv[0],'  numerical kernel dim (sv<1e-8*sv0):',int((sv<sv[0]*1e-8).sum()))
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
def overlap(S):
    return max(int(sum((S[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))
# true u: align each w(m) to true SH(m): u_m = <w,SH>/|<w,SH>| * conj? solve w*u ~ SH: u = SH/w component-wise along w
SH=np.array([to_spec(s_true[k]) for k in range(3)])
ut=np.zeros(N,dtype=complex)
for m in range(N):
    ip=np.vdot(W[:,m],SH[:,m])
    ut[m]=ip/abs(ip)
print('true u: pairing viol %.2e, |u| dev %.2e'%(np.abs(ut[perm]-np.conj(ut)).max(),np.abs(np.abs(ut)-1).max()))
xt=np.concatenate([ut[half].real,ut[half].imag])
print('true u residual ||Ah xt||=%.4e (vs sv[-1]=%.2e)'%(np.linalg.norm(Ah@xt),sv[-1]))
St=np.array([from_spec(ut*W[k]) for k in range(3)])
Str=np.clip(np.round(St),-9,9).astype(np.int64)
print('true-u reconstruction: slack %.4f overlap %d/768'%(np.abs(St-Str).mean(),overlap(Str)))
np.save('ut_true.npy',ut); np.save('W_eig.npy',W); np.save('Ah.npy',Ah)
