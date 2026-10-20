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
# constraints Im[ conj(tw_i) h_ki(u) ] = 0 ; h_ki = (1/N) sum_m u_m W_k(m) E[i,m]
# full complex linear map u -> residual vector r (768 real): build as real matrices acting on (a,b)
def build_rows(Wk):
    WEm=Wk[None,:]*E
    reW,imW=WEm.real,WEm.imag
    ct=np.conj(tw); cR,cI=ct.real,ct.imag
    Aa=(cR*imW[:, :]+cI*reW)/N   # coefficient of a_m  (im(c*h)=cR*Im h + cI*Re h; Im h = a imW + b reW; Re h = a reW - b imW)
    Ab=(cR*reW-cI*imW)/N
    return np.hstack([Aa,Ab])
A=np.vstack([build_rows(W[k]) for k in range(3)])
# pairing: a_perm=a, b_perm=-b -> unknowns A_half(128) + B_half(128): a = Pa @ ah etc.
half=[m for m in range(N) if m<perm[m]]
Hm=np.zeros((N,len(half)))
for j,m in enumerate(half): Hm[m,j]=1; Hm[perm[m],j]=1
Hs=np.zeros((N,len(half)))
for j,m in enumerate(half): Hs[m,j]=1; Hs[perm[m],j]=-1
Ah=A[:,:N]@Hm + A[:,N:]@Hs   # 768 x 256
# spectral init: smallest eigenvector of Ah^T Ah
B=Ah.T@Ah
wv,vv=np.linalg.eigh(B)
u0=vv[:,0]
# check truth residual scale
print('sv tail of Ah: ',np.round(np.linalg.svd(Ah,compute_uv=False)[-4:],5))
def resid(uv): return np.linalg.norm(Ah@uv)
print('||A u_spec||=%.4f  (||u||=%.3f)'%(resid(u0),np.linalg.norm(u0)))
# Gauss-Newton on torus: parametrize u = exp(i*(phi0 + P psi)), psi in R^255 (fix gauge)
def u_of(psi,phi00=0.0):
    full=np.concatenate(([phi00],psi))
    # map half-vector to full via Hm/Hs indices: build index arrays
    ph=np.zeros(N); ph[half]=full[1:]; ph[perm[half]]=full[1:]
    ph[half[0] if False else 0]=full[0] if (0 in half) else ph[0]
    return ph
# simpler: unknowns = phi on half indices (128), u_m=exp(i phi_j), u_perm=exp(-i phi_j)
half_idx=np.array(half)
def uvec(phi):
    u=np.zeros(N,dtype=complex)
    u[half_idx]=np.exp(1j*phi)
    u[perm[half_idx]]=np.exp(-1j*phi)
    return u
def lin_rows(phi):
    u=uvec(phi)
    # derivative of Im[...] wrt phi_j: d u_m/d phi_j = i u_m (m=half), -i conj... build numerically? analytic:
    # r = A @ [a;b] with a,b from u. dr/dphi_j: use chain: du/dphi
    cols=np.zeros((Ah.shape[0],len(half)))
    for j,m in enumerate(half):
        du1=1j*u[m]*Hm[m,j]*0  # placeholder
    return cols
# do GN numerically with analytic Jacobian via complex step is messy; use small eps finite diff on 128 dims (cheap: 768x128)
def r_of(phi):
    u=uvec(phi); x=np.concatenate([u.real,u.imag])
    return Ah@x
eps=1e-6
phi=0.5*(lambda t:t)(np.zeros(len(half)))  # init from spectral? spectral u0 is real-coef... get phases:
# init phases from spectral u0 mapped: sign + magnitude ignored; take phi = angle of best unit-modulus fit
uu=np.zeros(N,dtype=complex)
uu[half_idx]=np.exp(1j*np.angle(u0[half_idx]+1e-12)*0+np.angle(u0[half_idx]*0+1)*0) if False else np.ones(len(half))
# better init: use u0 signs: s_j = sign(u0)
sgn=np.sign(u0[half_idx]); sgn[sgn==0]=1
phi=np.arccos(np.clip(sgn,-1,1))*0  # 0 -> u=1*sgn not unit... u=sgn is unit (±1)!
# iterate GN
for it in range(60):
    u=uvec(np.concatenate(([0],phi))[1:] if len(phi) else phi) if False else uvec_from(sgn,phi)
    break
# cleaner implementation below
def uvec2(signs,phi):
    u=np.zeros(N,dtype=complex)
    mag=signs.astype(float)
    u[half_idx]=mag*np.exp(1j*phi)
    u[perm[half_idx]]=mag*np.exp(-1j*phi)  # conj for real signs
    return u
phi=np.zeros(len(half))
for it in range(80):
    u=uvec2(sgn,phi); x=np.concatenate([u.real,u.imag]); r=Ah@x
    J=np.zeros((len(r),len(half)))
    for j in range(len(half)):
        ph2=phi.copy(); ph2[j]+=eps
        u2=uvec2(sgn,ph2); x2=np.concatenate([u2.real,u2.imag])
        J[:,j]=(Ah@x2-r)/eps
    dphi,_,_,_=np.linalg.lstsq(J,-r,rcond=None)
    phi=phi+0.7*dphi
    if np.linalg.norm(dphi)<1e-10: break
u=uvec2(sgn,phi)
S=np.array([from_spec(u*W[k]) for k in range(3)])
Sr=np.clip(np.round(S),-9,9).astype(np.int64)
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
ov=max(int(sum((Sr[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))
print('ang-sync GN: final ||r||=%.4f  slack=%.3f  overlap=%d/768'%(np.linalg.norm(Ah@np.concatenate([u.real,u.imag])),np.abs(S-Sr).mean(),ov))
