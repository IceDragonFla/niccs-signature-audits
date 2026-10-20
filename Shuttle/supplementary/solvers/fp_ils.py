import numpy as np
from fpylll import IntegerMatrix, LLL, GSO, CVP
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
PHI=np.exp(-2j*np.pi*np.outer(idx,idx)/N)*tw[None,:]
def build_system(S0):
    SH=np.array([to_spec(S0[k]) for k in range(3)])
    rows=[];rhs=[]
    for k,kp in xpairs:
        R=M[k,kp]-C3*SH[k]*np.conj(SH[kp])
        Ak=C3*np.conj(SH[kp])[:,None]*PHI
        Akp=C3*np.conj(PHI*SH[k][:,None])
        Ap=np.zeros((512,768))
        Ap[:,k*256:(k+1)*256]=np.vstack([Ak.real,Ak.imag])
        Ap[:,kp*256:(kp+1)*256]+=np.vstack([Akp.real,Akp.imag])
        rows.append(Ap); rhs.append(np.r_[R.real,R.imag])
    return np.vstack(rows),np.concatenate(rhs)
S0=np.load('Sr_deep2.npy').astype(np.int64)
A,b=build_system(S0)
# reduce dimension via QR projection: solve in row space of A (rank<=768)
# lattice: columns sigma*e_i plus A rows -> use GSO-based Babai on B = [sigma*I | A^T] rows? Build 768 x (768+1536) basis rows
sigma=np.sqrt(np.mean(np.abs(b)**2)/4)*2
B=np.zeros((768,768+1536))
B[:,:768]=sigma*np.eye(768)
B[:,768:]=A.T   # rows: basis vectors (sigma e_i , A_i-row?) careful: we want lattice {(sigma*delta, A delta)}
print('building integer matrix...',flush=True)
SC=1000
Bi=(B*SC).round().astype(np.int64)
Ml=IntegerMatrix(Bi.shape[0],Bi.shape[1])
for i in range(Bi.shape[0]):
    for j in range(Bi.shape[1]):
        Ml[i,j]=int(Bi[i,j])
print('LLL on 768 x 2304...',flush=True)
LLL.reduction(Ml)
print('LLL done; Babai...',flush=True)
Mgor=GSO.Mat(Ml,float_type='ld')
Mgor.update_gso()
tar=(np.concatenate([np.zeros(768),(b*SC)])).round().astype(np.int64)
w=CVP(Mgor)
cv=np.array(w.closest_vector(tar.tolist()))
delta=(cv[:768]/(sigma*SC)).round().astype(np.int64).ravel()
print('|delta|_1=',int(np.abs(delta).sum()))
S1=S0+delta.reshape(3,256)
print('overlap after Babai CVP:',fam_overlap(S1,s_true))
np.save('Sr_fpils.npy',S1)
