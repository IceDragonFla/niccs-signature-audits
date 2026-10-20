import numpy as np, sys
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N); perm=(1-idx)%N
to_spec=lambda x: np.fft.fft(np.asarray(x,float)*tw)
from_spec=lambda sh: np.real(np.conj(tw)*np.fft.ifft(sh))
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
import struct
def get_M(fn):
    d=open(fn,'rb').read(); off=0
    n,=struct.unpack_from('<q',d,off); off+=8
    Pre=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
    Pim=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
    bs=np.frombuffer(d,dtype='<f8',count=3,offset=off); bq=np.frombuffer(d,dtype='<f8',count=3,offset=off+24)
    P=(Pre+1j*Pim)/n
    var=bq/(n*N)-(bs/(n*N))**2
    M=P.copy()
    for k in range(3): M[k,k]-=N*var[k]
    return M,P,n

DATA=sys.argv[1] if len(sys.argv)>1 else 'corr4m'
TRUTH=sys.argv[2] if len(sys.argv)>2 else 's_true_4m.bin'
Tcorr=sys.argv[3] if len(sys.argv)>3 else None   # optional common-bias tensor to subtract
M,P,n=get_M(DATA)
if Tcorr:
    T=np.load(Tcorr); M=M-T; print('subtracted common bias tensor')
s_true=np.fromfile(TRUTH,dtype=np.int64).reshape(3,N)
xpairs=[(0,1),(0,2),(1,2)]
num=sum((M[k,kp]*np.conj(to_spec(s_true[k])*np.conj(to_spec(s_true[kp])))).sum() for k,kp in xpairs).real
den=sum((np.abs(to_spec(s_true[k])*np.conj(to_spec(s_true[kp])))**2).sum() for k,kp in xpairs).real
C3=num/den
print('data=%s NSIG=%d C3=%.3f'%(DATA,n,C3))
def xcost(S):
    SH=np.array([to_spec(S[k]) for k in range(3)])
    return float(sum(np.sum(np.abs(M[k,kp]-C3*SH[k]*np.conj(SH[kp]))**2) for k,kp in xpairs))
print('cost(s_true)=%.4e'%xcost(s_true))
# eigvecs + V
Wv=np.zeros((3,N),dtype=complex); lam=np.zeros(N)
for m in range(N):
    G=(M[:,:,m]+M[:,:,m].conj().T)/2
    l,v=np.linalg.eigh(G); Wv[:,m]=v[:,-1]; lam[m]=l[-1]
for m in range(N):
    mp=perm[m]
    if m<mp:
        a=0.5*(Wv[:,m]+np.conj(Wv[:,mp])); Wv[:,m]=a; Wv[:,mp]=np.conj(a)
V=Wv*np.sqrt(np.maximum(lam,0)/C3)
# true-u oracle check
SHt=np.array([to_spec(s_true[k]) for k in range(3)])
ut=np.zeros(N,dtype=complex)
for m in range(N):
    ip=np.vdot(Wv[:,m],SHt[:,m]); ut[m]=ip/abs(ip)
St=np.array([from_spec(ut*V[k]) for k in range(3)])
Sr=np.clip(np.round(St),-9,9).astype(np.int64)
dev=np.abs(St-Sr)
print('ORACLE true-u: dev_mean=%.4f dev_max=%.3f WRONG=%d overlap=%d/768'%(dev.mean(),dev.max(),int((dev>0.4).sum()),fam_overlap(Sr,s_true)))
# BGS + CD solver tournament
def bgs(S,iters=80):
    S=S.copy()
    for it in range(iters):
        moved=False
        for k in range(3):
            oth=[j for j in range(3) if j!=k]
            SH=np.array([to_spec(S[j]) for j in range(3)])
            numk=M[k,oth[0]]*SH[oth[0]]+M[k,oth[1]]*SH[oth[1]]
            denk=C3*(np.abs(SH[oth[0]])**2+np.abs(SH[oth[1]])**2)
            SHk=numk/np.maximum(denk,1e-9)
            Snew=np.clip(np.round(from_spec(SHk)),-9,9).astype(np.int64)
            if not np.array_equal(Snew,S[k]): moved=True
            S[k]=Snew
        if not moved: break
    return S
BAS=np.exp(-2j*np.pi*np.outer(idx,idx)/N)*tw[None,:]
def cd(S,sweeps=25):
    SH=np.array([to_spec(S[k]) for k in range(3)])
    for sw in range(sweeps):
        improved=False
        for t in range(768):
            k,i=divmod(t,N)
            cur=float(np.sum([np.sum(np.abs(M[a,b]-C3*((SH[k] if a==k else SH[a])*np.conj(SH[k] if b==k else SH[b])))**2) for a,b in xpairs]))
            bestv=S[k][i]; bestc=cur
            for dv in (-2,-1,0,1,2):
                if dv==S[k][i]: continue
                SHk=SH[k]+(dv-S[k][i])*BAS[:,i]
                c=float(np.sum([np.sum(np.abs(M[a,b]-C3*((SHk if a==k else SH[a])*np.conj(SHk if b==k else SH[b])))**2) for a,b in xpairs]))
                if c<bestc-1e-3: bestc=c; bestv=dv
            if bestv!=S[k][i]:
                S[k][i]=bestv; SH[k]=to_spec(S[k]); improved=True
        if not improved: break
    return S
rng=np.random.default_rng(2027)
starts=[]
for t in range(int(sys.argv[4]) if len(sys.argv)>4 else 24):
    starts.append(rng.choice([-2,-1,0,1,2],size=(3,N),p=[.07,.17,.52,.17,.07]).astype(np.int64))
results=[]
for ii,S0 in enumerate(starts):
    S=cd(bgs(S0))
    c=xcost(S)
    results.append((c,S))
    if ii%6==0: print('  start %d: cost %.4e'%(ii,c),flush=True)
results.sort(key=lambda x:x[0])
print('tournament best 5:',[('%.3e'%c) for c,_ in results[:5]])
bS=results[0][1]
print('best overlap=%d/768 cost=%.4e'%(fam_overlap(bS,s_true),results[0][0]))
np.save('Sr_4m_best.npy',bS)
# also true-u + polish as sanity
Sp=cd(Sr.copy())
print('oracle-polish overlap=%d/768 cost=%.4e'%(fam_overlap(Sp,s_true),xcost(Sp)))
np.save('Sr_4m_oracle.npy',Sp)
