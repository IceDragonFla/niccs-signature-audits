import numpy as np, time
from scipy.optimize import least_squares
H=128; N=256
def rmul_int(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj_int(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
gam00=rmul_int(s[0],rconj_int(s[0],N),N)
E0=gam00[0::2].astype(float); O0=gam00[1::2].astype(float)
def sigma(a): return a*((-1)**np.arange(N))
nrel=rmul_int(s[0],sigma(s[0]),N)[0::2].astype(float)
xtrue=np.concatenate([s[0][0::2],s[0][1::2]]).astype(float)
W=[np.zeros((H,H)) for _ in range(H)]
for k in range(H):
    for i in range(H):
        j1=k-i
        if 0<=j1<H: W[k][i,j1]+=1.0
        j2=k+H-i
        if 0<=j2<H: W[k][i,j2]-=1.0
T=np.zeros((H,H))
for j in range(H):
    e=np.zeros(H); e[j]=1.0
    T[:,j]=np.r_[e[0],-e[:0:-1]]
Ry=np.zeros((H,H))
for k in range(1,H): Ry[k,k-1]=1.0
Ry[0,H-1]=-1.0
Rinv=np.zeros((H,H))
for k in range(H-1): Rinv[k,k+1]=1.0
Rinv[H-1,0]=-1.0
def sym(M): return 0.5*(M+M.T)
Qs=[]; cs=[]
for k in range(H):
    F1=sym(W[k]@T); Q=np.zeros((2*H,2*H)); Q[:H,:H]=F1; Q[H:,H:]=F1
    Qs.append(Q); cs.append(E0[k])
Cts=[(W[k]@T).T+sum(Rinv[k,m]*(W[m]@T) for m in range(H)) for k in range(H)]
for k in range(H):
    Q=np.zeros((2*H,2*H)); Q[:H,H:]=0.5*Cts[k]; Q[H:,:H]=0.5*Cts[k].T
    Qs.append(Q); cs.append(O0[k])
Fbs=[sum(Ry[k,m]*W[m] for m in range(H)) for k in range(H)]
for k in range(H):
    Q=np.zeros((2*H,2*H)); Q[:H,:H]=sym(W[k]); Q[H:,H:]=-sym(Fbs[k])
    Qs.append(Q); cs.append(nrel[k])
tw=np.exp(1j*np.pi*np.arange(N)/N)
def emb_row(j):
    row=np.zeros(2*H,dtype=complex)
    for m in range(H):
        row[m]=tw[2*m]*np.exp(-2j*np.pi*j*2*m/N)
        row[H+m]=tw[2*m+1]*np.exp(-2j*np.pi*j*(2*m+1)/N)
    return row
rows=[emb_row(j) for j in range(N)]
for j in range(N):
    r=rows[j]
    Q=np.outer(r.real,r.real)+np.outer(r.imag,r.imag)
    Qs.append(Q); cs.append(float(abs(r@xtrue)**2))
Qs=np.array(Qs); cs=np.array(cs)
Qflat=Qs.reshape(len(Qs),-1)
def resid(x):
    return Qflat@(np.outer(x,x).ravel())-cs
def jac(x):
    return 2*(Qs@x)
def negshift(v,j):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
truth=np.r_[s[0][0::2],s[0][1::2]]
def fam(x):
    return max(int((x==np.r_[eps*negshift(s[0][0::2],d),eps*negshift(s[0][1::2],d)]).sum()) for eps in (1,-1) for d in range(N))
print('sanity: resid(truth)=%.2e'%np.linalg.norm(resid(xtrue)))
rng=np.random.default_rng(5)
t0=time.time()
sols=[]
for rs in range(60):
    x0=rng.uniform(-2,2,size=2*H)
    try:
        sol=least_squares(resid,x0,jac=jac,method='trf',xtol=1e-14,ftol=1e-14,max_nfev=200)
        f=np.linalg.norm(sol.fun)
        if f<1e-6:
            xr=np.round(sol.x)
            ov=fam(xr.astype(int))
            sols.append((ov,f))
            if ov>=250:
                print('*** HIT rs=%d ov=%d f=%.2e'%(rs,ov,f),flush=True)
    except Exception:
        pass
    if rs%10==9:
        print('rs %d (%.0fs): %d solutions, best ov=%s'%(rs+1,time.time()-t0,len(sols),max([o for o,_ in sols]) if sols else '-'),flush=True)
if sols:
    ovs=sorted(set(o for o,_ in sols))
    print('solution overlaps distribution:',{o:sum(1 for v,_ in sols if v==o) for o in ovs})
