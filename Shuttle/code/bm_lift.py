import numpy as np
import time
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
forms=[]; cvals=[]
for k in range(H):
    F1=sym(W[k]@T)
    Q=np.zeros((2*H,2*H)); Q[:H,:H]=F1; Q[H:,H:]=F1
    forms.append(Q); cvals.append(E0[k])
for k in range(H):
    C=(W[k]@T).T+sum(Rinv[k,m]*(W[m]@T) for m in range(H))
    Q=np.zeros((2*H,2*H)); Q[:H,H:]=0.5*C; Q[H:,:H]=0.5*C.T
    forms.append(Q); cvals.append(O0[k])
for k in range(H):
    Fb=sum(Ry[k,m]*W[m] for m in range(H))
    Q=np.zeros((2*H,2*H)); Q[:H,:H]=sym(W[k]); Q[H:,H:]=-sym(Fb)
    forms.append(Q); cvals.append(nrel[k])
tw=np.exp(1j*np.pi*np.arange(N)/N)
def emb_row(j):
    row=np.zeros(2*H,dtype=complex)
    for m in range(H):
        row[m]=tw[2*m]*np.exp(-2j*np.pi*j*2*m/N)
        row[H+m]=tw[2*m+1]*np.exp(-2j*np.pi*j*(2*m+1)/N)
    return row
for j in range(N):
    r=emb_row(j)
    Q=np.outer(r.real,r.real)+np.outer(r.imag,r.imag)
    forms.append(Q); cvals.append(float(abs(r@xtrue)**2))
Qs=np.array(forms); cs=np.array(cvals)
print('forms %d verified err=%.2e'%(len(Qs),max(abs((Qs@xtrue*xtrue).sum(axis=1)-cs))))
def negshift(v,j):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
def famscore(x):
    return max(int((x==np.r_[eps*negshift(s[0][0::2],d),eps*negshift(s[0][1::2],d)]).sum()) for eps in (1,-1) for d in range(N))
rng=np.random.default_rng(7)
r=4
best=(1e18,None)
t0=time.time()
for restart in range(40):
    R=rng.normal(size=(2*H,r))*0.3
    lr=2e-4
    for it in range(4000):
        vals=np.einsum('kij,ia,ja->k',Qs,R,R)
        g=np.einsum('k,kij,ja->ia',2*(vals-cs),Qs,R)
        R=R-lr*g
        if it%1000==999:
            f=np.linalg.norm(np.einsum('kij,ia,ja->k',Qs,R,R)-cs)
    X=R@R.T
    f=np.linalg.norm(np.einsum('kij,ia,ja->k',Qs,R,R)-cs)
    if f<best[0]: best=(f,R.copy())
    if restart%8==0:
        print('restart %d: f=%.3e  (%.0fs)'%(restart,f,time.time()-t0),flush=True)
f,R=best
print('best f=%.3e'%f)
X=R@R.T
wv,vv=np.linalg.eigh(X)
print('top eigs:',np.round(wv[-4:],2))
v=vv[:,-1]
sc=np.sqrt(max(wv[-1],0))
v=v/np.linalg.norm(v)
Ab=v[:H]*sc*np.sqrt(2*H)*np.linalg.norm(v[:H])/(np.linalg.norm(v[:H])+1e-12) if False else v[:H]/(np.linalg.norm(v[:H])+1e-12)*np.sqrt(wv[-1])
Bb=v[H:]/(np.linalg.norm(v[H:])+1e-12)*np.sqrt(wv[-1])
print('cand norms: %.2f %.2f'%(np.linalg.norm(Ab),np.linalg.norm(Bb)))
# instead: try direct extraction assuming rank-1-ish: take leading column pattern
v2=vv[:,-1]
sign=np.sign(v2[:H].sum())
v2=v2*sign
Ab=v2[:H]; Bb=v2[H:]
Ar=np.round(Ab*3)/3 if False else None
print('overlap(normalized-rounded):',famscore(np.round(np.r_[Ab,Bb]*3).astype(int)) if False else 'scale-free check')
sc2=np.linalg.norm(xtrue)/np.linalg.norm(v2)
xhat=v2*sc2
xr=np.round(xhat)
print('scaled rounded overlap: %d/256 maxdev=%.3f'%(famscore(xr.astype(int)) if False else int((xr==np.r_[s[0][0::2],s[0][1::2]]).sum()),np.abs(xhat-xr).max()))
np.save('bm_best.npy',R)
