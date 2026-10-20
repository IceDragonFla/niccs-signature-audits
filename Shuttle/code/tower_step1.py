import os, sys, time
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
FPLLL.set_precision(212)
N=256; H=128
def rmul(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
def negsh(v,j,n):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
def sigma(a): return a*((-1)**np.arange(N))
def Nrel(a):
    p=rmul(a,sigma(a),N); return p[0::2].copy()
n_s0=Nrel(s[0])
n01=Nrel(rmul(s[0],rconj(s[1],N),N))
n02=Nrel(rmul(s[0],rconj(s[2],N),N))
# attacker ideal in R' (from rounded gammas == truth, verified separately)
g01=np.load('ghat_01_incl.npy'); g01=np.round(g01).astype(np.int64)
g02=np.load('ghat_02_incl.npy'); g02=np.round(g02).astype(np.int64)
a01=Nrel(g01); a02=Nrel(g02)
print('attacker n01==truth:',np.array_equal(a01,n01),' n02==truth:',np.array_equal(a02,n02))
G=np.array([negsh(a01,j,H) for j in range(H)]+[negsh(a02,j,H) for j in range(H)])
M=IntegerMatrix(2*H,H)
for i in range(2*H):
    for j in range(H): M[i,j]=int(G[i][j])
t0=time.time(); LLL.reduction(M)
rows=np.array([[M[i,j] for j in range(H)] for i in range(2*H)])
nz=[np.array(r,dtype=np.int64) for r in rows if any(r)]
B=np.vstack(nz[:H])
norms=np.linalg.norm(B.astype(float),axis=1)
print('LLL %.0fs ||b1||=%.1f target=194.4'%(time.time()-t0,norms.min()),flush=True)
def found_scan(Bx):
    for v in Bx[np.argsort(np.linalg.norm(Bx.astype(float),axis=1))[:60]]:
        if any(v)==0: continue
        for eps in (1,-1):
            for j in range(H):
                if np.array_equal(v,eps*negsh(n_s0,j,H)):
                    return v,(eps,j)
    return None,None
v,gauge=found_scan(B)
print('LLL direct hit:',v is not None, gauge,flush=True)
if v is None:
    from fpylll import load_strategies_json
    strat=load_strategies_json(os.path.expanduser('~/libs-fplll/share/fplll/strategies/default.json'))
    for beta in (20,30,40,50,60):
        t1=time.time()
        try:
            BKZ.reduction(M,BKZ.Param(block_size=beta,max_duration=3600,float_type='mpfr',strategies=strat))
        except Exception as e:
            print('BKZ-%d exc %r'%(beta,e),flush=True)
        Bx=np.array([[M[i,j] for j in range(H)] for i in range(H)],dtype=np.int64)
        n1=float(np.min(np.linalg.norm(Bx.astype(float),axis=1)))
        v,gauge=found_scan(Bx)
        print('BKZ-%d (%.0fs): ||b1||=%.1f hit=%s %s'%(beta,time.time()-t1,n1,v is not None,gauge),flush=True)
        if v is not None: break
if v is not None:
    print('*** STEP-1 COMPLETE: N(s0) recovered, gauge=(%+d, y^%d)'%gauge,flush=True)
    np.save('tower_Ns0.npy',n_s0)
    A=s[0][0::2]; Bb=s[0][1::2]
    np.save('tower_AB.npy',np.vstack([A,Bb]))
