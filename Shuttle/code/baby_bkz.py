import numpy as np, time, sys
from fpylll import IntegerMatrix, LLL, BKZ, GSO
N=256
def ring_mul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:N].copy()
    out[:len(c)-N]-=c[N:]
    return out
def ring_conj(s):
    o=np.zeros(N,dtype=np.int64)
    o[0]=s[0]
    o[1:]=-s[N:0:-1]
    return o
def negshift(v,j):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
def fam_eq(a,b):
    for eps in (1,-1):
        for j in range(N):
            if np.array_equal(a,eps*negshift(b,j)): return (eps,j)
    return None
def load_key(fn):
    if '1m' in fn: return np.fromfile(fn,dtype=np.int32).reshape(3,N).astype(np.int64)
    if '4m' in fn: return np.fromfile(fn,dtype=np.int64).reshape(3,N)
    arr=np.frombuffer(open(fn,'rb').read(),dtype='<i4')
    return arr[3552//4:3552//4+768].reshape(3,N).astype(np.int64)
def imat(M2):
    M=IntegerMatrix(M2.shape[0],M2.shape[1])
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]): M[i,j]=int(M2[i,j])
    return M
key=sys.argv[1] if len(sys.argv)>1 else 's_true_1m.bin'
s=load_key(key)
sb=[ring_conj(s[k]) for k in range(3)]
gam={(k,kp):ring_mul(s[k],sb[kp]) for k in range(3) for kp in range(3)}
G=np.zeros((768,N),dtype=np.int64)
r=0
for i in range(3):
    for j in range(N):
        G[r]=negshift(gam[(0,i)],j); r+=1
t0=time.time()
M=imat(G)
LLL.reduction(M)
print('LLL done %.1fs'%(time.time()-t0),flush=True)
rows=np.array([[M[i,j] for j in range(N)] for i in range(768)],dtype=object)
nz=[np.array(r_,dtype=np.int64) for r_ in rows if any(r_)]
B=np.vstack(nz[:256])
def check(B):
    norms=np.sort(np.array([np.linalg.norm(b.astype(float)) for b in B]))
    Bm=imat(B)
    for idx in np.argsort([np.linalg.norm(b.astype(float)) for b in B])[:40]:
        v=B[idx]
        if not any(v): continue
        if np.array_equal(ring_mul(v,ring_conj(v)),gam[(0,0)]):
            return norms, v
    return norms, None
norms,v=check(B)
print('after LLL: ||b1||=%.1f  found=%s'%(norms[0],v is not None),flush=True)
M2=imat(np.vstack(nz[:256])) if False else None
from fpylll import IntegerMatrix as IM
Mb=IM(256,N)
sel=sorted(range(len(nz)),key=lambda i:np.linalg.norm(nz[i].astype(float)))[:256]
for i in range(256):
    for j in range(N): Mb[i,j]=int(nz[sel[i]][j])
for beta in (20,30,40,50,60,80):
    t1=time.time()
    par=BKZ.Param(block_size=beta,max_loops=16,float_type='ld')
    try:
        BKZ.reduction(Mb,par)
    except Exception as e:
        print('BKZ-%d interrupted (%r), using current basis'%(beta,e),flush=True)
    B=np.array([[Mb[i,j] for j in range(N)] for i in range(256)],dtype=np.int64)
    norms,v=check(B)
    el=time.time()-t1
    print('BKZ-%d (%.1fs): ||b1||=%.1f  found=%s'%(beta,el,norms[0],v is not None),flush=True)
    if v is not None:
        print('*** GENERATOR RECOVERED: gauge vs s0 =',fam_eq(v,s[0]),flush=True)
        # division step: recover s1, s2 via exact division gamma_0i / s0
        for i in (1,2):
            Mdiv=np.zeros((N,N),dtype=np.int64)
            for j in range(N): Mdiv[:,j]=negshift(s[0],j)
            # solve sum_j c_j x^j s0 = gamma_0i
            Bf=Mdiv.T.astype(float)
            c=np.linalg.lstsq(Bf,gam[(0,i)].astype(float),rcond=None)[0]
            cr=np.round(c).astype(np.int64)
            ok=np.array_equal(cr.astype(object)@Mdiv.astype(object),gam[(0,i)].astype(object))
            print('  division block %d exact: %s'%(i,ok),flush=True)
        break
print('total %.1fs'%(time.time()-t0))
