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
    # a == ± x^j b for some j?
    for eps in (1,-1):
        for j in range(N):
            if np.array_equal(a,eps*negshift(b,j)): return (eps,j)
    return None

def load_key(fn):
    if fn.endswith('s_true_1m.bin'):
        return np.fromfile(fn,dtype=np.int32).reshape(3,N).astype(np.int64)
    if fn.endswith('s_true_4m.bin'):
        return np.fromfile(fn,dtype=np.int64).reshape(3,N)
    arr=np.frombuffer(open(fn,'rb').read(),dtype='<i4')
    return arr[3552//4:3552//4+768].reshape(3,N).astype(np.int64)

def imat_from(M2):
    Ml=IntegerMatrix(M2.shape[0],M2.shape[1])
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            Ml[i,j]=int(M2[i,j])
    return Ml

def babai_solve(B,s):
    # B: 256x256 reduced basis rows; solve B c = s over Z (float Babai + exact verify)
    Bf=B.astype(float)
    c=np.linalg.lstsq(Bf.T,s.astype(float),rcond=None)[0]
    cr=np.round(c).astype(np.int64)
    ok=np.array_equal(cr.astype(object)@B.astype(object),s.astype(object))
    if not ok:
        # nearest-plane style iterative refinement
        for _ in range(4):
            r=(s.astype(object)-(cr.astype(object)@B.astype(object)))
            d=np.linalg.lstsq(Bf.T,r.astype(float),rcond=None)[0]
            cr=cr+np.round(d).astype(np.int64)
            if np.array_equal(cr.astype(object)@B.astype(object),s.astype(object)):
                ok=True; break
    return ok

def experiment(tag,s,use_bkz=False,verbose=True):
    t0=time.time()
    sb=[ring_conj(s[k]) for k in range(3)]
    gam={}
    for k in range(3):
        for kp in range(3):
            gam[(k,kp)]=ring_mul(s[k],sb[kp])
    # sanity: gamma(0)=||s_k||^2 on diag; cross check twisted spectrum
    for k in range(3):
        assert gam[(k,k)][0]==int((s[k].astype(np.int64)**2).sum())
    tw=np.exp(1j*np.pi*np.arange(N)/N)
    for k in range(3):
        f=np.fft.fft(s[k].astype(float)*tw)
        g2=np.fft.ifft(f*np.conj(f))*np.conj(tw)
        assert np.abs(g2-gam[(k,k)]).max()<1e-6
    if verbose: print('%s: ring arithmetic verified (time %.1fs)'%(tag,time.time()-t0))
    # ideal J = (gamma_00)+(gamma_01)+(gamma_02); generators x^j*gamma_0i
    G=np.zeros((768,N),dtype=np.int64)
    r=0
    for i in range(3):
        for j in range(N):
            G[r]=negshift(gam[(0,i)],j); r+=1
    t1=time.time()
    Ml=imat_from(G)
    LLL.reduction(Ml)
    # extract 256 nonzero independent rows
    rows=np.array([[Ml[i,j] for j in range(N)] for i in range(768)],dtype=object)
    nz=[rows[i] for i in range(768) if any(rows[i])]
    # rank check via numpy float
    Bf=np.array([list(map(float,r_)) for r_ in nz])
    rank=np.linalg.matrix_rank(Bf)
    assert rank==256, 'rank %d'%rank
    # take first 256 independent (they span the lattice after LLL: nonzero rows are a basis)
    B=np.array([np.array(r_,dtype=np.int64) for r_ in nz[:256]])
    t2=time.time()
    if use_bkz:
        Mb=imat_from(np.vstack([B]))
        par=BKZ.Param(block_size=20)
        BKZ.reduction(Mb,par)
        B=np.array([[Mb[i,j] for j in range(N)] for i in range(256)],dtype=np.int64)
    t3=time.time()
    # membership: s0 in J ?
    ok=babai_solve(B,s[0])
    # generator recovery: short vectors v, test v*conj(v)==gamma_00
    norms=np.array([int(np.linalg.norm(b.astype(float))) for b in B])
    order=np.argsort(norms)
    found=None
    for idx in order[:20]:
        v=B[idx]
        if not any(v): continue
        vv=ring_mul(v,ring_conj(v))
        if np.array_equal(vv,gam[(0,0)]):
            found=v; break
    t4=time.time()
    res=dict(tag=tag,member=ok,found=found is not None,
             v1norm=float(norms[order[0]]),s0norm=float(np.linalg.norm(s[0].astype(float))),
             t_ideal=t2-t1,t_bkz=t3-t2,t_total=time.time()-t0)
    if found is not None:
        rel=fam_eq(found,s[0])
        res['gauge']=rel
        # division step: s1bar = gamma_01 / s0 ; check integrality via solve
        Mdiv=np.zeros((N,N),dtype=np.int64)
        for j in range(N):
            Mdiv[:,j]=negshift(s[0],j)   # columns = x^j s0 -> Mdiv @ c = sum c_j x^j s0
        ok1=babai_solve(Mdiv.T,gam[(0,1)])  # want q with q*s0... careful: solve s0*q = gamma01 -> q coefficients
        res['division']=bool(ok1)
    if verbose:
        print('%s: s0 in J: %s | generator found: %s | ||v1||=%.1f ||s0||=%.1f | gauge=%s | t(LLL %.1fs BKZ %.1fs total %.1fs)'%(
            tag,ok,res['found'],res['v1norm'],res['s0norm'],res.get('gauge'),res['t_ideal'],res['t_bkz'],res['t_total']))
    return res

if __name__=='__main__':
    keys=[('1M-main','s_true_1m.bin'),('4M-main','s_true_4m.bin'),
          ('mk1','mk1.bin_key.bin'),('mk2','mk2.bin_key.bin'),('mk3','mk3.bin_key.bin')]
    allres=[]
    for tag,fn in keys:
        try:
            s=load_key(fn)
            allres.append(experiment(tag,s,use_bkz=False))
        except Exception as e:
            print(tag,'ERROR',repr(e))
