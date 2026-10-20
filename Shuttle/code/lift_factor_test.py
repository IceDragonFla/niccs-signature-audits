import numpy as np, time, sys
import sympy
from sympy import Poly, symbols, ZZ
N=256; H=128
def rmul(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
A=s[0][0::2].copy(); B=s[0][1::2].copy()
# honest N~(y) = A(y)^2 - y*B(y)^2, deg_y <= 255
cAA=np.convolve(A,A).astype(object)
cBB=np.convolve(B,B).astype(object)
Ntil=np.zeros(256,dtype=object)
Ntil[:len(cAA)]+=cAA
Ntil[1:1+len(cBB)]-=cBB
print('N~ coeffs: max|.|=%d  deg=%d'%(abs(Ntil).max(),np.nonzero(Ntil)[0].max()))
# sanity: N~ mod (y^128+1) == N(s0) in R'
def Nrel(a):
    sig=a*((-1)**np.arange(N))
    p=rmul(a,sig,N); return p[0::2]
n_s0=Nrel(s[0])
red=Ntil[:H].copy()
red[:H]-=Ntil[H:]
print('N~ mod (y^128+1) == N(s0):',np.array_equal(red,n_s0.astype(object)))
# route (iii) endgame: factor F(x) = N~(x^2) over Z[x]
x,y=symbols('x y')
Fcoeff=np.zeros(511,dtype=object)
Fcoeff[0::2]=Ntil  # N~(x^2): coeff of x^{2k} = N~_k
Fx=Poly(list(Fcoeff),x,domain=ZZ)
t0=time.time()
print('factoring F(x)=N~(x^2), deg %d...'%(511-np.nonzero(Fcoeff)[0].min()*0 or 510),flush=True)
fac=Fx.factor_list()
print('factorization %.0fs: %d factors'%(time.time()-t0,len(fac[1])))
for f,m in fac[1]:
    print('  deg %d mult %d head-coeff %s'%(f.degree(),m,str(f.LC())[:12]),flush=True)
# check: is s(x) one of the factors (up to content)?
sx=Poly(list(s[0].astype(object)),x,domain=ZZ)
hit=False
for f,m in fac[1]:
    q,r=divmod(sx,f)
    if r==0:
        print('*** s(x) IS a factor (deg %d) ***'%f.degree()); hit=True
if not hit:
    # s might be a product of several factors
    prods=[1]
    for f,m in fac[1]:
        prods_new=[]
        for p in prods:
            prods_new.append(p)
            if m==1: prods_new.append(p*f)
        prods=prods_new
        if len(prods)>4096: break
    for p in prods:
        if p==1: continue
        q,r=divmod(sx,p)
        if r==0 and p.degree()==sx.degree():
            print('*** s(x) = product of factors (deg %d) ***'%p.degree()); hit=True; break
print('route-iii endgame valid:',hit)
