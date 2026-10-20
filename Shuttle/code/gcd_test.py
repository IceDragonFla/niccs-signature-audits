import numpy as np
from sympy import Poly, symbols, ZZ, gcd as pgcd
N=256
def rmul_int(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj_int(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
x=symbols('x')
def honest(vec):
    nz=np.nonzero(vec)[0]
    return Poly(list(vec.astype(int)),x,domain=ZZ) if len(nz) else Poly(0,x,domain=ZZ)
gam01=rmul_int(s[0],rconj_int(s[1],N),N)
gam02=rmul_int(s[0],rconj_int(s[2],N),N)
def sigma(a): return a*((-1)**np.arange(N))
Ns=rmul_int(s[0],sigma(s[0]),N)   # N(s0) as R-element (even poly)
sx=honest(s[0])
G1=honest(gam01); G2=honest(gam02); NR=honest(Ns)
print('deg: gam01=%d gam02=%d N=%d sx=%d'%(G1.degree(),G2.degree(),NR.degree(),sx.degree()))
import time
t0=time.time()
g12=pgcd(G1,G2)
print('gcd(gam01,gam02): deg %d (%.1fs)'%(g12.degree(),time.time()-t0))
q,r=divmod(sx,g12)
print('  s | gcd?',r==0)
g1n=pgcd(G1,NR)
print('gcd(gam01,N): deg %d'%(g1n.degree()))
q,r=divmod(sx,g1n)
print('  s | gcd1N?',r==0, ' deg gcd=%d'%g1n.degree())
gall=pgcd(g12,NR)
print('gcd(all three): deg %d'%gall.degree())
q,r=divmod(sx,gall)
print('  s | gcdAll?',r==0)
