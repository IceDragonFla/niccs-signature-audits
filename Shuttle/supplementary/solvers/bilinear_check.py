import numpy as np
N=256; H=128
def rmul(a,b,n):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:n].copy(); out[:len(c)-n]-=c[n:]; return out
def rconj(s,n):
    o=np.zeros(n,dtype=np.int64); o[0]=s[0]; o[1:]=-s[n:0:-1]; return o
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
def sigma(a): return a*((-1)**np.arange(N))
def Nrel(a):
    p=rmul(a,sigma(a),N); return p[0::2].copy()
def ymul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:H].copy(); out[:len(c)-H]-=c[H:]
    return out
def ytimes(P):
    o=np.zeros(H,dtype=np.int64); o[1:]=P[:H-1]; o[0]=-P[H-1]
    return o
def yconj(p,n=H):
    o=np.zeros(n,dtype=np.int64); o[0]=p[0]; o[1:]=-p[n:0:-1]; return o
A=s[0][0::2].copy(); B=s[0][1::2].copy()
# gamma00 = s0 * conj(s0) in R (exact, attacker-derivable via gamma00=gamma01*gamma20/gamma21)
gam00=rmul(s[0],rconj(s[0],N),N)
# even part E0 = (gamma00 + sigma(gamma00))/2 -> in R': coeffs at even positions
g_even=(gam00[0::2].copy()).astype(np.int64)  # gamma00 already even? s*conj(s): check
print('gamma00 even?',np.abs(gam00[1::2]).max()==0)
# E0 = A*conj(A) + B*conj(B) in R'
E0=ymul(A,yconj(A))+ymul(B,yconj(B))
print('check E0 identity: gamma00_even == A*Ab + B*Bb :',np.array_equal(g_even,E0))
n_s0=Nrel(s[0])   # = A^2 - y B^2 in R'
print('check Nform: A^2-yB^2==N:',np.array_equal(ymul(A,A)-ytimes(ymul(B,B)),n_s0))
# bilinear identity: Nbar*A^2 + N*Abar^2 - 2*E0*(A*Abar) == N*Nbar - E0^2
LHS=ymul(yconj(n_s0),ymul(A,A))+ymul(n_s0,ymul(yconj(A),yconj(A)))-2*ymul(E0,ymul(A,yconj(A)))
RHS=ymul(n_s0,yconj(n_s0))-ymul(E0,E0)
print('check BILINEAR in A:',np.array_equal(LHS,RHS))
# same for B:  derive analog: from (2) B^2=(A^2-N)/y and (1) BBb = E0-AAb
# => (A^2-N)(Abar^2-Nbar)/y/(-y^127?) -- we already collapsed; do B-version directly:
LHS_B=ymul(yconj(n_s0),ymul(ytimes(ymul(B,B)),ytimes(ymul(B,B)))) if False else None
# direct B-elimination: A^2 = N + yB^2 ; Ab^2 = Nb + yb Bb^2 (yb = conj of y in R' = -y^127 = ytimes-based)
# E0 = AAb + BBb -> AAb = E0 - BBb
# (N+yB^2)(Nb+ybBb^2) = (E0-BBb)^2 with y*yb = y*conj(y): y*conj(y)=1 in eval |.|, as ring: y*(1/y)=1
# => N*Nb + N*ybBb^2 + Nb*yB^2 + y*yb*(BBb)^2 = E0^2 - 2E0BBb + (BBb)^2 ; y*yb=1
# => Nb*yB^2 + N*ybBb^2 - 2E0BBb + (1-1)(BBb)^2... : Nb*yB^2 + N*yb*Bb^2 = E0^2 - NNb + 2E0(BBb)
# verify numerically:
ybBb2=yconj(ytimes(ymul(B,B)))  # yb*Bbar^2
yB2=ytimes(ymul(B,B))
LB=ymul(yconj(n_s0),yB2)+ymul(n_s0,ybBb2)-2*ymul(E0,ymul(B,yconj(B)))
RB=ymul(E0,E0)-ymul(n_s0,yconj(n_s0))
print('check BILINEAR in B:',np.array_equal(2*ymul(E0,ymul(B,yconj(B))),ymul(E0,E0)-ymul(n_s0,yconj(n_s0))-LB+2*ymul(E0,ymul(B,yconj(B)))) if False else np.array_equal(LB,RB-2*ymul(E0,ymul(B,yconj(B)))+2*ymul(E0,ymul(B,yconj(B)))))
# cleaner verify of B identity: Nb*yB^2 + N*ybBb2 = E0^2 - NNb + 2E0*(BBb)
RB2=ymul(E0,E0)-ymul(n_s0,yconj(n_s0))+2*ymul(E0,ymul(B,yconj(B)))
print('check BILINEAR in B (clean):',np.array_equal(LB,RB2))
