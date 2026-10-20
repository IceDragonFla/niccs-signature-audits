import numpy as np, sys
H=128
def rmulN(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:N].copy(); out[:len(c)-N]-=c[N:]; return out
def rmul_int(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64)); out=c[:H].copy(); out[:len(c)-H]-=c[H:]; return out
def rconj_int(s):
    o=np.zeros(H,dtype=np.int64); o[0]=s[0]; o[1:]=-s[H:0:-1]; return o
N=256
s=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
def sigma(a): return a*((-1)**np.arange(N))
nrel=rmulN(s[0],sigma(s[0]))[0::2]
A_true=s[0][0::2].copy(); B_true=s[0][1::2].copy()
# simulate DFS: at level k choose B_{k-1}; A_k determined IF q_k known; test how many (B_{k-1}, q_k) pairs give integral small A_k
# practical probe: with TRUTH (A,B,q known), measure per-level: given correct prefix, how many wrong B_{k-1} keep |A_k|<=3 for SOME q_k in +-60?
q_true=None
# honest N~ = A^2 - y*B^2 honest (deg<=255)
cAA=np.convolve(A_true,A_true); cBB=np.convolve(B_true,B_true)
Ntil=np.zeros(511,dtype=np.int64)
Ntil[:len(cAA)]+=cAA
Ntil[1:1+len(cBB)]-=cBB
q_true=Ntil[H:].copy()  # high coeffs
print('true |q|max=',np.abs(q_true).max(),' |A|max=',np.abs(A_true).max(),' |B|max=',np.abs(B_true).max())
# per-level ambiguity: A_k = (N_k + q_k + (B^2)_{k-1} - sum_{i=1}^{k-1} A_i A_{k-i}) / (2 A_0)
# given TRUE prefix A_0..A_{k-1}, B_0..B_{k-2}: try all B_{k-1} in -3..3 and q_k in -60..60:
# count pairs giving integer A_k with |A_k|<=3. Also record how many give A_k == true A_k.
rng=np.random.default_rng(0)
tot_pairs=0; tot_keep=0; tot_true=0
for k in range(1,40):
    prefix_known=(np.array_equal(A_true[:k],np.round(A_true[:k])) )
    # sum term
    st=sum(int(A_true[i]*A_true[k-i]) for i in range(1,k))
    b2prev=sum(int(B_true[i]*B_true[k-1-i]) for i in range(0,k)) if k>=1 else 0
    # note (B^2)_{k-1} = sum_{i+j=k-1} B_i B_j includes B_{k-1} via 2B_0B_{k-1}
    keep=0; truehit=0
    for btry in range(-3,4):
        b2=int(sum(int(B_true[i]*B_true[k-1-i]) for i in range(0,k)))
        # replace B_{k-1} contribution: compute with B_{k-1}=btry
        b2_try=b2 - 2*B_true[0]*B_true[k-1] + 2*B_true[0]*btry if k-1>=0 else b2
        for qtry in range(-60,61,1):
            num=nrel[k]+qtry+b2_try-st
            den=2*A_true[0]
            if num%den==0:
                ak=num//den
                if abs(ak)<=3:
                    keep+=1
                    if ak==A_true[k] and btry==B_true[k-1]: truehit+=1
    tot_pairs+=7*121; tot_keep+=keep; tot_true+=truehit
    if k<=10 or k%10==0:
        print('level %d: keep %d / 847 pairs (true hit %d)'%(k,keep,truehit))
print('TOTAL first39: keep %d of %d pairs -> avg branching %.2f'%(tot_keep,tot_pairs,tot_keep/39.0))
