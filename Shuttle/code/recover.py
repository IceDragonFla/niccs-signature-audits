import numpy as np
N=256
idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
s_true=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)

def to_spec(s): return np.fft.fft(s.astype(float)*tw)
def from_spec(sh): return np.real(np.conj(tw)*np.fft.ifft(sh))

d=open('corr1m','rb').read()
off=0
NSIG,=np.frombuffer(d,dtype=np.int64,count=1,offset=off); off+=8
Pre=np.frombuffer(d,dtype=np.float64,count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
Pim=np.frombuffer(d,dtype=np.float64,count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
bsum=np.frombuffer(d,dtype=np.float64,count=3,offset=off); off+=24
bsq=np.frombuffer(d,dtype=np.float64,count=3,offset=off)
P=(Pre+1j*Pim)/NSIG
var=bsq/(NSIG*N)-(bsum/(NSIG*N))**2
print("NSIG",NSIG,"block vars",var)
M=P.copy()
for k in range(3): M[k,k]-=N*var[k]
SHt=[to_spec(s_true[k]) for k in range(3)]
Gt=np.array([[SHt[k]*np.conj(SHt[kp]) for kp in range(3)] for k in range(3)])
num=sum((M[k,kp]*np.conj(Gt[k,kp])).sum() for k in range(3) for kp in range(3)).real
den=sum((np.abs(Gt[k,kp])**2).sum() for k in range(3) for kp in range(3)).real
c3=num/den
res=np.abs(M-c3*Gt).sum()/np.abs(c3*Gt).sum()
print(f"c3={c3:.4f} gram residual={res:.4f} (per-entry SNR ~ {1/res:.1f})")

def build_W(cscale):
    W=np.zeros((3,N),dtype=complex)
    for m in range(N):
        G=M[:,:,m]/cscale
        G=(G+G.conj().T)/2
        lam,v=np.linalg.eigh(G)
        W[:,m]=np.sqrt(max(lam[-1],0))*v[:,-1]
    for m in range(N//2):
        mp=N-1-m
        if m==mp: W[:,m]=np.real(W[:,m]); continue
        W[:,mp]=np.conj(W[:,m])
    return W

def run_AP(W, th, iters=300):
    for it in range(iters):
        S=np.clip(np.round(np.array([from_spec(np.exp(1j*th)*W[k]) for k in range(3)])),-9,9)
        acc=np.zeros(N,dtype=complex)
        for k in range(3): acc+=to_spec(S[k])*np.conj(W[k])
        th=np.angle(acc)
        th=0.5*(th-th[::-1])
    return th

def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
def overlap(Sr):
    return max(int(sum((Sr[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))

rng=np.random.default_rng(2026)
best=None
for cs in [c3*f for f in (0.9,0.95,1.0,1.05,1.1)]:
    W=build_W(cs)
    for r in range(10):
        th0=np.zeros(N) if r==0 else 0.5*(lambda t:t-t[::-1])(rng.uniform(0,2*np.pi,N))
        th=run_AP(W,th0)
        S=np.array([from_spec(np.exp(1j*th)*W[k]) for k in range(3)])
        Sr=np.clip(np.round(S),-9,9).astype(np.int64)
        sc=np.abs(S-Sr).mean()
        ov=overlap(Sr)
        if best is None or (sc,-ov)<(best[0],-best[1]): best=(sc,Sr,ov,cs,W.copy())
    print(f"cs {cs:.1f}: best slack {best[0]:.3f} overlap {best[1] if isinstance(best[1],int) else best[2]}/{768}")
sc,Sr,ov,cs,W=best
print(f"AP stage: slack {sc:.3f} overlap {ov}/768")

# coordinate-descent polish on ||M - c3*Gram(S)||^2
def gram_of(S):
    SH=[to_spec(S[k]) for k in range(3)]
    return np.array([[SH[k]*np.conj(SH[kp]) for kp in range(3)] for k in range(3)])
def cost(S):
    G=gram_of(S)
    return float((np.abs(M-c3*G)**2).sum())
S=Sr.astype(np.int64).copy()
cur=cost(S)
improved=True
sweep=0
while improved and sweep<12:
    improved=False; sweep+=1
    for k in range(3):
        # incremental: changing S[k][i] by delta changes SH[k] by delta*twist*delta_basis
        SH=[to_spec(S[j]) for j in range(3)]
        base_e=np.fft.fft(np.eye(N)[0]*0)  # placeholder
        for i in range(N):
            # effect of S[k][i] += dv : SH[k] += dv * e_i_spectrum, e_i_spectrum[m] = tw_i * e^{+2pi i m i /N}
            ei = tw[i]*np.exp(2j*np.pi*idx*i/N)
            bestv=S[k][i]; bestc=cur
            for dv in range(-9,10):
                if dv==S[k][i]: continue
                dSH=(dv-S[k][i])*ei
                c=cur
                # recompute affected cost terms exactly for row/col k
                SHk=SH[k]+dSH
                c=0.0
                for a in range(3):
                    for b in range(3):
                        Ga = SHk if a==k else SH[a]
                        Gb = SHk if b==k else SH[b]
                        c+=float((np.abs(M[a,b]-c3*(Ga*np.conj(Gb)))**2).sum())
                if c<bestc-1e-9:
                    bestc=c; bestv=dv
            if bestv!=S[k][i]:
                S[k][i]=bestv
                SH[k]=to_spec(S[k])
                cur=bestc
                improved=True
    print(f"sweep {sweep}: cost {cur:.3e}, overlap {overlap(S)}/768")
np.save('S_final.npy', S)
ov=overlap(S)
print(f"FINAL overlap: {ov}/768  cost {cur:.3e}")
if ov==768:
    print("*** EXACT RECOVERY (up to global shift/sign) ***")
    for sg in (1,-1):
        for dd in range(N):
            if all(np.array_equal(S[k], sg*negshift(s_true[k],dd)) for k in range(3)):
                print(f"relation: S = {sg:+} * (s . X^{dd})"); break
        else: continue
        break
