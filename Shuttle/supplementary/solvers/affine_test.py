import numpy as np, struct
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N)
def load_acc(fn):
    d=open(fn,'rb').read(); off=0
    n,=struct.unpack_from('<q',d,off); off+=8
    Pre=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
    Pim=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
    bs=np.frombuffer(d,dtype='<f8',count=3,offset=off); bq=np.frombuffer(d,dtype='<f8',count=3,offset=off+24)
    P=(Pre+1j*Pim)/n
    var=bq/(n*N)-(bs/(n*N))**2
    for k in range(3): P[k,k]-=N*var[k]
    return n,P
def gram(S):
    SH=np.array([np.fft.fft(S[k].astype(float)*tw) for k in range(3)])
    return np.array([[SH[k]*np.conj(SH[kp]) for kp in range(3)] for k in range(3)])
def to_time(X):
    return np.array([[np.real(np.conj(tw)*np.fft.ifft(X[k,kp])) for kp in range(3)] for k in range(3)])
def rmul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:N].copy(); out[:len(c)-N]-=c[N:]
    return out
def rconj(x):
    o=np.zeros(N,dtype=np.int64); o[0]=x[0]; o[1:]=-x[N:0:-1]
    return o
s_star=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
gamma_star=np.array([[rmul(s_star[k],rconj(s_star[kp])) for kp in range(3)] for k in range(3)])
# sample set: (name, M, s)  -- all 1M, all with the SAME mechanism (real loop)
samples=[]
def add(name,acc,s):
    n,M=load_acc(acc)
    samples.append((name,M,s))
add('main','corr1m',s_star)
for i in (1,2,3):
    arr=np.frombuffer(open('mk%d.bin_key.bin'%i,'rb').read(),dtype='<i4')
    s_i=arr[3552//4:3552//4+768].reshape(3,N).astype(np.int64)
    add('mk%d'%i,'mk%d.bin'%i,s_i)
rng=np.random.default_rng(99)
add('rand','einR3.bin',np.fromfile('s0_rand.bin',dtype=np.int32).reshape(3,N).astype(np.int64))
add('fin321','einR1.bin',np.fromfile('s0_fin321.bin',dtype=np.int32).reshape(3,N).astype(np.int64))
# LOO: fit channel-convolutional affine T = conv_w(G) + b on other keys, invert on held-out
# per channel (k,kp): T(m) - b(m) = sum_D w(D) G(m+D); fit shared w per channel + per-m bias b via LS on train keys
def fit_apply(hold):
    train=[x for x in samples if x[0]!=hold]
    hM=[x for x in samples if x[0]==hold][0][1]
    res={}
    Ds=range(-8,9)
    for k in range(3):
        for kp in range(k,3):
            rows=[];rhs=[]
            for name,M,s in train:
                G=gram(s)[k,kp]
                T=M[k,kp]-(fit_c3 if False else 0)  # T raw = M - c3*G with global c3 per sample? use per-sample c3
            # redo properly below
    return res
# simpler per-sample c3 then per-channel kernel fit
def c3of(M,G):
    num=sum((M[k,kp]*np.conj(G[k,kp])).sum() for k in range(3) for kp in range(k,3)).real
    den=sum((np.abs(G[k,kp])**2).sum() for k in range(3) for kp in range(k,3)).real
    return num/den
def Ts_of(M,s):
    G=gram(s)
    return M-c3of(M,G)*G, G
Ts_list=[]
for name,M,s in samples:
    T,G=Ts_of(M,s)
    Ts_list.append((name,T,G,M))
Ds=list(range(-8,9))
for hold in [x[0] for x in samples]:
    train=[x for x in Ts_list if x[0]!=hold]
    hT,hG,hM=[x for x in Ts_list if x[0]==hold][0][1:]
    # corrected estimate for held-out: Ghat = (M - b - conv_w(Ghat))^-1 ... iterate:
    # init Ghat0 = M/c3 ; T_hat = conv_w(Ghat)+b ; Ghat1 = (M - b - conv_w(Ghat0))/c3 ... fixed point
    for (k,kp) in [(0,0),(0,1),(0,2)]:
        A=[];y=[]
        for name,T,G,M in train:
            for m in range(N):
                row=[G[k,kp][(m+d)%N] for d in Ds]+[1.0]
                A.append(row); y.append(T[k,kp][m])
        A=np.array(A);y=np.array(y)
        w,res_,rank,sv=np.linalg.lstsq(A,y,rcond=None)
        # hold-out correction iterate
        c3h=c3of(hM,hG)
        Gh=hM[k,kp]/c3h
        for it in range(8):
            Th=sum(w[d+8]*Gh[(m+d)%N] for m in range(0)) if False else np.array([sum(w[dd]*Gh[(m+d)%N] for dd,d in enumerate(Ds)) for m in range(N)])+w[-1]
            Gh=(hM[k,kp]-Th)/c3h
        # error vs exact gamma (time domain) for this channel
        err=np.real(np.conj(tw)*np.fft.ifft(Gh))-gamma_star[k,kp] if hold=='main' else None
        if err is not None:
            print('%s LOO ch(%d,%d): per-lag err rms=%.3f max=%.3f #>0.5=%d'%(hold,k,kp,err.std(),np.abs(err).max(),int((np.abs(err)>0.5).sum())),flush=True)
