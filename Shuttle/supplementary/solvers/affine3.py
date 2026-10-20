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
def rmul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:N].copy(); out[:len(c)-N]-=c[N:]
    return out
def rconj(x):
    o=np.zeros(N,dtype=np.int64); o[0]=x[0]; o[1:]=-x[N:0:-1]
    return o
def gamma_exact(S):
    return np.array([[rmul(S[k],rconj(S[kp])) for kp in range(3)] for k in range(3)])
# ---- (1) empirical noise floor: corrA vs corrB (same key, 2M, independent streams)
nA,MA=load_acc('corrA.bin'); nB,MB=load_acc('corrB.bin')
s4=np.fromfile('s_true_4m.bin',dtype=np.int64).reshape(3,N)
G4=gram(s4)
c3=sum((MA[k,kp]*np.conj(G4[k,kp])).sum() for k in range(3) for kp in range(k,3)).real/\
   sum((np.abs(G4[k,kp])**2).sum() for k in range(3) for kp in range(k,3)).real
diff=(MA-MB)/c3
td=np.array([[np.real(np.conj(tw)*np.fft.ifft(diff[k,kp])) for kp in range(3)] for k in range(3)])
print('=== empirical noise calibration (corrA vs corrB, 2M, independent) ===')
print('cross per-lag: rms=%.4f (theory 2M: %.4f)  diag rms=%.4f'%(
    np.array([td[0,1],td[0,2],td[1,2]]).std(), np.sqrt(2*(1758*np.sqrt(2))**2/256)/c3*0.5,
    np.array([td[0,0],td[1,1],td[2,2]]).std()/np.sqrt(2)))
noise_1M=np.array([td[0,1],td[0,2],td[1,2]]).std()/np.sqrt(2)*np.sqrt(2)  # eta per 2M sample; scale to 1M: *sqrt2
print('=> eta per 2M lag rms=%.4f; per 1M=%.4f; per 4M=%.4f'%(
    np.array([td[0,1],td[0,2],td[1,2]]).std()/np.sqrt(2),
    np.array([td[0,1],td[0,2],td[1,2]]).std(),
    np.array([td[0,1],td[0,2],td[1,2]]).std()/2))
# ---- (2) affine LOO with 4M key held out (train: six 1M keys)
samples=[]
def add(name,acc,s):
    n,M=load_acc(acc); samples.append((name,M,s))
add('main','corr1m',np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64))
for i in (1,2,3):
    arr=np.frombuffer(open('mk%d.bin_key.bin'%i,'rb').read(),dtype='<i4')
    add('mk%d'%i,'mk%d.bin'%i,arr[3552//4:3552//4+768].reshape(3,N).astype(np.int64))
add('rand','einR3.bin',np.fromfile('s0_rand.bin',dtype=np.int32).reshape(3,N).astype(np.int64))
add('fin321','einR1.bin',np.fromfile('s0_fin321.bin',dtype=np.int32).reshape(3,N).astype(np.int64))
def c3of(M,G):
    num=sum((M[k,kp]*np.conj(G[k,kp])).sum() for k in range(3) for kp in range(k,3)).real
    den=sum((np.abs(G[k,kp])**2).sum() for k in range(3) for kp in range(k,3)).real
    return num/den
Ts=[]
for name,M,s in samples:
    G=gram(s); Ts.append((name,M-c3of(M,G)*G,G,M,c3of(M,G)))
hold='4M'
nH,MH=load_acc('corr4m')
GH=gram(s4); c3H=c3of(MH,GH)
TH=MH-c3H*GH
gam4=gamma_exact(s4)
Ds=list(range(-8,9))
print('=== affine LOO: hold-out = 4M key (noise floor per-lag 4M = %.4f) ==='%(
    np.array([td[0,1],td[0,2],td[1,2]]).std()/2))
for (k,kp) in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]:
    A=[];y=[]
    for name,T,G,M,c3s in Ts:
        Gm=G[k,kp]
        for m in range(N):
            A.append([Gm[(m+d)%N] for d in Ds]+[1.0]); y.append(T[k,kp][m])
    w,_,_,_=np.linalg.lstsq(np.array(A),np.array(y),rcond=None)
    Gh=MH[k,kp]/c3H
    for it in range(10):
        Th=np.array([sum(w[dd]*Gh[(m+d)%N] for dd,d in enumerate(Ds)) for m in range(N)])+w[-1]
        Gh=(MH[k,kp]-Th)/c3H
    err=np.real(np.conj(tw)*np.fft.ifft(Gh))-gam4[k,kp]
    print('ch(%d,%d): per-lag err rms=%.4f max=%.4f #>0.5=%d'%(k,kp,err.std(),np.abs(err).max(),int((np.abs(err)>0.5).sum())))
    # kernel spectrum: min |c3 + Lhat(omega)|
    Lh=np.array([sum(w[dd]*np.exp(-2j*np.pi*om*(d)/N) for dd,d in enumerate(Ds)) for om in range(N)])
    mn=np.abs(c3H+Lh).min()
    if (k,kp) in [(0,1),(0,2)]:
        print('   inversion spectrum: min|c3+L(w)|=%.4f (c3=%.2f) => amplification max %.2fx'%(mn,c3H,c3H/mn))
