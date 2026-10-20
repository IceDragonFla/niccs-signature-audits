import numpy as np, struct, sys
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
def combine(files,out):
    T=None
    for f in files:
        n,pr,pi,bs,bq=load_acc_raw(f)
        if T is None: T=[n,pr,pi,bs,bq]
        else:
            T[0]+=n; T[1]=T[1]+pr; T[2]=T[2]+pi; T[3]=T[3]+bs; T[4]=T[4]+bq
    n,pr,pi,bs,bq=T
    with open(out,'wb') as o:
        o.write(struct.pack('<q',n)); o.write(np.asarray(pr,dtype='<f8').tobytes())
        o.write(np.asarray(pi,dtype='<f8').tobytes()); o.write(np.asarray(bs,dtype='<f8').tobytes()); o.write(np.asarray(bq,dtype='<f8').tobytes())
    return n
def load_acc_raw(fn):
    d=open(fn,'rb').read(); off=0
    n,=struct.unpack_from('<q',d,off); off+=8
    Pre=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).copy(); off+=9*N*8
    Pim=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).copy(); off+=9*N*8
    bs=np.frombuffer(d,dtype='<f8',count=3,offset=off).copy(); bq=np.frombuffer(d,dtype='<f8',count=3,offset=off+24).copy()
    return n,Pre,Pim,bs,bq
def gram(S):
    SH=np.array([np.fft.fft(S[k].astype(float)*tw) for k in range(3)])
    return np.array([[SH[k]*np.conj(SH[kp]) for kp in range(3)] for k in range(3)])
def c3of(M,G):
    num=sum((M[k,kp]*np.conj(G[k,kp])).sum() for k in range(3) for kp in range(k,3)).real
    den=sum((np.abs(G[k,kp])**2).sum() for k in range(3) for kp in range(k,3)).real
    return num/den
def rmul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:N].copy(); out[:len(c)-N]-=c[N:]
    return out
def rconj(x):
    o=np.zeros(N,dtype=np.int64); o[0]=x[0]; o[1:]=-x[N:0:-1]
    return o
s_star=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
gam=np.array([[rmul(s_star[k],rconj(s_star[kp])) for kp in range(3)] for k in range(3)])
# combine campaign
n=combine(['camp%d.bin'%i for i in range(8)],'campall')
print('campaign total NSIG=%d'%n)
nM,MH=load_acc('campall')
GH=gram(s_star); c3H=c3of(MH,GH)
print('c3=%.3f'%c3H)
# training keys
samples=[('main','corr1m',s_star)]
for i in (1,2,3):
    arr=np.frombuffer(open('mk%d.bin_key.bin'%i,'rb').read(),dtype='<i4')
    samples.append(('mk%d'%i,'mk%d.bin'%i,arr[888:888+768].reshape(3,N).astype(np.int64)))
samples.append(('rand','einR3.bin',np.fromfile('s0_rand.bin',dtype=np.int32).reshape(3,N).astype(np.int64)))
samples.append(('fin','einR1.bin',np.fromfile('s0_fin321.bin',dtype=np.int32).reshape(3,N).astype(np.int64)))
# NOTE: training includes 'main' (corr1m) itself - the SAME key as target: that adds direct T(s*) info at 1M.
# Report both with and without.
Ds=list(range(-8,9))
def fit_L(chans, include_main):
    L={}
    for (k,kp) in chans:
        A=[];y=[]
        for name,acc,s in samples:
            if name=='main' and not include_main: continue
            nT,MT=load_acc(acc); G=gram(s)
            T=MT-c3of(MT,G)*G
            Gm=G[k,kp]
            for m in range(N):
                A.append([Gm[(m+d)%N] for d in Ds]+[1.0]); y.append(T[k,kp][m])
        w,_,_,_=np.linalg.lstsq(np.array(A),np.array(y),rcond=None)
        L[(k,kp)]=w
    return L
def correct(L,chans):
    out={}
    for (k,kp) in chans:
        w=L[(k,kp)]
        Gh=MH[k,kp]/c3H
        for it in range(12):
            Th=np.array([sum(w[dd]*Gh[(m+d)%N] for dd,d in enumerate(Ds)) for m in range(N)])+w[-1]
            Gh=(MH[k,kp]-Th)/c3H
        out[(k,kp)]=Gh
    return out
chans=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
for inc in (True,False):
    L=fit_L(chans,inc)
    Gc=correct(L,chans)
    tag='L-incl-main' if inc else 'L-excl-main'
    print('=== %s ==='%tag)
    for (k,kp) in chans:
        gh=Gc[(k,kp)]
        err=np.real(np.conj(tw)*np.fft.ifft(gh))-gam[k,kp]
        wrong=int((np.abs(err)>0.5).sum())
        print(' ch(%d,%d): rms=%.4f max=%.4f WRONG=%d/256'%(k,kp,err.std(),np.abs(err).max(),wrong))
    # the decisive channels
    w01=int((np.abs(np.real(np.conj(tw)*np.fft.ifft(Gc[(0,1)]))-gam[0,1])>0.5).sum())
    w02=int((np.abs(np.real(np.conj(tw)*np.fft.ifft(Gc[(0,2)]))-gam[0,2])>0.5).sum())
    print(' >>> gamma01 wrong=%d, gamma02 wrong=%d (0 = exact)  [ideal input ready: %s]'%(
        w01,w02, 'YES' if w01+w02==0 else 'needs %d flips'%(w01+w02)))
np.save('MH_campaign.npy',MH)
