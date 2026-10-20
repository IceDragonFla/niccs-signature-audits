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
s_star=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
n,Mstar=load_acc('corr1m')
Gs=gram(s_star)
# exact integer gamma for reference
def rmul(a,b):
    c=np.convolve(a.astype(np.int64),b.astype(np.int64))
    out=c[:N].copy(); out[:len(c)-N]-=c[N:]
    return out
def rconj(x):
    o=np.zeros(N,dtype=np.int64); o[0]=x[0]; o[1:]=-x[N:0:-1]
    return o
gam_exact=np.array([[rmul(s_star[k],rconj(s_star[kp])) for kp in range(3)] for k in range(3)])
# common c3 fitted on Mstar
c3=sum((Mstar[k,kp]*np.conj(Gs[k,kp])).sum() for k in range(3) for kp in range(3)).real / \
    sum((np.abs(Gs[k,kp])**2).sum() for k in range(3) for kp in range(3)).real
print('c3* = %.4f'%c3)
Tstar=Mstar-c3*Gs   # oracle correction (contains eta*)
def report(tag, Tcorr):
    Mcorr=Mstar-Tcorr
    e_time=to_time(Mcorr/c3) - gam_exact   # error vs exact integer gamma (gamma of Gram has scale 1? check)
    # note: to_time(Gs) should equal gam_exact; verify scale first
    cross=[e_time[0,1],e_time[0,2],e_time[1,2]]
    need=[e_time[0,0],e_time[0,1],e_time[0,2]]
    print('%-14s | cross: rms=%.3f max=%.3f | needed(00,01,02): rms=%.3f max=%.3f | #>0.5(all9)=%d'%(
        tag, np.abs(np.array(cross)).std(), np.abs(np.array(cross)).max(),
        np.abs(np.array(need)).std(), np.abs(np.array(need)).max(),
        int((np.abs(e_time)>0.5).sum())))
# scale sanity
chk=to_time(Gs)
print('scale check |to_time(Gs)-gamma_exact| max = %.2e'%np.abs(chk-gam_exact).max())
report('ORACLE T(s*)', Tstar)
for tag,acc,s0f in [('R0 s* repro','einR0.bin','s0_truth.bin'),
                    ('R1 S_final(321)','einR1.bin','s0_fin321.bin'),
                    ('R2 Sr_trueu(357)','einR2.bin','s0_trueu357.bin'),
                    ('R3 random','einR3.bin','s0_rand.bin')]:
    try:
        n0,M0=load_acc(acc)
        s0=np.fromfile(s0f,dtype=np.int32).reshape(3,N).astype(np.int64)
        T0=M0-c3*gram(s0)
        report(tag,T0)
        # decompose: Tstar - T0 deterministic part vs noise
        dT=Tstar-T0
        print('   |T*-T0| freq-rms=%.1f (indep-noise ~%.1f) => %s'%(
            np.sqrt(np.mean(np.abs(dT[0,1])**2)), 1758*np.sqrt(2),
            'PAIRED' if np.sqrt(np.mean(np.abs(dT[0,1])**2))<1758 else 'independent'))
    except FileNotFoundError:
        print(tag,'not ready')
