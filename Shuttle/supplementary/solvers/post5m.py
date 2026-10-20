import numpy as np, struct, sys
N=256; idx=np.arange(N); tw=np.exp(1j*np.pi*idx/N); perm=(1-idx)%N

def load_acc(fn):
    d=open(fn,'rb').read()
    off=0
    NSIG,=struct.unpack_from('<q',d,off); off+=8
    Pre=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
    Pim=np.frombuffer(d,dtype='<f8',count=9*N,offset=off).reshape(3,3,N); off+=9*N*8
    bs=np.frombuffer(d,dtype='<f8',count=3,offset=off); off+=24
    bq=np.frombuffer(d,dtype='<f8',count=3,offset=off)
    return NSIG,Pre,Pim,bs,bq

def combine(files,out):
    T=None
    for f in files:
        n,pr,pi,bs,bq=load_acc(f)
        if T is None: T=[n,pr.copy(),pi.copy(),bs.copy(),bq.copy()]
        else:
            T[0]+=n; T[1]+=pr; T[2]+=pi; T[3]+=bs; T[4]+=bq
    n,pr,pi,bs,bq=T
    with open(out,'wb') as o:
        o.write(struct.pack('<q',n)); o.write(pr.astype('<f8').tobytes())
        o.write(pi.astype('<f8').tobytes()); o.write(bs.astype('<f8').tobytes()); o.write(bq.astype('<f8').tobytes())
    return n

def stats(fn,s_true,tag):
    n,pr,pi,bs,bq=load_acc(fn)
    P=(pr+1j*pi)/n
    var=bq/(n*N)-(bs/(n*N))**2
    M=P.copy()
    for k in range(3): M[k,k]-=N*var[k]
    to_spec=lambda x: np.fft.fft(np.asarray(x,float)*tw)
    SHt=np.array([to_spec(s_true[k]) for k in range(3)])
    Gt=np.array([[SHt[k]*np.conj(SHt[kp]) for kp in range(3)] for k in range(3)])
    num=sum((M[k,kp]*np.conj(Gt[k,kp])).sum() for k in range(3) for kp in range(3)).real
    den=sum((np.abs(Gt[k,kp])**2).sum() for k in range(3) for kp in range(3)).real
    c3=num/den
    res=np.abs(M-c3*Gt).sum()/np.abs(c3*Gt).sum()
    # eigvec error + true-u reconstruction
    Wv=np.zeros((3,N),dtype=complex); lam=np.zeros(N)
    errs=[]
    for m in range(N):
        G=(M[:,:,m]+M[:,:,m].conj().T)/2
        l,v=np.linalg.eigh(G); Wv[:,m]=v[:,-1]; lam[m]=l[-1]
        t=SHt[:,m]
        ct=np.abs(np.vdot(t,Wv[:,m]))/(np.linalg.norm(t)*np.linalg.norm(Wv[:,m]))
        errs.append(np.degrees(np.arccos(np.clip(ct,-1,1))))
    errs=np.array(errs)
    ut=np.zeros(N,dtype=complex)
    for m in range(N):
        ip=np.vdot(Wv[:,m],SHt[:,m]); ut[m]=ip/abs(ip)
    V=Wv*np.sqrt(np.maximum(lam,0)/c3)
    from_spec=lambda sh: np.real(np.conj(tw)*np.fft.ifft(sh))
    St=np.array([from_spec(ut*V[k]) for k in range(3)])
    Sr=np.clip(np.round(St),-9,9).astype(np.int64)
    dev=np.abs(St-Sr)
    def negshift(s,dd):
        o=np.roll(s,dd)
        if dd: o[:dd]=-o[:dd]
        return o
    ov=max(int(sum((Sr[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))
    def fam(S,T):
        b=0
        for eps in (1,-1):
            for dd in range(N):
                b=max(b,int(sum((S[k]==eps*negshift(T[k],dd)).sum() for k in range(3))))
        return b
    C3=c3
    xpairs=[(0,1),(0,2),(1,2)]
    def xcost(S):
        SH=np.array([to_spec(S[k]) for k in range(3)])
        return float(sum(np.sum(np.abs(M[k,kp]-C3*SH[k]*np.conj(SH[kp]))**2) for k,kp in xpairs))
    print('%s: NSIG=%d c3=%.3f gram_res=%.4f eigvec_med=%.1fdeg p90=%.1f | true-u recon: dev_mean=%.4f dev_max=%.3f overlap=%d/768 cost_true=%.4e'%(
        tag,n,c3,res,np.median(errs),np.percentile(errs,90),dev.mean(),dev.max(),fam(Sr,s_true),xcost(s_true)))
    # bias/noise decomposition cross pairs
    rmsR=[];rmsN=[]
    for k in range(3):
        for kp in range(k+1,3):
            rmsR.append(np.sqrt(np.mean(np.abs(M[k,kp]-c3*Gt[k,kp])**2)))
            rmsN.append(np.sqrt(np.mean(np.abs(P[k,k])*np.abs(P[kp,kp])/n)))
    print('   bias/noise: rms_resid=%s rms_noise=%s'%(np.round(rmsR,1),np.round(rmsN,1)))
    return dict(M=M,P=P,c3=c3,s_true=s_true)

if __name__=='__main__':
    s1=np.fromfile('s_true_1m.bin',dtype=np.int32).reshape(3,N).astype(np.int64)
    # combined 5M for main key
    n=combine(['corr1m','corrA','corrB'],'corr5m')
    print('combined NSIG=',n)
    stats('corr5m',s1,'5M-main')
    stats('corr1m',s1,'1M-main(ref)')
    # verify key match between corrA dump and main
    import hashlib
    kA=open('corrA_key.bin','rb').read()[:55]
    kM=open('corr1m_key.bin','rb').read()[:55]
    print('keygen seed match A vs 1M:',kA==kM)
    # multi-key stats + LOO bias learning
    MK=[]
    for i in (1,2,3):
        kb=open('mk%d_key.bin'%i,'rb').read()
        arr=np.frombuffer(kb,dtype='<i4')
        off=None
        for o in range(1264//4, len(arr)-768, 64):
            v=arr[o:o+768]
            if np.abs(v).max()<=2 and (v!=0).mean()>0.3:
                off=o; break
        s_i=arr[off:off+768].reshape(3,N).astype(np.int64)
        print('mk%d: auto-located block offset %d bytes, density %.3f'%(i,off*4,(s_i!=0).mean()))
        MK.append(stats('mk%d.bin'%i,s_i,'mk%d'%i))
    # LOO bias learning: T_i = M_i - c3_i * Gram(s_i); test common component
    Ts=[]
    for i,st in enumerate(MK):
        to_spec=lambda x: np.fft.fft(np.asarray(x,float)*tw)
        SH=np.array([to_spec(st['s_true'][k]) for k in range(3)])
        G=np.array([[SH[k]*np.conj(SH[kp]) for kp in range(3)] for k in range(3)])
        Ts.append(st['M']-st['c3']*G)
    Tmean=np.mean(Ts,axis=0)
    print('T rms per key:',[round(float(np.sqrt(np.mean(np.abs(T)**2))),1) for T in Ts])
    for i,T in enumerate(Ts):
        r=T-Tmean
        print('  key%d: |T-Tmean|/|T| = %.3f'%(i+1,np.linalg.norm(r)/np.linalg.norm(T)))
