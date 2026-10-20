import numpy as np
N=256; KVEC=4; R=825.0
d=open('irs_realkey.bin','rb').read()
# header: sk_tilde KVEC polys (int32 x N)
skt=np.frombuffer(d,dtype='<i4',count=KVEC*N).reshape(KVEC,N).astype(np.int64)
off=KVEC*N*4
# per trial: seedY SEEDBYTES + seedC CHALLENGESEEDBYTES + c N*4 + y KVEC polys + z KVEC polys
# infer record size: remaining/(count) -- find SB,CB
rem=len(d)-off
print('remaining bytes',rem)
for SB,CB,T in [(32,32,2000),(48,32,2000),(32,16,2000),(32,32,2063),(48,16,2000)]:
    if rem % (SB+CB+4*N+2*KVEC*N*4)==0:
        print('fits T=%d with SB=%d CB=%d rec=%d'%(rem//(SB+CB+4*N+2*KVEC*N*4),SB,CB,SB+CB+4*N+2*KVEC*N*4)); break
rec=SB+CB+4*N+2*KVEC*N*4
T=rem//rec
def negsh(v,j):
    o=np.roll(v,j)
    if j: o[:j]=-o[:j]
    return o
tw=np.exp(1j*np.pi*np.arange(N)/N)
rows=[]
for q in range(T):
    o=off+q*rec
    c=np.frombuffer(d,dtype='<i4',count=N,offset=o+SB+CB)
    y=np.frombuffer(d,dtype='<i4',count=KVEC*N,offset=o+SB+CB+4*N).reshape(KVEC,N).astype(np.int64)
    z=np.frombuffer(d,dtype='<i4',count=KVEC*N,offset=o+SB+CB+4*N+KVEC*N*4).reshape(KVEC,N).astype(np.int64)
    supp=np.nonzero(c)[0]
    if len(supp)==0: continue
    # v_j = e_j * X^j * skt (blockwise negshift per block; e_j applies to all blocks)
    # z - y accumulated: compute Delta per block
    D=(z-y).astype(np.int64)
    for jj,j in enumerate(supp):
        ej=1 if c[j]>0 else -1
        # v_j blocks: ej * negshift(skt_blk, j)
        vb=np.array([ej*negsh(skt[k],j) for k in range(KVEC)])
        Vv=int((vb.astype(float)**2).sum())
        t=float((y.astype(float)*vb).sum())
        proj=float((D.astype(float)*vb).sum())
        f=1 if proj>0 else -1
        rows.append((t,Vv,f))
    if q>=1999: break
rows=np.array(rows)
print('transitions:',len(rows))
# empirical b(t): bin by tau = t/r^2
tau=rows[:,0]/(R*R)
f=rows[:,2]
Vtyp=np.median(rows[:,1])
print('V median',Vtyp)
bins=np.linspace(-0.3,0.3,25)
idx=np.digitize(tau,bins)
print(' tau-bin   n     E[f]_emp   b_pred')
def b_pred(t,V,r=R,M=15):
    # hit prob = sum_i (exp(hi/2r^2)-exp(lo/2r^2)), intervals [lo,hi] intersect (-inf,0]
    s=0.0
    for i in range(M):
        lo=-2*(2*i+1)*t-(2*i+1)**2*V
        hi=-4*i*t-4*i*i*V
        a,b=max(lo,-1e18),min(hi,0.0)
        if b>a: s+=np.exp(b/(2*r*r))-np.exp(a/(2*r*r))
    return min(max(s,0.0),1.0)
for bi in range(1,25):
    m=idx==bi
    if m.sum()<50: continue
    tm=tau[m].mean()
    emp=f[m].mean()
    bp=b_pred(tm*R*R,Vtyp)
    print('%8.4f %5d  %+8.4f  %8.4f'%(tm,m.sum(),emp,2*bp-1))
