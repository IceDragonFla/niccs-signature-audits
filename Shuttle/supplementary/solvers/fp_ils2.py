exec(open('fp_ils.py').read().split('Ml=IntegerMatrix(Bi.shape')[0])
from fpylll import IntegerMatrix, LLL, GSO
SC=1000
Bi=(B*SC).round().astype(np.int64)
Ml=IntegerMatrix(Bi.shape[0],Bi.shape[1])
for i in range(Bi.shape[0]):
    for j in range(Bi.shape[1]):
        Ml[i,j]=int(Bi[i,j])
print('LLL...',flush=True)
LLL.reduction(Ml)
print('GSO...',flush=True)
Mgor=GSO.Mat(Ml,float_type='ld')
Mgor.update_gso()
Br=np.array([ [Ml[i,j] for j in range(Bi.shape[1])] for i in range(Bi.shape[0])],dtype=float)
# Gram-Schmidt of Br (rows) manually
bs=Br.copy(); mu=np.zeros((768,768))
for i in range(768):
    for j in range(i):
        mu[i,j]=np.dot(Br[i],bs[j])/np.dot(bs[j],bs[j])
        bs[i]-=mu[i,j]*bs[j]
tar=np.concatenate([np.zeros(768),b*SC])
tp=tar.copy(); coef=np.zeros(768)
for i in range(767,-1,-1):
    c=round(np.dot(tp,bs[i])/np.dot(bs[i],bs[i]))
    coef[i]=c; tp-=c*bs[i]
v=coef@Br
dhat=v[:768]/(sigma*SC)
delta=np.round(dhat).astype(np.int64)
print('|delta|_1=',int(np.abs(delta).sum()),'frac zero %.2f'%(delta==0).mean())
S1=S0+delta.reshape(3,256)
def negshift(s,dd):
    o=np.roll(s,dd)
    if dd: o[:dd]=-o[:dd]
    return o
ov=max(int(sum((S1[k]==negshift(s_true[k],dd)).sum() for k in range(3))) for dd in range(N))
print('overlap after nearest-plane CVP:',ov,'/768')
np.save('Sr_fpils.npy',S1)
