import numpy as np, sys
seed=int(sys.argv[1]); nst=int(sys.argv[2]); Tc=sys.argv[3] if len(sys.argv)>3 else ''
exec(open('final_stack.py').read().split('rng=np.random.default_rng')[0].replace("DATA=sys.argv[1] if len(sys.argv)>1 else 'corr4m'",'DATA="corr4m"').replace("TRUTH=sys.argv[2] if len(sys.argv)>2 else 's_true_4m.bin'",'TRUTH="s_true_4m.bin"').replace("Tcorr=sys.argv[3] if len(sys.argv)>3 else None",'Tcorr="%s"'%Tc if Tc else 'Tcorr=None'))
rng=np.random.default_rng(seed)
best=(1e18,None)
for ii in range(nst):
    S0=rng.choice([-2,-1,0,1,2],size=(3,N),p=[.07,.17,.52,.17,.07]).astype(np.int64)
    S=cd(bgs(S0))
    c=xcost(S)
    if c<best[0]: best=(c,S); print('  s%d new best %.4e ov %d'%(seed,c,fam_overlap(S,s_true)),flush=True)
np.save('tour_best_%d.npy'%seed,best[1])
print('FINAL seed%d cost %.4e overlap %d'%(seed,best[0],fam_overlap(best[1],s_true)))
