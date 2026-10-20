"""gap_tests2.py — final localization: Flit-512 boundary sweep; ZEN-512
dual_hybrid with compression-noise-augmented error."""
import sys, os, math

HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/niccs_est/lattice-estimator")

from estimator.schemes import SparseTernary, DiscreteGaussian
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_primal import PrimalUSVP
from estimator.lwe_dual import dual_hybrid
from estimator.reduction import ADPS16

CS = ADPS16()

# 1. Flit-512: find estimator's actual beta_min at d=3570 (per-sign var 7/8)
p = LWEParameters(n=2048, q=3329, Xs=DiscreteGaussian(math.sqrt(7 / 8)),
                  Xe=DiscreteGaussian(math.sqrt(7 / 8)), m=1521)
lo, hi = None, None
for b in range(1885, 1961, 1):
    c = PrimalUSVP.cost_gsa(beta=b, params=p, d=3570, red_cost_model=CS)
    if float(c["rop"]) != float("inf"):
        print(f"  Flit512 DG var7/8 d=3570: first feasible beta = {b} "
              f"(log2rop={math.log2(float(c['rop'])):.1f}, CoreSVP={0.292*b:.1f})", flush=True)
        break
else:
    print("  Flit512: no feasible beta <= 1960 at d=3570", flush=True)
# also scan a few d around 3570 at the claim beta
for d in (3500, 3540, 3570, 3600, 3640):
    for b in (1880, 1885, 1890):
        c = PrimalUSVP.cost_gsa(beta=b, params=p, d=d, red_cost_model=CS)
        ok = float(c["rop"]) != float("inf")
        if ok:
            print(f"  Flit512 feasible at d={d} beta={b}: log2rop={math.log2(float(c['rop'])):.1f}", flush=True)

# 2. ZEN-512 RLWE dual_hybrid with compression noise (769->256: var 0.752)
p2 = LWEParameters(n=2048, q=769, Xs=SparseTernary(p=256, n=2048),
                   Xe=DiscreteGaussian(math.sqrt(0.25 + 0.752)), m=2048)
try:
    r = dual_hybrid(p2, red_cost_model=CS)
    d = {k: float(r[k]) for k in ("beta", "rop") if k in r}
    d["C"] = 0.292 * d["beta"]
    d["log2rop"] = math.log2(d.pop("rop"))
    print(f"ZEN512 dual_hybrid Xe=var1.0 (compression incl.): {d}", flush=True)
except Exception as e:
    print(f"ZEN512 compression test: ERR {str(e)[:120]}", flush=True)
