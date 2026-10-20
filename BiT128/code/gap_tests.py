"""gap_tests.py — verify claimed boundary points via estimator's own cost_gsa."""
import sys, os, math

HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/niccs_est/lattice-estimator")

from estimator.schemes import SparseTernary, DiscreteGaussian
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_primal import PrimalUSVP
from estimator.reduction import ADPS16

CS = ADPS16()

print("== cost_gsa at claimed boundary points ==", flush=True)
p = LWEParameters(n=2048, q=3329, Xs=SparseTernary(p=896, n=2048),
                  Xe=SparseTernary(p=896, n=2048), m=1521)
for b in (1884, 1885, 1886):
    try:
        c = PrimalUSVP.cost_gsa(beta=b, params=p, d=3570, red_cost_model=CS)
        print(f"  Flit512 SparseTernary d=3570 beta={b}: rop={float(c['rop'])!r}", flush=True)
    except Exception as e:
        print(f"  Flit512 beta={b}: ERR {e}", flush=True)

p2 = LWEParameters(n=2048, q=1409, Xs=DiscreteGaussian(math.sqrt(2 / 3)),
                   Xe=DiscreteGaussian(math.sqrt(2 / 3)), m=1431)
for b in (1997, 1998, 1999):
    try:
        c = PrimalUSVP.cost_gsa(beta=b, params=p2, d=3480, red_cost_model=CS)
        print(f"  NEV-R3 d=3480 beta={b}: rop={float(c['rop'])!r}", flush=True)
    except Exception as e:
        print(f"  NEV-R3 beta={b}: ERR {e}", flush=True)

p3 = LWEParameters(n=2048, q=769, Xs=DiscreteGaussian(math.sqrt(3 / 16)),
                   Xe=DiscreteGaussian(math.sqrt(3 / 16)), m=1192)
for b in (1800, 1801, 1802):
    try:
        c = PrimalUSVP.cost_gsa(beta=b, params=p3, d=3241, red_cost_model=CS)
        print(f"  ZEN512 d=3241 beta={b}: rop={float(c['rop'])!r}", flush=True)
    except Exception as e:
        print(f"  ZEN512 beta={b}: ERR {e}", flush=True)

print("== primal_usvp optimizer with dense-Gaussian proxy, m capped ==", flush=True)
p4 = LWEParameters(n=2048, q=3329, Xs=DiscreteGaussian(math.sqrt(7 / 8)),
                   Xe=DiscreteGaussian(math.sqrt(7 / 8)), m=1521)
try:
    r = PrimalUSVP()(p4, red_cost_model=CS)
    print(f"  Flit512 DG var7/8 m<=1521: {dict(r)}", flush=True)
except Exception as e:
    print(f"  Flit512 m-cap: ERR {str(e)[:120]}", flush=True)

p5 = LWEParameters(n=2048, q=1409, Xs=DiscreteGaussian(math.sqrt(2 / 3)),
                   Xe=DiscreteGaussian(math.sqrt(2 / 3)), m=1431)
try:
    r = PrimalUSVP()(p5, red_cost_model=CS)
    print(f"  NEV-R3 DG var2/3 m<=1431: {dict(r)}", flush=True)
except Exception as e:
    print(f"  NEV-R3 m-cap: ERR {str(e)[:120]}", flush=True)
