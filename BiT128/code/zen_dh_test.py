"""zen_dh_test.py — ZEN-512 RLWE dual_hybrid: proxy sensitivity test."""
import sys, os

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("EST_PATH", HOME + "/niccs_est/lattice-estimator"))

from estimator.schemes import SparseTernary, DiscreteGaussian
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_dual import dual_hybrid
from estimator.reduction import ADPS16

CS = ADPS16()
for tag, Xs, Xe in [
    ("SparseTernary(256) fixed-weight", SparseTernary(p=256, n=2048), SparseTernary(p=256, n=2048)),
    ("DG(0.5) dense proxy", DiscreteGaussian(0.5), DiscreteGaussian(0.5)),
]:
    p = LWEParameters(n=2048, q=769, Xs=Xs, Xe=Xe, m=2048)
    try:
        r = dual_hybrid(p, red_cost_model=CS)
        d = {k: float(r[k]) for k in ("beta", "rop") if k in r}
        d["C"] = 0.292 * d["beta"]
        print(f"ZEN512 RLWE dual_hybrid {tag}: {d}", flush=True)
    except Exception as e:
        print(f"ZEN512 {tag}: ERR {str(e)[:100]}", flush=True)
