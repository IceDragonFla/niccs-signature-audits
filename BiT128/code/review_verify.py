"""review_verify.py — Reproduce the reviewer's max_beta diagnostics + full-cost
recomputation for ZEN-512 dual_hybrid (incl. compression case).

Does NOT modify estimator source; raises the cap only in this process.
"""
import sys, os, math

HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/niccs_est/lattice-estimator")

import estimator.lwe_primal as LP
import estimator.lwe_dual as LD
LP.max_beta_global = 2200
LD.max_beta_global = 2200

from estimator.schemes import SparseTernary, DiscreteGaussian, CenteredBinomial
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_primal import primal_usvp
from estimator.lwe_dual import dual_hybrid
from estimator.reduction import ADPS16

CS = ADPS16()

print("== A. default-cap infeasible cases with cap=2200 ==", flush=True)
cases = [
    ("Flit-512 fixed-weight var7/8", 2048, 3329,
     SparseTernary(p=896, n=2048), SparseTernary(p=896, n=2048)),
    ("ZEN-512 NTRU LWE(b=0) var3/16", 2048, 769,
     SparseTernary(p=192, n=2048), SparseTernary(p=192, n=2048)),
    ("NEV-R3 fixed-weight var1366/2048", 2048, 1409,
     SparseTernary(p=683, n=2048), SparseTernary(p=683, n=2048)),
    ("NEV-R3 uniform ternary var2/3 (DG proxy)", 2048, 1409,
     DiscreteGaussian(math.sqrt(2 / 3)), DiscreteGaussian(math.sqrt(2 / 3))),
]
for tag, n, q, xs, xe in cases:
    p = LWEParameters(n=n, q=q, Xs=xs, Xe=xe, m=n)
    try:
        r = primal_usvp(p, red_cost_model=CS)
        beta, d, rop = float(r["beta"]), float(r["d"]), float(r["rop"])
        print(f"  {tag}: beta={beta:.0f} d={d:.0f} log2rop={math.log2(rop):.3f} "
              f"coresvp={0.292*beta:.1f}", flush=True)
    except Exception as e:
        print(f"  {tag}: ERR {str(e)[:90]}", flush=True)

print("== B. ZEN-512 RLWE dual_hybrid FULL COST ==", flush=True)
for tag, xe in [
    ("spec (S1/8 var 1/4)", SparseTernary(p=256, n=2048)),
    ("compression incl. (var 1.002)", DiscreteGaussian(math.sqrt(0.25 + 0.752))),
]:
    p = LWEParameters(n=2048, q=769, Xs=SparseTernary(p=256, n=2048), Xe=xe, m=2048)
    try:
        r = dual_hybrid(p, red_cost_model=CS)
        beta, rop = float(r["beta"]), float(r["rop"])
        extras = {k: float(v) for k, v in r.items()
                  if k in ("m", "d", "mem", "repetitions", "zeta", "h1")}
        print(f"  {tag}: beta={beta:.0f} log2rop={math.log2(rop):.4f} "
              f"(0.292*beta={0.292*beta:.1f}) {extras}", flush=True)
    except Exception as e:
        print(f"  {tag}: ERR {str(e)[:90]}", flush=True)

print("== C. ZEN-128/256 RLWE dual_hybrid full cost (conservative rows) ==", flush=True)
for tag, n, xs, xe in [
    ("ZEN-128 (s=B1+S3/32, e=B2)", 512,
     DiscreteGaussian(math.sqrt(7 / 8)), CenteredBinomial(2)),
    ("ZEN-256 (s=e=B1)", 1024, CenteredBinomial(1), CenteredBinomial(1)),
]:
    p = LWEParameters(n=n, q=769, Xs=xs, Xe=xe, m=n)
    try:
        r = dual_hybrid(p, red_cost_model=CS)
        beta, rop = float(r["beta"]), float(r["rop"])
        print(f"  {tag}: beta={beta:.0f} log2rop={math.log2(rop):.3f}", flush=True)
    except Exception as e:
        print(f"  {tag}: ERR {str(e)[:90]}", flush=True)
