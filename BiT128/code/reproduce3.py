"""reproduce3.py — Working version: dict access on Cost objects."""
import sys, time
sys.path.insert(0, "/root/lattice-estimator")
from sage.all import ZZ, RDF
from estimator.schemes import Kyber512, SparseTernary, CenteredBinomial
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_primal import primal_usvp
from estimator.reduction import ADPS16, MATZOV

CORE_SVP = ADPS16()
MATZOV_MODEL = MATZOV()


def make_ntru(n, q, w):
    Xs = SparseTernary(p=w//2, n=n)
    Xe = SparseTernary(p=w//2, n=n)
    return LWEParameters(n=n, q=q, Xs=Xs, Xe=Xe, m=n)

def make_cbd(n, q, eta):
    Xs = CenteredBinomial(eta, n=n)
    Xe = CenteredBinomial(eta, n=n)
    return LWEParameters(n=n, q=q, Xs=Xs, Xe=Xe, m=n)


def run(params, label, cc, cq, model, mname):
    t0 = time.time()
    try:
        res = primal_usvp(params, red_cost_model=model)
        beta = float(res["beta"])
        c = 0.292 * beta
        q_b = 0.257 * beta
        t = time.time() - t0
        print(f"  {label:<14} {mname:<8} β={beta:>4.0f}  C={c:>6.1f}(claim {cc:>5.1f} d={c-cc:>+6.1f})  "
              f"Q={q_b:>6.1f}(claim {cq:>5.1f} d={q_b-cq:>+6.1f})  {t:.0f}s")
    except Exception as e:
        print(f"  {label:<14} {mname:<8} ERROR: {type(e).__name__}: {e}")


# Validation
print("=" * 100)
print("Validation: Kyber-512 (expected CoreSVP ≈ 117)")
print("=" * 100)
run(Kyber512, "Kyber-512", 117.0, 108.0, CORE_SVP, "CoreSVP")
run(Kyber512, "Kyber-512", 117.0, 108.0, MATZOV_MODEL, "MATZOV")

# FLIT
print("\n" + "=" * 100)
print("FLIT — NTRU sparse ternary (per Sec 3)")
print("=" * 100)
for n, q, w, cc, cq in [(512,769,64,134.0,129.7), (1024,769,256,269.5,253.2), (2048,3329,896,529.1,488.5)]:
    p = make_ntru(n, q, w)
    run(p, f"Flit{n}", cc, cq, CORE_SVP, "CoreSVP")
    run(p, f"Flit{n}", cc, cq, MATZOV_MODEL, "MATZOV")

# DTRU
print("\n" + "=" * 100)
print("DTRU — NTRU CBD-1 (per Sec 3)")
print("=" * 100)
for n, q, cc, cq in [(648,3457,164,144), (768,3457,198,174), (1024,3457,271,238), (2048,3457,567,498)]:
    p = make_cbd(n, q, 1)
    run(p, f"DTRU-{n}", cc, cq, CORE_SVP, "CoreSVP")
    run(p, f"DTRU-{n}", cc, cq, MATZOV_MODEL, "MATZOV")

# ZEN
print("\n" + "=" * 100)
print("ZEN — NTRU sparse 1/8 (per Sec 5)")
print("=" * 100)
for n, q, w, cc, cq in [(512,769,64,128.5,116.6), (1024,769,128,257.5,233.7), (2048,769,256,519.9,477.5)]:
    p = make_ntru(n, q, w)
    run(p, f"ZEN-{n}", cc, cq, CORE_SVP, "CoreSVP")
    run(p, f"ZEN-{n}", cc, cq, MATZOV_MODEL, "MATZOV")

# NEV
print("\n" + "=" * 100)
print("NEV — NTRU fixed weight 472/1409 (per Sec 8)")
print("=" * 100)
for n, q, cc, cq in [(1409,4673,135,119), (1409,7681,284,250), (1409,12289,583,513)]:
    p = make_ntru(n, q, 472)
    run(p, f"NEV-{q}", cc, cq, CORE_SVP, "CoreSVP")
    run(p, f"NEV-{q}", cc, cq, MATZOV_MODEL, "MATZOV")

print("\nDone.")
