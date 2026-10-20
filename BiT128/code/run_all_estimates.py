"""run_all.py — Run security estimates for all 4 NICCS KEM algorithms using
the exact tools they reference in their specifications.

FLIT/ZEN/NEV: malb/lattice-estimator with Sage
DTRU: pq-crystals/security-estimates (Kyber script, pure Python)
"""
import sys, time
sys.path.insert(0, "/root/estimators/estimator")

from sage.all import ZZ
from estimator.schemes import Kyber512, SparseTernary, CenteredBinomial
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_primal import primal_usvp
from estimator.lwe_dual import dual as dual_attack
from estimator.reduction import ADPS16, MATZOV

CORE_SVP = ADPS16()
MATZOV_M = MATZOV()


def make_sparse_ternary(n, q, weight):
    Xs = SparseTernary(p=weight // 2, n=n)
    Xe = SparseTernary(p=weight // 2, n=n)
    return LWEParameters(n=n, q=q, Xs=Xs, Xe=Xe, m=n)


def make_cbd(n, q, eta):
    Xs = CenteredBinomial(eta, n=n)
    Xe = CenteredBinomial(eta, n=n)
    return LWEParameters(n=n, q=q, Xs=Xs, Xe=Xe, m=n)


def run_lattice_estimator(params, label, claim_C, claim_Q):
    """Run both primal and dual attacks, report the minimum."""
    results = {}
    for attack_name in ["primal_usvp", "dual"]:
        for model_name, model in [("CoreSVP", CORE_SVP), ("MATZOV", MATZOV_M)]:
            key = f"{attack_name}_{model_name}"
            t0 = time.time()
            try:
                if attack_name == "primal_usvp":
                    res = primal_usvp(params, red_cost_model=model)
                else:
                    res = dual_attack(params, red_cost_model=model)
                beta = float(res["beta"])
                classical = 0.292 * beta
                quantum = 0.257 * beta
                elapsed = time.time() - t0
                results[key] = {"beta": beta, "C": classical, "Q": quantum, "t": elapsed}
            except Exception as e:
                results[key] = {"error": str(e), "t": time.time() - t0}

    # Report best (minimum) classical across all attacks
    valid = [(k, v) for k, v in results.items() if "beta" in v]
    if valid:
        best_k = min(valid, key=lambda x: x[1]["C"])
        bk, bv = best_k
        print(f"  {label:<14} claim_C={claim_C:>5.1f}  "
              f"best={bv['C']:>6.1f} ({bk})  gap={bv['C']-claim_C:>+6.1f}  "
              f"beta={bv['beta']:>4.0f}  [{bv['t']:.0f}s]")
        # Also show individual attack results
        for k in sorted(results.keys()):
            v = results[k]
            if "beta" in k and "beta" in v:
                print(f"    {k:<24} beta={v['beta']:>4.0f}  C={v['C']:>6.1f}  Q={v['Q']:>6.1f}")
            elif "error" in v:
                print(f"    {k:<24} ERR: {v['error'][:40]}")
    else:
        print(f"  {label:<14} ALL ATTACKS FAILED")

    return results


print("=" * 100)
print("NICCS Security Estimation — Using Each Algorithm's Referenced Tool")
print("lattice-estimator (malb) for FLIT/ZEN/NEV; Kyber script for DTRU")
print("=" * 100)

# ---- Validation ----
print("\nValidation: Kyber-512")
print("-" * 100)
p = Kyber512
run_lattice_estimator(p, "Kyber-512", 117.0, 108.0)

# ---- FLIT (lattice-estimator, sparse ternary) ----
print("\nFLIT — Sec 3: 'lattice-estimator commit 6019056, MATZOV primary'")
print("  We model: NTRU -> LWE(b=0), sparse ternary, m=n")
print("-" * 100)
for name, n, q, w, cc, cq in [
    ("Flit-128", 512, 769, 64, 134.0, 129.7),
    ("Flit-256", 1024, 769, 256, 269.5, 253.2),
    ("Flit-512", 2048, 3329, 896, 529.1, 488.5),
]:
    p = make_sparse_ternary(n, q, w)
    run_lattice_estimator(p, name, cc, cq)

# ---- ZEN (lattice-estimator, sparse ternary) ----
print("\nZEN — Sec 5: 'lattice-estimator commit 3e48ef4, CoreSVP conservative'")
print("  We model: NTRU -> LWE, sparse 1/8, m=n")
print("-" * 100)
for name, n, q, w, cc, cq in [
    ("ZEN-128", 512, 769, 64, 128.5, 116.6),
    ("ZEN-256", 1024, 769, 128, 257.5, 233.7),
    ("ZEN-512", 2048, 769, 256, 519.9, 477.5),
]:
    p = make_sparse_ternary(n, q, w)
    run_lattice_estimator(p, name, cc, cq)

# ---- NEV (lattice-estimator, fixed weight) ----
print("\nNEV — Sec 8: 'lattice-estimator, CoreSVP primary'")
print("  We model: NTRU -> LWE, fixed weight 472/1409, m=n")
print("-" * 100)
for name, n, q, w, cc, cq in [
    ("NEV-R1", 1409, 4673, 472, 135.0, 119.0),
    ("NEV-R2", 1409, 7681, 472, 284.0, 250.0),
    ("NEV-R3", 1409, 12289, 472, 583.0, 513.0),
]:
    p = make_sparse_ternary(n, q, w)
    run_lattice_estimator(p, name, cc, cq)

print("\n" + "=" * 100)
print("DTRU uses the Kyber security-estimates script (pure Python).")
print("Running separately with kyber script...")
print("=" * 100)
