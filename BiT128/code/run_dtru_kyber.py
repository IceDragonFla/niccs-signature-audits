"""run_dtru_kyber.py — DTRU estimates using the Kyber security-estimates script.

DTRU Sec 3: 'Kyber security-estimates script, progressive sieving 0.292/0.2563'
This script is pure Python (no Sage needed).
"""
import sys, math, time
sys.path.insert(0, "/root/estimators/kyber")
from MLWE_security import MLWE_optimize_attack, LWE_primal_cost, LWE_dual_cost, svp_classical, svp_quantum
from math import sqrt, log2, pi

# DTRU parameters from Sec 2, Table 1
# NTRU on tricyclotomic ring, CBD eta=1 (sigma = sqrt(1/2))
dtru_params = [
    ("DTRU-648",  648,  3457, 164.0, 144.0),
    ("DTRU-768",  768,  3457, 198.0, 174.0),
    ("DTRU-1024", 1024, 3457, 271.0, 238.0),
    ("DTRU-2048", 2048, 3457, 567.0, 498.0),
]

print("=" * 90)
print("DTRU — Kyber security-estimates script (as referenced in DTRU Sec 3)")
print("Parameters: NTRU on tricyclotomic ring, CBD eta=1, sigma=sqrt(1/2)")
print("=" * 90)

for name, n, q, claim_C, claim_Q in dtru_params:
    # CBD eta=1: sigma = sqrt(eta/2) = sqrt(0.5)
    s = sqrt(0.5)
    max_m = n  # use m = n samples

    t0 = time.time()
    try:
        # Classical (progressive sieving: 0.292 * beta)
        m_c, b_c, c_c = MLWE_optimize_attack(
            q, n, max_m, s,
            cost_attack=LWE_primal_cost,
            cost_svp=svp_classical,
            verbose=False
        )
        # Quantum (progressive sieving: 0.2563 * beta)
        m_q, b_q, c_q = MLWE_optimize_attack(
            q, n, max_m, s,
            cost_attack=LWE_primal_cost,
            cost_svp=svp_quantum,
            verbose=False
        )
        elapsed = time.time() - t0
        # c_c and c_q are log2 costs
        print(f"  {name:<14} n={n:>5} q={q:>5}  "
              f"beta={b_c:>4}  C={c_c:>6.1f} (claim {claim_C:>5.1f}, gap {c_c-claim_C:>+6.1f})  "
              f"Q={c_q:>6.1f} (claim {claim_Q:>5.1f}, gap {c_q-claim_Q:>+6.1f})  "
              f"m={m_c}  [{elapsed:.0f}s]")
    except Exception as e:
        print(f"  {name:<14} ERROR: {type(e).__name__}: {e}")

print("\nNote: svp_classical uses 0.292*beta, svp_quantum uses 0.2563*beta")
print("These are the 'progressive sieving' coefficients DTRU references.")
