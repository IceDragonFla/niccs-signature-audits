"""gap_localize.py — Verify claimed (m, beta) points against the standard uSVP
GSA condition (as implemented in malb/lattice-estimator lwe_primal.cost_gsa),
and reverse-engineer the noise level implied by a claimed point.

Condition (non-homogeneous, tau = sigma_e):
  lhs = log(sigma_e * sqrt(beta))           [since sigma_e^2(beta-1) + tau^2 = sigma_e^2 * beta]
  rhs = log(delta_beta)*(2*beta - d - 1)
        + (log(tau) + log(xi)*n + log(q)*(d - n - 1)) / d
  success iff lhs <= rhs,  d = n + m + 1, xi = sigma_e/sigma_s if sigma_s < sigma_e else 1
delta_beta = ((pi*beta)^(1/beta) * beta/(2*pi*e))^(1/(2(beta-1)))   [Chen]
"""
from math import log, sqrt, pi, e, exp
from scipy.optimize import brentq  # type: ignore


def delta(beta):
    return ((pi * beta) ** (1 / beta) * beta / (2 * pi * e)) ** (1 / (2 * (beta - 1)))


def margin(n, q, sig_s, sig_e, m, beta, homogeneous=False):
    """lhs - rhs (positive => condition VIOLATED => attack does not succeed)."""
    d = n + m + 1
    dl = log(delta(beta))
    xi = sig_e / sig_s if sig_s < sig_e else 1.0
    if homogeneous:
        lhs = log(sqrt(sig_e**2 * (beta - 1)))
        rhs = dl * (2 * beta - d - 1) + (log(xi) * n + log(q) * (d - n - 1)) / d
    else:
        tau = sig_e
        lhs = log(sqrt(sig_e**2 * (beta - 1) + tau**2))
        rhs = dl * (2 * beta - d - 1) + (log(tau) + log(xi) * n + log(q) * (d - n - 1)) / d
    return lhs - rhs


def implied_sigma_e(n, q, sig_s, m, beta):
    """Solve for sigma_e such that margin == 0 (boundary of success)."""
    f = lambda se: margin(n, q, sig_s, se, m, beta)
    lo, hi = 1e-3, 10.0
    if f(lo) * f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-12)


print("=== NEV-R3: claimed (m, beta) points vs standard GSA condition ===")
# NTRU as RLWE instance (their Sec 8.2): s ~ T1/3 (var 2/3), e ~ T1/3 (var 2/3)
sig = sqrt(2 / 3)
for tag, m, b, ss, se in [
    ("NTRU claim (1431,1998)", 1431, 1998, sig, sig),
    ("RLWE claim (1421,2043)", 1421, 2043, sig, 1.0),  # r~T1/3, e~B2
]:
    mg = margin(2048, 1409, ss, se, m, b)
    print(f"  {tag}: margin(lhs-rhs) = {mg:+.4f}  -> "
          f"{'VIOLATED (attack fails)' if mg > 0 else 'satisfied'}")

# For reference: at what beta does the condition become satisfied at their m?
print("\n  minimal beta satisfying condition at their m (bisection on beta):")
for tag, m, ss, se in [("NTRU m=1431", 1431, sig, sig), ("RLWE m=1421", 1421, sig, 1.0)]:
    lo, hi = 200, 4000
    f = lambda b: margin(2048, 1409, ss, se, m, b)
    if f(lo) < 0:
        print(f"    {tag}: already satisfied at beta=200??")
        continue
    if f(lo) * f(hi) > 0:
        print(f"    {tag}: no beta in [200,4000] satisfies -> infeasible at this m")
        continue
    bb = brentq(f, lo, hi, xtol=1)
    print(f"    {tag}: beta_min ~ {bb:.0f}  (CoreSVP {0.292*bb:.1f} bits)")

print("\n=== Flit-512: implied sigma_e of claimed (d, beta) = (3570, 1885) ===")
# d = n + m + 1 = 3570 -> n=2048, m=1521. Secret f ~ per-sign 7/16 -> var 7/8.
sig_s_impl = sqrt(7 / 8)   # implementation reading
sig_s_spec = sqrt(7 / 16)  # prose reading
se = implied_sigma_e(2048, 3329, sig_s_impl, 1521, 1885)
print(f"  implied sigma_e (given impl secret var 7/8): {se:.4f}  "
      f"(var {se**2:.4f}; spec per-sign var {7/8:.4f}, prose var {7/16:.4f})")
if se:
    print(f"  ratio vs impl var: {se**2/(7/8):.2f}x   vs prose var: {se**2/(7/16):.2f}x")
# also with prose secret
se2 = implied_sigma_e(2048, 3329, sig_s_spec, 1521, 1885)
print(f"  implied sigma_e (given prose secret var 7/16): {se2:.4f} (var {se2**2:.4f})"
      if se2 else "  (no solution with prose secret)")

print("\n=== reference: our reproduced points pass the same check ===")
for tag, n, q, ss, se, m, b in [
    ("Flit-256 impl (1727,915)", 1024, 769, sqrt(5/16), sqrt(1/2), 702, 915),
    ("Flit-128 impl (861,407)", 512, 769, sqrt(1/4), sqrt(5/8), 348, 407),
]:
    mg = margin(n, q, ss, se, m, b)
    print(f"  {tag}: margin = {mg:+.4f} -> {'ok' if mg <= 0 else 'VIOLATED'}")
