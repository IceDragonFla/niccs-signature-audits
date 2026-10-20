"""zen_dtru_localize.py — Local gap math (no estimator needed).

1. ZEN-512 NTRU: min over m of beta_min(m) via standard GSA condition;
   compare with implied claim beta = 526.2/0.292 ~ 1802.
2. DTRU-648: implied sigma via Kyber-script-style condition sweep.
"""
from math import log, sqrt, pi, e
from scipy.optimize import brentq  # type: ignore


def delta(b):
    return ((pi * b) ** (1 / b) * b / (2 * pi * e)) ** (1 / (2 * (b - 1)))


def margin(n, q, ss, se, m, b):
    d = n + m + 1
    dl = log(delta(b))
    xi = se / ss if ss < se else 1.0
    lhs = log(sqrt(se**2 * (b - 1) + se**2))
    rhs = dl * (2 * b - d - 1) + (log(se) + log(xi) * n + log(q) * (d - n - 1)) / d
    return lhs - rhs


print("=== ZEN-512 NTRU: standard-GSA minimal beta (optimizing m) ===")
n, q = 2048, 769
var = 3 / 16          # S_3/32 per ZEN convention: Pr[+-1]=3/32 each
ss = se = sqrt(var)
best = None
for m in range(n // 2, n + 1, 8):
    f = lambda bb: margin(n, q, ss, se, m, bb)
    lo, hi = 200, 3400
    try:
        if f(lo) < 0:
            continue
        if f(lo) * f(hi) > 0:
            continue
        bb = brentq(f, lo, hi, xtol=1)
    except Exception:
        continue
    if best is None or bb < best[0]:
        best = (bb, m)
print(f"  min beta ~ {best[0]:.0f} at m={best[1]}  -> CoreSVP {0.292*best[0]:.1f} bits")
print(f"  claim 526.2 implies beta ~ {526.2/0.292:.0f}")

print("\n=== NEV-R1/R2 sanity: our optimizer-found betas pass boundary check ===")
for tag, n_, q_, ss_, se_, m_, b_ in [
    ("NEV-R1 NTRU (486,464)", 512, 1409, sqrt(1.5), sqrt(1.5), 486, 464),
    ("NEV-R2 NTRU (826,975)", 1024, 1409, 1.0, 1.0, 826, 975),
]:
    mg = margin(n_, q_, ss_, se_, m_, b_)
    print(f"  {tag}: margin {mg:+.4f}")
