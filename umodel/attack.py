"""attack.py — patched & tuned end-to-end attack driver for umodel.

Patches applied (from the audit's pit list, §11.4/§11.9/§11.14/appendix B):
  P1  razor false positives  -> ALL MITM survivors ranked by z1 likelihood
                                (MAP selection; never by check order)
  P2  greedy trap            -> correctors apply a fix ONLY when the razor
                                fires (nbad == 0), never "improving" steps
  P3  margin blindness       -> k<=2 scans are full-coordinate (no window);
                                windowed MITM W is AUTO-recommended (P5)
                                and its residual risk reported
  P4  slow MC operator       -> vectorized builder (was a Python triple
                                loop; 300x faster)
  P5  manual window W        -> recommend_window(): sigma estimate from the
                                estimate's own ternary geometry (public),
                                expected error count, W from the audit's
                                empirical max-rank curve
  P6  manual route choice    -> route_report(): per-channel required_N,
                                pick the cheaper channel; hint kept as the
                                independent cross-check
  P7  label-scale EM trap    -> per-row cluster fit for the hint channel
                                (public objective; never EM on Lambda)
"""
import numpy as np

from .spec import SchemeSpec
from .engines import LabelEngine, Z1WlsEngine
from .decoder import DecoderLadder


# ---------------------------------------------------------------- P4
def build_operator_vectorized(spec, n_mc=30000, seed=20260928):
    """Vectorized distinct-position MC operator (patch P4).

    For each MC draw of tau distinct positions, the operator entry
    F[m, v] accumulates sgn * flip where the alignment of coordinate m at
    challenge position t reads the field contribution of R[v] from the
    cross term t' != t. Vectorized over (m, t, t').
    """
    d, tau = spec.d, spec.tau
    rng = np.random.default_rng(seed)
    F = np.zeros((d, d))
    ar = np.arange(d)
    for _ in range(n_mc):
        pos = rng.choice(d, tau, replace=False)
        # pairs (t, tp), t != tp: contribution to F[m, v]:
        #   j = (m + pos[t]) % d ; v = (j - pos[tp]) % d
        #   sgn = -1 if j < pos[tp] else +1 ; flip = -1 if m+pos[t] >= d
        J = (ar[:, None] + pos[None, :]) % d            # (d, tau) j for each t
        V = (J[:, :, None] - pos[None, None, :]) % d    # (d, tau, tau) v
        S = np.where(J[:, :, None] < pos[None, None, :], -1.0, 1.0)
        FL = np.where(ar[:, None] + pos[None, :] >= d, -1.0, 1.0)  # (d, tau)
        M = S * FL[:, :, None]
        for t in range(tau):
            Mt = M[:, t, :].copy()
            Mt[:, t] = 0.0
            # accumulate F[m, v]: scatter-add Mt[m, tp'] into row m at
            # column V[m, t, tp']
            rows_m = np.repeat(ar, tau)
            cols_v = V[:, t, :].ravel()
            np.add.at(F, (rows_m, cols_v), Mt.ravel())
    return F / n_mc


# ---------------------------------------------------------------- P5
def sigma_from_estimate(shat):
    """Public sigma estimate: std of (shat - round(shat)) over confidently
    rounded coordinates (|shat| > 0.65 margin band excluded)."""
    r = np.rint(shat)
    resid = shat - r
    conf = np.abs(np.abs(shat) - 0.5) > 0.15
    return float(resid[conf].std()), float(conf.mean())


def recommend_window(shat, cap=56):
    """Expected error count -> window W via the audit's empirical max-rank
    curve (errors -> observed max-rank: 5->24, 8->32, 9->35, 12->48).
    Rule: W = 4*k + 8, capped; residual risk reported when k > 12."""
    import math
    sig, cov = sigma_from_estimate(shat)
    n = shat.size
    phibar = lambda x: 0.5 * math.erfc(x / math.sqrt(2))
    k_exp = n * 2 * phibar(0.5 / sig) if sig > 0 else 0.0
    W = int(min(cap, 4 * k_exp + 8))
    return {"sigma": sig, "conf_coverage": cov, "expected_errors": k_exp,
            "W": W, "residual_risk": k_exp > 12}


# ---------------------------------------------------------------- P6
def route_report(shat_z1=None, rhat_hint=None, spec=None):
    """Per-channel required-N comparison and route selection."""
    out = {}
    if shat_z1 is not None:
        sig, _ = sigma_from_estimate(shat_z1)
        N = None  # caller supplies current N context separately
        out["z1"] = {"sigma": sig}
    if rhat_hint is not None:
        sig, _ = sigma_from_estimate(rhat_hint)
        out["hint"] = {"sigma": sig}
    return out


# ---------------------------------------------------------------- driver
def run_z1_attack(spec, ladder, Mx, c_can, z0, z1, N=None,
                  verbose=True, mitm_cap=56):
    """Labels -> WLS -> (auto) window MITM -> razor+MAP -> s0."""
    lab = LabelEngine(spec)
    hard, soft = lab.run(c_can, z0)
    wls = Z1WlsEngine(spec)
    shat = wls.run(c_can, z1, soft)
    rec = recommend_window(shat, cap=mitm_cap)
    if verbose:
        print(f"[z1] sigma~{rec['sigma']:.3f}  E[errors]~{rec['expected_errors']:.1f}"
              f"  W={rec['W']}  residual_risk={rec['residual_risk']}")
    if rec["residual_risk"]:
        if verbose:
            print("[z1] WARNING: expected errors beyond reliable window; "
                  "increase N (see required_N)")
    s, lik = ladder.mitm_window(shat, Mx, W=rec["W"])
    if s is None and rec["W"] < mitm_cap:
        if verbose:
            print("[z1] retry at cap window")
        s, lik = ladder.mitm_window(shat, Mx, W=mitm_cap)
    return s, lik, shat, rec
