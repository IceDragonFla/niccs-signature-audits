"""pipeline.py (v3) — the universal attack driver.

Contract: at ANY sample size N the pipeline returns either a decoded,
self-certified candidate or a certified analog answer (per-coordinate
interval + required-N prediction). It never raises on "not enough data"
and never fuses channels outside the constrained decoder (composition
law). Ground truth, if supplied, enters only the [SCORING] fields.
"""
import math

import numpy as np

from .spec import check_applicability
from .linalg import build_relation_matrix, invert_modq, center_modq
from .channels import LabelChannel, ShiftRegressionChannel, \
    QuantizedMomentChannel
from .diagnose import SharedLatentDiagnosis
from .engines import Z1WlsEngine
from .attack import sigma_from_estimate, recommend_window


def probit_upper_tail(p):
    """z with 0.5*erfc(z/sqrt(2)) = p (bisection; no scipy)."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sigma_full(shat):
    """Full-residual sigma estimate (public): std(x - rint(x)) over ALL
    coordinates, no confidence-band selection. Unbiased for sigma below
    ~0.4, saturating toward 0.289 above ~0.7 (folding) — always at least
    as large as the band-limited proxy, hence the conservative choice."""
    x = np.asarray(shat).ravel()
    return float(np.std(x - np.rint(x)))


class UniversalPipeline:
    def __init__(self, spec, pub=None):
        self.spec = spec
        self.applicability = check_applicability(spec)
        self.pub = pub
        self.Mx = self.Ainv = self.b1f = None
        if pub is not None:
            self.Mx = build_relation_matrix(spec, pub["A0"])
            self.Ainv = invert_modq(self.Mx, spec.q)
            self.b1f = (spec.relation.pk_scale
                        * (pub["b1"] % spec.q)).ravel()
        self._ladder = None

    # ----------------------------------------------------------------
    def _decoder(self):
        if self._ladder is None:
            from .decoder import DecoderLadder
            self._ladder = DecoderLadder(self.spec, self.Ainv, self.b1f,
                                         self.pub["A0"])
        return self._ladder

    def diagnose(self, c, z0, z1, n_control=2):
        lab = LabelChannel(self.spec)
        wls = Z1WlsEngine(self.spec)
        diag = SharedLatentDiagnosis(self.spec, lab, wls)
        return diag.run(c, z0.astype(np.int64), z1, n_control=n_control)

    # ------------------------------------------------------- estimate
    def estimate(self, c, z0=None, z1=None, h=None):
        """Per-channel estimates; returns (report, estimates dict)."""
        spec = self.spec
        rep, est = {"channels": {}}, {}
        hard = soft = None
        if "labels" in spec.channels and z0 is not None:
            lab = LabelChannel(spec)
            hard, soft, lam = lab.run(c, z0)
            est["hard"], est["soft"] = hard, soft
            rep["channels"]["labels"] = {
                "mean_abs_soft": float(np.abs(soft).mean()),
                "agree_rate_soft": float((soft > 0).mean())}
        if "z1-regression" in spec.channels and z1 is not None:
            reg = ShiftRegressionChannel(spec)
            shat, info = reg.run(c, z1, soft if hard is not None
                                 else np.ones(c.shape[0]))
            sig, cov = sigma_from_estimate(shat)
            # conservative certified sigma: the band-limited proxy
            # underestimates once the confidence band empties
            sig_cert = max(sig, sigma_full(shat))
            # folding saturation: no public residual estimator can see
            # sigma beyond ~0.3 (x -> x - rint(x) wraps); flag it and let
            # the interval_only branch fall back to the alphabet box
            saturated = bool(sigma_full(shat) >= 0.27)
            if saturated:
                # fold-saturation fallback: the public MASK LAW sigma does
                # not fold (Var of the declared/measured mask model, a
                # scheme-level constant) — use it as the certified sigma
                n_sig = c.shape[0]
                ch_reg = spec.channels["z1-regression"]
                var_y = ch_reg.mask.variance()
                # VARIANCE statement under a no-bias assumption: the law
                # describes the estimator's variance when the acceptance
                # conditioning does not bias the regression signal.
                # Acceptance CAN erase signal in principle (truncation
                # counterexample), so scheme-level required_N claims need
                # a real-transcript check, not just this law.
                sig_law = math.sqrt(var_y / (spec.tau * n_sig))
                sig_cert = max(sig_cert, sig_law)
            est["shat"] = shat
            n = c.shape[0]
            rep["channels"]["z1-regression"] = {
                **info, "sigma": sig_cert, "sigma_residual_proxy": sig,
                "conf_coverage": cov, "sigma_saturated": saturated,
                "required_N_sigma0.5": (sig_cert * math.sqrt(n) / 0.5) ** 2
                if sig_cert > 0 else float("inf")}
        if "hint-operator" in spec.channels and h is not None:
            if hard is None:
                rep["channels"]["hint-operator"] = {
                    "error": "moment channel needs labels; none available"}
            else:
                mom = QuantizedMomentChannel(spec)
                Phi = mom.accumulate(c, h, hard)
                rhat = mom.solve(Phi, c.shape[0],
                                 s_eff_grid=np.arange(0.90, 0.97, 0.001))
                sig_r, _ = sigma_from_estimate(rhat)
                est["rhat"] = rhat
                est["Phi"] = Phi
                n = c.shape[0]
                rep["channels"]["hint-operator"] = {
                    "sigma": sig_r,
                    "required_N_sigma0.5": (sig_r * math.sqrt(n) / 0.5) ** 2
                    if sig_r > 0 else float("inf")}
        return rep, est

    # --------------------------------------------------------- decode
    def _razor_box(self, R):
        lo, hi = self.spec.relation.r_box
        return bool(np.all((R >= lo) & (R <= hi)))

    def decode(self, est, mitm_cap=56):
        """R-space ladder (rhat) or s-space ladder (shat); graceful."""
        spec = self.spec
        out = {"status": "no_estimate", "stage": None, "candidate": None}
        if self.pub is None:
            return {**out, "error": "no public material (pub) supplied"}
        cand_s = cand_R = None
        primary = None
        # ---- R-space route (hint estimate): rounding -> pair_scan
        if "rhat" in est:
            primary = "hint"
            ladder = self._decoder()
            R, s = ladder.rounding(est["rhat"])
            stage = "rounding" if R is not None else None
            if R is None:
                R, s = ladder.pair_scan(est["rhat"])
                stage = "pair_scan" if R is not None else None
            if R is not None:
                cand_s, cand_R, out["stage"] = s, R, stage
        # ---- s-space route (regression estimate): rounding_s -> MITM
        if "shat" in est and cand_s is None:
            primary = primary or "z1"
            shf = est["shat"].reshape(-1)
            s1 = np.clip(np.rint(shf),
                         min(spec.relation.s_alphabet),
                         max(spec.relation.s_alphabet)).astype(np.int64)
            R1 = center_modq(self.Mx @ s1 - self.b1f, spec.q)
            if self._razor_box(R1):
                cand_s, cand_R, out["stage"] = s1, R1, "rounding"
            else:
                rec = recommend_window(est["shat"], cap=mitm_cap)
                out["window"] = rec
                # memory-safe window cap: meet-in-the-middle tables are
                # 2^ceil(W/2) x NC int64; keep them under ~512 MB
                NC = 6
                w_half = int(math.log2(512e6 / (NC * 8)))
                w_cap = min(mitm_cap, 2 * w_half)
                ladder = self._decoder()
                s2, lik = ladder.mitm_window(est["shat"].reshape(-1),
                                             self.Mx, W=min(rec["W"], w_cap))
                if s2 is None and rec["W"] > w_cap:
                    out["window_marginal"] = (
                        f"recommended W={rec['W']} exceeds safe cap "
                        f"{w_cap}; tried W={w_cap}")
                if s2 is not None:
                    R2 = center_modq(self.Mx @ s2 - self.b1f, spec.q)
                    cand_s, cand_R = s2, R2
                    out["stage"] = "mitm"
                    out["mitm_lik"] = lik
        if cand_s is None:
            out["status"] = "interval_only"
            if "shat" in est:
                x = est["shat"].reshape(-1)
                sig_c = max(sigma_from_estimate(x.reshape(
                    est["shat"].shape))[0], sigma_full(x))
                lo_b = np.clip(np.rint(x) - 1,
                               min(spec.relation.s_alphabet),
                               max(spec.relation.s_alphabet)).astype(np.int64)
                hi_b = np.clip(np.rint(x) + 1,
                               min(spec.relation.s_alphabet),
                               max(spec.relation.s_alphabet)).astype(np.int64)
                out["alphabet_box"] = (lo_b.reshape(est["shat"].shape),
                                       hi_b.reshape(est["shat"].shape))
                # P(s outside box) per coordinate = 2*Q(1.5/sigma); report
                # the box failure bound so the interval stays honest
                out["box_failure_bound"] = float(
                    x.size * 2 * 0.5 * math.erfc(1.5 / sig_c
                                                 / math.sqrt(2)))
            return out
        out["status"] = "recovered"
        out["candidate"] = cand_s.reshape(spec.L, spec.d)
        out["R"] = cand_R.reshape(spec.K, spec.d)
        return out

    # ------------------------------------------------------------- run
    def run(self, c, z0=None, z1=None, h=None, ground=None, mitm_cap=56,
            diagnose=True):
        N = c.shape[0]
        rep = {"spec": self.spec.name, "N": int(N),
               "applicability": self.applicability}
        if not self.applicability["applicable"]:
            rep["status"] = "refused"
            rep["reasons"] = self.applicability["failures"]
            return rep
        if diagnose and (z0 is not None and z1 is not None):
            rep["diagnosis"] = self.diagnose(c, z0, z1)
        crep, est = self.estimate(c, z0=z0, z1=z1, h=h)
        rep.update(crep)
        # decode-target sigma: expected errors <= 12 at n coordinates
        n_coords = self.spec.L * self.spec.d
        sig_dec = 0.5 / probit_upper_tail(6.0 / n_coords)
        best = None
        for key in ("z1-regression", "hint-operator"):
            if key in rep.get("channels", {}) and "sigma" in rep["channels"][key]:
                s_ = rep["channels"][key]["sigma"]
                if best is None or s_ < best[1]:
                    best = (key, s_)
        rep["primary_channel"] = best[0] if best else None
        dec = self.decode(est, mitm_cap=mitm_cap)
        if dec["status"] == "recovered":
            val = self.validate(dec["candidate"], ground=ground)
            rep["validation"] = val
            e_hat, b0_hat = self.spec.relation.derive_equivalent_key(
                dec["R"].reshape(-1))
            rep["equivalent_key"] = {"e_hat": e_hat, "b0_hat": b0_hat}
        else:
            val = None
        rep["decode"] = {k: v for k, v in dec.items()
                         if k not in ("candidate", "R")}
        if dec["status"] != "recovered" and best is not None:
            sig = best[1]
            rep["decode"]["required_N_decode"] = float(
                (sig * math.sqrt(N) / sig_dec) ** 2)
            rep["decode"]["sigma_decode_target"] = sig_dec
            rep["decode"]["certified_interval_sigma"] = 3.0 * sig
            sat = rep["channels"].get(best[0], {}).get("sigma_saturated",
                                                       False)
            rep["decode"]["sigma_saturated"] = sat
            if sat:
                rep["decode"]["required_N_note"] = (
                    "lower bound: the public sigma proxy is fold-saturated; "
                    "the true requirement is higher")
        if ground is not None and dec["status"] == "recovered":
            sc = val["scoring"]
            rep["scoring"] = {"s0_exact": sc["s0_exact"],
                              "equivalent_key_exact": sc["equivalent_key_exact"]}
            if ground.get("e") is not None:
                sc2 = self._full_score(dec["candidate"], e_hat, b0_hat, ground)
                rep["scoring"]["original_key_exact"] = sc2
        return rep

    def _full_score(self, s_c, e_hat, b0_hat, ground):
        from .validator import CandidateValidator
        v = CandidateValidator(self.spec, self.pub["A0"], self.pub["b1"])
        return v.score(s_c, e_hat=e_hat, b0_hat=b0_hat,
                       s0_true=ground.get("s"), e_true=ground.get("e"),
                       b0_true=ground.get("b0"))["original_key_exact"]

    # ------------------------------------------------------- validate
    def validate(self, s_c, ground=None):
        from .validator import CandidateValidator
        v = CandidateValidator(self.spec, self.pub["A0"], self.pub["b1"])
        v1, _ = v.check_alphabet(s_c)
        v2, _ = v.check_R_range(s_c)
        v3, _ = v.check_relation(s_c)
        out = {"V1_alphabet": v1, "V2_R_range": v2, "V3_relation": v3,
               "key_valid": bool(v1 and v2 and v3)}
        if ground is not None:
            out["scoring"] = v.score(
                s_c, e_hat=None, b0_hat=None,
                s0_true=ground.get("s"), e_true=ground.get("e"),
                b0_true=ground.get("b0"))
        return out
