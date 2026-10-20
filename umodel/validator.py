"""validator.py — independent candidate verification (no secret access).

v3: ring-aware relation matrix (via linalg), equivalent-key scoring, and
the four-outcome separation kept from v2.

  V1  alphabet check: every s_c coordinate in the secret alphabet
  V2  range check:    every R_c coordinate in the declared r_box
  V3  relation check: A0 * s_c == pk_scale * b1 + R_c (mod q), exactly
  V4  signing check:  candidate produces signatures the ORIGINAL reference
                      verifier accepts (mechanism check, callable hook)

Scoring (four separate outcomes, never conflated):
  score_s0_exact          candidate s_c equals TRUE s_0        [SCORING]
  score_equivalent_key    candidate R equals TRUE R (the equivalence
                          invariant of this family)             [SCORING]
  score_original_key      candidate (s,e,b0) all equal truth    [SCORING]
  score_key_valid         V1-V3 hold (public data only)
"""
import numpy as np

from .linalg import build_relation_matrix, center_modq


class CandidateValidator:
    """All checks use public data; ground truth enters only via explicit
    scoring calls (marked [SCORING])."""

    def __init__(self, spec, A0, b1):
        self.spec = spec
        self.d = spec.d
        self.q = spec.q
        self.Mx = build_relation_matrix(spec, A0)
        self.b1 = b1
        self.s_alphabet = set(spec.relation.s_alphabet)
        self.r_lo, self.r_hi = spec.relation.r_box
        self.scale = spec.relation.pk_scale

    def _cen(self, x):
        return center_modq(x, self.q)

    def _R_of_s(self, s_c):
        s = np.asarray(s_c, dtype=np.int64).ravel()
        b1f = (self.scale * (self.b1 % self.q)).ravel()
        return self._cen(self.Mx @ s - b1f)

    # ------------------------------------------------------------ V1
    def check_alphabet(self, s_c):
        s = np.asarray(s_c).ravel()
        lo, hi = min(self.s_alphabet), max(self.s_alphabet)
        in_set = np.all(np.isin(s, list(self.s_alphabet)))
        bad = int(np.sum(~np.isin(s, list(self.s_alphabet))))
        return bool(in_set), {"n_violations": bad}

    # ------------------------------------------------------------ V2
    def check_R_range(self, s_c, R_c=None):
        R = self._R_of_s(s_c) if R_c is None else np.asarray(R_c).ravel()
        ok = bool(np.all((R >= self.r_lo) & (R <= self.r_hi)))
        bad = int(np.sum((R < self.r_lo) | (R > self.r_hi)))
        return ok, {"R_c": R, "n_violations": bad}

    # ------------------------------------------------------------ V3
    def check_relation(self, s_c, R_c=None):
        s = np.asarray(s_c, dtype=np.int64).ravel()
        b1f = (self.scale * (self.b1 % self.q)).ravel()
        lhs = (self.Mx @ s) % self.q
        if R_c is not None:
            rhs = (b1f + np.asarray(R_c, dtype=np.int64).ravel()) % self.q
            return bool(np.array_equal(lhs, rhs)), {}
        R = self._cen(lhs - b1f)
        return True, {"R_c": R}

    # ------------------------------------------------------------ V4
    def check_signing_ability(self, s_c, e_hat, b0_hat, forge_fn):
        return forge_fn(s_c, e_hat, b0_hat)

    # ------------------------------------------------------------ scoring
    def score(self, s_c, e_hat=None, b0_hat=None,
              s0_true=None, e_true=None, b0_true=None):
        """[SCORING] ground truth used only inside this method."""
        v1_ok, v1d = self.check_alphabet(s_c)
        v2_ok, v2d = self.check_R_range(s_c)
        v3_ok, v3d = self.check_relation(s_c)
        out = {
            "key_valid": bool(v1_ok and v2_ok and v3_ok),   # public only
            "s0_exact": None,
            "equivalent_key_exact": None,
            "original_key_exact": None,
        }
        if s0_true is not None:
            out["s0_exact"] = bool(np.array_equal(
                np.asarray(s_c).reshape(np.asarray(s0_true).shape),
                np.asarray(s0_true)))
        R_c = v2d.get("R_c", v3d.get("R_c"))
        if R_c is not None and s0_true is not None and e_true is not None \
                and b0_true is not None:
            R_true = np.asarray(b0_true).ravel() - np.asarray(e_true).ravel()
            out["equivalent_key_exact"] = bool(np.array_equal(
                R_c.reshape(R_true.shape), R_true))
            if e_hat is not None and b0_hat is not None:
                out["original_key_exact"] = bool(
                    out["s0_exact"]
                    and np.array_equal(np.asarray(e_hat).reshape(
                        np.asarray(e_true).shape), np.asarray(e_true))
                    and np.array_equal(np.asarray(b0_hat).reshape(
                        np.asarray(b0_true).shape), np.asarray(b0_true)))
        return out
