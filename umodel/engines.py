"""engines.py — back-compatible wrappers over the v3 universal channels.

The v3 estimators live in channels.py (ring-aware, mask-aware, chunked).
These wrappers preserve the v2 API used by run_demos.py and the reports.
"""
import numpy as np

from .channels import LabelChannel, ShiftRegressionChannel, \
    QuantizedMomentChannel


class LabelEngine(LabelChannel):
    """v2 API: run(c_can, z0) -> (hard, soft)."""

    def run(self, c_can, z0):
        hard, soft, lam = super().run(c_can, z0)
        return hard, soft


class Z1WlsEngine(ShiftRegressionChannel):
    """v2 API: run(c_can, z1, soft) -> shat (L, d)."""

    def run(self, c_can, z1, soft):
        s, info = super().run(c_can, z1, soft)
        return s

    def required_N(self, shat_sigma, current_N, target_sigma=0.5):
        """sigma ~ N^{-1/2} for fresh-mask noise (validated to 4M on BiT)."""
        const = shat_sigma * np.sqrt(current_N)
        return float((const / target_sigma) ** 2)


class HintOperatorEngine(QuantizedMomentChannel):
    """v2 API: run(c_can, h, hard_labels, operator=None) -> (Phi, Fop)."""

    def run(self, c_can, h, hard_labels, operator=None):
        if operator is not None:
            self._Fop = operator
        Phi = self.accumulate(c_can, h, hard_labels)
        Fop = operator if operator is not None else self.build_operator()
        return Phi, Fop
