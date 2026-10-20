"""Diagnosis: does the shared-latent leak exist in this data at all?

Split-half signal test (the audit's permutation-control methodology):
  - Estimate the secret from the first and second halves of the transcript
    (z1 WLS with inferred labels).
  - leak present  <=> corr(shat_A, shat_B) is high while the same estimate
    under label-shuffled controls collapses toward zero.
  - On a FIXED variant (independent per-block signs) the inferred labels do
    not align the z1 block, the statistic cancels, and the correlation stays
    at the control level -> "no attack", regardless of sample count.
"""
import numpy as np


class SharedLatentDiagnosis:
    def __init__(self, spec, label_engine, wls_engine):
        self.spec = spec
        self.labels = label_engine
        self.wls = wls_engine

    def _labels(self, c_can, z0):
        out = self.labels.run(c_can, z0)
        return out[0], out[1]          # tolerate 2- or 3-value APIs

    def _wls(self, c_can, z1, soft):
        out = self.wls.run(c_can, z1, soft)
        return out[0] if isinstance(out, tuple) else out

    def _halves_corr(self, c_can, z1, hard, soft):
        n = c_can.shape[0] // 2
        shA = self._wls(c_can[:n], z1[:n], soft[:n])
        shB = self._wls(c_can[n:], z1[n:], soft[n:])
        ca = np.corrcoef(shA.ravel(), shB.ravel())[0, 1]
        return float(ca)

    def run(self, c_can, z0, z1, n_control=2):
        hard, soft = self._labels(c_can, z0)
        val = self._halves_corr(c_can, z1, hard, soft)
        rng = np.random.default_rng(0)
        ctrls = []
        for _ in range(n_control):
            perm = rng.permutation(len(hard))
            ctrls.append(self._halves_corr(c_can, z1, hard[perm], soft[perm]))
        ctrl = float(np.mean(ctrls))
        leak = bool(val - ctrl > 0.3)
        return {
            "D_split_half_corr": {"value": val, "control": ctrl, "leak": leak},
        }
