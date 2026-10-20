"""assay.py (v4) — data-driven applicability: how much latent information do
the PUBLIC data actually carry?

Motivated by the CS-128 real-instance test: the scheme HAS a per-chunk
bimodal latent (oracle-verified), but the published marginals carry ~0
information about it (measured label accuracy 50.3%, corr 0.004) because
the BLISS-style symmetric acceptance conditions it away. A framework that
only checks the SPEC would promise an attack the data cannot deliver.

The assay measures, from public data only:
  A1 label self-consistency (chunked): split the label evidence per
     signature into two disjoint halves; corr(soft_A, soft_B) vs the
     chance level. Informative labels correlate; random labels do not.
  A2 label-vs-regression coupling: with soft labels, does the aligned
     regression develop split-half structure? (the pipeline's diagnosis)
  A3 effective label information: 0.5*log2(1+corr^2/(1-corr^2)) bits per
     latent, reported with its standard error.

Verdicts:
  "labels_informative"  -> label-dependent routes applicable
  "labels_uninformative"-> latent (if declared) is published-invisible;
                           label-dependent routes REFUSED with this reason
  "no_labels_declared"  -> scheme declares no latent; regression-only
"""
import numpy as np


class LatentInformationAssay:
    def __init__(self, spec):
        self.spec = spec

    # ------------------------------------------------------------ A1
    def label_split_half(self, c, z0, chunk_sizes, lut, n_boot=20, seed=0):
        """Per-chunk soft-label evidence split-half correlation.

        c: (n,d) published challenge; z0: (n,d) label response; lut:
        callable(value, cval) -> per-position LLR; chunk_sizes: the latent
        chunk structure. Returns dict with corr, chance level, and bits."""
        n, d = c.shape
        nzpos = np.nonzero(c)
        rows = nzpos[0]
        order = np.argsort(rows, kind="stable")
        pos = nzpos[1][order]
        sig = rows[order]
        chunk_id = np.searchsorted(np.cumsum(chunk_sizes),
                                   np.arange(len(pos)) % sum(chunk_sizes),
                                   side="right")
        vals = c[sig, pos]
        obs = z0[sig, pos].astype(np.float64)
        llr = lut(obs, vals)                        # per-position evidence
        half = self.spec.d // 2                     # split by coefficient
        m_lo, m_hi = pos < half, pos >= half
        lamA = np.zeros((n, len(chunk_sizes)))
        lamB = np.zeros((n, len(chunk_sizes)))
        np.add.at(lamA, (sig[m_lo], chunk_id[m_lo]), llr[m_lo])
        np.add.at(lamB, (sig[m_hi], chunk_id[m_hi]), llr[m_hi])
        keep = (lamA != 0) | (lamB != 0)
        a, b = lamA[keep].ravel(), lamB[keep].ravel()
        if len(a) < 100:
            return {"verdict": "insufficient_data", "n_pairs": len(a)}
        corr = float(np.corrcoef(a, b)[0, 1])
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_boot):
            p = rng.permutation(len(b))
            null.append(float(np.corrcoef(a, b[p])[0, 1]))
        null = float(np.mean(np.abs(null)))
        sig_flag = abs(corr) > 3 * (1 / np.sqrt(len(a)) + 1e-6)
        bits = (0.5 * np.log2(1 + corr ** 2 / max(1e-12, 1 - corr ** 2))
                if sig_flag else 0.0)
        return {
            "verdict": "labels_informative" if sig_flag
            else "labels_uninformative",
            "corr_split_half": corr, "null_level": null,
            "n_pairs": int(len(a)), "bits_per_latent": float(bits),
            "se": float(1 / np.sqrt(len(a))),
        }

    # ------------------------------------------------------------ A2/A3
    def regression_split_half(self, run_regression, n):
        """run_regression(idx) -> (L,d) estimate on signatures idx."""
        half = n // 2
        shA = run_regression(np.arange(0, half))
        shB = run_regression(np.arange(half, n))
        if shA is None or shB is None:
            return {"verdict": "insufficient_data"}
        corr = float(np.corrcoef(shA.ravel(), shB.ravel())[0, 1])
        return {"verdict": "signal" if corr > 0.3 else "no_signal_at_N",
                "corr": corr, "N": int(n)}
