"""cs_moment_bound.py (v6, round 3) — PRECISE sample complexity of the
second-moment (and higher) channels for CS-128, replacing the v4/v5 rough
estimate "~1e8".

All first moments of z1 die under the chunked latent (E[eps_i] = 0). The
lowest surviving statistics are second moments; the informative ones are
the CROSS-BLOCK products (they carry the relative phases of the s-spectra):

    g_t = z_{t,j,nu} * conj(z_{t,j',nu}),   mu = E[g] = s_hat_j s_hat_j'^* * S_nu
    S_nu = sum_i E|lambda_{c_i,nu}|^2   (computable exactly from c_pub)

We measure Var(g) per (nu, block pair) EMPIRICALLY from the audit dump and
report the N needed for relative precision eps on the product, i.e. the
input to any phase-retrieval + rounding decoder. Higher-even moments have
strictly worse mask-noise scaling (argument in the report); odd moments
vanish identically.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel import niccs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    ".."))


def run(n_use=50_000, eps_rel=0.2):
    data = niccs.cs_ingest_full(os.path.join(ROOT, "tmp", "cs128", "cs_full",
                                             "dump.bin"),
                                n_rows=n_use)
    ground = niccs.cs_ground(os.path.join(ROOT, "tmp", "cs128", "cs_full",
                                          "ground.bin"))
    d = 256
    z1 = data["z1"].astype(np.float64)
    N = z1.shape[0]
    # frequency domain per block
    zh = np.fft.fft(np.pad(z1, ((0, 0), (0, 0), (0, d))), axis=-1)[:, :, 1::2]
    # exact S_nu from the published challenges
    cs = niccs.cs_chunk_spectra(data["c_pub"], (4, 4, 4, 4, 4, 3))
    S = sum(np.abs(lam) ** 2 for lam in cs)          # (N, d)
    Sbar = S.mean(0)                                  # E over challenges

    s_true = ground["s"]
    sh = np.fft.fft(np.pad(s_true.astype(np.float64), ((0, 0), (0, d))),
                    axis=-1)[:, 1::2]

    pairs = [(0, 1), (0, 2), (1, 2)]
    reqs = []
    for (j, jp) in pairs:
        g = zh[:, j, :] * np.conj(zh[:, jp, :])       # (N, d)
        mu_emp = g.mean(0)
        mu_theo = sh[j] * np.conj(sh[jp]) * Sbar
        var_g = g.var(0)
        sig2 = np.abs(mu_theo) ** 2
        ok = sig2 > 0
        req = var_g[ok] / (sig2[ok] * eps_rel ** 2)
        reqs.append(req)
        coh = np.abs(mu_emp) / np.sqrt(var_g / N)
        print(f"pair ({j},{jp}): |mu_theo| median "
              f"{np.median(np.abs(mu_theo)):.3e}, empirical/theoretical "
              f"corr {np.corrcoef(np.abs(mu_emp), np.abs(mu_theo))[0,1]:.3f}, "
              f"coherence@N={N}: median {np.median(coh):.2f}")
    allr = np.concatenate(reqs)
    # phase retrieval needs the products accurate AND enough to round
    # after global-phase resolution; report quantiles
    q = np.percentile(allr, [50, 90, 99])
    print(f"\nrequired_N for {eps_rel:.0%} relative precision on the "
          f"cross-block products:")
    print(f"  median {q[0]:.3g}   p90 {q[1]:.3g}   p99 {q[2]:.3g}")
    # and for the magnitudes-only channel |E|z|^2| (no phase info)
    var_m = (np.abs(zh[:, 0, :]) ** 2).var(0)
    mu_m = np.abs(sh[0]) ** 2 * Sbar
    ok = mu_m > 0
    req_m = var_m[ok] / (mu_m[ok] ** 2 * eps_rel ** 2)
    print(f"magnitude-only channel: median required_N "
          f"{np.percentile(req_m, 50):.3g}")
    return {"pair_median": float(q[0]), "pair_p90": float(q[1]),
            "pair_p99": float(q[2]), "magnitude_median":
            float(np.percentile(req_m, 50))}


if __name__ == "__main__":
    run()
