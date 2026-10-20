"""cs_em_coldstart.py (v12) — EM cold-start quantification for CS-128.

Question: how much side information on s (k known coordinates) does the
label-regression loop need to bootstrap? Iterates
    soft labels (Gaussian model given shat) -> chunked WLS -> shat
and tracks sigma(shat) per iteration for a grid of k. k=0 must stall
(the public cold start); some k* breaks through — the wall's thickness.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel import niccs
from umodel.cs_exact_llr import negconv_chunk, CUM

VAR0, VAR1 = 29.2 ** 2, 661.0 ** 2


def run(dump_path, ground_path, n_use=20000, ks=(0, 50, 200, 768),
        iters=6, seed=7):
    data = niccs.cs_ingest_full(dump_path, n_rows=n_use)
    ground = niccs.cs_ground(ground_path, a_scale=1)
    d, N = 256, data["z0"].shape[0]
    s_true = ground["s"].astype(np.float64)
    z0 = data["z0"].astype(np.float64)
    z1 = data["z1"].astype(np.float64)
    eps = data["eps_chunks"].astype(np.float64)
    # per-chunk responses (fixed)
    cs_lam = niccs.cs_chunk_spectra(data["c_pub"], (4, 4, 4, 4, 4, 3))
    resp_s = []
    cd_all = []
    pos_all = np.nonzero(data["c_pub"])[1].reshape(N, 23)
    cdense = np.zeros((N, 6, d), np.int64)
    cid = np.searchsorted(CUM[1:], np.arange(23), side="right")
    rows = np.repeat(np.arange(N), 23)
    cdense[rows, cid[np.tile(np.arange(23), N)],
           pos_all.ravel()] = data["c_pub"].reshape(N, d)[
        rows, pos_all.ravel()]
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        shat = np.zeros((3, d))
        if k:
            idx = rng.choice(3 * d, k, replace=False)
            flat = s_true.ravel().copy()
            shat = np.zeros(3 * d)
            shat[idx] = flat[idx]
            shat = shat.reshape(3, d)
        traj = [float((shat - s_true).std())]
        for it in range(iters):
            # per-chunk LLR given shat (z0 and z1 parts, mean-field)
            rs = [negconv_chunk(cdense[:, i], shat) for i in range(6)]
            # base: z1 minus other chunks at their CURRENT soft eps
            w = np.full((N, 6), 0.5)     # start neutral each iter? no:
            # use previous soft labels (persist across iters)
            if it == 0:
                soft = np.zeros((N, 6))
            lam = np.zeros((N, 6))
            for i in range(6):
                others = sum(soft[:, j][:, None, None] * rs[j]
                             for j in range(6) if j != i)
                cdi = cdense[:, i]
                lam += 0  # placeholder
                l0 = (((z0 - (others.sum(1) if others.ndim == 3 else 0)
                        + cdi) ** 2).sum(1)
                      - ((z0 - others.sum(1) - cdi) ** 2).sum(1)) / (2 * VAR0)
                r_i = rs[i]
                l1 = (((z1 - others - r_i) ** 2).sum((1, 2))
                      - ((z1 - others + r_i) ** 2).sum((1, 2))) / (2 * VAR1)
                lam[:, i] = np.nan_to_num(l0 + l1)
            soft = np.tanh(np.clip(lam, -30, 30) / 2)
            # chunked WLS with soft weights
            lam_eff = sum(soft[:, i][:, None] * cs_lam[i] for i in range(6))
            zh = np.fft.fft(np.pad(z1, ((0, 0), (0, 0), (0, d))),
                            axis=-1)[:, :, 1::2]
            B = np.einsum("tk,tlk->lk", np.conj(lam_eff), zh)
            Dn = np.einsum("tk->k", np.abs(lam_eff) ** 2)
            Z = np.exp(1j * np.pi * (2 * np.arange(d)[None, :] + 1)
                       * np.arange(d)[:, None] / d)
            shat = np.stack([np.real(Z @ (B[i] / Dn)) / d
                             for i in range(3)])
            traj.append(float((shat - s_true).std()))
        out[k] = traj
        print(f"k={k:4d}: sigma trajectory: "
              + " -> ".join(f"{t:.3f}" for t in traj))
    return out


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "..", ".."))
    full = os.path.join(ROOT, "tmp", "cs128", "cs_full")
    run(os.path.join(full, "dump.bin"), os.path.join(full, "ground.bin"),
        n_use=int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
