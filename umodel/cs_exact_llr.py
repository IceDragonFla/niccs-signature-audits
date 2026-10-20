"""cs_exact_llr.py (v5) — exact-acceptance-model information bound for the
CS-128 chunked bimodal latent.

Closes the v4 honest boundary "assay detecting no label information does
not prove no public-data attack exists": we evaluate the EXACT per-signature
log-likelihood of the latent eps under the true signing distribution
(mask prior + BLISS-style acceptance + norm indicators, all reconstructed
from sampling.c), and measure the information it carries about eps.

Model (exact, per signature):
  states:  Y_i = y + sum_{i'<=i} eps_i' t_i'   (post-chunk-i partial
           responses), t_i = (c_i, c_i*s, c_i*e) with y = fresh masks;
  norms:   |Y0_i|_inf <= B0-(i+1); |Y1_i|_inf <= B1-cum_i;
           |Y2_i|_inf <= B2-cum_i;
  accept:  logB_i = log2(premul_i) + log2(C_i + E_i)
           logA_i = log2(premul'_i) + log2(F_i) + log2(M0_i)
           C_i = fre_0(|Y0_i - t0| at chunk pos) * fre_1(|Y1_i - t1|)
                 * fre_2(|Y2_i - t2|)
           E_i = same with +t;  F_i = same on |Y_i|;
           P(accept chunk i) = min(1, 2^(logA_i - logB_i)).
  published density of z given eps prop. to
      P_mask(y_eps) * prod_i min(1, 2^(logA_i-logB_i)) * 1[norms]
  where y_eps = z - sum_i eps_i t_i.

The all-flip Bayes factor LLR = loglik(eps) - loglik(-eps) uses, under the
alternative, the implied masks y' = y + 2*sum eps_i t_i (same prior family).
The e-part published value is a quantized projection of the raw response;
we evaluate the raw-response likelihood exactly (oracle arm) and note the
quantization-consistency factor as the one unmodeled (symmetric) term.
"""
import json
import math
import os

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_T = json.load(open(os.path.join(_DIR, "_cs_fre_tables.json")))

LOG_FRE0 = np.log(np.array(_T["fre_0"], dtype=np.float64))
LOG_FRE1 = np.log(np.array(_T["fre_1"], dtype=np.float64))
LOG_FRE2 = np.log(np.array(_T["fre_2"], dtype=np.float64))
# Raw fixed-point semantics: the exp-test compares the EXACT integers
# premul'_raw * F_raw * M0_raw * 2^sum(coe')  vs  premul_raw * (C+E)_raw *
# 2^sum(coe). The per-factor fre coe sums are equal across C/E/F (same
# factor counts), so they cancel in the ratio; premul/M0 enter RAW (no
# 2^64 normalization).
LOG_PREMUL = (np.log(np.array(_T["fre_premul"], dtype=np.float64))
              + np.array(_T["fre_premul_coe"], dtype=np.float64) * math.log(2))
LOG_PREMUL_P = (np.log(np.array(_T["fre_premul_prime"], dtype=np.float64))
                + np.array(_T["fre_premul_prime_coe"], dtype=np.float64)
                * math.log(2))
LOG_M0 = np.log(np.array(_T["M0"], dtype=np.float64))

B0, B1, B2 = 116, 2779, 1962
SIZES = (4, 4, 4, 4, 4, 3)
CUM = np.cumsum((0,) + SIZES)
MAX0, MAX1, MAX2 = len(LOG_FRE0) - 1, len(LOG_FRE1) - 1, len(LOG_FRE2) - 1


def mask_logpmf(a, b, half_width):
    """Exact pmf of (sum of a U[0,2^b-1]) - (sum of a U[0,2^b-1]) as a
    callable x -> log P(x) on integers, zeros outside."""
    base = np.full(2 ** b, 1.0 / 2 ** b)
    conv = np.array([1.0])
    for _ in range(2 * a):
        conv = np.convolve(conv, base)
    n = len(conv)
    vals = np.arange(n) - a * (2 ** b - 1)
    sel = np.abs(vals) <= half_width
    vals, p = vals[sel], conv[sel]
    vmin, vmax = int(vals[0]), int(vals[-1])
    tab = np.full(vmax - vmin + 1, -400.0)
    tab[p > 0] = np.log(p[p > 0])

    def f(x):
        idx = np.clip(np.rint(x).astype(np.int64) - vmin, 0, len(tab) - 1)
        return tab[idx]

    return f


LP0 = mask_logpmf(5, 5, 160)
LP1 = mask_logpmf(10, 9, 5110)      # full support: 10*(2^9-1)
LP2 = mask_logpmf(6, 9, 3066)       # full support: 6*(2^9-1)


def _cl(a, tab, mx):
    return tab[np.clip(np.abs(a).astype(np.int64), 0, mx)]


def negconv_chunk(cd, sblocks):
    """(N, nblocks, d) = c_i * s_j for each block; cd (N,d) chunk values."""
    N = cd.shape[0]
    nb = sblocks.shape[0]
    d = cd.shape[1]
    ar = np.arange(d)
    out = np.zeros((N, nb, d), np.float64)
    for t in np.nonzero(cd.any(0))[0]:
        col = cd[:, int(t)][:, None, None]
        delta = ar - int(t)
        flip = np.where(delta < 0, -1.0, 1.0)
        out += col * (sblocks[:, delta % d] * flip).astype(np.float64)
    return out


def loglik(y0, y1, y2, eps, chunk_pos, resp_s, resp_e, cdense):
    """Exact log-likelihood of one hypothesis given the raw responses.

    y0 (N,d), y1 (N,3,d), y2 (N,3,d): implied fresh masks under the
    hypothesis; eps (N,6); chunk_pos (tau,) positions; resp_s/resp_e
    (N,6,3,d): per-chunk challenge responses for s and e; cdense (N,6,d):
    per-chunk dense challenge (values at chunk positions)."""
    N = y0.shape[0]
    ll = LP0(y0).sum(1) + LP1(y1).sum((1, 2)) + LP2(y2).sum((1, 2))
    Y0 = y0.astype(np.float64)
    Y1 = y1.astype(np.float64).copy()
    Y2 = y2.astype(np.float64).copy()
    rows = np.arange(N)[:, None]
    for i in range(6):
        st, en = CUM[i], CUM[i + 1]
        cp = chunk_pos[st:en]
        e_i = eps[:, i][:, None]
        t0 = cdense[:, i, :][:, cp]                      # (N, sz)
        Y0c = Y0[:, cp]
        Y1 += e_i[:, :, None] * resp_s[:, i]
        Y2 += e_i[:, :, None] * resp_e[:, i]
        t1 = e_i[:, :, None] * resp_s[:, i]              # post-state minus pre
        t2 = e_i[:, :, None] * resp_e[:, i]
        logC = _cl(Y0c - t0, LOG_FRE0, MAX0).sum(1) \
            + _cl(Y1 - t1, LOG_FRE1, MAX1).sum((1, 2)) \
            + _cl(Y2 - t2, LOG_FRE2, MAX2).sum((1, 2))
        logE = _cl(Y0c + t0, LOG_FRE0, MAX0).sum(1) \
            + _cl(Y1 + t1, LOG_FRE1, MAX1).sum((1, 2)) \
            + _cl(Y2 + t2, LOG_FRE2, MAX2).sum((1, 2))
        logF = _cl(Y0c, LOG_FRE0, MAX0).sum(1) \
            + _cl(Y1, LOG_FRE1, MAX1).sum((1, 2)) \
            + _cl(Y2, LOG_FRE2, MAX2).sum((1, 2))
        logC_E = np.logaddexp(logC, logE)
        logB = LOG_PREMUL[i] + logC_E
        logA = LOG_PREMUL_P[i] + logF + LOG_M0[i]
        # Bernoulli acceptance; the hard-abort indicator (A <= B) is
        # dropped: the C's fixed-point C+E alignment shift overflows for
        # exponent gaps >= 64 (documented implementation deviation), so the
        # absolute calibration is only exact on well-aligned chunks
        ll = ll + np.minimum(0.0, logA - logB)
        # norm indicators on the post state
        cum = CUM[i + 1]
        ll = ll + (np.abs(Y0).max(1) <= B0 - (i + 1)) \
            + (np.abs(Y1).max((1, 2)) <= B1 - cum) \
            + (np.abs(Y2).max((1, 2)) <= B2 - cum)
        ll = np.where((np.abs(Y0).max(1) <= B0 - (i + 1))
                      & (np.abs(Y1).max((1, 2)) <= B1 - cum)
                      & (np.abs(Y2).max((1, 2)) <= B2 - cum),
                      ll, -1e9)
    return ll


def exact_llr_battery(data, ground, n_use=5000, batch=250):
    """[SCORING] All-flip and per-chunk-flip exact LLRs under the true eps.

    data: dict from cs_ingest on the FULL dump (with y2); ground: cs_ground.
    Returns a report dict."""
    d = 256
    ar = np.arange(d)
    N = min(n_use, data["z0"].shape[0])
    cpub = data["c_pub"][:N]
    eps = data["eps_chunks"][:N].astype(np.float64)
    y0 = data["y0"][:N].astype(np.float64)
    y1 = data["y1"][:N].astype(np.float64)
    y2 = data["y2"][:N].astype(np.float64)
    s = ground["s"].astype(np.float64)
    e = ground["e"].astype(np.float64)
    pos = np.array(np.nonzero(cpub[0:1]))[2] if False else np.nonzero(cpub)[1].reshape(-1, 23)[0]
    chunk_pos = pos  # same structure every signature (sorted)
    # per-chunk dense challenges
    cdense = np.zeros((N, 6, d))
    posall = np.nonzero(cpub)[1].reshape(N, 23)   # sorted per signature
    chunk_id = np.searchsorted(CUM[1:], np.arange(23), side="right")
    for j in range(23):
        cdense[np.arange(N), chunk_id[j], posall[:, j]] =             cpub.reshape(N, d)[np.arange(N), posall[:, j]]
    llr_all = np.zeros(N)
    llr_chunk = np.zeros((N, 6))
    for st in range(0, N, batch):
        en = min(batch, N - st)
        sl = slice(st, st + en)
        rs = np.stack([negconv_chunk(cdense[sl, i], s) for i in range(6)],
                      axis=1)                      # (en,6,3,d)
        re_ = np.stack([negconv_chunk(cdense[sl, i], e) for i in range(6)],
                       axis=1)
        ll_true = loglik(y0[sl], y1[sl], y2[sl], eps[sl], chunk_pos,
                         rs, re_, cdense[sl])
        # all-flip: implied masks y' = y + 2 * sum eps_i t_i
        t0_full = np.einsum("ni,nd->nd", eps[sl],
                            np.zeros((en, d)))  # placeholder; t0 sum below
        sum_t0 = np.zeros((en, d))
        for i in range(6):
            sum_t0 += eps[sl, i][:, None] * cdense[sl, i]
        sum_ts = np.einsum("ni,nijd->njd", eps[sl], rs)
        sum_te = np.einsum("ni,nijd->njd", eps[sl], re_)
        ll_flip = loglik(y0[sl] + 2 * sum_t0, y1[sl] + 2 * sum_ts,
                         y2[sl] + 2 * sum_te, -eps[sl], chunk_pos,
                         rs, re_, cdense[sl])
        llr_all[sl] = ll_true - ll_flip
        for i in range(6):
            eps_i = eps[sl].copy()
            eps_i[:, i] *= -1
            # implied masks: flip only chunk i
            llr_chunk[sl, i] = ll_true - loglik(
                y0[sl] - 2 * eps[sl, i][:, None] * cdense[sl, i],
                y1[sl] - 2 * eps[sl, i][:, None, None] * rs[:, i],
                y2[sl] - 2 * eps[sl, i][:, None, None] * re_[:, i],
                eps_i, chunk_pos, rs, re_, cdense[sl])
    BIG = 1e8
    killed_all = (llr_all > BIG) | (llr_all < -BIG)
    killed_chunk = np.abs(llr_chunk) > BIG
    cont_all = llr_all[np.abs(llr_all) < BIG]
    rep = {
        "n": int(N),
        "allflip_frac_outlier": float(killed_all.mean()),
        "chunkflip_frac_killed": killed_chunk.mean(0).tolist(),
        "allflip_mean": float(cont_all.mean()) if len(cont_all) else 0.0,
        "allflip_se": (float(cont_all.std() / np.sqrt(len(cont_all)))
                       if len(cont_all) else 0.0),
        "allflip_p_pos": (float((cont_all > 0).mean()) if len(cont_all)
                          else 0.5),
        "chunkflip_mean": (np.nanmean(np.where(killed_chunk, np.nan,
                                                llr_chunk), 0).tolist()),
        "chunkflip_se": (np.nanstd(np.where(killed_chunk, np.nan, llr_chunk),
                                   0)
                         / np.sqrt(N - killed_chunk.sum(0))).tolist(),
    }
    return rep
