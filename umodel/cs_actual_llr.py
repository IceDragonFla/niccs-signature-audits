"""cs_actual_llr.py (v6, round 1) — ACTUAL-implementation exact LLR battery.

Uses the bit-exact fixed-point port (cs_fixedpoint.chunk_exp_test, validated
240/240 against the reference binary's captured exp-test values) instead of
the intended-model float logs of cs_exact_llr. Also quantifies the
intended-vs-actual acceptance deviation per chunk.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel.cs_fixedpoint import chunk_exp_test
from umodel.cs_exact_llr import (negconv_chunk, CUM, B0, B1, B2, LP0, LP1,
                                 LP2)
from umodel import niccs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    ".."))


def _chunk_states(y0, y1, y2, eps, cdense, s, e, pos):
    """Per-chunk post states and responses for one hypothesis."""
    d = y0.shape[0]
    Y0 = y0.astype(np.int64).copy()
    Y1 = y1.astype(np.int64).copy()
    Y2 = y2.astype(np.int64).copy()
    out = []
    for i in range(6):
        e_i = int(eps[i])
        rs = negconv_chunk(cdense[i:i + 1].astype(np.float64),
                           s.astype(np.float64))[0].astype(np.int64)
        re_ = negconv_chunk(cdense[i:i + 1].astype(np.float64),
                            e.astype(np.float64))[0].astype(np.int64)
        Y0 = Y0 + e_i * cdense[i].astype(np.int64)
        Y1 = Y1 + e_i * rs
        Y2 = Y2 + e_i * re_
        out.append((Y0.copy(), Y1.copy(), Y2.copy(),
                    (e_i * cdense[i].astype(np.int64)),
                    e_i * rs, e_i * re_))
    return out


def loglik_actual_z01(y0, y1, eps, cdense, s, pos):
    """EXACT (z0,z1)-channel log-likelihood: y0/y1 are PINNED by the
    published z0/z1 under each hypothesis (no latent marginalization), so
    this is a rigorous information measure of the directly observed
    channels. Acceptance uses only the fre_0/fre_1 factors and the y0/y1
    norm indicators; the y2-dependent layers (fre_2, z2 quantization) are
    excluded (they require marginalizing the unpinned y2)."""
    ll = float(LP0(y0).sum() + LP1(y1).sum())
    Y0 = y0.astype(np.int64).copy()
    Y1 = y1.astype(np.int64).copy()
    # responses for the s-part only
    rs_all = []
    for i in range(6):
        rs_all.append(negconv_chunk(cdense[i:i + 1].astype(np.float64),
                                    s.astype(np.float64))[0].astype(np.int64))
    for i in range(6):
        e_i = int(eps[i])
        cp = pos[CUM[i]:CUM[i + 1]]
        Y0 = Y0 + e_i * cdense[i].astype(np.int64)
        Y1 = Y1 + e_i * rs_all[i]
        t0 = (e_i * cdense[i].astype(np.int64))[cp]
        # fre_0 at chunk positions, fre_1 on all coords (C/E/F)
        r = _chunk_exp_z01(Y0[cp], Y1, t0, e_i * rs_all[i], cp, i)
        if r["abort"]:
            return -1e9
        ll += r["log2_p"] * math.log(2)
        cum = CUM[i + 1]
        if not (np.abs(Y0).max() <= B0 - (i + 1)
                and np.abs(Y1).max() <= B1 - cum):
            return -1e9
    return ll


def loglik_actual(y0, y1, y2, eps, cdense, s, e, pos):
    """Exact log-likelihood under the ACTUAL implementation (bit-exact
    acceptance + norm indicators + analytic mask prior)."""
    ll = float(LP0(y0).sum() + LP1(y1).sum() + LP2(y2).sum())
    cum = 0
    for i in range(6):
        Y0, Y1, Y2, t0full, t1, t2 = None, None, None, None, None, None
        st = _chunk_states(y0, y1, y2, eps, cdense, s, e, pos)
        Y0, Y1, Y2, t0full, t1, t2 = st[i]
        cp = pos[CUM[i]:CUM[i + 1]]
        r = chunk_exp_test(Y0[cp], Y1, Y2, t0full[cp], t1, t2, cp, i)
        if r["abort"]:
            return -1e9
        ll += r["log2_p"] * math.log(2)
        cum = CUM[i + 1]
        if not (np.abs(Y0).max() <= B0 - (i + 1)
                and np.abs(Y1).max() <= B1 - cum
                and np.abs(Y2).max() <= B2 - cum):
            return -1e9
    return ll


def _chunk_exp_z01(y0c, Y1, t0, t1, cp, i):
    """Fixed-point exp-test with ONLY the fre_0 (chunk positions) and
    fre_1 (all coords) factors — the y2-free part of the acceptance."""
    from umodel.cs_fixedpoint import _approx, FRE0, FRE1, COE0, COE1,         PREMUL, PREMUL_COE, PREMUL_P, PREMUL_P_COE, M0, M64, _shr
    C = E = F = 1
    bC = bE = bF = 0
    for u in range(len(cp)):
        base = int(y0c[u]); tv = int(t0[u])
        for w, arg in ((0, base - tv), (1, base + tv), (2, base)):
            a = abs(int(arg))
            if w == 0:
                C, ex = _approx(C, FRE0[a]); bC += ex + COE0[a]
            elif w == 1:
                E, ex = _approx(E, FRE0[a]); bE += ex + COE0[a]
            else:
                F, ex = _approx(F, FRE0[a]); bF += ex + COE0[a]
    for j in range(Y1.shape[0]):
        for u in range(Y1.shape[1]):
            base = int(Y1[j, u]); tv = int(t1[j, u])
            for w, arg in ((0, base - tv), (1, base + tv), (2, base)):
                a = abs(int(arg))
                if w == 0:
                    C, ex = _approx(C, FRE1[a]); bC += ex + COE1[a]
                elif w == 1:
                    E, ex = _approx(E, FRE1[a]); bE += ex + COE1[a]
                else:
                    F, ex = _approx(F, FRE1[a]); bF += ex + COE1[a]
    diff = bC - bE
    if diff < 0:
        Bx = (C ^ E) & M64
        bC, bE = bE, bC
    else:
        Bx = 0
    Ew = _shr((E ^ Bx) & M64, bC - bE)
    Csum = ((C ^ Bx) + Ew) & M64
    if Csum < Ew:
        bC += 1
        Csum = (1 << 63) | (Csum >> 1)
    Bm, eB = _approx(PREMUL[i], Csum)
    bB = eB + bC + PREMUL_COE[i]
    Am, eA = _approx(PREMUL_P[i], F)
    bA = eA + bF + PREMUL_P_COE[i]
    Am, eA2 = _approx(Am, M0[i])
    bA += eA2
    bA -= bB
    abort = (bA > 0) or (bA == 0 and Am > Bm)
    A_shift = _shr(Am, -bA) if bA < 0 else Am
    p = (A_shift / Bm) if Bm else 1.0
    return {"abort": abort, "p": min(1.0, p) if not abort else 0.0,
            "log2_p": math.log2(min(1.0, p)) if not abort else 0.0}


def battery(dump_path, ground_path, n_use=300, report_every=50):
    data = niccs.cs_ingest_full(dump_path, n_rows=n_use)
    ground = niccs.cs_ground(ground_path)
    d = 256
    s = ground["s"]
    e = ground["e"]
    N = min(n_use, data["z0"].shape[0])
    llr_all = np.zeros(N)
    llr_chunk = np.zeros((N, 6))
    p_true_log2 = []
    for k in range(N):
        cpub = data["c_pub"][k:k + 1]
        eps = data["eps_chunks"][k].astype(np.int64)
        y0 = data["y0"][k]
        y1 = data["y1"][k]
        y2 = data["y2"][k]
        pos = np.nonzero(cpub)[1]
        cdense = np.zeros((6, d), np.int64)
        chunk_id = np.searchsorted(CUM[1:], np.arange(23), side="right")
        for j in range(23):
            cdense[chunk_id[j], pos[j]] = cpub[0, pos[j]]
        ll_true = loglik_actual(y0, y1, y2, eps, cdense, s, e, pos)
        # all-flip
        sum_t0 = sum(int(eps[i]) * cdense[i] for i in range(6))
        st_t = _chunk_states(y0, y1, y2, eps, cdense, s, e, pos)
        sum_t1 = sum(int(eps[i]) * st_t[i][4] for i in range(6))
        sum_t2 = sum(int(eps[i]) * st_t[i][5] for i in range(6))
        ll_flip = loglik_actual(y0 + 2 * sum_t0, y1 + 2 * sum_t1,
                                y2 + 2 * sum_t2, -eps, cdense, s, e, pos)
        llr_all[k] = ll_true - ll_flip
        for i in range(6):
            eps_i = eps.copy()
            eps_i[i] *= -1
            llr_chunk[k, i] = ll_true - loglik_actual(
                y0 - 2 * int(eps[i]) * cdense[i],
                y1 - 2 * int(eps[i]) * st_t[i][4],
                y2 - 2 * int(eps[i]) * st_t[i][5],
                eps_i, cdense, s, e, pos)
        if (k + 1) % report_every == 0:
            print(f"  ... {k + 1}/{N}", file=sys.stderr)
    BIG = 1e8
    killed = np.abs(llr_chunk) > BIG
    cont = llr_all[np.abs(llr_all) < BIG]
    return {
        "n": int(N),
        "allflip_frac_outlier": float((np.abs(llr_all) > BIG).mean()),
        "allflip_mean": float(cont.mean()) if len(cont) else 0.0,
        "allflip_se": float(cont.std() / np.sqrt(len(cont))) if len(cont)
        else 0.0,
        "chunkflip_frac_killed": killed.mean(0).tolist(),
        "chunkflip_mean": np.nanmean(np.where(killed, np.nan,
                                              llr_chunk), 0).tolist(),
        "chunkflip_se": (np.nanstd(np.where(killed, np.nan, llr_chunk), 0)
                         / np.sqrt(N - killed.sum(0))).tolist(),
    }


def battery_z01(dump_path, ground_path, n_use=300, report_every=50):
    """EXACT (z0,z1)-channel information battery (see loglik_actual_z01)."""
    data = niccs.cs_ingest_full(dump_path, n_rows=n_use)
    ground = niccs.cs_ground(ground_path)
    d = 256
    s = ground["s"]
    N = min(n_use, data["z0"].shape[0])
    llr_all = np.zeros(N)
    llr_chunk = np.zeros((N, 6))
    for k in range(N):
        cpub = data["c_pub"][k:k + 1]
        eps = data["eps_chunks"][k].astype(np.int64)
        y0 = data["y0"][k]
        y1 = data["y1"][k]
        pos = np.nonzero(cpub)[1]
        cdense = np.zeros((6, d), np.int64)
        chunk_id = np.searchsorted(CUM[1:], np.arange(23), side="right")
        for j in range(23):
            cdense[chunk_id[j], pos[j]] = cpub[0, pos[j]]
        rs_all = [negconv_chunk(cdense[i:i + 1].astype(np.float64),
                                s.astype(np.float64))[0].astype(np.int64)
                  for i in range(6)]
        ll_true = loglik_actual_z01(y0, y1, eps, cdense, s, pos)
        sum_t0 = sum(int(eps[i]) * cdense[i] for i in range(6))
        sum_t1 = sum(int(eps[i]) * rs_all[i] for i in range(6))
        ll_flip = loglik_actual_z01(y0 + 2 * sum_t0, y1 + 2 * sum_t1,
                                    -eps, cdense, s, pos)
        llr_all[k] = ll_true - ll_flip
        for i in range(6):
            eps_i = eps.copy()
            eps_i[i] *= -1
            llr_chunk[k, i] = ll_true - loglik_actual_z01(
                y0 - 2 * int(eps[i]) * cdense[i],
                y1 - 2 * int(eps[i]) * rs_all[i], eps_i, cdense, s, pos)
        if (k + 1) % report_every == 0:
            print(f"  ... {k + 1}/{N}", file=sys.stderr)
    BIG = 1e8
    killed = np.abs(llr_chunk) > BIG
    cont = llr_all[np.abs(llr_all) < BIG]
    return {
        "n": int(N),
        "allflip_frac_outlier": float((np.abs(llr_all) > BIG).mean()),
        "allflip_mean": float(cont.mean()) if len(cont) else 0.0,
        "allflip_se": float(cont.std() / np.sqrt(len(cont))) if len(cont)
        else 0.0,
        "chunkflip_frac_killed": killed.mean(0).tolist(),
        "chunkflip_mean": np.nanmean(np.where(killed, np.nan,
                                              llr_chunk), 0).tolist(),
        "chunkflip_se": (np.nanstd(np.where(killed, np.nan, llr_chunk), 0)
                         / np.sqrt(N - killed.sum(0))).tolist(),
    }


if __name__ == "__main__":
    full = os.path.join(ROOT, "tmp", "cs128", "cs_full")
    rep = battery(os.path.join(full, "dump.bin"),
                  os.path.join(full, "ground.bin"),
                  n_use=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    import json
    print(json.dumps(rep, indent=1, default=str))
