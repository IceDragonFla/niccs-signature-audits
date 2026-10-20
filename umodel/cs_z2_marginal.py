"""cs_z2_marginal.py (v7) — exact marginal likelihood of the published z2
under each latent hypothesis, closing the y2-marginalization boundary.

Algebra (per coefficient j, hypothesis eps-tilde):
  y0' = z0 - c'        (pinned)      y1' = z1 - c'*s   (pinned)
  ỹ'  = y2' + c'*(e - t0)            (y2' unpinned, prior = mask pmf)
  The HighBits argument of the z2 transform,
      w2q - 2*ỹ'  =  2(A*y1' - y0'*t1hat - c'*(e-t0)) - q*m   (mod 2q),
  is INDEPENDENT of y2' (the y2' terms cancel); call its 2q-representative
  arg(j).  The published z2[j] = center32(HighBits(w2q[j]) - HighBits(arg))
  where w2q = lift2q(2*modq(A*y1' + y2' - y0'*t1hat) - parity(y0')*q) DOES
  depend on y2'.  Hence for each candidate y2'_j = v (support of the mask)
  we evaluate the exact published z2 and accumulate the prior measure of
  the v that reproduce the observation:

      L_z2(eps-tilde) = prod_j  sum_v P2(v) * 1[z2_pred(eps-tilde, v)_j =
                                              z2_obs[j]]

The acceptance's y2-part (fre_2 factors) is not marginalized here; the v6
plug-in showed its asymmetry contribution is small (actual==intended).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel import niccs
from umodel.cs_exact_llr import negconv_chunk, CUM, LP2

Q = 32257
DQ = 2 * Q
ALPHA = 2016
T1HAT = 32          # beta = 5
SIZES = (4, 4, 4, 4, 4, 3)
_SUPPORT = np.arange(-1900, 1901)
_LOGP2 = LP2(_SUPPORT)          # log pmf on the support grid


def high_bits(a):
    """Exact port of bits-hints.c HighBits (alpha=2016, q=32257) for a in
    [0, 2q)."""
    a = np.asarray(a, dtype=np.int64)
    hi = a >> 11
    lo = (a & 2047) + (hi << 5)
    hi = hi - ((1007 - lo) >> 31)          # +1 iff lo >= 1008
    hi = hi & ((hi - 32) >> 31)            # clamp to 0 if hi >= 32
    return hi


def lift2q(x):
    return ((x % DQ) + DQ) % DQ


def modq(x):
    return ((x % Q) + Q) % Q


def z2_loglik(y0p, y1p, cprime, eps, cdense, s, e, t0, t1, A0, z2_obs):
    """log L_z2(hypothesis) for one signature. y0p/y1p: pinned masks under
    the hypothesis; cprime: dense flipped challenge under the hypothesis;
    z2_obs: (3,256) published. Returns log-measure (sum over coords)."""
    d = 256
    k = 3
    ll = 0.0
    p2 = np.exp(_LOGP2)
    v = _SUPPORT
    for r in range(k):
        er = e[r] - t0[r]      # transform uses c'*(e - t0)
        cr_conv = _conv_dense(cprime, er)                    # c'*(e - t0)
        base = (_conv_dense(A0[r][0], y1p[0])
                + _conv_dense(A0[r][1], y1p[1])
                + _conv_dense(A0[r][2], y1p[2])
                - T1HAT * _conv_dense(y0p, t1[r]))           # A*y1' - y0'*t1^
        par = (y0p & 1) * Q if r == 0 else np.zeros(d, np.int64)
        mbase = modq(base[:, None] + v[None, :])              # (d, |sup|)
        w2q = lift2q(2 * mbase - par[:, None])
        wh = high_bits(w2q)                                   # (d, |sup|)
        # exact arg per candidate v (the q*m parity term matters:
        # arg = 2*modq(base+v) - par*q - 2*(v + c'*(e-t0)))
        arg = 2 * mbase - par[:, None] - 2 * (v[None, :]
                                              + cr_conv[:, None])
        arg = ((arg % DQ) + DQ) % DQ
        r_hi = high_bits(arg)
        diff = wh - r_hi
        z2 = diff & 31
        z2 = z2 - (((15 - z2) >> 31) & 32)                   # centered
        match = (z2 == z2_obs[r][:, None])
        meas = (p2[None, :] * match).sum(1)                  # (d,)
        meas = np.maximum(meas, 1e-300)
        ll += float(np.log(meas).sum())
    return ll


def _conv_dense(a, b):
    """Negacyclic convolution of two dense int vectors (length 256)."""
    d = len(a)
    ar = np.arange(d)
    out = np.zeros(d, np.int64)
    nz = np.nonzero(a)[0]
    for t in nz:
        delta = ar - int(t)
        flip = np.where(delta < 0, -1, 1)
        out += int(a[t]) * (b[delta % d] * flip)
    return out


def battery(dump_path, ground_path, n_use=150):
    data = niccs.cs_ingest_full(dump_path, n_rows=n_use)
    # both audit dumps store the RAW INTT output; true plain A needs the
    # Montgomery factor 2^-32 mod q = 23435 (validated on the key relation)
    ground = niccs.cs_ground(ground_path, a_scale=23435)
    d = 256
    s = ground["s"].astype(np.int64)
    e = ground["e"].astype(np.int64)
    t1 = ground["t1"].astype(np.int64)
    A0 = ground["A0"].astype(np.int64)
    N = min(n_use, data["z0"].shape[0])
    llr = np.zeros(N)
    for kk in range(N):
        cpub = data["c_pub"][kk]
        eps = data["eps_chunks"][kk].astype(np.int64)
        z0 = data["z0"][kk].astype(np.int64)
        z1 = data["z1"][kk].astype(np.int64)
        z2o = data["z2"][kk].astype(np.int64)
        pos = np.array(np.nonzero(cpub))[0]
        cdense = np.zeros((6, d), np.int64)
        chunk_id = np.searchsorted(CUM[1:], np.arange(23), side="right")
        for j in range(23):
            cdense[chunk_id[j], pos[j]] = cpub[pos[j]]
        # pinned masks under each hypothesis
        def pinned(epsv):
            cp = np.zeros(d, np.int64)
            for i in range(6):
                cp = cp + epsv[i] * cdense[i]
            rs = [_conv_one(cdense, i, s) for i in range(6)]
            y0p = z0 - cp
            y1p = z1 - sum(epsv[i] * rs[i] for i in range(6))
            return y0p, y1p, cp
        y0t, y1t, cpt = pinned(eps)
        y0f, y1f, cpf = pinned(-eps)
        t0 = ground['t0'].astype(np.int64)
        ll_true = z2_loglik(y0t, y1t, cpt, eps, cdense, s, e, t0, t1, A0, z2o)
        ll_flip = z2_loglik(y0f, y1f, cpf, -eps, cdense, s, e, t0, t1, A0, z2o)
        llr[kk] = ll_true - ll_flip
        if (kk + 1) % 25 == 0:
            print(f"  ... {kk + 1}/{N}", file=sys.stderr)
    BIG = 1e8
    cont = llr[np.abs(llr) < BIG]
    # METRIC (v17): this is the MEAN LOG-LIKELIHOOD RATIO under the true
    # hypothesis (a KL-flavored quantity), NOT the mutual information:
    # I(eps; T | s, pk) <= H(eps) <= 6 ln 2 ~ 4.16 nats per signature.
    # A mean LLR above that cap indicates heavy-tailed per-signature
    # behavior (most of the mass in rare near-deterministic reveals).
    return {"n": int(N),
            "frac_outlier": float((np.abs(llr) > BIG).mean()),
            "mean_llr": float(cont.mean()) if len(cont) else 0.0,
            "mi_cap_nats": 6 * math.log(2),
            "se": float(cont.std() / np.sqrt(len(cont))) if len(cont)
            else 0.0,
            "p_pos": float((cont > 0).mean()) if len(cont) else 0.5}


def battery_chunkwise(dump_path, ground_path, n_use=40):
    """Per-chunk z2-marginal LLRs (given-key information structure)."""
    data = niccs.cs_ingest_full(dump_path, n_rows=n_use)
    ground = niccs.cs_ground(ground_path, a_scale=1)
    d = 256
    s = ground["s"].astype(np.int64)
    e = ground["e"].astype(np.int64)
    t0g = ground["t0"].astype(np.int64)
    t1 = ground["t1"].astype(np.int64)
    A0 = ground["A0"].astype(np.int64)
    N = min(n_use, data["z0"].shape[0])
    llr_c = np.zeros((N, 6))
    for kk in range(N):
        cpub = data["c_pub"][kk]
        eps = data["eps_chunks"][kk].astype(np.int64)
        z0 = data["z0"][kk].astype(np.int64)
        z1 = data["z1"][kk].astype(np.int64)
        z2o = data["z2"][kk].astype(np.int64)
        pos = np.nonzero(cpub)[0]
        cdense = np.zeros((6, d), np.int64)
        cid = np.searchsorted(CUM[1:], np.arange(23), side="right")
        for j in range(23):
            cdense[cid[j], pos[j]] = cpub[pos[j]]
        rs = [_conv_one(cdense, i, s) for i in range(6)]

        def pinned(epsv):
            cp = sum(epsv[i] * cdense[i] for i in range(6))
            y0p = z0 - cp
            y1p = z1 - sum(epsv[i] * rs[i] for i in range(6))
            return y0p, y1p, cp

        y0t, y1t, cpt = pinned(eps)
        ll_t = z2_loglik(y0t, y1t, cpt, eps, cdense, s, e, t0g, t1, A0, z2o)
        for i in range(6):
            eps_i = eps.copy()
            eps_i[i] *= -1
            y0i, y1i, cpi = pinned(eps_i)
            llr_c[kk, i] = ll_t - z2_loglik(y0i, y1i, cpi, eps_i, cdense,
                                            s, e, t0g, t1, A0, z2o)
    med = np.median(llr_c, 0)
    return {"n": int(N),
            "chunk_median": med.tolist(),
            "chunk_p75": np.percentile(llr_c, 75, 0).tolist(),
            "frac_pos": (llr_c > 0).mean(0).tolist()}


def _conv_one(cdense, i, s):
    return negconv_chunk(cdense[i:i + 1].astype(np.float64),
                         s.astype(np.float64))[0].astype(np.int64)


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "..", ".."))
    full = os.path.join(ROOT, "tmp", "cs128", "cs_full")
    rep = battery(os.path.join(full, "dump.bin"),
                  os.path.join(full, "ground.bin"),
                  n_use=int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    import json
    print(json.dumps(rep, indent=1, default=str))
