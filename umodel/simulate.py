"""simulate.py — spec-driven synthetic scheme instances.

Everything here is derived from the spec alone: ring convention, signed or
unsigned challenges, mask widths, K/L rows. This is what makes a transfer
test honest — the simulator has no BiT-specific code paths.

Keygen follows the family shape:
    b = A0 * s + e  (mod q),  b = 16*b1 + b0 (centered 16-decomposition),
    R = b0 - e  in the declared r_box,  A0*s = 16*b1 + R  (mod q).

Transcripts follow the universal channel model:
    z0 = y0 + eps * c                      (labels channel, narrow mask)
    z1 = y1 + eps_z1 * (c * s)             (regression channel, wide mask)
with eps the per-signature shared latent (vulnerable family) or fresh
independent signs (fixed variant, negative control).
"""
import numpy as np

from .linalg import build_relation_matrix, center_modq


def _triangular(rng, shape, g):
    """Centered triangular mask: y in [-(g-1), g-1], P ~ g - |y|."""
    T = np.arange(-(g - 1), g)
    W = (g - np.abs(T)).astype(np.float64)
    cum = np.cumsum(W / W.sum())
    return T[np.searchsorted(cum, rng.random(shape))]


def _mask_sample(rng, shape, mask):
    """Draw from the declared MaskModel (triangular / uniform /
    empirical-table)."""
    if mask is None:
        return np.zeros(shape)
    if mask.kind == "uniform":
        return rng.integers(-(mask.half_width - 1), mask.half_width + 1,
                            shape)
    if mask.kind == "empirical":
        v, p = mask.pmf()
        v = np.asarray(v)
        cum = np.cumsum(np.asarray(p, dtype=np.float64))
        return v[np.searchsorted(cum, rng.random(shape))]
    if mask.kind == "gaussian":
        v, p = mask.pmf()
        v = np.asarray(v)
        cum = np.cumsum(np.asarray(p, dtype=np.float64))
        return v[np.searchsorted(cum, rng.random(shape))]
    return _triangular(rng, shape, mask.half_width)


def _challenges(rng, n, spec):
    """tau distinct positions per signature. The dense canonical challenge
    carries the PUBLISHED values (+1, or +/-1 when signed) — one convention
    with the response convolution and the label LLR. Returns dense (n, d)
    int8, positions (n, tau), values (n, tau)."""
    pos = rng.random((n, spec.d)).argpartition(spec.tau, axis=1)[:, :spec.tau]
    vals = np.ones((n, spec.tau), np.int8)
    if spec.challenge_signed:
        vals = (rng.integers(0, 2, (n, spec.tau)) * 2 - 1).astype(np.int8)
    c = np.zeros((n, spec.d), np.int8)
    np.put_along_axis(c, pos, vals, axis=1)
    return c, pos, vals


def _conv_response(spec, pos, vals, s):
    """(c * s)[k] = sum_t c_t s[(k-p_t)%d] flip(k-p_t); (n, L, d) int32."""
    n = pos.shape[0]
    d, L = spec.d, spec.L
    ar = np.arange(d)
    a = np.zeros((n, L, d), np.int32)
    for t in range(spec.tau):
        p = pos[:, t:t + 1]                       # (n,1)
        idx = (ar[None, :] - p) % d               # (n,d)
        if spec.ring.kind == "negacyclic":
            sgn = np.where(ar[None, :] - p < 0, -1.0, 1.0)
        else:
            sgn = np.ones((n, d))
        a += (vals[:, t:t + 1, None] * sgn[:, None, :].astype(np.int32)
              * s[:, idx].transpose(1, 0, 2))
    return a


def gen_keypair(spec, seed):
    """Family keygen. Returns dict with secrets, public rows, and the
    decoded relation material (Mx, Ainv, b1f)."""
    from .linalg import invert_modq
    rng = np.random.default_rng(seed)
    d, K, L, q = spec.d, spec.K, spec.L, spec.q
    s = rng.integers(-1, 2, (L, d)).astype(np.int64)
    e = rng.integers(-1, 2, (K, d)).astype(np.int64)
    A0 = rng.integers(0, q, (K, L, d)).astype(np.int64)
    Mx = build_relation_matrix(spec, A0)
    t = center_modq(Mx @ s.reshape(-1) + e.reshape(-1), q)     # b, centered
    lo, hi = spec.relation.b0_box
    half = spec.relation.pk_scale // 2
    b0 = ((t + half) % spec.relation.pk_scale) + lo            # centered low
    b1 = (t - b0) // spec.relation.pk_scale                    # exact division
    R = b0.reshape(K, d) - e                                   # in r_box
    # v17: rectangular relations (K != L) are legal for ESTIMATE-ONLY
    # specs; the inverse (needed by the decoder razor) exists only when
    # the relation is square — record None otherwise.
    if Mx.shape[0] == Mx.shape[1]:
        Ainv = invert_modq(Mx, q)
        assert Ainv is not None, "relation matrix singular; resample seed"
    else:
        Ainv = None
    b1m = b1.reshape(K, d) % q
    return {"s": s, "e": e, "b0": b0.reshape(K, d), "b1": b1m,
            "A0": A0, "Mx": Mx, "Ainv": Ainv,
            "b1f": (spec.relation.pk_scale * b1m % q).ravel(), "R": R}


def gen_transcript(spec, key, n, seed, independent_signs=False,
                   chunk=50_000):
    """Public transcript: c (n,d) int8, z0 (n,d) int16, z1 (n,L,d) int16,
    eps (n,) int8 (the shared latent; kept for [SCORING] only)."""
    rng = np.random.default_rng(seed)
    d, L = spec.d, spec.L
    ch_lab = spec.channels.get("labels")
    ch_reg = spec.channels.get("z1-regression")
    g0 = ch_lab.mask.half_width if ch_lab else 0
    g1 = ch_reg.mask.half_width if ch_reg else 0
    c = np.empty((n, d), np.int8)
    dt = np.int32 if max(g0, g1) > 30000 else np.int16
    z0 = np.empty((n, d), dt) if g0 else None
    z1 = np.empty((n, L, d), dt) if g1 else None
    eps = np.empty(n, np.int8)
    s = key["s"].astype(np.int32)
    done = 0
    while done < n:
        m = min(chunk, n - done)
        c_c, pos, vals = _challenges(rng, m, spec)
        e_sign = np.where(rng.random(m) < 0.5, 1, -1).astype(np.int8)
        eps[done:done + m] = e_sign
        c[done:done + m] = c_c
        if g0:
            z0[done:done + m] = _mask_sample(rng, (m, d),
                                             spec.channels["labels"].mask)                 + e_sign[:, None] * c_c
        if g1:
            if spec.shared_latent == "none":
                e_z1 = np.ones(m, np.int8)
            else:
                e_z1 = (np.where(rng.random(m) < 0.5, 1, -1).astype(np.int8)
                        if independent_signs else e_sign)
            resp = _conv_response(spec, pos, vals.astype(np.int32), s)
            z1[done:done + m] = _mask_sample(rng, (m, L, d),
                                             spec.channels["z1-regression"]
                                             .mask)                 + e_z1[:, None, None] * resp
        done += m
    return {"c": c, "z0": z0, "z1": z1, "eps": eps}
