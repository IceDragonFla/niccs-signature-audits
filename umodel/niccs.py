"""niccs.py (v4) — adapters for REAL NICCS candidate implementations.

Two instances, both owner-authorized audit datasets generated from the
submitters' reference C code with a dump hook (tmp/cs128, tmp/compass128):

  CS-128        BLISS-style bimodal per-chunk signs. STRUCTURE PRESENT
                (oracle-verified) but the symmetric acceptance conditions
                the published marginals: the label channel carries ~0
                information (measured 50.3% accuracy). The framework must
                REFUSE the label-dependent routes with that reason.
  COMPASS-128   textbook Dilithium clone. NO latent; plain regression is
                mask-limited (Var(y) = 3.58e8 -> required_N ~ 4.8e10).

Both specs are declarative: ring, masks (empirical where measured), chunk
structure, relation. The pipeline runs unchanged on them.
"""
import numpy as np

from .spec import (SchemeSpec, ChannelSpec, PublicKeyRelation,
                   RingStructure, MaskModel)


# ---------------------------------------------------------------- CS-128
def cs128(y0_pmf=None, y1_pmf=None):
    """CS-128 spec. Mask pmfs default to analytic priors; the attack code
    passes measured (public) tables — see cs128_public_masks()."""
    y0 = MaskModel("empirical", 116, y0_pmf if y0_pmf else _cs_y0_default())
    y1 = MaskModel("empirical", 2779, y1_pmf if y1_pmf else _cs_y1_default())
    return SchemeSpec(
        name="CS-128", d=256, q=32257, tau=23, challenge_signed=True,
        ring=RingStructure("negacyclic"), K=3, L=3,
        shared_latent="chunked-signs", chunk_sizes=(4, 4, 4, 4, 4, 3),
        gamma0=116, gamma1=2779,
        relation=PublicKeyRelation(
            form="A0*s + e = t1*2^beta + t0 (mod q); razor: "
                 "center(t1*2^beta - A0*s) = e - t0 in r_box",
            s_alphabet=(-1, 0, 1), r_box=(-17, 17), pk_scale=32,
            b0_box=(-16, 15), derive="clip"),
        channels={
            "labels": ChannelSpec("labels", "labels", "z0, c (chunked)",
                                  "eps_chunks", y0),
            "z1-regression": ChannelSpec(
                "z1-regression", "shift_regression", "z1, c, labels",
                "s", y1, needs_labels=True),
        })


def _cs_y0_default():
    """Analytic prior for the y0 mask (sum of uniforms) — replaced by the
    measured public table in the attack."""
    v = np.arange(-110, 111)
    # Irwin-Hall-like prior; only used when no measured table is supplied
    p = np.exp(-v ** 2 / (2 * 29.2 ** 2))
    return (tuple(v), tuple(p / p.sum()))


def _cs_y1_default():
    v = np.arange(-2555, 2556)
    p = np.exp(-v ** 2 / (2 * 643.0 ** 2))
    return (tuple(v), tuple(p / p.sum()))


def cs128_public_masks(cpub, z0, z1):
    """Measure the CS-128 mask tables from PUBLIC data.

    y0: z0 at non-challenge positions (no sign contamination).
    y1: not directly measurable without s; return the analytic prior with
    the measured variance from public z1 (variance of z1 upper-bounds
    Var(y1); the regression's sigma law uses the measured Var(y1) from
    the oracle arm only in [SCORING] contexts)."""
    nz = cpub != 0
    y0_samples = z0[~nz].astype(np.float64)
    v, cnt = np.unique(y0_samples, return_counts=True)
    p = cnt / cnt.sum()
    return (tuple(v.tolist()), tuple(p.tolist()))


def cs_ingest(dump_path, n_rows=None):
    """Parse a CS-128 dump.bin into arrays. Row = 6192 B:
    cwave 32 | c_pub int8 256 | c' int32 256 | z0 i16 256 | z1 i16 3*256 |
    z2 i8 3*256 | y0 i16 256 [oracle] | y1 i16 3*256 [oracle] |
    bits i8 16 [oracle]."""
    ROW = 6192
    raw = np.memmap(dump_path, dtype=np.uint8, mode="r")
    n = len(raw) // ROW if n_rows is None else n_rows

    def field(off, nby, dt, shape):
        idx = np.arange(n)[:, None] * ROW + off + np.arange(nby)[None, :]
        return raw[idx].copy().reshape(-1).view(dt).reshape(n, *shape)

    out = {
        "cwave": field(0, 32, np.uint8, (32,)),
        "c_pub": field(32, 256, np.int8, (256,)),
        "c_prime": field(288, 1024, np.int32, (256,)),        # [oracle]
        "z0": field(1312, 512, np.int16, (256,)),
        "z1": field(1824, 1536, np.int16, (3, 256)),
        "z2": field(3360, 768, np.int8, (3, 256)),
        "y0": field(4128, 512, np.int16, (256,)),             # [oracle]
        "y1": field(4640, 1536, np.int16, (3, 256)),          # [oracle]
        "bits": field(6176, 16, np.int8, (16,))[:, :6],       # [oracle]
    }
    out["eps_chunks"] = 1 - 2 * out["bits"].astype(np.int64)  # [oracle]
    return out


def cs_ingest_full(dump_path, n_rows=None):
    """Parse the FULL CS-128 dump (with y2 masks + packed sig).
    Row = 9276 B: sig 1548 | cwave 32 | c_pub 256 | c' 1024 | z0 512 |
    z1 1536 | z2 768 | y0 512 | y1 1536 | y2 1536 | bits 16."""
    ROW = 9276
    raw = np.memmap(dump_path, dtype=np.uint8, mode="r")
    n = len(raw) // ROW if n_rows is None else n_rows

    def field(off, nby, dt, shape):
        idx = np.arange(n)[:, None] * ROW + off + np.arange(nby)[None, :]
        return raw[idx].copy().reshape(-1).view(dt).reshape(n, *shape)

    out = {
        "sig": field(0, 1548, np.uint8, (1548,)),
        "cwave": field(1548, 32, np.uint8, (32,)),
        "c_pub": field(1580, 256, np.int8, (256,)),
        "c_prime": field(1836, 1024, np.int32, (256,)),      # [oracle]
        "z0": field(2860, 512, np.int16, (256,)),
        "z1": field(3372, 1536, np.int16, (3, 256)),
        "z2": field(4908, 768, np.int8, (3, 256)),
        "y0": field(5676, 512, np.int16, (256,)),            # [oracle]
        "y1": field(6188, 1536, np.int16, (3, 256)),         # [oracle]
        "y2": field(7724, 1536, np.int16, (3, 256)),         # [oracle]
        "bits": field(9260, 16, np.int8, (16,))[:, :6],      # [oracle]
    }
    out["eps_chunks"] = 1 - 2 * out["bits"].astype(np.int64)
    return out


def cs_ground(ground_path, a_scale=23435):
    """a_scale: Montgomery correction for the dumped plain-domain A.
    The generator dumps A after INTT (raw, factor 2^32); the true plain
    A = raw * (2^32)^{-1} mod q = raw * 23435 (validated via the key
    relation A*s + e = 32*t1 + t0, 768/768)."""
    g = np.fromfile(ground_path, dtype="<i4", count=21 * 256)
    d = 256
    A0 = g[12 * d:21 * d].reshape(3, 3, d) % 32257
    if a_scale != 1:
        A0 = (A0 * a_scale) % 32257
    out = {
        "s": g[:3 * d].reshape(3, d),
        "e": g[3 * d:6 * d].reshape(3, d),
        "t0": g[6 * d:9 * d].reshape(3, d),
        "t1": g[9 * d:12 * d].reshape(3, d),
        "A0": A0.astype(np.int64),
    }
    return out


def cs_chunk_spectra(cpub, chunk_sizes=(4, 4, 4, 4, 4, 3)):
    """Per-chunk dense challenges and their negacyclic spectra."""
    n, d = cpub.shape
    rows_nz, cols_nz = np.nonzero(cpub)
    order = np.argsort(rows_nz, kind="stable")
    rows_nz, cols_nz = rows_nz[order], cols_nz[order]
    vals = cpub[rows_nz, cols_nz].astype(np.int64)
    out = []
    per = sum(chunk_sizes)
    starts = np.cumsum((0,) + chunk_sizes)
    pad = np.pad(cpub.astype(np.float64), ((0, 0), (0, d)))
    Ffull = np.fft.fft(pad, axis=-1)                     # (n, 2d)
    for i in range(len(chunk_sizes)):
        ci = np.zeros((n, d))
        gidx = np.arange(len(cols_nz))
        sel = (gidx % per >= starts[i]) & (gidx % per < starts[i + 1])
        ci[rows_nz[sel], cols_nz[sel]] = vals[sel]
        cip = np.pad(ci, ((0, 0), (0, d)))
        out.append(np.fft.fft(cip, axis=-1)[:, 1::2])
    return out


# ------------------------------------------------------------ COMPASS-128
def shuttle128():
    """SHUTTLE-128 structural spec (v9). Hidden per-challenge-position sign
    latent of a NEW kind: deterministic normalization (flip iff
    <z_partial, v> <= 0 with v a rotated stretched secret) plus secret IRS
    flags — tau=42 hidden bits per signature, claimed distributionally
    erased by the rejection-free IRS (z ~ D_Gaussian exactly). Binary
    all-+1 published challenge; Gaussian wide mask sigma=825/256.
    Channels: none declared — the label channel requires instrumentation
    of the IRS internals (test pending); the regression channel needs the
    hidden signed challenge (unavailable publicly)."""
    from .spec import MaskModel as MM
    return SchemeSpec(
        name="SHUTTLE-128", d=256, q=15361, tau=42, challenge_signed=False,
        ring=RingStructure("negacyclic"), K=3, L=3,
        shared_latent="deterministic-normalization",
        relation=PublicKeyRelation(
            form="A*[1,s,e']^T = q*j (mod 2q) at the secret (structural)",
            s_alphabet=tuple(range(-16, 17)), r_box=(-(2 ** 7), 2 ** 7),
            pk_scale=2, b0_box=(-(2 ** 6), 2 ** 6 - 1), derive="clip"),
        channels={})   # applicability: structure only; test pending


def rhyme_geometry_synthetic():
    """Rhyme GEOMETRY, synthetic transcripts (v11). Gaussian mask
    (sigma=122), d=256, q=3329, tau=30, K=2, L=3, eta=2, signed challenge.
    NOTE: geometry only — Rhyme's mod-2q parity commitment and
    challenge-keyed tables are not modeled; this spec exercises the
    Gaussian-mask channel machinery."""
    from .spec import MaskModel as MM
    return SchemeSpec(
        name="Rhyme-geometry-synthetic", d=256, q=3329, tau=30,
        challenge_signed=True, ring=RingStructure("negacyclic"),
        K=3, L=3, shared_latent="single-sign", gamma1=388,
        relation=PublicKeyRelation(
            form="A0*s + e = t1*2^d + t0 (mod q) [geometry placeholder]",
            s_alphabet=tuple(range(-2, 3)), r_box=(-(2 ** 6), 2 ** 6),
            pk_scale=2 ** 7, b0_box=(-(2 ** 6), 2 ** 6 - 1), derive="clip"),
        channels={
            "z1-regression": ChannelSpec(
                "z1-regression", "shift_regression", "z1, c, labels", "s",
                MM("gaussian", 388, sigma=122.0), needs_labels=True),
        })



def compass128():
    return SchemeSpec(
        name="COMPASS-SIG-128", d=256, q=2081281, tau=30,
        challenge_signed=True, ring=RingStructure("negacyclic"),
        K=3, L=4, shared_latent="none", gamma1=32768,
        relation=PublicKeyRelation(
            form="A0*s1 + s2 = t1*2^13 + t0 (mod q)",
            s_alphabet=(-1, 0, 1), r_box=(-(2 ** 12) - 1, 2 ** 12),
            pk_scale=2 ** 13, b0_box=(-(2 ** 12), 2 ** 12 - 1),
            derive="clip"),
        channels={
            "z1-regression": ChannelSpec(
                "z1-regression", "shift_regression", "z1, c", "s",
                MaskModel("uniform", 32768)),
        })


def compass_ingest(dump_path, n_rows=None):
    """Row = 4384 B: ctilde 32 | cp int8 256 | z i16 4*256 | y i16 4*256."""
    ROW = 4384
    raw = np.memmap(dump_path, dtype=np.uint8, mode="r")
    n = len(raw) // ROW if n_rows is None else n_rows

    def field(off, nby, dt, shape):
        idx = np.arange(n)[:, None] * ROW + off + np.arange(nby)[None, :]
        return raw[idx].copy().reshape(-1).view(dt).reshape(n, *shape)

    return {
        "ctilde": field(0, 32, np.uint8, (32,)),
        "c": field(32, 256, np.int8, (256,)),
        "z1": field(288, 2048, np.int16, (4, 256)),
        "y": field(2336, 2048, np.int16, (4, 256)),       # [oracle]
    }


def compass_ground(ground_path):
    g = np.fromfile(ground_path, dtype="<i4", count=(4 + 3 + 3 + 12) * 256)
    d = 256
    return {
        "s": g[:4 * d].reshape(4, d),
        "s2": g[4 * d:7 * d].reshape(3, d),
        "t0": g[7 * d:10 * d].reshape(3, d),
        "A0": g[10 * d:22 * d].reshape(3, 4, d),
    }
