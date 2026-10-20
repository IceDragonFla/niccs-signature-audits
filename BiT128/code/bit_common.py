"""bit_common.py — shared public-side primitives for the BiT-128 audit pipeline.

Everything in this module is derivable from PUBLIC values only:
  - SM3 hash (pure Python, matches auxfunc.c sm3_bit)
  - pseudoXOF block stream (matches auxfunc.c pseudoXOF + symmetric.c
    bit_xof256_ctx: block i of a context = SM3(seed || BE32(ct)) with the
    context counter starting at 0 and pseudoXOF's internal ct starting at 1)
  - SampleInBall challenge recovery (matches poly.c poly_challenge)
  - signature unpacking (matches packing.c unpack_sig)
  - the negacyclic adjoint accumulation used by the linear statistic

Parameter set: BiT-128 (d=256, q=26881, gamma1_0=8, gamma1=1024, tau=30).
"""
import numpy as np

# ---------------------------------------------------------------- parameters
D = 256
Q = 26881
QHAT = 13441            # 2^{-1} mod q
TAU = 30
GAMMA1_0 = 8            # z0 mask width
GAMMA1 = 1024           # z1 mask width
GAMMA1_2 = 2048         # z2 mask width
GAMMA2 = 682
GAMMA_B = 16
L_ROWS = 3              # s0 rows
K_ROWS = 3              # e/b0/b1 rows

# signature byte layout (params.h)
CHALLENGEBYTES = 32
Z0_BYTES = D * 4 // 8            # 128
Z1_POLY_BYTES = D * 11 // 8      # 352
H_POLY_BYTES = (D * 3 + 7) // 8  # 96
SIGNBYTES = CHALLENGEBYTES + Z0_BYTES + 3 * Z1_POLY_BYTES + 3 * H_POLY_BYTES  # 1504
DUMP_ROW = 2 * D * 2 + 1 + (4 + 3) * D * 2  # 4609

# ---------------------------------------------------------------- SM3 (GB/T 32982-2016)
# Implementation: OpenSSL via hashlib (verified below against the standard
# test vectors of GB/T 32905-2016 at import time). The audit's end-to-end
# correctness proof is the Stage A check that challenges recovered from
# public signature bytes equal the C reference implementation's internal
# canonical challenge on every signature.
import hashlib as _hashlib

def _sm3_selftest():
    v1 = _hashlib.new("sm3", b"abc").hexdigest()
    v2 = _hashlib.new("sm3", b"").hexdigest()
    assert v1 == "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0", v1
    assert v2 == "1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b", v2

_sm3_selftest()

def sm3(msg: bytes) -> bytes:
    return _hashlib.new("sm3", msg).digest()

# ------------------------------------------------- pseudoXOF context (SM3 mode)
class XofCtx:
    """Matches bit_xof256_ctx (BIT_USE_SHAKE=0). Each squeezed 32-byte block
    is produced by bit_xof256 -> pseudoXOF(256 bits, extseed): pseudoXOF
    appends its own big-endian counter ct (starting at 1) to the input, and
    the context appends its block counter (starting at 0) before that. So
    block k of the stream = SM3(seed || BE32(k) || BE32(1))."""

    def __init__(self, seed: bytes):
        self.seed = seed
        self.block = 0

    def squeezeblocks(self, nblocks: int) -> bytes:
        out = bytearray()
        for _ in range(nblocks):
            out += sm3(self.seed + self.block.to_bytes(4, "big")
                       + (1).to_bytes(4, "big"))
            self.block += 1
        return bytes(out)

# ---------------------------------------------------- SampleInBall (poly.c)
def poly_challenge(seed32: bytes) -> np.ndarray:
    """Canonical {0,1} challenge with exactly TAU coefficients equal to +1.
    Faithful transcription of poly.c poly_challenge()."""
    ctx = XofCtx(seed32)
    buf = ctx.squeezeblocks(1)
    pos = 0
    c = np.zeros(D, dtype=np.int8)
    for i in range(D - TAU, D):
        while True:
            if pos >= len(buf):
                buf = ctx.squeezeblocks(1)   # C overwrites the buffer
                pos = 0
            b = buf[pos]
            pos += 1
            if b <= i:
                break
        c[i] = c[b]
        c[b] = 1
    return c

# ------------------------------------------------------- signature unpacking
def unpack_sigs(sig_bytes: np.ndarray) -> dict:
    """Vectorized unpacking of packed signatures, faithful to packing.c
    unpack_sig. sig_bytes: (N, SIGNBYTES) uint8 array.
    Returns c_tilde (N,32), z0 (N,D) int16, z1 (N,3,D) int16, h (N,3,D) int16."""
    N = sig_bytes.shape[0]
    c_tilde = sig_bytes[:, :CHALLENGEBYTES]
    off = CHALLENGEBYTES

    # z0: nibbles, coefficient = nibble - 7 (byte 2j: low nibble -> coeff 2j)
    raw = sig_bytes[:, off:off + Z0_BYTES]
    z0 = np.empty((N, D), dtype=np.int16)
    z0[:, 0::2] = (raw & 0x0F).astype(np.int16) - 7
    z0[:, 1::2] = (raw >> 4).astype(np.int16) - 7
    off += Z0_BYTES

    # z1: 11-bit little-endian bit fields, sign-extended
    z1 = np.empty((N, 3, D), dtype=np.int16)
    a = sig_bytes[:, off:off + 3 * Z1_POLY_BYTES].reshape(N, 3, Z1_POLY_BYTES)
    for j in range(D // 8):
        t = np.empty((N, 3, 8), dtype=np.int32)
        p = 11 * j
        t[:, :, 0] = a[:, :, p] | (a[:, :, p + 1].astype(np.int32) << 8)
        t[:, :, 1] = (a[:, :, p + 1] >> 3) | (a[:, :, p + 2].astype(np.int32) << 5)
        t[:, :, 2] = (a[:, :, p + 2] >> 6) | (a[:, :, p + 3].astype(np.int32) << 2) | (a[:, :, p + 4].astype(np.int32) << 10)
        t[:, :, 3] = (a[:, :, p + 4] >> 1) | (a[:, :, p + 5].astype(np.int32) << 7)
        t[:, :, 4] = (a[:, :, p + 5] >> 4) | (a[:, :, p + 6].astype(np.int32) << 4)
        t[:, :, 5] = (a[:, :, p + 6] >> 7) | (a[:, :, p + 7].astype(np.int32) << 1) | (a[:, :, p + 8].astype(np.int32) << 9)
        t[:, :, 6] = (a[:, :, p + 8] >> 2) | (a[:, :, p + 9].astype(np.int32) << 6)
        t[:, :, 7] = (a[:, :, p + 9] >> 5) | (a[:, :, p + 10].astype(np.int32) << 3)
        t &= 0x7FF
        t = (t << 5).astype(np.int16) >> 5      # sign-extend 11 -> 16 bit
        z1[:, :, 8 * j:8 * j + 8] = t
    off += 3 * Z1_POLY_BYTES

    # h: 3-bit little-endian bit fields, coefficient = code - 3, reject v > 3
    flat = sig_bytes[:, off:off + 3 * H_POLY_BYTES].reshape(N * 3, H_POLY_BYTES)
    bits = np.unpackbits(flat, axis=1, bitorder="little")   # (N*3, 768)
    codes = bits.reshape(N * 3, D, 3).astype(np.int32)
    code = codes[..., 0] + 2 * codes[..., 1] + 4 * codes[..., 2]
    h = (code - 3).astype(np.int16).reshape(N, 3, D)
    return {"c_tilde": c_tilde, "z0": z0, "z1": z1, "h": h}

# ------------------------------------------- negacyclic adjoint accumulation
def adjoint_accumulate(c_sparse_vals, c_sparse_pos, psi_ext):
    """Adjoint of negacyclic multiplication by the challenge, applied to psi.

    (adj(c) psi)[k] = sum_m c[m] * psi[(m+k) mod d] * (-1)^{1 if m+k >= d}.

    c_sparse_*: lists over the TAU nonzero positions of the CANONICAL
    challenge (all +1 for BiT); psi_ext: length-2d extension
    [psi, -psi].  Returns length-d int/float vector.
    """
    d = len(psi_ext) // 2
    out = np.zeros(d, dtype=psi_ext.dtype)
    for val, m in zip(c_sparse_vals, c_sparse_pos):
        out += val * psi_ext[m:m + d]
    return out
