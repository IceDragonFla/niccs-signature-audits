"""public_decoders.py (v5) — pure-Python decoders for the REAL public
signature bytes of the NICCS instances under test.

Closes the v4 honest-boundary item "the Python signature-byte decoder has
not yet been ported": everything the attack consumes is now derived from
the packed signature bytes (+ public key), no C helper needed.

  CS-128        rANS entropy decoder (223/89/7-symbol tables, L=2^23),
                6-bit low-part unpack, SM3 CTR-DRNG (SP800-90A-style,
                SEEDLEN=55) + SampleInBall port for the challenge.
  COMPASS-128   16-bit LE z unpack (z = gamma1 - packed), shake256
                SampleInBall port for the challenge.

Each decoder is verified bit-exactly against the instrumented dumps in
run_niccs_test.test_public_decoders().
"""
import hashlib
import json
import os
import struct

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- SM3 DRNG
OUTLEN = 32
SEEDLEN = 55


def _sm3(msg: bytes) -> bytes:
    return hashlib.new("sm3", msg).digest()


def _be32(x: int) -> bytes:
    return struct.pack(">I", x)


def _inc_bn(b: bytearray):
    for i in range(len(b) - 1, -1, -1):
        b[i] = (b[i] + 1) & 0xFF
        if b[i]:
            break


def _plus4(a: bytes, b: bytes, c: bytes, d: bytes) -> bytes:
    out = bytearray(SEEDLEN)
    carry = 0
    for i in range(SEEDLEN - 1, -1, -1):
        s = a[i] + b[i] + c[i] + d[i] + carry
        carry = s // 256
        out[i] = s & 0xFF
    return bytes(out)


def _sm3_df(input_string: bytes) -> bytes:
    nblocks = (SEEDLEN + OUTLEN - 1) // OUTLEN
    temp = b""
    for counter in range(1, nblocks + 1):
        temp += _sm3(bytes([counter]) + _be32(8 * SEEDLEN) + input_string)
    return temp[:SEEDLEN]


class Sm3CtrDrng:
    """Faithful port of the NICCS shared drng.c (instantiate + generate)."""

    def __init__(self, nonce: bytes):
        seed = _sm3_df(nonce)
        self.V = bytearray(seed[:SEEDLEN])
        self.C = bytearray(_sm3_df(b"\x00" + bytes(self.V))[:SEEDLEN])
        self.rc = bytearray(SEEDLEN)          # big number, starts 0
        self.rc[-1] = 1                       # instantiate increments rc

    def generate(self, n_bits: int) -> bytes:
        n_bytes = (n_bits + 7) // 8
        m = (n_bits + OUTLEN * 8 - 1) // (OUTLEN * 8)
        data = bytearray(self.V)
        out = bytearray()
        for _ in range(m):
            out += _sm3(bytes(data))
            _inc_bn(data)
        out = out[:n_bytes]
        if n_bytes >= 1 and n_bits % 8:
            out[-1] &= (0xFF << (8 - n_bits % 8)) & 0xFF
        # state update: V <- V + H + C + reseed_counter
        H = bytearray(SEEDLEN)
        H[SEEDLEN - OUTLEN:] = _sm3(b"\x03" + bytes(self.V))
        self.V = bytearray(_plus4(bytes(self.V), bytes(H),
                                  bytes(self.C), bytes(self.rc)))
        _inc_bn(self.rc)
        return bytes(out)


# ------------------------------------------------------------ CS challenge
def cs_sample_in_ball(cwave: bytes, n=256, tau=23) -> np.ndarray:
    """Port of sampling.c SampleInBall (DRNG over cwave)."""
    state = Sm3CtrDrng(cwave)
    buf = state.generate(((tau + 7) >> 3) * 8)
    signs = int.from_bytes(buf[: (tau + 7) >> 3], "little")
    c = np.zeros(n, dtype=np.int64)
    for i in range(n - tau, n):
        while True:
            j = state.generate(8)[0]
            if j <= i:
                break
        c[i] = c[j]
        c[j] = 1 - 2 * (signs & 1)
        signs >>= 1
    return c


# ------------------------------------------------------------------- rANS
RANS_L = 1 << 23


class RansDecoder:
    def __init__(self, table, scale_bits):
        self.scale_bits = scale_bits
        self.mask = (1 << scale_bits) - 1
        M = 1 << scale_bits
        slot = np.empty(M, dtype=np.int32)
        for s, (start, freq) in enumerate(table):
            slot[start:start + freq] = s
        self.slot = slot
        self.start = np.array([t[0] for t in table], dtype=np.int64)
        self.freq = np.array([t[1] for t in table], dtype=np.int64)

    def decode(self, buf: bytes, cnum: int):
        """buf = [4-byte LE initial state][payload...]; returns cnum values
        (symbol - num/2 handled by caller)."""
        x = int.from_bytes(buf[:4], "little")
        assert RANS_L <= x < (RANS_L << 8), "bad rANS state"
        ptr = 4
        end = len(buf)
        out = np.empty(cnum, dtype=np.int64)
        for i in range(cnum):
            s = int(self.slot[x & self.mask])
            out[i] = s
            x = int(self.freq[s]) * (x >> self.scale_bits) \
                + (x & self.mask) - int(self.start[s])
            while x < RANS_L and ptr < end:
                x = (x << 8) | buf[ptr]
                ptr += 1
        assert x == RANS_L, "rANS final state mismatch"
        assert ptr == end, "rANS length mismatch"
        return out


def _load_cs_tables():
    with open(os.path.join(_DIR, "_cs_rans_tables.json")) as f:
        return json.load(f)


_CS_T = _load_cs_tables()
Z0_TABLE = _CS_T["dsyms_z0"]       # 223 symbols, value = s - 111
Z1_TABLE = _CS_T["dsyms_hb_z1"]    # 89 symbols,  value = s - 44
Z2_TABLE = _CS_T["dsyms_z2"]       # 7 symbols,   value = s - 3, M = 2^10

# CS-128 signature layout (params: d=256, k=3, l=3, HBYTES=32)
CS_SIG_LEN = 32 + 240 + 3 * 192 + 545 + 155          # 1548


def cs_unpack_sig(sig: bytes, n=256, l=3, k=3):
    """Full public decode of a CS-128 signature -> (c, z0, z1, z2)."""
    assert len(sig) == CS_SIG_LEN
    cwave = sig[:32]
    c = cs_sample_in_ball(cwave, n=n)
    # z0 block: [cnt LE16][stream] padded to 240
    b = sig[32:32 + 240]
    cnt0 = int.from_bytes(b[:2], "little")
    z0 = RansDecoder(Z0_TABLE, 11).decode(b[2:2 + cnt0], n) - 111
    # z1 low parts: 3 x 192 bytes, 6-bit fields, value = 32 - bits
    off = 272
    low = np.empty((l, n), dtype=np.int64)
    for i in range(l):
        raw = sig[off + i * 192: off + (i + 1) * 192]
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8),
                             bitorder="little").reshape(-1, 6)
        vals = bits @ (1 << np.arange(6))
        low[i] = 32 - vals
    # z1 high parts: [cnt][stream] padded to 545
    b = sig[848:848 + 545]
    cnt1 = int.from_bytes(b[:2], "little")
    hi = RansDecoder(Z1_TABLE, 11).decode(b[2:2 + cnt1], l * n) - 44
    hi = hi.reshape(l, n)
    z1 = (hi << 6) + low
    # z2 block: [cnt][stream] padded to 155 (M = 2^10)
    b = sig[1393:1393 + 155]
    cnt2 = int.from_bytes(b[:2], "little")
    z2 = RansDecoder(Z2_TABLE, 10).decode(b[2:2 + cnt2], k * n) - 3
    return c, z0, z1, z2.reshape(k, n)


# ------------------------------------------------------------ COMPASS-128
def sm3_pseudoxof(msg: bytes, length: int) -> bytes:
    """NICCS pseudoXOF (auxfunc.c): blocks SM3(msg || BE32(ct)), ct = 1..,
    concatenated, prefix-consistent. NOTE: COMPASS's "shake256" API is this
    SM3 construction in disguise (symmetric-shake.c), not Keccak."""
    out = b""
    ct = 1
    while len(out) < length:
        out += _sm3(msg + ct.to_bytes(4, "big"))
        ct += 1
    return out[:length]


def compass_sample_in_ball(ctilde: bytes, n=256, tau=30) -> np.ndarray:
    """Port of poly.c poly_challenge. The XOF behind the shake256 API is
    the SM3 pseudoXOF stream of the absorbed seed; the challenge sampler
    is the standard Dilithium Fisher-Yates."""
    stream = sm3_pseudoxof(ctilde, 4096)
    signs = int.from_bytes(stream[:8], "little")
    pos = 8
    c = np.zeros(n, dtype=np.int64)
    for i in range(n - tau, n):
        while True:
            b = stream[pos]
            pos += 1
            if b <= i:
                break
        c[i] = c[b]
        c[b] = 1 - 2 * (signs & 1)
        signs >>= 1
    return c


def compass_unpack_sig(sig: bytes, n=256, l=4, gamma1=1 << 15):
    """Public decode of a COMPASS-SIG-128 signature -> (c, z)."""
    ctilde = sig[:32]
    c = compass_sample_in_ball(ctilde, n=n)
    z = np.empty((l, n), dtype=np.int64)
    for i in range(l):
        raw = sig[32 + i * 512: 32 + (i + 1) * 512]
        packed = np.frombuffer(raw, dtype="<u2").astype(np.int64)
        z[i] = gamma1 - packed
    return c, z
