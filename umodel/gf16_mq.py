"""gf16_mq.py (v20, Milestone C) — the SECOND algebraic backend: GF(16)
multivariate-quadratic analysis for Origami-128 (bit-exact port of the
reference implementation's public-map semantics), plus the structural
forgery test the port enabled.

Port pieces (validated against the reference C):
  SM3 (OpenSSL), pseudoXOF counter-KDF, buffer-SHAKE (SM3 stand-in),
  GF(16) tables, nibble packing, derive_bytes, gf_stream,
  Fisher-Yates permutation, residual schedule, evaluate_public_map,
  hash_to_field, pseudohash_512.

Structural finding under test: the public map contains ONLY cross terms
y_non * y_oil in the permutation coordinates defined by the PUBLIC
pk_seed — so the oil blocks are publicly computable and the signature
equation is LINEAR in the oil variables once the non-oil prefix is fixed.
If confirmed by the unmodified reference verifier, Origami-128
signatures are forgeable WITHOUT the secret key via three 32x32 and one
8x8 GF(16) linear solve.
"""
import hashlib
import os
import struct

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- SM3 KDF
def _sm3(msg: bytes) -> bytes:
    return hashlib.new("sm3", msg).digest()


def pseudo_xof(msg: bytes, outlen: int) -> bytes:
    out = b""
    ct = 1
    while len(out) < outlen:
        out += _sm3(msg + struct.pack(">I", ct))
        ct += 1
    return out[:outlen]


def pseudohash_512(msg: bytes, hmac_key: bytes) -> bytes:
    m = msg + b"\x02\x00"
    h1 = _sm3(m)
    # HMAC-SM3 with fixed key: ipad/opad 64-byte block
    k = hmac_key + b"\x00" * (64 - len(hmac_key))
    k1 = _sm3(bytes(x ^ 0x5C for x in k) + h1)  # note: C uses k1=HMAC(key,m)
    # C: k1 = HMAC(key, 0x02 0x00 || m); h1 = SM3(m||0x02 0x00);
    #    h2 = SM3(k1 || h1); out = h1||h2  -- exact order per auxfunc.c
    inner = _sm3(bytes(x ^ 0x36 for x in k) + m)
    k1 = _sm3(bytes(x ^ 0x5C for x in k) + inner)
    h2 = _sm3(k1 + h1)
    return h1 + h2


class BufferShake:
    """origami 'shake' = concat-absorb (cap 208B) + sequential squeeze of
    pseudoXOF(buffer)."""

    def __init__(self):
        self.buf = bytearray()
        self.off = 0

    def absorb(self, data: bytes):
        room = 208 - len(self.buf)
        if room > 0:
            self.buf += data[:room]

    def finalize(self):
        pass

    def squeeze(self, n: int) -> bytes:
        out = pseudo_xof(bytes(self.buf), self.off + n)[self.off:
                                                        self.off + n]
        self.off += n
        return out


def derive_bytes(seed, label, a, b, c, extra=b""):
    st = BufferShake()
    st.absorb(seed)
    st.absorb(label)
    st.absorb(struct.pack("<III", a, b, c))
    st.absorb(extra)
    st.finalize()
    return st.squeeze(4096)  # callers slice


# ---------------------------------------------------------------- GF(16)
F_STAR = [1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9]
LOG = {v: i for i, v in enumerate(F_STAR)}


def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else F_STAR[(LOG[a] + LOG[b]) % 15]


def gf_inv(a):
    return F_STAR[(15 - LOG[a]) % 15]


GF_MUL = np.vectorize(gf_mul) if False else None  # tables used directly

_MUL = np.zeros((16, 16), dtype=np.uint8)
for _a in range(16):
    for _b in range(16):
        _MUL[_a, _b] = gf_mul(_a, _b)
_INV = np.zeros(16, dtype=np.uint8)
for _a in range(1, 16):
    _INV[_a] = gf_inv(_a)


def expand_gf(data: bytes, count: int):
    """nibbles, even index = LOW nibble."""
    b = np.frombuffer(data, dtype=np.uint8, count=(count + 1) // 2)
    lo = (b & 0xF).astype(np.int64)
    hi = (b >> 4).astype(np.int64)
    vals = np.empty(2 * len(b), dtype=np.int64)
    vals[0::2] = lo
    vals[1::2] = hi
    return vals[:count]


# ---------------------------------------------------------------- zones
ZONES = [  # (n_offset, flat_v, flat_o, flat_m, m_offset)
    (0, 24, 40, 32, 0),
    (64, 24, 40, 32, 32),
    (128, 24, 40, 32, 64),
    (192, 0, 8, 8, 96),
]
N_TOTAL, M_TOTAL = 200, 104
PARAM_ID = 0x4F4D0080
TOTAL_ELIGIBLE = 350208
STRIDE = 12545
RPK_NIBBLES = 5952


def gen_permutation(seed, label, domain, count):
    """Fisher-Yates: absorb seed+label+u32le(domain) ONLY (no b/c);
    sequential 2-byte squeeze per step."""
    st = BufferShake()
    st.absorb(seed)
    st.absorb(label)
    st.absorb(struct.pack("<I", domain))
    st.finalize()
    perm = list(range(count))
    for i in range(count - 1, 0, -1):
        rnd = st.squeeze(2)
        j = (rnd[0] | (rnd[1] << 8)) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return np.array(perm)


class GFStream:
    def __init__(self, seed, label, a, b, extra=b""):
        self.seed, self.label, self.a, self.b, self.extra = seed, label, a, b, extra
        self.block = 0
        self.buf = self._block(0)
        self.pos = 0

    def _block(self, j):
        st = BufferShake()
        st.absorb(self.seed)
        st.absorb(self.label)
        st.absorb(struct.pack("<III", self.a, self.b, PARAM_ID))
        st.absorb(struct.pack("<I", j) + self.extra)
        st.finalize()
        return st.squeeze(4096)

    def next_nibbles(self, k):
        out = []
        while len(out) < k:
            if self.pos >= 4096:
                self.block += 1
                self.buf = self._block(self.block)
                self.pos = 0
            b = self.buf[self.pos]
            self.pos += 1
            out.append(b & 0xF)
            out.append(b >> 4)
        return out[:k]


def residual_offset(pk_seed):
    acc = (0x9E3779B97F4A7C15 ^ PARAM_ID) & 0xFFFFFFFFFFFFFFFF
    for b in pk_seed:
        acc ^= (b + 0x9E3779B97F4A7C15 + ((acc << 6) & 0xFFFFFFFFFFFFFFFF)
                + (acc >> 2)) & 0xFFFFFFFFFFFFFFFF
    return int(acc % TOTAL_ELIGIBLE)


class OrigamiPK:
    def __init__(self, pk_bytes):
        assert struct.unpack("<I", pk_bytes[:4])[0] == PARAM_ID
        self.pk_seed = pk_bytes[4:20]
        rpk = expand_gf(pk_bytes[20:20 + 2976], RPK_NIBBLES)
        self.R = rpk
        # gen_permutation returns the SHUFFLED perm = C's secret_to_public
        # (forward: secret_index -> public_index). The variable mapping
        # needs the INVERSE (public_index -> secret_index).
        self.secret_to_public = gen_permutation(
            self.pk_seed, b"secret-to-public", 0, N_TOTAL)
        self.public_to_secret = np.zeros(N_TOTAL, dtype=np.int64)
        self.public_to_secret[self.secret_to_public] = np.arange(N_TOTAL)
        self.r_off = residual_offset(self.pk_seed)
        # per-equation P_affine streams are heavy; build coefficient arrays
        # lazily per evaluation is too slow — precompute all coefficients:
        # 104 eqs x (non x oil) nibbles
        self._coeffs = {}
        self._build_all_coeffs()

    def _build_all_coeffs(self):
        zone_base = []
        acc = 0
        for (n_off, fv, fo, fm, m_off) in ZONES:
            zone_base.append(acc)
            acc += fm * (n_off + fv) * fo
        total = acc
        assert total == TOTAL_ELIGIBLE
        coeffs = np.zeros(total, dtype=np.int64)
        for z, (n_off, fv, fo, fm, m_off) in enumerate(ZONES):
            non_count = n_off + fv
            for eq in range(fm):
                eq_global = m_off + eq
                gs = GFStream(self.pk_seed, b"P_affine", eq_global, 200)
                nib = gs.next_nibbles(non_count * fo)
                idx0 = zone_base[z] + eq * non_count * fo
                coeffs[idx0:idx0 + non_count * fo] = nib
        # residual overrides
        idx = np.arange(total, dtype=np.int64)
        rank = (idx * STRIDE + self.r_off) % TOTAL_ELIGIBLE
        sel = rank < RPK_NIBBLES
        coeffs[sel] = self.R[rank[sel]]
        self.all_coeffs = coeffs
        self.zone_base = zone_base

    def evaluate(self, sigma):
        """Ppub(sigma): 104 GF(16) values (sigma = public coords)."""
        y = np.zeros(N_TOTAL, dtype=np.int64)
        y[self.public_to_secret] = sigma
        res = np.zeros(M_TOTAL, dtype=np.int64)
        for z, (n_off, fv, fo, fm, m_off) in enumerate(ZONES):
            non_count = n_off + fv
            oil_start = n_off + fv
            yv = y[:non_count]
            yo = y[oil_start:oil_start + fo]
            for eq in range(fm):
                idx0 = self.zone_base[z] + eq * non_count * fo
                c = self.all_coeffs[idx0:idx0 + non_count * fo] \
                    .reshape(non_count, fo)
                # acc = sum_{non,oil} c * y_non * y_oil  (XOR-add in GF(16))
                prod = _MUL[np.ix_(yv, yo)]  # (non, oil)
                cp = _MUL[c, prod].ravel()
                # XOR-reduce
                acc = 0
                for v in cp.tolist():
                    acc ^= v
                res[m_off + eq] = acc
        return res


def hash_to_field(digest: bytes, salt: bytes):
    st = BufferShake()
    st.absorb(b"target")
    st.absorb(digest[:64])
    st.absorb(salt)
    st.finalize()
    raw = st.squeeze(52)
    return expand_gf(raw, M_TOTAL)


def sigma_from_sig(sig: bytes):
    return expand_gf(sig[:100], N_TOTAL)


# ------------------------------------------------ forge (structural test)
def forge(pk: OrigamiPK, digest: bytes, salt: bytes, rng, try_max=64):
    """Key-free signature: per zone fix non-oil prefix + 8 free oil coords,
    solve the linear system for the rest."""
    target = hash_to_field(digest, salt)
    y = np.zeros(N_TOTAL, dtype=np.int64)
    for z, (n_off, fv, fo, fm, m_off) in enumerate(ZONES):
        non_count = n_off + fv
        oil_start = n_off + fv
        if fo == fm:      # tail zone: square system
            free, sol = 0, fo
        else:
            free, sol = fo - fm, fm
        for attempt in range(try_max):
            y[:non_count] = rng.integers(0, 16, non_count)
            free_idx = rng.choice(fo, free, replace=False) if free else \
                np.array([], dtype=int)
            y[oil_start + free_idx] = rng.integers(0, 16, free) if free \
                else np.array([], dtype=int)
            sol_idx = np.setdiff1d(np.arange(fo), free_idx)
            # build A (fm x sol), b = target - 0  (no constant: no vv terms)
            A = np.zeros((fm, sol), dtype=np.int64)
            for eq in range(fm):
                idx0 = pk.zone_base[z] + eq * non_count * fo
                c = pk.all_coeffs[idx0:idx0 + non_count * fo] \
                    .reshape(non_count, fo)
                row = np.zeros(sol, dtype=np.int64)
                for ni in range(non_count):
                    if y[ni] == 0:
                        continue
                    crow = _MUL[y[ni], c[ni, :]]  # y_non * c  per oil
                    row ^= _MUL[crow[sol_idx], 0] * 0  # placeholder
                    # accumulate crow[sol_idx] (coeff* y_non pre-multiplied)
                    row = row ^ crow[sol_idx]
                A[eq] = row
            rhs = target[m_off:m_off + fm]
            x = _solve_gf16(A, rhs)
            if x is not None:
                y[oil_start + sol_idx] = x
                break
        else:
            return None
    sigma = np.zeros(N_TOTAL, dtype=np.int64)
    sigma[pk.secret_to_public] = y
    sig = bytearray(116)
    packed = np.packbits(sigma.reshape(-1, 4) @ (1 << np.arange(4)),
                         bitorder="little") if False else None
    # nibble pack: even LOW
    out = np.zeros(100, dtype=np.uint8)
    for i in range(0, N_TOTAL, 2):
        out[i // 2] = (sigma[i] & 0xF) | ((sigma[i + 1] & 0xF) << 4)
    sig[:100] = out
    sig[100:116] = salt
    return bytes(sig)


def _solve_gf16(A, b):
    """Gaussian elimination over GF(16); None if singular."""
    A = A.copy() % 16
    b = b.copy()
    m, n = A.shape
    for col in range(min(m, n)):
        piv = None
        for r in range(col, m):
            if A[r, col] != 0:
                piv = r
                break
        if piv is None:
            return None
        if piv != col:
            A[[col, piv]] = A[[piv, col]]
            b[[col, piv]] = b[[piv, col]]
        inv = _INV[A[col, col]]
        A[col] = _MUL[A[col], inv]
        b[col] = _MUL[b[col], inv]
        for r in range(m):
            if r != col and A[r, col] != 0:
                f = A[r, col]
                A[r] ^= _MUL[f, A[col]] if False else _MUL[np.full(n, f), A[col]]
                b[r] ^= _MUL[f, b[col]]
    return b
