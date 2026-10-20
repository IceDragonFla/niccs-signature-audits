"""spec.py (v3) — fully declarative scheme specification.

A spec is the ONLY scheme-specific input to the framework. Engines,
simulators, decoders and the validator read the spec, never scheme code.

Universal-model scope (checked by check_applicability, not assumed):
  - the scheme lives in a commutative ring R_q = Z_q[x]/(x^d - a) with
    a = +1 (cyclic) or a = -1 (negacyclic), d a power of two;
  - the public key satisfies a LINEAR relation over R_q between a secret
    of small alphabet and small public rows, with a bounded residual R;
  - at least one leakage channel of the three implemented kinds is
    declared (labels / shift_regression / quantized_moment).

Everything outside this class (hash-based, isogeny, ..., or rings we have
no spectral basis for) is REFUSED by check_applicability with a reason —
the framework is universal over the Fiat-Shamir-with-aborts module-lattice
family, and states that scope explicitly instead of silently failing.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------- ring
@dataclass(frozen=True)
class RingStructure:
    """Signed-shift algebra of R_q.

    Convention (one convention everywhere: generator, relation matrix,
    regression, operator):
        (c * s)[k] = sum_t c_t * s[(k - p_t) mod d] * flip(k - p_t),
        flip = -1 iff (k - p_t) < 0 for negacyclic, +1 always for cyclic.
    """
    kind: str = "negacyclic"          # "negacyclic" | "cyclic"

    def flip(self, delta) -> float:
        if self.kind == "cyclic":
            return 1.0
        return np_sign(delta)

    def spectrum(self, c):
        """Eigenvalues of the challenge-shift operator, per signature.

        c: (..., d) canonical challenges -> (..., d) complex eigenvalues.
        negacyclic: odd bins of the length-2d FFT; cyclic: all bins of the
        length-d FFT."""
        import numpy as np
        d = c.shape[-1]
        if self.kind == "cyclic":
            return np.fft.fft(c.astype(np.float64), n=d, axis=-1)
        pad = ((0, 0),) * (c.ndim - 1) + ((0, d),)
        return np.fft.fft(np.pad(c.astype(np.float64), pad), axis=-1)[..., 1::2]

    def inverse_basis(self, d):
        """Z with s = Re(Z @ s_hat) / d for the chosen basis."""
        import numpy as np
        k = np.arange(d)
        if self.kind == "cyclic":
            return np.exp(2j * np.pi * k[None, :] * k[:, None] / d)
        return np.exp(1j * np.pi * (2 * k[None, :] + 1) * k[:, None] / d)

    def alignment_flip(self, m_plus_p, d):
        """Sign flip for the adjoint alignment psi[(m+p) mod d]:
        -1 when the index wraps (negacyclic), +1 for cyclic."""
        if self.kind == "cyclic":
            return 1.0
        return -1.0 if m_plus_p >= d else 1.0

    def describe(self):
        poly = "x^d + 1" if self.kind == "negacyclic" else "x^d - 1"
        return f"Z_q[x]/({poly})"


def np_sign(delta):
    import numpy as np
    return np.where(delta < 0, -1.0, 1.0)


# ---------------------------------------------------------------- masks
@dataclass(frozen=True)
class MaskModel:
    """Fresh-mask distribution.
    kind='triangular': y in [-(g-1), g-1], P(y) ~ g - |y| (BiT centered).
    kind='uniform':    y uniform in [-(g-1), g] (Dilithium/COMPASS).
    kind='empirical':  pmf supplied as a table (measured from public data,
                       e.g. acceptance-conditioned masks of BLISS-style
                       schemes); half_width is the support bound."""
    kind: str = "triangular"
    half_width: int = 0               # gamma
    sigma: float = 0.0                # Gaussian sigma (kind='gaussian')
    pmf_table: tuple = ()             # (values, probs) for kind='empirical' '

    def support(self):
        g = self.half_width
        return range(-(g - 1), g + 1)

    def pmf(self):
        import numpy as np
        if self.kind == "gaussian":
            sig = self.sigma if self.sigma > 0 else self.half_width / 2.0
            v = np.arange(-4 * int(sig) - 2, 4 * int(sig) + 3)
            p = np.exp(-v * v / (2 * sig * sig))
            return v, p / p.sum()
        if self.kind == "empirical":
            if len(self.pmf_table) == 2 and len(self.pmf_table[0]):
                return (np.asarray(self.pmf_table[0]),
                        np.asarray(self.pmf_table[1], dtype=np.float64))
            g = max(self.half_width, 1)
            v = np.arange(-g, g + 1)
            sig = g / 2.2
            p = np.exp(-v * v / (2 * sig * sig))
            return v, p / p.sum()
        if self.kind == "uniform":
            y = np.arange(-(self.half_width - 1), self.half_width + 1)
            p = np.full_like(y, 1.0 / len(y), dtype=np.float64)
            return y, p
        g = self.half_width
        y = np.arange(-(g - 1), g)
        w = g - np.abs(y)
        return y, w / w.sum()

    def variance(self):
        y, p = self.pmf()
        return float((p * y * y).sum())


# ---------------------------------------------------------------- channels
CHANNEL_KINDS = ("labels", "shift_regression", "quantized_moment")


@dataclass(frozen=True)
class ChannelSpec:
    name: str                          # e.g. "z1-regression"
    kind: str                          # one of CHANNEL_KINDS
    observes: str                      # which public values it consumes
    estimates: str                     # linked quantity ("eps"|"s"|"R")
    mask: Optional[MaskModel] = None   # fresh-mask model (if masked)
    needs_labels: bool = False         # consumes per-signature sign labels


# ---------------------------------------------------------------- relation
@dataclass
class PublicKeyRelation:
    """A0 * s = pk_scale * b1 + R (mod q), R coordinate-wise in r_box.

    derive_equivalent_key: the R -> (e_hat, b0_hat) map. The clip rule is
    the BiT-family rule (unique at the R-box edges, three-way class in the
    middle); a scheme with a different target map overrides via subclass
    or by passing derive='none'."""
    form: str = "A0*s = 16*b1 + R mod q"
    s_alphabet: Tuple[int, ...] = (-1, 0, 1)
    r_box: Tuple[int, int] = (-9, 8)
    pk_scale: int = 16
    b0_box: Tuple[int, int] = (-8, 7)
    derive: str = "clip"

    def derive_equivalent_key(self, R):
        """R = b0 - e (audit-verified convention): pick e_hat in
        {-1,0,1} so that b0_hat = R + e_hat lands in the b0 box.
        At the box edges: R >= hi -> e_hat = -1; R <= lo -> e_hat = +1.
        (v17 fix: the previous version returned the OPPOSITE sign at the
        26 boundary coordinates — validated against the audit's
        forge-proven key_hat_h_real.bin.)"""
        import numpy as np
        R = np.rint(np.asarray(R)).astype(np.int64)
        lo, hi = self.b0_box
        e = np.where(R >= hi + 1, -1, np.where(R <= lo - 1, 1, 0))
        e = np.where((R == hi + 1) & (e == -1), -1, e)  # R=8 -> -1
        # general edges: R outside [lo, hi] by 1 -> move back inside
        e = np.where(R > hi, -1, np.where(R < lo, 1, 0))
        b0 = R + e
        return e.astype(np.int64), b0.astype(np.int64)


# ---------------------------------------------------------------- spec
@dataclass
class SchemeSpec:
    name: str
    d: int                            # ring degree (power of two)
    q: int                            # modulus (prime for the inverse)
    tau: int                          # challenge weight
    challenge_signed: bool            # False: all +1 (BiT); True: +/-1
    ring: RingStructure
    K: int                            # pk rows
    L: int                            # secret rows
    shared_latent: str                # "single-sign" | "independent-signs"
                                      # | "chunked-signs" | "none"
                                      # | "deterministic-normalization"
    chunk_sizes: tuple = ()           # latent structure for chunked schemes
                                      # (e.g. CS: (4,4,4,4,4,3) per-signature)
    channels: Dict[str, ChannelSpec] = field(default_factory=dict)
    relation: PublicKeyRelation = field(default_factory=PublicKeyRelation)
    gamma1: int = 0                   # regression mask width (0: none)
    gamma1_2: int = 0                 # z2 mask width (hint quantization)
    gamma2: int = 0                   # hint cell width
    gamma0: int = 0                   # label mask width (0: no labels channel)

    # ------------------------------------------------------------ factories
    @staticmethod
    def bit128() -> "SchemeSpec":
        return SchemeSpec(
            name="BiT-128", d=256, q=26881, tau=30, challenge_signed=False,
            ring=RingStructure("negacyclic"), K=3, L=3,
            shared_latent="single-sign", gamma1=1024, gamma1_2=2048,
            gamma2=682, gamma0=8,
            channels={
                "labels": ChannelSpec("labels", "labels", "z0, c", "eps",
                                      MaskModel("triangular", 8)),
                "z1-regression": ChannelSpec(
                    "z1-regression", "shift_regression", "z1, c, labels",
                    "s", MaskModel("triangular", 1024), needs_labels=True),
                "hint-operator": ChannelSpec(
                    "hint-operator", "quantized_moment", "h, c, labels",
                    "R", MaskModel("triangular", 2048), needs_labels=True),
            })

    @staticmethod
    def bit128_fixed() -> "SchemeSpec":
        """Repaired variant: independent signs per response block."""
        s = SchemeSpec.bit128()
        s.name = "BiT-128-fixed"
        s.shared_latent = "independent-signs"
        return s

    @staticmethod
    def scaled(d=128, tau=20) -> "SchemeSpec":
        """Scaled synthetic instance, same algebra as BiT-128."""
        s = SchemeSpec.bit128()
        s.name = f"scaled-d{d}-t{tau}"
        s.d, s.tau = d, tau
        return s

    @staticmethod
    def dlt_toy() -> "SchemeSpec":
        """Transfer instance: Dilithium-FLAVORED toy. Different ring (cyclic),
        different modulus (8380417), SIGNED challenge, narrower masks, L=K=1.
        Declares only the channels its structure actually has (no hint)."""
        return SchemeSpec(
            name="DLT-toy-c128", d=128, q=8380417, tau=19,
            challenge_signed=True, ring=RingStructure("cyclic"), K=1, L=1,
            shared_latent="single-sign", gamma1=256, gamma0=4,
            relation=PublicKeyRelation(
                form="A0*s = 16*b1 + R mod q (cyclic)",
                s_alphabet=(-1, 0, 1), r_box=(-9, 8), pk_scale=16,
                b0_box=(-8, 7), derive="clip"),
            channels={
                "labels": ChannelSpec("labels", "labels", "z0, c", "eps",
                                      MaskModel("triangular", 4)),
                "z1-regression": ChannelSpec(
                    "z1-regression", "shift_regression", "z1, c, labels",
                    "s", MaskModel("triangular", 256), needs_labels=True),
            })


# ---------------------------------------------------------------- checks
def check_applicability(spec: SchemeSpec) -> dict:
    """Refuse cleanly what the framework cannot legally touch."""
    fails = []
    d = spec.d
    if d & (d - 1):
        fails.append(f"ring degree d={d} is not a power of two: the "
                     "spectral BACKEND lacks an FFT basis here "
                     "(implementation/efficiency limitation of this "
                     "backend, NOT a mathematical impossibility — a "
                     "non-FFT backend could in principle handle it)")
    if spec.ring.kind not in ("negacyclic", "cyclic"):
        fails.append(f"ring kind '{spec.ring.kind}' not implemented")
    # modulus prime (needed for the relation inverse); cheap Fermat test
    if _is_probable_prime(spec.q):
        pass
    else:
        fails.append(f"modulus q={spec.q} not prime (relation may be "
                     "non-invertible)")
    if not spec.channels:
        fails.append("no channels declared")
    for cname, ch in spec.channels.items():
        if ch.kind not in CHANNEL_KINDS:
            fails.append(f"channel '{cname}': unknown kind '{ch.kind}'")
        if ch.kind == "labels" and (ch.mask is None or ch.mask.half_width <= 0):
            fails.append(f"channel '{cname}': labels channel needs a mask")
        if ch.kind == "shift_regression" and ch.mask is None:
            fails.append(f"channel '{cname}': regression needs a mask model")
        if ch.kind == "quantized_moment" and (
                not spec.challenge_signed and spec.ring.kind != "negacyclic"):
            fails.append(
                f"channel '{cname}': moment operator for UNSIGNED challenges "
                "is derived only for the negacyclic family; SIGNED "
                "challenges are covered (E[C^T C] = tau*I, Fop = 0 — see "
                "operator_lemma.py)")
    if spec.challenge_signed and spec.tau > d:
        fails.append("challenge weight exceeds degree")
    return {"applicable": not fails, "failures": fails,
            "ring": spec.ring.describe(), "channels": list(spec.channels)}


def _is_probable_prime(n, k=16):
    import numpy as np
    if n < 2 or n % 2 == 0:
        return n == 2
    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    rng = np.random.default_rng(3)
    for _ in range(k):
        a = int(rng.integers(2, n - 1))
        x = pow(a, s, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True
