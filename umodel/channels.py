"""channels.py (v3) — the three universal channel estimators.

Each estimator consumes ONLY public transcript data plus the spec, and
outputs an estimate of a LINKED quantity with a public error model:

  LabelChannel          narrow-mask response  -> per-signature sign labels
                        (hard + soft). Any mask width, signed or unsigned
                        challenges.
  ShiftRegressionChannel  masked linear response z = y + eps*(c*s) ->
                        per-frequency WLS in the ring's spectral basis ->
                        s-hat. Any ring kind, any mask width, chunked.
  QuantizedMomentChannel  quantized observation whose first moment is a
                        deterministic linear functional of a linked
                        quantity, via a PUBLIC Monte-Carlo operator built
                        from (d, tau, ring) constants only.

Scope gate: the moment-channel operator is derived for the unsigned
negacyclic family (closed form verified in operator_lemma.py). For other
ring/challenge combinations check_applicability refuses the channel with
an explicit reason instead of emitting an unvalidated operator.
"""
import numpy as np

from .spec import MaskModel


def _clip_int(x, lo, hi):
    return np.clip(x, lo, hi)


# ---------------------------------------------------------------- labels
class LabelChannel:
    """LLR labels from the narrow-mask response z0 = y0 + eps*c.

    Per coordinate: LLR_k = log f0(z_k - c_k) - log f0(z_k + c_k), summed
    over the challenge support. Handles c in {-1,0,1} (signed) and {0,1}
    (unsigned) uniformly; f0 comes from the declared mask model."""

    def __init__(self, spec):
        self.spec = spec
        ch = spec.channels.get("labels")
        if ch is None:
            raise ValueError("spec declares no labels channel")
        self.g0 = ch.mask.half_width
        yv, pv = ch.mask.pmf()
        if ch.mask.kind == "empirical":
            # table on its own value grid; snap queries to the nearest tabulated value
            self._vals = np.asarray(yv, dtype=np.float64)
            self._probs = np.asarray(pv, dtype=np.float64)
            self._emp = True
            self._lut = self._empirical_lut()
            return
        self._emp = False
        z = yv
        f0v = pv
        # LUT over z0 grid [-(g0), g0] (support widened by c = +/-1),
        # columns c = -1, 0, +1
        zgrid = np.arange(-self.g0, self.g0 + 1)

        def log_f0(val):
            inside = np.abs(val) <= self.g0 - 1
            p = np.where(inside, f0v[np.clip(val + self.g0 - 1, 0, len(f0v) - 1)],
                         0.0)
            with np.errstate(divide="ignore"):
                return np.log(p)

        lut = np.zeros((2 * self.g0 + 1, 3))
        for ci, cval in enumerate((-1, 0, 1)):
            with np.errstate(divide="ignore", invalid="ignore"):
                el = log_f0(zgrid - cval) - log_f0(zgrid + cval)
            lut[:, ci] = np.clip(np.where(np.isfinite(el), el,
                                          np.sign(el) * 40), -40, 40)
        self._lut = lut

    def _empirical_lut(self):
        """LLR LUT for a tabulated mask pmf: query grid = the union of
        {v - 1, v, v + 1} over tabulated values, clipped to the support
        bound; probability off-table = 0."""
        v = self._vals
        grid = np.unique(np.concatenate([v - 1, v, v + 1]).astype(np.int64))
        keep = (grid >= -self.g0) & (grid <= self.g0)
        grid = grid[keep]

        def p_of(x):
            # probability that the mask equals x (0 if not tabulated)
            idx = np.searchsorted(v.astype(np.int64), x)
            idx_c = np.clip(idx, 0, len(v) - 1)
            hit = v.astype(np.int64)[idx_c] == x
            return np.where(hit, self._probs[idx_c], 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            lut = np.zeros((len(grid), 3))
            for ci, cval in enumerate((-1, 0, 1)):
                el = np.log(p_of(grid - cval)) - np.log(p_of(grid + cval))
                lut[:, ci] = np.clip(np.where(np.isfinite(el), el,
                                              np.sign(el) * 40), -40, 40)
        self._grid = grid
        self._gmin = int(grid[0])
        return lut

    def _lookup(self, obs, cvals):
        """LLR of observations at challenge values; obs/cvals same shape."""
        o = np.clip(np.rint(obs), -self.g0, self.g0).astype(np.int64)
        cc = np.clip(cvals, -1, 1).astype(np.int64) + 1
        if getattr(self, "_emp", False):
            # snap to the tabulated query grid (nearest tabulated point)
            gi = np.searchsorted(self._grid, o)
            gi = np.clip(gi, 0, len(self._grid) - 1)
            left = np.clip(gi - 1, 0, len(self._grid) - 1)
            dleft = np.abs(self._grid[left] - o)
            dright = np.abs(self._grid[gi] - o)
            gi = np.where(dleft <= dright, left, gi)
            return self._lut[gi, cc]
        return self._lut[o + self.g0, cc]

    def run(self, c, z0):
        """c: (n, d) canonical challenges (int), z0: (n, d) narrow response.
        Returns (hard in {+1,-1}, soft in (-1,1), lam)."""
        lam = self._lookup(z0, c).sum(axis=1)
        hard = np.where(lam > 0, 1, -1)
        soft = np.tanh(np.clip(lam, -40, 40) / 2)
        return hard, soft, lam

    def run_chunked(self, c, z0, chunk_sizes):
        """Chunked latents (e.g. CS-128): per-chunk labels from the label
        response at the chunk's challenge positions. c: (n,d) SIGNED
        published values; returns hard (n,n_chunks), soft (n,n_chunks)."""
        n, d = c.shape
        per_pos = self._lookup(z0, c)                   # (n,d), 0 outside
        rows, cols = np.nonzero(c)
        order = np.argsort(rows, kind="stable")
        rows, cols = rows[order], cols[order]
        nch = len(chunk_sizes)
        chunk_id = np.arange(len(cols)) % sum(chunk_sizes)
        chunk_id = np.searchsorted(np.cumsum(chunk_sizes), chunk_id,
                                   side="right")
        lam = np.zeros((n, nch))
        np.add.at(lam, (rows, chunk_id), per_pos[rows, cols])
        hard = np.where(lam > 0, 1, -1)
        soft = np.tanh(np.clip(lam, -40, 40) / 2)
        return hard, soft


# ---------------------------------------------------------- regression
class ShiftRegressionChannel:
    """Per-frequency weighted least squares for z = y + eps*(c*s).

    In the ring's spectral basis the challenge-shift operator is diagonal,
    so each frequency solves independently:
        shat_nu = sum_t w_t conj(lam_t) z_t / sum_t w_t^2 |lam_t|^2,
    with w = soft labels (E[eps|public]); then s = Re(Z shat)/d.
    Chunked over signatures; works for any ring kind in the spec."""

    def __init__(self, spec):
        self.spec = spec
        ch = spec.channels.get("z1-regression")
        if ch is None:
            raise ValueError("spec declares no shift_regression channel")
        self.d = spec.d
        self._Zinv = spec.ring.inverse_basis(self.d)

    def accumulate(self, c, z, soft, chunk=50_000):
        """Stream accumulation. c: (n,d); z: (n,L,d); soft: (n,).
        Returns (B (L,d), Dn (d,), n_used)."""
        n = c.shape[0]
        L = z.shape[1]
        B = np.zeros((L, self.d), dtype=complex)
        Dn = np.zeros(self.d)
        for st in range(0, n, chunk):
            en = min(chunk, n - st)
            lam = self.spec.ring.spectrum(c[st:st + en])            # (c,d)
            zh = self.spec.ring.spectrum(z[st:st + en])             # (c,L,d)
            w = soft[st:st + en]
            B += np.einsum("t,tk,tlk->lk", w, np.conj(lam), zh)
            Dn += np.einsum("t,tk->k", w * w, np.abs(lam) ** 2)
        return B, Dn, n

    def solve(self, B, Dn):
        dead = int((Dn <= 0).sum())
        Dsafe = np.where(Dn > 0, Dn, 1.0)
        shat_freq = B / Dsafe[None, :]
        s = np.stack([np.real(self._Zinv @ shat_freq[i]) / self.d
                      for i in range(shat_freq.shape[0])])
        return s, dead

    def run(self, c, z, soft, chunk=50_000):
        B, Dn, n = self.accumulate(c, z, soft, chunk)
        s, dead = self.solve(B, Dn)
        return s, {"dead_frequencies": dead, "n_used": n}

    def accumulate_chunked(self, chunk_spectra, z, W, chunk=20_000):
        """Chunked-latent accumulation (e.g. CS-128).

        chunk_spectra: list (per chunk) of (n, d) complex spectra;
        W: (n, n_chunks) soft label weights; z: (n, L, d). The effective
        design per signature is lambda_eff = sum_i W_i lambda_i."""
        n = z.shape[0]
        L = z.shape[1]
        B = np.zeros((L, self.d), dtype=complex)
        Dn = np.zeros(self.d)
        ones = np.ones(chunk)
        for st in range(0, n, chunk):
            en = min(chunk, n - st)
            lam_eff = None
            for i, lam_i in enumerate(chunk_spectra):
                w = W[st:st + en, i][:, None]
                term = w * lam_i[st:st + en]
                lam_eff = term if lam_eff is None else lam_eff + term
            zh = self.spec.ring.spectrum(z[st:st + en])
            B += np.einsum("t,tk,tlk->lk", ones[:en], np.conj(lam_eff), zh)
            Dn += np.einsum("t,tk->k", ones[:en], np.abs(lam_eff) ** 2)
        return B, Dn, n

    def run_chunked(self, chunk_spectra, z, W, chunk=20_000):
        B, Dn, n = self.accumulate_chunked(chunk_spectra, z, W, chunk)
        s, dead = self.solve(B, Dn)
        return s, {"dead_frequencies": dead, "n_used": n,
                   "label_weight_rms": float(np.sqrt((W ** 2).mean()))}


# ------------------------------------------------------------ moment
class QuantizedMomentChannel:
    """First-moment channel through a PUBLIC Monte-Carlo operator.

    v3 scope: the operator derivation (and its closed form, see
    operator_lemma.py) is established for the unsigned negacyclic family.
    build_operator refuses other combinations with a reason — the caller
    is expected to have gated via check_applicability."""

    def __init__(self, spec, n_mc=30000, seed=20260928):
        self.spec = spec
        self.n_mc = n_mc
        self._seed = seed

    def build_operator(self):
        spec = self.spec
        if spec.challenge_signed:
            # lemma (v5): independent +/-1 challenge values kill the
            # cross-position terms -> E[C^T C] = tau*I exactly -> the field
            # part is ZERO (no deblurring). Ring-independent.
            return np.zeros((spec.d, spec.d))
        if spec.ring.kind != "negacyclic":
            raise NotImplementedError(
                "moment-channel operator for UNSIGNED challenges is "
                "derived only for the negacyclic family; refused rather "
                "than emitted unvalidated")
        d, tau = spec.d, spec.tau
        rng = np.random.default_rng(self._seed)
        F = np.zeros((d, d))
        ar = np.arange(d)
        for _ in range(self.n_mc):
            pos = rng.choice(d, tau, replace=False)
            J = (ar[:, None] + pos[None, :]) % d
            V = (J[:, :, None] - pos[None, None, :]) % d
            S = np.where(J[:, :, None] < pos[None, None, :], -1.0, 1.0)
            FL = np.where(ar[:, None] + pos[None, :] >= d, -1.0, 1.0)
            M = S * FL[:, :, None]
            for t in range(tau):
                Mt = M[:, t, :].copy()
                Mt[:, t] = 0.0
                rows_m = np.repeat(ar, tau)
                cols_v = V[:, t, :].ravel()
                np.add.at(F, (rows_m, cols_v), Mt.ravel())
        return F / self.n_mc

    def accumulate(self, c, h, hard, chunk=5000):
        """Aligned first-moment accumulator (adjoint alignment, spec ring).

        c: (n,d); h: (n,K,d); hard: (n,) sign labels. Returns Phi (K,d)."""
        spec = self.spec
        d, tau = spec.d, spec.tau
        m_range = np.arange(d)
        K = h.shape[1]
        Phi = np.zeros((K, d))
        for st in range(0, c.shape[0], chunk):
            en = min(chunk, c.shape[0] - st)
            cc = c[st:st + en]
            hh = h[st:st + en]
            eps = hard[st:st + en]
            idx = np.nonzero(cc)[1].reshape(en, tau)
            if spec.ring.kind == "cyclic":
                # plain gather, no wrap flip: (en, K, tau, d) -> (en,tau,K,d)
                g = np.stack([hh[:, k, (idx + m_range[None, :]) % d]
                              for k in range(K)], axis=1)
                g = g.transpose(0, 2, 1, 3)
            else:
                psi_ext = np.concatenate([hh, -hh], axis=2)
                psiT = np.ascontiguousarray(psi_ext.transpose(2, 0, 1))
                rows = np.repeat(np.arange(en), tau)
                g = psiT[idx.reshape(-1)[:, None] + m_range[None, :], rows[:, None], :]
                g = np.ascontiguousarray(
                    g.reshape(en, tau, d, hh.shape[1]).transpose(0, 1, 3, 2))
            Phi += (g * eps[:, None, None, None]).sum(axis=(0, 1))
        return Phi

    def solve(self, Phi, N, s_eff_grid=None, gamma2=None, tau=None):
        """Rhat = -(gamma2) * (signal + N*Fop)^{-1} Phi, scale fit public.

        s_eff absorbs label accuracy; the fit objective is the public
        near-integer alignment cos(2*pi*R)."""
        spec = self.spec
        d = spec.d
        tau = spec.tau if tau is None else tau
        gamma2 = spec.gamma2 if gamma2 is None else gamma2
        Fop = self.build_operator() if not hasattr(self, "_Fop") else self._Fop
        M = N * tau
        out = []
        for i in range(Phi.shape[0]):
            if s_eff_grid is None:
                Rh = np.linalg.solve(M * np.eye(d) + N * Fop, -gamma2 * Phi[i])
                out.append(Rh)
                continue
            best = None
            for s in s_eff_grid:
                Rh = np.linalg.solve(float(s) * M * np.eye(d) + N * Fop,
                                     -gamma2 * Phi[i])
                o = float(np.cos(2 * np.pi * Rh).mean())
                if best is None or o > best[0]:
                    best = (o, Rh)
            out.append(best[1])
        return np.stack(out)
