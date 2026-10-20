"""Decoder ladder: sigma-adaptive, self-certifying (public-relation razor).

Stages: rounding -> pair-scan -> windowed MITM (CPU). The razor is the
public relation check — the ONLY point where cross-channel/estimate fusion
may legally occur (composition law).
"""
import numpy as np


class DecoderLadder:
    def __init__(self, spec, Ainv, b1f, A0):
        self.spec = spec
        self.Ainv, self.b1f, self.A0 = Ainv, b1f, A0
        n = Ainv.shape[0]
        self.n = n
        self.q = spec.q

    def cen(self, x):
        return ((x % self.q) + self.q // 2) % self.q - self.q // 2

    def s_of_R(self, R):
        return self.cen((self.Ainv @ ((self.b1f + R.reshape(-1)) % self.q))
                        % self.q)

    def razor(self, R):
        s = self.s_of_R(R)
        lo, hi = self.spec.relation.s_alphabet[0], self.spec.relation.s_alphabet[-1]
        return int(((s < lo) | (s > hi)).sum()) == 0, s

    # ---------------- stage 1: rounding ----------------
    def rounding(self, Rhat):
        lo, hi = self.spec.relation.r_box
        R = np.clip(np.rint(Rhat), lo, hi).astype(np.int64)
        ok, s = self.razor(R)
        return (R, s) if ok else (None, None)

    # ---------------- stage 2: pair scan ----------------
    def pair_scan(self, Rhat, max_stage=2):
        lo, hi = self.spec.relation.r_box
        R = np.clip(np.rint(Rhat), lo, hi).astype(np.int64)
        ok, _ = self.razor(R)
        if ok:
            return self.rounding(Rhat)
        Ac = self.Ainv
        c0 = self.s_of_R(R)
        n = self.n
        for da in (-1, 1, -2, 2):
            for ma in range(n):
                if not (lo <= R.reshape(-1)[ma] + da <= hi):
                    continue
                cu = self.cen(c0 + da * Ac[:, ma])
                if int(((cu < -1) | (cu > 1)).sum()) == 0:
                    R2 = R.copy().reshape(-1); R2[ma] += da
                    R2 = R2.reshape(R.shape)
                    ok2, s2 = self.razor(R2)
                    if ok2:
                        return R2, s2
        if max_stage < 2:
            return None, None
        for da in (-1, 1, -2, 2):
            for ma in range(n):
                if not (lo <= R.reshape(-1)[ma] + da <= hi):
                    continue
                u = c0 + da * Ac[:, ma]
                for db in (-1, 1, -2, 2):
                    U = (u[:, None] + db * Ac) % self.q
                    C = ((U + self.q // 2) % self.q) - self.q // 2
                    okc = ((C >= -1) & (C <= 1)).sum(axis=0)
                    for mb in np.nonzero(okc == n)[0]:
                        if mb == ma:
                            continue
                        if not (lo <= R.reshape(-1)[mb] + db <= hi):
                            continue
                        R2 = R.copy().reshape(-1)
                        R2[ma] += da; R2[mb] += db
                        R2 = R2.reshape(R.shape)
                        ok2, s2 = self.razor(R2)
                        if ok2:
                            return R2, s2
        return None, None

    # ---------------- stage 3: windowed MITM (z1-style estimates) --------
    def mitm_window(self, shat, Mx, W):
        """Double-key exact-match join over a confidence-ranked window,
        applied to a TERNARY estimate shat (s0-space). Returns (s, lik) or
        (None, None)."""
        q, n = self.q, self.n
        shf = shat.reshape(-1)
        s_round = np.clip(np.rint(shf), -1, 1).astype(np.int64)
        marg = np.abs(np.abs(shf) - 0.5)
        order = np.argsort(marg)
        win = order[:W]
        deltas = np.array([0 - s_round[m] if s_round[m] != 0
                           else (1 if shf[m] > 0 else -1) for m in win],
                          dtype=np.int64)
        NC = 6
        scr = np.linspace(0, n - 1, NC).astype(int)
        base = ((Mx[scr] @ s_round) - self.b1f[scr]) % q
        cols = np.zeros((W, NC), dtype=np.int64)
        for k_, m in enumerate(win):
            e = np.zeros(n, dtype=np.int64)
            e[m] = deltas[k_]
            cols[k_] = (Mx[scr] @ e) % q
        WA, WB = W // 2, W - W // 2
        def build(w, c):
            NS = 1 << w
            S = np.zeros((NS, NC), dtype=np.int64)
            for k_ in range(w):
                h = 1 << k_
                S[h:h + h] = (S[:h] + c[k_]) % q
            return S
        SA = build(WA, cols[:WA]); SB = build(WB, cols[WA:])
        keyB = SB[:, 0] * q + SB[:, 1]
        orderB = np.argsort(keyB); keyB_sorted = np.sort(keyB)
        rhi = self.spec.relation.r_box[1]
        ACC = np.concatenate([np.arange(0, rhi + 1), np.arange(q - rhi, q)])
        NA = len(SA)
        best = None
        for t1 in ACC:
            need1 = (int(t1) - base[0] - SA[:, 0]) % q
            for t2 in ACC:
                need2 = (int(t2) - base[1] - SA[:, 1]) % q
                key = need1 * q + need2
                # chunked searchsorted over SA (2^28 needles)
                SB_ = 8_000_000
                lo_parts, hi_parts = [], []
                for st_ in range(0, NA, SB_):
                    kl = key[st_:st_ + SB_]
                    lo_parts.append(np.searchsorted(keyB_sorted, kl, "left"))
                    hi_parts.append(np.searchsorted(keyB_sorted, kl, "right"))
                lo_ = np.concatenate(lo_parts) if lo_parts else np.array([], np.intp)
                hi_ = np.concatenate(hi_parts) if hi_parts else np.array([], np.intp)
                widths = hi_ - lo_
                tot = int(widths.sum())
                if tot == 0:
                    continue
                # chunked expansion (patch: cap-run memory blowup)
                CHUNK = 4_000_000
                csw = np.cumsum(widths)
                ai_l, bi_l = [], []
                done = 0
                while done < tot:
                    en = min(done + CHUNK, tot)
                    k = np.arange(done, en)          # global pair index
                    ai_c = np.searchsorted(csw, k, "right")
                    base_off = np.where(ai_c == 0, 0, csw[ai_c - 1])
                    bi_idx = lo_[ai_c] + (k - base_off)
                    ai_l.append(ai_c)
                    bi_l.append(orderB[bi_idx])
                    done = en
                ai = np.concatenate(ai_l)
                bi = np.concatenate(bi_l)
                good = np.ones(tot, dtype=bool)
                for cc in range(2, NC):
                    u = (SA[ai, cc] + SB[bi, cc] + base[cc]) % q
                    good &= (u < rhi + 1) | (u >= q - rhi)
                if not good.any():
                    continue
                for a_, b_ in zip(ai[good].tolist(), bi[good].tolist()):
                    s_c = s_round.copy()
                    for k_ in range(WA):
                        if (a_ >> k_) & 1:
                            s_c[win[k_]] += deltas[k_]
                    for k_ in range(WB):
                        if (b_ >> k_) & 1:
                            s_c[win[WA + k_]] += deltas[WA + k_]
                    Rv = self.cen((Mx @ s_c) - self.b1f)
                    ok, s = self.razor(Rv)
                    if ok:
                        lik = float(((shf - s_c) ** 2).sum())
                        if best is None or lik < best[0]:
                            best = (lik, s_c)
        if best is None:
            return None, None
        return best[1], best[0]
