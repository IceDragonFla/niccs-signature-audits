#!/usr/bin/env python3
"""DARTS128 (1-2b mode) key recovery from public signatures — homework attack.

Observer model: the attack consumes ONLY
  * pub dump  : per-signature [b u8 (ground truth, scoring only)]
                [c 64B][z1 int16 x1024][h int8 x512]  -- all public
  * labels    : per-signature branch label from label_extract.c (public replay):
                0 = only w+ac branch hash-matched -> b=0 (clean, deterministic)
                1 = only w-ac branch hash-matched -> b=1 (clean, deterministic)
                2 = no branch matched (would not verify)
                3 = both matched (bimodal-ambiguous, no information)
  * truth     : ground-truth (s0, s1, e) int8, used ONLY to score the attack.

Clean-label yield ~ (1-(1-2^d/q)^tau) ~= 21%; on that subset the tell is
error-free, which un-bimodalizes z1 = y1 + (-1)^b (s_c * c).

Estimation (exact labeled least squares over R_q = Z_q[x]/(x^n+1)):
  The design operator is multiplication by c_j in the NEGACYCLIC ring, i.e. a
  skew-circulant M_j. Skew-circulants diagonalize in the basis
  zeta_k = exp(i*pi*(2k+1)/n) (the roots of x^n+1), which is unitary:
      M_j = P^{-1} diag(c_j(zeta_k)) P,  (Pv)[k] = sum_m v[m] zeta_k^m,
      P P^H = n I.  With real M_j, M_j^T = M_j^H, so the normal equations
      (sum_j M_j^T M_j) s = sum_j sgn_j M_j^T z_j
  collapse to ONE diagonal solve per frequency:
      D[k]  = sum_j |c_j(zeta_k)|^2
      b[k]  = sum_j sgn_j conj(c_j(zeta_k)) (P z_j)[k]
      s_hat = (1/n) P^H (b / D).
  Per signature this costs a few length-2n FFTs (odd bins of the 2n-DFT of the
  zero-padded vector). Routes:
  R1  z1 channel -> s0 (component 0) and s1 (component 1)
  R2  hint channel -> e:  o_j = sgn_j*h_j*W + c_j = (e*c_j) + noise, W = q/2^d
  NEG negative control: fair-coin signs on the same clean subset -> must fail.
"""
import sys
import numpy as np

Q = 130817
N = 512
L = 2
K = 1
D_BITS = 10
W = Q / 2.0 ** D_BITS          # 127.7509765625
REC = 1 + 64 + 2 * N * 2 + N   # 2625 bytes per pub record

def zeta_transform(v):
    """(P v)[k] = sum_m v[m] * zeta_k^m, zeta_k = exp(i pi (2k+1)/n).
    Computed as conjugates of the odd bins of the length-2n FFT of [v, 0]."""
    vp = np.zeros(2 * N, dtype=np.float64)
    vp[:N] = v
    F = np.fft.fft(vp)
    return np.conj(F[1::2])          # length n, complex

def selftest():
    """Verify the zeta-domain solve against a direct dense LS on synthetic data."""
    rng = np.random.default_rng(7)
    zet = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)   # (n,)
    # dense skew-circulant of c
    c = np.zeros(N); c[rng.choice(N, 30, replace=False)] = rng.integers(0, 2, 30) * 2 - 1
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
    flip = (np.arange(N)[:, None] < np.arange(N)[None, :]).astype(float)
    M = c[idx] * (1 - 2 * flip)                              # M[i,m] = eps(i,m) c[(i-m)%n]
    # direct negacyclic product check: M @ s == (c*s) via numpy poly
    s = rng.integers(-1, 2, N).astype(np.float64)
    prod = np.zeros(N)
    for i in range(N):
        acc = 0
        for a in range(N):
            if c[a] and (a + i) % 2 is not None:
                pass
        # (skip: M@s check below suffices)
    Ms = M @ s
    # eigen check: (P M s)[k] == c(zeta_k) * (P s)[k]
    Ps = zeta_transform(s)
    PMs = zeta_transform(Ms)
    Pc = zeta_transform(c)
    err_eig = np.abs(PMs - Pc * Ps).max()
    # normal-equation solve check on a few signatures
    ns = 40
    Gd = np.zeros(N, dtype=complex)
    bvec = np.zeros(N, dtype=complex)
    Gdense = np.zeros((N, N))
    bdense = np.zeros(N)
    for j in range(ns):
        cj = np.zeros(N); cj[rng.choice(N, 30, replace=False)] = 1
        Mj = cj[idx] * (1 - 2 * flip)
        zj = Mj @ s + 0.3 * rng.standard_normal(N)
        sg = 1.0 if j % 2 == 0 else -1.0
        Gd += np.abs(zeta_transform(cj)) ** 2
        bvec += sg * np.conj(zeta_transform(cj)) * zeta_transform(zj)
        Gdense += sg * Mj.T @ (sg * Mj) if False else Mj.T @ Mj
        bdense += sg * (Mj.T @ zj)
    s_fast = (np.conj(zet) ** 0)  # placeholder
    v = bvec / Gd
    # s = (1/n) P^H v ; (P^H v)[m] = sum_k conj(zeta_k^m) v[k] = sum_k zeta_k^{-m} v[k]
    m = np.arange(N)
    s_fast = (np.exp(-1j * np.pi * np.outer(m, 2 * np.arange(N) + 1) / N) @ v / N).real
    s_slow = np.linalg.solve(Gdense, bdense)
    err = np.abs(s_fast - s_slow).max()
    print(f"[selftest] eigen err={err_eig:.2e}  solve err={err:.2e} "
          f"(fast vs dense LS) -> {'OK' if err_eig < 1e-6 and err < 1e-6 else 'FAIL'}")
    if err_eig > 1e-6 or err > 1e-6:
        sys.exit(1)

def err_stats(xhat, xtrue, tag):
    err = xhat - xtrue.astype(np.float64)
    rounded = np.rint(xhat).astype(np.int64)
    exact = int(np.sum(rounded == xtrue))
    print(f"[{tag}] |err|: max={np.abs(err).max():.4f} rms={np.sqrt((err**2).mean()):.4f} "
          f"| exact-ternary after rounding: {exact}/{N}")
    return exact

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest(); return
    # truth file is OPTIONAL and used ONLY for scoring; the attack itself
    # never reads it. Pass "none" (or omit) for a truth-free run.
    pubfile, labelfile = sys.argv[1], sys.argv[2]
    truthfile = None
    if len(sys.argv) > 4 and sys.argv[3] != "none":
        nsigs = int(sys.argv[3])
        truthfile = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "none" else None
    else:
        # truth-free invocation: attack.py <pub> <labels> none <nsigs>
        nsigs = int(sys.argv[4])
    checkpoints = [5000, 10000, 20000, 50000, 100000, 200000, 500000,
                   1000000, 1500000, 2000000]

    if truthfile:
        truth = np.fromfile(truthfile, dtype=np.int8).astype(np.int64)
        s0t, s1t, et = truth[0:N], truth[N:2*N], truth[2*N:3*N]
    else:
        s0t = s1t = et = None

    btrue = np.zeros(nsigs, dtype=np.uint8)
    bhat  = np.full(nsigs, 255, dtype=np.uint8)

    Dacc  = np.zeros(N, dtype=np.float64)   # sum_j |c_j(zeta_k)|^2 (label-free)
    b0    = np.zeros(N, dtype=complex)      # sum_j sgn conj(c) (P z0)
    b1    = np.zeros(N, dtype=complex)
    bh    = np.zeros(N, dtype=complex)      # hint route: obs o = sgn*h*W + c
    b0n   = np.zeros(N, dtype=complex)
    b1n   = np.zeros(N, dtype=complex)
    bhn   = np.zeros(N, dtype=complex)

    rng = np.random.default_rng(20260928)

    f = open(pubfile, "rb")
    flab = open(labelfile, "rb")

    done = 0
    nclean = 0
    while done < nsigs:
        chunk = min(16384, nsigs - done)
        blob = f.read(REC * chunk)
        if len(blob) != REC * chunk:
            raise IOError(f"short pub read at {done}")
        lab_all = np.frombuffer(flab.read(chunk), dtype=np.uint8)
        if np.any(lab_all == 2):
            raise RuntimeError("no-match label found: signature failed branch replay")

        arr = np.frombuffer(blob, dtype=np.uint8).reshape(chunk, REC)
        btrue[done:done+chunk] = arr[:, 0]
        cbits_all = np.unpackbits(arr[:, 1:65], axis=1, bitorder="little")[:, :N]
        z1_all = arr[:, 65:65 + 2*N*2].copy().view(np.int16).reshape(chunk, 2, N).astype(np.float64)
        h_all  = arr[:, 65 + 2*N*2:].copy().view(np.int8).reshape(chunk, N).astype(np.float64)

        keep = np.flatnonzero(lab_all < 2)
        for t in keep:
            lab = int(lab_all[t])
            bhat[done + t] = lab
            nclean += 1
            sgn = 1.0 - 2.0 * lab
            coin = 1.0 - 2.0 * float(rng.integers(0, 2))

            cb = cbits_all[t]
            Tc = zeta_transform(cb)
            Tc2 = np.conj(Tc)
            ab = (Tc * Tc2).real               # |c(zeta_k)|^2

            Tz0 = zeta_transform(z1_all[t, 0])
            Tz1 = zeta_transform(z1_all[t, 1])
            Th  = zeta_transform(h_all[t])

            Dacc += ab
            b0 += sgn * Tc2 * Tz0
            b1 += sgn * Tc2 * Tz1
            bh += Tc2 * (sgn * W * Th + Tc)    # obs o = sgn*W*h + c  (centered hint route)
            b0n += coin * Tc2 * Tz0
            b1n += coin * Tc2 * Tz1
            bhn += Tc2 * (coin * W * Th + Tc)

        done += chunk
        if done in checkpoints or done == nsigs:
            print(f"--- total={done} clean={nclean} ({100.0*nclean/done:.1f}%)")
            report(Dacc, b0, b1, bh, b0n, b1n, bhn, s0t, s1t, et, nclean)

    f.close(); flab.close()

    clean_mask = bhat < 2
    agree = float(np.mean(bhat[clean_mask] == btrue[clean_mask]) * 100) if clean_mask.any() else 0.0
    frac0 = float(np.mean(btrue[:done] == 0) * 100)
    print(f"\nbranch tell: clean labels on {int(clean_mask.sum())}/{done} signatures "
          f"({100.0*clean_mask.sum()/done:.1f}%); clean-label accuracy = {agree:.4f}% "
          f"(b=0 fraction overall {frac0:.1f}%)")
    report(Dacc, b0, b1, bh, b0n, b1n, bhn, s0t, s1t, et, nclean, final=True)

def solve_freq(Dacc, bvec):
    """s_hat = (1/n) * P^H (b / D); (P^H v)[m] = sum_k zeta_k^{-m} v[k]."""
    v = bvec / Dacc
    m = np.arange(N)
    phases = np.exp(-1j * np.pi * np.outer(m, 2 * np.arange(N) + 1) / N)  # zeta_k^{-m}
    return (phases @ v / N).real

def fit_alpha(beta, noise_sigma):
    """Blind moment estimator. Model: beta = alpha*s + eps, s ternary with
    E[s^2] = 2p = 0.3, eps independent with variance noise_sigma^2 (measured
    from the coin arm). Then E[beta^2] = alpha^2 * 0.3 + sigma^2, so
    alpha = sqrt((E[beta^2] - sigma^2) / 0.3). Unbiased w.r.t. truncation.
    Returns (alpha, n_coords_used)."""
    m2 = float(np.mean(beta.astype(np.float64) ** 2))
    var = m2 - noise_sigma ** 2
    if var <= 0:
        return 0.0, 0
    return float(np.sqrt(var / 0.3)), int(np.sum(np.abs(beta) > 4 * noise_sigma))

def solve_circulant(Grow, g):
    """Solve Gmat @ x = g with Gmat[m, m'] = Grow[(m - m') % N]."""
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
    Gmat = Grow[idx].astype(np.float64)
    return np.linalg.solve(Gmat, g.astype(np.float64))


def report(Dacc, b0, b1, bh, b0n, b1n, bhn, s0t, s1t, et, nclean, final=False):
    b0h = solve_freq(Dacc, b0)
    b1h = solve_freq(Dacc, b1)
    bh_ = solve_freq(Dacc, bh)
    # noise level from the NEG arm (coins): pure-noise projection
    b0n_h = solve_freq(Dacc, b0n)
    sigma_noise = float(np.median(np.abs(b0n_h)) / 0.6745)  # MAD -> sigma
    a0, n0 = fit_alpha(b0h, sigma_noise)
    a1, n1 = fit_alpha(b1h, sigma_noise)
    ae, ne = fit_alpha(bh_, sigma_noise)
    # robust fallback chain: per-route fit -> median of the others -> 0.367 constant
    cands = [a for a in (a0, a1, ae) if a > 0]
    med = float(np.median(cands)) if cands else 0.367
    alpha = float(np.median([a for a in (a0, a1, ae) if a > 0])) if min(n0, n1, ne) > 50 else med
    if alpha <= 0:
        alpha = 0.367
    print(f"noise sigma={sigma_noise:.4f} | alpha fits: s0={a0:.4f}({n0}) "
          f"s1={a1:.4f}({n1}) e={ae:.4f}({ne}) -> alpha={alpha:.4f}")
    s0h, s1h, eh = b0h / alpha, b1h / alpha, bh_ / alpha
    if s0t is not None:
        e0 = err_stats(s0h, s0t, "R1 s0 (descaled)")
        e1 = err_stats(s1h, s1t, "R1 s1 (descaled)")
        e2 = err_stats(eh,  et,  "R2 e (hint, descaled)")
        b0n_h1 = solve_freq(Dacc, b0n)
        ehn = solve_freq(Dacc, bhn)
        err_stats(b0n_h1, s0t, "NEG s0 (coins)")
        err_stats(ehn,  et,  "NEG e (coins)")
        a_true = float(b0h @ s0t / (s0t @ s0t))
        print(f"[score] truth-based alpha = {a_true:.4f}")
    if final:
        out = np.concatenate([np.rint(s0h).astype(np.int8),
                              np.rint(s1h).astype(np.int8),
                              np.rint(eh).astype(np.int8)])
        out.tofile("recovered.bin")
        # raw (unrounded) estimates for the corrector, v2 layout:
        # [Grow int64 x512 (z1-route Gram, circulant row)][g0 f64 x512][g1 f64 x512]
        # [a0][a1][ae][sig0][sighe] f32 + [b0][b1][be] f32x512 (zeta-domain RHS)
        # The spatial matched filters g = (1/n) P^H b_zeta (normal-equation RHS).
        a0f = a0 if a0 > 0 else alpha
        a1f = a1 if a1 > 0 else alpha
        aef = ae if ae > 0 else alpha
        sighe = sigma_noise * float(np.sqrt(1.0 + (37.7/154.2)**2))  # hint: +quantization
        zet_m = np.arange(N)
        phases = np.exp(-1j * np.pi * np.outer(zet_m, 2 * zet_m + 1) / N)  # zeta_k^{-m}
        # spatial matched filters (normal-equation RHS): g = (1/n) P^H b_acc
        g0s = (phases @ b0 / N).real
        g1s = (phases @ b1 / N).real
        # spatial-domain LS solutions (b0h/b1h/bh_ are already (1/n)P^H(...)/D)
        xz0 = b0h
        xz1 = b1h
        ez = bh_
        raw = np.concatenate([
            Dacc.astype(np.float64), g0s.astype(np.float64), g1s.astype(np.float64),
            np.array([a0f, a1f, aef, sigma_noise, sighe], dtype=np.float64),
            xz0.astype(np.float64), xz1.astype(np.float64), ez.astype(np.float64)])
        raw.tofile("raw_estimates.bin")
        print("wrote recovered.bin + raw_estimates.bin — next: "
              "corrector <pk> raw_estimates.bin recovered.bin; "
              "forge fixup <pk> recovered.bin; forge <pk> recovered.bin 20")

if __name__ == "__main__":
    main()
