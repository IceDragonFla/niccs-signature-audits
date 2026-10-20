#!/usr/bin/env python3
"""diag.py — measure the actual noise/bias structure of the DARTS128 dump.

Answers:
  (1) empirical sigma of y coordinates (via std of z1 components; the +-sc
      contribution has std ~3, negligible)
  (2) scale of the labeled-LS estimate vs the true secret:
      alpha = <s_hat, s_true> / <s_true, s_true>  ->  shrinkage (1-kappa)
  (3) NEG-arm rms vs sigma_y/sqrt(N*tau) prediction
  (4) histogram of s_hat (real arm)
"""
import sys
import numpy as np

Q = 130817; N = 512; D_BITS = 10; W = Q / 2.0 ** D_BITS
REC = 1 + 64 + 2 * N * 2 + N

def zeta_transform(v):
    vp = np.zeros(2 * N); vp[:N] = v
    return np.conj(np.fft.fft(vp)[1::2])

def solve_freq(Dacc, bvec):
    m = np.arange(N)
    phases = np.exp(-1j * np.pi * np.outer(m, 2 * np.arange(N) + 1) / N)
    return (phases @ (bvec / Dacc) / N).real

def main():
    pubfile, labelfile, truthfile, nsigs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    truth = np.fromfile(truthfile, dtype=np.int8).astype(np.int64)
    s0t = truth[0:N]

    f = open(pubfile, "rb"); flab = open(labelfile, "rb")
    z0s, z1s, hs = [], [], []
    Dacc = np.zeros(N); b0 = np.zeros(N, dtype=complex); b0n = np.zeros(N, dtype=complex)
    rng = np.random.default_rng(1)
    done = nclean = 0
    while done < nsigs:
        chunk = min(16384, nsigs - done)
        arr = np.frombuffer(f.read(REC * chunk), dtype=np.uint8).reshape(chunk, REC)
        lab_all = np.frombuffer(flab.read(chunk), dtype=np.uint8)
        z1 = arr[:, 65:65 + 2*N*2].copy().view(np.int16).reshape(chunk, 2, N).astype(np.float64)
        h  = arr[:, 65 + 2*N*2:].copy().view(np.int8).reshape(chunk, N).astype(np.float64)
        if done == 0:
            z0s, z1s, hs = z1[:, 0].ravel(), z1[:, 1].ravel(), h.ravel()
        keep = np.flatnonzero(lab_all < 2)
        cbits_all = np.unpackbits(arr[:, 1:65], axis=1, bitorder="little")[:, :N]
        for t in keep:
            sgn = 1.0 - 2.0 * int(lab_all[t])
            coin = 1.0 - 2.0 * float(rng.integers(0, 2))
            cb = cbits_all[t]
            Tc = zeta_transform(cb); Tc2 = np.conj(Tc)
            Dacc += (Tc * Tc2).real
            Tz0 = zeta_transform(z1[t, 0])
            b0 += sgn * Tc2 * Tz0
            b0n += coin * Tc2 * Tz0
            nclean += 1
        done += chunk
    f.close(); flab.close()

    print(f"clean={nclean}")
    s_y = (np.std(z0s), np.std(z1s), np.std(hs))
    print(f"std(z1[0])={s_y[0]:.1f}  std(z1[1])={s_y[1]:.1f}  std(h)={s_y[2]:.3f}")
    print(f"-> sigma_y = {s_y[0]:.1f} (ball-uniform prediction 154.3; "
          f"noise-floor sigma/sqrt(N*tau) = {s_y[0]/np.sqrt(nclean*30):.4f})")

    shat = solve_freq(Dacc, b0)
    shn = solve_freq(Dacc, b0n)
    alpha = float(shat @ s0t / (s0t @ s0t))
    err = shat - s0t
    print(f"scale alpha=<s_hat,s_true>/<s_true,s_true> = {alpha:.4f}  "
          f"-> conditioning shrinkage kappa = {1-alpha:.4f}")
    print(f"rms(s_hat - s_true)={np.sqrt((err**2).mean()):.4f}  "
          f"rms(removed-scale)={np.sqrt(((shat - alpha*s0t)**2).mean()):.4f}")
    print(f"NEG-arm rms={np.sqrt(((shn - s0t)**2).mean()):.4f}  "
          f"std(shn)={np.std(shn):.4f}")
    print(f"residual after scale removal vs sigma/sqrt(N*tau): "
          f"{np.sqrt(((shat - alpha*s0t)**2).mean()):.4f} vs "
          f"{s_y[0]/np.sqrt(nclean*30):.4f}")
    # histogram of shat in 0.25 bins over [-2, 2]
    bins = np.arange(-2.0, 2.01, 0.25)
    hist, _ = np.histogram(shat, bins)
    print("s_hat histogram [-2,2] step .25:")
    for i in range(len(hist)):
        bar = "#" * int(60 * hist[i] / hist.max())
        print(f"  {bins[i]:+.2f}: {hist[i]:5d} {bar}")

if __name__ == "__main__":
    main()
