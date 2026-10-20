#!/usr/bin/env python3
"""em_test.py — EM soft-label refinement for the DARTS128 (1-2b) attack.

Model: z1[i] = y1[i] + sigma * (s_i * c),  sigma = (-1)^b in {+-1},
y1 ~ ball noise with per-coord sigma_y = 154.2. The hash branch tell resolves
sigma on only ~21% (clean) signatures; the remaining 79% are ambiguous. But
given a current estimate s_hat, the DIRECT likelihood resolves most of them:
    LLR_j = 2 * (<s_hat0*c, z1[0]> + <s_hat1*c, z1[1]>) / sigma_y^2
    m_j = E[sigma_j | z1] = tanh(LLR_j / 2)
For DARTS-128, ||s*c|| ~ 191 gives per-signature LLR mean ~3.1 (comp 0 only)
/ ~6.2 (both components) -> ~4% posterior flip rate -> ambiguous signatures
carry ~85% of clean information. EM: rebuild the soft RHS with m_j for ALL
signatures, re-solve, repeat. The Gram matrix is label-free (computed once).

usage: em_test.py <pub> <labels> <truth> <nsigs> [iters]
"""
import sys
import numpy as np

Q = 130817
N = 512
L = 2
REC = 1 + 64 + 2 * N * 2 + N
SIG_Y2 = 154.2 ** 2

def zeta_batch(V):
    """V: (m, N) real -> (m, N) complex conjugate-odd-bin transform."""
    vp = np.zeros((V.shape[0], 2 * N))
    vp[:, :N] = V
    return np.conj(np.fft.fft(vp, axis=1)[:, 1::2])

def solve_freq(Dacc, bvec):
    m = np.arange(N)
    phases = np.exp(-1j * np.pi * np.outer(m, 2 * m + 1) / N)
    return (phases @ (bvec / Dacc) / N).real

def main():
    pubfile, labelfile, truthfile, nsigs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    truth = np.fromfile(truthfile, dtype=np.int8).astype(np.float64)

    Dclean = np.zeros(N)
    Dall = np.zeros(N)
    b0_hard = np.zeros(N, dtype=complex)
    b1_hard = np.zeros(N, dtype=complex)
    m = np.zeros(nsigs)
    decided = np.zeros(nsigs, dtype=bool)                       # label state: 0 = ambiguous
    f = open(pubfile, "rb"); flab = open(labelfile, "rb")
    done = 0
    while done < nsigs:
        chunk = min(16384, nsigs - done)
        arr = np.frombuffer(f.read(REC * chunk), dtype=np.uint8).reshape(chunk, REC)
        lab = np.frombuffer(flab.read(chunk), dtype=np.uint8)
        cbits = np.unpackbits(arr[:, 1:65], axis=1, bitorder="little")[:, :N]
        z1 = arr[:, 65:65 + 2*N*2].copy().view(np.int16).reshape(chunk, 2, N).astype(np.float64)
        Tc = zeta_batch(cbits.astype(np.float64))
        ab = (Tc * np.conj(Tc)).real
        Dall += ab.sum(axis=0)
        Tz0 = zeta_batch(z1[:, 0, :])
        Tz1 = zeta_batch(z1[:, 1, :])
        sel = lab < 2
        Dclean += ab[sel].sum(axis=0)
        sgn = 1.0 - 2.0 * lab[sel].astype(np.float64)
        b0_hard += (sgn[:, None] * np.conj(Tc[sel]) * Tz0[sel]).sum(axis=0)
        b1_hard += (sgn[:, None] * np.conj(Tc[sel]) * Tz1[sel]).sum(axis=0)
        mj = np.zeros(chunk); mj[sel] = sgn
        m[done:done+chunk] = mj
        decided[done:done+chunk] = sel
        done += chunk
    f.close(); flab.close()
    print(f"pass1 done: hard labels {int(np.sum(m != 0))}/{nsigs}")

    alpha = 0.367                              # conditioning shrinkage (measured)
    b_soft0 = b0_hard.copy()
    b_soft1 = b1_hard.copy()
    s_est = np.zeros(2 * N)
    for it in range(iters + 1):
        G = Dclean if it == 0 else Dall
        sh0 = solve_freq(G, b_soft0)
        sh1 = solve_freq(G, b_soft1)
        # oracle descale for scoring (alpha from truth projection)
        a0 = float(sh0 @ truth[:N] / (truth[:N] @ truth[:N]))
        a1 = float(sh1 @ truth[N:2*N] / (truth[N:2*N] @ truth[N:2*N]))
        s_est = np.concatenate([sh0 / a0, sh1 / a1])
        wrong = int(np.sum(np.rint(s_est).astype(np.int64) != truth[:2*N].astype(np.int64)))
        rms = float(np.sqrt(((s_est - truth[:2*N]) ** 2).mean()))
        print(f"iter {it}: rms={rms:.4f}  wrong={wrong}/1024  (alpha {a0:.4f}/{a1:.4f})")
        if it == iters:
            break

        # EM pass: posterior soft labels for ambiguous signatures
        S = zeta_batch(s_est.reshape(2, N))          # (2, N) zeta of each component
        S0c = np.conj(S[0]); S1c = np.conj(S[1])
        b_soft0 = np.zeros(N, dtype=complex)
        b_soft1 = np.zeros(N, dtype=complex)
        f = open(pubfile, "rb"); flab = open(labelfile, "rb")
        done = 0
        namb = 0
        while done < nsigs:
            chunk = min(16384, nsigs - done)
            arr = np.frombuffer(f.read(REC * chunk), dtype=np.uint8).reshape(chunk, REC)
            lab = np.frombuffer(flab.read(chunk), dtype=np.uint8)
            cbits = np.unpackbits(arr[:, 1:65], axis=1, bitorder="little")[:, :N]
            z1 = arr[:, 65:65 + 2*N*2].copy().view(np.int16).reshape(chunk, 2, N).astype(np.float64)
            Tc = zeta_batch(cbits.astype(np.float64))
            Tz0 = zeta_batch(z1[:, 0, :])
            Tz1 = zeta_batch(z1[:, 1, :])
            # u_j = <s0*c, z0> + <s1*c, z1>  (Parseval: <a,b> = (1/n) sum conj(Pa) Pb)
            u_tot = ((np.conj(S[0]) * Tc * Tz0).sum(axis=1) +
                     (np.conj(S[1]) * Tc * Tz1).sum(axis=1)).real / N
            amb = ~decided[done:done+chunk]
            mj = m[done:done+chunk].copy()
            llr = 2 * u_tot / SIG_Y2
            mj[amb] = np.tanh(llr[amb] / 2)
            namb += int(amb.sum())
            b_soft0 += (mj[:, None] * np.conj(Tc) * Tz0).sum(axis=0)
            b_soft1 += (mj[:, None] * np.conj(Tc) * Tz1).sum(axis=0)
            m[done:done+chunk] = mj
            done += chunk
        f.close(); flab.close()
        print(f"  EM pass: soft labels applied to {namb} ambiguous signatures")

if __name__ == "__main__":
    main()
