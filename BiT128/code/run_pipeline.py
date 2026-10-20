#!/usr/bin/env python3
# run_pipeline.py — BiT-128 equivalent-key recovery from PUBLIC data only.
#
# This is the standalone pipeline of our attack (hint channel + operator
# inversion + full-solution consistency corrector). Inputs are the two
# accumulated hint-channel statistics (obtained from public signatures by
# code/hint_channel_fast.py), the Monte-Carlo field operator, and the PUBLIC
# key material (pub_A0.bin / pub_b1.bin, produced from pk.bin by pubmat).
#
# Route (ours, distinct from the z1-regression route of the concurrent work):
#   1. per-row label-scale fit by integer clustering (public objective)
#   2. solve (s*M*I + N*Fop) R = -gamma2 * Phi (+ row-0 b-term)
#   3. round R to [-9, 8] and validate the FULL window R in [-9,8]^768
#   4. staged consistency corrector (OURS): full mod-q solve
#      s0 = A0^-1 (16 b1 + R); accept a candidate correction ONLY if the
#      entire 768-dim solution lands in {-1,0,1} (nbad = 0). No confidence
#      windows, no Gray-code enumeration.
#   5. complete the equivalent key (e_hat, b0_hat).
#
# Scoring against the real key (optional, --score ground.bin) is never used
# for any decision.
import argparse
import hashlib
import numpy as np

D, Q, TAU, G2 = 256, 26881, 30, 682

def center(x):
    return ((x % Q) + Q // 2) % Q - Q // 2

def load_pub(path, n):
    return np.fromfile(path, dtype="<i2").astype(np.int64)[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--datasets", nargs="+",
                    default=["phi_hint_real_1000000", "phi_hint_sim3m"],
                    help="Phi accumulators to combine (attack arm 0)")
    ap.add_argument("--nsig", type=int, default=4_000_000,
                    help="total signatures across the combined datasets")
    ap.add_argument("--score", default=None,
                    help="optional ground.bin for AFTER-THE-FACT scoring only")
    ap.add_argument("--out", default="key_hat.bin")
    args = ap.parse_args()

    A0 = load_pub(f"{args.artifacts}/pub_A0.bin", 3 * 3 * D).reshape(3, 3, D)
    b1 = load_pub(f"{args.artifacts}/pub_b1.bin", 3 * D).reshape(3, D)

    Phi = np.zeros((3, D))
    for name in args.datasets:
        Phi += np.load(f"{args.artifacts}/{name}.npy")[0].astype(np.float64)
    N = args.nsig
    M = N * TAU
    print(f"[data] datasets={args.datasets}  N={N}  M={M}", flush=True)

    Fop = np.load(f"{args.artifacts}/Fop_150000.npy").astype(np.float64)
    m_axis = np.arange(D)
    N_F = N * Fop
    b1f = (16 * (b1 % Q)).reshape(3 * D)

    # ---------------- A0^-1 (mod q, Gauss-Jordan) ----------------
    r_ = np.arange(D)
    Mx = np.zeros((3 * D, 3 * D), np.int64)
    for i in range(3):
        for p in range(3):
            idxm = (r_[:, None] - r_[None, :]) % D
            sgn = np.where(r_[None, :] > r_[:, None], -1, 1)
            Mx[i * D:(i + 1) * D, p * D:(p + 1) * D] = A0[i, p][idxm] * sgn
    n = 3 * D
    Aaug = np.concatenate([Mx % Q, np.eye(n, dtype=np.int64)], axis=1).astype(np.int64)
    for k in range(n):
        piv = k + int(np.argmax(Aaug[k:, k] != 0))
        Aaug[[k, piv]] = Aaug[[piv, k]]
        inv = pow(int(Aaug[k, k]), Q - 2, Q)
        Aaug[k] = (Aaug[k] * inv) % Q
        col = Aaug[:, k].copy()
        col[k] = 0
        Aaug = (Aaug - col[:, None] * Aaug[k][None, :]) % Q
    Ainv = Aaug[:, n:]

    def s0_from_R(Rv):
        return center((Ainv @ ((b1f + Rv.reshape(3 * D)) % Q)) % Q)

    def nbad_of(Rv):
        c = s0_from_R(Rv)
        return int(np.sum((c < -1) | (c > 1)))

    # ---------------- per-row scale fit + operator inversion ----------------
    Rhat = np.zeros((3, D))
    s_row = []
    for i in range(3):
        phi = Phi[i].copy()
        if i == 0:
            phi = phi + (0.924 * 0.5 * M * (1.0 - 2.0 * m_axis / D)) / 682.0
        rhs = -G2 * phi
        best = None
        for s in np.arange(0.88, 1.001, 0.001):
            Rh = np.linalg.solve(s * M * np.eye(D) + N_F, rhs)
            o = float(np.cos(2 * np.pi * Rh).mean())
            if best is None or o > best[0]:
                best = (o, float(s), Rh)
        o, s, Rh = best
        s_row.append(s)
        Rhat[i] = Rh
        print(f"[fit] row{i}: s={s:.3f} cluster_obj={o:.4f}", flush=True)

    Rint = np.clip(np.rint(Rhat), -9, 8).astype(np.int64)
    nbad = nbad_of(Rint)
    print(f"[solve] window-valid R coordinates: {768 - nbad}/768 "
          f"(nbad={nbad})", flush=True)

    # ---------------- staged corrector (OURS: full-solution razor) ----------
    def scan(k, W):
        margins = np.abs(Rhat - Rint).reshape(3 * D)
        worst = list(np.argsort(margins)[::-1][:W])
        best = None
        base = Rcur.reshape(3 * D)
        import itertools as _it
        for combo in _it.combinations(worst, k):
            for dset in _it.product(range(-9, 10), repeat=k):
                if all(dd == 0 for dd in dset):
                    continue
                Rtry = base.copy()
                ok = True
                for w, dd in zip(combo, dset):
                    newv = int(base[w]) + dd
                    if newv < -9 or newv > 8:
                        ok = False
                        break
                    Rtry[w] = newv
                if not ok:
                    continue
                nb = nbad_of(Rtry)
                if best is None or nb < best[0]:
                    best = (nb, list(zip(combo, dset)))
                    if nb == 0:
                        return best
        return best

    Rcur = Rint.copy()
    for k, W in ((1, 3 * D), (2, 12), (3, 10)):
        if nbad == 0:
            break
        res = scan(k, W)
        if res is not None and res[0] == 0:
            nb, fixes = res
            for w, dd in fixes:
                Rcur.reshape(3 * D)[w] += dd
            nbad = 0
            print(f"[corrector] stage k={k}: closed with {fixes}", flush=True)
            break
        print(f"[corrector] stage k={k}: best nbad="
              f"{res[0] if res else None} (not applied)", flush=True)

    s0v = s0_from_R(Rcur)
    assert nbad == 0, "recovery did not reach a fully consistent key"
    ehat = np.where(Rcur >= 8, -1, np.where(Rcur <= -9, 1, 0)).astype(np.int64)
    b0hat = Rcur + ehat
    assert np.all((b0hat >= -8) & (b0hat < 8)) and np.abs(ehat).max() <= 1

    key = (s0v.reshape(3, D).astype(np.int16).tobytes()
           + ehat.astype(np.int16).tobytes() + b0hat.astype(np.int16).tobytes())
    h = hashlib.sha256(key).hexdigest()
    with open(args.out, "wb") as f:
        f.write(key)
    print(f"[key] equivalent key written to {args.out}")
    print(f"[key] sha256 = {h}")
    print(f"[key] e_hat nonzero at {int((ehat != 0).sum())} coordinates")

    if args.score:
        g = np.fromfile(args.score, dtype="<i2").astype(np.int64)
        s0_true = g[:3 * D].reshape(3, D)
        print(f"[score] s0 coefficients matching the original key: "
              f"{int((s0v.reshape(3, D) == s0_true).sum())}/768 "
              f"(never used for any decision)")

if __name__ == "__main__":
    main()
