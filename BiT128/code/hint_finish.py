# hint_finish.py — final hint-channel pipeline (public data only).
#
#   1. Exact field operator Fop by Monte-Carlo over DISTINCT challenge
#      positions (SampleInBall marginal), full wrap algebra, vectorized.
#   2. R-hat = (s*M*I + N*Fop)^-1 * (-682*Phi + b0corr), s = label scale.
#      Oracle arm: s = 1.  Attack arm: s fitted PUBLICLY by integer
#      clustering of R-hat(s) over a 1-D grid (no oracle quantities).
#   3. A0^{-1}-consistency finisher: for the few coordinates whose R is
#      misclassified, test single-coordinate corrections delta and accept
#      the unique correction whose solved s0 lands entirely in {-1,0,1}.
#   4. s0 = A0^{-1}(16*b1 + R) mod q  ->  equivalent key (e_hat, b0_hat).
import json
import sys
import numpy as np

D, Q, TAU = 256, 26881, 30
G2 = 682
DATA = sys.argv[1] if len(sys.argv) > 1 else "real"
SUF = sys.argv[2] if len(sys.argv) > 2 else "1000000"
NSIG = int(SUF)
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 150000
rng = np.random.default_rng(20260928)

g16 = np.fromfile("tmp/bit128_1m/ground.bin", dtype="<i2").astype(np.int64)
s0_true = g16[:3 * D].reshape(3, D)
b1_true = g16[6 * D:9 * D].reshape(3, D)
A0 = g16[12 * D:].reshape(3, 3, D)

def negacyc(a, s):
    conv = np.convolve(a, s)
    conv = np.concatenate([conv, [0]])
    return (conv[:D] - conv[D:2 * D]) % Q

def center(x):
    return ((x % Q) + Q // 2) % Q - Q // 2

tv = np.zeros((3, D), np.int64)
for i in range(3):
    acc = np.zeros(D, np.int64)
    for p in range(3):
        acc = (acc + negacyc(A0[i, p], s0_true[p])) % Q
    tv[i] = acc
R_true = np.stack([center(tv[i] - 16 * (b1_true[i] % Q)) for i in range(3)])

# ---------------------------------------------------------------- MC operator
rows = np.arange(D)
mrow = rows[:, None]
def one_draw_mat():
    mat = np.zeros(D * D)
    pos = rng.choice(D, size=TAU, replace=False)
    for t in range(TAU):
        j = (rows + pos[t]) % D
        flip = np.where(j < rows, -1.0, 1.0)
        vm = (j[:, None] - pos[None, :]) % D
        cm = np.where(j[:, None] < pos[None, :], -1.0, 1.0) * flip[:, None]
        cm[:, t] = 0.0
        mat += np.bincount((mrow * D + vm).ravel(), weights=cm.ravel(), minlength=D * D)
    return mat

Fop = np.zeros((D, D))
for it in range(NDRAW):
    Fop += one_draw_mat().reshape(D, D)
Fop /= NDRAW
np.save(f"tmp/attack2/Fop_{NDRAW}.npy", Fop)
print(f"[mc] draws={NDRAW} row-sum mean={Fop.sum(axis=1).mean():+.2f} "
      f"std={Fop.sum(axis=1).std():.2f}", flush=True)

# ---------------------------------------------------------------- solve arms
Phi = np.load(f"tmp/attack2/phi_hint_{DATA}_{SUF}.npy").astype(np.float64)
M = NSIG * TAU
m_axis = np.arange(D)
b0corr = 0.5 * M * (1.0 - 2.0 * m_axis / D)

def rhs_of(arm):
    rhs = np.zeros((3, D))
    for i in range(3):
        phi = Phi[arm, i].copy()
        if i == 0:
            phi = phi + b0corr / 682.0
        rhs[i] = -G2 * phi
    return rhs

def lambda_caps():
    """Recompute the z0-channel LLR sum per signature (public data)."""
    z_grid = np.arange(-7, 8)
    with np.errstate(divide="ignore", invalid="ignore"):
        el = np.log(np.where(np.abs(z_grid - 1) <= 7, (8 - np.abs(z_grid - 1)) / 64.0, 0.0))             - np.log(np.where(np.abs(z_grid + 1) <= 7, (8 - np.abs(z_grid + 1)) / 64.0, 0.0))
    el = np.where(np.isfinite(el), el, np.sign(el) * 40.0)
    el = np.clip(el, -40.0, 40.0)
    LUT = np.zeros((15, 2)); LUT[:, 1] = el
    ROWt = 4609
    raw = np.memmap("tmp/bit128_1m/dump.bin" if DATA == "real" else "tmp/sim1m/dump.bin",
                    dtype=np.uint8, mode="r", shape=(NSIG, ROWt))
    B = 50_000
    out = np.empty(NSIG)
    for st in range(0, NSIG, B):
        sl = slice(st, min(st + B, NSIG))
        cc = raw[sl, 0:512].copy().view("<i2").reshape(-1, D).astype(np.int64)
        z0 = raw[sl, 1025:1537].copy().view("<i2").reshape(-1, D).astype(np.int64)
        out[sl] = LUT[z0 + 7, cc].sum(axis=1)
    return out

def em_label_accuracy(lam):
    """Public estimate of P(sign(Lambda)=eps) from the symmetric mixture."""
    hist, edges = np.histogram(lam, bins=801, range=(-40.0, 40.0))
    g = hist.astype(np.float64) + 1e-9
    mid = len(f_hist := g) // 2
    # asymmetric init: lean the + component toward Lambda>0 to escape the
    # symmetric degenerate fixed point of the mixture EM
    lean = np.where(np.arange(len(g)) > mid, 1.6, 0.4)
    f = (g * lean)
    f /= f.sum()
    for _ in range(500):
        tot = f + f[::-1]
        p_plus = np.divide(f, tot, out=np.full_like(f, 0.5), where=tot > 0)
        f_new = g * p_plus                    # + component density (unnormalized)
        f_new /= f_new.sum()
        if np.max(np.abs(f_new - f)) < 1e-13:
            f = f_new
            break
        f = f_new
    # accuracy = P(Lambda>0 | eps=+) + 0.5*P(Lambda=0 | eps=+)
    acc = float(f[mid + 1:].sum() + 0.5 * f[mid])
    return min(max(acc, 0.5), 1.0)

KAPPA = 1.0      # effective real-data slope (wrap asymmetry cancels in selection)

def solve_R(Fop_use, s, rhs):
    A = (s * KAPPA * M) * np.eye(D) + NSIG * Fop_use
    return np.stack([np.linalg.solve(A, rhs[i]) for i in range(3)]), A

def icm_snap(A, rhs, R0, iters=30):
    """Coordinate descent on ||A r - rhs||^2 over integers in [-9,8]."""
    r = np.clip(np.rint(R0), -9, 8).astype(np.float64)
    diag = np.diag(A).copy()
    for it in range(iters):
        grad = A @ r - rhs
        changed = 0
        for m in range(D):
            step = grad[m] / diag[m]
            base = r[m] - step
            for cand in (np.floor(base), np.ceil(base)):
                c = float(np.clip(cand, -9, 8))
                if c != r[m]:
                    newgrad = grad + A[:, m] * (c - r[m])
                    # accept only if the quadratic objective decreases
                    if (newgrad @ newgrad) < (grad @ grad):
                        grad = newgrad
                        r[m] = c
                        changed += 1
        if changed == 0:
            break
    return r

# ---------------- A0^-1 (computed once; used by the s-razor and finisher)
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
    if Aaug[piv, k] == 0:
        raise RuntimeError("singular A0")
    if piv != k:
        Aaug[[k, piv]] = Aaug[[piv, k]]
    inv = pow(int(Aaug[k, k]), Q - 2, Q)
    Aaug[k] = (Aaug[k] * inv) % Q
    col = Aaug[:, k].copy()
    col[k] = 0
    Aaug = (Aaug - col[:, None] * Aaug[k][None, :]) % Q
Ainv_m = Aaug[:, n:]
b1f = (16 * (b1_true % Q)).reshape(3 * D)

def s0_from_R(Rv):
    return (Ainv_m @ ((b1f + Rv.reshape(3 * D)) % Q)) % Q

def nbad_of(Rv):
    c = center(s0_from_R(Rv))
    return int(np.sum((c < -1) | (c > 1)))

def scan1(Rbase, worst=None):
    Rflat = Rbase.reshape(3 * D)
    best = None
    for m in range(3 * D):
        base = int(Rflat[m])
        for delta in range(-9, 10):
            newv = base + delta
            if newv < -9 or newv > 8 or delta == 0:
                continue
            Rtry = Rflat.copy()
            Rtry[m] = newv
            nb = nbad_of(Rtry)
            if best is None or nb < best[0]:
                best = (nb, [(m, newv)])
                if nb == 0:
                    return best
    return best

def scan2(Rbase, worst):
    Rflat = Rbase.reshape(3 * D)
    best = None
    for ai in range(len(worst)):
        for bi in range(ai + 1, len(worst)):
            ma, mb = int(worst[ai]), int(worst[bi])
            for da in range(-3, 4):
                newa = int(Rflat[ma]) + da
                if newa < -9 or newa > 8:
                    continue
                for db in range(-3, 4):
                    newb = int(Rflat[mb]) + db
                    if newb < -9 or newb > 8:
                        continue
                    Rtry = Rflat.copy()
                    Rtry[ma] = newa
                    Rtry[mb] = newb
                    nb = nbad_of(Rtry)
                    if best is None or nb < best[0]:
                        best = (nb, [(ma, newa), (mb, newb)])
                        if nb == 0:
                            return best
    return best

def scan3(Rbase, worst):
    import itertools
    Rflat = Rbase.reshape(3 * D)
    best = None
    for combo in itertools.combinations(range(len(worst)), 3):
        ma, mb, mc = (int(worst[j]) for j in combo)
        for da in range(-2, 3):
            newa = int(Rflat[ma]) + da
            if newa < -9 or newa > 8:
                continue
            for db in range(-2, 3):
                newb = int(Rflat[mb]) + db
                if newb < -9 or newb > 8:
                    continue
                for dc in range(-2, 3):
                    newc = int(Rflat[mc]) + dc
                    if newc < -9 or newc > 8:
                        continue
                    Rtry = Rflat.copy()
                    Rtry[ma] = newa
                    Rtry[mb] = newb
                    Rtry[mc] = newc
                    nb = nbad_of(Rtry)
                    if best is None or nb < best[0]:
                        best = (nb, [(ma, newa), (mb, newb), (mc, newc)])
                        if nb == 0:
                            return best
    return best


out = {"data": DATA, "nsig": NSIG, "ndraw": NDRAW, "arms": {}}
res_all = {}
for arm, arm_name in ((1, "oracle"), (0, "attack")):
    rhs = rhs_of(arm)
    if arm_name == "oracle":
        s_best = 1.0
        Rhat, A = solve_R(Fop, s_best, rhs)
        obj = float(np.cos(2 * np.pi * Rhat).mean())
    else:
        # fine clustering grid; each candidate gets the single-fix pass and is
        # scored by (post-single-fix nbad, then the clustering objective)
        cand = []
        for s in np.round(np.arange(0.900, 0.962, 0.002), 4):
            Rh, _ = solve_R(Fop, float(s), rhs)
            o = float(np.cos(2 * np.pi * Rh).mean())
            Ri = np.clip(np.rint(Rh), -9, 8).astype(np.int64)
            nb0 = nbad_of(Ri)
            if nb0 > 0:
                mg = np.abs(Rh - Ri).reshape(3 * D)
                w12 = np.argsort(mg)[::-1][:12]
                r1 = scan1(Ri, w12)
                nb = r1[0] if r1 is not None else nb0
                fx = r1[1] if (r1 is not None and r1[0] == 0) else []
                if nb > 0:
                    r2 = scan2(Ri, w12)
                    if r2 is not None and r2[0] < nb:
                        nb, fx = r2
            else:
                fx = []
            cand.append((nb, -o, float(s), Rh, Ri, fx))
            print(f"[{arm_name}] s={s:.3f} obj={o:.4f} raw_nbad={nb0} "
                  f"post12={nb}", flush=True)
            if nb == 0:
                break
        cand.sort(key=lambda t: (t[0], t[1]))
        nb_best, negobj, s_best, Rhat, Rbase, fix_list = cand[0]
        obj = -negobj
        A = s_best * KAPPA * M * np.eye(D) + NSIG * Fop
        print(f"[{arm_name}] grid best: s={s_best:.3f} post12 nbad={nb_best}",
              flush=True)
    Rint = np.clip(np.rint(Rhat), -9, 8).astype(np.int64)
    acc = float((Rint == R_true).mean())
    # ICM integer snap (per row)
    Rint_icm = np.stack([icm_snap(A, rhs[i], Rhat[i]).astype(np.int64)
                         for i in range(3)])
    acc_icm = float((Rint_icm == R_true).mean())
    if arm_name == "oracle":
        use = Rint_icm if acc_icm >= acc else Rint
    else:
        use = Rbase.copy()
        for m, newv in fix_list:
            use.reshape(3 * D)[m] = newv
    res_all[arm_name] = {"s_fit": s_best, "cluster_obj": obj,
                         "direct_acc": acc, "n_wrong": int((Rint != R_true).sum()),
                         "icm_acc": acc_icm, "icm_n_wrong": int((Rint_icm != R_true).sum()),
                         "per_row": [float((Rint[i] == R_true[i]).mean()) for i in range(3)]}
    print(f"[{arm_name}] s={s_best:.3f} direct acc={acc:.4f} "
          f"({res_all[arm_name]['n_wrong']}/768) icm acc={acc_icm:.4f} "
          f"({res_all[arm_name]['icm_n_wrong']}/768)", flush=True)

    # ---------------- consistency finisher
    Rcur = use.copy()
    nbad = nbad_of(Rcur)

    if arm_name == "attack":
        print(f"[{arm_name}] finisher start nbad={nbad}", flush=True)
    if nbad > 0:
        margins = np.abs(Rhat - Rint).reshape(3 * D)
        worst = np.argsort(margins)[::-1][:12]
        for stage, fn in ((1, scan1), (2, scan2), (3, scan3)):
            if nbad == 0:
                break
            res = fn(Rcur, worst)
            if res is not None and res[0] < nbad:
                nb, fixes = res
                if nb == 0:
                    for m, newv in fixes:
                        Rcur.reshape(3 * D)[m] = newv
                    nbad = 0
                    print(f"[{arm_name}] stage-{stage} closed (fixes={fixes})",
                          flush=True)
                else:
                    print(f"[{arm_name}] stage-{stage} best nb={nb} (not applied)",
                          flush=True)
                    if stage == 3:
                        nbad = res[0] if res[0] < nbad else nbad
    s0v = s0_from_R(Rcur)
    c = center(s0v)
    s0_exact = bool(np.array_equal(c.reshape(3, D), s0_true))
    n_match = int((c.reshape(3, D) == s0_true).sum())
    res_all[arm_name].update({
        "finisher_nbad_left": int(nbad),
        "s0_match": n_match, "s0_exact": s0_exact,
    })
    print(f"[{arm_name}] finisher: nbad_left={int(nbad)} -> s0 match {n_match}/768 "
          f"exact={s0_exact}", flush=True)

    if s0_exact:
        ehat = np.where(Rcur >= 8, -1, np.where(Rcur <= -9, 1, 0)).astype(np.int64)
        b0hat = Rcur + ehat
        assert np.all((b0hat >= -8) & (b0hat < 8)) and np.abs(ehat).max() <= 1
        with open(f"tmp/attack2/key_hat_h_{arm_name}.bin", "wb") as f:
            for p in range(3):
                f.write(c.reshape(3, D)[p].astype(np.int16).tobytes())
            for i in range(3):
                f.write(ehat[i].astype(np.int16).tobytes())
            for i in range(3):
                f.write(b0hat[i].astype(np.int16).tobytes())
        res_all[arm_name]["key_written"] = True
        res_all[arm_name]["e_hat_nonzero"] = int((ehat != 0).sum())
        print(f"[{arm_name}] equivalent key written (e_hat nonzero at "
              f"{int((ehat != 0).sum())} coords)", flush=True)

out["arms"] = res_all
with open(f"tmp/attack2/hint_finish_{DATA}_{SUF}.json", "w") as f:
    json.dump(out, f, indent=1)
print("[done]", flush=True)
