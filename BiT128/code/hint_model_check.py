# hint_model_check.py — tier-1 validation of the hint-channel model.
#
# Model under test (derived from sign.c / poly.c / sample.c):
#   wa = w - z2 + b0*c_signed (+ c_signed on row 0 when b=0)  [hint argument]
#      = base + (R * c_signed)                                [y2 cancels]
#   w  = base + y2
#   h[j] = fold40(cell(w[j]) - cell(wa[j])),  cell(x) = ((x mod q)+341)//682
#   => h[j] is a 7-level quantization of Delta[j] = y2[j] - a[j],
#      a[j] = (R*c_signed)[j] (+ c_signed[j] on row 0 when b=0),
#      y2 ~ Tri(+/-2047) fresh randomness, base uniform mod q.
#   => E[h | a] ~= -a/682 (slope -1/682, small wrap losses).
#
# Checks on REAL accepted signatures (first NSIG of the 1M dump):
#   1. oracle R from ground truth satisfies R == b0 - e (algebra regression);
#   2. empirical E[h | a] (signed-a bins) against the exact-cell Monte-Carlo table;
#   3. pooled first-order signal T = mean(-a * h * eps) vs permutation control.
import json
import numpy as np

D, Q, TAU = 256, 26881, 30
G2, G2HALF = 682, 341
NSIG = 100_000
ROW = 4609
DUMP = "tmp/bit128_1m/dump.bin"
GROUND = "tmp/bit128_1m/ground.bin"
OUT = "tmp/attack2/hint_model_check.json"

rng = np.random.default_rng(20260926)

# ---------------------------------------------------------------- ground truth
g = np.fromfile(GROUND, dtype="<i2").astype(np.int64)
s0 = g[:3 * D].reshape(3, D)
e_vec = g[3 * D:6 * D].reshape(3, D)
b1 = g[6 * D:9 * D].reshape(3, D)
b0 = g[9 * D:12 * D].reshape(3, D)
A0 = g[12 * D:].reshape(3, 3, D)

def negacyc(a, s):
    conv = np.convolve(a, s)
    conv = np.concatenate([conv, [0]])          # index 511 is always zero
    return (conv[:D] - conv[D:2 * D]) % Q

tv = np.zeros((3, D), np.int64)
for i in range(3):
    acc = np.zeros(D, np.int64)
    for p in range(3):
        acc = (acc + negacyc(A0[i, p], s0[p])) % Q
    tv[i] = acc

def center(x):
    return ((x % Q) + Q // 2) % Q - Q // 2

R = np.stack([center(tv[i] - 16 * (b1[i] % Q)) for i in range(3)])
alg_exact = bool(np.array_equal(R, b0 - e_vec))
print(f"[algebra] R == b0 - e exactly: {alg_exact}", flush=True)

# ---------------------------------------------------------------- MC model table
# discrete Tri(+/-2047): P(t) proportional to 2048 - |t|
W = np.concatenate([np.arange(1, 2048), [2048], np.arange(2047, 0, -1)]).astype(np.float64)
W /= W.sum()
Wc = np.cumsum(W)
T_GRID = np.concatenate([np.arange(-2047, 0), [0], np.arange(1, 2048)])

def cell(x):
    return ((x % Q) + G2HALF) // G2

A_MAX = 150
MC_N = 400_000
mc_mean_pos = np.zeros(A_MAX + 1)       # E[h | a = +v], v = 0..150
mc_tab_pos = np.zeros((A_MAX + 1, 7))   # P(h=k | a=+v), k = -3..3
b_samp = rng.integers(0, Q, size=MC_N)
cell_b = cell(b_samp)
y_samp = T_GRID[np.searchsorted(Wc, rng.random(MC_N))]
for v in range(A_MAX + 1):
    hraw = cell(b_samp + y_samp - v) - cell_b
    h = np.where(hraw > 20, hraw - 40, np.where(hraw < -20, hraw + 40, hraw))
    mc_mean_pos[v] = h.mean()
    sel = np.abs(h) <= 3
    mc_tab_pos[v] = np.bincount(h[sel] + 3, minlength=7) / sel.sum()
# odd symmetry (wrap losses are O(|a|/q), absorbed as model residual)
mc_mean_signed = lambda a: np.where(a >= 0, mc_mean_pos[np.clip(a, 0, A_MAX)],
                                    -mc_mean_pos[np.clip(-a, 0, A_MAX)])
fit = np.polyfit(np.arange(1, 61), mc_mean_pos[1:61], 1)
slope_per_a = fit[0]
kappa = -slope_per_a * G2
print(f"[model] dE[h|a]/da = {slope_per_a:.6e}  (theory -1/682 = {-1/G2:.6e}),  "
      f"kappa = {kappa:.4f}", flush=True)

# ---------------------------------------------------------------- real data
raw = np.memmap(DUMP, dtype=np.uint8, mode="r", shape=(NSIG, ROW))
c_can = raw[:, 0:512].copy().view("<i2").reshape(NSIG, D).astype(np.int64)
b_int = raw[:, 512].astype(np.int64)
h_all = raw[:, 3073:4609].copy().view("<i2").reshape(NSIG, 3, D).astype(np.int64)
eps = 1 - 2 * b_int                       # oracle bimodal sign (scoring only)

# a[n,i,j] = eps_n*(R_i*c_can_n)[j] + (i==0)*(b_n==0)*eps_n*c_can_n[j]
c_ext = np.concatenate([c_can, -c_can], axis=1).astype(np.float64)   # (N,512)
Rf = np.fft.rfft(np.concatenate([R, np.zeros((3, D))], axis=1), axis=1)  # (3,512)
Cf = np.fft.rfft(c_ext, axis=1)
a_all = np.empty((3, NSIG, D))
for i in range(3):
    conv = np.fft.irfft(Rf[i][None, :] * Cf, n=512, axis=1)
    a_all[i] = conv[:, :D]
    del conv
a_all[0] += ((b_int == 0) * eps)[:, None] * c_can                         # row-0 b term
print(f"[data] |a| max = {np.abs(a_all).max():.0f}, std = {a_all.std():.2f}", flush=True)

# empirical E[h|a] on signed-a bins
BIN = 2 * A_MAX + 1
ai = np.clip(np.rint(a_all).astype(np.int64), -A_MAX, A_MAX) + A_MAX     # (3,N,256)
emp_mean = np.zeros((3, BIN)); emp_cnt = np.zeros((3, BIN), np.int64)
for i in range(3):
    hrow = h_all[:, i, :]
    emp_mean[i] = np.bincount(ai[i].ravel(), weights=hrow.ravel().astype(np.float64),
                              minlength=BIN)
    emp_cnt[i] = np.bincount(ai[i].ravel(), minlength=BIN)
emp_mean /= np.maximum(emp_cnt, 1)
a_axis = np.arange(-A_MAX, A_MAX + 1)
model_line = np.array([mc_mean_signed(np.array([v]))[0] for v in a_axis])
cmp_mask = np.abs(a_axis) <= 60
model_err = float(np.abs(emp_mean[:, cmp_mask] - model_line[None, cmp_mask]).max())
print(f"[check] max |E_emp[h|a] - E_model[h|a]| over |a|<=60: {model_err:.4f}", flush=True)

# pooled first-order signal: T = mean(-a * h * eps) per (n,j) observation
T = np.zeros(3)
for i in range(3):
    T[i] = float(np.mean(eps[:, None] * h_all[:, i, :] * (-a_all[i])))
perm = rng.permutation(NSIG)
T_ctrl = np.zeros(3)
for i in range(3):
    T_ctrl[i] = float(np.mean(eps[perm][:, None] * h_all[:, i, :] * (-a_all[i])))
print(f"[signal] T rows = {T}, perm control = {T_ctrl}", flush=True)

out = {
    "nsig": NSIG, "algebra_R_equals_b0_minus_e": alg_exact,
    "mc_slope_per_a": float(slope_per_a), "theory_slope": -1.0 / G2,
    "kappa": float(kappa),
    "max_abs_emp_minus_model_mean_h_a_le_60": model_err,
    "T_rows": T.tolist(), "T_perm_control": T_ctrl.tolist(),
    "mc_mean_pos_head": mc_mean_pos[:16].tolist(),
    "emp_mean_row0_head": emp_mean[0][A_MAX:A_MAX + 16].tolist(),
    "emp_mean_row1_head": emp_mean[1][A_MAX:A_MAX + 16].tolist(),
    "emp_mean_row2_head": emp_mean[2][A_MAX:A_MAX + 16].tolist(),
    "mc_tab_pos_a0": mc_tab_pos[0].tolist(),
    "mc_tab_pos_a60": mc_tab_pos[60].tolist(),
    "R_std_per_row": R.std(axis=1).tolist(),
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(f"[done] wrote {OUT}", flush=True)
