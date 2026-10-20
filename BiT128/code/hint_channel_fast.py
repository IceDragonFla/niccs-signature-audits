# hint_channel_fast.py — batched Phi accumulation for the hint channel.
# Same statistic as hint_channel.py but with the 30 per-t gathers fused into
# one (B*tau) gather, and the dump path/NSIG as arguments.
#   usage: python hint_channel_fast.py <dump.bin> <nsig> <tag>
# Saves tmp/attack2/phi_hint_<tag>.npy  (3, 3, 256) = (attack, oracle, perm).
import sys
import numpy as np

D, TAU = 256, 30
ROW = 4609
DUMP, NSIG, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3]

raw = np.memmap(DUMP, dtype=np.uint8, mode="r", shape=(NSIG, ROW))
B = 25_000
NBLK = (NSIG + B - 1) // B
Phi_hat = np.zeros((3, D), np.int64)
Phi_orc = np.zeros((3, D), np.int64)
Phi_perm = np.zeros((3, D), np.int64)
n_label_err = 0
n_b0 = 0
rng = np.random.default_rng(20260927)
m_range = np.arange(D)

# z0 label LUT (public constants)
z_grid = np.arange(-7, 8)
f0 = lambda x: np.where(np.abs(x) <= 7, (8 - np.abs(x)) / 64.0, 0.0)
with np.errstate(divide="ignore", invalid="ignore"):
    el = np.log(f0(z_grid - 1)) - np.log(f0(z_grid + 1))
el = np.clip(np.where(np.isfinite(el), el, np.sign(el) * 40.0), -40.0, 40.0)
LUT2 = np.zeros((15, 2))
LUT2[:, 1] = el

n_done = 0
for blk in range(NBLK):
    st = blk * B
    en = min(st + B, NSIG)
    b = en - st
    sl = slice(st, en)
    c_can = raw[sl, 0:512].copy().view("<i2").reshape(b, D).astype(np.int64)
    b_int = raw[sl, 512].astype(np.int64)
    z0 = raw[sl, 1025:1537].copy().view("<i2").reshape(b, D).astype(np.int64)
    h = raw[sl, 3073:4609].copy().view("<i2").reshape(b, 3, D)
    eps_orc = (1 - 2 * b_int).astype(np.int64)
    n_b0 += int((b_int == 0).sum())
    Lam = LUT2[z0 + 7, c_can].sum(axis=1)
    eps_hat = np.where(Lam > 0, 1, -1).astype(np.int64)
    n_label_err += int((eps_hat != eps_orc).sum())

    idx = np.nonzero(c_can)[1].reshape(b, TAU)
    psi_ext = np.concatenate([h, -h], axis=2)                    # (b,3,512)
    psiT = np.ascontiguousarray(psi_ext.transpose(2, 0, 1))      # (512,b,3)
    rows_rep = np.repeat(np.arange(b), TAU)                      # (b*tau,)
    idx_flat = idx.reshape(-1)                                   # (b*tau,)
    eps_perm = eps_orc[rng.permutation(b)]

    gath = psiT[idx_flat[:, None] + m_range[None, :], rows_rep[:, None], :]
    gath = np.ascontiguousarray(
        gath.reshape(b, TAU, D, 3).transpose(0, 1, 3, 2))         # (b,tau,3,256)
    Phi_hat += (gath * eps_hat[:, None, None, None]).sum(axis=(0, 1))
    Phi_orc += (gath * eps_orc[:, None, None, None]).sum(axis=(0, 1))
    Phi_perm += (gath * eps_perm[:, None, None, None]).sum(axis=(0, 1))
    n_done += b
    if (blk + 1) % 20 == 0:
        print(f"[pass] {n_done}/{NSIG}", flush=True)

M = NSIG * TAU
label_acc = 1.0 - n_label_err / NSIG
P0 = n_b0 / NSIG
print(f"[pass] done: M={M}, label_acc={label_acc:.4f}, P(b=0)={P0:.4f}", flush=True)
np.save(f"tmp/attack2/phi_hint_{TAG}.npy",
        np.stack([Phi_hat, Phi_orc, Phi_perm]).astype(np.int64))
with open(f"tmp/attack2/phi_hint_{TAG}_meta.json", "w") as f:
    import json
    json.dump({"tag": TAG, "nsig": NSIG, "label_acc": label_acc, "P_b0": P0}, f)
print(f"[done] wrote phi_hint_{TAG}.npy", flush=True)
