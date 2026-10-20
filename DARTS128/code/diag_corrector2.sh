#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
python3 - <<'PY'
import numpy as np
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.float64)
alpha, sig = 0.367, 0.0614                     # 1M-quality noise floor
rng = np.random.default_rng(7)
s0 = truth[:512].copy(); s1 = truth[512:1024].copy()
# force two near-boundary WRONG decisions: true 0 -> est +0.6 ; true 1 -> est +0.4 (rounds to 0)
i_a = int(np.flatnonzero(s0==0)[10]); s0[i_a] = 0.6
i_b = int(np.flatnonzero(s1==1)[10]); s1[i_b] = 0.4
b0 = alpha*s0 + rng.normal(0, sig, 512)
b1 = alpha*s1 + rng.normal(0, sig, 512)
be = alpha*truth[1024:] + rng.normal(0, sig, 512)
raw = np.concatenate([np.array([alpha, sig], dtype=np.float32),
                      b0.astype(np.float32), b1.astype(np.float32), be.astype(np.float32)])
raw.tofile('raw_sim1m.bin')
x0 = b0/alpha; x1 = b1/alpha
k0 = int(np.sum(np.rint(x0)!=s0+ (s0*0)))  # wrong = rounded != truth
err0 = np.rint(x0).astype(int) - s0.astype(int); err0[i_a] = 1  # est rounds 0.6->1 vs forced... recompute below
true0 = truth[:512].astype(int)
# after noise the planted coords: x0[i_a] ~ 0.6 -> rounds to 1 (true 0): wrong
# x1[i_b] ~ 0.4 -> rounds to 0 (true 1): wrong
k0 = int(np.sum(np.rint(x0).astype(int) != true0))
k1 = int(np.sum(np.rint(x1).astype(int) != truth[512:1024].astype(int)))
print(f'planted: s0[{i_a}] 0->~0.6 (rounds 1), s1[{i_b}] 1->~0.4 (rounds 0)')
print(f'simulated wrong coeffs: s0={k0} s1={k1} (incl. planted)')
PY
echo "== corrector on simulated 1M estimate with 2 planted near-boundary errors =="
./corrector pk.bin raw_sim1m.bin rec_sim1m.bin
cmp rec_sim1m.bin truth.bin && echo "SIM 1M PASS: corrector output == TRUTH (byte-identical)" || echo "SIM 1M FAIL"
