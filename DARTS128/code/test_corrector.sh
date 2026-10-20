#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
SRC=~/darts_work/DARTS128
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
gcc -O3 -march=native -std=gnu99 -fcommon -I$SRC -o corrector corrector.c $SRC/sign.c $SRCS -lm 2>&1 | grep -E 'error' && exit 1
echo "== build ok =="

# 1) regenerate raw estimates from the existing 500k dump
python3 attack.py pub.bin labels.bin truth.bin 500000 2>&1 | tail -2

# 2) corrector on the REAL 500k estimate (expect failure: k_s ~ 34 > scan depth)
echo "== corrector on real 500k estimate =="
./corrector pk.bin raw_estimates.bin rec_corr_500k.bin || echo "(failed as predicted at 500k)"

# 3) mechanical validation: simulate a 1M-quality estimate from the 500k truth
python3 - <<'PY'
import numpy as np
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.float64)
alpha, sig = 0.367, 0.0614            # 1M-quality: sigma_eps = 154/sqrt(210k*30)
raw = np.concatenate([np.array([alpha, sig], dtype=np.float32),
                      (alpha*truth[:512]   + np.random.default_rng(1).normal(0, sig, 512)).astype(np.float32),
                      (alpha*truth[512:1024]+ np.random.default_rng(2).normal(0, sig, 512)).astype(np.float32),
                      (alpha*truth[1024:]  + np.random.default_rng(3).normal(0, sig, 512)).astype(np.float32)])
raw.tofile('raw_sim1m.bin')
err = np.rint(raw[2:2+1024]/alpha) - truth[:1024]
print('simulated 1M-quality estimate: wrong coeffs in (s0,s1) =', int(np.sum(err[:512]!=0)), int(np.sum(err[512:]!=0)))
PY
echo "== corrector on simulated 1M estimate =="
./corrector pk.bin raw_sim1m.bin rec_sim1m.bin
cmp rec_sim1m.bin truth.bin && echo "SIM 1M: corrector output == TRUTH (byte-identical)"
