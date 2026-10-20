#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
SRC=~/darts_work/DARTS128
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
gcc -O3 -march=native -std=gnu99 -fcommon -I$SRC -o gen_dump gen_dump.c sign_dump.c $SRC/sign.c $SRCS -lm
DARTS_KEY_SEED=a1b2c3d4e5f6a7b8a1b2c3d4e5f6a7b8a1b2c3d4e5f6a7b8a1b2c3d4e5f6a7b8 ./gen_dump keygen pk_seed.bin sk_seed.bin truth_seed.bin
./gen_dump keygen pk_def.bin sk_def.bin truth_def.bin
cmp -s pk_seed.bin pk_def.bin && echo "SEED FAILED: same key" || echo "SEED OK: keys differ"
cmp -s sk_seed.bin sk.bin && echo "note: seeded key == main-run key" || echo "seeded key differs from main-run key (expected)"
rm -f pk_seed.bin sk_seed.bin truth_seed.bin pk_def.bin sk_def.bin truth_def.bin
