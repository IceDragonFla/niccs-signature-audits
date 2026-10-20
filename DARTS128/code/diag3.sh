#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
SRC=${SRC:-~/darts_work/DARTS128}
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
gcc -O3 -march=native -std=gnu99 -fcommon -I$SRC -o corrector corrector.c $SRCS -lm
DARTS_TRUTH=truth.bin ./corrector pk.bin raw_sim1m.bin rec_sim1m.bin || true
