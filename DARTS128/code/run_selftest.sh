#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
SRC=~/darts_work/DARTS128
SRCS="$SRC/sign.c $SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
gcc -O2 -std=gnu99 -fcommon -I$SRC -o selftest_cols selftest_cols.c $SRCS -lm
./selftest_cols pk.bin truth.bin
