#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
SRC=~/darts_work/DARTS128
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
gcc -O3 -march=native -std=gnu99 -fcommon -I$SRC -o forge forge.c $SRC/sign.c $SRCS -lm
./forge pk.bin recovered_hint.bin 20
