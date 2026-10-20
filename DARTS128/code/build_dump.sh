#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
SRC=${SRC:-~/darts_work/DARTS128}
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
gcc -O3 -march=native -std=gnu99 -fcommon -I$SRC -o dump_cols dump_cols.c $SRCS -lm
echo "build ok: dump_cols"
