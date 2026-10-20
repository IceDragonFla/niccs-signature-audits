#!/usr/bin/env bash
# build_tools.sh — build all attack tools (local or remote; run from the tool dir)
set -e
cd "$(dirname "$0")"
SRC=${SRC:-~/darts_work/DARTS128}
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
CFL="-O3 -march=native -std=gnu99 -fcommon -I$SRC"
gcc $CFL -o gen_dump gen_dump.c sign_dump.c $SRC/sign.c $SRCS -lm
gcc $CFL -o label_extract label_extract.c $SRCS -lm
gcc $CFL -o forge forge.c $SRC/sign.c $SRCS -lm
gcc $CFL -o corrector corrector.c $SRCS -lm
gcc $CFL -o scan3 scan3.c $SRCS -lm
gcc $CFL -o dump_cols dump_cols.c $SRCS -lm
echo "build ok: gen_dump label_extract forge corrector"
