#!/usr/bin/env bash
# adaptive_stop.sh — adaptive-stopping validation run for one key.
# usage: adaptive_stop.sh <workdir-tag> <pk> <sk> [seed]
# Batches of 2.5e5: cumulative 5e5, 7.5e5, 1e6, 1.25e6, 1.5e6.
# After each batch: LS -> scan3 (tolerant collapse) -> fixup;
# closure = fixup non-ternary == 0  <=>  s exact. Stops at the first closure.
set -e
TAG=${1:?tag}
PK=${2:?pk}
SK=${3:?sk}
SEED=${4:-}
cd "$(dirname "$0")"
mkdir -p work_$TAG && cd work_$TAG
SRC=${SRC:-$HOME/darts_work/DARTS128}
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
CFL="-O3 -march=native -std=gnu99 -fcommon -I$SRC"
gcc $CFL -o gen_dump_$TAG gen_dump.c sign_dump.c $SRC/sign.c $SRCS -lm
gcc $CFL -o label_extract_$TAG label_extract.c $SRCS -lm
gcc $CFL -o scan3_$TAG scan3.c $SRCS -lm
gcc $CFL -o forge_$TAG forge.c $SRC/sign.c $SRCS -lm
gcc $CFL -o dump_cols_$TAG dump_cols.c $SRCS -lm

if [ ! -f pk.bin ]; then
  if [ -n "$SEED" ]; then DARTS_KEY_SEED=$SEED ./gen_dump_$TAG keygen pk.bin sk.bin truth.bin
  else ./gen_dump_$TAG keygen pk.bin sk.bin truth.bin; fi
fi
./dump_cols_$TAG $PK cols.bin > /dev/null

B=0
ACC_P=pub_acc_$TAG.bin; ACC_L=lab_acc_$TAG.bin
: > $ACC_P; : > $ACC_L
check() {  # total sigs
  local total=$1
  python3 ../attack.py $ACC_P $ACC_L none $total > /dev/null
  ./scan3_$TAG $PK cols.bin raw_estimates.bin rec.bin 3 > /dev/null 2>&1 || true
  cp rec.bin rec_fix.bin
  local NT=$(./forge_$TAG fixup $PK rec_fix.bin | grep -oE 'non-ternary=[0-9]+' | cut -d= -f2)
  echo "  batch-total $total: fixup non-ternary=$NT"
  [ "$NT" = "0" ]
}

BATCH=250000
for B in 1 2 3 4 5; do
  total=$((B * BATCH))
  start=$(((B-1) * BATCH))
  echo "== $TAG: batch $B (total $total) =="
  for k in 0 1 2 3 4 5 6 7; do
    local_s=$((start + k * BATCH / 8))
    local_n=$((BATCH / 8))
    ./gen_dump_$TAG sign sk_$TAG.bin $local_s $local_n pub_${TAG}_b${B}_$k.bin sig_${TAG}_b${B}_$k.bin > /dev/null 2>&1 &
  done
  wait
  for k in 0 1 2 3 4 5 6 7; do
    local_s=$((start + k * BATCH / 8))
    ./label_extract_$TAG $PK sig_${TAG}_b${B}_$k.bin $local_s $local_n lab_${TAG}_b${B}_$k.bin > /dev/null 2>&1 &
  done
  wait
  for k in 0 1 2 3 4 5 6 7; do
    cat pub_${TAG}_b${B}_$k.bin >> $ACC_P
    cat lab_${TAG}_b${B}_$k.bin >> $ACC_L
  done
  if check $total; then
    echo "== $TAG CLOSED at $total =="
    exit 0
  fi
done
echo "== $TAG: NOT CLOSED at 1.5e6 =="
exit 2
