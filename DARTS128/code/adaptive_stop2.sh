#!/usr/bin/env bash
# adaptive_stop2.sh — keyB (seeded) adaptive validation, self-contained.
set -e
TAG=keyB
PK=pk.bin
SK=sk.bin
SEED=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
cd ~/darts_work/repro/work_keyB
SRC=$HOME/darts_work/DARTS128
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
CFL="-O3 -march=native -std=gnu99 -fcommon -I$SRC"
gcc $CFL -o gen_dump gen_dump.c sign_dump.c $SRC/sign.c $SRCS -lm
gcc $CFL -o label_extract label_extract.c $SRCS -lm
gcc $CFL -o scan3 scan3.c $SRCS -lm
gcc $CFL -o forge forge.c $SRC/sign.c $SRCS -lm
gcc $CFL -o dump_cols dump_cols.c $SRCS -lm

if [ ! -f pk.bin ]; then
  DARTS_KEY_SEED=$SEED ./gen_dump keygen pk.bin sk.bin truth.bin
fi
./dump_cols pk.bin cols.bin > /dev/null

ACC_P=pub_acc.bin; ACC_L=lab_acc.bin
: > $ACC_P; : > $ACC_L
check() {
  local total=$1
  python3 attack.py $ACC_P $ACC_L none $total > /dev/null
  ./scan3 pk.bin cols.bin raw_estimates.bin rec.bin 3 > /dev/null 2>&1 || true
  cp rec.bin rec_fix.bin
  local NT=$(./forge fixup pk.bin rec_fix.bin | grep -oE 'non-ternary=[0-9]+' | cut -d= -f2)
  echo "  batch-total $total: fixup non-ternary=$NT"
  [ "$NT" = "0" ]
}
BATCH=250000
for B in 1 2 3 4 5; do
  total=$((B * BATCH)); start=$(((B-1) * BATCH))
  echo "== keyB: batch $B (total $total) =="
  for k in 0 1 2 3 4 5 6 7; do
    local_s=$((start + k * BATCH / 8))
    ./gen_dump sign sk.bin $local_s $((BATCH / 8)) pub_b${B}_$k.bin sig_b${B}_$k.bin > /dev/null 2>&1 &
  done
  wait
  for k in 0 1 2 3 4 5 6 7; do
    local_s=$((start + k * BATCH / 8))
    ./label_extract pk.bin sig_b${B}_$k.bin $local_s $((BATCH / 8)) lab_b${B}_$k.bin > /dev/null 2>&1 &
  done
  wait
  for k in 0 1 2 3 4 5 6 7; do
    cat pub_b${B}_$k.bin >> $ACC_P
    cat lab_b${B}_$k.bin >> $ACC_L
  done
  if check $total; then echo "== keyB CLOSED at $total =="; exit 0; fi
done
echo "== keyB: NOT CLOSED at 1.5e6 =="
exit 2
