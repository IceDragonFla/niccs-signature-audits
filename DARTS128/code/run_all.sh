#!/usr/bin/env bash
# run_all.sh — DARTS128 (1-2b) attack pipeline on the WSL server.
# Stages: build -> smoke(2k) -> full(200k) -> labels -> recovery -> fixup -> forge
set -e
cd ~/darts_work/attack
SRC=~/darts_work/DARTS128
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
CFL="-O3 -march=native -std=gnu99 -fcommon -I$SRC"

echo "== build =="
gcc $CFL -o gen_dump gen_dump.c sign_dump.c $SRC/sign.c $SRCS -lm
gcc $CFL -o label_extract label_extract.c $SRCS -lm
gcc $CFL -o forge forge.c $SRC/sign.c $SRCS -lm
echo "build ok"

NS=${NS:-200000}
SMOKE=${SMOKE:-2000}

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "== smoke: $SMOKE signatures =="
  ./gen_dump keygen pk_smoke.bin sk_smoke.bin truth_smoke.bin
  ./gen_dump sign sk_smoke.bin 0 $SMOKE pub_smoke.bin sig_smoke.bin
  ./label_extract pk_smoke.bin sig_smoke.bin 0 $SMOKE labels_smoke.bin
  python3 attack.py pub_smoke.bin labels_smoke.bin truth_smoke.bin $SMOKE
  echo "== smoke end =="
fi

if [ "${FULL:-1}" = "1" ]; then
  echo "== full: $NS signatures =="
  [ -f pk.bin ] || ./gen_dump keygen pk.bin sk.bin truth.bin
  if [ ! -f pub.bin ] || [ "$(stat -c%s pub.bin)" -ne $((2625 * NS)) ]; then
    ./gen_dump sign sk.bin 0 $NS pub.bin sig.bin
  else
    echo "pub.bin already complete, skipping generation"
  fi
  ./label_extract pk.bin sig.bin 0 $NS labels.bin
  python3 attack.py pub.bin labels.bin truth.bin $NS 2>&1 | tee attack_log.txt
  cp recovered.bin recovered_hint.bin
  ./forge fixup pk.bin recovered_hint.bin
  ./forge pk.bin recovered_hint.bin 20
  echo "== full end =="
fi
