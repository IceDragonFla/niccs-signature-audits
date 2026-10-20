#!/usr/bin/env bash
# run_parallel.sh — 8-worker parallel attack pipeline for N signatures.
# Workers sign/label disjoint index ranges (signing is deterministic in (sk,msg)),
# artifacts are concatenated in range order, then attack.py runs once.
set -e
cd "$(dirname "$0")"
SRC=~/darts_work/DARTS128
SRCS="$SRC/packing.c $SRC/polyvec.c $SRC/poly.c $SRC/ntt.c $SRC/reduce.c \
      $SRC/sampler.c $SRC/encoding.c $SRC/polyfix.c $SRC/polymat.c $SRC/fft.c \
      $SRC/fixpoint.c $SRC/symmetric.c $SRC/drng.c $SRC/auxfunc.c"
CFL="-O3 -march=native -std=gnu99 -fcommon -I$SRC"

NS=${NS:-2000000}
W=${W:-8}
PER=$((NS / W))

echo "== build =="
gcc $CFL -o gen_dump gen_dump.c sign_dump.c $SRC/sign.c $SRCS -lm
gcc $CFL -o label_extract label_extract.c $SRCS -lm
gcc $CFL -o forge forge.c $SRC/sign.c $SRCS -lm

echo "== keygen =="
# DARTS_KEY_SEED (hex, >=1 byte): instantiate the reference DRNG to obtain a
# key DIFFERENT from the deterministic zero-state keypair. Unset => default.
[ -f pk.bin ] || DARTS_KEY_SEED="$DARTS_KEY_SEED" ./gen_dump keygen pk.bin sk.bin truth.bin

echo "== generate: $NS signatures, $W workers =="
pids=()
for k in $(seq 0 $((W - 1))); do
  s=$((k * PER))
  ./gen_dump sign sk.bin $s $PER pub_$k.bin sig_$k.bin > gen_$k.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "generation done"

echo "== labels: $W workers =="
pids=()
for k in $(seq 0 $((W - 1))); do
  s=$((k * PER))
  ./label_extract pk.bin sig_$k.bin $s $PER labels_$k.bin > lab_$k.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
cat lab_*.log | grep "branch stats" || true

echo "== concatenate =="
cat $(for k in $(seq 0 $((W - 1))); do echo pub_$k.bin; done) > pub.bin
cat $(for k in $(seq 0 $((W - 1))); do echo sig_$k.bin; done) > sig.bin
cat $(for k in $(seq 0 $((W - 1))); do echo labels_$k.bin; done) > labels.bin
ls -la pub.bin labels.bin

echo "== attack =="
python3 attack.py pub.bin labels.bin truth.bin $NS 2>&1 | tee attack_log.txt

echo "== staged flip corrector (BiT-style solution-consistency) =="
./corrector pk.bin raw_estimates.bin recovered.bin
cp recovered.bin recovered_hint.bin
./forge fixup pk.bin recovered_hint.bin
./forge pk.bin recovered_hint.bin 20
echo "== all done =="
