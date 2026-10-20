#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
S="$HOME/darts_work/DARTS128"
NS=1000000
echo "== second key (seeded) =="
DARTS_KEY_SEED=deadbeefcafebabe0123456789abcdefdeadbeefcafebabe0123456789abcdef ./gen_dump keygen pk2.bin sk2.bin truth2.bin
for k in 0 1 2 3 4 5 6 7; do
  s=$((k * 125000))
  ./gen_dump sign sk2.bin $s 125000 pub2_$k.bin sig2_$k.bin > /dev/null 2>&1 &
done
wait
echo "gen done"
for k in 0 1 2 3 4 5 6 7; do
  s=$((k * 125000))
  ./label_extract pk2.bin sig2_$k.bin $s 125000 labels2_$k.bin > /dev/null 2>&1 &
done
wait
echo "labels done"
cat pub2_0.bin pub2_1.bin pub2_2.bin pub2_3.bin pub2_4.bin pub2_5.bin pub2_6.bin pub2_7.bin > pub2.bin
cat labels2_0.bin labels2_1.bin labels2_2.bin labels2_3.bin labels2_4.bin labels2_5.bin labels2_6.bin labels2_7.bin > labels2.bin
python3 attack.py pub2.bin labels2.bin truth2.bin $NS 2>&1 | grep -E 'R1 s0|branch tell' | head -2
./dump_cols pk2.bin cols2.bin > /dev/null
./scan3 pk2.bin cols2.bin raw_estimates.bin rec2.bin 3
cp rec2.bin rec2_fix.bin
./forge fixup pk2.bin rec2_fix.bin
./forge pk2.bin rec2_fix.bin 20
cmp -n 1024 rec2_fix.bin truth2.bin && echo "KEY2 @1M: S EXACT" || echo "KEY2 @1M: s wrong"
cmp rec2_fix.bin truth2.bin && echo "KEY2 @1M: FULL KEY == TRUTH"
