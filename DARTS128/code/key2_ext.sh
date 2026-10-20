#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
NS1=1000000; NEXT=250000
for k in 0 1 2 3 4 5 6 7; do
  s=$((NS1 + k*31250))
  ./gen_dump sign sk2.bin $s 31250 pub3_$k.bin sig3_$k.bin > /dev/null 2>&1 &
done
wait
for k in 0 1 2 3 4 5 6 7; do
  s=$((NS1 + k*31250))
  ./label_extract pk2.bin sig3_$k.bin $s 31250 labels3_$k.bin > /dev/null 2>&1 &
done
wait
cat pub3_*.bin > pub3.bin; cat labels3_*.bin > labels3.bin
cat pub2.bin pub3.bin > pub25.bin
cat labels2.bin labels3.bin > labels25.bin
echo "extended to 1250000"
python3 attack.py pub25.bin labels25.bin truth2.bin 1250000 2>&1 | grep -E 'R1 s0 .descaled' | head -1
./dump_cols pk2.bin cols2.bin > /dev/null
./scan3 pk2.bin cols2.bin raw_estimates.bin rec25.bin 3
cp rec25.bin rec25_fix.bin
./forge fixup pk2.bin rec25_fix.bin
./forge pk2.bin rec25_fix.bin 20
cmp -n 1024 rec25_fix.bin truth2.bin && echo "KEY2 @1.25M: S EXACT" || echo "KEY2 @1.25M: s wrong"
cmp rec25_fix.bin truth2.bin && echo "KEY2 @1.25M: FULL KEY == TRUTH"
