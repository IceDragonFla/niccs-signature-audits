#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
NS2=1250000
for k in 0 1 2 3 4 5 6 7; do
  s=$((NS2 + k*31250))
  ./gen_dump sign sk2.bin $s 31250 pub4_$k.bin sig4_$k.bin > /dev/null 2>&1 &
done
wait
for k in 0 1 2 3 4 5 6 7; do
  s=$((NS2 + k*31250))
  ./label_extract pk2.bin sig4_$k.bin $s 31250 labels4_$k.bin > /dev/null 2>&1 &
done
wait
cat pub4_*.bin > pub4.bin; cat labels4_*.bin > labels4.bin
cat pub25.bin pub4.bin > pub_f.bin
cat labels25.bin labels4.bin > labels_f.bin
echo "extended to 1500000"
python3 attack.py pub_f.bin labels_f.bin truth2.bin 1500000 2>&1 | grep -E 'R1 s0 .descaled' | head -1
./scan3 pk2.bin cols2.bin raw_estimates.bin rec_f.bin 3
cp rec_f.bin rec_f_fix.bin
./forge fixup pk2.bin rec_f_fix.bin
./forge pk2.bin rec_f_fix.bin 20
cmp -n 1024 rec_f_fix.bin truth2.bin && echo "KEY2 @1.5M: S EXACT" || echo "KEY2 @1.5M: s wrong"
cmp rec_f_fix.bin truth2.bin && echo "KEY2 @1.5M: FULL KEY == TRUTH"
