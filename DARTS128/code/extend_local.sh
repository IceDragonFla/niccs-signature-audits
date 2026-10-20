#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
S="$HOME/darts_work/DARTS128"
for k in 0 1 2 3 4 5 6 7; do
  s=$((500000 + k*156250))
  ./gen_dump sign sk.bin $s 156250 pub_e$k.bin sig_e$k.bin > /dev/null 2>&1 &
done
wait
echo "gen done"
for k in 0 1 2 3 4 5 6 7; do
  s=$((500000 + k*156250))
  ./label_extract pk.bin sig_e$k.bin $s 156250 labels_e$k.bin > /dev/null 2>&1 &
done
wait
echo "labels done"
cp pub.bin pub5.bin; cp labels.bin labels5.bin
cat pub_e0.bin pub_e1.bin pub_e2.bin pub_e3.bin pub_e4.bin pub_e5.bin pub_e6.bin pub_e7.bin > pub_e.bin
cat labels_e0.bin labels_e1.bin labels_e2.bin labels_e3.bin labels_e4.bin labels_e5.bin labels_e6.bin labels_e7.bin > labels_e.bin
cat pub5.bin pub_e.bin > pub15.bin
cat labels5.bin labels_e.bin > labels15.bin
ls -la pub15.bin labels15.bin
