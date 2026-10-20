#!/usr/bin/env bash
cd ~/darts_work/attack
for NN in 750000 800000 850000 900000 950000 1000000 1050000 1100000; do
  python3 attack.py pub15.bin labels15.bin truth.bin $NN > /dev/null 2>&1
  rm -f rec_p.bin
  OUT=$(./scan3 pk.bin cols.bin raw_estimates.bin rec_p.bin 3 2>&1)
  R=$(echo "$OUT" | grep -oE 'collapse: depth [0-9]|already consistent' | head -1)
  if cmp -n 1024 -s rec_p.bin truth.bin; then C1=EXACT; else C1=NO; fi
  W=$(python3 -c "
import numpy as np
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
t = np.fromfile('truth.bin', dtype=np.int8).astype(np.float64)
off = 3*512+5
w = int(np.sum(np.rint(np.concatenate([raw[off:off+512]/raw[1536], raw[off+512:off+1024]/raw[1537]])) != t[:1024]))
print(w)")
  echo "N=$NN wrong=$W result='${R:-FAILED}' s=$C1"
done
