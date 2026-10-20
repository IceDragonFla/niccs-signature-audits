#!/usr/bin/env bash
cd ~/darts_work/attack
for NN in 800000 900000 950000 1050000 1100000; do
  python3 attack.py pub15.bin labels15.bin truth.bin $NN > /dev/null 2>&1
  ./dump_cols pk.bin cols.bin > /dev/null 2>&1
  R=$(./scan3 pk.bin cols.bin raw_estimates.bin rec_p.bin 3 2>&1 | grep -cE 'collapse|already')
  if cmp -n 1024 -s rec_p.bin truth.bin; then C1=EXACT; else C1=wrong; fi
  W=$(python3 -c "
import numpy as np
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
t = np.fromfile('truth.bin', dtype=np.int8).astype(np.float64)
off = 3*512+5
w = int(np.sum(np.rint(np.concatenate([raw[off:off+512]/raw[1536], raw[off+512:off+1024]/raw[1537]])) != t[:1024]))
print(w)")
  echo "N=$NN wrong=$W scan-collapse=$R s=$C1"
done
