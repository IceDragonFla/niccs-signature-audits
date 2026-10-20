#!/usr/bin/env bash
cd ~/darts_work/attack
for NN in 50000 100000 200000 500000 1000000 1500000 1750000; do
  python3 attack.py pub15.bin labels15.bin truth.bin $NN > /dev/null 2>&1
  W=$(python3 -c "
import numpy as np
raw = np.fromfile('raw_estimates.bin', dtype=np.float64)
t = np.fromfile('truth.bin', dtype=np.int8).astype(np.float64)
off = 3*512+5
a0, a1 = raw[1536], raw[1537]
d = np.concatenate([raw[off:off+512]/a0, raw[off+512:off+1024]/a1]) - t[:1024]
print('%d %.6f' % (int(np.sum(np.rint(d) != t[:1024])), np.sqrt((d**2).mean())))")
  echo "N=$NN wrong_rms=$W"
done
