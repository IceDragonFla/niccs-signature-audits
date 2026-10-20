#!/usr/bin/env bash
set -e
cd ~/darts_work/attack
python3 - <<'PY'
import numpy as np
truth = np.fromfile('truth.bin', dtype=np.int8).astype(np.float64)
alpha = 0.367
# test A: zero-noise estimate -> initial count_bad MUST be 0
raw = np.concatenate([np.array([alpha, 0.0], dtype=np.float32),
                      (alpha*truth[:512]).astype(np.float32),
                      (alpha*truth[512:1024]).astype(np.float32),
                      (alpha*truth[1024:]).astype(np.float32)])
raw.tofile('raw_exact.bin')
# test B: zero-noise s0/s1 but 3 planted e-side... no - plant 2 KNOWN s-flips near expected-ambiguous positions
s0 = truth[:512].copy(); s1 = truth[512:1024].copy()
# flip two coords of s0: 0->1 and 1->-1 (deltas +1 and -2)
i0 = int(np.flatnonzero(s0==0)[0]); s0[i0] = 1
i1 = int(np.flatnonzero(s0==1)[0]); s1f = s1.copy(); s1[i1] = -1
raw2 = np.concatenate([np.array([alpha, 0.0], dtype=np.float32),
                       (alpha*s0).astype(np.float32),
                       (alpha*s1).astype(np.float32),
                       (alpha*truth[1024:]).astype(np.float32)])
raw2.tofile('raw_flip2.bin')
np.save('/tmp/flipinfo.npy', np.array([i0, i1]))
print('planted flips at s0[%d]=+1 (delta +1), s0[%d]=-1 (delta -2)' % (i0, i1))
PY
echo "== test A: exact estimate (expect: already consistent) =="
./corrector pk.bin raw_exact.bin rec_A.bin || true
echo "== test B: exact + 2 planted flips (expect: pair-scan collapse) =="
./corrector pk.bin raw_flip2.bin rec_B.bin || true
cmp rec_B.bin truth.bin && echo "TEST B PASS: output == truth" || echo "TEST B FAIL"
