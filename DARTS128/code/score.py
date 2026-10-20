#!/usr/bin/env python3
"""score.py — standalone scoring for the DARTS128 (1-2b) attack artifacts.

Separated from the attack pipeline per review: the attack scripts
(attack.py / corrector / scan3 / label_extract / mp_decode) never read the
secret key; this script is the ONLY place where the recovered key is compared
against the ground truth, and it is run after the attack has finished.

usage:
  score.py <recovered.bin> <truth.bin>              # byte comparison + counts
  score.py <raw_estimates.bin> <truth.bin> --raw    # continuous estimate rms
"""
import sys
import numpy as np

N = 512

def main():
    recp, truthp = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "final"
    truth = np.fromfile(truthp, dtype=np.int8).astype(np.int64)
    if mode == "--raw":
        raw = np.fromfile(recp, dtype=np.float64)
        off = 3 * N + 5
        a0, a1 = raw[3*N], raw[3*N+1]
        est = np.concatenate([raw[off:off+N] / a0, raw[off+N:off+2*N] / a1])
        err = est - truth[:2*N].astype(np.float64)
        print(f"descaled estimate rms = {np.sqrt((err**2).mean()):.4f}  "
              f"wrong(after rounding) = {int(np.sum(np.rint(est) != truth[:2*N]))}/1024")
        return
    rec = np.fromfile(recp, dtype=np.int8)
    if len(rec) != 3 * N:
        print(f"unexpected size {len(rec)} (want {3*N})"); sys.exit(1)
    s_wrong = int(np.sum(rec[:2*N] != truth[:2*N]))
    e_wrong = int(np.sum(rec[2*N:3*N] != truth[2*N:3*N]))
    print(f"s wrong = {s_wrong}/1024   e wrong = {e_wrong}/512")
    if s_wrong == 0 and e_wrong == 0:
        print("RECOVERED KEY == GROUND TRUTH (byte-identical)")
    else:
        print("recovered key differs from ground truth")

if __name__ == "__main__":
    main()
