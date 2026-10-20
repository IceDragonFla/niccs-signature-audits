"""calibrate_fixedpoint.py — validate the bit-exact fixed-point port against
EXPLINE captures from the reference binary (accepted chunks only)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from umodel import niccs
from umodel.cs_fixedpoint import chunk_exp_test
from umodel.cs_exact_llr import negconv_chunk, CUM

DBG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cs128", "cs_dbg")
lines = open(os.path.join(DBG, "expline.log")).read().splitlines()
# group: last 6 EXPLINE before each EXPDONE = accepted attempt
groups, cur = [], []
for ln in lines:
    if ln.startswith("EXPLINE"):
        cur.append([int(x) for x in ln.split()[1:]])
    elif ln.startswith("EXPDONE"):
        groups.append(cur[-6:])
        cur = []

data = niccs.cs_ingest_full(os.path.join(DBG, "dump.bin"),
                            n_rows=len(groups))
ground = niccs.cs_ground(os.path.join(DBG, "ground.bin"))
d = 256
s = ground["s"].astype(np.int64)
e = ground["e"].astype(np.int64)

n_ok = n_tot = 0
fields_ok = {k: 0 for k in ("C", "E", "F", "bC", "bE", "bF", "A", "B", "bA", "bB")}
for k, grp in enumerate(groups[:40]):
    cpub = data["c_pub"][k:k+1]
    eps = data["eps_chunks"][k].astype(np.int64)
    y0 = data["y0"][k].astype(np.int64)
    y1 = data["y1"][k].astype(np.int64)
    y2 = data["y2"][k].astype(np.int64)
    pos = np.nonzero(cpub)[1]
    cdense = np.zeros((6, d), np.int64)
    chunk_id = np.searchsorted(CUM[1:], np.arange(23), side="right")
    for j in range(23):
        cdense[chunk_id[j], pos[j]] = cpub[0, pos[j]]
    Y0 = y0.copy()
    Y1 = y1.copy()
    Y2 = y2.copy()
    for i in range(6):
        cp = pos[CUM[i]:CUM[i+1]]
        e_i = eps[i]
        rs = negconv_chunk(cdense[i:i+1], s)[0]
        re_ = negconv_chunk(cdense[i:i+1], e)[0]
        Y0 = Y0 + e_i * cdense[i]
        Y1 = Y1 + e_i * rs
        Y2 = Y2 + e_i * re_
        t0 = (e_i * cdense[i])[cp]
        t1 = e_i * rs
        t2 = e_i * re_
        r = chunk_exp_test(Y0[cp], Y1, Y2, t0, t1, t2, cp, i)
        ref = grp[i]   # bA bB A B C E F bC bE bF
        n_tot += 1
        checks = {"bA": (r["bA"], ref[0]), "bB": (r["bB"], ref[1]),
                  "A": (r["A"], ref[2]), "B": (r["B"], ref[3]),
                  "C": (r["C"], ref[4]), "E": (r["E"], ref[5]),
                  "F": (r["F"], ref[6]), "bC": (r["bC"], ref[7]),
                  "bE": (r["bE"], ref[8]), "bF": (r["bF"], ref[9])}
        allok = True
        for name, (mine, refv) in checks.items():
            ok = int(mine) == int(refv)
            fields_ok[name] += ok
            allok &= ok
        n_ok += allok
        if not allok and n_tot <= 12:
            bad = {n_: (m, rv) for n_, (m, rv) in checks.items()
                   if int(m) != int(rv)}
            print(f"sig {k} chunk {i} MISMATCH: {bad}")
print(f"\naccepted chunks fully matching: {n_ok}/{n_tot}")
for name, cnt in fields_ok.items():
    print(f"  field {name}: {cnt}/{n_tot}")
