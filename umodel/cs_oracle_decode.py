"""cs_oracle_decode.py (v14) — the oracle-arm decode closure for CS-128.

Validates that the "perfect labels => N ~ 4.8e5 (decode semantics)" bound
is real: with the oracle chunk signs on the fresh 500k dump, run the
chunked WLS -> sigma ~0.195 -> expected ~8 rounding errors -> MITM window
decode -> razor + validation against ground truth. A complete key
recovery conditioned on labels — the empirical anchor of the cold-start
wall (public labels are worth ~0, so the wall stands).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel import niccs
from umodel.channels import ShiftRegressionChannel
from umodel.spec import SchemeSpec
from umodel.linalg import build_relation_matrix, invert_modq, center_modq
from umodel.decoder import DecoderLadder
from umodel.cs_exact_llr import CUM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    ".."))
DUMP = os.path.join(ROOT, "tmp", "cs128", "cs_500k", "dump.bin")
GROUND = os.path.join(ROOT, "tmp", "cs128", "cs_500k", "ground.bin")


def main(block=50_000):
    spec = SchemeSpec(
        name="CS-128-decode", d=256, q=32257, tau=23, challenge_signed=True,
        ring=niccs.RingStructure("negacyclic"), K=3, L=3,
        shared_latent="chunked-signs", chunk_sizes=(4, 4, 4, 4, 4, 3),
        channels={"z1-regression": niccs.ChannelSpec(
            "z1-regression", "shift_regression", "z1, c, labels", "s",
            niccs.MaskModel("empirical", 2779))},
        relation=niccs.PublicKeyRelation(
            form="A0*s + e = 32*t1 + t0; razor: center(32*t1 - A0*s) "
                 "= e - t0 in r_box",
            s_alphabet=(-1, 0, 1), r_box=(-17, 17), pk_scale=32,
            b0_box=(-16, 15), derive="clip"))
    reg = ShiftRegressionChannel(spec)
    B = None
    N = 0
    raw = np.memmap(DUMP, dtype=np.uint8, mode="r")
    ROW = 9276
    total = len(raw) // ROW
    for st in range(0, total, block):
        en = min(block, total - st)
        data = niccs.cs_ingest_full(DUMP, n_rows=total)  # memmap view
        cpub = data["c_pub"][st:st + en]
        z1 = data["z1"][st:st + en].astype(np.float64)
        eps = data["eps_chunks"][st:st + en].astype(np.float64)
        cs_lam = niccs.cs_chunk_spectra(cpub, (4, 4, 4, 4, 4, 3))
        b_, d_, n_ = reg.accumulate_chunked(cs_lam, z1, eps)
        B = b_ if B is None else B + b_
        Dn = d_ if N == 0 else Dn + d_
        N += n_
        print(f"  accumulated {N}/{total}", file=sys.stderr)
    shat = reg.solve(B, Dn if N else Dn)[0]
    ground = niccs.cs_ground(GROUND, a_scale=23435)
    s_true = ground["s"]
    sig = float((shat - s_true).std())
    nerr = int((np.clip(np.rint(shat), -1, 1) != s_true).sum())
    print(f"oracle-label regression on {N:,}: sigma = {sig:.4f}, "
          f"rounding errors = {nerr}/768")
    # decode: razor on the CS relation
    A0 = ground["A0"].astype(np.int64)
    Mx = build_relation_matrix(spec, A0)
    Ainv = invert_modq(Mx, spec.q)
    b1f = (32 * (ground["t1"] % spec.q)).ravel()
    ladder = DecoderLadder(spec, Ainv, b1f, A0)
    rec_w = min(46, 4 * nerr + 16)
    s_hat, lik = ladder.mitm_window(shat.reshape(-1), Mx, W=rec_w)
    how = f"MITM(W={rec_w})"
    if s_hat is None:
        # greedy razor-guided corrector (staged single moves; the CS
        # reference's own corrector strategy): from the rounded start,
        # apply alphabet-neighbor moves that reduce the R-violation count
        s_c = np.clip(np.rint(shat.reshape(-1)), -1, 1).astype(np.int64)
        R = center_modq(Mx @ s_c - b1f, spec.q)
        def nbad(sc, Rc):
            return int(((Rc < -17) | (Rc > 17)).sum()
                       + ((sc < -1) | (sc > 1)).sum())
        nb = nbad(s_c, R)
        for _ in range(60):
            if nb == 0:
                break
            best = None
            Rc = R.copy()
            for m in range(768):
                for dv in (-1, 1):
                    v2 = s_c[m] + dv
                    if v2 < -1 or v2 > 1:
                        continue
                    R2 = Rc + dv * center_modq(Mx[:, m], spec.q)
                    R2 = center_modq(R2, spec.q)
                    nb2 = int(((R2 < -17) | (R2 > 17)).sum())
                    if best is None or nb2 < best[0]:
                        best = (nb2, m, dv)
            if best is None or best[0] >= nb:
                break
            nb, m, dv = best
            s_c[m] += dv
            R = center_modq(R + dv * center_modq(Mx[:, m], spec.q),
                            spec.q)
        s_hat = s_c
        how = f"greedy corrector (final nbad={nb})"
    exact = bool(np.array_equal(s_hat.reshape(3, 256), s_true))
    R = center_modq(Mx @ s_hat - b1f, spec.q)
    print(f"{how} -> s0 exact: {exact}; "
          f"R box ok: {bool(np.all((R >= -17) & (R <= 17)))}")
    return exact


if __name__ == "__main__":
    main()
