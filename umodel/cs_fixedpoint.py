"""cs_fixedpoint.py (v6, round 1) — bit-exact port of the CS-128 reference
rejection-sampling arithmetic, INCLUDING its undefined-behavior shift
semantics (x86: 64-bit shift counts are taken mod 64).

Closes the v5 open boundary (i): the exact-LLR information bound was
computed under the INTENDED acceptance model; the shipped implementation
takes E >> (bC-bE) with exponent gaps up to ~500, which on x86 shifts by
(bC-bE) & 63 — a different acceptance probability. This module reproduces
the ACTUAL distribution bit-exactly, validated against instrumented
EXPLINE captures from the reference binary.

Semantics ported from sampling.c:
  Approx(B, A0, A1): B = top-64 of the 128-bit product; returns excess bits.
  per-factor accumulation: bX += Approx(...) + coe.
  C+E merge: conditional XOR swap, then E >> (bC-bE)  [UB: & 63],
             C = C + E, renormalize on carry.
  exp-test: bB = Approx(premul, C) + bC + premul_coe,
            bA = Approx(premul', F) + bF + premul'_coe + Approx(A, M0),
            bA -= bB; abort if bA > 0 or (== 0 and A > B);
            A >>= -bA  [UB: & 63];  accept iff rand64 < A given < B.
"""
import json
import math
import os

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_T = json.load(open(os.path.join(_DIR, "_cs_fre_tables.json")))
FRE0 = _T["fre_0"]
FRE1 = _T["fre_1"]
FRE2 = _T["fre_2"]
COE0 = _T["fre_0_coe"]
COE1 = _T["fre_1_coe"]
COE2 = _T["fre_2_coe"]
PREMUL = _T["fre_premul"]
PREMUL_COE = _T["fre_premul_coe"]
PREMUL_P = _T["fre_premul_prime"]
PREMUL_P_COE = _T["fre_premul_prime_coe"]
M0 = _T["M0"]

M64 = (1 << 64) - 1
UB_SHIFT = True          # emulate x86 shift-count masking


def _shr(x, n):
    """x >> n with x86 UB semantics (count mod 64) when n >= 64."""
    if n < 0:
        n = 0
    if n >= 64 and UB_SHIFT:
        n &= 63
    return x >> n


def _approx(a0, a1):
    """Returns (mant, excess)."""
    p = a0 * a1
    bits = p.bit_length()
    if bits > 64:
        e = bits - 64
        return p >> e, e
    return p, 0


def chunk_exp_test(y0c, Y1, Y2, t0, t1, t2, chunk_pos, i):
    """One chunk's exp-test under the ACTUAL implementation semantics.

    y0c: (sz,) y0 state at chunk positions (post); Y1: (l,d), Y2: (k,d)
    post states; t0/t1/t2: chunk responses. Returns
    (A_mant, B_mant, bA_rel, accept_prob) where accept_prob = A/B
    (mantissa ratio at B's scale, as the Bernoulli uses it)."""
    # ---- accumulate C, E, F over the three component groups
    C = E = F = 1
    bC = bE = bF = 0

    def acc(val, table, coe_tab):
        nonlocal C, E, F, bC, bE, bF
        for which in (0, 1, 2):        # 0 = C (minus), 1 = E (plus), 2 = F
            pass

    # y0 part (chunk positions only)
    for u in range(len(chunk_pos)):
        base = int(round(y0c[u]))
        tv = int(t0[u])
        for which, arg in ((0, base - tv), (1, base + tv), (2, base)):
            a = abs(int(arg))
            if which == 0:
                C, e = _approx(C, FRE0[a])
                bC += e + COE0[a]
            elif which == 1:
                E, e = _approx(E, FRE0[a])
                bE += e + COE0[a]
            else:
                F, e = _approx(F, FRE0[a])
                bF += e + COE0[a]
    # y1 / y2 parts (all coords)
    for Y, t, tab, coe in ((Y1, t1, FRE1, COE1), (Y2, t2, FRE2, COE2)):
        for j in range(Y.shape[0]):
            for u in range(Y.shape[1]):
                base = int(round(Y[j, u]))
                tv = int(t[j, u])
                for which, arg in ((0, base - tv), (1, base + tv),
                                   (2, base)):
                    a = abs(int(arg))
                    if which == 0:
                        C, e = _approx(C, tab[a])
                        bC += e + coe[a]
                    elif which == 1:
                        E, e = _approx(E, tab[a])
                        bE += e + coe[a]
                    else:
                        F, e = _approx(F, tab[a])
                        bF += e + coe[a]
    # ---- C+E merge (with UB shift)
    diff = bC - bE
    if diff < 0:                      # conditional XOR swap
        u_ = (bC ^ bE)
        Bx = (C ^ E) & M64
        bC, bE = bC ^ u_, bE ^ u_
    else:
        u_ = 0
        Bx = 0
    Ew = ((E ^ Bx) & M64)
    Ew = _shr(Ew, bC - bE)
    Csum = ((C ^ Bx) + Ew) & M64
    if Csum < Ew:
        bC += 1
        Csum = (1 << 63) | (Csum >> 1)
    # ---- exp-test
    Bm, eB = _approx(PREMUL[i], Csum)
    bB = eB + bC + PREMUL_COE[i]
    Am, eA = _approx(PREMUL_P[i], F)
    bA = eA + bF + PREMUL_P_COE[i]
    Am, eA2 = _approx(Am, M0[i])
    bA += eA2
    bA -= bB
    # hard abort if bA > 0, or bA == 0 and A > B
    abort = (bA > 0) or (bA == 0 and Am > Bm)
    A_shift = _shr(Am, -bA) if bA < 0 else Am
    p = (A_shift / Bm) if Bm else 1.0
    return {"A": Am, "B": Bm, "bA": bA, "bB": bB, "C": Csum, "E": Ew,
            "F": F, "bC": bC, "bE": bE, "bF": bF, "abort": abort,
            "p": min(1.0, p) if not abort else 0.0,
            "log2_p": (math.log2(min(1.0, p)) if not abort else 0.0)}
