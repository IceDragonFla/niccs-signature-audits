"""selftest.py — numeric verification of the v3 universal machinery.

Every check must pass before any demo is trusted:
  T1  one-convention algebra: generator convolution == relation matrix
  T2  regression exact-recovery identity, both ring kinds, signed/unsigned
  T3  label LUT == v2 engine on the BiT grid; signed-challenge sanity
  T4  moment operator == closed-form lemma (unsigned negacyclic);
      refusal for out-of-scope combos
  T5  simulated keygen satisfies the public relation exactly
  T6  applicability gate accepts both instances, refuses bogus specs
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel.spec import SchemeSpec, check_applicability, RingStructure
from umodel.linalg import (build_relation_matrix, kernel_matrix,
                           invert_modq, center_modq)
from umodel.channels import (LabelChannel, ShiftRegressionChannel,
                             QuantizedMomentChannel)
from umodel import simulate

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}  {detail}")


def ref_conv(spec, pos, vals, s):
    n, d = pos.shape
    L = s.shape[0]
    out = np.zeros((n, L, d), np.int64)
    for i in range(n):
        for t in range(spec.tau):
            p = int(pos[i, t])
            cv = int(vals[i, t])
            for k in range(d):
                delta = k - p
                fl = 1 if spec.ring.kind == "cyclic" else \
                    (-1 if delta < 0 else 1)
                out[i, :, k] += cv * fl * s[:, (k - p) % d]
    return out


def t1_algebra():
    print("T1: one-convention algebra (conv == Mx @ s)")
    for spec in (SchemeSpec.bit128(), SchemeSpec.dlt_toy()):
        d = spec.d
        rng = np.random.default_rng(1)
        pos = rng.random((4, d)).argpartition(spec.tau, axis=1)[:, :spec.tau]
        vals = np.ones((4, spec.tau), np.int64)
        if spec.challenge_signed:
            vals = rng.integers(0, 2, (4, spec.tau)) * 2 - 1
        s = rng.integers(-1, 2, (spec.L, d)).astype(np.int64)
        a = simulate._conv_response(spec, pos, vals, s)
        # dense challenge matrix form: (c*s) = Cdense @ s with Cdense built
        # from the same kernels convention as build_relation_matrix
        ok = True
        for i in range(4):
            c_can = np.zeros(d, np.int64)
            for t in range(spec.tau):
                c_can[pos[i, t]] = vals[i, t]
            Mxc = kernel_matrix(spec, c_can)
            lhs = np.stack([Mxc @ s[p] for p in range(spec.L)])
            ok &= np.array_equal(lhs, a[i])
        check(f"{spec.name}", ok, f"ring={spec.ring.kind}")


def t2_regression():
    print("T2: regression exact-recovery identity (eps known)")
    for spec in (SchemeSpec.bit128(), SchemeSpec.dlt_toy()):
        d = spec.d
        rng = np.random.default_rng(2)
        n = 3
        pos = rng.random((n, d)).argpartition(spec.tau, axis=1)[:, :spec.tau]
        vals = np.ones((n, spec.tau), np.int64)
        if spec.challenge_signed:
            vals = rng.integers(0, 2, (n, spec.tau)) * 2 - 1
        s = rng.integers(-1, 2, (spec.L, d)).astype(np.int64)
        eps = np.where(rng.random(n) < 0.5, 1, -1)
        resp = simulate._conv_response(spec, pos, vals, s)
        c = np.zeros((n, d), np.int8)
        np.put_along_axis(c, pos, vals.astype(np.int8), axis=1)
        z = (eps[:, None, None] * resp).astype(np.int16)
        reg = ShiftRegressionChannel(spec)
        shat, info = reg.run(c, z, eps.astype(np.float64))
        err = np.abs(shat - s).max()
        check(f"{spec.name}", err < 1e-6,
              f"max|shat-s|={err:.2e} dead={info['dead_frequencies']}")


def t3_labels():
    print("T3: label LUT consistency")
    spec = SchemeSpec.bit128()
    lab = LabelChannel(spec)
    rng = np.random.default_rng(3)
    n = 2000
    pos = rng.random((n, spec.d)).argpartition(spec.tau, axis=1)[:, :spec.tau]
    c = np.zeros((n, spec.d), np.int8)
    np.put_along_axis(c, pos, 1, axis=1)
    eps = np.where(rng.random(n) < 0.5, 1, -1)
    y0 = simulate._triangular(rng, (n, spec.d), 8)
    z0 = y0 + eps[:, None] * c
    hard, soft, lam = lab.run(c, z0)
    acc = float((hard == eps).mean())
    check("bit128 label accuracy ~96%", 0.94 < acc < 0.985, f"acc={acc:.4f}")
    # signed challenge, narrow mask: much higher accuracy
    spec2 = SchemeSpec.dlt_toy()
    lab2 = LabelChannel(spec2)
    pos2 = rng.random((n, spec2.d)).argpartition(spec2.tau,
                                                 axis=1)[:, :spec2.tau]
    vals2 = rng.integers(0, 2, (n, spec2.tau)) * 2 - 1
    c2 = np.zeros((n, spec2.d), np.int8)
    np.put_along_axis(c2, pos2, vals2.astype(np.int8), axis=1)
    eps2 = np.where(rng.random(n) < 0.5, 1, -1)
    y02 = simulate._triangular(rng, (n, spec2.d), 4)
    z02 = y02 + eps2[:, None] * c2
    hard2, soft2, _ = lab2.run(c2, z02)
    acc2 = float((hard2 == eps2).mean())
    check("dlt_toy label accuracy > 99%", acc2 > 0.99, f"acc={acc2:.4f}")


def t4_operator():
    print("T4: moment operator vs closed-form lemma")
    from umodel.operator_lemma import closed_form_operator
    mom = QuantizedMomentChannel(SchemeSpec.bit128(), n_mc=4000, seed=7)
    Fop = mom.build_operator()
    Fcf, rho = closed_form_operator(256, 30)
    rel = float(np.linalg.norm(Fop - Fcf) / np.linalg.norm(Fcf))
    check("negacyclic/unsigned matches closed form", rel < 0.02,
          f"rel_err={rel:.4f} (MC noise at 4k draws)")
    # v5 lemma: signed challenges -> E[C^T C] = tau*I -> field operator 0
    spec_toy = SchemeSpec.dlt_toy()
    Fop_signed = QuantizedMomentChannel(spec_toy).build_operator()
    check("signed challenge -> zero field operator",
          float(np.abs(Fop_signed).max()) == 0.0)
    # numeric cross-check of the lemma at the toy parameters
    rng7 = np.random.default_rng(7)
    ar7 = np.arange(spec_toy.d)
    ECC = np.zeros((spec_toy.d, spec_toy.d))
    for _ in range(3000):
        pos = rng7.choice(spec_toy.d, spec_toy.tau, replace=False)
        vals = rng7.integers(0, 2, spec_toy.tau) * 2 - 1
        C = np.zeros((spec_toy.d, spec_toy.d))
        for t in range(spec_toy.tau):
            j = (ar7 + pos[t]) % spec_toy.d
            C[ar7, j] = np.where(ar7 + pos[t] >= spec_toy.d, -1.0, 1.0)                 * vals[t]
        ECC += C.T @ C
    ECC /= 3000
    rel = np.linalg.norm(ECC - spec_toy.tau * np.eye(spec_toy.d)) /         (spec_toy.tau * spec_toy.d ** 0.5)
    check("signed lemma numeric (E[C^T C] = tau I)", rel < 0.05,
          f"rel={rel:.4f} (MC noise)")
    # unsigned + cyclic still refused
    bad_spec = SchemeSpec.dlt_toy()
    bad_spec.challenge_signed = False
    try:
        QuantizedMomentChannel(bad_spec).build_operator()
        check("unsigned/cyclic refused", False)
    except NotImplementedError:
        check("unsigned/cyclic refused", True)


def t5_keygen():
    print("T5: simulated keygen satisfies the public relation")
    for spec in (SchemeSpec.bit128(), SchemeSpec.dlt_toy()):
        key = simulate.gen_keypair(spec, seed=11)
        q = spec.q
        lhs = (key["Mx"] @ key["s"].reshape(-1)) % q
        rhs = (key["b1f"] + key["R"].reshape(-1)) % q
        ok = np.array_equal(lhs, rhs)
        box = bool(np.all((key["R"] >= spec.relation.r_box[0])
                          & (key["R"] <= spec.relation.r_box[1])))
        tri = bool(np.all(np.isin(key["s"], (-1, 0, 1))))
        check(f"{spec.name}", ok and box and tri,
              f"relation={ok} r_box={box} ternary={tri}")


def t6_applicability():
    print("T6: applicability gate")
    a1 = check_applicability(SchemeSpec.bit128())
    a2 = check_applicability(SchemeSpec.dlt_toy())
    check("bit128 accepted", a1["applicable"])
    check("dlt_toy accepted", a2["applicable"])
    bad = SchemeSpec.dlt_toy()
    bad.d = 100
    a3 = check_applicability(bad)
    check("non-pow2 degree refused", not a3["applicable"])
    import dataclasses
    from umodel.spec import RingStructure
    bad2 = dataclasses.replace(SchemeSpec.bit128(),
                               ring=RingStructure("cyclic"))  # unsigned+cyclic
    a4 = check_applicability(bad2)
    check("bit128 unsigned/cyclic refused (moment operator scope)",
          not a4["applicable"])


def t7_chunked_channel():
    """Chunked latent channel: oracle-sign exact recovery + chunk label API."""
    print("T7: chunked-sign channels (CS-style)")
    from umodel.channels import LabelChannel, ShiftRegressionChannel
    from umodel import niccs
    import numpy as np
    rng = np.random.default_rng(5)
    spec = niccs.cs128()
    sizes = spec.chunk_sizes
    d, tau = spec.d, spec.tau
    n = 400
    # synthetic transcript following CS semantics: z0 = y0 + sum eps_i c_i,
    # z1 = y1 + sum eps_i (c_i * s)
    s = rng.integers(-1, 2, (spec.L, d)).astype(np.int64)
    pos = rng.random((n, d)).argpartition(tau, axis=1)[:, :tau]
    vals = (rng.integers(0, 2, (n, tau)) * 2 - 1).astype(np.int64)
    ordr = np.argsort(pos, axis=1)
    pos = np.take_along_axis(pos, ordr, axis=1)
    vals = np.take_along_axis(vals, ordr, axis=1)
    c = np.zeros((n, d), np.int64)
    np.put_along_axis(c, pos, vals, axis=1)
    eps = np.where(rng.random((n, len(sizes))) < 0.5, 1.0, -1.0)
    # chunk spectra via the adapter
    c8 = c.astype(np.int8)
    chunk_spectra = niccs.cs_chunk_spectra(c8, sizes)
    # exact responses: z1 = sum_i eps_i (c_i * s) with true signs, no noise
    ar = np.arange(d)
    resp = np.zeros((n, spec.L, d), np.float64)
    starts = np.cumsum((0,) + sizes)
    for i, sz in enumerate(sizes):
        for j in range(sz):
            t = pos[:, starts[i] + j]                 # (n,)
            v = vals[:, starts[i] + j] * eps[:, i]    # (n,)
            delta = ar[None, :] - t[:, None]          # (n,d)
            flip = np.where(delta < 0, -1.0, 1.0)      # (n,d)
            sp = s[:, delta % d].astype(np.float64) * flip[None, :, :]
            resp += v[:, None, None] * sp.transpose(1, 0, 2)
    reg = ShiftRegressionChannel(spec)
    sh, _ = reg.run_chunked(chunk_spectra, resp, eps)
    err = np.abs(sh - s).max()
    check("chunked regression exact recovery", err < 1e-6, f"max err {err:.2e}")
    # chunk label API runs and returns the right shapes
    lab = LabelChannel(spec)
    y0 = rng.integers(-60, 61, (n, d)).astype(np.int16)
    z0 = y0 + np.zeros((n, d))
    for i, sz in enumerate(sizes):
        z0[np.arange(n)[:, None], pos[:, starts[i]:starts[i]+sz]] +=             (eps[:, i][:, None] * vals[:, starts[i]:starts[i]+sz]).astype(np.int16)
    hard, soft = lab.run_chunked(c8, z0, sizes)
    check("chunked label shapes", hard.shape == (n, len(sizes))
          and soft.shape == hard.shape)


def t8_assay():
    """Assay separates informative from uninformative labels (synthetic)."""
    print("T8: latent-information assay")
    from umodel.channels import LabelChannel
    from umodel.assay import LatentInformationAssay
    from umodel.spec import SchemeSpec, ChannelSpec, RingStructure, MaskModel
    import numpy as np
    rng = np.random.default_rng(6)
    for tag, g0, expect in (("informative", 4, "labels_informative"),
                            ("wide-mask", 64, "labels_uninformative")):
        spec = SchemeSpec(
            name=f"assay-{tag}", d=64, q=8380417, tau=12,
            challenge_signed=True, ring=RingStructure("negacyclic"),
            K=1, L=1, shared_latent="chunked-signs", chunk_sizes=(12,),
            channels={"labels": ChannelSpec("labels", "labels", "z0,c",
                                            "eps", MaskModel("uniform", g0))})
        n = 4000
        d, tau = spec.d, spec.tau
        pos = rng.random((n, d)).argpartition(tau, axis=1)[:, :tau]
        vals = (rng.integers(0, 2, (n, tau)) * 2 - 1).astype(np.int64)
        c = np.zeros((n, d), np.int64)
        np.put_along_axis(c, pos, vals, axis=1)
        eps = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        y0 = rng.integers(-(g0 - 1), g0, (n, d))
        z0 = y0 + eps[:, None] * c
        lab = LabelChannel(spec)
        assay = LatentInformationAssay(spec)
        a = assay.label_split_half(c.astype(np.int8), z0, (tau,),
                                   lambda o, v: lab._lookup(o, v))
        check(f"assay verdict [{tag}]", a["verdict"] == expect,
              f"corr={a.get('corr_split_half', float('nan')):+.3f} "
              f"-> {a['verdict']}")


def t9_public_decoders():
    """Known-answer checks for the pure-Python public decoders (no local
    audit data needed): SM3 CTR-DRNG stream, SM3 pseudoXOF, rANS tables."""
    print("T9: public decoder known-answer checks")
    from umodel import public_decoders as pd
    import numpy as np
    # DRNG: reference values captured from the C implementation
    cwave = bytes((i * 7 + 1) & 0xFF for i in range(32))
    st = pd.Sm3CtrDrng(cwave)
    ok_v = bytes(st.V[:8]).hex() == "c8d8aaf4c315511c"
    b = st.generate(24)
    ok_g = b[:3].hex() == "f8be16"
    check("SM3 CTR-DRNG instantiate/generate (C reference vector)",
          ok_v and ok_g, f"V={bytes(st.V[:4]).hex()} b={b[:3].hex()}")
    # SampleInBall port vs C vector (single positions test omitted; the
    # full-vector check runs in run_niccs_test with local data)
    c = pd.cs_sample_in_ball(cwave)
    nz = int((c != 0).sum())
    vals_ok = bool(np.all(np.isin(np.unique(c[c != 0]), (-1, 1))))
    check("CS SampleInBall shape/values", nz == 23 and vals_ok)
    # COMPASS challenge via SM3 pseudoXOF
    c2 = pd.compass_sample_in_ball(cwave)
    nz2 = int((c2 != 0).sum())
    check("COMPASS SampleInBall shape/values", nz2 == 30)
    # rANS table integrity
    t = pd._load_cs_tables() if hasattr(pd, "_load_cs_tables") else         {k: v for k, v in (("dsyms_z0", pd.Z0_TABLE),
                           ("dsyms_hb_z1", pd.Z1_TABLE),
                           ("dsyms_z2", pd.Z2_TABLE))}
    ok_t = (sum(f for _, f in t["dsyms_z0"]) == 2048
            and sum(f for _, f in t["dsyms_hb_z1"]) == 2048
            and sum(f for _, f in t["dsyms_z2"]) == 1024)
    check("rANS symbol tables complete (freq sums)", ok_t)


def t10_fixedpoint():
    """v6: fixed-point Approx invariants."""
    from umodel.cs_fixedpoint import _approx
    import numpy as np
    for x, y in ((1, 15150943174432), (2 ** 63, 2 ** 63 - 1),
                 (12345678901234567890, 9876543210987654321)):
        m, e = _approx(x, y)
        ok = (m << e) <= x * y < ((m + 1) << e) if e else m == x * y
        check(f"Approx({x.bit_length()}b,{y.bit_length()}b) reconstructs",
              bool(ok))



if __name__ == "__main__":
    t1_algebra()
    t2_regression()
    t3_labels()
    t4_operator()
    t5_keygen()
    t6_applicability()
    t7_chunked_channel()
    t8_assay()
    t9_public_decoders()
    t10_fixedpoint()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
