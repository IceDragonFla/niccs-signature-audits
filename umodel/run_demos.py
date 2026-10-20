"""run_demos.py — seven demonstrations of the umodel framework (v3):

  Demo 1  BiT-128 hint channel, real data artifacts  -> key reproduced
  Demo 2  BiT-128 z1 channel + MITM decoder, artifacts -> key reproduced
  Demo 3  Diagnosis on a VULNERABLE synthetic instance (leak present)
  Demo 4  Diagnosis on the FIXED variant (independent signs -> no leak)
  Demo 5  TRANSFER: DLT-toy (cyclic ring, signed challenge, q=8380417,
          different widths) end-to-end through UniversalPipeline —
          zero engine changes, equivalent key recovered and scored
  Demo 6  TRANSFER of the diagnosis: toy vulnerable vs toy fixed variant
  Demo 7  Data-volume sweep on the toy: sigma(N) ~ N^-1/2 law, decode
          boundary vs required_N prediction, interval-only below it

Demos 3-7 generate synthetic transcripts with the spec-driven simulator
(umodel.simulate); demos 1-2 read audit artifacts.
"""
import hashlib
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel import (SchemeSpec, LabelEngine, Z1WlsEngine,
                    HintOperatorEngine, DecoderLadder, SharedLatentDiagnosis)
from umodel.pipeline import UniversalPipeline
from umodel import simulate

D, Q = 256, 26881


def load_common():
    g16 = np.fromfile("tmp/bit128_1m/ground.bin", dtype="<i2").astype(np.int64)
    s0_ref = g16[:3*D].reshape(3, D)
    b1 = g16[6*D:9*D].reshape(3, D)
    A0 = g16[12*D:].reshape(3, 3, D)
    r_ = np.arange(D)
    Mx = np.zeros((3*D, 3*D), np.int64)
    for i in range(3):
        for p in range(3):
            idxm = (r_[:, None] - r_[None, :]) % D
            sgn = np.where(r_[None, :] > r_[:, None], -1, 1)
            Mx[i*D:(i+1)*D, p*D:(p+1)*D] = A0[i, p][idxm] * sgn
    n = 3*D
    Aug = np.concatenate([Mx % Q, np.eye(n, dtype=np.int64)], axis=1).astype(np.int64)
    for k in range(n):
        piv = k + int(np.argmax(Aug[k:, k] != 0))
        Aug[[k, piv]] = Aug[[piv, k]]
        iv = pow(int(Aug[k, k]), Q-2, Q)
        Aug[k] = (Aug[k]*iv) % Q
        col = Aug[:, k].copy(); col[k] = 0
        Aug = (Aug - col[:, None]*Aug[k][None, :]) % Q
    return s0_ref, b1, A0, Mx, Aug[:, n:]


def demo1(spec, ladder):
    print("=" * 62)
    print("[Demo 1] BiT-128 hint channel (artifacts, N=1.5M)")
    eng = HintOperatorEngine(spec)
    Phi = np.load("tmp/attack2/phi_xc_1500.npy").astype(np.float64)
    Fop = np.load("tmp/attack2/Fop_150000.npy").astype(np.float64)
    N = 1_500_000
    M = N * spec.tau
    m_axis = np.arange(spec.d)
    Rh = np.zeros((3, spec.d))
    for i in range(3):
        best = None
        for s in np.arange(0.90, 0.97, 0.001):
            phi = Phi[i].copy()
            if i == 0:
                phi = phi + (float(s)*0.5*M*(1.0-2.0*m_axis/spec.d))/spec.gamma2
            Rh_ = np.linalg.solve(float(s)*M*np.eye(spec.d) + N*Fop,
                                  -spec.gamma2*phi)
            o = float(np.cos(2*np.pi*Rh_).mean())
            if best is None or o > best[0]:
                best = (o, Rh_)
        Rh[i] = best[1]
    R, s0v = ladder.rounding(Rh)
    if R is None:
        R, s0v = ladder.pair_scan(Rh)
    return s0v, float((R is not None))


def demo2(spec, ladder, Mx):
    print("=" * 62)
    print("[Demo 2] BiT-128 z1 channel + MITM decoder (artifact, N=175k)")
    sh = np.load("tmp/attack2/z1_shat_175000.npy")
    s, lik = ladder.mitm_window(sh, Mx, W=37)
    return s, lik


# ------------- synthetic tier-1 generator (spec-driven, v2 legacy) -------------
def gen_synthetic(spec, n_sigs, seed, independent_signs=False):
    """v2 legacy generator (negacyclic BiT shape) for demos 3-4."""
    rng = np.random.default_rng(seed)
    d, tau, g1 = spec.d, spec.tau, spec.gamma1
    L = spec.L
    s0 = rng.integers(-1, 2, (L, d))
    T1 = np.arange(-g1 + 1, g1)
    W1 = (g1 - np.abs(T1)).astype(np.float64); W1 /= W1.sum()
    cum1 = np.cumsum(W1)
    y1 = T1[np.searchsorted(cum1, rng.random((n_sigs, L, d)))]
    pos = np.stack([rng.choice(d, tau, replace=False) for _ in range(n_sigs)])
    ar = np.arange(d)
    a = np.zeros((n_sigs, L, d))
    for t in range(tau):
        p_ = pos[:, t]
        idx = (ar[None, :] - p_[:, None]) % d
        sgn = np.where(ar[None, :] - p_[:, None] < 0, -1.0, 1.0)
        a += sgn[:, None, :] * s0[:, idx].transpose(1, 0, 2)
    eps_true = np.where(rng.random(n_sigs) < 0.5, 1.0, -1.0)
    eps_z1 = (np.where(rng.random(n_sigs) < 0.5, 1.0, -1.0)
              if independent_signs else eps_true)
    z1 = y1 + eps_z1[:, None, None] * a
    c_can = np.zeros((n_sigs, d), np.int64)
    np.put_along_axis(c_can, pos, 1, axis=1)
    T0 = np.arange(-7, 8)
    W0 = (8 - np.abs(T0)).astype(np.float64); W0 /= W0.sum()
    y0 = T0[np.searchsorted(np.cumsum(W0), rng.random((n_sigs, d)))]
    z0 = y0 + eps_true[:, None] * c_can
    return s0, c_can, z0, z1, eps_true


def demo_diagnosis(spec, n_sigs=60_000, seed=7, independent=False):
    tag = "FIXED (independent signs)" if independent else "VULNERABLE"
    print("=" * 62)
    print(f"[Demo {'4' if independent else '3'}] diagnosis on {spec.name} — {tag}")
    s0, c_can, z0, z1s, eps_true = gen_synthetic(spec, n_sigs, seed,
                                                 independent_signs=independent)
    lab = LabelEngine(spec)
    wls = Z1WlsEngine(spec)
    diag = SharedLatentDiagnosis(spec, lab, wls)
    res = diag.run(c_can, z0.astype(np.int64), z1s)
    r = res["D_split_half_corr"]
    print(f"  split-half corr = {r['value']:+.3f} "
          f"(shuffled control {r['control']:+.3f}) -> leak: {r['leak']}")
    if r["leak"]:
        hard, soft = lab.run(c_can, z0.astype(np.int64))
        sh = wls.run(c_can, z1s, soft)
        sigma = float((sh - s0).std())
        reqN = wls.required_N(sigma, n_sigs, 0.5)
        print(f"  z1 estimate sigma = {sigma:.3f} @ N={n_sigs}; "
              f"predicted N for sigma=0.5: {reqN:,.0f}")
    else:
        print("  -> no attack regardless of sample count (fix verified)")
    return res


# ================= v3 transfer + volume demos =================
def demo5_transfer(n_sigs=200_000):
    print("=" * 62)
    print(f"[Demo 5] TRANSFER: {SchemeSpec.dlt_toy().name} end-to-end "
          f"(cyclic ring, signed challenge, q=8380417), N={n_sigs:,}")
    spec = SchemeSpec.dlt_toy()
    key = simulate.gen_keypair(spec, seed=42)
    tr = simulate.gen_transcript(spec, key, n_sigs, seed=43)
    pub = {"A0": key["A0"], "b1": key["b1"]}
    pipe = UniversalPipeline(spec, pub)
    rep = pipe.run(tr["c"], z0=tr["z0"], z1=tr["z1"], ground=key)
    d = rep["diagnosis"]["D_split_half_corr"]
    print(f"  diagnosis: split-half corr {d['value']:+.3f} "
          f"(control {d['control']:+.3f}) -> leak={d['leak']}")
    ch = rep["channels"]["z1-regression"]
    print(f"  z1 estimate: sigma={ch['sigma']:.4f}  "
          f"required_N(sigma=0.5)={ch['required_N_sigma0.5']:,.0f}")
    print(f"  decode: stage={rep['decode']['stage']} "
          f"status={rep['decode']['status']}")
    print(f"  validation: {rep['validation']}")
    print(f"  scoring: {rep.get('scoring')}")
    return rep


def demo6_toy_diagnosis(n_sigs=60_000):
    spec = SchemeSpec.dlt_toy()
    for tag, indep in (("VULNERABLE", False), ("FIXED (independent signs)",
                                               True)):
        print("-" * 62)
        print(f"[Demo 6] toy diagnosis — {tag}")
        key = simulate.gen_keypair(spec, seed=99)
        tr = simulate.gen_transcript(spec, key, n_sigs, seed=100,
                                     independent_signs=indep)
        pipe = UniversalPipeline(spec)
        rep_d = pipe.diagnose(tr["c"], tr["z0"], tr["z1"])
        d = rep_d["D_split_half_corr"]
        print(f"  split-half corr = {d['value']:+.3f} "
              f"(control {d['control']:+.3f}) -> leak: {d['leak']}")


def demo7_volume_sweep(ns=(2_500, 5_000, 10_000, 25_000, 50_000, 100_000,
                           200_000, 400_000)):
    print("=" * 62)
    print("[Demo 7] data-volume sweep on DLT-toy (independent of N by "
          "contract)")
    spec = SchemeSpec.dlt_toy()
    key = simulate.gen_keypair(spec, seed=42)
    pub = {"A0": key["A0"], "b1": key["b1"]}
    pipe = UniversalPipeline(spec, pub)
    n_max = max(ns)
    tr = simulate.gen_transcript(spec, key, n_max, seed=43)
    rows = []
    for n in ns:
        rep = pipe.run(tr["c"][:n], z0=tr["z0"][:n], z1=tr["z1"][:n],
                       ground=key, diagnose=False)
        ch = rep["channels"].get("z1-regression", {})
        sig = ch.get("sigma", float("nan"))
        dec = rep["decode"]
        # [SCORING] truth-based sigma for the scaling law
        _, est = pipe.estimate(tr["c"][:n], z0=tr["z0"][:n], z1=tr["z1"][:n])
        sh_err = (float(np.std(est["shat"] - key["s"]))
                  if "shat" in est else None)
        rows.append((n, sig, sh_err, dec.get("stage"), dec.get("status"),
                     dec.get("required_N_decode", float("nan")),
                     rep.get("scoring", {}).get("s0_exact")))
    print(f"  {'N':>8}  {'sig_pub':>8}  {'sig_true':>9}  {'stage':>10}  "
          f"{'status':>14}  {'req_N':>7}  exact")
    for n, sig, st, stage, status, req, ex in rows:
        print(f"  {n:>8,}  {sig:>8.4f}  {st:>9.4f}  {str(stage):>10}  "
              f"{status:>14}  {req:>7,.0f}  {ex}")
    # sigma ~ N^-1/2 fit on the truth-based sigma
    st_arr = [r[2] for r in rows if r[2] is not None and np.isfinite(r[2])]
    logs = np.polyfit(np.log(np.array(ns, float)), np.log(st_arr), 1)
    print(f"  fit log sigma_true = {logs[0]:.3f} x log N  (expect -0.5)")
    req = [r[5] for r in rows if np.isfinite(r[5])]
    if req:
        print(f"  required_N_decode (constant by the N^-1/2 law): "
              f"{np.mean(req):,.0f}")
    return rows



def demo8_niccs():
    """[needs local audit data] Real NICCS implementations through the
    framework: CS-128 (chunked bimodal latent) and COMPASS-SIG-128 (no
    latent). Graceful skip when the dumps are absent."""
    print("=" * 62)
    print("[Demo 8] REAL NICCS instances (CS-128 @200k, COMPASS-128 @100k)")
    try:
        from umodel import run_niccs_test as bat
    except ImportError:
        print("  battery module missing; skip")
        return
    import os
    if not os.path.exists(bat.CS_DUMP):
        print("  audit dumps not present in this checkout; skip")
        return
    bat.test_cs128()
    print()
    bat.test_compass128()


if __name__ == "__main__":
    spec = SchemeSpec.bit128()
    have_artifacts = all(os.path.exists(p_) for p_ in (
        "tmp/bit128_1m/ground.bin", "tmp/attack2/phi_xc_1500.npy",
        "tmp/attack2/Fop_150000.npy", "tmp/attack2/z1_shat_175000.npy",
        "tmp/attack2/key_hat_h_real.bin"))
    if have_artifacts:
        s0_ref, b1, A0, Mx, Ainv = load_common()
        b1f = (16*(b1 % Q)).reshape(3*D)
        ladder = DecoderLadder(spec, Ainv, b1f, A0)
        href = hashlib.sha256(open("tmp/attack2/key_hat_h_real.bin","rb").read()).hexdigest()

        s1, ok1 = demo1(spec, ladder)
        m1 = int((s1.reshape(3, D) == s0_ref).sum()) if s1 is not None else 0
        h1 = hashlib.sha256(s1.reshape(3, D).astype(np.int16).tobytes()).hexdigest() if s1 is not None else ""
        print(f"  -> s0 {m1}/768, key-material match: {h1 == href[:len(h1)] or m1 == 768}")

        s2, lik2 = demo2(spec, ladder, Mx)
        m2 = int((s2.reshape(3, D) == s0_ref).sum()) if s2 is not None else 0
        print(f"  -> MITM lik={lik2:.2f}, s0 {m2}/768")

        demo_diagnosis(SchemeSpec.bit128(), independent=False)
        demo_diagnosis(SchemeSpec.bit128_fixed(), independent=True)

    else:
        print("[Demos 1-4 skipped: audit artifacts not present in this checkout]")

    demo5_transfer()
    demo6_toy_diagnosis()
    demo7_volume_sweep()
    demo8_niccs()
    print("=" * 62)
    print("[umodel] demos complete")
