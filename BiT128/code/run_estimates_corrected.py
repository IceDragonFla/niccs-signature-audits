"""remote_battery_v2.py — CORRECTED parameter battery (from design docs Table 2/3/1).

Root cause found: v1 used wrong secret/error distributions and wrong NEV params.
This battery uses the EXACT documented parameters:

FLIT (spec Table 2): density-ternary T_a, variance = a
  128: RLWE r~T1/8, e~T5/16; NTRU f~T1/8, g~T5/16 (n=512, q=769)
  256: r~T5/32, e~T1/4; f~T5/32, g~T1/4 (n=1024, q=769)
  512: all T7/16 (n=2048, q=3329)
ZEN (spec Table 3): (Dg, Df, Ds, De)
  128: (B1+S1/8, B1, B1+S3/32, B2) n=512 q=769
  256: (S3/16, S1/8, B1, B1) n=1024 q=769
  512: (S3/32, S3/32, S1/8, S1/8) n=2048 q=769
NEV (spec Table 2): R-series all q=1409
  R1: n=512,  f,g,r,e ~ B3
  R2: n=1024, f,g,r,e ~ B2
  R3: n=2048, f,g ~ T1/3; r ~ T1/3, e ~ B2
DTRU (spec Table 1): Kyber script, (kr, ke):
  648: (B2,B2); 768: (B4,B2); 1024: (B3,B2); 1536: (B2,B1); 2048: (B2,B1)

Proxies: B_k -> CenteredBinomial(k); S_a/T_a -> SparseTernary(p=a*n/2)
         B1+S_x -> DiscreteGaussian(sqrt(1/2 + x)) (variance-matched)

Usage: python remote_battery_v2.py <flit|zen|nev|dtru>
"""
import sys, os, time, json, math

HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/niccs_est/lattice-estimator")

from estimator.schemes import SparseTernary, CenteredBinomial, DiscreteGaussian
from estimator.lwe_parameters import LWEParameters
from estimator.ntru_parameters import NTRUParameters
from estimator.lwe_primal import primal_usvp
from estimator.lwe_dual import dual as dual_attack
from estimator.lwe_dual import dual_hybrid
from estimator.ntru_primal import primal_bdd
from estimator.reduction import ADPS16, MATZOV

CORE_SVP, MATZOV_M = ADPS16(), MATZOV()


def ST(density, n):
    """Density-ternary via fixed-weight SparseTernary (variance & density matched)."""
    return SparseTernary(p=int(round(density * n / 2)), n=n)


def DG(sigma):
    return DiscreteGaussian(sigma)


def run_one(fn, params, model, label, key, out):
    t0 = time.time()
    try:
        res = fn(params, red_cost_model=model)
        d = {"beta": float(res["beta"]), "C_coresvp": 0.292 * float(res["beta"]),
             "Q_coresvp": 0.257 * float(res["beta"]),
             "log2rop": math.log2(float(res["rop"])) if float(res["rop"]) > 0 else None,
             "t": round(time.time() - t0, 1)}
    except Exception as e:
        d = {"error": f"{type(e).__name__}: {e}"[:180]}
    print(f"[{label}] {key}: {d}", flush=True)
    out[key] = d


def battery(label, n, q, Xs_r, Xe_r, Xs_ntru, Xe_ntru, claim, results, circulant=True):
    """RLWE (primal/dual/dual_hybrid) + NTRU primal_bdd, each x CoreSVP/MATZOV."""
    sc = {"claim": claim, "rlwe": {}, "ntru": {}}
    params = LWEParameters(n=n, q=q, Xs=Xs_r, Xe=Xe_r, m=n)
    for key, fn in [("primal_usvp", primal_usvp), ("dual", dual_attack),
                    ("dual_hybrid", dual_hybrid)]:
        for mn, m in [("CoreSVP", CORE_SVP), ("MATZOV", MATZOV_M)]:
            if key == "dual_hybrid" and mn == "MATZOV":
                continue  # dual_hybrid MATZOV is very slow; CoreSVP captures it
            run_one(fn, params, m, label, f"{key}_{mn}", sc["rlwe"])
    nt = NTRUParameters(n=n, q=q, Xs=Xs_ntru, Xe=Xe_ntru,
                        ntru_type="circulant" if circulant else "matrix")
    for mn, m in [("CoreSVP", CORE_SVP), ("MATZOV", MATZOV_M)]:
        run_one(primal_bdd, nt, m, label, f"primal_bdd_{mn}", sc["ntru"])
    results[label] = sc
    return sc


def dump(group, results):
    with open(f"{HOME}/niccs_est/v2_results_{group}.json", "w") as f:
        json.dump(results, f, indent=1, default=float)


def group_flit():
    res = {}
    battery("Flit128-RLWE/NTRU", 512, 769,
            ST(1 / 8, 512), ST(5 / 16, 512), ST(1 / 8, 512), ST(5 / 16, 512),
            {"rlwe_cs": 118.8, "ntru_cs": 118.6, "matzov_min": 134.0}, res)
    dump("flit", res)
    battery("Flit256", 1024, 769,
            ST(5 / 32, 1024), ST(1 / 4, 1024), ST(5 / 32, 1024), ST(1 / 4, 1024),
            {"rlwe_cs": 267.2, "ntru_cs": 267.2, "matzov_min": 269.5}, res)
    dump("flit", res)
    battery("Flit512", 2048, 3329,
            ST(7 / 16, 2048), ST(7 / 16, 2048), ST(7 / 16, 2048), ST(7 / 16, 2048),
            {"rlwe_cs": 550.4, "ntru_cs": 550.1, "matzov_min": 529.1}, res)
    dump("flit", res)


def group_zen():
    res = {}
    battery("ZEN128", 512, 769,
            DG(math.sqrt(11 / 16)), CenteredBinomial(2),          # s=B1+S3/32, e=B2
            CenteredBinomial(1), DG(math.sqrt(3 / 4)),            # f=B1, g=B1+S1/8
            {"rlwe_cs": 129.4, "ntru_cs": 128.5}, res)
    dump("zen", res)
    battery("ZEN256", 1024, 769,
            CenteredBinomial(1), CenteredBinomial(1),             # s=e=B1
            ST(1 / 8, 1024), ST(3 / 16, 1024),                    # f=S1/8, g=S3/16
            {"rlwe_cs": 261.9, "ntru_cs": 257.5}, res)
    dump("zen", res)
    battery("ZEN512", 2048, 769,
            ST(1 / 8, 2048), ST(1 / 8, 2048),                     # s=e=S1/8
            ST(3 / 32, 2048), ST(3 / 32, 2048),                   # f=g=S3/32
            {"rlwe_cs": 519.9, "ntru_cs": 526.2}, res)
    dump("zen", res)


def group_nev():
    res = {}
    battery("NEV-R1", 512, 1409,
            CenteredBinomial(3), CenteredBinomial(3),
            CenteredBinomial(3), CenteredBinomial(3),
            {"rlwe_primal_cs": 135, "ntru_cs": 135}, res)
    dump("nev", res)
    battery("NEV-R2", 1024, 1409,
            CenteredBinomial(2), CenteredBinomial(2),
            CenteredBinomial(2), CenteredBinomial(2),
            {"rlwe_primal_cs": 284, "ntru_cs": 284}, res)
    dump("nev", res)
    battery("NEV-R3", 2048, 1409,
            ST(1 / 3, 2048), CenteredBinomial(2),                 # r~T1/3, e~B2
            ST(1 / 3, 2048), ST(1 / 3, 2048),                     # f=g~T1/3
            {"rlwe_primal_cs": 596, "ntru_cs": 583}, res)
    dump("nev", res)


def group_dtru():
    """Kyber script with documented (kr, ke); try secret-sigma and error-sigma."""
    sys.path.insert(0, f"{HOME}/niccs_est/security-estimates")
    from MLWE_security import MLWE_optimize_attack, LWE_primal_cost, svp_classical
    res = {}
    sets = [
        ("DTRU-648", 648, 3457, 2, 2, 164.0),
        ("DTRU-768", 768, 3457, 4, 2, 198.0),
        ("DTRU-1024", 1024, 3457, 3, 2, 271.0),
        ("DTRU-1536", 1536, 3457, 2, 1, 411.0),
        ("DTRU-2048", 2048, 3457, 2, 1, 567.0),
    ]
    for name, n, q, kr, ke, claim in sets:
        sc = {"claim": claim, "kr": kr, "ke": ke, "runs": {}}
        for tag, s in [("sigma_r", math.sqrt(kr / 2)), ("sigma_e", math.sqrt(ke / 2)),
                       ("sigma_eff", math.sqrt((kr + ke) / 4))]:
            t0 = time.time()
            try:
                m, b, c = MLWE_optimize_attack(q, n, n, s, cost_attack=LWE_primal_cost,
                                               cost_svp=svp_classical, verbose=False)
                sc["runs"][tag] = {"s": s, "beta": b, "C": c, "m": m,
                                   "t": round(time.time() - t0, 1)}
            except Exception as e:
                sc["runs"][tag] = {"s": s, "error": str(e)[:120]}
            print(f"[{name}] {tag}: {sc['runs'][tag]}", flush=True)
        res[name] = sc
        dump("dtru", res)


if __name__ == "__main__":
    {"flit": group_flit, "zen": group_zen, "nev": group_nev,
     "dtru": group_dtru}[sys.argv[1]]()
    print("GROUP DONE", flush=True)
