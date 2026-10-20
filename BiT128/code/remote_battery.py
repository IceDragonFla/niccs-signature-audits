"""remote_battery.py — Group-parallel driver for the full NICCS estimation battery.

Usage: sage -python remote_battery.py <group>
  groups: flit | zen | nev   (flit also runs the Kyber-512 validation first)

Runs primal_usvp + dual x CoreSVP/MATZOV for each parameter set in the group,
writes results_<group>.json alongside stdout log.
"""
import sys, os, time, json

HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/niccs_est/lattice-estimator")

from estimator.schemes import Kyber512, SparseTernary
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_primal import primal_usvp
from estimator.lwe_dual import dual as dual_attack
from estimator.reduction import ADPS16, MATZOV

CORE_SVP = ADPS16()
MATZOV_M = MATZOV()

GROUPS = {
    "flit": [("Kyber512-val", None, None, None, 117.0, 108.0)] + [
        ("Flit-128", 512, 769, 64, 134.0, 129.7),
        ("Flit-256", 1024, 769, 256, 269.5, 253.2),
        ("Flit-512", 2048, 3329, 896, 529.1, 488.5),
    ],
    "zen": [
        ("ZEN-128", 512, 769, 64, 128.5, 116.6),
        ("ZEN-256", 1024, 769, 128, 257.5, 233.7),
        ("ZEN-512", 2048, 769, 256, 519.9, 477.5),
    ],
    "nev": [
        ("NEV-R1", 1409, 4673, 472, 135.0, 119.0),
        ("NEV-R2", 1409, 7681, 472, 284.0, 250.0),
        ("NEV-R3", 1409, 12289, 472, 583.0, 513.0),
    ],
}


def run_group(group):
    out = {}
    for label, n, q, w, claim_C, claim_Q in GROUPS[group]:
        if label == "Kyber512-val":
            params = Kyber512
        else:
            Xs = SparseTernary(p=w // 2, n=n)
            Xe = SparseTernary(p=w // 2, n=n)
            params = LWEParameters(n=n, q=q, Xs=Xs, Xe=Xe, m=n)
        res_scheme = {"claim_C": claim_C, "claim_Q": claim_Q, "attacks": {}}
        for atk_name in ["primal_usvp", "dual"]:
            for mdl_name, model in [("CoreSVP", CORE_SVP), ("MATZOV", MATZOV_M)]:
                key = f"{atk_name}_{mdl_name}"
                t0 = time.time()
                try:
                    if atk_name == "primal_usvp":
                        res = primal_usvp(params, red_cost_model=model)
                    else:
                        res = dual_attack(params, red_cost_model=model)
                    d = {k: float(res[k]) for k in ("beta",) if k in res}
                    d["log2"] = float(res.get("rop", res.get("cost", 0)))
                    d["C_coresvp"] = 0.292 * d["beta"]
                    d["Q_coresvp"] = 0.257 * d["beta"]
                    d["t"] = round(time.time() - t0, 1)
                    res_scheme["attacks"][key] = d
                except Exception as e:
                    res_scheme["attacks"][key] = {"error": f"{type(e).__name__}: {e}"[:200]}
                print(f"[{label}] {key}: {res_scheme['attacks'][key]}", flush=True)
        valid = [(k, v) for k, v in res_scheme["attacks"].items() if "beta" in v]
        if valid:
            bk, bv = min(valid, key=lambda x: x[1]["C_coresvp"])
            res_scheme["best"] = {"key": bk, "C": bv["C_coresvp"], "Q": bv["Q_coresvp"],
                                  "gap_C": bv["C_coresvp"] - claim_C}
        out[label] = res_scheme
        with open(f"{HOME}/niccs_est/results_{group}.json", "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    run_group(sys.argv[1])
