"""run_spec_battery.py (v15) — one-command acceptance battery for any spec.

The transfer-validation protocol formalized: given a SchemeSpec, run
applicability -> synthetic keypair+transcript -> diagnosis -> per-channel
estimate (labels + regression at a small N) -> verdict, and print a
one-line summary suitable for accepting a new scheme instance into the
framework. Used by selftest T11 and demo runs.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umodel.spec import check_applicability
from umodel import simulate
from umodel.channels import LabelChannel, ShiftRegressionChannel
from umodel.assay import LatentInformationAssay


def run_spec_battery(spec, n_sigs=20_000, seed=2026, verbose=True):
    rep = {"spec": spec.name, "applicability": check_applicability(spec)}
    if not rep["applicability"]["applicable"]:
        rep["verdict"] = "refused"
        rep["reasons"] = rep["applicability"]["failures"]
        if verbose:
            print(f"[{spec.name}] REFUSED: {rep['reasons']}")
        return rep
    try:
        key = simulate.gen_keypair(spec, seed=seed)
        tr = simulate.gen_transcript(spec, key, n_sigs, seed=seed + 1)
    except AssertionError as exc:
        rep["verdict"] = ("simulator-incompatible (square-relation "
                          f"limitation: {exc})")
        if verbose:
            print(f"[{spec.name}] {rep['verdict']}")
        return rep
    N = tr["c"].shape[0]
    hard = soft = None
    if "labels" in spec.channels and tr.get("z0") is not None:
        lab = LabelChannel(spec)
        hard, soft, _ = lab.run(tr["c"], tr["z0"])
        assay = LatentInformationAssay(spec)

        def lut(o, v):
            return lab._lookup(o, v)
        a = assay.label_split_half(tr["c"], tr["z0"],
                                   spec.chunk_sizes or (spec.tau,), lut)
        rep["assay"] = a["verdict"]
    if "z1-regression" in spec.channels and tr.get("z1") is not None:
        reg = ShiftRegressionChannel(spec)
        w = soft if soft is not None else np.ones(N)
        if spec.shared_latent == "chunked-signs":
            cs = None
            from umodel import niccs
            if hasattr(niccs, "cs_chunk_spectra"):
                cs = niccs.cs_chunk_spectra(tr["c"], spec.chunk_sizes)
                sh, info = reg.run_chunked(cs, tr["z1"].astype(np.float64),
                                           w)
        else:
            sh, info = reg.run(tr["c"], tr["z1"].astype(np.float64), w)
        rep["regression_sigma"] = float((sh - key["s"]).std())
        var_y = spec.channels["z1-regression"].mask.variance()
        rep["regression_law"] = float(np.sqrt(var_y / (spec.tau * N)))
    # verdict
    if rep.get("assay") == "labels_uninformative" and \
            spec.shared_latent in ("single-sign", "chunked-signs"):
        rep["verdict"] = "latent-invisible (label routes refused)"
    elif spec.shared_latent == "none":
        rep["verdict"] = ("mask-limited (required_N sigma=0.5 ~ "
                          f"{var_y / (spec.tau * 0.25):.3g})")
    elif spec.shared_latent == "deterministic-normalization":
        rep["verdict"] = "structure-only (instrumentation pending)"
    else:
        rep["verdict"] = "label-informative (attack route applicable)"
    if verbose:
        print(f"[{spec.name}] {rep['verdict']}  "
              f"sigma={rep.get('regression_sigma', float('nan')):.3f} "
              f"(law {rep.get('regression_law', float('nan')):.3f})")
    from umodel.verdict import verdict_from_battery
    rep["verdict_report"] = verdict_from_battery(spec, rep)
    rep["state"] = rep["verdict_report"].distinguish_key_states()
    if verbose:
        print("  " + rep["verdict_report"].line())
        print("  state: " + rep["state"])
    return rep


if __name__ == "__main__":
    from umodel.spec import SchemeSpec
    from umodel.niccs import (compass128, dilithium2,
                              shuttle128,
                              rhyme_geometry_synthetic)
    for f in (SchemeSpec.bit128, SchemeSpec.dlt_toy, compass128,
              rhyme_geometry_synthetic, shuttle128):
        run_spec_battery(f())
