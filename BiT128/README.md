# BiT-128: Full Equivalent-Key Recovery from Public Signatures (Hint-Channel Operator Attack)

[中文说明（README.zh-CN.md）](README.zh-CN.md)

Independent cryptanalysis of **BiT-128** (candidate *sign-02* of the
Next-generation Commercial Cryptographic Algorithms program), performed under
owner authorization on the submitter's own reference implementation.

**Result.** From **4,000,000 ordinary signatures** and the public key only —
no side channel, no instrumentation, no timing — we recover
`s0` exactly (768/768 coefficients), complete an **equivalent signing key**,
and produce fresh-message forgeries accepted by the **unmodified reference C
verifier (20/20, 4 negative controls rejected)**.

**Provenance.** This owner-authorized audit was initiated on **2026-09-20**.
The core defect localization (joint-distribution leakage of the shared
bimodal sign), the `z0` label channel, the equivalent-key algebra, and the
reference-verifier forgery loop were established and logged in the first
phase of the audit; the hint channel and its operator treatment followed.
All artifacts are date-stamped and deterministically reproducible.

---

## The attack (our route)

```
public signatures ──► z0-channel sign labels (96.2%)
        │
        ▼
HINT CHANNEL: in the signing-side hint argument
    w_a = w − z2 + b0*c_signed (+ row-0 b-term),
    the y2 terms of w and z2 cancel — w_a is INDEPENDENT of the fresh mask:
    w_a = base − (R * c_signed) + ⋯,  R = b0 − e
    ⇒ the published hint h quantizes (w − w_a) = y2 − (R * c_signed) − ⋯
      where y2 is fresh known-density noise and R is the entire unknown
      of the equivalent key: h is a direct first-order observation of R.
        │
        ▼
OPERATOR LAW: the aligned statistic carries a deterministic structure
that is a LINEAR functional of R. The alignment-combinatorics part is the
exact expectation under the distinct-position challenge model, computed by
Monte-Carlo (F_op); acceptance-conditioning terms are unmodeled and
empirically bounded (post-inversion residual σ ≈ 0.10 at 4×10⁶).
        │
        ▼
OPERATOR INVERSION + PER-ROW SCALE FIT: solve
    (M·I + N·F_op) R̂ = −(γ2/s)·Φ − P₀·M·(1 − 2m/d)·𝟙[i=0]
    with the label scale s fitted per row by an integer-clustering objective.
        │
        ▼
FULL-SOLUTION CONSISTENCY CORRECTOR: solve the mod-q system
    s0 = A0⁻¹(16·b1 + R)
    and accept a candidate correction ONLY if the ENTIRE 768-dim solution
    lands in {−1,0,1} ("nbad = 0"). Staged single→pair→triple scans.
        │
        ▼
EQUIVALENT KEY (ê, b̂0) ──► fresh-message forgery, accepted by the
                            unmodified reference verifier (20/20)
```

At N = 4×10⁶ the direct classification is **already exact (0 wrong
coordinates, no correction needed)**; the corrector is defense-in-depth.

## What is exact and what is equivalent

* `s0` is recovered **exactly**: 768/768 coefficients identical to the
  original key, cross-validated by two independent recovery routes and by
  the uniqueness of the mod-q solution under the window constraints.
* `(e, b0)` is completed **equivalently**: the public key pins only
  R = b0 − e; any (ê, b̂0 = R + ê) with ê ∈ {−1,0,1}, b̂0 ∈ [−8,8) signs
  identically. Our completion keeps ê = 0 wherever allowed (26 forced
  boundary coordinates). The verifier cannot distinguish the two keys'
  signatures; the two routes recover the **same** key byte-for-byte.

## Three-way comparison

A detailed three-way comparison of independent BiT-128 attacks (our
hint-channel operator route, the concurrent z1-regression route, and the
SSD deconvolution route of ePrint 2026/2232) is in
[paper/three_way_comparison.md](paper/three_way_comparison.md).

## Universal model (umodel/)

The BiT-128 attack above is the first validated instance of `umodel`, a
scheme-agnostic equivalent-key-recovery framework shipped at the repository top level in `../umodel/`
(architecture write-up: ../umodel/ARCHITECTURE_v17v18.md):

* **Declarative spec.** Everything scheme-specific — ring algebra, mask
  models, channel list, public-key relation, equivalent-key rule — lives in
  one `SchemeSpec`. Engines, the simulator, the decoder ladder and the
  validator read the spec, never scheme code.
* **Machine-checked scope.** `check_applicability` refuses what is not yet
  derived (e.g. the moment-channel operator for signed challenges or
  non-power-of-two rings) with explicit reasons, instead of emitting
  unvalidated operators.
* **Transfer validated.** On a cyclic-ring toy instance (cyclic ring
  Z_q[x]/(x^d-1), signed challenges, q=8380417, different mask widths,
  K=L=1) the same engines recover the equivalent key end-to-end with zero
  engine changes: diagnosis split-half corr +0.991 (control -0.009),
  rounding decode, V1-V3 public checks pass, s0 exact and the equivalent
  key exact against ground truth.
* **Data-volume-agnostic contract.** At any N the pipeline returns either a
  self-certified candidate or a certified interval plus a required-N
  prediction (coordinate-error-target semantics); the sweep verifies
  sigma proportional to N^(-1/2) (fitted exponent -0.497) and the empirical
  decode boundary matches the prediction. A fold-saturation limit of the
  public sigma proxy (beyond sigma ~ 0.3) is reported with explicit
  lower-bound semantics rather than hidden.
* **Tested on REAL NICCS candidates (v4).** Instrumented the submitters'
  reference implementations of two more lattice candidates and ran the
  same engines on real transcripts (needs the local audit dumps):
  - **CS-128** (BLISS-style per-chunk bimodal signs, 200k signatures):
    the latent structure exists (oracle-verified; with perfect labels
    76k signatures would suffice), but the symmetric rejection sampling
    makes the published marginals information-invisible (label accuracy
    52.6%, split-half assay -0.0002 vs null +0.0005) — the framework's
    data-driven applicability gate REFUSES the label routes with that
    reason instead of producing garbage. CS-128's security rests entirely
    on the rejection-sampling symmetry.
  - **COMPASS-SIG-128** (FS-aborts, no latent, 100k signatures):
    sigma(N) matches the law at all scales; mask-limited with
    required N ~ 4.8x10^7; split-half correlation -0.04 (no signal).
  The verdict matrix (BiT / DLT-toy / CS-128 / COMPASS-128) demonstrates
  the contract end to end: four regimes, four honest outcomes, same code.
* **Honest boundaries resolved (v5).** Each "explicitly not claimed" item
  from v4 is closed or quantified: (1) the moment operator for SIGNED
  challenges is derived — E[C^T C] = tau*I exactly (cross terms vanish),
  so signed-challenge schemes need no deblurring at all; (2) fold
  saturation of the public sigma proxy falls back to the public mask law;
  (3) required-N is reported in BOTH semantics (coordinate / decode);
  (4) pure-Python decoders for the REAL packed signature bytes (CS-128
  rANS + full SM3 CTR-DRNG port; COMPASS 16-bit z — including the finding
  that its "shake256" API is SM3-pseudoXOF in disguise) verify 200/200
  against the instrumented dumps; (5) the CS-128 latent information is
  QUANTIFIED by an exact-acceptance-model LLR: +0.180 +/- 0.023 nats per
  signature (8 sigma, NOT zero — the symmetric acceptance is imperfect)
  plus 0.19%/chunk deterministic norm-boundary reveals, implying even an
  optimal classifier needs ~2x10^6 signatures (practical: ~3x10^8), vs
  7.6x10^4 with perfect labels. A reference-implementation deviation
  (undefined-behavior alignment shift in the C+E merge when exponent gaps
  exceed 64) is documented as an unmodeled layer.
* **v6 (multi-round boundary closure, incl. one retraction).** (1) The
  v5 "alignment-shift UB" finding is RETRACTED — it was a format-string
  misalignment in our own instrumentation printf (shift counts in the
  reference binary never exceed 2); the real deviation is fixed-point
  rounding, ported BIT-EXACTLY (validated 240/240 against captured
  exp-test values), which does NOT change the information (actual-impl
  LLR +0.182 +/- 0.028 vs intended +0.180 +/- 0.023). (2) The y2-layer
  is formalized: y0/y1 are pinned by (z0,z1), y2 requires
  marginalization; the rigorous (z0,z1)-channel information is
  +0.122 +/- 0.032 nats/signature, bracketing the total in [0.12, 0.18].
  (3) The second-moment route is closed PRECISELY: measured estimator
  variance gives required_N = 4.0e10 (median), 4.7e12 (p90); odd moments
  through the same engines: mask-limited at 5.9e8 — the verdict matrix
  now spans five regimes (BiT / DLT-toy / CS-128 / COMPASS-128).

```bash
python3 ../umodel/selftest.py     # 29 numeric checks (ring identities, LUTs,
                               # operator closed form, keygen relation, gates,
                               # chunked channels, latent-information assay)
python3 ../umodel/run_demos.py    # demos 3-7 are self-contained; demos 1-2 and 8
                               # need local audit artifacts (BiT dumps, NICCS
                               # CS-128/COMPASS-128 dumps) and skip otherwise
```

## Repository layout

```
code/
  bit_common.py            SM3 / pseudoXOF / challenge recovery / sig unpacking (public side)
  hint_channel_fast.py     Φ_R accumulation from a signature dump (public data)
  hint_model_check.py      tier-1 observation-model validation (MC table, field checks)
  hint_solve.py            analytic-operator solve
  hint_finish.py           Monte-Carlo operator + per-row scales + staged corrector
  hint_perrow.py           combined-dataset driver (real + sim3m → zero-error closure)
  run_pipeline.py          ★ standalone public-data pipeline (works from artifacts/ alone)
  cross_check.py           independent cross-validation battery
  forge_all.c / verify_all.c / pubmat.c / simulate.c   reference C drivers (unmodified signer/verifier)
../umodel/                 scheme-agnostic recovery framework (top level; v3): spec / channels /
                           pipeline / decoder / validator + selftest + demos
artifacts/                 public accumulated statistics, operator, public key, recovered equivalent key
paper/                     technical report, primer, slides, and the universal-model
                           report (LaTeX sources + PDFs)
```

**Not included** (secret or oversized): `ground.bin` (real key), `sk.bin`,
signature dumps (`dump.bin`), raw signature files. `run_pipeline.py` works
entirely from `artifacts/`; `--score` optionally accepts a `ground.bin` for
after-the-fact accuracy reporting only.

## Reproduce

```bash
# 1) from the shipped artifacts (public statistics only; minutes, CPU only)
python3 code/run_pipeline.py --artifacts artifacts --out key_hat.bin
#    expect: sha256 4191a44b4bf1e081175986302b0d30605bb86f827dbe02c80d90ef7f704d5a11

# 2) from scratch (needs the BiT reference implementation and ~4M signatures;
#    see paper §Reproducibility)
#    accumulate:  python3 code/hint_channel_fast.py <dump.bin> <nsig> <tag>
#    close:       python3 code/hint_perrow.py real sim3m
#    forge (original C signer+verifier): forge_all.exe pk.bin key_hat.bin 20
```

Headline numbers: label accuracy 96.19%; N=10⁶ oracle-arm 99.87% direct
(single fix closes); attack arm 99.09% (7 iid boundary crossings);
**N=4×10⁶ direct zero errors → s0 768/768 exact**; forgery 20/20 accepted,
all 4 negative controls rejected; eight-way cross-validation passed
(incl. 3,000,000 signatures re-verified by the original C verifier, zero
failures).

## Correction log (transparency)

* An earlier version of our notes claimed the z1-regression route "saturates
  at 65–69% accuracy and never converges". **That claim was retracted**: the
  saturation was an artifact of our earlier statistic
  (ψ(z)=sign(z)/(1024−|z|)) combined with a sign-threshold decision rule.
  The plain weighted least-squares regression on the raw responses converges
  with RMSE ∝ N^(−1/2) — verified on our own data (RMSE 0.184 at 2×10⁵
  signatures, closed after public-key correction). The cross-dataset
  deterministic structure we measured (correlation 0.9907 at zero positions)
  is real but does not impede a properly normalized regression.
* The hint channel, its operator law, per-row scale fit, and the
  full-solution corrector are the contributions of this work. A second,
  independent recovery route strengthens the result: both routes recover
  the same equivalent key byte-for-byte.
* v3 revisions (external review): fixed a sign typo in the estimator
  equation (implementation always used the correct sign); corrected the
  label-scale treatment (s scales all deterministic terms); replaced the
  "exact tent kernel" wording — the analytic tent kernel is an iid-position
  approximation (diagonal exceeds the exact N·τ by (τ−1)/d = 11.3%;
  Frobenius error vs the distinct-position Monte-Carlo operator: 10.9%);
  stated acceptance-conditioning terms as unmodeled and empirically
  bounded; retracted the σ(N)-floor formula and its self-contradictory
  threshold in favor of the expected-crossing-count criterion; fixed the
  cell-interval description and q = 39·682 + 283; clarified that the
  equivalent key is verification-equivalent (not
  output-distribution-identical) and that the overlapping-dataset check is
  a consistency check rather than independent-data replication.

## Ethics

Owner-authorized audit of the submitter's own reference implementation,
prior to submission review. Attack inputs are public keys and public
signatures only; secret-key material was used solely for after-the-fact
scoring. The defect is design-level (joint vs marginal independence); see
the paper §Mitigations.

## References

* L. Ducas, A. Durmus, T. Lepoint, V. Lyubashevsky, *Lattice signatures and
  bimodal Gaussians*, CRYPTO 2013 (ePrint 2013/383).
* V. Lyubashevsky, *Fiat–Shamir with aborts*, ASIACRYPT 2009.
* T. Espitau, P.-A. Fouque, B. Gérard, M. Tibouchi, *Side-channel attacks on
  BLISS*, CCS 2017 (ePrint 2017/505) — requires runtime leakage; our attack
  does not.
* BiT submission, NGCC Round 1 (sign-02), NICCS 2026.
