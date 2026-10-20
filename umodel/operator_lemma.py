# operator_lemma.py — Closed-form position-exclusive operator: derivation,
# numerical verification, and scope statement.
#
# LEMMA (supervisor-derived, numerically verified here):
#   Let C be the negacyclic convolution matrix of a weight-τ challenge
#   with τ distinct positions drawn uniformly from {0,...,d-1}. Then
#
#     E[C^T C] = τ·I + ρ·(K_mv - d·I)
#
#   where ρ = τ(τ-1) / (d(d-1)) and K_mv[m,v] = d - 2|m-v|.
#
#   The field operator Fop used in our pipeline (which excludes same-position
#   terms) therefore has the closed-form model expectation:
#
#     Fop = E[C^T C - τ·I] = ρ·(K_mv - d·I)
#
# PROOF:
#   Write C = Σ_{t=1}^{τ} S_{p_t} where S_p is the signed negacyclic shift
#   matrix for position p (S_p[m, (m+p)%d] = ±1). Then
#     C^T C = Σ_{t,t'} S_{p_t}^T S_{p_{t'}}.
#   For t = t' (same position): S_p^T S_p = I for any p. There are τ such
#   terms, contributing τ·I — this is the signal diagonal.
#   For t ≠ t' (different positions): E[S_{p_t}^T S_{p_{t'}}] over distinct
#   p_t, p_{t'} drawn uniformly. The entry [m, v] is nonzero when the
#   alignment couples coordinate m (via p_t) to coordinate v (via p_{t'}).
#   The probability that a specific ordered pair (p_t, p_{t'}) takes values
#   (p, p') with p ≠ p' is 1/(d(d-1)) for each pair. The count of such
#   pairs is τ(τ-1). The sum Σ_{p≠p'} S_p^T S_{p'}[m,v] counts the number
#   of (p, p') pairs that connect m to v — this count equals K_mv[m,v] - d
#   (total connections minus diagonal-only connections). Therefore
#     E[Σ_{t≠t'} S_{p_t}^T S_{p_{t'}}] = ρ·(K_mv - d·I).
#
# NUMERICAL VERIFICATION (this file):
#   Direct MC of E[C^T C] vs closed form at (d, τ) ∈ {(32,6), (64,10),
#   (128,20), (256,30)}: relative error ≤ 0.008 (8000 draws, consistent
#   with MC noise). Diagonal exactly τ in all cases.
#   Our 150k-draw Fop vs ρ(K-dI): relative error 0.0024, diagonal exactly 0.
#
# EXTENSION (v5): SIGNED challenges.
#   When the challenge coefficients are independent uniform +/-1 (CS-128 family), the cross-position terms carry the factor
#   E[val_t * val_t'] = 0 for t != t', so
#
#       E[C^T C] = tau * I          (signed, independent +/-1 values)
#
#   i.e. the tent-kernel field term DISAPPEARS: the first-moment system is
#   purely diagonal (no deblurring needed; the field operator Fop = 0).
#   Verified numerically below at (64,10), (128,20), (256,30): deviation
#   from tau*I at MC-noise level (~1.6% at 4000 draws), while the unsigned
#   closed form is off by 54-72%. Consequences:
#     - schemes with a published quantized hint AND signed challenges have
#       a SIMPLER moment channel (the v3 refusal "derived only for the
#       unsigned family" is upgraded: the signed case needs no operator
#       beyond tau*I);
#     - the BiT-style deblurring step is specific to the unsigned family.
#
# SCOPE — what this lemma does and does NOT cover:
#   COVERED:
#     - Uniform distinct positions (SampleInBall marginal, before acceptance)
#     - Canonical challenge with all +1 coefficients
#     - Negacyclic convolution in Z_q[x]/(x^d+1)
#     - The expectation of the alignment combinatorics ONLY
#
#   NOT COVERED (remain as unmodeled layers):
#     - Label weights: real labels have 96.2% accuracy, not 100%; the
#       effective signal scale s_eff absorbs this but is fitted, not derived
#     - Acceptance conditioning: z2 rejection, |h|_∞ ≤ 3 truncation, w1
#       sparse check all condition the published distribution on secret
#       geometry — these change the effective operator by unmodeled terms
#     - Quantization slope: the 7-level quantizer introduces κ ≈ 1.011
#       (measured, not derived from first principles)
#     - Row-0 b-term: the (b=0) indicator adds a known row-0 correction
#       (handled separately in the pipeline)
#
#   Therefore: Fop = ρ(K-dI) is the model expectation under the stated
#   sampling model. It eliminates the MC layer for the alignment
#   combinatorics. It does NOT give the full signal operator under the
#   real acceptance distribution, and cannot be used alone to predict
#   sample complexity.
import numpy as np

def closed_form_operator(d, tau):
    """E[C^T C] = tau*I + rho*(K_mv - d*I); field part Fop = rho*(K_mv - d*I)."""
    rho = tau * (tau - 1) / (d * (d - 1))
    K_mv = d - 2 * np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    Fop_closed = rho * (K_mv - d * np.eye(d))
    return Fop_closed, rho

def verify_signed(d, tau, n_mc=4000, seed=43):
    """E[C^T C] for SIGNED challenges vs tau*I (and vs the unsigned closed
    form, which must FAIL)."""
    rng = np.random.default_rng(seed)
    ar = np.arange(d)
    ECC = np.zeros((d, d))
    for _ in range(n_mc):
        pos = rng.choice(d, tau, replace=False)
        vals = rng.integers(0, 2, tau) * 2 - 1
        C = np.zeros((d, d))
        for t in range(tau):
            p = pos[t]
            j = (ar + p) % d
            flip = np.where(ar + p >= d, -1.0, 1.0)
            C[ar, j] = flip * vals[t]
        ECC += C.T @ C
    ECC /= n_mc
    rel_diag = np.linalg.norm(ECC - tau * np.eye(d)) / (tau * np.sqrt(d))
    Fop_closed, _ = closed_form_operator(d, tau)
    closed_unsigned = tau * np.eye(d) + Fop_closed
    rel_uns = np.linalg.norm(ECC - closed_unsigned) / np.linalg.norm(closed_unsigned)
    return rel_diag, rel_uns


def verify(d, tau, n_mc=8000, seed=42):
    rng = np.random.default_rng(seed)
    ar = np.arange(d)
    ECC = np.zeros((d, d))
    for _ in range(n_mc):
        pos = rng.choice(d, tau, replace=False)
        C = np.zeros((d, d))
        for t in range(tau):
            p = pos[t]
            j = (ar + p) % d
            flip = np.where(ar + p >= d, -1.0, 1.0)
            C[ar, j] = flip
        ECC += C.T @ C
    ECC /= n_mc
    Fop_closed, rho = closed_form_operator(d, tau)
    E_closed = tau * np.eye(d) + Fop_closed
    rel = np.linalg.norm(ECC - E_closed) / np.linalg.norm(E_closed)
    return rel, ECC.diagonal().mean(), rho

if __name__ == "__main__":
    print("Closed-form position-exclusive operator verification")
    print("=" * 60)
    for d, tau in [(32, 6), (64, 10), (128, 20), (256, 30)]:
        rel, diag, rho = verify(d, tau)
        print(f"  d={d:3d} tau={tau:2d}: rel_err={rel:.6f}  "
              f"diag={diag:.3f} (expect {tau})  rho={rho:.6f}")

    print("Signed-challenge extension: E[C^T C] = tau*I")
    print("-" * 60)
    for d, tau in [(64, 10), (128, 20), (256, 30)]:
        rd, ru = verify_signed(d, tau)
        print(f"  d={d:3d} tau={tau:2d}: |ECC-tauI|/|tauI| = {rd:.4f} "
              f"(MC noise); vs unsigned closed form = {ru:.4f} (rejected)")

    # Compare against our production Fop
    Fop = np.load("tmp/attack2/Fop_150000.npy")
    Fop_cf, rho = closed_form_operator(256, 30)
    rel = np.linalg.norm(Fop - Fop_cf) / np.linalg.norm(Fop_cf)
    print(f"\n  Production Fop (150k draws) vs closed form: rel_err = {rel:.6f}")
    print(f"  Fop diagonal: {Fop.diagonal().mean():.6f} (closed form: 0)")
    print(f"  Full signal diag: {(30 + Fop.diagonal()).mean():.3f} (should be 30)")
    print(f"  Tent approximation diag would be: {30 * (1 + 29/256):.3f} (11.33% excess)")
