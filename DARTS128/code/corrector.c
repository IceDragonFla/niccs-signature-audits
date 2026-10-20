/* corrector.c — combined weighted-LS corrector for the DARTS128 (1-2b) attack.
 *
 * The labeled z1-LS alone rounds to a ternary candidate with a heavy-tailed
 * number of wrong coefficients (~34 at N=5e5, ~8 at N=1e6, 0 at N=2e6). This
 * corrector adds the 512 KEY IDENTITY equations
 *     (A0*s0 + A1*s1)[i] = qhat*beta[i] - e_obs[i]   (mod q)
 * as extra observations, where e_obs = round(be/ae) is the hint-channel
 * estimate of e (per-coefficient error ~2-4% at N=5e5). The identity
 * equations' noise (~0.15 in q-units) is ~1000x smaller than the z1 equations'
 * noise (sigma_y ~ 154), so with weight w = (sig_y/sig_eta)^2 they pin the
 * row-space of A (512 of the 1024 unknown directions) essentially exactly;
 * the kernel directions keep the z1 precision. The combined per-coordinate
 * rounding noise drops by ~sqrt(2), and a staged (single->pair->triple) flip
 * scan over the most ambiguous coordinates cleans the remaining handful of
 * errors. Success criterion: the identity residual r = center(qhat*beta -
 * A*s) is ENTIRELY ternary (count_bad == 0) — no false positives.
 *
 * usage: corrector <pk> <raw_estimates.bin> <out_recovered.bin> [topk]
 *   raw_estimates.bin (attack.py v2, all float64):
 *     [Grow x512][g0 x512][g1 x512][a0][a1][ae][sig0][sighe][xz0 x512][xz1 x512][xe x512]
 *   out: [s0 int8 x512][s1 int8 x512][e int8 x512]
 */
#include "sign.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "polymat.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int comp, m, delta; } flip_t;

static long center_q(long v) {
    v %= Q;
    if (v < 0) v += Q;
    if (v > Q / 2) v -= Q;
    return v;
}

static long col[2][N][N];       /* col[c][m] = A_c * x^m, reference-path semantics */

static long count_bad(const long *r) {
    long bad = 0;
    for (int i = 0; i < N; ++i)
        if (r[i] > 1 || r[i] < -1) bad++;
    return bad;
}

/* dense 2N x 2N solver: Gaussian elimination with partial pivoting */
static void solve2N(double *H, double *rhs, double *out) {
    int n = 2 * N;
    for (int k = 0; k < n; ++k) {
        int piv = k;
        double best = fabs(H[k * n + k]);
        for (int i = k + 1; i < n; ++i)
            if (fabs(H[i * n + k]) > best) { best = fabs(H[i * n + k]); piv = i; }
        if (piv != k)
            for (int j = k; j < n; ++j) {
                double t = H[k * n + j]; H[k * n + j] = H[piv * n + j]; H[piv * n + j] = t;
            }
        double t = rhs[k]; rhs[k] = rhs[piv]; rhs[piv] = t;
        for (int i = k + 1; i < n; ++i) {
            double f = H[i * n + k] / H[k * n + k];
            if (f != 0) {
                for (int j = k + 1; j < n; ++j) H[i * n + j] -= f * H[k * n + j];
                rhs[i] -= f * rhs[k];
            }
            H[i * n + k] = 0;
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double v = rhs[i];
        for (int j = i + 1; j < n; ++j) v -= H[i * n + j] * out[j];
        out[i] = v / H[i * n + i];
    }
}

int main(int argc, char **argv) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    polyvecl A[K];
    polyveck A0v;
    uint8_t seedA1[SEEDBYTES];
    double Grow[N];
    double g0[N], g1[N];
    double tail[5 + 3 * N];
    long s[2][N], r[N];
    int topk = 96;

    if (argc < 4) {
        fprintf(stderr, "usage: %s <pk> <raw_estimates> <out> [topk]\n", argv[0]);
        return 1;
    }
    if (argc >= 5) topk = atoi(argv[4]);

    { FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
      if (fread(pk, 1, sizeof(pk), f) != sizeof(pk)) { fprintf(stderr, "short pk\n"); return 1; }
      fclose(f); }
    { FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 1; }
      if (fread(Grow, 8, N, f) != N) { fprintf(stderr, "short raw (Grow)\n"); return 1; }
      if (fread(g0, 8, N, f) != N) { fprintf(stderr, "short raw (g0)\n"); return 1; }
      if (fread(g1, 8, N, f) != N) { fprintf(stderr, "short raw (g1)\n"); return 1; }
      if (fread(tail, 8, 5 + 3 * N, f) != 5 + 3 * N) { fprintf(stderr, "short raw (tail)\n"); return 1; }
      fclose(f); }
    const double a0 = tail[0], a1 = tail[1], ae = tail[2];
    const double *xz0 = tail + 5, *xz1 = tail + 5 + N, *xe = tail + 5 + 2 * N;
    printf("corrector: a0=%.4f a1=%.4f ae=%.4f topk=%d\n", a0, a1, ae, topk);

    unpack_pk(&A0v, seedA1, pk);
    polymatkl_expand(A, seedA1);
    for (unsigned k = 0; k < 1; ++k) A[k].vec[0] = A0v.vec[k];

    /* precompute all columns A_c * x^m through the reference product path */
    for (int c = 0; c < 2; ++c)
        for (int m = 0; m < N; ++m) {
            polyvecl sh;
            polyveck res;
            memset(&sh, 0, sizeof(sh));
            sh.vec[c].coeffs[m] = 1;
            poly_ntt(&sh.vec[c]);
            polymatkl_pointwise_montgomery(&res, A, &sh);
            polyveck_invntt_tomont(&res);
            polyveck_freeze(&res);
            for (int i = 0; i < N; ++i) col[c][m][i] = res.vec[0].coeffs[i];
        }

    /* e_obs from the hint route */
    long e_obs[N];
    for (int i = 0; i < N; ++i) {
        long v = (long)llround(xe[i] / ae);
        if (v > 1) v = 1; if (v < -1) v = -1;
        e_obs[i] = v;
    }

    /* combined weighted-LS system (single round; the hint outliers act as
     * observation noise and average out through the LS fit):
     * [Gz1 + w*AtA] s = [g + w*At(t)],  t = qhat*beta - e_obs
     * unknown vector = (s0[0..511], s1[0..511]) */
    const double sig_y = 154.2, sig_eta = 0.30;
    const double w = (sig_y * sig_y) / (sig_eta * sig_eta);
    static double H[2 * N][2 * N];
    static double rhs[2 * N];
    printf("building combined system (w=%.3g)...\n", w);

    static double At_t[2][N];
    static long tvec[N];
    for (int i = 0; i < N; ++i) tvec[i] = center_q((i == 0 ? (long)HALF_Q : 0) - e_obs[i]);
    for (int c = 0; c < 2; ++c)
        for (int m = 0; m < N; ++m) {
            double acc = 0;
            const long *cc = col[c][m];
            for (int i = 0; i < N; ++i) acc += (double)cc[i] * tvec[i];
            At_t[c][m] = acc;
        }

    for (int a = 0; a < 2; ++a)
        for (int m = 0; m < N; ++m) {
            const int row = a * N + m;
            for (int j = 0; j < N; ++j) {
                const double gz = Grow[(m - j + N) % N];
                H[row][j] = (a == 0) ? gz : 0.0;
                H[row][N + j] = (a == 1) ? gz : 0.0;
            }
            rhs[row] = (a == 0 ? g0[m] : g1[m]);
            for (int c2 = 0; c2 < 2; ++c2)
                for (int j = 0; j < N; ++j) {
                    const long *ci = col[a][m], *cj = col[c2][j];
                    long acc = 0;
                    for (int i = 0; i < N; ++i) acc += ci[i] * cj[i];
                    H[row][c2 * N + j] += w * (double)acc;
                }
            rhs[row] += w * At_t[a][m];
        }

    /* solve */
    static double sol[2 * N];
    long sc[2][N];
    solve2N(&H[0][0], rhs, sol);
    for (int c = 0; c < 2; ++c)
        for (int m = 0; m < N; ++m) {
            long v = (long)llround(sol[c * N + m]);
            if (v > 1) v = 1; if (v < -1) v = -1;
            sc[c][m] = v;
        }

    /* diagnostic: elimination residual + objective at truth vs at solution */
    {
        const char *tp = getenv("DARTS_TRUTH");
        double resid_max = 0, resid_sum = 0;
        for (int i = 0; i < 2 * N; ++i) {
            double v = rhs[i];
            for (int j = 0; j < 2 * N; ++j) v -= H[i][j] * sol[j];
            if (fabs(v) > resid_max) resid_max = fabs(v);
            resid_sum += v * v;
        }
        printf("elimination residual: max=%.3g l2=%.3g\n", resid_max, sqrt(resid_sum));
        if (tp) {
            static int8_t truth[3 * N];
            FILE *f = fopen(tp, "rb");
            if (f && fread(truth, 1, sizeof(truth), f) == sizeof(truth)) {
                fclose(f);
                double Jt = 0, Js = 0;
                for (int i = 0; i < 2 * N; ++i) {
                    double dt = 0, ds = 0;
                    for (int j = 0; j < 2 * N; ++j) {
                        double tv = (j < N ? (double)truth[j] : (double)truth[N + j - N]);
                        dt += H[i][j] * tv;
                        ds += H[i][j] * sol[j];
                    }
                    dt -= rhs[i]; ds -= rhs[i];
                    Jt += dt * dt; Js += ds * ds;
                }
                printf("objective: J(true)=%.4g  J(sol)=%.4g  %s\n",
                       Jt, Js, Jt < Js ? "TRUTH BETTER -> solver numerics failed"
                                       : "SOL BETTER -> model/weight issue");
            } else fclose(f);
        }
    }

    /* collapse check */
    {
        long Pacc[N];
        for (int i = 0; i < N; ++i) Pacc[i] = 0;
        for (int c = 0; c < 2; ++c)
            for (int m = 0; m < N; ++m) {
                if (sc[c][m] == 0) continue;
                const long sv = sc[c][m];
                const long *cc = col[c][m];
                for (int i = 0; i < N; ++i) Pacc[i] += sv * cc[i];
            }
        for (int i = 0; i < N; ++i) r[i] = center_q((i == 0 ? (long)HALF_Q : 0) - Pacc[i]);
    }
    long bad0 = count_bad(r);
    printf("combined-LS count_bad = %ld / %d\n", bad0, N);

    /* optional truth debugging (placed before the scan; also on the failed path) */
    {
        const char *tp = getenv("DARTS_TRUTH");
        if (tp) {
            static int8_t truth[3 * N];
            FILE *f = fopen(tp, "rb");
            if (f && fread(truth, 1, sizeof(truth), f) == sizeof(truth)) {
                fclose(f);
                long errs = 0, eobs_err = 0;
                double rms = 0;
                for (int c = 0; c < 2; ++c)
                    for (int m = 0; m < N; ++m) {
                        if (sc[c][m] != truth[c * N + m]) errs++;
                        double dv = sol[c * N + m] - truth[c * N + m];
                        rms += dv * dv;
                    }
                for (int i = 0; i < N; ++i)
                    if (e_obs[i] != truth[2 * N + i]) eobs_err++;
                printf("truth-debug: combined rms=%.4f  s wrong=%ld/1024  e_obs wrong=%ld/512\n",
                       sqrt(rms / (2 * N)), errs, eobs_err);
            } else fclose(f);
        }
    }

    if (bad0 != 0) {
        /* staged flip scan ranked by the continuous solution's ambiguity */
        printf("staged flip scan...\n");
        static flip_t cand[2 * 512]; int nflip = 0;
        int idx[2 * N]; double dv[2 * N]; int nc = 0;
        for (int c = 0; c < 2; ++c)
            for (int m = 0; m < N; ++m) {
                double d = fabs(sol[c * N + m] - (double)sc[c][m]);
                idx[nc] = c * N + m; dv[nc] = d; nc++;
            }
        for (int i = 0; i < topk && i < nc; ++i) {
            int best = i;
            for (int j = i + 1; j < nc; ++j) if (dv[j] > dv[best]) best = j;
            if (best != i) {
                int ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
                double td = dv[i]; dv[i] = dv[best]; dv[best] = td;
            }
        }
        int ntop = (topk < nc) ? topk : nc;
        for (int i = 0; i < ntop; ++i) {
            int c = idx[i] / N, m = idx[i] % N;
            for (int v = -1; v <= 1; ++v) {
                int d = v - (int)sc[c][m];
                if (d != 0) { cand[nflip].comp = c; cand[nflip].m = m; cand[nflip].delta = d; nflip++; }
            }
        }
        long rb[N]; int found = 0;
        for (int st = 1; st <= 3 && !found; ++st)
            for (int i = 0; i < nflip && !found; ++i)
                for (int j = (st >= 2 ? i : nflip - 1); j < (st >= 2 ? nflip : i + 1) && !found; ++j)
                    for (int k2 = (st >= 3 ? j + 1 : j); k2 < (st >= 3 ? nflip : j + 1) && !found; ++k2) {
                        flip_t fl[3];
                        int nfl = 0;
                        fl[nfl++] = cand[i];
                        if (st >= 2) fl[nfl++] = cand[j];
                        if (st >= 3) fl[nfl++] = cand[k2];
                        memcpy(rb, r, sizeof(r));
                        for (int f2 = 0; f2 < nfl; ++f2) {
                            const long *cc = col[fl[f2].comp][fl[f2].m];
                            for (int ii = 0; ii < N; ++ii)
                                rb[ii] = center_q(rb[ii] - (long)fl[f2].delta * cc[ii]);
                        }
                        if (count_bad(rb) == 0) {
                            memcpy(r, rb, sizeof(r)); found = 1;
                            for (int f2 = 0; f2 < nfl; ++f2) sc[fl[f2].comp][fl[f2].m] += fl[f2].delta;
                        }
                    }
        if (!found) {
            printf("corrector FAILED: no collapse (increase N)\n");
            return 2;
        }
        printf("collapse found via staged flip scan\n");
    }
    for (int c = 0; c < 2; ++c) for (int m = 0; m < N; ++m) s[c][m] = sc[c][m];

    /* optional truth debugging */
    {
        const char *tp = getenv("DARTS_TRUTH");
        if (tp) {
            static int8_t truth[3 * N];
            FILE *f = fopen(tp, "rb");
            if (f && fread(truth, 1, sizeof(truth), f) == sizeof(truth)) {
                fclose(f);
                long errs = 0, eobs_err = 0;
                double rms = 0;
                for (int c = 0; c < 2; ++c)
                    for (int m = 0; m < N; ++m) {
                        if (s[c][m] != truth[c * N + m]) errs++;
                        double dv = sol[c * N + m] - truth[c * N + m];
                        rms += dv * dv;
                    }
                for (int i = 0; i < N; ++i)
                    if (e_obs[i] != truth[2 * N + i]) eobs_err++;
                printf("truth-debug: combined rms=%.4f  s wrong=%ld/1024  e_obs wrong=%ld/512\n",
                       sqrt(rms / (2 * N)), errs, eobs_err);
            } else fclose(f);
        }
    }

    {
        static int8_t out[3 * N];
        for (int i = 0; i < N; ++i) out[i] = (int8_t)s[0][i];
        for (int i = 0; i < N; ++i) out[N + i] = (int8_t)s[1][i];
        for (int i = 0; i < N; ++i) out[2 * N + i] = (int8_t)r[i];
        FILE *f = fopen(argv[3], "wb");
        if (!f) { perror(argv[3]); return 1; }
        fwrite(out, 1, sizeof(out), f);
        fclose(f);
    }
    printf("wrote %s (s0|s1|e int8, identity-consistent)\n", argv[3]);
    return 0;
}
