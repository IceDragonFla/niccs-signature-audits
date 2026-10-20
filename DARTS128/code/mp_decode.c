/* mp_decode.c — matching-pursuit finisher for the DARTS128 (1-2b) attack.
 *
 * The identity residual r = center_q(t - A*s)  (t = qhat*beta - e_obs)
 * correlates with the near-orthogonal columns of A:
 *     rho_j = <col_j, r>/||col_j||^2  ~=  -delta_j  +  noise(0.28),
 * where delta_j = s_j - s_true_j is the candidate's error and the noise is
 * cross-leakage (~4.4% per other error) plus mod-q wrap leakage (~0.04).
 * Greedy pursuit: flip the coordinate with the largest |rho_j| (ternary-
 * constrained), update r incrementally, check COLLAPSE (all |r_i| <= 1)
 * after every flip. Corrects tens of errors without any exhaustive search.
 *
 * usage: mp_decode <pk> <cols.bin> <raw_estimates.bin> <out> [maxflips]
 *   cols.bin: dump_cols output (1024 x 512 int64, reference-path columns)
 *   raw_estimates.bin: attack.py v2 (all float64)
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

static long center_q(long v) {
    v %= Q;
    if (v < 0) v += Q;
    if (v > Q / 2) v -= Q;
    return v;
}

static long count_bad(const long *r) {
    long bad = 0;
    for (int i = 0; i < N; ++i)
        if (r[i] > 1 || r[i] < -1) bad++;
    return bad;
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
    int maxflips = 600;

    if (argc < 5) {
        fprintf(stderr, "usage: %s <pk> <cols.bin> <raw_estimates> <out> [maxflips]\n", argv[0]);
        return 1;
    }
    if (argc >= 6) maxflips = atoi(argv[5]);

    { FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
      if (fread(pk, 1, sizeof(pk), f) != sizeof(pk)) { fprintf(stderr, "short pk\n"); return 1; }
      fclose(f); }
    static long col[2 * N][N];
    { FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 1; }
      if (fread(col, 8, 2 * N * N, f) != 2 * N * N) { fprintf(stderr, "short cols\n"); return 1; }
      fclose(f); }
    { FILE *f = fopen(argv[3], "rb"); if (!f) { perror(argv[3]); return 1; }
      if (fread(Grow, 8, N, f) != N) { fprintf(stderr, "short raw\n"); return 1; }
      if (fread(g0, 8, N, f) != N) { fprintf(stderr, "short raw\n"); return 1; }
      if (fread(g1, 8, N, f) != N) { fprintf(stderr, "short raw\n"); return 1; }
      if (fread(tail, 8, 5 + 3 * N, f) != 5 + 3 * N) { fprintf(stderr, "short raw\n"); return 1; }
      fclose(f); }
    (void)Grow; (void)g0; (void)g1;
    const double a0 = tail[0], a1 = tail[1], ae = tail[2];
    const double *xz0 = tail + 5, *xz1 = tail + 5 + N, *xe = tail + 5 + 2 * N;
    printf("mp_decode: a0=%.4f a1=%.4f ae=%.4f maxflips=%d\n", a0, a1, ae, maxflips);

    unpack_pk(&A0v, seedA1, pk);
    polymatkl_expand(A, seedA1);
    for (unsigned k = 0; k < 1; ++k) A[k].vec[0] = A0v.vec[k];

    /* column norms */
    static double colnorm[2 * N];
    for (int j = 0; j < 2 * N; ++j) {
        double sq = 0;
        for (int i = 0; i < N; ++i) sq += (double)col[j][i] * col[j][i];
        colnorm[j] = sq;
    }

    /* initial ternary candidate from the descaled z1 estimate */
    for (int c = 0; c < 2; ++c)
        for (int m = 0; m < N; ++m) {
            double x = (c == 0 ? xz0[m] / a0 : xz1[m] / a1);
            long v = (long)llround(x);
            if (v > 1) v = 1; if (v < -1) v = -1;
            s[c][m] = v;
        }

    /* e_obs and t */
    long e_obs[N], t[N];
    for (int i = 0; i < N; ++i) {
        long v = (long)llround(xe[i] / ae);
        if (v > 1) v = 1; if (v < -1) v = -1;
        e_obs[i] = v;
    }
    for (int i = 0; i < N; ++i) t[i] = center_q((i == 0 ? (long)HALF_Q : 0) - e_obs[i]);

    /* residual r = center_q(t - A s) */
    for (int i = 0; i < N; ++i) {
        long acc = 0;
        for (int c = 0; c < 2; ++c)
            for (int m = 0; m < N; ++m)
                if (s[c][m]) {
                    long v = col[c * N + m][i];
                    acc += (long)s[c][m] * v;
                }
        r[i] = center_q(t[i] - acc);
    }
    printf("initial count_bad = %ld / %d\n", count_bad(r), N);

    /* greedy pursuit */
    static int hist[2 * 1024]; int nh = 0;
    int collapsed = (count_bad(r) == 0);
    int flips = 0;
    while (!collapsed && flips < maxflips) {
        /* correlations rho_j = <col_j, r>/norm_j */
        int best_j = -1; double best_rho = 0.45; int best_d = 0;
        for (int j = 0; j < 2 * N; ++j) {
            double acc = 0;
            const long *cj = col[j];
            for (int i = 0; i < N; ++i) acc += (double)cj[i] * r[i];
            double rho = acc / colnorm[j];
            int c = j / N, m = j % N;
            long d = -(long)llround(rho);
            long nv = s[c][m] + d;
            if (nv < -1 || nv > 1 || d == 0) continue;
            /* tabu: avoid immediate undo of the last flip */
            if (nh >= 2 && hist[nh - 1] == j && hist[nh - 2] == j) continue;
            if (fabs(rho) > best_rho) { best_rho = fabs(rho); best_j = j; best_d = (int)d; }
        }
        if (best_j < 0) break;   /* no confident move left */
        int c = best_j / N, m = best_j % N;
        s[c][m] += best_d;
        const long *cj = col[best_j];
        for (int i = 0; i < N; ++i) r[i] = center_q(r[i] - (long)best_d * cj[i]);
        hist[nh++] = best_j;
        flips++;
        if (count_bad(r) == 0) collapsed = 1;
        if (flips % 25 == 0)
            printf("  flip %d: last rho=%.3f count_bad=%ld\n", flips, best_rho, count_bad(r));
    }

    long bad = count_bad(r);
    printf("pursuit: flips=%d collapsed=%s (count_bad=%ld)\n",
           flips, collapsed ? "YES" : "NO", bad);
    if (!collapsed) { printf("mp_decode FAILED: no collapse\n"); return 2; }

    {
        static int8_t out[3 * N];
        for (int i = 0; i < N; ++i) out[i] = (int8_t)s[0][i];
        for (int i = 0; i < N; ++i) out[N + i] = (int8_t)s[1][i];
        for (int i = 0; i < N; ++i) out[2 * N + i] = (int8_t)r[i];  /* identity residual */
        FILE *f = fopen(argv[4], "wb");
        if (!f) { perror(argv[4]); return 1; }
        fwrite(out, 1, sizeof(out), f);
        fclose(f);
    }
    printf("wrote %s\n", argv[4]);
    return 0;
}
