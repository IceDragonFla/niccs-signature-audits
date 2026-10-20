/* scan3.c — unrestricted depth-3 Gray/pair+coverage-AND finisher.
 *
 * Candidate s (from the labeled LS) has e_w wrong coordinates (unknown where).
 * Residual r = center_q(t - A s), t = qhat*beta - e_obs. A flip set F = {(j,d)}
 * closes the attack iff center_q(r - sum_F d_j col_j) is entirely ternary
 * (|.|<=1 at every coordinate; the e_obs mismatches live in +-1).
 *
 * Depth 3 with ALL 1024 coordinates x 4 delta values, no ranking restriction
 * (the |sol-round| statistic carries no information about error locations).
 * Enumeration: for every ordered pair (i,di),(j,dj) compute the pair residual
 * and require the LAST flip (k,dk) to cover ALL remaining non-ternary
 * coordinates simultaneously (early-exit AND test). Collapse -> exact key.
 *
 * usage: scan3 <pk> <cols.bin> <raw_estimates.bin> <out> [maxflips]
 */
#include "sign.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "polymat.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static long center_q(long v) {
    v %= Q;
    if (v < 0) v += Q;
    if (v > Q / 2) v -= Q;
    return v;
}

/* collapse criterion: the residual at the true s equals e_true - e_obs, and
 * the hint-route clipping can push individual eta entries to +-2/+-3 — so
 * collapse requires |r_i| <= 3 (a wrong candidate gives r ~ uniform +-65408,
 * so P(false collapse) = (7/130817)^512 = 0). */
static long count_bad(const long *r) {
    long bad = 0;
    for (int i = 0; i < N; ++i)
        if (r[i] > 3 || r[i] < -3) bad++;
    return bad;
}

#define DK 4
static const int dvals[DK] = {-2, -1, 1, 2};

int main(int argc, char **argv) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    polyvecl A[K];
    polyveck A0v;
    uint8_t seedA1[SEEDBYTES];
    double tail[5 + 3 * N];
    long s[2][N], r[N];
    int maxflips = 3;

    if (argc < 5) {
        fprintf(stderr, "usage: %s <pk> <cols.bin> <raw_estimates> <out> [depth]\n", argv[0]);
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
      double Grow[N], g0[N], g1[N];
      if (fread(Grow, 8, N, f) != N) { fprintf(stderr, "short raw\n"); return 1; }
      if (fread(g0, 8, N, f) != N) { fprintf(stderr, "short raw\n"); return 1; }
      if (fread(g1, 8, N, f) != N) { fprintf(stderr, "short raw\n"); return 1; }
      if (fread(tail, 8, 5 + 3 * N, f) != 5 + 3 * N) { fprintf(stderr, "short raw\n"); return 1; }
      }
    const double a0 = tail[0], a1 = tail[1], ae = tail[2];
    const double *xz0 = tail + 5, *xz1 = tail + 5 + N, *xe = tail + 5 + 2 * N;

    unpack_pk(&A0v, seedA1, pk);
    polymatkl_expand(A, seedA1);
    for (unsigned k = 0; k < 1; ++k) A[k].vec[0] = A0v.vec[k];

    for (int c = 0; c < 2; ++c)
        for (int m = 0; m < N; ++m) {
            double x = (c == 0 ? xz0[m] / a0 : xz1[m] / a1);
            long v = (long)llround(x);
            if (v > 1) v = 1; if (v < -1) v = -1;
            s[c][m] = v;
        }

    long e_obs[N], t[N];
    for (int i = 0; i < N; ++i) {
        long v = (long)llround(xe[i] / ae);
        if (v > 1) v = 1; if (v < -1) v = -1;
        e_obs[i] = v;
    }
    for (int i = 0; i < N; ++i) t[i] = center_q((i == 0 ? (long)HALF_Q : 0) - e_obs[i]);

    for (int i = 0; i < N; ++i) {
        long acc = 0;
        for (int c = 0; c < 2; ++c)
            for (int m = 0; m < N; ++m)
                if (s[c][m]) acc += (long)s[c][m] * col[c * N + m][i];
        r[i] = center_q(t[i] - acc);
    }
    long bad0 = count_bad(r);
    printf("initial count_bad = %ld / %d\n", bad0, N);
    if (bad0 == 0) { printf("already consistent\n"); goto write_out; }

    {
        int found = 0;
        /* depth 1 */
        for (int j = 0; j < 2 * N && !found; ++j) {
            int c = j / N, m = j % N;
            for (int d = 0; d < DK && !found; ++d) {
                long dv = dvals[d], nv = s[c][m] + dv;
                if (nv < -1 || nv > 1) continue;
                long r2[N];
                for (int i = 0; i < N; ++i) r2[i] = center_q(r[i] - dv * col[j][i]);
                if (count_bad(r2) == 0) {
                    memcpy(r, r2, sizeof(r2));
                    s[c][m] = nv; found = 1;
                    printf("collapse: depth 1 at coord %d delta %ld\n", j, dv);
                }
            }
        }
        /* depth 2: pair covers all bads */
        for (int j1 = 0; j1 < 2 * N && !found; ++j1) {
            int c1 = j1 / N, m1 = j1 % N;
            for (int d1 = 0; d1 < DK && !found; ++d1) {
                long dv1 = dvals[d1], nv1 = s[c1][m1] + dv1;
                if (nv1 < -1 || nv1 > 1) continue;
                long r1[N];
                for (int i = 0; i < N; ++i) r1[i] = center_q(r[i] - dv1 * col[j1][i]);
                for (int j2 = 0; j2 < 2 * N && !found; ++j2) {
                    if (j2 == j1) continue;
                    int c2 = j2 / N, m2 = j2 % N;
                    for (int d2 = 0; d2 < DK && !found; ++d2) {
                        long dv2 = dvals[d2], nv2 = s[c2][m2] + dv2;
                        if (nv2 < -1 || nv2 > 1) continue;
                        long r2[N];
                        for (int i = 0; i < N; ++i) r2[i] = center_q(r1[i] - dv2 * col[j2][i]);
                        if (count_bad(r2) == 0) {
                            memcpy(r, r2, sizeof(r2));
                            s[c1][m1] = nv1; s[c2][m2] = nv2; found = 1;
                            printf("collapse: depth 2 at coords (%d,%ld) (%d,%ld)\n",
                                   j1, dv1, j2, dv2);
                        }
                    }
                }
            }
        }
        /* depth 3: pair residual + last-flip coverage-AND (early-exit) */
        if (!found && maxflips >= 3) {
            printf("depth 3 coverage-AND scan...\n");
            volatile int found_v = 0;
            long fs[3], fds[3];
            #pragma omp parallel for schedule(dynamic, 4)
            for (int j1 = 0; j1 < 2 * N; ++j1) {
                if (found_v) continue;
                int c1 = j1 / N, m1 = j1 % N;
                for (int d1 = 0; d1 < DK && !found_v; ++d1) {
                    long dv1 = dvals[d1], nv1 = s[c1][m1] + dv1;
                    if (nv1 < -1 || nv1 > 1) continue;
                    long r1[N];
                    for (int i = 0; i < N; ++i) r1[i] = center_q(r[i] - dv1 * col[j1][i]);
                    for (int j2 = 0; j2 < 2 * N && !found_v; ++j2) {
                        if (j2 == j1) continue;
                        int c2 = j2 / N, m2 = j2 % N;
                        for (int d2 = 0; d2 < DK && !found_v; ++d2) {
                            long dv2 = dvals[d2], nv2 = s[c2][m2] + dv2;
                            if (nv2 < -1 || nv2 > 1) continue;
                            /* pair residual + bad list */
                            long r2[N];
                            int nb = 0, blist[N];
                            for (int i = 0; i < N; ++i) {
                                r2[i] = center_q(r1[i] - dv2 * col[j2][i]);
                                if (r2[i] > 1 || r2[i] < -1) blist[nb++] = i;
                            }
                            if (nb == 0) { /* depth-2 found inside */
                                #pragma omp critical
                                {
                                    if (!found_v) { memcpy(r, r2, sizeof(r2)); found_v = 1;
                                        fs[0] = j1; fds[0] = dv1; fs[1] = j2; fds[1] = dv2; }
                                }
                                continue;
                            }
                            /* last flip (k, dk) must fix every bad coord */
                            for (int k = 0; k < 2 * N && !found_v; ++k) {
                                int c3 = k / N, m3 = k % N;
                                for (int d3 = 0; d3 < DK && !found_v; ++d3) {
                                    long dv3 = dvals[d3], nv3 = s[c3][m3] + dv3;
                                    if (nv3 < -1 || nv3 > 1) continue;
                                    const long *ck = col[k];
                                    int ok = 1;
                                    for (int b = 0; b < nb; ++b) {
                                        long v = center_q(r2[blist[b]] - dv3 * ck[blist[b]]);
                                        if (v > 1 || v < -1) { ok = 0; break; }
                                    }
                                    if (ok) {
                                        #pragma omp critical
                                        {
                                            if (!found_v) {
                                                memcpy(r, r2, sizeof(r2));
                                                for (int b = 0; b < nb; ++b)
                                                    r[blist[b]] = center_q(r2[blist[b]] - dv3 * col[k][blist[b]]);
                                                found_v = 1;
                                                fs[0] = j1; fds[0] = dv1;
                                                fs[1] = j2; fds[1] = dv2;
                                                fs[2] = k;  fds[2] = dv3;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            found = found_v;
            if (found) {
                for (int f = 0; f < 3; ++f) {
                    int c = (int)(fs[f]) / N, m = (int)(fs[f]) % N;
                    s[c][m] += (int)fds[f];
                }
                printf("collapse: depth 3\n");
            }
        }
        if (!found) {
            /* a non-collapsed residual does NOT imply s is wrong: e_obs
             * outliers (|eta| up to 3+) leave |r| in 2..3+ even when s is
             * exact. Always write the candidate; the downstream fixup
             * (e = qhat*beta - A s, non-ternary count) is the discriminator:
             * non-ternary==0 <=> s exact. */
            printf("scan3: no collapse within depth %d (candidate may still be "
                   "exact; fixup decides)\n", maxflips);
        }
    }

write_out:
    {
        static int8_t out[3 * N];
        for (int i = 0; i < N; ++i) out[i] = (int8_t)s[0][i];
        for (int i = 0; i < N; ++i) out[N + i] = (int8_t)s[1][i];
        for (int i = 0; i < N; ++i) out[2 * N + i] = (int8_t)r[i];
        FILE *f = fopen(argv[4], "wb");
        if (!f) { perror(argv[4]); return 1; }
        fwrite(out, 1, sizeof(out), f);
        fclose(f);
    }
    printf("wrote %s\n", argv[4]);
    return 0;
}
