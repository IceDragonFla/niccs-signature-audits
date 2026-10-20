/*
 * harness_corr.c -- stream NSIG shipped signatures, accumulate the spectral
 * leak Gram P[k][k'][m] = sum_q zhat_k(m) conj(zhat_k'(m)) for blocks 1..3
 * (the attacker-visible s-blocks of z1), plus per-block sum/sumsq.
 * Dumps: count, P (3x3x256 complex as re,im doubles), sums, sumsqs.
 * argv[1]=rand file, argv[2]=NSIG, argv[3]=out .bin
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "packing.h"
#include "rounding.h"
#include "symmetric.h"
#include "drng.h"
#include "SIG_AlgorithmInstance.h"

DRNG_ctx drng_algorithm;

#define PI 3.14159265358979323846

/* iterative radix-2 FFT on 256 complex points (in-place), sign=-1 forward */
static void fft256(double *re, double *im)
{
    static double cwr[128], cwi[128];
    static int init = 0;
    int i, j, len, half, k;
    if (!init) {
        for (i = 0; i < 128; ++i) {
            cwr[i] = cos(-2.0 * PI * i / 256.0);
            cwi[i] = sin(-2.0 * PI * i / 256.0);
        }
        init = 1;
    }
    for (i = 1, j = 0; i < 256; ++i) {
        int bit = 128;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (len = 2; len <= 256; len <<= 1) {
        half = len >> 1;
        int step = 256 / len;
        for (i = 0; i < 256; i += len)
            for (k = 0; k < half; ++k) {
                double wr = cwr[k * step], wi = cwi[k * step];
                double ur = re[i + k], ui = im[i + k];
                double vr = re[i + k + half] * wr - im[i + k + half] * wi;
                double vi = re[i + k + half] * wi + im[i + k + half] * wr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
            }
    }
}

int main(int argc, char **argv)
{
    uint8_t pk[4200], skraw[3000];
    unsigned long long pklen, sklen, siglen;
    long NSIG, q;
    int k, kp, rtn;
    unsigned i;
    FILE *f, *o;
    static uint8_t randbuf[1400000000];
    size_t rn, roff;
    double twist_re[256], twist_im[256];
    static double Pre[3][3][256], Pim[3][3][256];
    static double blk_sum[3], blk_sumsq[3];
    double zre[3][256], zim[3][256];
    double z1re[4][256], z1im[4][256];

    if (argc != 4 && argc != 5) { fprintf(stderr, "usage: %s rand NSIG out [roff0]\n", argv[0]); return 1; }

    NSIG = atol(argv[2]);
    long roff0 = (argc == 5) ? atol(argv[4]) : (55 + 4096);
    long nfail = 0;
    f = fopen(argv[1], "rb");
    if (!f) { perror("rand"); return 1; }
    rn = fread(randbuf, 1, sizeof(randbuf), f);
    fclose(f);

    for (i = 0; i < 256; ++i) {
        twist_re[i] = cos(PI * i / 256.0);
        twist_im[i] = sin(PI * i / 256.0);
    }

    if (getenv("BASEKEY")) {
        FILE* kf = fopen(getenv("BASEKEY"), "rb");
        size_t rd;
        if (!kf) { fprintf(stderr, "basekey open fail\n"); return 1; }
        rd = fread(pk, 1, CRYPTO_PUBLICKEYBYTES, kf);
        rd += fread(skraw, 1, CRYPTO_SECRETKEYBYTES, kf);
        fclose(kf);
        (void)rd; pklen = CRYPTO_PUBLICKEYBYTES; sklen = CRYPTO_SECRETKEYBYTES;
    } else {
    init_random_number(&drng_algorithm, randbuf, 55 * 8);
    rtn = sig_keygen(pk, &pklen, skraw, &sklen);
    if (rtn) { fprintf(stderr, "keygen %d\n", rtn); return 1; } }
    rtn = 0;
    if (rtn) { fprintf(stderr, "keygen %d\n", rtn); return 1; }
    /* dump pk+sk for ground truth */
    { char fn[512]; snprintf(fn, sizeof fn, "%s_key.bin", argv[3]);
      f = fopen(fn, "wb"); fwrite(pk, 1, (size_t)pklen, f);
      fwrite(skraw, 1, (size_t)sklen, f);
      { uint8_t seedA2[SEEDBYTES], masterK2[CHALLENGESEEDBYTES], tr2[CHALLENGESEEDBYTES];
        poly b2[EM], s2[ELL], ep2[EM], skf[KVEC], skt2[KVEC];
        unsigned kk, ii;
        if (unpack_sk(seedA2, b2, masterK2, tr2, s2, ep2, skraw) == 0) {
            for (kk = 0; kk < N; ++kk) skf[0].coeffs[kk] = 0;
            skf[0].coeffs[0] = 1;
            for (ii = 0; ii < ELL; ++ii) skf[1 + ii] = s2[ii];
            for (ii = 0; ii < EM; ++ii) skf[1 + ELL + ii] = ep2[ii];
            stretch_s(skt2, skf);
            fwrite(s2, sizeof(poly), ELL, f);
            fwrite(ep2, sizeof(poly), EM, f);
            fwrite(skt2, sizeof(poly), KVEC, f);
        } }
      fclose(f); }

    { const char* s0f = getenv("S0FILE");
      if (s0f) {
        FILE* g = fopen(s0f, "rb");
        int32_t buf[3*N];
        poly s2[ELL], ep2[EM];
        uint8_t seedA2[SEEDBYTES], masterK2[CHALLENGESEEDBYTES], tr2[CHALLENGESEEDBYTES];
        poly b2[EM];
        long nfail = 0;
        if (!g || fread(buf, 4, 3*N, g) != 3*N) { fprintf(stderr, "s0file fail\n"); return 1; }
        fclose(g);
        if (unpack_sk(seedA2, b2, masterK2, tr2, s2, ep2, skraw) != 0) { fprintf(stderr, "unpack fail\n"); return 1; }
        { unsigned k2, i2, w = 0;
          for (k2 = 0; k2 < ELL; ++k2) for (i2 = 0; i2 < N; ++i2) s2[k2].coeffs[i2] = buf[w++];
          for (k2 = 0; k2 < EM; ++k2) for (i2 = 0; i2 < N; ++i2) ep2[k2].coeffs[i2] = buf[w++]; }
        pack_sk(skraw, seedA2, b2, masterK2, tr2, s2, ep2);
        fprintf(stderr, "S0 injected from %s\n", s0f);
      } }
    roff = roff0;
    for (q = 0; q < NSIG; ++q) {
        uint8_t m[32], sig[CRYPTO_BYTES];
        uint8_t seedC[CHALLENGESEEDBYTES];
        poly z1[Z1LEN], hint[EM];
        if (roff + 64 > rn) roff = roff0;
        memcpy(m, randbuf + roff, 32); roff += 32;
        init_random_number(&drng_algorithm, randbuf + roff, 55 * 8); roff += 55;
        rtn = sig_sign(skraw, sklen, m, 32, sig, &siglen);
        if (rtn != 0) { fprintf(stderr, "sign fail at %ld\n", q); return 1; }
        if (unpack_sig(seedC, z1, hint, sig) != 0) { fprintf(stderr, "unpack fail\n"); return 1; }
        for (k = 0; k < Z1LEN; ++k) {
            for (i = 0; i < 256; ++i) {
                double v = (double)z1[k].coeffs[i];
                z1re[k][i] = v * twist_re[i];
                z1im[k][i] = v * twist_im[i];
            }
            fft256(z1re[k], z1im[k]);
        }
        for (k = 0; k < 3; ++k) {
            memcpy(zre[k], z1re[k + 1], sizeof(double) * 256);
            memcpy(zim[k], z1im[k + 1], sizeof(double) * 256);
        }
        for (k = 0; k < 3; ++k)
            for (kp = 0; kp < 3; ++kp)
                for (i = 0; i < 256; ++i)
                    Pre[k][kp][i] += zre[k][i] * zre[kp][i] + zim[k][i] * zim[kp][i],
                    Pim[k][kp][i] += zim[k][i] * zre[kp][i] - zre[k][i] * zim[kp][i];
        for (k = 0; k < 3; ++k)
            for (i = 0; i < 256; ++i) {
                double v = (double)z1[k + 1].coeffs[i];
                blk_sum[k] += v; blk_sumsq[k] += v * v;
            }
        if ((q & 0xFFFF) == 0) { fprintf(stderr, "%ld\r", q); fflush(stderr); }
    }

    fprintf(stderr, "sign failures: %ld / %ld\n", nfail, NSIG);
    o = fopen(argv[3], "wb");
    fwrite(&NSIG, sizeof(long), 1, o);
    fwrite(Pre, sizeof(double), 3 * 3 * 256, o);
    fwrite(Pim, sizeof(double), 3 * 3 * 256, o);
    fwrite(blk_sum, sizeof(double), 3, o);
    fwrite(blk_sumsq, sizeof(double), 3, o);
    fclose(o);
    return 0;
}
