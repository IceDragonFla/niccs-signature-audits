/*
 * harness_forge.c -- complete the key-recovery attack:
 *   input: pk file (from *_key.bin), recovered s (3x256 int32, possibly a
 *          few wrong coefficients), rand file, fresh message (from rand).
 *   1. e'_rec = b - (a + A.s) centered (exact if s correct)
 *   2. greedy residual correction of wrong s coefficients via |e'| outliers
 *   3. rebuild sk (own masterK, tr = HashPK(pk)), sign a fresh message,
 *      verify.
 * argv[1]=key file, argv[2]=s_rec file, argv[3]=rand file, argv[4]=out prefix
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "packing.h"
#include "rounding.h"
#include "symmetric.h"
#include "xof.h"
#include "drng.h"
#include "SIG_AlgorithmInstance.h"

DRNG_ctx drng_algorithm;

static int violations(const poly ep[EM])
{
    int v = 0, p; unsigned i;
    for (p = 0; p < EM; ++p)
        for (i = 0; i < N; ++i)
            if (ep[p].coeffs[i] > 8 || ep[p].coeffs[i] < -8) ++v;
    return v;
}

int main(int argc, char **argv)
{
    FILE *f;
    uint8_t pk[4200], sk[CRYPTO_SECRETKEYBYTES], skraw[3000];
    uint8_t seedA[SEEDBYTES], masterK[CHALLENGESEEDBYTES], tr[CHALLENGESEEDBYTES];
    poly b[EM], s[ELL], ep[EM], ep0[EM];
    poly16 agen[EM], hAgen[EM * ELL];
    uint8_t msg[32], sig[CRYPTO_BYTES], tr_in[1 + 4200];
    unsigned long long siglen, vlen, sklen2;
    xof_ctx ctx;
    int k, iter, rtn, p;
    unsigned i, t, delta;
    size_t pklen = CRYPTO_PUBLICKEYBYTES;

    if (argc != 5) { fprintf(stderr, "usage: %s key.bin s_rec.bin rand.bin prefix\n", argv[0]); return 1; }
    f = fopen(argv[1], "rb");
    fread(pk, 1, pklen, f);
    fread(skraw, 1, CRYPTO_SECRETKEYBYTES, f); /* ground truth (unused) */
    fclose(f);
    f = fopen(argv[2], "rb");
    fread(s, sizeof(poly), ELL, f);
    fclose(f);
    { FILE *g = fopen(argv[3], "rb"); static uint8_t rb[1 << 16];
      size_t rn = fread(rb, 1, sizeof rb, g); fclose(g);
      memcpy(msg, rb, 32);
      memcpy(masterK, rb + 32, CHALLENGESEEDBYTES);
      init_random_number(&drng_algorithm, rb + 64, 55 * 8);
      (void)rn; }

    if (unpack_pk(seedA, b, pk) != 0) { fprintf(stderr, "unpack_pk fail\n"); return 1; }
    expand_a(agen, hAgen, seedA);

    /* e' = centered(b - (a + A s)) via keygen_bproduct with e = 0 */
    {
        poly e0[EM];
        poly b0[EM];
        memset(e0, 0, sizeof e0);
        keygen_bproduct(b0, agen, hAgen, s, e0);
        for (p = 0; p < EM; ++p)
            for (i = 0; i < N; ++i)
                { int32_t v = freeze(b[p].coeffs[i] - b0[p].coeffs[i]);
                  ep[p].coeffs[i] = (v > Q / 2) ? v - Q : v; }
    }
    printf("initial |e'|>8 violations: %d\n", violations(ep));

    /* greedy correction: adjust single s coefficients to kill violations */
    for (iter = 0; iter < 60 && violations(ep); ++iter) {
        int bestv = violations(ep), bi = -1, bk = 0, bt = 0, bd = 0;
        for (k = 0; k < ELL; ++k)
            for (t = 0; t < N; ++t) {
                int sv = s[k].coeffs[t];
                for (delta = 0; delta < 6; ++delta) {
                    int d = (int[]){1, -1, 2, -2, 3, -3}[delta];
                    poly st[ELL], e0[EM], b0[EM];
                    memcpy(st, s, sizeof st);
                    memset(e0, 0, sizeof e0);
                    st[k].coeffs[t] = sv + d;
                    keygen_bproduct(b0, agen, hAgen, st, e0);
                    for (p = 0; p < EM; ++p)
                        for (i = 0; i < N; ++i) {
                            int32_t v = freeze(b[p].coeffs[i] - b0[p].coeffs[i]);
                            ep0[p].coeffs[i] = (v > Q / 2) ? v - Q : v;
                        }
                    int vv = violations(ep0);
                    if (vv < bestv) { bestv = vv; bi = iter; bk = k; bt = t; bd = d; }
                }
            }
        if (bi < 0) { printf("correction stalled at %d violations\n", bestv); break; }
        s[bk].coeffs[bt] += bd;
        {
            poly e0[EM], b0[EM];
            memset(e0, 0, sizeof e0);
            keygen_bproduct(b0, agen, hAgen, s, e0);
            for (p = 0; p < EM; ++p)
                for (i = 0; i < N; ++i) {
                    int32_t v = freeze(b[p].coeffs[i] - b0[p].coeffs[i]);
                    ep[p].coeffs[i] = (v > Q / 2) ? v - Q : v;
                }
        }
        printf("iter %d: s[%d][%u] += %d -> violations %d\n", iter, bk, bt, bd, violations(ep));
    }
    printf("final violations: %d\n", violations(ep));

    /* tr = HashPK(pk) = XOF(0x05 || pk) */
    tr_in[0] = DS_HASH_PK;
    memcpy(tr_in + 1, pk, pklen);
    xof256_init(&ctx, tr_in, 1 + pklen);
    xof256_squeeze(&ctx, tr, CHALLENGESEEDBYTES);

    pack_sk(sk, seedA, b, masterK, tr, s, ep);
    printf("forged-key sk assembled; signing fresh message...\n");
    rtn = sig_sign(sk, CRYPTO_SECRETKEYBYTES, msg, 32, sig, &siglen);
    printf("sig_sign -> %d (siglen=%llu)\n", rtn, (unsigned long long)siglen);
    rtn = sig_verify(pk, pklen, sig, siglen, msg, 32);
    printf("*** sig_verify -> %d  (0 = FORGED SIGNATURE ACCEPTED) ***\n", rtn);

    { char fn[512]; snprintf(fn, sizeof fn, "%s_srec.bin", argv[4]);
      f = fopen(fn, "wb"); fwrite(s, sizeof(poly), ELL, f); fclose(f);
      snprintf(fn, sizeof fn, "%s_sig.bin", argv[4]);
      f = fopen(fn, "wb"); fwrite(sig, 1, (size_t)siglen, f); fclose(f); }
    (void)sklen2; (void)vlen;
    return 0;
}
