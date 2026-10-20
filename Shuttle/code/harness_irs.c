/*
 * harness_irs.c -- isolated IRS (rejection-free masking transition) test.
 *
 * Fixed key from sig_keygen under a deterministic DRNG; T trials of
 * (y=sample_y, c=sample_c, reject_sample). Dumps sk_tilde, per-trial
 * seeds and the resulting z vectors for offline bias analysis.
 * argv[1]=rand file, argv[2]=#trials, argv[3]=output dump
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
#include "irs.h"
#include "symmetric.h"
#include "drng.h"
#include "SIG_AlgorithmInstance.h"

DRNG_ctx drng_algorithm;

int main(int argc, char **argv)
{
    uint8_t pk[4000], sk[4000], skraw[CRYPTO_SECRETKEYBYTES];
    uint8_t seedA[SEEDBYTES], masterK[CHALLENGESEEDBYTES],
        tr[CHALLENGESEEDBYTES];
    poly b[EM], s[ELL], ep[EM], sk_full[KVEC], sk_tilde[KVEC];
    unsigned long long pklen, sklen, sgnlen, smlen = 0;
    FILE *f, *o;
    static uint8_t randbuf[1 << 21];
    size_t rn, roff = 0;
    int i, T;

    if (argc != 4) { fprintf(stderr, "usage: %s rand.bin trials out.dump\n", argv[0]); return 1; }
    T = atoi(argv[2]);
    f = fopen(argv[1], "rb");
    if (!f) { perror("rand"); return 1; }
    rn = fread(randbuf, 1, sizeof(randbuf), f);
    fclose(f);
    if (rn < 65536) { fprintf(stderr, "rand too small\n"); return 1; }

    /* deterministic keygen: init the scheme DRNG from the rand file */
    if (rn < SEEDLEN) return 1;
    init_random_number(&drng_algorithm, randbuf, SEEDLEN * 8);
    if (getenv("USE_KEYFILE")) {
        FILE *kf = fopen(getenv("USE_KEYFILE"), "rb");
        fseek(kf, (long)CRYPTO_PUBLICKEYBYTES, SEEK_SET);
        if (!kf || fread(skraw, 1, CRYPTO_SECRETKEYBYTES, kf) != CRYPTO_SECRETKEYBYTES) {
            fprintf(stderr, "keyfile fail"); return 1;
        }
        fclose(kf);
    } else if (sig_keygen(pk, &pklen, skraw, &sklen) != 0) {
        fprintf(stderr, "keygen failed"); return 1;
    }
    (void)pk; (void)sgnlen; (void)smlen;

    if (unpack_sk(seedA, b, masterK, tr, s, ep, skraw) != 0) {
        fprintf(stderr, "unpack_sk failed\n"); return 1;
    }
    {   /* sk_full[0] = polynomial 1 */
        unsigned k;
        for (k = 0; k < N; ++k) sk_full[0].coeffs[k] = 0;
        sk_full[0].coeffs[0] = 1;
    }
    for (i = 0; i < ELL; ++i) sk_full[1 + i] = s[i];
    for (i = 0; i < EM; ++i) sk_full[1 + ELL + i] = ep[i];
    stretch_s(sk_tilde, sk_full);

    o = fopen(argv[3], "wb");
    if (!o) { perror("out"); return 1; }
    fwrite(sk_tilde, sizeof(poly), KVEC, o);

    roff = SEEDLEN;
    for (i = 0; i < T; i++) {
        uint8_t seedY[SEEDBYTES], seedC[CHALLENGESEEDBYTES];
        uint8_t irs_seed[1 + SEEDBYTES];
        poly y[KVEC], z[KVEC], c;
        xof_ctx ctx;
        int j;

        if (roff + SEEDBYTES + CHALLENGESEEDBYTES > rn) { roff = SEEDLEN; }
        memcpy(seedY, randbuf + roff, SEEDBYTES); roff += SEEDBYTES;
        memcpy(seedC, randbuf + roff, CHALLENGESEEDBYTES); roff += CHALLENGESEEDBYTES;

        sample_y(y, seedY);
        sample_c(&c, seedC);
        irs_seed[0] = DS_IRS;
        memcpy(irs_seed + 1, seedY, SEEDBYTES);
        xof256_init(&ctx, irs_seed, 1 + SEEDBYTES);
        reject_sample(&ctx, z, y, &c, sk_tilde);

        fwrite(seedY, 1, SEEDBYTES, o);
        fwrite(seedC, 1, CHALLENGESEEDBYTES, o);
        fwrite(c.coeffs, sizeof(int32_t), N, o);
        fwrite(y, sizeof(poly), KVEC, o);
        fwrite(z, sizeof(poly), KVEC, o);
    }
    fclose(o);
    return 0;
}
