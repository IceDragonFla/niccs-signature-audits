/* gen_dump.c — data generation for the DARTS128 (1-2b) hint-channel attack homework.
 *
 * Two modes:
 *   gen_dump keygen <pkfile> <skfile> <truthfile>
 *       Generate one keypair; write pk, sk, and the ternary ground truth
 *       (s0, s1, e as int8, 512 coefficients each).
 *   gen_dump sign <skfile> <start> <nsigs> <pubfile> <sigfile>
 *       Sign messages msg = LE32(start .. start+nsigs-1) with the instrumented
 *       signer (exposes the bimodal bit b). Per signature write:
 *         pubfile record: [b u8][c 64B][z1 int16 x1024 (centered)][h int8 x512] = 2625 B
 *         sigfile record: packed signature (CRYPTO_SIGNATUREBYTES bytes)
 *       z1/c/h are what any observer can derive from the public signature alone;
 *       b is ground truth used ONLY for scoring the attack.
 */
#include "sign.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "drng.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int crypto_sign_signature_dump(uint8_t *sig, size_t *siglen, const uint8_t *m,
                               size_t mlen, const uint8_t *sk, uint8_t *b_out);

extern DRNG_ctx drng_algorithm;

static void write_all(const char *path, const void *buf, size_t len, const char *mode) {
    FILE *f = fopen(path, mode);
    if (!f) { perror(path); exit(1); }
    if (fwrite(buf, 1, len, f) != len) { fprintf(stderr, "short write %s\n", path); exit(1); }
    fclose(f);
}

static void le32(uint8_t *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = (v >> 24) & 0xff;
}

static void dump_truth(const char *path, const uint8_t *sk) {
    polyvecl A[K];
    poly s0;
    polyvecl_1 s1;
    polyveck e;
    uint8_t key[SEEDBYTES];
    static int8_t truth[3 * N];
    unsigned int i;

    unpack_sk(A, &s0, &s1, &e, key, sk);
    for (i = 0; i < N; ++i) truth[i]         = (int8_t)s0.coeffs[i];
    for (i = 0; i < N; ++i) truth[N + i]     = (int8_t)s1.vec[0].coeffs[i];
    for (i = 0; i < N; ++i) truth[2 * N + i] = (int8_t)e.vec[0].coeffs[i];
    write_all(path, truth, sizeof(truth), "wb");
}

static void do_keygen(const char *pkfile, const char *skfile, const char *truthfile) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES], sk[CRYPTO_SECRETKEYBYTES];
    /* The reference crypto_sign_keypair draws from the zero-initialized global
     * DRNG (deterministic across processes). When DARTS_KEY_SEED is set,
     * instantiate the DRNG first so a distinct key can be generated. */
    const char *seedhex = getenv("DARTS_KEY_SEED");
    if (seedhex && strlen(seedhex) >= 2) {
        uint8_t seed[32];
        memset(seed, 0, sizeof(seed));
        for (int i = 0; i < 32 && seedhex[2 * i] && seedhex[2 * i + 1]; ++i) {
            unsigned v; sscanf(seedhex + 2 * i, "%2x", &v); seed[i] = (uint8_t)v;
        }
        init_random_number(&drng_algorithm, seed, sizeof(seed));
    }
    crypto_sign_keypair(pk, sk);
    write_all(pkfile, pk, sizeof(pk), "wb");
    write_all(skfile, sk, sizeof(sk), "wb");
    dump_truth(truthfile, sk);
    printf("keypair written: pk=%zu sk=%zu bytes\n", sizeof(pk), sizeof(sk));
}

static void do_sign(const char *skfile, long start, long nsigs,
                    const char *pubfile, const char *sigfile) {
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t sig[CRYPTO_SIGNATUREBYTES + 16];
    uint8_t msg[4];
    size_t siglen;
    uint8_t b;
    FILE *fp, *fs;

    static uint8_t rec[1 + 64 + 2 * N * 2 + N]; /* b + c + z1(int16x1024) + h(int8x512) */

    { FILE *f = fopen(skfile, "rb"); if (!f) { perror(skfile); exit(1); }
      if (fread(sk, 1, sizeof(sk), f) != sizeof(sk)) { fprintf(stderr, "short sk read\n"); exit(1); }
      fclose(f); }

    fp = fopen(pubfile, "wb"); if (!fp) { perror(pubfile); exit(1); }
    fs = fopen(sigfile, "wb"); if (!fs) { perror(sigfile); exit(1); }

    for (long j = 0; j < nsigs; ++j) {
        poly c;
        polyveck h;
        polyvecl lowbits_z1, highbits_z1, z1;
        static int16_t z1buf[2 * N];
        static int8_t hbuf[N];
        unsigned int i, k;
        long v;

        le32(msg, (uint32_t)(start + j));
        if (crypto_sign_signature_dump(sig, &siglen, msg, 4, sk, &b)) {
            fprintf(stderr, "signature failed at j=%ld\n", j);
            exit(1);
        }
        if (siglen != CRYPTO_SIGNATUREBYTES) {
            fprintf(stderr, "bad siglen at j=%ld\n", j);
            exit(1);
        }

        if (unpack_sig(&c, &lowbits_z1, &highbits_z1, &h, sig)) {
            fprintf(stderr, "unpack failed at j=%ld\n", j);
            exit(1);
        }
        polyvecl_compose(&z1, &lowbits_z1, &highbits_z1);

        rec[0] = b;
        memcpy(rec + 1, sig, 64); /* challenge c bit-packed */

        for (i = 0; i < L; ++i) {
            for (k = 0; k < N; ++k) {
                v = z1.vec[i].coeffs[k];
                if (v > Q / 2) v -= Q;          /* center mod q */
                if (v < -30000 || v > 30000) { fprintf(stderr, "z1 range\n"); exit(1); }
                z1buf[i * N + k] = (int16_t)v;
            }
        }
        for (k = 0; k < N; ++k) {
            v = h.vec[0].coeffs[k];
            /* decode_h returns the mod-2^d representative of the signed hint
             * difference Comp(w) - Comp(arg2); near q the mask adds +-2^d
             * (encoder edge symbols at +-1020). Re-center modularly into
             * (-2^(d-1), 2^(d-1)]: |true value| <= ~|z2|*2^d/q + 1 = ~50. */
            v = ((v % (1 << D)) + (1 << D)) % (1 << D);
            if (v >= (1 << (D - 1))) v -= (1 << D);
            if (v < -127 || v > 127) { fprintf(stderr, "h range %ld\n", v); exit(1); }
            hbuf[k] = (int8_t)v;
        }

        memcpy(rec + 65, z1buf, sizeof(z1buf));
        memcpy(rec + 65 + sizeof(z1buf), hbuf, sizeof(hbuf));

        fwrite(rec, 1, sizeof(rec), fp);
        fwrite(sig, 1, CRYPTO_SIGNATUREBYTES, fs);

        if ((j & 0xFFFF) == 0) { printf("\rj=%ld", j); fflush(stdout); }
    }
    printf("\ndone: %ld signatures\n", nsigs);
    fclose(fp); fclose(fs);
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "keygen") && argc == 5) {
        do_keygen(argv[2], argv[3], argv[4]);
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "sign") && argc == 7) {
        do_sign(argv[2], atol(argv[3]), atol(argv[4]), argv[5], argv[6]);
        return 0;
    }
    fprintf(stderr,
        "usage:\n  %s keygen <pk> <sk> <truth>\n  %s sign <sk> <start> <nsigs> <pub> <sig>\n",
        argv[0], argv[0]);
    return 1;
}
