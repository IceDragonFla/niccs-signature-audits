/* forge.c — forgery closure test for the DARTS128 (1-2b) homework attack.
 *
 * Loads a RECOVERED secret (s0, s1, e as int8, 512 coefficients each — the
 * output of attack.py), builds a standard secret key with an arbitrary K
 * (verification never checks K; K only shapes the y-sampler), then signs
 * FRESH messages with the UNMODIFIED reference crypto_sign_signature and
 * verifies with the UNMODIFIED reference crypto_sign_verify.
 *
 * Negative controls:
 *   (a) one coefficient of e flipped (breaks the pk identity) -> must fail
 *   (b) one bit of a valid signature flipped                  -> must fail
 *   (c) valid signature checked under a different public key  -> must fail
 *
 * usage: forge <pk> <sec_int8_file> <nforge>
 *   sec_int8_file layout: [s0 512B][s1 512B][e 512B]
 */
#include "sign.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void le32(uint8_t *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = (v >> 24) & 0xff;
}

static void load_sec(const char *path, poly *s0, polyvecl_1 *s1, polyveck *e) {
    static int8_t buf[3 * N];
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(buf, 1, sizeof(buf), f) != sizeof(buf)) { fprintf(stderr, "short sec\n"); exit(1); }
    fclose(f);
    for (unsigned int i = 0; i < N; ++i) s0->coeffs[i] = buf[i];
    for (unsigned int i = 0; i < N; ++i) s1->vec[0].coeffs[i] = buf[N + i];
    for (unsigned int i = 0; i < N; ++i) e->vec[0].coeffs[i] = buf[2 * N + i];
}

static void build_sk(uint8_t sk[CRYPTO_SECRETKEYBYTES], const uint8_t *pk,
                     const poly *s0, const polyvecl_1 *s1, const polyveck *e) {
    uint8_t key[SEEDBYTES];
    memset(key, 0xA5, sizeof(key)); /* arbitrary K: any value yields a valid signer */
    pack_sk(sk, pk, s0, s1, e, key);
}

/* fixup mode: recompute e from the MN-LWE key identity
 *     A0*s0 + A1*s1 + e = (Q+1)/2 * beta   (mod q)
 * using the same arithmetic path as the reference signer. If the recovered
 * (s0, s1) are exact, e* is exact regardless of the hint-channel quality. */
static void do_fixup(const char *pkfile, const char *secfile) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    poly s0; polyvecl_1 s1; polyveck e;
    polyvecl A[K];
    polyveck A0, res, qbeta;
    polyvecl sh;
    uint8_t seedA1[SEEDBYTES];
    static int8_t buf[3 * N];
    long changed = 0, nonternary = 0;
    unsigned int i, k;

    { FILE *f = fopen(pkfile, "rb"); if (!f) { perror(pkfile); exit(1); }
      if (fread(pk, 1, sizeof(pk), f) != sizeof(pk)) { fprintf(stderr, "short pk\n"); exit(1); }
      fclose(f); }
    { FILE *f = fopen(secfile, "rb+"); if (!f) { perror(secfile); exit(1); }
      if (fread(buf, 1, sizeof(buf), f) != sizeof(buf)) { fprintf(stderr, "short sec\n"); exit(1); }
      for (i = 0; i < N; ++i) s0.coeffs[i] = buf[i];
      for (i = 0; i < N; ++i) s1.vec[0].coeffs[i] = buf[N + i];
      for (i = 0; i < N; ++i) e.vec[0].coeffs[i] = buf[2 * N + i];

      unpack_pk(&A0, seedA1, pk);
      polymatkl_expand(A, seedA1);
      for (k = 0; k < K; ++k) A[k].vec[0] = A0.vec[k];

      poly_ntt(&s0);
      polyvecl_1_ntt(&s1);
      sh.vec[0] = s0;
      for (i = 0; i < L - 1; ++i) sh.vec[i + 1] = s1.vec[i];
      polymatkl_pointwise_montgomery(&res, A, &sh);
      polyveck_invntt_tomont(&res);

      polyveck_halfq_beta(&qbeta);   /* ((Q+1)/2, 0, ..., 0) */
      polyveck_sub(&res, &qbeta, &res);  /* e* = qhat*beta - (A0 s0 + A1 s1) */

      for (i = 0; i < N; ++i) {
          long v = res.vec[0].coeffs[i], vc;
          if (v > Q / 2) v -= Q;
          vc = buf[2 * N + i];
          if (v < -1 || v > 1) nonternary++;
          if (v != vc) changed++;
          buf[2 * N + i] = (int8_t)v;
      }
      rewind(f);
      fwrite(buf, 1, sizeof(buf), f);
      fclose(f);
    }
    printf("fixup: e recomputed from pk identity; changed=%ld/512, non-ternary=%ld\n",
           changed, nonternary);
}

int main(int argc, char **argv) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES], sk[CRYPTO_SECRETKEYBYTES];
    uint8_t sig[CRYPTO_SIGNATUREBYTES + 16];
    uint8_t msg[4];
    size_t siglen;
    poly s0, s0_bad;
    polyvecl_1 s1;
    polyveck e, e_bad;
    long nforge, acc = 0;
    unsigned int zero_idx = 0;

    if (argc == 4 && !strcmp(argv[1], "fixup")) {
        do_fixup(argv[2], argv[3]);
        return 0;
    }
    if (argc != 4) {
        fprintf(stderr, "usage: %s [fixup] <pk> <sec> [nforge]\n", argv[0]);
        return 1;
    }
    nforge = atol(argv[3]);

    { FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
      if (fread(pk, 1, sizeof(pk), f) != sizeof(pk)) { fprintf(stderr, "short pk\n"); return 1; }
      fclose(f); }

    load_sec(argv[2], &s0, &s1, &e);
    build_sk(sk, pk, &s0, &s1, &e);

    /* ---- positives: fresh messages, reference signer + verifier ---- */
    for (long k = 0; k < nforge; ++k) {
        le32(msg, 0x40000000u + (uint32_t)k); /* never signed during data generation */
        if (crypto_sign_signature(sig, &siglen, msg, 4, sk)) {
            fprintf(stderr, "forged signing failed at k=%ld\n", k);
            return 1;
        }
        if (crypto_sign_verify(sig, siglen, msg, 4, pk) == 0) acc++;
    }
    printf("forgery: %ld/%ld accepted by unmodified reference verifier\n", acc, nforge);

    /* ---- negative (a): tamper e on 64 zero coordinates ----
     * NOTE: a SINGLE-coefficient e-tamper still verifies with prob
     * (1-2^d/q)^tau ~= 0.79 because the hint hash only binds e through
     * quantization-boundary crossings at the ~tau support positions of
     * e_diff*c — a scheme property (loose e-binding), documented in
     * DARTS.md §9.4. Tampering 64 coefficients makes the control decisive. */
    {
        polyveck e_bad = e;
        int flipped = 0;
        unsigned int k = 0;
        while (flipped < 64 && k < N) {
            if (e_bad.vec[0].coeffs[k] == 0) { e_bad.vec[0].coeffs[k] = 1; flipped++; }
            k++;
        }
        int r;
        build_sk(sk, pk, &s0, &s1, &e_bad);
        le32(msg, 0x40000000u);
        if (crypto_sign_signature(sig, &siglen, msg, 4, sk) == 0) {
            r = crypto_sign_verify(sig, siglen, msg, 4, pk);
            printf("negative (a) tampered-e (64 coeffs): verify=%s\n",
                   r == 0 ? "ACCEPT (BAD)" : "reject (ok)");
        }
    }

    /* ---- negative (b): flip one bit of a valid signature ---- */
    {
        int r;
        build_sk(sk, pk, &s0, &s1, &e);
        le32(msg, 0x40000000u + 1000);
        if (crypto_sign_signature(sig, &siglen, msg, 4, sk) == 0) {
            sig[700] ^= 0x10;
            r = crypto_sign_verify(sig, siglen, msg, 4, pk);
            printf("negative (b) tampered-sig: verify=%s\n", r == 0 ? "ACCEPT (BAD)" : "reject (ok)");
        }
    }

    /* ---- negative (c): valid signature under a tampered public key ----
     * NOTE: crypto_sign_keypair draws from the zero-initialized global DRNG
     * (the reference never instantiates it outside the KAT harness), so a
     * fresh keypair in a fresh process REPRODUCES pk1 exactly. Tamper one
     * byte of the A0 section instead. */
    {
        uint8_t pk2[CRYPTO_PUBLICKEYBYTES];
        int r;
        memcpy(pk2, pk, sizeof(pk2));
        pk2[SEEDBYTES + 10] ^= 0x01;
        le32(msg, 0x40000000u + 2000);
        if (crypto_sign_signature(sig, &siglen, msg, 4, sk) == 0) {
            r = crypto_sign_verify(sig, siglen, msg, 4, pk2);
            printf("negative (c) tampered-pk: verify=%s\n", r == 0 ? "ACCEPT (BAD)" : "reject (ok)");
        }
    }

    return acc == nforge ? 0 : 2;
}
