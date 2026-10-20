/* sign_dump.c — instrumented copy of DARTS128 (1-2b mode) crypto_sign_signature.
 * Identical logic to the patched reference sign.c, plus:
 *   - function renamed crypto_sign_signature_dump
 *   - extra out-param *b_out exposing the bimodal bit b (ground truth for scoring)
 * The global DRNG context is extern'd (signing path is deterministic and does not use it).
 */
#include "sign.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyfix.h"
#include "polymat.h"
#include "polyvec.h"
#include "drng.h"
#include "symmetric.h"
#include <inttypes.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

extern DRNG_ctx drng_algorithm;

int crypto_sign_signature_dump(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, uint8_t *b_out) {

    uint8_t buf[POLYVECK_HIGHBITS_PACKEDBYTES] = {0};
    uint8_t seedbuf[CRHBYTES] = {0};
    uint8_t key[SEEDBYTES] = {0};
    uint8_t mu[CRHBYTES] = {0};

    uint8_t b;
    uint16_t counter = 0;

    uint64_t reject1, reject2, reject3;

    poly s0; polyvecl_1 s1; polyveck e;

    polyvecl A[K];
    polyveck w;
    polyveck h, htmp;

    polyfixvecl y1, z1, z1tmp;
    polyfixveck y2, z2, z2tmp;
    polyvecl z1rnd;
    polyveck z2rnd;
    polyvecl highbits_z1, lowbits_z1;

    poly c, c_hat;
    poly c_1_2b;
    poly w0, w0_highbits;
    poly w0_cb, w0_cb_highbits;

    polyvecl cs;
    polyveck ce;

    sm3_xof_state state;

    unsigned int i;

    unpack_sk(A, &s0, &s1, &e, key, sk);

    sm3_xof_absorb_twice(&state, sk, CRYPTO_PUBLICKEYBYTES, m, mlen);
    sm3_xof_squeeze(mu, CRHBYTES, &state);
    sm3_xof_absorb_twice(&state, key, SEEDBYTES, mu, CRHBYTES);
    sm3_xof_squeeze(seedbuf, CRHBYTES, &state);

    poly_ntt(&s0);
    polyvecl_1_ntt(&s1);
    polyveck_ntt(&e);

reject:
    counter = polyfixveclk_sample_hyperball(&y1, &y2, &b, seedbuf, counter);

    polyfixvecl_round(&z1rnd, &y1);
    polyfixveck_round(&z2rnd, &y2);

    /*--------------------- Compute w -------------------------*/
    polyvecl_ntt(&z1rnd);
    polymatkl_pointwise_montgomery(&w, A, &z1rnd);
    polyveck_invntt_tomont(&w);
    polyveck_add(&w, &w, &z2rnd);
    polyveck_freeze(&w);

    /*------------------ Generate challenge c -----------------*/
    polyveck_pack_highbits(buf, &w);
    poly_challenge(&c, buf, mu);

    /*--------------------- Compress Check ---------------------*/
    w0 = w.vec[0];
    poly_mul_1_2b(&c_1_2b, &c, b & 0x1);
    poly_add(&w0_cb, &w0, &c_1_2b); // w0_cb = w0 + (1-2b)c

    poly_freeze(&w0);
    poly_freeze(&w0_cb);

    poly_compress(&w0_highbits, &w0);
    poly_compress(&w0_cb_highbits, &w0_cb);

    int compress_reject = poly_equal(&w0_highbits, &w0_cb_highbits);
    if (compress_reject == 0) {
        goto reject;
    }

    /*-------------------- Compute (cs, ce) --------------------*/
    c_hat = c;
    poly_ntt(&c_hat);
    poly_basemul_montgomery(&cs.vec[0], &c_hat, &s0);

    for (i = 0; i < L - 1; ++i)
        poly_basemul_montgomery(&cs.vec[i + 1], &c_hat, &s1.vec[i]);
    for (i = 0; i < K; ++i)
        poly_basemul_montgomery(&ce.vec[i], &c_hat, &e.vec[i]);

    polyvecl_invntt_tomont(&cs);
    polyveck_invntt_tomont(&ce);

    /*------------ Compute z = y + (1-2b) * (c * s) ---------------*/
    polyvecl_cneg(&cs, b & 0x1);
    polyveck_cneg(&ce, b & 0x1);
    polyfixvecl_add(&z1, &y1, &cs);
    polyfixveck_add(&z2, &y2, &ce);

    /*-------------------- Check z1, z2 --------------------*/
    reject1 = ((uint64_t)B1SQ * LN * LN - polyfixveclk_sqnorm2(&z1, &z2)) >> 63;
    reject1 &= 1;

    polyfixvecl_double(&z1tmp, &z1);
    polyfixveck_double(&z2tmp, &z2);
    polyfixfixvecl_sub(&z1tmp, &z1tmp, &y1);
    polyfixfixveck_sub(&z2tmp, &z2tmp, &y2);

    reject2 = (polyfixveclk_sqnorm2(&z1tmp, &z2tmp) - BSQ * LN * LN) >> 63;
    reject2 &= 1;
    reject3 = ((uint64_t)B0SQ * LN * LN - polyfixveclk_sqnorm2(&z1, &z2)) >> 63;
    reject3 &= 1;

    if (reject1 || (reject2 && reject3 && ((b & 0x2) == 0))) {
        goto reject;
    }

    /*-------------------- Prepare signature --------------------*/
    polyfixvecl_round(&z1rnd, &z1);
    polyfixveck_round(&z2rnd, &z2);

    polyvecl_freeze(&z1rnd);
    polyveck_freeze(&z2rnd);

    polyvecl_lowbits(&lowbits_z1, &z1rnd);
    polyvecl_compress(&highbits_z1, &z1rnd);

    // Compute h = Compress(w, d) - Compress(w - round(z2) + (1-2b)cβ, d)
    polyveck_compress(&htmp, &w);
    polyveck_sub(&w, &w, &z2rnd);
    poly_add(&w.vec[0], &w.vec[0], &c_1_2b); // w.vec[0] += (1-2b)c

    polyveck_freeze(&w);

    polyveck_compress(&h, &w);
    polyveck_sub(&h, &htmp, &h);

    if (pack_sig(sig, &c, &lowbits_z1, &highbits_z1, &h)) {
        goto reject;
    }

    if (b_out) {
        *b_out = b & 0x1;
    }

    *siglen = CRYPTO_SIGNATUREBYTES;

    return 0;
}
