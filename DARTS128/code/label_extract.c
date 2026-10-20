/* label_extract.c — PUBLIC-SIDE branch tell for DARTS128 (1-2b mode).
 *
 * The 1-2b update forces verification to try BOTH branches
 *     w1 = h + Compress(A*z1 ± (Q+1)/2 * c * beta, d)
 * because the verifier does not know the bimodal bit b. The true branch always
 * hash-matches; the FALSE branch also matches whenever the +-c*beta shift
 * crosses no quantization boundary at the tau support positions of c
 * (probability ~ (1-2^d/q)^tau ~= 0.79). The observer therefore computes the
 * full match set and labels:
 *     0 -> only "+q_hat*c*beta" matched  (b = 0, the "w+ac" side)   [clean]
 *     1 -> only "-q_hat*c*beta" matched  (b = 1, the "w-ac" side)   [clean]
 *     2 -> no branch matched (verification would fail)
 *     3 -> both branches matched (bimodal-ambiguous, no label)
 * Clean labels are deterministic (error-free); the ambiguous majority is discarded.
 *
 * usage: label_extract <pkfile> <sigfile> <start> <nsigs> <labelfile>
 */
#include "sign.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "polymat.h"
#include "symmetric.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void le32(uint8_t *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = (v >> 24) & 0xff;
}

int main(int argc, char **argv) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sig[CRYPTO_SIGNATUREBYTES + 16];
    uint8_t msg[4], mu[CRHBYTES];
    uint8_t buf[POLYVECK_HIGHBITS_PACKEDBYTES];
    uint8_t seedA1[SEEDBYTES];
    long start, nsigs;
    unsigned int i;
    long cnt[4] = {0, 0, 0, 0};
    FILE *fpk, *fsig, *flab;

    if (argc != 6) {
        fprintf(stderr, "usage: %s <pk> <sig> <start> <nsigs> <labels>\n", argv[0]);
        return 1;
    }
    start = atol(argv[3]);
    nsigs = atol(argv[4]);

    fpk = fopen(argv[1], "rb"); if (!fpk) { perror(argv[1]); return 1; }
    if (fread(pk, 1, sizeof(pk), fpk) != sizeof(pk)) { fprintf(stderr, "short pk\n"); return 1; }
    fclose(fpk);

    fsig = fopen(argv[2], "rb"); if (!fsig) { perror(argv[2]); return 1; }
    flab = fopen(argv[5], "wb"); if (!flab) { perror(argv[5]); return 1; }

    /* branch-independent public state */
    polyvecl A[K];
    polyveck A0, h, w1, Az1_base, Az1_cb, Az1_cb_comp;
    poly c, c1, c_half_q;
    polyvecl z1, lowbits_z1, highbits_z1;
    sm3_xof_state state;

    unpack_pk(&A0, seedA1, pk);
    polymatkl_expand(A, seedA1);
    for (i = 0; i < K; ++i) {
        A[i].vec[0] = A0.vec[i];
    }

    for (long j = 0; j < nsigs; ++j) {
        int branch, match_plus = 0, match_minus = 0, label;

        if (fread(sig, 1, CRYPTO_SIGNATUREBYTES, fsig) != CRYPTO_SIGNATUREBYTES) {
            fprintf(stderr, "short sig read at j=%ld\n", j);
            return 1;
        }
        le32(msg, (uint32_t)(start + j));

        if (unpack_sig(&c, &lowbits_z1, &highbits_z1, &h, sig)) {
            fprintf(stderr, "unpack_sig failed at j=%ld\n", j);
            return 1;
        }
        polyvecl_compose(&z1, &lowbits_z1, &highbits_z1);

        polyvecl_ntt(&z1);
        poly_mul_halfq(&c_half_q, &c);

        polymatkl_pointwise_montgomery(&Az1_base, A, &z1);
        polyveck_invntt_tomont(&Az1_base);

        sm3_xof_absorb_twice(&state, pk, CRYPTO_PUBLICKEYBYTES, msg, 4);
        sm3_xof_squeeze(mu, CRHBYTES, &state);

        for (branch = 0; branch < 2; ++branch) {
            int equal = 1;

            Az1_cb = Az1_base;
            if (branch == 0)
                poly_add(&Az1_cb.vec[0], &Az1_base.vec[0], &c_half_q);
            else
                poly_sub(&Az1_cb.vec[0], &Az1_base.vec[0], &c_half_q);
            polyveck_freeze(&Az1_cb);

            polyveck_compress(&Az1_cb_comp, &Az1_cb);
            polyveck_add(&w1, &Az1_cb_comp, &h);
            polyveck_pack_compressed(buf, &w1);
            poly_challenge(&c1, buf, mu);

            for (i = 0; i < N; ++i) {
                if (c.coeffs[i] != c1.coeffs[i]) { equal = 0; break; }
            }
            if (equal) {
                if (branch == 0) match_plus = 1; else match_minus = 1;
            }
        }

        if (match_plus && !match_minus) label = 0;        /* w+ac side only */
        else if (!match_plus && match_minus) label = 1;   /* w-ac side only */
        else if (!match_plus && !match_minus) label = 2;  /* would not verify */
        else label = 3;                                   /* both: ambiguous */

        cnt[label]++;
        fputc(label, flab);

        if ((j & 0xFFFF) == 0) { printf("\rj=%ld", j); fflush(stdout); }
    }

    printf("\nbranch stats: +only(b=0)=%ld  -only(b=1)=%ld  no-match=%ld  both(ambiguous)=%ld\n",
           cnt[0], cnt[1], cnt[2], cnt[3]);
    fclose(fsig); fclose(flab);
    return 0;
}
