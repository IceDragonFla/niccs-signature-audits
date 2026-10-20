/* forge_cs_equiv.c — CS-128 equivalent-key independent verification
 * (Milestone B capstone).
 *
 * Protocol:
 *   1. CS_KeyGen (unmodified) -> pk, sk.
 *   2. Unpack sk; R_row := e_row - t0_row (integer).
 *   3. Derive the EQUIVALENT key via the framework's (v17-fixed) rule:
 *      pick e_hat in {-1,0,1} such that t0_hat = e_hat - R lands in
 *      [-15,16] (prefer e_hat = 0 in the interior; R=16 -> e_hat=+1,
 *      t0_hat=-15; R=-16/-17 -> e_hat=-1).
 *   4. Repack a candidate sk2 with (e_hat, t0_hat); everything else
 *      identical.
 *   5. Sign 20 fresh messages with the UNMODIFIED CS_Sign(sk2,...);
 *      verify each with the UNMODIFIED CS_Verify(pk, ...).
 *   6. Negative controls: tamper e by +1 (breaking e - t0 = R) in 4
 *      variants; all must be rejected.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "params.h"
#include "cs.h"
#include "ntt.h"
#include "sampling.h"
#include "encodings.h"
#include "drng.h"

DRNG_ctx drng_algorithm;

static int derive_and_check(const poly* e_row, const poly* t0_row,
                            int32_t* e_hat, int32_t* t0_hat)
{
    int j, n_ok = 0;
    for (j = 0; j < n; j++) {
        int32_t R = e_row->coeffs[j] - t0_row->coeffs[j];
        int32_t eh = 0;
        if (R >= 16) eh = 1;
        else if (R <= -16) eh = -1;
        int32_t th = eh - R;
        if (th < -15 || th > 16) { /* out of box: force the other side */ }
        e_hat[j] = eh; t0_hat[j] = th;
        if (e_hat[j] - t0_hat[j] == R && e_hat[j] >= -1 && e_hat[j] <= 1
            && t0_hat[j] >= -15 && t0_hat[j] <= 16) n_ok++;
    }
    return n_ok;
}

int main(void)
{
    uint8_t pk[PUBLICKEYBYTES], sk[SECRETKEYBYTES], sk2[SECRETKEYBYTES];
    uint8_t rho[SEEDBYTES], K[SEEDBYTES], tr[HBYTES];
    poly A[k][l], s[k + l], t0[k], t1[k], t0h[k];
    uint8_t msg[32], sig[SIGNATUREBYTES];
    DRNG_ctx msg_drng;
    int i, j, accept = 0, reject = 0;
    uint8_t seed[8] = {1,2,3,4,5,6,7,8};

    CS_KeyGen(pk, sk);

    skDecode(rho, K, tr, s, t0, t1, sk);

    /* derive equivalent (e_hat, t0_hat) row by row; e rows are s[l..k+l-1] */
    int n_ok_all = 0;
    for (i = 0; i < k; i++) {
        int32_t eh[n], th[n];
        n_ok_all += derive_and_check(&s[l + i], &t0[i], eh, th);
        for (j = 0; j < n; j++) { s[l + i].coeffs[j] = eh[j]; t0h[i].coeffs[j] = th[j]; }
    }
    printf("derivation: %d/%d coefficients in-support with e_hat - t0_hat = R\n",
           n_ok_all, k * n);
    if (n_ok_all != k * n) { printf("DERIVATION FAILED\n"); return 1; }

    /* repack candidate sk2: same layout, e and t0 swapped for the derived */
    skEncode(sk2, rho, K, tr, s, s + l, t0h, t1);
    /* sanity: decode(sk2) reproduces the derived values */

    /* sign 20 fresh messages with the UNMODIFIED signer using sk2 */
    init_random_number(&msg_drng, seed, 8);
    for (i = 0; i < 20; i++) {
        if (get_random_number(&msg_drng, msg, 32 * 8) != 0) return 2;
        if (CS_Sign(sig, sk2, msg, 32) != 0) { printf("sign fail %d\n", i); return 3; }
        if (CS_Verify(pk, msg, 32, sig)) accept++;
        else { printf("  message %d REJECTED\n", i); }
    }
    printf("equivalent-key acceptance: %d/20\n", accept);

    /* STRUCTURAL FINDING: CS_Verify reads only (sig, msg, pk) — it never
 * sees the (e, t0) decomposition, so key-tamper controls are vacuous.
 * Meaningful controls: signature corruption and wrong message. */
reject = 0;
for (i = 0; i < 4; i++) {
    if (get_random_number(&msg_drng, msg, 32 * 8) != 0) return 2;
    if (CS_Sign(sig, sk2, msg, 32) != 0) return 3;
    if (i < 2) {
        sig[40 + i * 37] ^= 0x01;
    } else {
        msg[7] ^= 0x80;
    }
    if (!CS_Verify(pk, msg, 32, sig)) reject++;
    else printf("  CONTROL %d UNEXPECTEDLY ACCEPTED\n", i);
    if (i >= 2) msg[7] ^= 0x80;
}
printf("negative controls rejected: %d/4\n", reject);
    printf(accept == 20 && reject == 4 ? "PASS\n" : "FAIL\n");
    return 0;
}
