/*
 * harness_sign.c -- end-to-end public-data attack harness.
 * Generates NSIG real signatures under the shipped (or fixed) ref build,
 * dumps: pk, ground-truth sk + stretched secret, and the attacker-visible
 * z1 (4 polys) decoded from each signature via the verify-side unpack_sig.
 * argv[1]=rand file, argv[2]=NSIG, argv[3]=out prefix
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
#include "drng.h"
#include "SIG_AlgorithmInstance.h"

DRNG_ctx drng_algorithm;

int main(int argc, char **argv)
{
    uint8_t pk[4200], skraw[3000];
    unsigned long long pklen, sklen, siglen;
    int NSIG, q, i, k, rtn;
    FILE *f, *oz;
    static uint8_t randbuf[1 << 24];
    size_t rn, roff;
    uint8_t seedA[SEEDBYTES], masterK[CHALLENGESEEDBYTES],
        tr[CHALLENGESEEDBYTES];
    poly b[EM], s[ELL], ep[EM], sk_full[KVEC], skt[KVEC];
    char fn[512];

    if (argc != 4) { fprintf(stderr, "usage: %s rand.bin NSIG outprefix\n", argv[0]); return 1; }
    NSIG = atoi(argv[2]);
    f = fopen(argv[1], "rb");
    if (!f) { perror("rand"); return 1; }
    rn = fread(randbuf, 1, sizeof(randbuf), f);
    fclose(f);
    if (rn < 65536 + 64 * (size_t)NSIG) { fprintf(stderr, "rand too small\n"); return 1; }

    init_random_number(&drng_algorithm, randbuf, 55 * 8);
    rtn = sig_keygen(pk, &pklen, skraw, &sklen);
    if (rtn) { fprintf(stderr, "keygen %d\n", rtn); return 1; }

    if (unpack_sk(seedA, b, masterK, tr, s, ep, skraw) != 0) return 1;
    {   unsigned _k; for (_k = 0; _k < N; ++_k) sk_full[0].coeffs[_k] = 0;
        sk_full[0].coeffs[0] = 1; }
    for (i = 0; i < ELL; ++i) sk_full[1 + i] = s[i];
    for (i = 0; i < EM; ++i) sk_full[1 + ELL + i] = ep[i];
    stretch_s(skt, sk_full);

    /* ground truth dump: pk, sk, s (3 polys), e' (3 polys), skt (7 polys) */
    snprintf(fn, sizeof fn, "%s_key.bin", argv[3]);
    f = fopen(fn, "wb");
    fwrite(pk, 1, (size_t)pklen, f);
    fwrite(skraw, 1, (size_t)sklen, f);
    fwrite(s, sizeof(poly), ELL, f);
    fwrite(ep, sizeof(poly), EM, f);
    fwrite(skt, sizeof(poly), KVEC, f);
    fclose(f);

    snprintf(fn, sizeof fn, "%s_z1.bin", argv[3]);
    oz = fopen(fn, "wb");

    roff = 65536;
    for (q = 0; q < NSIG; q++) {
        uint8_t m[32], sig[CRYPTO_BYTES];
        uint8_t seedC[CHALLENGESEEDBYTES];
        poly z1[Z1LEN], hint[EM];

        memcpy(m, randbuf + roff, 32); roff += 32;
        rtn = sig_sign(skraw, sklen, m, 32, sig, &siglen);
        if (rtn != 0) { fprintf(stderr, "sign %d at q=%d\n", rtn, q); return 1; }
        if (unpack_sig(seedC, z1, hint, sig) != 0) {
            fprintf(stderr, "unpack_sig failed at q=%d\n", q); return 1;
        }
        for (k = 0; k < Z1LEN; ++k)
            fwrite(z1[k].coeffs, sizeof(int32_t), N, oz);
        {   /* pack the public challenge support: c has weight TAU ones */
            poly cc; uint8_t cbits[(N + 7) / 8] = {0}; unsigned t2;
            sample_c(&cc, seedC);
            for (t2 = 0; t2 < N; ++t2)
                if (cc.coeffs[t2]) cbits[t2 >> 3] |= (uint8_t)(1u << (t2 & 7));
            fwrite(cbits, 1, (N + 7) / 8, oz);
        }
    }
    fclose(oz);
    (void)masterK; (void)tr; (void)b; (void)seedA;
    return 0;
}
