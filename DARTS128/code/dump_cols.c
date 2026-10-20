/* dump_cols.c — export the 1024 reference-path columns A_c * x^m + the
 * normal-domain A rows through the identity's own arithmetic, for the Python
 * two-stage estimator.
 *
 * usage: dump_cols <pk> <out.bin>
 *   out.bin: 1024*512 int64 columns (col[c*N+m][i] = (A_c*x^m)[i], c=0..1)
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

static long col[2][N][N];

int main(int argc, char **argv) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    polyvecl A[K];
    polyveck A0v;
    uint8_t seedA1[SEEDBYTES];

    if (argc != 3) { fprintf(stderr, "usage: %s <pk> <out>\n", argv[0]); return 1; }
    { FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
      if (fread(pk, 1, sizeof(pk), f) != sizeof(pk)) { fprintf(stderr, "short pk\n"); return 1; }
      fclose(f); }
    unpack_pk(&A0v, seedA1, pk);
    polymatkl_expand(A, seedA1);
    for (unsigned k = 0; k < 1; ++k) A[k].vec[0] = A0v.vec[k];

    for (int c = 0; c < 2; ++c)
        for (int m = 0; m < N; ++m) {
            polyvecl sh;
            polyveck res;
            memset(&sh, 0, sizeof(sh));
            sh.vec[c].coeffs[m] = 1;
            poly_ntt(&sh.vec[c]);
            polymatkl_pointwise_montgomery(&res, A, &sh);
            polyveck_invntt_tomont(&res);
            polyveck_freeze(&res);
            for (int i = 0; i < N; ++i) col[c][m][i] = res.vec[0].coeffs[i];
        }

    FILE *f = fopen(argv[2], "wb");
    if (!f) { perror(argv[2]); return 1; }
    fwrite(col, 8, 2 * N * N, f);
    fclose(f);
    printf("wrote %s (%d columns)\n", argv[2], 2 * N);
    return 0;
}
