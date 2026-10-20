/*
 * diff_kernel.c -- cross-backend differential test of every
 * distribution-relevant kernel of SHUTTLE-128.
 *
 * argv[1] = random-bytes file (>= 64 KiB), argv[2] = output dump.
 * Compiled identically inside each backend directory (ref / avx2 / avx512),
 * so all namespace renames apply automatically via params.h -> config.h.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "params.h"
#include "sampler.h"
#include "sampler_u.h"
#include "approx_exp.h"
#include "approx_log.h"
#include "rcdt_tables.h"
#include "polyvec.h"
#include "drng.h"

/* KAT_SIG.c owns this in the real build; the harness must provide it. */
DRNG_ctx drng_algorithm;

#define BUF_SZ (1u << 20)
static uint8_t buf[BUF_SZ];

int main(int argc, char **argv)
{
    FILE *f, *o;
    size_t n, off;
    int32_t out[32];
    int i, x, y, k, s;

    if (argc != 3) { fprintf(stderr, "usage: %s rand.bin out.dump\n", argv[0]); return 1; }
    f = fopen(argv[1], "rb");
    if (!f) { perror("rand"); return 1; }
    n = fread(buf, 1, BUF_SZ, f);
    fclose(f);
    if (n < 65536) { fprintf(stderr, "rand too small\n"); return 1; }
    o = fopen(argv[2], "wb");
    if (!o) { perror("out"); return 1; }

    /* 1) base scans: wide table + every noise table over the stream */
    for (off = 0; off + 384 <= n; off += 384) {
        cdt_scan96(out, buf + off, SHUTTLE_RCDT_Z, 36, 32);
        fwrite(out, 4, 32, o);
    }
    for (off = 0; off + 384 <= n; off += 384) {
        cdt_scan96(out, buf + off, SHUTTLE_RCDT_NOISE_0_85, 9, 32);
        fwrite(out, 4, 32, o);
    }
    for (off = 0; off + 384 <= n; off += 384) {
        cdt_scan96(out, buf + off, SHUTTLE_RCDT_NOISE_0_90, 10, 32);
        fwrite(out, 4, 32, o);
    }
    for (off = 0; off + 384 <= n; off += 384) {
        cdt_scan96(out, buf + off, SHUTTLE_RCDT_NOISE_1_00, 11, 32);
        fwrite(out, 4, 32, o);
    }

    /* 2) ApproxExp full domain (9472 inputs), scalar kernel */
    for (x = 0; x <= 36; x++)
        for (y = 0; y < 256; y++) {
            uint64_t v = shuttle_exp_accept_poly_q64(x, y);
            fwrite(&v, 8, 1, o);
        }
    /* 2b) x4 batched kernel over a permuted input order */
    {
        int xs[4], ys[4];
        uint64_t vo[4];
        for (i = 0; i * 4 + 3 < 37 * 256; i++) {
            for (k = 0; k < 4; k++) {
                int t = i * 4 + k;
                xs[k] = t & 31;          /* permuted domain coverage */
                ys[k] = (t * 37) & 255;
            }
            shuttle_exp_accept_poly_q64_x4(xs, ys, vo);
            fwrite(vo, 8, 4, o);
        }
    }

    /* 3) ApproxLog grid: 4 segments x 1024 points + boundary patterns */
    for (k = 0; k < 4; k++) {
        int64_t v;
        uint64_t xq;
        for (i = 0; i < 1024; i++) {
            xq = ((uint64_t)i * ((UINT64_C(1) << 55) - 1) / 1023) << 9;
            v = shuttle_log2_frac_q62((uint32_t)k, xq);
            fwrite(&v, 8, 1, o);
        }
        xq = 0; v = shuttle_log2_frac_q62((uint32_t)k, xq); fwrite(&v, 8, 1, o);
        xq = ~UINT64_C(0); v = shuttle_log2_frac_q62((uint32_t)k, xq); fwrite(&v, 8, 1, o);
        xq = UINT64_C(1) << 63; v = shuttle_log2_frac_q62((uint32_t)k, xq); fwrite(&v, 8, 1, o);
    }

    /* 4) gauss_finalize grid */
    for (x = 0; x <= 36; x += 5)
        for (y = 0; y < 256; y += 17) {
            uint64_t p = shuttle_exp_accept_poly_q64(x, y);
            uint8_t tail[8];
            memcpy(tail, buf + ((x * 256 + y) * 8) % (n - 8), 8);
            for (s = 0; s < 2; s++) {
                int32_t zv;
                int keep = gauss_finalize(&zv, x, y, p, tail, (uint32_t)s);
                fwrite(&zv, 4, 1, o);
                fwrite(&keep, 4, 1, o);
            }
        }

    /* 5) sampler_u_decode over 18-byte blocks */
    for (off = 0; off + 18 <= n; off += 18) {
        sampler_u_res r = sampler_u_decode(buf + off, buf + off + 10);
        fwrite(&r.a, 4, 1, o);
        fwrite(&r.frac_q62, 8, 1, o);
    }

    /* 6) full wide sampler: sample_y over 4096 seeds -> 4096*7*256 coeffs */
    for (i = 0; i < 4096 && (size_t)(16 * (i + 1)) <= n; i++) {
        poly yy[KVEC];
        sample_y(yy, buf + 16 * (size_t)i);
        fwrite(yy, sizeof(poly), KVEC, o);
    }

    fclose(o);
    return 0;
}
