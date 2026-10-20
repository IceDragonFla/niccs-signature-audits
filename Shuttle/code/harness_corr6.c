/* harness_corr6.c -- streaming-rand version of the spectral accumulator.
   usage: h_corr6 rand.bin NSIG out [roff0]
   randbuf is a 512MB sliding window refilled from the file as roff advances;
   file-absolute offsets; no seed reuse. Optional S0FILE/BASEKEY as in corr5. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "packing.h"
#include "rounding.h"
#include "symmetric.h"
#include "drng.h"
#include "SIG_AlgorithmInstance.h"

DRNG_ctx drng_algorithm;

#define PI 3.14159265358979323846
#define BUFSZ ((size_t)1 << 29)

static FILE *rf;
static uint8_t *randbuf;
static long wlo, whi;   /* file-absolute window [wlo,whi) */

static void fill_window(long pos)
{
    long start = pos;
    if (fseek(rf, start, SEEK_SET) != 0) { fprintf(stderr, "seek fail\n"); exit(1); }
    size_t got = fread(randbuf, 1, BUFSZ, rf);
    wlo = start; whi = start + (long)got;
}

static void fft256(double *re, double *im)
{
    static double cwr[128], cwi[128];
    static int init = 0;
    int i, j, len, half, k;
    if (!init) {
        for (i = 0; i < 128; ++i) { cwr[i] = cos(-2.0*PI*i/256.0); cwi[i] = sin(-2.0*PI*i/256.0); }
        init = 1;
    }
    for (i = 1, j = 0; i < 256; ++i) { int bit = 128; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
        if (i < j) { double t; t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; } }
    for (len = 2; len <= 256; len <<= 1) {
        half = len >> 1; int step = 256 / len;
        for (i = 0; i < 256; i += len)
            for (k = 0; k < half; ++k) {
                double wr = cwr[k*step], wi = cwi[k*step];
                double vr = re[i+k+half]*wr - im[i+k+half]*wi;
                double vi = re[i+k+half]*wi + im[i+k+half]*wr;
                re[i+k] += vr; im[i+k] += vi; re[i+k+half] -= vr; im[i+k+half] -= vi;
            }
    }
}

int main(int argc, char **argv)
{
    uint8_t pk[4200], skraw[3000];
    unsigned long long pklen, sklen, siglen;
    long NSIG, q;
    int k, kp, rtn;
    unsigned i;
    static double Pre[3][3][256], Pim[3][3][256];
    static double blk_sum[3], blk_sumsq[3];
    double zre[3][256], zim[3][256], z1re[4][256], z1im[4][256];
    long roff0 = (argc == 5) ? atol(argv[4]) : 0;
    long nfail = 0, naccum = 0;

    if (argc != 4 && argc != 5) { fprintf(stderr, "usage: %s rand NSIG out [roff0]\n", argv[0]); return 1; }
    NSIG = atol(argv[2]);
    rf = fopen(argv[1], "rb");
    if (!rf) { perror("rand"); return 1; }
    randbuf = malloc(BUFSZ);
    if (!randbuf) { fprintf(stderr, "oom\n"); return 1; }
    fill_window(0);
    fseek(rf, 0, SEEK_END);
    { long fsz = ftell(rf); if (roff0 + 64 > fsz) { fprintf(stderr, "roff0 beyond file\n"); return 1; } }

    {   /* key: from BASEKEY if given, else keygen from file head */
        const char* bk = getenv("BASEKEY");
        if (bk) {
            FILE* kf = fopen(bk, "rb");
            if (!kf || fread(pk,1,CRYPTO_PUBLICKEYBYTES,kf)!=CRYPTO_PUBLICKEYBYTES
                   || fread(skraw,1,CRYPTO_SECRETKEYBYTES,kf)!=CRYPTO_SECRETKEYBYTES) { fprintf(stderr,"basekey fail\n"); return 1; }
            fclose(kf); pklen=CRYPTO_PUBLICKEYBYTES; sklen=CRYPTO_SECRETKEYBYTES;
        } else {
            init_random_number(&drng_algorithm, randbuf, 55*8);
            rtn = sig_keygen(pk, &pklen, skraw, &sklen);
            if (rtn) { fprintf(stderr, "keygen %d\n", rtn); return 1; }
        }
        { char fn[512]; snprintf(fn, sizeof fn, "%s_key.bin", argv[3]);
          FILE* df = fopen(fn, "wb");
          fwrite(pk,1,(size_t)pklen,df); fwrite(skraw,1,(size_t)sklen,df);
          { uint8_t seedA2[SEEDBYTES], masterK2[CHALLENGESEEDBYTES], tr2[CHALLENGESEEDBYTES];
            poly b2[EM], s2[ELL], ep2[EM], skf[KVEC], skt2[KVEC]; unsigned kk,ii;
            if (unpack_sk(seedA2,b2,masterK2,tr2,s2,ep2,skraw)==0) {
                const char* s0f = getenv("S0FILE");
                if (s0f) { FILE* g=fopen(s0f,"rb"); int32_t buf[3*N]; unsigned w=0;
                  if (!g || fread(buf,4,3*N,g)!=3*N) { fprintf(stderr,"s0 fail\n"); return 1; }
                  fclose(g);
                  for (kk=0;kk<ELL;++kk) for (ii=0;ii<N;++ii) s2[kk].coeffs[ii]=buf[w++];
                  for (kk=0;kk<EM;++kk) for (ii=0;ii<N;++ii) ep2[kk].coeffs[ii]=buf[w++];
                  pack_sk(skraw,seedA2,b2,masterK2,tr2,s2,ep2); } }
            for (kk=0;kk<N;++kk) skf[0].coeffs[kk]=0;
            skf[0].coeffs[0]=1;
            for (ii=0;ii<ELL;++ii) skf[1+ii]=s2[ii];
            for (ii=0;ii<EM;++ii) skf[1+ELL+ii]=ep2[ii];
            stretch_s(skt2,skf);
            fwrite(s2,sizeof(poly),ELL,df); fwrite(ep2,sizeof(poly),EM,df); fwrite(skt2,sizeof(poly),KVEC,df); }
          fclose(df); }
    }

    { long roff = roff0 ? roff0 : (55 + 4096);
      if (roff + 64 >= whi) fill_window(roff);
      for (q = 0; q < NSIG; ++q) {
        uint8_t m[32], sig[CRYPTO_BYTES];
        uint8_t seedC[CHALLENGESEEDBYTES];
        poly z1[Z1LEN], hint[EM];
        if (roff + 87 + 64 >= whi) {
            fill_window(roff);
            if (roff + 64 >= whi) { fprintf(stderr, "rand exhausted at %ld\n", q); break; }
        }
        memcpy(m, randbuf + (roff - wlo), 32); roff += 32;
        init_random_number(&drng_algorithm, randbuf + (roff - wlo), 55*8); roff += 55;
        rtn = sig_sign(skraw, sklen, m, 32, sig, &siglen);
        naccum++;
        if (unpack_sig(seedC, z1, hint, sig) != 0) { fprintf(stderr, "unpack fail\n"); return 1; }
        for (k = 0; k < Z1LEN; ++k) {
            for (i = 0; i < 256; ++i) {
                double twist_re = cos(PI*i/256.0), twist_im = sin(PI*i/256.0);
                double v = (double)z1[k].coeffs[i];
                z1re[k][i] = v*twist_re; z1im[k][i] = v*twist_im;
            }
            fft256(z1re[k], z1im[k]);
        }
        for (k = 0; k < 3; ++k) { memcpy(zre[k], z1re[k+1], sizeof(double)*256); memcpy(zim[k], z1im[k+1], sizeof(double)*256); }
        for (k = 0; k < 3; ++k) for (kp = 0; kp < 3; ++kp) for (i = 0; i < 256; ++i)
            Pre[k][kp][i] += zre[k][i]*zre[kp][i] + zim[k][i]*zim[kp][i],
            Pim[k][kp][i] += zim[k][i]*zre[kp][i] - zre[k][i]*zim[kp][i];
        for (k = 0; k < 3; ++k) for (i = 0; i < 256; ++i) {
            double v = (double)z1[k+1].coeffs[i]; blk_sum[k] += v; blk_sumsq[k] += v*v; }
        if ((q & 0x3FFFF) == 0) { fprintf(stderr, "%ld\n", q); fflush(stderr); }
      }
    }
    fprintf(stderr, "sign failures: %ld / %ld\n", nfail, NSIG);
    { FILE* o = fopen(argv[3], "wb");
      fwrite(&NSIG, sizeof(long), 1, o);
      fwrite(Pre, sizeof(double), 3*3*256, o);
      fwrite(Pim, sizeof(double), 3*3*256, o);
      fwrite(blk_sum, sizeof(double), 3, o);
      fwrite(blk_sumsq, sizeof(double), 3, o);
      fclose(o); }
    return 0;
}
