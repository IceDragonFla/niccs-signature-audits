/* isolate: linearity of the reference product chain */
#include "sign.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "polymat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long center_q(long v){ v%=Q; if(v<0)v+=Q; if(v>Q/2)v-=Q; return v; }
static long r_col0[N];
static long col[2][N][N];

static void prod(const polyvecl A[], const polyvecl *sh, polyveck *res){
    polymatkl_pointwise_montgomery(res, A, sh);
    polyveck_invntt_tomont(res);
    polyveck_freeze(res);
}

int main(int argc, char**argv){
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    polyvecl A[K]; polyveck A0v; uint8_t seedA1[SEEDBYTES];
    static int8_t truth[3*N];
    FILE*f=fopen(argv[1],"rb"); if(fread(pk,1,sizeof(pk),f)!=sizeof(pk))return 1; fclose(f);
    f=fopen(argv[2],"rb"); if(fread(truth,1,sizeof(truth),f)!=sizeof(truth))return 1; fclose(f);
    unpack_pk(&A0v,seedA1,pk);
    polymatkl_expand(A,seedA1);
    for(unsigned k=0;k<1;++k) A[k].vec[0]=A0v.vec[k];

    poly s0; polyvecl_1 s1v;
    for(int i=0;i<N;++i) s0.coeffs[i]=truth[i];
    for(int i=0;i<N;++i) s1v.vec[0].coeffs[i]=truth[N+i];
    printf("s_true[0..5]: %d %d %d %d %d %d | e_true[0..5]: %d %d %d %d %d %d\n",
        truth[0],truth[1],truth[2],truth[3],truth[4],truth[5],
        truth[2*N],truth[2*N+1],truth[2*N+2],truth[2*N+3],truth[2*N+4],truth[2*N+5]);

    /* unit columns through the chain */
    for(int c=0;c<2;++c) for(int m=0;m<4;++m){
        polyvecl shu; polyveck resu;
        memset(&shu,0,sizeof(shu));
        shu.vec[c].coeffs[m]=1;
        poly_ntt(&shu.vec[c]);
        prod(A,&shu,&resu);
        for(int i=0;i<N;++i) col[c][m][i]=resu.vec[0].coeffs[i];
    }
    printf("col[0][0][0..3]: %ld %ld %ld %ld\n", col[0][0][0],col[0][0][1],col[0][0][2],col[0][0][3]);
    printf("col[0][1][0..3]: %ld %ld %ld %ld\n", col[0][1][0],col[0][1][1],col[0][1][2],col[0][1][3]);
    printf("col[1][0][0..3]: %ld %ld %ld %ld\n", col[1][0][0],col[1][0][1],col[1][0][2],col[1][0][3]);

    /* one-shot with full true s */
    polyvecl sh; polyveck res;
    memset(&sh,0,sizeof(sh));
    poly s0h=s0; poly_ntt(&s0h); sh.vec[0]=s0h;
    poly s1h=s1v.vec[0]; poly_ntt(&s1h); sh.vec[1]=s1h;
    prod(A,&sh,&res);
    printf("oneshot res[0..5]: %ld %ld %ld %ld %ld %ld\n",
        res.vec[0].coeffs[0],res.vec[0].coeffs[1],res.vec[0].coeffs[2],
        res.vec[0].coeffs[3],res.vec[0].coeffs[4],res.vec[0].coeffs[5]);

    /* linearity check: one-shot of (s0 only) vs sum_m s0[m]*col[0][m] */
    polyvecl sh0; polyveck res0;
    memset(&sh0,0,sizeof(sh0));
    poly s0h2=s0; poly_ntt(&s0h2); sh0.vec[0]=s0h2;
    prod(A,&sh0,&res0);
    long acc[N]; for(int i=0;i<N;++i) acc[i]=0;
    for(int m=0;m<N;++m){
        long sv=truth[m]; if(!sv) continue;
        for(int i=0;i<N;++i) acc[i]+=sv*col[0][m][i];
    }
    long nlin=0;
    for(int i=0;i<N;++i){
        long d=center_q(res0.vec[0].coeffs[i]-acc[i]);
        if(d!=0) nlin++;
        if(i<5) printf("i=%d oneshot0=%ld colsum=%ld diff=%ld\n", i, res0.vec[0].coeffs[i], center_q(acc[i]), d);
    }
    printf("s0-only: nonlinear coords = %ld/512\n", nlin);

    /* identity check via column sum with full s */
    for(int c=0;c<2;++c) for(int m=0;m<N;++m){
        long sv=(c==0)?truth[m]:truth[N+m]; if(!sv) continue;
        for(int i=0;i<N;++i) acc[i]+= (c==1? 0:0); /* placeholder */
    }
    /* recompute cleanly: acc already has s0 part; add s1 part */
    for(int m=0;m<N;++m){
        long sv=truth[N+m]; if(!sv) continue;
        for(int i=0;i<N;++i) acc[i]+=sv*col[1][m][i];
    }
    long bad2=0;
    for(int i=0;i<N;++i){
        long e=center_q((i==0?(long)HALF_Q:0)-acc[i]);
        if(e>1||e<-1) bad2++;
        if(i<6) printf("cols e[%d]=%ld ", i, e);
    }
    printf("| column identity non-ternary=%ld/512\n", bad2);
    (void)r_col0;
    return 0;
}
