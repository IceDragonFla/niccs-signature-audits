/* Simulated annealing on the cross-Gram cost ||M - C3*SH SH^dagger||^2.
   usage: sa init.npy... actually binary input: S init (3*256 int32), Mx (3*3*256*2 float64), out best S */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#define N 256
#define PI 3.14159265358979323846
static double Mr[3][3][N], Mi[3][3][N];
static double C3=62.95;
static int S[3][N];
static double SHr[3][N], SHi[3][N];
static uint64_t rs=88172645463325252ULL;
static inline uint64_t rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static inline double rnd01(void){ return (rnd()>>11)*(1.0/9007199254740992.0); }
static void fft256(double*re,double*im,int inv){
    int i,j,len,half,k;
    for(i=1,j=0;i<N;i++){int bit=N>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;
        if(i<j){double t=re[i];re[i]=re[j];re[j]=t;t=im[i];im[i]=im[j];im[j]=t;}}
    for(len=2;len<=N;len<<=1){half=len>>1;
        for(i=0;i<N;i+=len)for(k=0;k<half;k++){
            double ang=(inv?2.0:-2.0)*PI*k/len;
            double wr=cos(ang),wi=sin(ang);
            double vr=re[i+k+half]*wr-im[i+k+half]*wi, vi=re[i+k+half]*wi+im[i+k+half]*wr;
            re[i+k]+=vr; im[i+k]+=vi; re[i+k+half]-=vr; im[i+k+half]-=vi;}}
}
static void recompute_SH(int k){
    double re[N],im[N]; int i;
    for(i=0;i<N;i++){ double tw_i=PI*i/N; re[i]=S[k][i]*cos(tw_i); im[i]=S[k][i]*sin(tw_i);}
    fft256(re,im,0);
    memcpy(SHr[k],re,sizeof(re)); memcpy(SHi[k],im,sizeof(im));
}
static double total_cost(void){
    double c=0; int k,kp,m;
    for(k=0;k<3;k++)for(kp=k+1;kp<3;kp++)for(m=0;m<N;m++){
        double gr=SHr[k][m]*SHr[kp][m]+SHi[k][m]*SHi[kp][m];
        double gi=SHi[k][m]*SHr[kp][m]-SHr[k][m]*SHi[kp][m];
        double dr=Mr[k][kp][m]-C3*gr, di=Mi[k][kp][m]-C3*gi;
        c+=dr*dr+di*di;
    }
    return c;
}
/* incremental: cost of pairs involving block k, with SH[k] replaced by (SH[k]+d*bas) */
static double pair_cost_k(int k, int i, double d){
    double re[N],im[N]; int m,kp;
    double twr=cos(PI*i/N), twi=sin(PI*i/N);
    /* basis vector: e_i * tw_i in time, fft -> tw_i * exp(-2 pi i m i /N) */
    double c=0;
    double nr[N],ni[N];
    memcpy(nr,SHr[k],sizeof(nr)); memcpy(ni,SHi[k],sizeof(ni));
    for(m=0;m<N;m++){ double ang=-2.0*PI*m*i/N; double br=twr*cos(ang)-twi*sin(ang), bi=twr*sin(ang)+twi*cos(ang);
        nr[m]+=d*br; ni[m]+=d*bi; }
    for(kp=0;kp<3;kp++){
        if(kp==k) continue;
        for(m=0;m<N;m++){
            double gr=nr[m]*SHr[kp][m]+ni[m]*SHi[kp][m];
            double gi=ni[m]*SHr[kp][m]-nr[m]*SHi[kp][m];
            double dr=Mr[k<kp?k:kp][k<kp?kp:k][m]-C3*gr, di=Mi[k<kp?k:kp][k<kp?kp:k][m]-C3*gi;
            c+=dr*dr+di*di;
        }
    }
    return c;
}
int main(int argc,char**argv){
    /* inputs: argv[1]=initS.bin (3*256 int32), argv[2]=Mx.bin (3*3*N*2 f64), argv[3]=out.bin, argv[4]=iters, argv[5]=seed */
    FILE*f=fopen(argv[1],"rb"); int32_t tmp[3*N]; fread(tmp,4,3*N,f); fclose(f);
    for(int k=0;k<3;k++)for(int i=0;i<N;i++) S[k][i]=tmp[k*N+i];
    f=fopen(argv[2],"rb"); double buf[3*3*N*2]; fread(buf,8,3*3*N*2,f); fclose(f);
    for(int k=0;k<3;k++)for(int kp=0;kp<3;kp++)for(int m=0;m<N;m++){Mr[k][kp][m]=buf[(k*3+kp)*N*2+2*m];Mi[k][kp][m]=buf[(k*3+kp)*N*2+2*m+1];}
    long long iters=atoll(argv[4]); rs=0x9E3779B97F4A7C15ULL^(uint64_t)atoll(argv[5]);
    for(int k=0;k<3;k++) recompute_SH(k);
    double cur=total_cost(), best=cur;
    int bestS[3][N]; memcpy(bestS,S,sizeof(S));
    double T0=3e8, T1=1e4;
    double T=T0;
    long long accept=0;
    for(long long it=0; it<iters; it++){
        T=T0*pow(T1/T0,(double)it/iters);
        int k=rnd()%3, i=rnd()%N;
        int dv;
        int r=rnd()%10;
        if(r<6) dv=(rnd()%2?1:-1);
        else if(r<9) dv=(int)(rnd()%5)-2;
        else dv=-(S[k][i]);
        int nv=S[k][i]+dv;
        if(nv<-3||nv>3||dv==0) continue;
        double before=pair_cost_k(k,i,0.0), after=pair_cost_k(k,i,(double)(nv-S[k][i]));
        double dc=after-before;
        if(dc<0 || rnd01()<exp(-dc/T)){
            S[k][i]=nv; accept++;
            /* update SH[k] incrementally */
            double d=(double)dv;
            for(int m=0;m<N;m++){ double ang=-2.0*PI*m*i/N; double twr=cos(PI*i/N), twi=sin(PI*i/N);
                double br=twr*cos(ang)-twi*sin(ang), bi=twr*sin(ang)+twi*cos(ang);
                SHr[k][m]+=d*br; SHi[k][m]+=d*bi; }
            cur+=dc;
            if(cur<best){ best=cur; memcpy(bestS,S,sizeof(S)); }
        }
    }
    f=fopen(argv[3],"wb");
    int32_t out[3*N];
    for(int k=0;k<3;k++)for(int i=0;i<N;i++) out[k*N+i]=bestS[k][i];
    fwrite(out,4,3*N,f); fclose(f);
    fprintf(stderr,"best=%.6e accept_rate=%.3f\n",best,(double)accept/iters);
    printf("%.6e\n",best);
    return 0;
}
