/* discover.c - demo of the nerve_discover.h single-header library.
 *
 * We hand the engine only noisy (x, y) samples from a hidden law and it hands
 * back the equation. Proof that "give it data, get an equation" is real.
 *
 * Build:  cc -O2 -std=c99 discover.c -o discover -lm      (then ./discover)
 */
#define NERVE_DISCOVER_IMPLEMENTATION
#include "../../nerve_discover.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* tiny deterministic RNG just for generating the demo datasets */
static unsigned long long g = 20260718ULL;
static double urand(void){ g ^= g>>12; g ^= g<<25; g ^= g>>27; return (double)((g*0x2545F4914F6CDD1DULL)>>11)*(1.0/9007199254740992.0); }
static double rrange(double a, double b){ return a + (b-a)*urand(); }
static double noise(void){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12; return sqrt(-2.0*log(u1))*cos(6.283185307179586*u2); }

int main(void){
    int i;

    /* ---- Target 1: the inverse-square law  F = m1*m2 / r^2  (2% noise) ---- */
    {
        enum { N = 160, NV = 3 };
        double X[N*NV], y[N];
        char *names[NV]; nd_expr *e; nd_options o;
        names[0] = "m1"; names[1] = "m2"; names[2] = "r";
        for (i = 0; i < N; i++){
            double m1 = rrange(1,5), m2 = rrange(1,5), r = rrange(1,3);
            X[i*NV+0] = m1; X[i*NV+1] = m2; X[i*NV+2] = r;
            y[i] = (m1*m2/(r*r)) * (1.0 + 0.02*noise());
        }
        o = nd_defaults(); o.seed = 424242; o.max_nodes = 20;
        e = nd_fit(X, y, N, NV, o);
        printf("Hidden law:  F = m1*m2 / r^2     [160 samples, 2%% noise]\n  ");
        nd_print(e, names, stdout);
        printf("\n  R2 = %.5f,  %d nodes\n\n", nd_r2(e, X, y, N, NV), nd_num_nodes(e));
        nd_free(e);
    }

    /* ---- Target 2: Kepler's third law  T = sqrt(a^3)  (2% noise) --------- */
    {
        enum { N = 120, NV = 1 };
        double X[N*NV], y[N];
        char *names[NV]; nd_expr *e; nd_options o;
        names[0] = "a";
        for (i = 0; i < N; i++){
            double a = rrange(0.5, 4.0);
            X[i] = a;
            y[i] = sqrt(a*a*a) * (1.0 + 0.02*noise());
        }
        o = nd_defaults(); o.seed = 777; o.max_nodes = 16;
        e = nd_fit(X, y, N, NV, o);
        printf("Hidden law:  T = sqrt(a^3)       [120 samples, 2%% noise]\n  ");
        nd_print(e, names, stdout);
        printf("\n  R2 = %.5f,  %d nodes\n", nd_r2(e, X, y, N, NV), nd_num_nodes(e));
        nd_free(e);
    }
    return 0;
}
