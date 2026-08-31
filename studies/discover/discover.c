/* discover.c - demo of the nerve_discover.h single-header library.
 *
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Three hidden laws, each handed to the engine as nothing but noisy numbers.
 * It gets no hint of the form, the operators, or the constants - only a table
 * of samples - and hands back the equation, plus the whole accuracy/complexity
 * front so you can see what each extra node actually bought.
 *
 * Build:  cc -O2 -std=c99 discover.c -o discover -lm      (then ./discover)
 */
#define NERVE_DISCOVER_IMPLEMENTATION
#include "../../nerve_discover.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NTRAIN 200
#define NTEST  200
#define MAXV     4

/* tiny deterministic RNG, just for generating the demo datasets */
static unsigned long long g = 20260718ULL;
static double urand(void){
    g ^= g >> 12; g ^= g << 25; g ^= g >> 27;
    return (double)((g * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}
static double rrange(double a, double b){ return a + (b - a) * urand(); }
static double gauss(void){
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

typedef struct {
    const char *title;      /* what the law is, for the reader only        */
    int         nv;
    const char *names[MAXV];
    double      lo[MAXV], hi[MAXV];
    double    (*f)(const double *v);
    double      noise;      /* relative noise added to y                   */
    unsigned int ops;
} target;

static double law_gravity (const double *v){ return v[0]*v[1] / (v[2]*v[2]); }
static double law_kepler  (const double *v){ return sqrt(v[0]*v[0]*v[0]); }
static double law_pendulum(const double *v){ return 6.2831853 * sqrt(v[0] / v[1]); }

int main(void){
    static const target targets[3] = {
        { "Newton:   F = m1*m2 / r^2",        3, {"m1","m2","r",0},
          {1,1,1,0}, {5,5,3,0}, law_gravity,  0.02, ND_OPS_ALGEBRA },
        { "Kepler:   T = a^(3/2)",            1, {"a",0,0,0},
          {0.5,0,0,0}, {4,0,0,0}, law_kepler, 0.02, ND_OPS_ALGEBRA },
        { "Pendulum: T = 2*pi*sqrt(L/g)",     2, {"L","g",0,0},
          {0.2,5,0,0}, {2.0,15,0,0}, law_pendulum, 0.01, ND_OPS_ALGEBRA }
    };
    int t;

    printf("nerve_discover - three hidden laws, recovered from samples alone\n");

    for (t = 0; t < 3; t++){
        const target *tg = &targets[t];
        double Xtr[NTRAIN * MAXV], ytr[NTRAIN];
        double Xte[NTEST  * MAXV], yte[NTEST];
        char *names[MAXV];
        nd_options o;
        nd_model  *m;
        const nd_expr *pick;
        int i, j;

        for (j = 0; j < tg->nv; j++) names[j] = (char *)tg->names[j];

        for (i = 0; i < NTRAIN + NTEST; i++){
            double v[MAXV];
            double *Xd = (i < NTRAIN) ? &Xtr[i * tg->nv] : &Xte[(i - NTRAIN) * tg->nv];
            double *yd = (i < NTRAIN) ? &ytr[i]          : &yte[i - NTRAIN];
            for (j = 0; j < tg->nv; j++){
                v[j] = rrange(tg->lo[j], tg->hi[j]);
                Xd[j] = v[j];
            }
            *yd = tg->f(v) * (1.0 + tg->noise * gauss());
        }

        o = nd_defaults();
        o.seed        = 20260831ULL + (unsigned long long)t;
        o.ops         = tg->ops;
        o.max_nodes   = 20;
        o.generations = 250;

        m = nd_fit(Xtr, ytr, NTRAIN, tg->nv, o);
        if (!m){ printf("  search failed\n"); continue; }

        printf("\n%s   [%d samples, %.0f%% noise]\n",
               tg->title, NTRAIN, tg->noise * 100.0);
        printf("  the accuracy/complexity front (R2 on 200 held-out samples):\n");
        for (i = 0; i < nd_count(m); i++){
            const nd_expr *e = nd_at(m, i);
            printf("    %2d nodes   R2 %8.5f   ", nd_complexity(e),
                   nd_r2(e, Xte, yte, NTEST, tg->nv));
            nd_print(e, names, stdout);
            putchar('\n');
        }

        pick = nd_knee(m, 0.05);
        printf("  -> the knee: the simplest equation worth its nodes\n     ");
        nd_print(pick, names, stdout);
        printf("\n     held-out R2 = %.5f\n", nd_r2(pick, Xte, yte, NTEST, tg->nv));
        nd_free(m);
    }
    return 0;
}
