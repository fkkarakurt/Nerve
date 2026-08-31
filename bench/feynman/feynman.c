/* feynman.c - run nerve_discover.h against the 100 Feynman equations.
 *
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * The benchmark that symbolic regression is measured by. For each equation the
 * engine is handed a table of sampled inputs and one output column, and gets
 * back nothing else: not the operators that built it, not the constants, not
 * which of the nine columns actually matter. It then has to write the formula.
 *
 * PROTOCOL
 *   - 300 training rows and 300 held-out rows per equation, variables drawn
 *     uniformly and independently from the intervals in equations.c.
 *   - Noise-free targets, as in the reference benchmark. What is being
 *     measured is whether the search finds the right closed form at all.
 *   - One run per equation, with a fixed per-equation seed. No restarts, no
 *     per-equation tuning: every equation gets the identical budget and the
 *     identical operator set.
 *   - The answer scored is the knee of the Pareto front - the simplest
 *     equation that concedes at most 1% of the accuracy the front spans -
 *     never the largest one. An engine that reports its most over-fitted
 *     candidate is grading its own homework.
 *   - SOLVED means held-out R^2 >= 0.999, the accuracy criterion used by
 *     SRBench (La Cava et al. 2021). R^2 is computed on data the search
 *     never saw.
 *
 * Build:  cc -O2 -std=c99 equations.c feynman.c -o feynman -lm
 * Run:    ./feynman              full 100-equation sweep
 *         ./feynman --quick      a 12-equation smoke test
 *         ./feynman --only I.12.2
 *         ./feynman --csv results.csv
 */
#define NERVE_DISCOVER_IMPLEMENTATION
#include "../../nerve_discover.h"
#include "equations.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NTRAIN 300
#define NTEST  300

/* The operator set the engine is allowed. Stated here rather than left to a
 * default, because it is part of the result: arcsine is NOT in it, so the two
 * equations that need one are expected to fail, and are still counted. */
#define BENCH_OPS (ND_OPS_DEFAULT | ND_OP_TANH)

/* deterministic sampler, independent of the engine's own RNG */
static unsigned long long rs = 1;
static double runit(void){
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return (double)((rs * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

static int finite_ok(double v){
    return (v == v) && v < 1e250 && v > -1e250;
}

/* Fill n rows of X and y for one equation. Rows whose target is not finite
 * are redrawn; a handful of intervals can still produce a singular draw. */
static int sample(const feyn_eq *eq, double *X, double *y, int n){
    int i, guard;
    for (i = 0; i < n; i++){
        for (guard = 0; guard < 100; guard++){
            double v[FEYN_MAXV];
            int j;
            for (j = 0; j < eq->nv; j++)
                v[j] = eq->lo[j] + (eq->hi[j] - eq->lo[j]) * runit();
            y[i] = eq->f(v);
            if (finite_ok(y[i])){
                for (j = 0; j < eq->nv; j++) X[(size_t)i * eq->nv + j] = v[j];
                break;
            }
        }
        if (guard >= 100) return 0;
    }
    return 1;
}

int main(int argc, char **argv){
    /* a spread across all three volumes and 1..9 variables, for --quick */
    static const char *quick[] = {
        "I.6.2a", "I.12.1", "I.12.2", "I.14.4", "I.25.13", "I.29.4",
        "I.34.27", "I.39.10", "II.3.24", "II.27.18", "II.38.14", "III.12.43"
    };
    const int nquick = (int)(sizeof quick / sizeof quick[0]);

    int    quick_mode = 0, show_front = 0, i, k, solved = 0, solved99 = 0, attempted = 0;
    const char *only = NULL, *csv_path = NULL;
    unsigned long long base_seed = 20260831ULL;
    FILE  *csv = NULL;
    double total_time = 0.0, r2_sum = 0.0;
    double *r2_all;

    for (i = 1; i < argc; i++){
        if      (!strcmp(argv[i], "--quick")) quick_mode = 1;
        else if (!strcmp(argv[i], "--front")) show_front = 1;
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--csv")  && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            base_seed = strtoull(argv[++i], NULL, 10);
        else {
            fprintf(stderr, "usage: %s [--quick] [--only ID] [--front] "
                            "[--csv FILE] [--seed N]\n", argv[0]);
            return 1;
        }
    }

    r2_all = (double *)malloc(sizeof(double) * (size_t)feyn_count);
    if (!r2_all) return 1;

    if (csv_path){
        csv = fopen(csv_path, "w");
        if (!csv){ fprintf(stderr, "cannot write %s\n", csv_path); free(r2_all); return 1; }
        fprintf(csv, "id,nvars,nodes,r2_test,seconds,equation\n");
    }

    printf("nerve_discover on the Feynman equations\n");
    printf("  %d train / %d held-out rows, noise-free, one run each, "
           "identical budget\n", NTRAIN, NTEST);
    printf("  solved = held-out R2 >= 0.999\n\n");
    printf("  %-11s %4s %6s %11s %8s  %s\n",
           "equation", "vars", "nodes", "R2 (test)", "sec", "recovered form");
    printf("  ---------------------------------------------------------------"
           "----------------\n");

    for (k = 0; k < feyn_count; k++){
        const feyn_eq *eq = &feyn_equations[k];
        double *Xtr, *ytr, *Xte, *yte;
        char   *names[FEYN_MAXV];
        char    text[2048];
        nd_options o;
        nd_model  *m;
        const nd_expr *pick;
        clock_t t0;
        double  secs, r2;

        if (only && strcmp(only, eq->id)) continue;
        if (quick_mode && !only){
            int hit = 0;
            for (i = 0; i < nquick; i++) if (!strcmp(quick[i], eq->id)) hit = 1;
            if (!hit) continue;
        }

        Xtr = (double *)malloc(sizeof(double) * (size_t)NTRAIN * eq->nv);
        Xte = (double *)malloc(sizeof(double) * (size_t)NTEST  * eq->nv);
        ytr = (double *)malloc(sizeof(double) * NTRAIN);
        yte = (double *)malloc(sizeof(double) * NTEST);
        if (!Xtr || !Xte || !ytr || !yte){
            free(Xtr); free(Xte); free(ytr); free(yte);
            fprintf(stderr, "out of memory on %s\n", eq->id);
            break;
        }

        rs = base_seed + (unsigned long long)k * 7919ULL;
        if (!sample(eq, Xtr, ytr, NTRAIN) || !sample(eq, Xte, yte, NTEST)){
            printf("  %-11s  (could not draw a finite sample)\n", eq->id);
            free(Xtr); free(Xte); free(ytr); free(yte);
            continue;
        }
        for (i = 0; i < eq->nv; i++) names[i] = (char *)eq->var[i];

        o = nd_defaults();
        o.seed        = base_seed + (unsigned long long)k;
        o.ops         = BENCH_OPS;
        o.population  = quick_mode ? 300 : 400;
        o.islands     = 5;
        o.generations = quick_mode ? 150 : 400;
        o.max_nodes   = 26;
        o.tune_iters  = 60;

        t0 = clock();
        m  = nd_fit(Xtr, ytr, NTRAIN, eq->nv, o);
        secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        total_time += secs;

        if (!m){
            printf("  %-11s  (search failed)\n", eq->id);
            free(Xtr); free(Xte); free(ytr); free(yte);
            continue;
        }

        /* the knee, not the biggest candidate on the front */
        pick = nd_knee(m, 0.01);
        r2   = nd_r2(pick, Xte, yte, NTEST, eq->nv);
        if (!finite_ok(r2)) r2 = -1.0;
        if (r2 < -1.0) r2 = -1.0;

        nd_format(pick, names, text, (int)sizeof text);

        attempted++;
        r2_all[attempted - 1] = r2;
        r2_sum += r2;
        if (r2 >= 0.999) solved++;
        if (r2 >= 0.99)  solved99++;

        printf("  %-11s %4d %6d %11.6f %8.1f  %s%s\n",
               eq->id, eq->nv, nd_complexity(pick), r2, secs,
               r2 >= 0.999 ? "" : "[miss] ", text);

        if (show_front){
            printf("      ground truth: %s\n", eq->formula);
            for (i = 0; i < nd_count(m); i++){
                const nd_expr *e = nd_at(m, i);
                nd_format(e, names, text, (int)sizeof text);
                printf("      %3d nodes  R2 %11.8f  %s\n", nd_complexity(e),
                       nd_r2(e, Xte, yte, NTEST, eq->nv), text);
            }
        }
        fflush(stdout);

        if (csv)
            fprintf(csv, "%s,%d,%d,%.8f,%.2f,\"%s\"\n",
                    eq->id, eq->nv, nd_complexity(pick), r2, secs, text);

        nd_free(m);
        free(Xtr); free(Xte); free(ytr); free(yte);
    }

    if (attempted > 0){
        double med;
        /* median R^2, by insertion sort - `attempted` is at most 100 */
        for (i = 1; i < attempted; i++){
            double key = r2_all[i];
            int j = i - 1;
            while (j >= 0 && r2_all[j] > key){ r2_all[j+1] = r2_all[j]; j--; }
            r2_all[j+1] = key;
        }
        med = (attempted & 1) ? r2_all[attempted/2]
                              : 0.5*(r2_all[attempted/2 - 1] + r2_all[attempted/2]);

        printf("\n  ---------------------------------------------------------"
               "----------------------\n");
        printf("  solved  (R2 >= 0.999) : %d / %d   (%.0f%%)\n",
               solved, attempted, 100.0 * solved / attempted);
        printf("  close   (R2 >= 0.99)  : %d / %d   (%.0f%%)\n",
               solved99, attempted, 100.0 * solved99 / attempted);
        printf("  median R2             : %.6f\n", med);
        printf("  mean   R2             : %.6f\n", r2_sum / attempted);
        printf("  total search time     : %.1f s  (%.1f s per equation)\n",
               total_time, total_time / attempted);
    }

    if (csv){ fclose(csv); printf("  results written to %s\n", csv_path); }
    free(r2_all);
    return 0;
}
