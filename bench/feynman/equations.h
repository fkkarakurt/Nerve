/* equations.h - the benchmark suite: 100 equations from the Feynman Lectures.
 *
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * This is the standard yardstick for symbolic regression. The set was
 * catalogued by Udrescu & Tegmark for AI Feynman (Sci. Adv. 6, eaay2631,
 * 2020) and is the accuracy track of SRBench (La Cava et al., NeurIPS 2021
 * Datasets & Benchmarks). Each entry carries the ground-truth formula, its
 * variables, and the interval each variable is sampled from.
 *
 * Sampling follows the AI Feynman convention: every variable is drawn
 * uniformly and independently from its stated interval, which is (1,5) unless
 * physics forces otherwise - a speed must stay below c, a denominator must
 * not straddle zero, an arcsine argument must stay inside [-1,1]. Every such
 * deviation is written into the table below rather than hidden in a script,
 * so the benchmark is reproducible from this file alone with no download.
 *
 * The engine is given nothing but the sampled columns and the target. It is
 * never told the operators, the constants, or how many variables actually
 * matter.
 */
#ifndef NERVE_BENCH_FEYNMAN_H
#define NERVE_BENCH_FEYNMAN_H

#define FEYN_MAXV 10

typedef struct {
    const char *id;                 /* Feynman Lectures reference           */
    const char *formula;            /* ground truth, for the report only    */
    int         nv;
    const char *var[FEYN_MAXV];
    double      lo [FEYN_MAXV];
    double      hi [FEYN_MAXV];
    double    (*f)(const double *v);
} feyn_eq;

extern const feyn_eq feyn_equations[];
extern const int     feyn_count;

#endif /* NERVE_BENCH_FEYNMAN_H */
