/*
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Nerve — nerve_discover.h test suite
 * ===========================================================================
 * Zero dependencies, one translation unit, no test framework.
 *
 *     gcc -O2 -std=c99 -Wall -Wextra tests/test_discover.c -o test_discover -lm
 *     ./test_discover
 *
 * A search engine is easy to test loosely and hard to test honestly. These
 * checks avoid asserting on any particular equation the search happens to
 * find today — that would break on any improvement to the search — and pin
 * down the things that must hold for every run instead: determinism, the
 * structural invariants of the Pareto front, agreement between what the
 * engine reports and what it evaluates, and clean behaviour on degenerate
 * input. The one accuracy assertion is a law simple enough that failing it
 * would mean the engine is broken, not unlucky.
 *
 * Every allocation path is exercised so the sanitizer build has something to
 * find if a tree is ever leaked or freed twice.
 */

#define NERVE_DISCOVER_IMPLEMENTATION
#include "../nerve_discover.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Tiny assertion harness ─────────────────────────────────────────────── */

static int g_checks = 0;
static int g_failed = 0;

static void begin(const char *name)
{
    printf("  %-52s", name);
    fflush(stdout);
}

static void end(void)
{
    printf("ok\n");
}

static void check(int cond, const char *what)
{
    g_checks++;
    if (!cond) {
        g_failed++;
        printf("\n    FAIL: %s\n", what);
    }
}

static void check_near(double got, double want, double tol, const char *what)
{
    g_checks++;
    if (!(fabs(got - want) <= tol)) {
        g_failed++;
        printf("\n    FAIL: %s (got %.10g, want %.10g +- %g)\n",
               what, got, want, tol);
    }
}

/* ── A deterministic sampler, independent of the engine's own RNG ───────── */

static unsigned long long g_rs = 12345;

static double urand(void)
{
    g_rs ^= g_rs >> 12; g_rs ^= g_rs << 25; g_rs ^= g_rs >> 27;
    return (double)((g_rs * 0x2545F4914F6CDD1DULL) >> 11)
           * (1.0 / 9007199254740992.0);
}

#define NROWS 120

/* y = 3*x0*x1 - the smallest law with a product in it. */
static void make_product(double *X, double *y, int n)
{
    int i;
    g_rs = 12345;
    for (i = 0; i < n; i++) {
        double a = 1.0 + 4.0 * urand();
        double b = 1.0 + 4.0 * urand();
        X[i * 2 + 0] = a;
        X[i * 2 + 1] = b;
        y[i] = 3.0 * a * b;
    }
}

/* ── Determinism ────────────────────────────────────────────────────────── */

static void test_same_seed_same_answer(void)
{
    double X[NROWS * 2], y[NROWS];
    char a[1024], b[1024];
    nd_options o;
    nd_model *m1, *m2;

    begin("same seed reproduces the same equation");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 99; o.population = 120; o.islands = 3; o.generations = 40;

    m1 = nd_fit(X, y, NROWS, 2, o);
    m2 = nd_fit(X, y, NROWS, 2, o);
    check(m1 != NULL && m2 != NULL, "both searches returned a model");
    if (m1 && m2) {
        check(nd_count(m1) == nd_count(m2), "identical front length");
        nd_format(nd_best(m1), NULL, a, (int)sizeof a);
        nd_format(nd_best(m2), NULL, b, (int)sizeof b);
        check(strcmp(a, b) == 0, "identical best equation");
    }
    nd_free(m1); nd_free(m2);
    end();
}

static void test_different_seed_is_allowed_to_differ(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;

    begin("a different seed still produces a valid model");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 4242; o.population = 120; o.islands = 3; o.generations = 40;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    check(nd_count(m) > 0, "front is not empty");
    nd_free(m);
    end();
}

/* ── The front's structural invariants ──────────────────────────────────── */

static void test_front_is_a_front(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;
    int i;

    begin("front ascends in size and improves in accuracy");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 7; o.population = 200; o.islands = 4; o.generations = 60;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        double prev_r2 = -2.0;
        int prev_nodes = 0;
        check(nd_count(m) > 0, "front is not empty");
        for (i = 0; i < nd_count(m); i++) {
            const nd_expr *e = nd_at(m, i);
            double r2 = nd_r2(e, X, y, NROWS, 2);
            check(e != NULL, "entry is present");
            check(nd_complexity(e) > prev_nodes, "strictly larger than the last");
            check(r2 > prev_r2 - 1e-9, "no less accurate than the last");
            check(nd_complexity(e) <= o.max_nodes, "within the node ceiling");
            prev_nodes = nd_complexity(e);
            prev_r2 = r2;
        }
        check(nd_at(m, -1) == NULL, "out-of-range index is NULL");
        check(nd_at(m, nd_count(m)) == NULL, "one past the end is NULL");
        check(nd_best(m) == nd_at(m, nd_count(m) - 1), "best is the last entry");
    }
    nd_free(m);
    end();
}

static void test_knee_is_no_larger_than_best(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;

    begin("the knee is never bigger than the most accurate entry");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 11; o.population = 200; o.islands = 4; o.generations = 60;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        const nd_expr *knee = nd_knee(m, 0.05);
        const nd_expr *best = nd_best(m);
        check(knee != NULL, "knee exists");
        check(nd_complexity(knee) <= nd_complexity(best),
              "knee is no more complex than best");
        /* zero slack must land on the most accurate entry */
        check(nd_knee(m, 0.0) != NULL, "zero-slack knee exists");
        /* an absurd slack must land on the simplest entry */
        check(nd_knee(m, 1e9) == nd_at(m, 0), "huge slack gives the simplest");
    }
    nd_free(m);
    end();
}

/* ── The engine agrees with itself ──────────────────────────────────────── */

static void test_r2_matches_manual_computation(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;

    begin("nd_r2 agrees with R2 computed from nd_eval");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 3; o.population = 150; o.islands = 3; o.generations = 40;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        const nd_expr *e = nd_best(m);
        double mean = 0.0, ssr = 0.0, sst = 0.0, manual;
        int i;
        for (i = 0; i < NROWS; i++) mean += y[i];
        mean /= NROWS;
        for (i = 0; i < NROWS; i++) {
            double d = nd_eval(e, &X[i * 2]) - y[i];
            double t = y[i] - mean;
            ssr += d * d; sst += t * t;
        }
        manual = 1.0 - ssr / sst;
        check_near(nd_r2(e, X, y, NROWS, 2), manual, 1e-9, "R2 agrees");
        check_near(nd_rmse(e, X, y, NROWS, 2), sqrt(ssr / NROWS), 1e-9,
                   "RMSE agrees");
    }
    nd_free(m);
    end();
}

static void test_format_is_bounded(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;

    begin("nd_format never writes past its buffer");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 5; o.population = 150; o.islands = 3; o.generations = 40;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        char small[8];
        char full[2048];
        int want, got;
        memset(small, 0x7f, sizeof small);
        want = nd_format(nd_best(m), NULL, full, (int)sizeof full);
        got  = nd_format(nd_best(m), NULL, small, (int)sizeof small);
        check(want == got, "returned length does not depend on capacity");
        check(small[sizeof small - 1] == '\0', "truncated output stays terminated");
        check(strlen(small) < sizeof small, "no overflow of the small buffer");
        check((int)strlen(full) == want || want >= (int)sizeof full,
              "full buffer holds the whole equation");
        /* a NULL expression must be handled, not dereferenced */
        check(nd_format(NULL, NULL, full, (int)sizeof full) == 0,
              "NULL expression formats as empty");
        check(full[0] == '\0', "NULL expression terminates the buffer");
    }
    nd_free(m);
    end();
}

static void test_printed_equations_are_unambiguous(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;
    char text[4096];
    int i;

    begin("printed equations bracket their reciprocals");
    make_product(X, y, NROWS);

    /* Reciprocals are the trap. A node printed as "1/x" instead of "(1/x)"
     * evaluates correctly and reads wrong the moment it lands on the right of
     * a division: a/1/x is (a/1)/x, which is not a/(1/x). Any equation this
     * library prints has to survive being read by a person. */
    o = nd_defaults();
    o.seed = 4242;
    o.ops = ND_OPS_ARITH | ND_OP_INV | ND_OP_SQR | ND_OP_NEG;
    o.population = 250; o.islands = 4; o.generations = 80;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        for (i = 0; i < nd_count(m); i++) {
            const char *p;
            int depth = 0, balanced = 1, naked = 0;
            nd_format(nd_at(m, i), NULL, text, (int)sizeof text);
            for (p = text; *p; p++) {
                if (*p == '(') depth++;
                else if (*p == ')') { depth--; if (depth < 0) balanced = 0; }
                /* "1/" must always open with a bracket right before it */
                if (p[0] == '1' && p[1] == '/' && (p == text || p[-1] != '('))
                    naked = 1;
            }
            check(balanced && depth == 0, "parentheses are balanced");
            check(!naked, "every reciprocal is bracketed");
        }
    }
    nd_free(m);
    end();
}

/* ── It can actually find a law ─────────────────────────────────────────── */

static void test_recovers_a_product_law(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;

    begin("recovers y = 3*x0*x1 exactly");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 20260831ULL;
    o.ops = ND_OPS_ARITH;
    o.population = 300; o.islands = 4; o.generations = 120;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        const nd_expr *knee = nd_knee(m, 0.01);
        check(nd_r2(knee, X, y, NROWS, 2) > 0.9999, "knee fits the data");
        check(nd_complexity(knee) <= 5, "and does it in at most five nodes");
    }
    nd_free(m);
    end();
}

static void test_ignores_an_irrelevant_column(void)
{
    enum { NV = 3 };
    double X[NROWS * NV], y[NROWS];
    nd_options o;
    nd_model *m;
    int i;

    begin("a pure-noise column does not wreck the fit");
    g_rs = 777;
    for (i = 0; i < NROWS; i++) {
        double a = 1.0 + 4.0 * urand();
        double b = 1.0 + 4.0 * urand();
        X[i * NV + 0] = a;
        X[i * NV + 1] = 100.0 * urand();   /* irrelevant */
        X[i * NV + 2] = b;
        y[i] = a / b;
    }

    o = nd_defaults();
    o.seed = 31337; o.ops = ND_OPS_ARITH;
    o.population = 300; o.islands = 4; o.generations = 120;
    m = nd_fit(X, y, NROWS, NV, o);
    check(m != NULL, "search returned a model");
    if (m) check(nd_r2(nd_knee(m, 0.01), X, y, NROWS, NV) > 0.999,
                 "still recovers x0/x2");
    nd_free(m);
    end();
}

/* ── Degenerate input must not crash ────────────────────────────────────── */

static void test_rejects_impossible_input(void)
{
    double X[8], y[4];
    nd_options o = nd_defaults();
    int i;

    begin("impossible input is refused, not crashed on");
    for (i = 0; i < 4; i++) { X[i * 2] = i; X[i * 2 + 1] = i; y[i] = i; }

    check(nd_fit(NULL, y, 4, 2, o) == NULL, "NULL X is refused");
    check(nd_fit(X, NULL, 4, 2, o) == NULL, "NULL y is refused");
    check(nd_fit(X, y, 1, 2, o) == NULL, "too few rows is refused");
    check(nd_fit(X, y, 4, 0, o) == NULL, "zero variables is refused");

    check(nd_count(NULL) == 0, "NULL model counts zero");
    check(nd_at(NULL, 0) == NULL, "NULL model indexes to NULL");
    check(nd_best(NULL) == NULL, "NULL model has no best");
    check(nd_knee(NULL, 0.05) == NULL, "NULL model has no knee");
    check(nd_complexity(NULL) == 0, "NULL expression has no nodes");
    nd_free(NULL);                      /* must be a no-op */
    end();
}

static void test_constant_target_is_survivable(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;
    int i;

    begin("a constant target does not divide by zero");
    g_rs = 99;
    for (i = 0; i < NROWS; i++) {
        X[i * 2 + 0] = urand();
        X[i * 2 + 1] = urand();
        y[i] = 7.0;
    }
    o = nd_defaults();
    o.seed = 2; o.population = 100; o.islands = 2; o.generations = 20;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        double v = nd_eval(nd_best(m), X);
        check(v == v, "prediction is a number");
        check_near(v, 7.0, 1e-6, "and it is the constant");
    }
    nd_free(m);
    end();
}

static void test_restricted_operator_set_is_respected(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;
    char text[2048];

    begin("an arithmetic-only search emits no transcendentals");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 8; o.ops = ND_OPS_ARITH;
    o.population = 200; o.islands = 3; o.generations = 60;
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search returned a model");
    if (m) {
        int i;
        for (i = 0; i < nd_count(m); i++) {
            nd_format(nd_at(m, i), NULL, text, (int)sizeof text);
            check(strstr(text, "sin")  == NULL, "no sin");
            check(strstr(text, "cos")  == NULL, "no cos");
            check(strstr(text, "exp")  == NULL, "no exp");
            check(strstr(text, "log")  == NULL, "no log");
            check(strstr(text, "tanh") == NULL, "no tanh");
        }
    }
    nd_free(m);
    end();
}

static void test_tiny_budgets_do_not_break(void)
{
    double X[NROWS * 2], y[NROWS];
    nd_options o;
    nd_model *m;

    begin("degenerate budgets are clamped, not honoured blindly");
    make_product(X, y, NROWS);

    o = nd_defaults();
    o.seed = 1;
    o.population  = 1;      /* below the floor        */
    o.islands     = 0;      /* below the floor        */
    o.generations = 0;      /* below the floor        */
    o.max_nodes   = 1;      /* below the floor        */
    o.tune_iters  = 0;      /* polish disabled        */
    o.ops         = 0;      /* no operators selected  */
    m = nd_fit(X, y, NROWS, 2, o);
    check(m != NULL, "search still returned a model");
    if (m) check(nd_count(m) > 0, "with at least one entry");
    nd_free(m);
    end();
}

/* ── Entry point ────────────────────────────────────────────────────────── */

int main(void)
{
    printf("\nnerve_discover test suite\n");

    printf("\n  determinism\n");
    test_same_seed_same_answer();
    test_different_seed_is_allowed_to_differ();

    printf("\n  the Pareto front\n");
    test_front_is_a_front();
    test_knee_is_no_larger_than_best();

    printf("\n  self-consistency\n");
    test_r2_matches_manual_computation();
    test_format_is_bounded();
    test_printed_equations_are_unambiguous();

    printf("\n  discovery\n");
    test_recovers_a_product_law();
    test_ignores_an_irrelevant_column();
    test_restricted_operator_set_is_respected();

    printf("\n  robustness\n");
    test_rejects_impossible_input();
    test_constant_target_is_survivable();
    test_tiny_budgets_do_not_break();

    printf("\n  %d checks, %d failed\n\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
