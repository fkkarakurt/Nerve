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
 * Nerve — core test suite
 * ===========================================================================
 * Zero dependencies, one translation unit, no test framework: it compiles
 * with the same one-line gcc invocation as everything else in this project.
 *
 *     gcc -O2 -std=c99 -Wall -Wextra tests/test_nerve.c -o test_nerve -lm
 *     ./test_nerve
 *
 * Exit status is 0 only when every check passes, so CI can gate on it.
 *
 * The library itself is ANSI C89. This harness is C99 so it can use
 * <stdint.h> for the independent RNG reference implementation — the whole
 * point of that check is to compare Nerve's C89-safe 32-bit emulation
 * against an exact-width type it is not allowed to rely on.
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <stdint.h>

/* ── Tiny assertion harness ─────────────────────────────────────────────── */

static int g_checks = 0;
static int g_failed = 0;
static const char *g_case = "";

static void begin(const char *name)
{
    g_case = name;
    printf("  %-52s", name);
    fflush(stdout);
}

static void end(void)
{
    printf("%s\n", g_failed ? "" : "ok");
}

static void fail(const char *fmt, ...)
{
    va_list ap;
    g_failed++;
    printf("\n      FAIL [%s] ", g_case);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}

#define CHECK(cond, ...) \
    do { g_checks++; if (!(cond)) fail(__VA_ARGS__); } while (0)

static int close_enough(float a, float b, float tol)
{
    float d = (float)fabs((double)(a - b));
    return d <= tol;
}

/* ── Reference xoshiro128** ─────────────────────────────────────────────────
 * Written straight from Blackman & Vigna with exact-width types. Nerve's own
 * generator emulates this in `unsigned long` so it stays inside C89; if the
 * masking were wrong, the two would diverge on LP64 (where unsigned long is
 * 64-bit) while still looking fine on Windows. This is the check that makes
 * "identical results on every platform" a fact rather than a hope.
 */
static uint32_t ref_s[4];

static uint32_t ref_rotl(uint32_t x, int k)
{
    return (uint32_t)((x << k) | (x >> (32 - k)));
}

static void ref_seed(uint32_t seed)
{
    int i;
    uint32_t z = seed;
    for (i = 0; i < 4; i++) {
        uint32_t t;
        z += UINT32_C(0x9E3779B9);
        t  = z;
        t  = (t ^ (t >> 16)) * UINT32_C(0x21F0AAAD);
        t  = (t ^ (t >> 15)) * UINT32_C(0x735A2D97);
        ref_s[i] = t ^ (t >> 15);
    }
    if ((ref_s[0] | ref_s[1] | ref_s[2] | ref_s[3]) == 0)
        ref_s[0] = UINT32_C(0x9E3779B9);
}

static uint32_t ref_next(void)
{
    uint32_t result = ref_rotl(ref_s[1] * 5u, 7) * 9u;
    uint32_t t      = ref_s[1] << 9;
    ref_s[2] ^= ref_s[0];
    ref_s[3] ^= ref_s[1];
    ref_s[1] ^= ref_s[2];
    ref_s[0] ^= ref_s[3];
    ref_s[2] ^= t;
    ref_s[3]  = ref_rotl(ref_s[3], 11);
    return result;
}

/* ── RNG tests ──────────────────────────────────────────────────────────── */

static void test_rng_matches_reference(void)
{
    const uint32_t seeds[] = { 0u, 1u, 42u, 12345u, 0x9E3779B9u, 0xFFFFFFFFu };
    size_t k;
    int i;

    begin("rng matches xoshiro128** reference");
    for (k = 0; k < sizeof(seeds) / sizeof(seeds[0]); k++) {
        nerve_seed((unsigned long)seeds[k]);
        ref_seed(seeds[k]);
        for (i = 0; i < 20000; i++) {
            unsigned long got  = nerve_rand_u32();
            uint32_t      want = ref_next();
            if (got != (unsigned long)want) {
                fail("seed %lu draw %d: got %lu, reference %lu",
                     (unsigned long)seeds[k], i, got, (unsigned long)want);
                end();
                return;
            }
        }
        g_checks++;
    }
    end();
}

static void test_rng_stays_in_32_bits(void)
{
    int i;
    begin("rng output never exceeds 32 bits");
    nerve_seed(7);
    for (i = 0; i < 50000; i++) {
        unsigned long v = nerve_rand_u32();
        CHECK(v <= 0xFFFFFFFFUL, "draw %d overflowed 32 bits: %lu", i, v);
        if (g_failed) break;
    }
    end();
}

static void test_rng_is_reproducible(void)
{
    unsigned long a[16], b[16];
    int i;

    begin("same seed reproduces the same stream");
    nerve_seed(2026);
    for (i = 0; i < 16; i++) a[i] = nerve_rand_u32();
    nerve_seed(2026);
    for (i = 0; i < 16; i++) b[i] = nerve_rand_u32();
    for (i = 0; i < 16; i++)
        CHECK(a[i] == b[i], "draw %d differs across reseed: %lu vs %lu",
              i, a[i], b[i]);
    end();
}

static void test_rng_seeds_differ(void)
{
    unsigned long a, b;
    begin("different seeds produce different streams");
    nerve_seed(1); a = nerve_rand_u32();
    nerve_seed(2); b = nerve_rand_u32();
    CHECK(a != b, "seeds 1 and 2 produced the same first draw (%lu)", a);
    end();
}

static void test_rng_zero_seed_is_not_degenerate(void)
{
    int i, nonzero = 0;
    begin("seed 0 does not collapse the state");
    nerve_seed(0);
    for (i = 0; i < 64; i++) if (nerve_rand_u32() != 0UL) nonzero++;
    CHECK(nonzero > 60, "seed 0 produced %d zeros in 64 draws", 64 - nonzero);
    end();
}

static void test_rng_float_range(void)
{
    int i;
    float lo = 1.0f, hi = 0.0f;
    double sum = 0.0;

    begin("rand_float stays in [0,1) and centres on 0.5");
    nerve_seed(99);
    for (i = 0; i < 200000; i++) {
        float v = nerve_rand_float();
        if (v < 0.0f || v >= 1.0f) {
            fail("draw %d out of [0,1): %f", i, (double)v);
            break;
        }
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        sum += (double)v;
    }
    g_checks++;
    CHECK(close_enough((float)(sum / 200000.0), 0.5f, 0.01f),
          "mean of 200k draws is %f, expected ~0.5", sum / 200000.0);
    CHECK(lo < 0.01f && hi > 0.99f,
          "draws did not span the interval: min %f max %f",
          (double)lo, (double)hi);
    end();
}

static void test_rng_golden_vector(void)
{
    /* Generated from the reference implementation above, not from Nerve's own
     * output, and pinned here forever. If a future refactor of the generator
     * silently changes the stream, every saved seed in every paper, notebook
     * and regression test that ever used Nerve would quietly stop reproducing.
     * This vector is what makes that change loud. */
    static const unsigned long golden[8] = {
        0x275D943DUL, 0xD9B9AAB4UL, 0x04A1304EUL, 0x36411021UL,
        0x88EE979CUL, 0xE1AAE265UL, 0xBA2CE7A8UL, 0x70BDBB6CUL
    };
    int i;

    begin("rng reproduces the pinned golden vector for seed 42");
    nerve_seed(42);
    for (i = 0; i < 8; i++) {
        unsigned long got = nerve_rand_u32();
        CHECK(got == golden[i], "draw %d: got 0x%08lX, pinned 0x%08lX",
              i, got, golden[i]);
    }
    end();
}

static void test_rand_below_respects_its_bound(void)
{
    const unsigned long bounds[] = { 1UL, 2UL, 3UL, 7UL, 10UL, 1000UL, 65536UL };
    size_t k;
    int i;

    begin("rand_below never returns its bound");
    nerve_seed(17);
    for (k = 0; k < sizeof(bounds) / sizeof(bounds[0]); k++) {
        for (i = 0; i < 20000; i++) {
            unsigned long v = nerve_rand_below(bounds[k]);
            if (v >= bounds[k]) {
                fail("bound %lu produced %lu", bounds[k], v);
                end();
                return;
            }
        }
        g_checks++;
    }
    end();
}

static void test_rand_below_is_unbiased(void)
{
    /* 7 does not divide 2^32, so a naive `rand() % 7` would over-represent the
     * low residues. With 700k draws over 7 buckets the expected count is 100k;
     * a chi-square statistic with 6 degrees of freedom exceeds 24.1 with
     * probability 0.0005, so the threshold below is a very loose net that a
     * genuinely biased generator would still blow straight through. */
    const int    B = 7;
    const long   N = 700000L;
    long counts[7];
    int i;
    double expected = (double)N / (double)B;
    double chi2 = 0.0;

    begin("rand_below is unbiased for a bound that splits 2^32 unevenly");
    for (i = 0; i < B; i++) counts[i] = 0;
    nerve_seed(4242);
    for (i = 0; i < N; i++) counts[nerve_rand_below((unsigned long)B)]++;

    for (i = 0; i < B; i++) {
        double d = (double)counts[i] - expected;
        chi2 += d * d / expected;
    }
    g_checks++;
    CHECK(chi2 < 24.1, "chi-square %.2f over %d buckets suggests bias", chi2, B);
    end();
    printf("      (chi-square %.2f, 6 dof, critical value 24.1 at p=0.0005)\n",
           chi2);
}

/* ── Initialisation tests ───────────────────────────────────────────────── */

static int weights_equal(const network_t *a, const network_t *b)
{
    int l, nu, nl;
    for (l = 1; l < a->no_of_layers; l++)
        for (nu = 0; nu < a->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= a->layer[l - 1].no_of_neurons; nl++)
                if (a->layer[l].neuron[nu].weight[nl] !=
                    b->layer[l].neuron[nu].weight[nl])
                    return 0;
    return 1;
}

static void test_init_is_reproducible(void)
{
    network_t *a, *b;

    begin("same seed gives bit-identical Xavier weights");
    nerve_seed(42); a = net_allocate(3, 4, 8, 2); net_initialize_xavier(a);
    nerve_seed(42); b = net_allocate(3, 4, 8, 2); net_initialize_xavier(b);
    CHECK(weights_equal(a, b), "Xavier init diverged under the same seed");
    net_free(a); net_free(b);
    end();

    begin("same seed gives bit-identical He weights");
    nerve_seed(7); a = net_allocate(3, 4, 8, 2); net_initialize_he(a);
    nerve_seed(7); b = net_allocate(3, 4, 8, 2); net_initialize_he(b);
    CHECK(weights_equal(a, b), "He init diverged under the same seed");
    net_free(a); net_free(b);
    end();
}

static void test_xavier_respects_its_bound(void)
{
    network_t *net;
    int l, nu, nl;
    int violations = 0;

    begin("Xavier weights lie inside +/- sqrt(6/(fan_in+fan_out))");
    nerve_seed(3);
    net = net_allocate(3, 10, 20, 5);
    net_initialize_xavier(net);
    for (l = 1; l < net->no_of_layers; l++) {
        int fi = net->layer[l - 1].no_of_neurons;
        int fo = net->layer[l].no_of_neurons;
        float r = (float)sqrt(6.0 / (double)(fi + fo));
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++) {
                float w = net->layer[l].neuron[nu].weight[nl];
                if (w < -r || w > r) violations++;
            }
    }
    CHECK(violations == 0, "%d weights fell outside the Xavier bound",
          violations);
    net_free(net);
    end();
}

/* ── Gradient check ─────────────────────────────────────────────────────────
 * The central correctness property of any backprop implementation, and the
 * one thing this library never verified. Nerve computes
 *
 *     grad = neuron.error * lower.output        (this is -dE/dw)
 *     delta = lr * grad + momentum * prev
 *     w    += delta
 *
 * so with momentum and L2 disabled, one net_train() step leaves
 *
 *     dE/dw = (w_before - w_after) / lr
 *
 * which we compare against a central finite difference of the loss that
 * net_compute_output_error() reports. Central differences are second-order
 * accurate; in float the useful floor is around 1e-2 relative, so we use a
 * combined absolute+relative tolerance and a smooth activation (tanh) to
 * avoid ReLU's non-differentiable kink.
 */

/* Central differences trade truncation error (grows as h^2) against
 * cancellation in (E(w+h) - E(w-h)) (grows as eps/h). For float the optimum
 * sits near (3*eps)^(1/3) ~ 7e-3; the value below was picked by measurement,
 * not by theory alone. Override to re-run the sweep. */
#ifndef NERVE_TEST_H
#define NERVE_TEST_H 7e-3f
#endif

/* Measured floor of the finite-difference noise (worst observed ~3e-5 across
 * the sweep), with room to spare; and the relative tolerance that applies once
 * a gradient is large enough for relative error to mean anything. */
#define FD_NOISE 1e-4f
#define FD_RTOL  1e-2f

typedef struct { int l, nu, nl; } widx_t;

static float loss_at(network_t *net, const float *x, const float *t)
{
    net_compute(net, x, NULL);
    return net_compute_output_error(net, t);
}

static void test_gradients_match_finite_differences(void)
{
    const float lr = 1e-3f;
    const float h  = NERVE_TEST_H;
    float x[3] = { 0.35f, -0.72f, 0.11f };
    float t[2] = { 0.8f, 0.2f };
    network_t *net;
    widx_t idx[512];
    float before[512], analytic[512];
    int count = 0, i, l, nu, nl;
    int compared = 0;
    float worst_abs = 0.0f, worst_rel = 0.0f;

    begin("backprop gradients match finite differences");

    nerve_seed(1234);
    net = net_allocate(3, 3, 5, 2);
    net_set_optimizer(net, NERVENET_OPTIMIZER_SGD);
    net_set_activation(net, NERVENET_ACTIVATION_TANH);
    net_set_output_activation(net, NERVENET_ACTIVATION_SIGMOID);
    net_set_loss(net, NERVENET_LOSS_MSE);
    net_set_momentum(net, 0.0f);      /* delta must be exactly lr * grad   */
    net_set_l2_lambda(net, 0.0f);     /* no weight decay in the gradient   */
    net_set_dropout(net, 0.0f);       /* deterministic forward pass        */
    net_set_learning_rate(net, lr);
    net_initialize_xavier(net);

    /* Index every weight, and snapshot it. */
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++) {
                if (count >= 512) break;
                idx[count].l = l; idx[count].nu = nu; idx[count].nl = nl;
                before[count] = net->layer[l].neuron[nu].weight[nl];
                count++;
            }

    /* One training step yields every analytic gradient at once. */
    net_compute(net, x, NULL);
    net_compute_output_error(net, t);
    net_train(net);

    for (i = 0; i < count; i++) {
        float after = net->layer[idx[i].l].neuron[idx[i].nu].weight[idx[i].nl];
        analytic[i] = (before[i] - after) / lr;
        /* restore, so the numeric pass sees the original weights */
        net->layer[idx[i].l].neuron[idx[i].nu].weight[idx[i].nl] = before[i];
    }

    for (i = 0; i < count; i++) {
        float *w = &net->layer[idx[i].l].neuron[idx[i].nu].weight[idx[i].nl];
        float orig = *w, ep, em, numeric, mag, abs_err, tol;

        *w = orig + h; ep = loss_at(net, x, t);
        *w = orig - h; em = loss_at(net, x, t);
        *w = orig;

        numeric = (ep - em) / (2.0f * h);
        mag     = (float)fabs((double)analytic[i]);
        if ((float)fabs((double)numeric) > mag) mag = (float)fabs((double)numeric);

        abs_err = (float)fabs((double)(numeric - analytic[i]));
        if (abs_err > worst_abs) worst_abs = abs_err;
        if (mag > 1e-6f && abs_err / mag > worst_rel) worst_rel = abs_err / mag;
        compared++;

        /* Combined absolute + relative tolerance, the standard criterion for
         * a float gradcheck. FD_NOISE is not a fudge factor: E is ~1e-1 here
         * while E(w+h)-E(w-h) is ~1e-5, so the subtraction cancels away most
         * of the mantissa and leaves an absolute floor of ~1e-5 on every
         * gradient, large or small. Judging small gradients by relative error
         * alone would therefore flag arithmetic that is in fact correct. */
        tol = FD_NOISE + FD_RTOL * mag;
        if (abs_err > tol)
            fail("weight[%d][%d][%d]: analytic %.6g, numeric %.6g "
                 "(abs err %.3g > tol %.3g)",
                 idx[i].l, idx[i].nu, idx[i].nl,
                 (double)analytic[i], (double)numeric,
                 (double)abs_err, (double)tol);
    }

    g_checks++;
    CHECK(compared > 10, "only %d gradients were non-trivial", compared);
    net_free(net);
    end();
    printf("      (%d gradients compared, worst abs err %.2e, worst rel %.2e)\n",
           compared, (double)worst_abs, (double)worst_rel);
}

/* ── Learning tests ─────────────────────────────────────────────────────── */

static void test_xor_converges(void)
{
    float inputs[4][2] = {{1,1},{1,0},{0,1},{0,0}};
    float targets[4]   = {0, 1, 1, 0};
    network_t *net;
    int i, epoch, correct = 0;
    float mse = 1.0f, out;

    begin("XOR converges from a fixed seed");
    nerve_seed(42);
    net = net_allocate(3, 2, 4, 1);
    net_set_optimizer(net, NERVENET_OPTIMIZER_ADAM);
    net_set_learning_rate(net, 0.01f);
    net_initialize_xavier(net);

    for (epoch = 1; epoch <= 10000; epoch++) {
        mse = 0.0f;
        for (i = 0; i < 4; i++) {
            net_compute(net, inputs[i], NULL);
            mse += net_compute_output_error(net, &targets[i]);
            net_train(net);
        }
        mse /= 4.0f;
        if (mse < 1e-4f) break;
    }
    for (i = 0; i < 4; i++) {
        net_compute(net, inputs[i], &out);
        if ((out > 0.5f) == (int)targets[i]) correct++;
    }
    CHECK(correct == 4, "only %d/4 XOR gates correct", correct);
    CHECK(mse < 1e-4f, "did not reach the MSE target in %d epochs (mse %g)",
          epoch, (double)mse);
    net_free(net);
    end();
}

static void test_softmax_is_a_distribution(void)
{
    float x[4] = { 0.5f, -0.3f, 0.9f, 0.1f };
    float out[3];
    network_t *net;
    float sum = 0.0f;
    int i;

    begin("softmax output is a probability distribution");
    nerve_seed(5);
    net = net_allocate(3, 4, 6, 3);
    net_set_classification(net);
    net_initialize_xavier(net);
    net_compute(net, x, out);

    for (i = 0; i < 3; i++) {
        CHECK(out[i] >= 0.0f && out[i] <= 1.0f,
              "class %d probability out of range: %f", i, (double)out[i]);
        sum += out[i];
    }
    CHECK(close_enough(sum, 1.0f, 1e-5f),
          "probabilities sum to %f, expected 1.0", (double)sum);
    net_free(net);
    end();
}

/* ── Persistence tests ──────────────────────────────────────────────────── */

static void test_save_load_roundtrip(void)
{
    network_t *net, *back;
    float x[4] = { 0.2f, 0.4f, -0.6f, 0.8f };
    float a[2], b[2];
    const char *txt = "test_roundtrip.net";
    const char *bin = "test_roundtrip.bin";
    int rc;

    begin("net_save/net_load preserves predictions");
    nerve_seed(11);
    net = net_allocate(3, 4, 6, 2);
    net_initialize_xavier(net);
    net_compute(net, x, a);

    rc = net_save(txt, net);            /* NOTE: filename first, then net */
    CHECK(rc != EOF, "net_save failed");
    back = net_load(txt);
    CHECK(back != NULL, "net_load returned NULL");
    if (back) {
        net_compute(back, x, b);
        CHECK(close_enough(a[0], b[0], 1e-5f) && close_enough(a[1], b[1], 1e-5f),
              "text round-trip changed the output: (%f,%f) vs (%f,%f)",
              (double)a[0], (double)a[1], (double)b[0], (double)b[1]);
        net_free(back);
    }
    remove(txt);
    net_free(net);
    end();

    begin("net_bsave/net_bload preserves predictions exactly");
    nerve_seed(12);
    net = net_allocate(3, 4, 6, 2);
    net_initialize_xavier(net);
    net_compute(net, x, a);

    rc = net_bsave(bin, net);
    CHECK(rc != EOF, "net_bsave failed");
    back = net_bload(bin);
    CHECK(back != NULL, "net_bload returned NULL");
    if (back) {
        net_compute(back, x, b);
        CHECK(a[0] == b[0] && a[1] == b[1],
              "binary round-trip is not bit-exact: (%f,%f) vs (%f,%f)",
              (double)a[0], (double)a[1], (double)b[0], (double)b[1]);
        net_free(back);
    }
    remove(bin);
    net_free(net);
    end();
}

static void test_copy_is_independent(void)
{
    network_t *net, *dup;
    float x[3] = { 0.1f, 0.2f, 0.3f };
    float a[1], b[1];

    begin("net_copy produces an equal but independent network");
    nerve_seed(21);
    net = net_allocate(3, 3, 5, 1);
    net_initialize_xavier(net);
    dup = net_copy(net);
    CHECK(dup != NULL, "net_copy returned NULL");
    if (dup) {
        net_compute(net, x, a);
        net_compute(dup, x, b);
        CHECK(a[0] == b[0], "copy predicts %f, original %f",
              (double)b[0], (double)a[0]);
        CHECK(weights_equal(net, dup), "copy has different weights");

        /* mutating the copy must not touch the original */
        net_set_weight(dup, 1, 0, 0, 12.5f);
        CHECK(net_get_weight(net, 1, 0, 0) != 12.5f,
              "writing to the copy modified the original");
        net_free(dup);
    }
    net_free(net);
    end();
}

static void test_validate_accepts_a_fresh_net(void)
{
    network_t *net;
    begin("net_validate accepts a freshly initialised network");
    nerve_seed(31);
    net = net_allocate(3, 2, 4, 1);
    net_initialize_xavier(net);
    CHECK(net_validate(net) != 0, "net_validate rejected a valid network");
    net_free(net);
    end();
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("\nNerve %s — test suite\n\n", net_get_version());

    printf("  deterministic RNG\n");
    test_rng_matches_reference();
    test_rng_stays_in_32_bits();
    test_rng_is_reproducible();
    test_rng_seeds_differ();
    test_rng_zero_seed_is_not_degenerate();
    test_rng_float_range();
    test_rng_golden_vector();
    test_rand_below_respects_its_bound();
    test_rand_below_is_unbiased();

    printf("\n  initialisation\n");
    test_init_is_reproducible();
    test_xavier_respects_its_bound();

    printf("\n  gradients\n");
    test_gradients_match_finite_differences();

    printf("\n  learning\n");
    test_xor_converges();
    test_softmax_is_a_distribution();

    printf("\n  persistence and structure\n");
    test_save_load_roundtrip();
    test_copy_is_independent();
    test_validate_accepts_a_fresh_net();

    printf("\n  %d checks, %d failed\n\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
