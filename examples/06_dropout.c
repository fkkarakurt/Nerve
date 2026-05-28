/*
 * Nerve — Example 06: Dropout Regularisation
 * Copyright (c) 2022 Fatih Küçükkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Demonstrates how dropout prevents memorisation of label noise.
 *
 * Task: binary classification with a circular decision boundary.
 *       Samples inside a circle (radius 0.5) are class 0; outside, class 1.
 *
 * The training set contains 10 deliberately mislabelled points mixed in with
 * 30 correct ones.  A massively over-parameterised network (2-128-128-2)
 * memorises the wrong labels without dropout, hurting test accuracy.
 * With dropout=0.5 the network cannot memorise 10 outliers and instead
 * learns the true circular boundary.
 *
 * Build:  gcc -O2 06_dropout.c -o dropout -lm
 * Run:    ./dropout
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_CORRECT   30
#define N_CORRUPT   10
#define N_TRAIN     (N_CORRECT + N_CORRUPT)
#define N_TEST      400
#define EPOCHS     5000

#define RADIUS  0.5f

/* True label: 0 = inside circle, 1 = outside */
static int true_label(float x, float y)
{ return ((x*x + y*y) < RADIUS*RADIUS) ? 0 : 1; }

static float randf(void) { return 2.0f * ((float)rand()/(float)RAND_MAX) - 1.0f; }

static float accuracy(network_t *net,
                      const float *in, const float *tgt, int n)
{
    return net_compute_accuracy(net, in, tgt, n, 2, 2);
}

int main(void)
{
    float      train_in[N_TRAIN * 2], train_tgt[N_TRAIN * 2];
    float      test_in[N_TEST  * 2], test_tgt[N_TEST  * 2];
    network_t *net_plain, *net_drop;
    float      ap, ad;
    int        i, lbl, flipped, epoch;

    /* Build training set: 30 correct + 10 with flipped labels */
    srand(77);
    flipped = 0;
    for (i = 0; i < N_TRAIN; i++) {
        float x = randf(), y = randf();
        train_in[i*2+0] = x;
        train_in[i*2+1] = y;
        lbl = true_label(x, y);
        /* flip the last N_CORRUPT labels */
        if (i >= N_CORRECT) { lbl = 1 - lbl; flipped++; }
        train_tgt[i*2+0] = (lbl == 0) ? 1.0f : 0.0f;
        train_tgt[i*2+1] = (lbl == 1) ? 1.0f : 0.0f;
    }

    /* Build clean test set on a uniform grid */
    srand(55);
    for (i = 0; i < N_TEST; i++) {
        float x = randf(), y = randf();
        test_in[i*2+0] = x;
        test_in[i*2+1] = y;
        lbl = true_label(x, y);
        test_tgt[i*2+0] = (lbl == 0) ? 1.0f : 0.0f;
        test_tgt[i*2+1] = (lbl == 1) ? 1.0f : 0.0f;
    }

    /* Both networks get identical Xavier init */
    srand(42);
    net_plain = net_allocate(4, 2, 128, 128, 2);
    net_set_optimizer(net_plain,  NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net_plain, NERVENET_ACTIVATION_RELU);
    net_initialize_he(net_plain);
    net_set_learning_rate(net_plain, 0.001f);

    srand(42);
    net_drop = net_allocate(4, 2, 128, 128, 2);
    net_set_optimizer(net_drop,  NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net_drop, NERVENET_ACTIVATION_RELU);
    net_initialize_he(net_drop);
    net_set_learning_rate(net_drop, 0.001f);
    net_set_dropout(net_drop, 0.5f);

    printf("Nerve %s -- Dropout vs No Dropout\n", net_get_version());
    printf("Task: binary classification — circular decision boundary\n");
    printf("Network: 2-128-128-2 (%d params) | Adam + ReLU | He Init\n",
           net_get_no_of_weights(net_plain));
    printf("Training: %d correct + %d mislabelled  |  Test: %d clean\n\n",
           N_CORRECT, flipped, N_TEST);

    printf("  Epoch |  No Dropout             |  Dropout p=0.5\n");
    printf("        |  Train Acc  Test Acc    |  Train Acc  Test Acc\n");
    printf("  ---------------------------------------------------------\n");

    for (epoch = 1; epoch <= EPOCHS; epoch++) {
        net_train_epoch(net_plain, train_in, train_tgt, N_TRAIN, 2, 2, 8);
        net_train_epoch(net_drop,  train_in, train_tgt, N_TRAIN, 2, 2, 8);

        if (epoch % 1000 == 0) {
            float tp = accuracy(net_plain, train_in, train_tgt, N_TRAIN);
            float td = accuracy(net_drop,  train_in, train_tgt, N_TRAIN);
            float vp = accuracy(net_plain, test_in,  test_tgt,  N_TEST);
            float vd = accuracy(net_drop,  test_in,  test_tgt,  N_TEST);
            printf("  %5d |   %5.1f%%     %5.1f%%    |   %5.1f%%     %5.1f%%\n",
                   epoch, tp*100, vp*100, td*100, vd*100);
        }
    }

    ap = accuracy(net_plain, test_in, test_tgt, N_TEST);
    ad = accuracy(net_drop,  test_in, test_tgt, N_TEST);

    printf("\n  No-dropout test accuracy : %.1f%%\n", ap * 100.0f);
    printf("  Dropout    test accuracy : %.1f%%\n",  ad * 100.0f);
    if (ad > ap)
        printf("  Dropout improved generalisation by %.1f pp\n",
               (ad - ap) * 100.0f);

    net_free(net_plain);
    net_free(net_drop);
    return 0;
}
