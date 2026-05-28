/*
 * Nerve — Example 07: Non-linear Spiral Classification
 * Copyright (c) 2022 Fatih Küçükkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Classifies three interleaved Archimedean spirals — a canonical
 * benchmark for non-linear separability that no linear model can solve.
 *
 * After training an ASCII grid shows the learned decision boundary.
 *
 * Architecture: 2-32-32-3 | Adam + ReLU | He Init
 *
 * Build:  gcc -O2 07_spiral.c -o spiral -lm
 * Run:    ./spiral
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_PER_CLASS  60          /* training points per class  */
#define N_TEST_CLS   20          /* test points per class       */
#define N_CLS         3
#define N_TRAIN      (N_PER_CLASS * N_CLS)
#define N_TEST       (N_TEST_CLS  * N_CLS)
#define N_IN          2
#define N_OUT         3
#define EPOCHS     2500

#define PI 3.14159265f

/* Grid dimensions for the decision boundary plot */
#define GRID_W  62
#define GRID_H  22

static const char CLASS_CHAR[N_CLS] = {'.', 'X', 'O'};

static void gen_spiral(float *inp, float *tgt, int n_per, int n_test,
                       float noise)
{
    int  cls, i, base;
    float theta, r, angle, px, py;

    for (cls = 0; cls < N_CLS; cls++) {
        /* training points */
        base = cls * n_per;
        for (i = 0; i < n_per; i++) {
            theta = (float)(i + 1) / (float)n_per * 2.5f * PI;
            r     = theta / (2.5f * PI);
            angle = theta + cls * 2.0f * PI / (float)N_CLS;
            px = r * (float)cos((double)angle) +
                 noise * (2.0f * ((float)rand()/(float)RAND_MAX) - 1.0f);
            py = r * (float)sin((double)angle) +
                 noise * (2.0f * ((float)rand()/(float)RAND_MAX) - 1.0f);
            inp[(base+i)*N_IN+0] = px;
            inp[(base+i)*N_IN+1] = py;
            tgt[(base+i)*N_OUT+0] = (cls == 0) ? 1.0f : 0.0f;
            tgt[(base+i)*N_OUT+1] = (cls == 1) ? 1.0f : 0.0f;
            tgt[(base+i)*N_OUT+2] = (cls == 2) ? 1.0f : 0.0f;
        }
        /* test points (denser, less noise) */
        base = N_TRAIN + cls * n_test;
        for (i = 0; i < n_test; i++) {
            theta = (float)(i + 1) / (float)n_test * 2.5f * PI;
            r     = theta / (2.5f * PI);
            angle = theta + cls * 2.0f * PI / (float)N_CLS;
            px = r * (float)cos((double)angle) +
                 (noise * 0.5f) * (2.0f * ((float)rand()/(float)RAND_MAX) - 1.0f);
            py = r * (float)sin((double)angle) +
                 (noise * 0.5f) * (2.0f * ((float)rand()/(float)RAND_MAX) - 1.0f);
            inp[(base+i)*N_IN+0] = px;
            inp[(base+i)*N_IN+1] = py;
            tgt[(base+i)*N_OUT+0] = (cls == 0) ? 1.0f : 0.0f;
            tgt[(base+i)*N_OUT+1] = (cls == 1) ? 1.0f : 0.0f;
            tgt[(base+i)*N_OUT+2] = (cls == 2) ? 1.0f : 0.0f;
        }
    }
}

static void print_boundary(network_t *net)
{
    char grid[GRID_H][GRID_W + 1];
    float xlo = -1.15f, xhi = 1.15f;
    float ylo = -1.15f, yhi = 1.15f;
    float inp[2];
    int   row, col, cls;

    memset(grid, ' ', sizeof(grid));
    for (row = 0; row < GRID_H; row++) grid[row][GRID_W] = '\0';

    for (row = 0; row < GRID_H; row++) {
        inp[1] = yhi - (float)row / (float)(GRID_H - 1) * (yhi - ylo);
        for (col = 0; col < GRID_W; col++) {
            inp[0] = xlo + (float)col / (float)(GRID_W - 1) * (xhi - xlo);
            cls    = net_classify(net, inp);
            grid[row][col] = CLASS_CHAR[cls];
        }
    }

    printf("  +");
    for (col = 0; col < GRID_W; col++) putchar('-');
    printf("+\n");
    for (row = 0; row < GRID_H; row++)
        printf("  |%s|\n", grid[row]);
    printf("  +");
    for (col = 0; col < GRID_W; col++) putchar('-');
    printf("+\n");
}

int main(void)
{
    float     *all_in, *all_tgt;
    float     *train_in, *train_tgt;
    float     *test_in,  *test_tgt;
    network_t *net;
    float      mse, acc;
    int        i, correct;

    all_in  = (float *)malloc((N_TRAIN + N_TEST) * N_IN  * sizeof(float));
    all_tgt = (float *)malloc((N_TRAIN + N_TEST) * N_OUT * sizeof(float));
    if (!all_in || !all_tgt) { fprintf(stderr, "out of memory\n"); return 1; }

    srand(7);
    gen_spiral(all_in, all_tgt, N_PER_CLASS, N_TEST_CLS, 0.07f);

    train_in  = all_in;
    train_tgt = all_tgt;
    test_in   = all_in  + N_TRAIN * N_IN;
    test_tgt  = all_tgt + N_TRAIN * N_OUT;

    srand(42);
    net = net_allocate(4, N_IN, 32, 32, N_OUT);
    net_set_optimizer(net,   NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net,  NERVENET_ACTIVATION_RELU);
    net_initialize_he(net);
    net_set_learning_rate(net, 0.003f);
    net_set_l2_lambda(net, 5e-5f);

    printf("Nerve %s -- 3-Class Spiral Classification\n", net_get_version());
    printf("Architecture: 2-32-32-3 | Adam + ReLU | He Init\n");
    printf("Dataset: %d train + %d test  (%d classes x %d/%d points)\n\n",
           N_TRAIN, N_TEST, N_CLS, N_PER_CLASS, N_TEST_CLS);
    printf("Training...\n");

    for (i = 1; i <= EPOCHS; i++) {
        mse = net_train_epoch(net, train_in, train_tgt,
                              N_TRAIN, N_IN, N_OUT, 32);
        if (i % 500 == 0) {
            acc = net_compute_accuracy(net, test_in, test_tgt,
                                       N_TEST, N_IN, N_OUT);
            printf("  Epoch %4d | MSE: %.4f | Test Acc: %.1f%%\n",
                   i, mse, acc * 100.0f);
        }
    }

    acc = net_compute_accuracy(net, test_in, test_tgt, N_TEST, N_IN, N_OUT);
    correct = (int)(acc * N_TEST + 0.5f);
    printf("\nFinal Test Accuracy: %.1f%%  (%d / %d)\n\n",
           acc * 100.0f, correct, N_TEST);

    printf("Decision Boundary (. = Class 0  X = Class 1  O = Class 2):\n");
    print_boundary(net);

    net_free(net);
    free(all_in);
    free(all_tgt);
    return 0;
}
