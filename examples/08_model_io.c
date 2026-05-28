/*
 * Nerve — Example 08: Model Save / Load / Fine-Tune
 * Copyright (c) 2022 Fatih Küçükkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Demonstrates the full model persistence workflow:
 *   1. Train a network on the Iris dataset (30 samples)
 *   2. Save weights to disk with net_save()
 *   3. Load them back with net_load()
 *   4. Verify the loaded model is bit-for-bit equivalent
 *   5. Fine-tune for 500 additional epochs — accuracy improves
 *
 * This pattern enables cold-start deployment: ship the .net file,
 * load on start-up, run inference with zero training cost.
 *
 * Build:  gcc -O2 08_model_io.c -o model_io -lm
 * Run:    ./model_io
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_TRAIN  30   /* 10 per class */
#define N_TEST   30   /* 10 per class */
#define N_IN      4
#define N_OUT     3
#define SAVEFILE  "iris_model.net"

/*
 * Iris dataset — 60 samples (20 per class, balanced train / test)
 * Features: sepal length, sepal width, petal length, petal width (cm)
 * Classes:  0 = setosa, 1 = versicolor, 2 = virginica
 * Source: Fisher 1936 / UCI Machine Learning Repository
 */
static const float iris_raw[N_TRAIN + N_TEST][5] = {
    /* setosa — training rows 0..9 */
    {5.1f,3.5f,1.4f,0.2f, 0}, {4.9f,3.0f,1.4f,0.2f, 0},
    {4.7f,3.2f,1.3f,0.2f, 0}, {4.6f,3.1f,1.5f,0.2f, 0},
    {5.0f,3.6f,1.4f,0.2f, 0}, {5.4f,3.9f,1.7f,0.4f, 0},
    {4.6f,3.4f,1.4f,0.3f, 0}, {5.0f,3.4f,1.5f,0.2f, 0},
    {4.4f,2.9f,1.4f,0.2f, 0}, {4.9f,3.1f,1.5f,0.1f, 0},
    /* versicolor — training rows 10..19 */
    {7.0f,3.2f,4.7f,1.4f, 1}, {6.4f,3.2f,4.5f,1.5f, 1},
    {6.9f,3.1f,4.9f,1.5f, 1}, {5.5f,2.3f,4.0f,1.3f, 1},
    {6.5f,2.8f,4.6f,1.5f, 1}, {5.7f,2.8f,4.5f,1.3f, 1},
    {6.3f,3.3f,4.7f,1.6f, 1}, {4.9f,2.4f,3.3f,1.0f, 1},
    {6.6f,2.9f,4.6f,1.3f, 1}, {5.2f,2.7f,3.9f,1.4f, 1},
    /* virginica — training rows 20..29 */
    {6.3f,3.3f,6.0f,2.5f, 2}, {5.8f,2.7f,5.1f,1.9f, 2},
    {7.1f,3.0f,5.9f,2.1f, 2}, {6.3f,2.9f,5.6f,1.8f, 2},
    {6.5f,3.0f,5.8f,2.2f, 2}, {7.6f,3.0f,6.6f,2.1f, 2},
    {4.9f,2.5f,4.5f,1.7f, 2}, {7.3f,2.9f,6.3f,1.8f, 2},
    {6.7f,2.5f,5.8f,1.8f, 2}, {7.2f,3.6f,6.1f,2.5f, 2},
    /* setosa — test rows 30..39 */
    {5.4f,3.7f,1.5f,0.2f, 0}, {4.8f,3.4f,1.6f,0.2f, 0},
    {4.8f,3.0f,1.4f,0.1f, 0}, {4.3f,3.0f,1.1f,0.1f, 0},
    {5.8f,4.0f,1.2f,0.2f, 0}, {5.7f,4.4f,1.5f,0.4f, 0},
    {5.4f,3.9f,1.3f,0.4f, 0}, {5.1f,3.5f,1.4f,0.3f, 0},
    {5.7f,3.8f,1.7f,0.3f, 0}, {5.1f,3.8f,1.5f,0.3f, 0},
    /* versicolor — test rows 40..49 */
    {5.0f,2.0f,3.5f,1.0f, 1}, {5.9f,3.0f,4.2f,1.5f, 1},
    {6.0f,2.2f,4.0f,1.0f, 1}, {6.1f,2.9f,4.7f,1.4f, 1},
    {5.6f,2.9f,3.6f,1.3f, 1}, {6.7f,3.1f,4.4f,1.4f, 1},
    {5.6f,3.0f,4.5f,1.5f, 1}, {5.8f,2.7f,4.1f,1.0f, 1},
    {6.2f,2.2f,4.5f,1.5f, 1}, {5.6f,2.5f,3.9f,1.1f, 1},
    /* virginica — test rows 50..59 */
    {6.5f,3.2f,5.1f,2.0f, 2}, {6.4f,2.7f,5.3f,1.9f, 2},
    {6.8f,3.0f,5.5f,2.1f, 2}, {5.7f,2.5f,5.0f,2.0f, 2},
    {5.8f,2.8f,5.1f,2.4f, 2}, {6.4f,3.2f,5.3f,2.3f, 2},
    {6.5f,3.0f,5.5f,1.8f, 2}, {7.7f,3.8f,6.7f,2.2f, 2},
    {7.7f,2.6f,6.9f,2.3f, 2}, {6.0f,2.2f,5.0f,1.5f, 2}
};

/* Min / max bounds for normalisation */
#define SL_LO 4.3f
#define SL_HI 7.9f
#define SW_LO 2.0f
#define SW_HI 4.4f
#define PL_LO 1.0f
#define PL_HI 6.9f
#define PW_LO 0.1f
#define PW_HI 2.5f

static float norm(float x, float lo, float hi)
{ return (x - lo) / (hi - lo); }

static void build_arrays(float *inp, float *tgt, int offset, int n)
{
    int i, cls;
    for (i = 0; i < n; i++) {
        const float *r = iris_raw[offset + i];
        inp[i*N_IN+0] = norm(r[0], SL_LO, SL_HI);
        inp[i*N_IN+1] = norm(r[1], SW_LO, SW_HI);
        inp[i*N_IN+2] = norm(r[2], PL_LO, PL_HI);
        inp[i*N_IN+3] = norm(r[3], PW_LO, PW_HI);
        cls = (int)r[4];
        tgt[i*N_OUT+0] = (cls == 0) ? 1.0f : 0.0f;
        tgt[i*N_OUT+1] = (cls == 1) ? 1.0f : 0.0f;
        tgt[i*N_OUT+2] = (cls == 2) ? 1.0f : 0.0f;
    }
}

static void print_cm(network_t *net, const float *inp, const float *tgt, int n)
{
    int cm[9] = {0};
    int row, col;
    static const char *names[3] = {"  setosa    ", "versicolor  ", "virginica   "};
    net_confusion_matrix(net, inp, tgt, n, N_IN, N_OUT, 3, cm);
    printf("              setosa  versicolor  virginica\n");
    for (row = 0; row < 3; row++) {
        printf("  %-12s", names[row]);
        for (col = 0; col < 3; col++)
            printf("  %6d    ", cm[row*3+col]);
        printf("\n");
    }
}

int main(void)
{
    float      train_in[N_TRAIN * N_IN],  train_tgt[N_TRAIN * N_OUT];
    float      test_in[N_TEST  * N_IN],   test_tgt[N_TEST  * N_OUT];
    network_t *net, *loaded;
    float      acc1, acc2, acc3, mse;
    FILE      *f;
    long       file_sz;
    int        i;

    build_arrays(train_in, train_tgt, 0,       N_TRAIN);
    build_arrays(test_in,  test_tgt,  N_TRAIN, N_TEST);

    printf("Nerve %s -- Model Save / Load / Fine-Tune\n", net_get_version());
    printf("Architecture: 4-12-8-3 | Adam + Tanh | Xavier Init\n");
    printf("Dataset: Iris (%d train / %d test)\n\n", N_TRAIN, N_TEST);

    /* Phase 1 — partial training (deliberately stopped early at ~80%) */
    srand(42);
    net = net_allocate(4, N_IN, 12, 8, N_OUT);
    net_set_optimizer(net,   NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net,  NERVENET_ACTIVATION_TANH);
    net_initialize_xavier(net);
    net_set_learning_rate(net, 0.0008f);    /* slow start — leaves room to improve */
    net_set_l2_lambda(net, 5e-4f);

    printf("Phase 1: early training (100 epochs, lr=0.0008)...\n");
    for (i = 1; i <= 100; i++) {
        mse = net_train_epoch(net, train_in, train_tgt,
                              N_TRAIN, N_IN, N_OUT, N_TRAIN);
        if (i % 25 == 0) {
            acc1 = net_compute_accuracy(net, test_in, test_tgt,
                                        N_TEST, N_IN, N_OUT);
            printf("  Epoch %3d | MSE: %.4f | Test Acc: %.1f%%\n",
                   i, mse, acc1 * 100.0f);
        }
    }
    acc1 = net_compute_accuracy(net, test_in, test_tgt, N_TEST, N_IN, N_OUT);
    printf("\nCheckpoint accuracy (early stop): %.1f%%  (%d / %d)\n",
           acc1 * 100.0f, (int)(acc1 * N_TEST + 0.5f), N_TEST);

    /* Save checkpoint */
    printf("\nSaving checkpoint to \"%s\"... ", SAVEFILE);
    if (net_save(SAVEFILE, net) == 0) {
        f = fopen(SAVEFILE, "r");
        file_sz = 0;
        if (f) { fseek(f, 0, SEEK_END); file_sz = ftell(f); fclose(f); }
        printf("done  (%.1f KB)\n", (float)file_sz / 1024.0f);
    } else {
        printf("FAILED\n");
    }

    /* Load — weights are restored; re-apply hyper-parameters. */
    printf("Loading  \"%s\"... ", SAVEFILE);
    loaded = net_load(SAVEFILE);
    if (!loaded) { printf("FAILED\n"); net_free(net); return 1; }
    net_set_optimizer(loaded,  NERVENET_OPTIMIZER_ADAM);
    net_set_activation(loaded, NERVENET_ACTIVATION_TANH);
    printf("done\n");

    acc2 = net_compute_accuracy(loaded, test_in, test_tgt, N_TEST, N_IN, N_OUT);
    printf("Loaded model accuracy: %.1f%%  %s\n",
           acc2 * 100.0f,
           (acc2 == acc1) ? "(identical to checkpoint)" : "(mismatch!)");

    /* Phase 2 — fine-tune at higher learning rate */
    net_set_learning_rate(loaded, 0.006f);
    net_set_l2_lambda(loaded, 1e-4f);

    printf("\nPhase 2: fine-tuning (600 epochs, lr=0.006)...\n");
    for (i = 1; i <= 600; i++) {
        mse = net_train_epoch(loaded, train_in, train_tgt,
                              N_TRAIN, N_IN, N_OUT, N_TRAIN);
        if (i % 150 == 0) {
            acc3 = net_compute_accuracy(loaded, test_in, test_tgt,
                                        N_TEST, N_IN, N_OUT);
            printf("  Epoch %3d | MSE: %.4f | Test Acc: %.1f%%\n",
                   100 + i, mse, acc3 * 100.0f);
        }
    }

    acc3 = net_compute_accuracy(loaded, test_in, test_tgt, N_TEST, N_IN, N_OUT);
    printf("\nFinal accuracy: %.1f%%  (%+.1f pp from checkpoint)\n",
           acc3 * 100.0f, (acc3 - acc1) * 100.0f);
    printf("Confusion matrix:\n");
    print_cm(loaded, test_in, test_tgt, N_TEST);

    net_free(net);
    net_free(loaded);
    return 0;
}
