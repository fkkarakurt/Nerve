/*
 * Nerve — Example 03: Iris Classification
 * Copyright (C) 2022 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Classifies the Fisher Iris dataset (150 samples, 4 features, 3 classes)
 * embedded directly in this file — no external data files needed.
 *
 * Features (normalised to [0,1]):
 *   sepal_length, sepal_width, petal_length, petal_width
 *
 * Classes:
 *   0 = Iris setosa
 *   1 = Iris versicolor
 *   2 = Iris virginica
 *
 * Build:  gcc -O2 03_iris.c -o iris -lm
 * Run:    ./iris
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Embedded Iris Dataset (Fisher 1936) ────────────────────────────────
 * Raw values: sepal_l, sepal_w, petal_l, petal_w
 * Min/Max for normalisation: [4.3,7.9] [2.0,4.4] [1.0,6.9] [0.1,2.5]
 * ─────────────────────────────────────────────────────────────────────── */
#define N_SAMPLES  150
#define N_FEATURES   4
#define N_CLASSES    3

static const float iris_raw[N_SAMPLES][N_FEATURES] = {
    /* Iris setosa (0) */
    {5.1f,3.5f,1.4f,0.2f},{4.9f,3.0f,1.4f,0.2f},{4.7f,3.2f,1.3f,0.2f},
    {4.6f,3.1f,1.5f,0.2f},{5.0f,3.6f,1.4f,0.2f},{5.4f,3.9f,1.7f,0.4f},
    {4.6f,3.4f,1.4f,0.3f},{5.0f,3.4f,1.5f,0.2f},{4.4f,2.9f,1.4f,0.2f},
    {4.9f,3.1f,1.5f,0.1f},{5.4f,3.7f,1.5f,0.2f},{4.8f,3.4f,1.6f,0.2f},
    {4.8f,3.0f,1.4f,0.1f},{4.3f,3.0f,1.1f,0.1f},{5.8f,4.0f,1.2f,0.2f},
    {5.7f,4.4f,1.5f,0.4f},{5.4f,3.9f,1.3f,0.4f},{5.1f,3.5f,1.4f,0.3f},
    {5.7f,3.8f,1.7f,0.3f},{5.1f,3.8f,1.5f,0.3f},{5.4f,3.4f,1.7f,0.2f},
    {5.1f,3.7f,1.5f,0.4f},{4.6f,3.6f,1.0f,0.2f},{5.1f,3.3f,1.7f,0.5f},
    {4.8f,3.4f,1.9f,0.2f},{5.0f,3.0f,1.6f,0.2f},{5.0f,3.4f,1.6f,0.4f},
    {5.2f,3.5f,1.5f,0.2f},{5.2f,3.4f,1.4f,0.2f},{4.7f,3.2f,1.6f,0.2f},
    {4.8f,3.1f,1.6f,0.2f},{5.4f,3.4f,1.5f,0.4f},{5.2f,4.1f,1.5f,0.1f},
    {5.5f,4.2f,1.4f,0.2f},{4.9f,3.1f,1.5f,0.1f},{5.0f,3.2f,1.2f,0.2f},
    {5.5f,3.5f,1.3f,0.2f},{4.9f,3.1f,1.5f,0.1f},{4.4f,3.0f,1.3f,0.2f},
    {5.1f,3.4f,1.5f,0.2f},{5.0f,3.5f,1.3f,0.3f},{4.5f,2.3f,1.3f,0.3f},
    {4.4f,3.2f,1.3f,0.2f},{5.0f,3.5f,1.6f,0.6f},{5.1f,3.8f,1.9f,0.4f},
    {4.8f,3.0f,1.4f,0.3f},{5.1f,3.8f,1.6f,0.2f},{4.6f,3.2f,1.4f,0.2f},
    {5.3f,3.7f,1.5f,0.2f},{5.0f,3.3f,1.4f,0.2f},
    /* Iris versicolor (1) */
    {7.0f,3.2f,4.7f,1.4f},{6.4f,3.2f,4.5f,1.5f},{6.9f,3.1f,4.9f,1.5f},
    {5.5f,2.3f,4.0f,1.3f},{6.5f,2.8f,4.6f,1.5f},{5.7f,2.8f,4.5f,1.3f},
    {6.3f,3.3f,4.7f,1.6f},{4.9f,2.4f,3.3f,1.0f},{6.6f,2.9f,4.6f,1.3f},
    {5.2f,2.7f,3.9f,1.4f},{5.0f,2.0f,3.5f,1.0f},{5.9f,3.0f,4.2f,1.5f},
    {6.0f,2.2f,4.0f,1.0f},{6.1f,2.9f,4.7f,1.4f},{5.6f,2.9f,3.6f,1.3f},
    {6.7f,3.1f,4.4f,1.4f},{5.6f,3.0f,4.5f,1.5f},{5.8f,2.7f,4.1f,1.0f},
    {6.2f,2.2f,4.5f,1.5f},{5.6f,2.5f,3.9f,1.1f},{5.9f,3.2f,4.8f,1.8f},
    {6.1f,2.8f,4.0f,1.3f},{6.3f,2.5f,4.9f,1.5f},{6.1f,2.8f,4.7f,1.2f},
    {6.4f,2.9f,4.3f,1.3f},{6.6f,3.0f,4.4f,1.4f},{6.8f,2.8f,4.8f,1.4f},
    {6.7f,3.0f,5.0f,1.7f},{6.0f,2.9f,4.5f,1.5f},{5.7f,2.6f,3.5f,1.0f},
    {5.5f,2.4f,3.8f,1.1f},{5.5f,2.4f,3.7f,1.0f},{5.8f,2.7f,3.9f,1.2f},
    {6.0f,2.7f,5.1f,1.6f},{5.4f,3.0f,4.5f,1.5f},{6.0f,3.4f,4.5f,1.6f},
    {6.7f,3.1f,4.7f,1.5f},{6.3f,2.3f,4.4f,1.3f},{5.6f,3.0f,4.1f,1.3f},
    {5.5f,2.5f,4.0f,1.3f},{5.5f,2.6f,4.4f,1.2f},{6.1f,3.0f,4.6f,1.4f},
    {5.8f,2.6f,4.0f,1.2f},{5.0f,2.3f,3.3f,1.0f},{5.6f,2.7f,4.2f,1.3f},
    {5.7f,3.0f,4.2f,1.2f},{5.7f,2.9f,4.2f,1.3f},{6.2f,2.9f,4.3f,1.3f},
    {5.1f,2.5f,3.0f,1.1f},{5.7f,2.8f,4.1f,1.3f},
    /* Iris virginica (2) */
    {6.3f,3.3f,6.0f,2.5f},{5.8f,2.7f,5.1f,1.9f},{7.1f,3.0f,5.9f,2.1f},
    {6.3f,2.9f,5.6f,1.8f},{6.5f,3.0f,5.8f,2.2f},{7.6f,3.0f,6.6f,2.1f},
    {4.9f,2.5f,4.5f,1.7f},{7.3f,2.9f,6.3f,1.8f},{6.7f,2.5f,5.8f,1.8f},
    {7.2f,3.6f,6.1f,2.5f},{6.5f,3.2f,5.1f,2.0f},{6.4f,2.7f,5.3f,1.9f},
    {6.8f,3.0f,5.5f,2.1f},{5.7f,2.5f,5.0f,2.0f},{5.8f,2.8f,5.1f,2.4f},
    {6.4f,3.2f,5.3f,2.3f},{6.5f,3.0f,5.5f,1.8f},{7.7f,3.8f,6.7f,2.2f},
    {7.7f,2.6f,6.9f,2.3f},{6.0f,2.2f,5.0f,1.5f},{6.9f,3.2f,5.7f,2.3f},
    {5.6f,2.8f,4.9f,2.0f},{7.7f,2.8f,6.7f,2.0f},{6.3f,2.7f,4.9f,1.8f},
    {6.7f,3.3f,5.7f,2.1f},{7.2f,3.2f,6.0f,1.8f},{6.2f,2.8f,4.8f,1.8f},
    {6.1f,3.0f,4.9f,1.8f},{6.4f,2.8f,5.6f,2.1f},{7.2f,3.0f,5.8f,1.6f},
    {7.4f,2.8f,6.1f,1.9f},{7.9f,3.8f,6.4f,2.0f},{6.4f,2.8f,5.6f,2.2f},
    {6.3f,2.8f,5.1f,1.5f},{6.1f,2.6f,5.6f,1.4f},{7.7f,3.0f,6.1f,2.3f},
    {6.3f,3.4f,5.6f,2.4f},{6.4f,3.1f,5.5f,1.8f},{6.0f,3.0f,4.8f,1.8f},
    {6.9f,3.1f,5.4f,2.1f},{6.7f,3.1f,5.6f,2.4f},{6.9f,3.1f,5.1f,2.3f},
    {5.8f,2.7f,5.1f,1.9f},{6.8f,3.2f,5.9f,2.3f},{6.7f,3.3f,5.7f,2.5f},
    {6.7f,3.0f,5.2f,2.3f},{6.3f,2.5f,5.0f,1.9f},{6.5f,3.0f,5.2f,2.0f},
    {6.2f,3.4f,5.4f,2.3f},{5.9f,3.0f,5.1f,1.8f}
};

static const int iris_labels[N_SAMPLES] = {
    0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2
};

static const char *class_names[N_CLASSES] = {
    "Iris setosa    ",
    "Iris versicolor",
    "Iris virginica "
};

/* ── Normalise features to [0,1] ──────────────────────────────────────── */
static const float feat_min[N_FEATURES] = {4.3f, 2.0f, 1.0f, 0.1f};
static const float feat_max[N_FEATURES] = {7.9f, 4.4f, 6.9f, 2.5f};

static void normalise(const float raw[N_FEATURES], float norm[N_FEATURES])
{
    int f;
    for (f = 0; f < N_FEATURES; f++)
        norm[f] = (raw[f] - feat_min[f]) / (feat_max[f] - feat_min[f]);
}

/* ── Build flat arrays + shuffle ─────────────────────────────────────── */
static void build_dataset(float *inputs, float *targets)
{
    int i, c;
    for (i = 0; i < N_SAMPLES; i++)
    {
        normalise(iris_raw[i], inputs + i * N_FEATURES);
        for (c = 0; c < N_CLASSES; c++)
            targets[i * N_CLASSES + c] = (iris_labels[i] == c) ? 1.0f : 0.0f;
    }
}

static void fisher_yates(int *order, int n)
{
    int i, j, t;
    for (i = n - 1; i > 0; i--)
    { j = rand() % (i + 1); t = order[i]; order[i] = order[j]; order[j] = t; }
}

/* ── Print confusion matrix ───────────────────────────────────────────── */
static void print_confusion(const int *matrix, int n)
{
    int i, j;
    printf("\n  Confusion Matrix (rows = true, cols = predicted):\n\n");
    printf("  %20s", "");
    for (j = 0; j < n; j++) printf("  %-16s", class_names[j]);
    printf("\n  %20s", "");
    for (j = 0; j < n; j++) printf("  %-16s", "----------------");
    putchar('\n');
    for (i = 0; i < n; i++)
    {
        printf("  %-20s", class_names[i]);
        for (j = 0; j < n; j++)
            printf("  %-16d", matrix[i * n + j]);
        putchar('\n');
    }
}

/* ── Main ─────────────────────────────────────────────────────────────── */
int main(void)
{
    float inputs[N_SAMPLES * N_FEATURES];
    float targets[N_SAMPLES * N_CLASSES];
    int   order[N_SAMPLES];

    float train_in[120 * N_FEATURES], train_tgt[120 * N_CLASSES];
    float test_in[30  * N_FEATURES],  test_tgt[30  * N_CLASSES];
    int   n_train = 120, n_test = 30;

    int   confusion[N_CLASSES * N_CLASSES];
    int   i, epoch;
    float mse, acc;
    network_t *net;

    srand((unsigned int)time(NULL));

    build_dataset(inputs, targets);

    /* 80/20 stratified-ish split via shuffle */
    for (i = 0; i < N_SAMPLES; i++) order[i] = i;
    fisher_yates(order, N_SAMPLES);

    for (i = 0; i < n_train; i++)
    {
        memcpy(train_in  + i * N_FEATURES, inputs  + order[i] * N_FEATURES,
               N_FEATURES * sizeof(float));
        memcpy(train_tgt + i * N_CLASSES,  targets + order[i] * N_CLASSES,
               N_CLASSES  * sizeof(float));
    }
    for (i = 0; i < n_test; i++)
    {
        memcpy(test_in  + i * N_FEATURES, inputs  + order[n_train + i] * N_FEATURES,
               N_FEATURES * sizeof(float));
        memcpy(test_tgt + i * N_CLASSES,  targets + order[n_train + i] * N_CLASSES,
               N_CLASSES  * sizeof(float));
    }

    /* Architecture: 4 → 12 → 8 → 3 */
    net = net_allocate(4, N_FEATURES, 12, 8, N_CLASSES);
    net_set_activation(net,    NERVENET_ACTIVATION_TANH);
    net_set_optimizer(net,     NERVENET_OPTIMIZER_ADAM);
    net_set_learning_rate(net, 0.005f);
    net_set_l2_lambda(net,     1e-4f);
    net_initialize_xavier(net);

    printf("Nerve %s — Iris Classification\n", net_get_version());
    printf("Architecture: 4-12-8-3 | Adam + Tanh + L2 | Xavier Init\n");
    printf("Dataset: %d train  /  %d test  (Fisher 1936)\n\n", n_train, n_test);

    for (epoch = 1; epoch <= 2000; epoch++)
    {
        mse = net_train_epoch(net, train_in, train_tgt,
                              n_train, N_FEATURES, N_CLASSES, 16);
        if (epoch % 200 == 0)
        {
            acc = net_compute_accuracy(net, test_in, test_tgt,
                                       n_test, N_FEATURES, N_CLASSES);
            printf("  Epoch %4d | MSE: %.4f | Test Acc: %.1f%%\n",
                   epoch, mse, acc * 100.0f);
        }
    }

    acc = net_compute_accuracy(net, test_in, test_tgt,
                               n_test, N_FEATURES, N_CLASSES);
    printf("\n  Final test accuracy: %.1f%% (%d / %d correct)\n",
           acc * 100.0f, (int)(acc * n_test + 0.5f), n_test);

    memset(confusion, 0, sizeof(confusion));
    net_confusion_matrix(net, test_in, test_tgt,
                         n_test, N_FEATURES, N_CLASSES, N_CLASSES, confusion);
    print_confusion(confusion, N_CLASSES);

    net_free(net);
    return 0;
}
