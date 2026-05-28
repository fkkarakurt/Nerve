/*
 * Nerve — Example 09: Predictive Maintenance
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
 * Real-world industrial use case: classify rotating machinery health
 * from four live sensor readings.
 *
 *   Inputs  : vibration (Hz), temperature (°C), pressure (bar), current (A)
 *   Outputs : Normal | Warning | Critical | Fault  (one-hot)
 *
 * Workflow
 *   1. Train a 4-16-8-4 network on 60 labeled sensor logs
 *   2. Evaluate on a held-out 20-sample test set
 *   3. Simulate a live monitoring session — 12 readings that tell the
 *      story of a machine degrading from Normal to Fault over time
 *
 * Architecture: 4-16-8-4 | Adam + ReLU | He Init
 *
 * Build:  gcc -O2 09_predictive_maintenance.c -o maintenance -lm
 * Run:    ./maintenance
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Dataset ───────────────────────────────────────────────────────────────
 * 80 labeled sensor readings: 20 per class.
 * First 60 = training (15 per class), last 20 = test (5 per class).
 * Columns: vibration (Hz), temperature (°C), pressure (bar), current (A)
 * Label  : 0=Normal  1=Warning  2=Critical  3=Fault
 * ────────────────────────────────────────────────────────────────────────── */
static const float sensors[80][5] = {
    /* Normal ─ train */
    {12.0f, 52.0f,  4.2f,  9.1f, 0}, {18.0f, 58.0f,  5.1f, 10.3f, 0},
    { 9.0f, 63.0f,  4.8f,  9.8f, 0}, {22.0f, 67.0f,  5.6f, 11.2f, 0},
    {15.0f, 55.0f,  3.9f,  9.5f, 0}, {11.0f, 61.0f,  4.4f, 10.1f, 0},
    {20.0f, 70.0f,  5.0f, 10.8f, 0}, { 8.0f, 48.0f,  3.5f,  8.9f, 0},
    {25.0f, 72.0f,  6.1f, 11.8f, 0}, {14.0f, 60.0f,  4.7f, 10.5f, 0},
    {17.0f, 65.0f,  5.3f, 11.0f, 0}, {10.0f, 54.0f,  4.1f,  9.3f, 0},
    {23.0f, 68.0f,  5.8f, 11.5f, 0}, {13.0f, 57.0f,  4.5f,  9.7f, 0},
    {19.0f, 64.0f,  4.9f, 10.6f, 0},
    /* Warning ─ train */
    {38.0f, 83.0f,  8.2f, 15.5f, 1}, {45.0f, 88.0f,  8.9f, 16.8f, 1},
    {52.0f, 92.0f,  9.3f, 17.2f, 1}, {40.0f, 85.0f,  8.5f, 16.1f, 1},
    {36.0f, 80.0f,  8.1f, 15.3f, 1}, {55.0f, 95.0f,  9.8f, 18.0f, 1},
    {48.0f, 90.0f,  9.1f, 17.5f, 1}, {42.0f, 87.0f,  8.7f, 16.5f, 1},
    {35.0f, 82.0f,  8.3f, 15.8f, 1}, {60.0f, 98.0f,  9.6f, 18.4f, 1},
    {43.0f, 86.0f,  8.6f, 16.3f, 1}, {50.0f, 93.0f,  9.2f, 17.9f, 1},
    {37.0f, 81.0f,  8.2f, 15.6f, 1}, {56.0f, 96.0f,  9.7f, 18.2f, 1},
    {44.0f, 89.0f,  8.8f, 16.7f, 1},
    /* Critical ─ train */
    {72.0f, 108.0f, 11.2f, 22.5f, 2}, {80.0f, 115.0f, 12.0f, 24.1f, 2},
    {88.0f, 122.0f, 12.8f, 26.3f, 2}, {75.0f, 110.0f, 11.5f, 23.0f, 2},
    {70.0f, 106.0f, 11.0f, 22.0f, 2}, {95.0f, 128.0f, 13.5f, 28.0f, 2},
    {83.0f, 118.0f, 12.3f, 25.2f, 2}, {78.0f, 113.0f, 11.8f, 23.8f, 2},
    {68.0f, 105.0f, 11.1f, 21.8f, 2}, {98.0f, 130.0f, 13.8f, 29.0f, 2},
    {82.0f, 117.0f, 12.2f, 25.0f, 2}, {90.0f, 124.0f, 13.0f, 27.0f, 2},
    {73.0f, 109.0f, 11.3f, 22.7f, 2}, {85.0f, 120.0f, 12.5f, 25.8f, 2},
    {76.0f, 111.0f, 11.6f, 23.2f, 2},
    /* Fault ─ train */
    {115.0f, 145.0f, 15.2f, 32.5f, 3}, {130.0f, 158.0f, 16.5f, 34.8f, 3},
    {148.0f, 172.0f, 17.8f, 36.2f, 3}, {122.0f, 152.0f, 16.0f, 33.5f, 3},
    {112.0f, 142.0f, 15.0f, 32.0f, 3}, {165.0f, 188.0f, 19.2f, 38.5f, 3},
    {140.0f, 165.0f, 17.2f, 35.5f, 3}, {135.0f, 161.0f, 16.8f, 35.0f, 3},
    {118.0f, 148.0f, 15.5f, 33.0f, 3}, {170.0f, 195.0f, 19.5f, 39.2f, 3},
    {142.0f, 167.0f, 17.3f, 35.8f, 3}, {155.0f, 180.0f, 18.5f, 37.5f, 3},
    {120.0f, 150.0f, 15.8f, 33.2f, 3}, {145.0f, 170.0f, 17.5f, 36.0f, 3},
    {128.0f, 156.0f, 16.3f, 34.5f, 3},
    /* ── test rows (5 per class) ── */
    /* Normal ─ test */
    { 7.0f, 50.0f,  3.8f,  8.7f, 0}, {21.0f, 71.0f,  5.5f, 11.3f, 0},
    {16.0f, 62.0f,  4.6f, 10.2f, 0}, {24.0f, 69.0f,  5.9f, 11.7f, 0},
    { 6.0f, 46.0f,  3.3f,  8.5f, 0},
    /* Warning ─ test */
    {39.0f, 84.0f,  8.4f, 16.0f, 1}, {58.0f, 97.0f,  9.5f, 18.6f, 1},
    {46.0f, 91.0f,  9.0f, 17.4f, 1}, {41.0f, 86.0f,  8.5f, 16.2f, 1},
    {62.0f, 99.0f,  9.9f, 19.0f, 1},
    /* Critical ─ test */
    {79.0f, 114.0f, 11.9f, 24.0f, 2}, {92.0f, 126.0f, 13.2f, 27.5f, 2},
    {74.0f, 110.0f, 11.4f, 22.9f, 2}, {96.0f, 129.0f, 13.6f, 28.5f, 2},
    {100.0f,132.0f, 14.0f, 30.0f, 2},
    /* Fault ─ test */
    {160.0f, 185.0f, 18.8f, 38.0f, 3}, {138.0f, 163.0f, 17.0f, 35.2f, 3},
    {125.0f, 154.0f, 16.2f, 33.8f, 3}, {168.0f, 192.0f, 19.3f, 38.8f, 3},
    {175.0f, 200.0f, 19.8f, 40.0f, 3}
};

/* Live readings that simulate machine degradation over time */
static const float live[12][4] = {
    { 14.0f,  58.0f,  4.5f,  9.8f},  /* 1  — Normal     */
    { 18.0f,  65.0f,  5.2f, 10.5f},  /* 2  — Normal     */
    { 25.0f,  72.0f,  5.8f, 11.2f},  /* 3  — Normal     */
    { 32.0f,  77.0f,  7.1f, 13.0f},  /* 4  — borderline */
    { 42.0f,  84.0f,  8.4f, 15.8f},  /* 5  — Warning    */
    { 51.0f,  92.0f,  9.1f, 17.2f},  /* 6  — Warning    */
    { 63.0f, 100.0f, 10.0f, 19.5f},  /* 7  — Warning↑   */
    { 74.0f, 109.0f, 11.3f, 22.8f},  /* 8  — Critical   */
    { 85.0f, 119.0f, 12.4f, 25.5f},  /* 9  — Critical   */
    { 96.0f, 130.0f, 13.8f, 28.9f},  /* 10 — Critical↑  */
    {118.0f, 148.0f, 15.8f, 33.2f},  /* 11 — Fault      */
    {142.0f, 168.0f, 17.4f, 36.1f}   /* 12 — Fault      */
};

/* Normalization bounds */
#define VIB_LO   0.0f
#define VIB_HI 200.0f
#define TMP_LO   0.0f
#define TMP_HI 200.0f
#define PRS_LO   0.0f
#define PRS_HI  20.0f
#define CUR_LO   0.0f
#define CUR_HI  40.0f

#define N_TRAIN  60
#define N_TEST   20
#define N_IN      4
#define N_OUT     4
#define N_CLS     4

static const char *STATUS[N_CLS] = {"NORMAL  ", "WARNING ", "CRITICAL", "FAULT   "};
static const char *ICON[N_CLS]   = {"[  OK  ]", "[ WARN ]", "[ CRIT ]", "[FAULT ]"};

static float norm(float x, float lo, float hi)
{ return (x - lo) / (hi - lo); }

static void build_arrays(float *inp, float *tgt, int offset, int n)
{
    int i, cls;
    for (i = 0; i < n; i++) {
        const float *r = sensors[offset + i];
        inp[i*N_IN+0] = norm(r[0], VIB_LO, VIB_HI);
        inp[i*N_IN+1] = norm(r[1], TMP_LO, TMP_HI);
        inp[i*N_IN+2] = norm(r[2], PRS_LO, PRS_HI);
        inp[i*N_IN+3] = norm(r[3], CUR_LO, CUR_HI);
        cls = (int)r[4];
        tgt[i*N_OUT+0] = (cls == 0) ? 1.0f : 0.0f;
        tgt[i*N_OUT+1] = (cls == 1) ? 1.0f : 0.0f;
        tgt[i*N_OUT+2] = (cls == 2) ? 1.0f : 0.0f;
        tgt[i*N_OUT+3] = (cls == 3) ? 1.0f : 0.0f;
    }
}

/* Print a proportional bar for a sensor value (0..hi) */
static void bar(float val, float hi, int width)
{
    int filled = (int)(val / hi * (float)width + 0.5f);
    int i;
    if (filled > width) filled = width;
    putchar('[');
    for (i = 0; i < filled; i++) putchar('=');
    for (i = filled; i < width; i++) putchar(' ');
    putchar(']');
}

static void print_reading(int idx, const float raw4[4], float conf[4])
{
    int cls = 0, j;
    float best = conf[0];
    for (j = 1; j < N_CLS; j++) if (conf[j] > best) { best = conf[j]; cls = j; }

    printf("\n  Reading %2d\n", idx);
    printf("    Vibration   %6.1f Hz   ", raw4[0]);
    bar(raw4[0], VIB_HI, 24); printf("\n");
    printf("    Temperature %6.1f C    ", raw4[1]);
    bar(raw4[1], TMP_HI, 24); printf("\n");
    printf("    Pressure    %6.1f bar  ", raw4[2]);
    bar(raw4[2], PRS_HI, 24); printf("\n");
    printf("    Current     %6.1f A    ", raw4[3]);
    bar(raw4[3], CUR_HI, 24); printf("\n");

    printf("    %s %s  Confidence: %.1f%%\n",
           ICON[cls], STATUS[cls], best * 100.0f);
    printf("    Normal: %4.1f%%  Warning: %4.1f%%  Critical: %4.1f%%  Fault: %4.1f%%\n",
           conf[0]*100, conf[1]*100, conf[2]*100, conf[3]*100);
}

int main(void)
{
    float      train_in[N_TRAIN * N_IN],  train_tgt[N_TRAIN * N_OUT];
    float      test_in[N_TEST  * N_IN],   test_tgt[N_TEST  * N_OUT];
    float      norm_in[N_IN], output[N_OUT];
    network_t *net;
    float      mse, acc;
    int        cm[N_CLS * N_CLS];
    int        i, row, col;

    build_arrays(train_in, train_tgt, 0,       N_TRAIN);
    build_arrays(test_in,  test_tgt,  N_TRAIN, N_TEST);

    printf("Nerve %s -- Predictive Maintenance\n", net_get_version());
    printf("Architecture: 4-16-8-4 | Adam + ReLU | He Init\n");
    printf("Inputs : Vibration (Hz), Temperature (C),");
    printf(" Pressure (bar), Current (A)\n");
    printf("Outputs: Normal / Warning / Critical / Fault\n");
    printf("Dataset: %d train / %d test\n\n", N_TRAIN, N_TEST);

    srand(42);
    net = net_allocate(4, N_IN, 16, 8, N_OUT);
    net_set_optimizer(net,   NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net,  NERVENET_ACTIVATION_RELU);
    net_initialize_he(net);
    net_set_learning_rate(net, 0.005f);
    net_set_l2_lambda(net, 1e-4f);

    printf("Training...\n");
    for (i = 1; i <= 1000; i++) {
        mse = net_train_epoch(net, train_in, train_tgt,
                              N_TRAIN, N_IN, N_OUT, N_TRAIN);
        if (i % 200 == 0) {
            acc = net_compute_accuracy(net, test_in, test_tgt,
                                       N_TEST, N_IN, N_OUT);
            printf("  Epoch %4d | MSE: %.4f | Test Acc: %.0f%%\n",
                   i, mse, acc * 100.0f);
        }
    }

    acc = net_compute_accuracy(net, test_in, test_tgt, N_TEST, N_IN, N_OUT);
    printf("\nTest accuracy: %.0f%%  (%d / %d)\n\n",
           acc * 100.0f, (int)(acc * N_TEST + 0.5f), N_TEST);

    memset(cm, 0, sizeof(cm));
    net_confusion_matrix(net, test_in, test_tgt,
                         N_TEST, N_IN, N_OUT, N_CLS, cm);
    printf("Confusion Matrix:\n");
    printf("              Normal  Warning  Critical   Fault\n");
    for (row = 0; row < N_CLS; row++) {
        printf("  %-10s", STATUS[row]);
        for (col = 0; col < N_CLS; col++)
            printf("  %6d   ", cm[row*N_CLS+col]);
        printf("\n");
    }

    /* ── Live monitoring simulation ─────────────────────────────────────── */
    printf("\n");
    printf("================================================================\n");
    printf("  LIVE MONITORING SIMULATION\n");
    printf("  Simulating machine degradation over 12 readings...\n");
    printf("================================================================\n");

    for (i = 0; i < 12; i++) {
        norm_in[0] = norm(live[i][0], VIB_LO, VIB_HI);
        norm_in[1] = norm(live[i][1], TMP_LO, TMP_HI);
        norm_in[2] = norm(live[i][2], PRS_LO, PRS_HI);
        norm_in[3] = norm(live[i][3], CUR_LO, CUR_HI);
        net_compute(net, norm_in, output);
        print_reading(i + 1, live[i], output);
    }

    printf("\n================================================================\n");
    printf("  Simulation complete.  Machine degradation detected:\n");
    printf("  Normal -> Warning -> Critical -> FAULT\n");
    printf("================================================================\n");

    net_free(net);
    return 0;
}
