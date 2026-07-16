/*
 * Nerve — Example 02: Function Approximation
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
 *
 * Demonstrates the universal approximation theorem:
 * a small MLP learns to approximate sin(x) on [-π, π].
 *
 * After training, renders a side-by-side ASCII plot of
 * the true function vs the learned approximation.
 *
 * Build:  gcc -O2 02_sine.c -o sine -lm
 * Run:    ./sine
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── ASCII plot ────────────────────────────────────────────────────────── */
#define PLOT_W  64
#define PLOT_H  20

static void render_plot(network_t *net, int n_samples)
{
    char canvas[PLOT_H][PLOT_W + 1];
    int  x, y, col, row;
    float t, inp, out, truth;

    memset(canvas, ' ', sizeof(canvas));
    for (y = 0; y < PLOT_H; y++) canvas[y][PLOT_W] = '\0';

    for (x = 0; x < n_samples; x++)
    {
        t     = (float)x / (float)(n_samples - 1); /* [0, 1] */
        inp   = t;                                   /* network input */
        truth = (float)sin(((double)t * 2.0 - 1.0) * 3.14159265);

        net_compute(net, &inp, &out);
        /* re-scale output from [0,1] to [-1,1] */
        out = out * 2.0f - 1.0f;

        col = (int)(t * (float)(PLOT_W - 1));

        /* truth — mark with '|' */
        row = (int)(((1.0f - truth) / 2.0f) * (float)(PLOT_H - 1));
        if (row >= 0 && row < PLOT_H && col >= 0 && col < PLOT_W)
            canvas[row][col] = '|';

        /* network — mark with '*' (overwrites '|' → shows as '#' on overlap) */
        row = (int)(((1.0f - out) / 2.0f) * (float)(PLOT_H - 1));
        if (row >= 0 && row < PLOT_H && col >= 0 && col < PLOT_W)
        {
            if (canvas[row][col] == '|')
                canvas[row][col] = '#'; /* overlap */
            else
                canvas[row][col] = '*';
        }
    }

    /* horizontal axis at y = 0 */
    row = PLOT_H / 2;
    for (col = 0; col < PLOT_W; col++)
        if (canvas[row][col] == ' ') canvas[row][col] = '-';

    printf("\n  sin(x) approximation — Nerve %s\n", net_get_version());
    printf("  %s\n", "Legend:  | truth    * network    # overlap    - zero");
    printf("  +1 ");
    for (col = 0; col < PLOT_W; col++) putchar('-');
    putchar('\n');
    for (y = 0; y < PLOT_H; y++)
    {
        if (y == 0)            printf("     |");
        else if (y == PLOT_H/2) printf("   0 |");
        else if (y == PLOT_H-1) printf("  -1 |");
        else                   printf("     |");
        printf("%s|\n", canvas[y]);
    }
    printf("     +");
    for (col = 0; col < PLOT_W; col++) putchar('-');
    printf("+\n");
    printf("       -pi%*s+pi\n", PLOT_W - 3, "");
}

/* ── Generate dataset ──────────────────────────────────────────────────── */
#define N_TRAIN  200
#define N_PLOT   200

int main(void)
{
    float train_in[N_TRAIN];
    float train_out[N_TRAIN];  /* target in [0, 1] */
    int   i, epoch;
    float t, mse, best_mse = 1e9f;
    network_t *net;

    nerve_seed(42);   /* fixed: this run is identical on every machine */

    /* Build dataset: x ∈ [0,1] → y = (sin(2πx-π) + 1) / 2  ∈ [0,1] */
    for (i = 0; i < N_TRAIN; i++)
    {
        t             = (float)i / (float)(N_TRAIN - 1);
        train_in[i]   = t;
        train_out[i]  = ((float)sin((2.0 * 3.14159265 * t) - 3.14159265) + 1.0f) * 0.5f;
    }

    /* Architecture: 1 → 16 → 16 → 1  (tanh hidden, sigmoid output) */
    net = net_allocate(4, 1, 16, 16, 1);
    net_set_activation(net,    NERVENET_ACTIVATION_TANH);
    net_set_optimizer(net,     NERVENET_OPTIMIZER_ADAM);
    net_set_learning_rate(net, 0.005f);
    net_initialize_xavier(net);

    printf("Nerve %s — sin(x) Approximation\n", net_get_version());
    printf("Architecture: 1-16-16-1 | Adam + Tanh | Xavier Init\n\n");

    for (epoch = 1; epoch <= 5000; epoch++)
    {
        mse = net_train_epoch(net, train_in, train_out,
                              N_TRAIN, 1, 1, 32);

        if (mse < best_mse) best_mse = mse;

        if (epoch % 500 == 0)
        {
            printf("  Epoch %4d | MSE: %.6f", epoch, mse);
            if (epoch % 1000 == 0)
            {
                int bar, prog = (epoch * 20) / 5000;
                printf("  [");
                for (bar = 0; bar < 20; bar++)
                    putchar(bar < prog ? '=' : ' ');
                printf("]");
            }
            putchar('\n');
        }

        if (mse < 2e-5f) break;
    }

    printf("\nBest MSE: %.6f\n", best_mse);

    render_plot(net, N_PLOT);

    /* Spot checks */
    {
        float angles[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
        printf("\n  Spot checks:\n");
        printf("  %-8s  %-10s  %-10s  %-10s\n",
               "x (norm)", "sin(x)", "network", "abs error");
        for (i = 0; i < 5; i++)
        {
            float x   = angles[i];
            float tru = ((float)sin((2.0 * 3.14159265 * x) - 3.14159265) + 1.0f) * 0.5f;
            float out;
            net_compute(net, &x, &out);
            printf("  %-8.2f  %-10.4f  %-10.4f  %-10.4f\n",
                   x, tru, out, (float)fabs(out - tru));
        }
    }

    net_free(net);
    return 0;
}
