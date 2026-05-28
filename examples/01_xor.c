/*
 * Nerve — Example 01: XOR
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
 * The simplest possible neural network example.
 * Trains a 2-4-1 MLP to learn the XOR function using Adam + Xavier init.
 *
 * Build:  gcc -O2 01_xor.c -o xor -lm
 * Run:    ./xor
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    /* XOR truth table */
    float inputs[4][2] = {{1,1}, {1,0}, {0,1}, {0,0}};
    float targets[4]   = {  0,     1,     1,     0  };

    float output;
    int   i, epoch;
    float mse;

    network_t *net;

    srand((unsigned int)time(NULL));

    net = net_allocate(3, 2, 4, 1);
    net_set_optimizer(net,  NERVENET_OPTIMIZER_ADAM);
    net_set_learning_rate(net, 0.01f);
    net_initialize_xavier(net);

    printf("Nerve %s — XOR Example\n", net_get_version());
    printf("Architecture: 2-4-1 | Adam | Xavier Init\n\n");

    for (epoch = 1; epoch <= 10000; epoch++)
    {
        mse = 0.0f;
        for (i = 0; i < 4; i++)
        {
            net_compute(net, inputs[i], NULL);
            mse += net_compute_output_error(net, &targets[i]);
            net_train(net);
        }
        mse /= 4.0f;

        if (epoch % 1000 == 0)
            printf("  Epoch %5d | MSE: %.6f\n", epoch, mse);

        if (mse < 1e-4f) break;
    }

    printf("\nResults after %d epochs:\n", epoch);
    printf("  %-12s  %-12s  %-12s\n", "Input", "Target", "Output");
    printf("  %-12s  %-12s  %-12s\n", "------", "------", "------");
    for (i = 0; i < 4; i++)
    {
        net_compute(net, inputs[i], &output);
        printf("  [%.0f, %.0f]       %.1f           %.4f  %s\n",
               inputs[i][0], inputs[i][1], targets[i], output,
               (output > 0.5f) == (int)targets[i] ? "OK" : "FAIL");
    }

    net_free(net);
    return 0;
}
