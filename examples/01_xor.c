/*
 * Nerve — Example 01: XOR
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

int main(int argc, char **argv)
{
    /* XOR truth table */
    float inputs[4][2] = {{1,1}, {1,0}, {0,1}, {0,0}};
    float targets[4]   = {  0,     1,     1,     0  };

    float output;
    int   i, epoch;
    float mse;

    network_t *net;

    /* A fixed seed makes this run identical on every machine and every libc.
     * Pass a seed on the command line to explore other initialisations. */
    nerve_seed(argc > 1 ? strtoul(argv[1], NULL, 10) : 42UL);

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
