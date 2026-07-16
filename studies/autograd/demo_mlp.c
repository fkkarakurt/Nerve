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
 * demo_mlp.c — SPIKE proof: an MLP trained entirely through autodiff.
 *
 * The forward pass below is just composition of ops (matmul, bias, relu,
 * softmax+cross-entropy). There is NO hand-written gradient anywhere — every
 * dW and db comes out of ng_backward(). This is the Phase 2 proof point:
 * a new layer becomes *expressible* without deriving its backprop by hand.
 *
 * Build:  gcc -O2 demo_mlp.c -o demo_mlp -lm
 */
#define NERVE_GRAD_IMPLEMENTATION
#include "nerve_grad.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N   12   /* samples */
#define IN   2   /* features */
#define H   16   /* hidden  */
#define C    3   /* classes */

static float frand(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }

static void he_init(tensor *t)
{
    int i, sz = t->rows * t->cols;
    float r = (float)sqrt(6.0 / (double)t->rows);
    for (i = 0; i < sz; i++) t->data[i] = 2.0f * r * frand();
}

int main(void)
{
    /* Three 2D clusters, 4 points each. */
    float X[N*IN] = {
        0.1f,0.1f, 0.2f,0.0f, 0.0f,0.2f, 0.15f,0.05f,   /* class 0 */
        0.9f,0.9f, 1.0f,0.8f, 0.8f,1.0f, 0.95f,0.85f,   /* class 1 */
        0.1f,0.9f, 0.0f,1.0f, 0.2f,0.8f, 0.05f,0.95f    /* class 2 */
    };
    int y[N] = {0,0,0,0, 1,1,1,1, 2,2,2,2};

    tensor *W1, *b1, *W2, *b2, *params[4];
    int epoch, i, j;

    srand(7);
    W1 = t_param(IN, H);  b1 = t_param(1, H);
    W2 = t_param(H,  C);  b2 = t_param(1, C);
    he_init(W1); he_init(W2);
    params[0]=W1; params[1]=b1; params[2]=W2; params[3]=b2;

    printf("Autodiff MLP  %d-%d-%d  | softmax + cross-entropy | no hand-coded gradients\n\n",
           IN, H, C);

    for (epoch = 1; epoch <= 400; epoch++) {
        tensor *xin, *h, *logits, *loss;

        ng_zero_grad(params, 4);

        /* ---- forward: pure op composition, batch of all N rows ---- */
        xin = t_param(N, IN);                 /* input as a constant leaf */
        for (i = 0; i < N*IN; i++) xin->data[i] = X[i];
        h      = t_relu(t_add_bias(t_matmul(xin, W1), b1));
        logits = t_add_bias(t_matmul(h, W2), b2);
        loss   = t_softmax_cross_entropy(logits, y);

        /* ---- backward: every gradient derived automatically ---- */
        ng_backward(loss);
        ng_sgd_step(params, 4, 0.5f);

        if (epoch % 50 == 0 || epoch == 1)
            printf("  epoch %3d   loss = %.5f\n", epoch, loss->data[0]);

        free(xin->data); free(xin->grad); free(xin);  /* input leaf is ours */
        ng_end_step();
    }

    /* ---- final accuracy via one more forward pass ---- */
    {
        tensor *xin = t_param(N, IN), *h, *logits;
        int correct = 0;
        for (i = 0; i < N*IN; i++) xin->data[i] = X[i];
        h      = t_relu(t_add_bias(t_matmul(xin, W1), b1));
        logits = t_add_bias(t_matmul(h, W2), b2);
        printf("\n  predictions:\n");
        for (i = 0; i < N; i++) {
            int best = 0;
            for (j = 1; j < C; j++)
                if (logits->data[i*C+j] > logits->data[i*C+best]) best = j;
            correct += (best == y[i]);
            printf("    sample %2d  pred=%d truth=%d %s\n",
                   i, best, y[i], best==y[i] ? "" : "  <-- wrong");
        }
        printf("\n  accuracy: %d/%d\n", correct, N);
        free(xin->data); free(xin->grad); free(xin);
        ng_end_step();
    }
    return 0;
}
