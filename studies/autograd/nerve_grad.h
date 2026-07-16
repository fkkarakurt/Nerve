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
 * nerve_grad.h — SPIKE: minimal reverse-mode autodiff for Nerve
 * ---------------------------------------------------------------------------
 * This is a de-risking prototype for ROADMAP Phase 2, NOT yet part of the
 * shipped library. It answers one question: can a readable, single-header
 * reverse-mode autodiff engine — the spine that makes convolutions, attention
 * and a pure-C Transformer *expressible instead of hand-coded* — stay tiny?
 *
 * The whole engine is below ~200 lines. Every operation records itself on a
 * tape at forward time; ng_backward() walks the tape in reverse and fills in
 * each tensor's gradient. No gradient is ever written by hand: you compose
 * forward ops, call backward, and the chain rule falls out.
 *
 * Tensors are 2D (rows x cols) — enough to express an MLP and, later, the
 * matmuls at the heart of attention. Single translation unit, libm only.
 *
 *   #define NERVE_GRAD_IMPLEMENTATION
 *   #include "nerve_grad.h"
 */
#ifndef NERVE_GRAD_H
#define NERVE_GRAD_H

typedef struct tensor tensor;
struct tensor {
    int      rows, cols;
    float   *data;      /* rows*cols values                                   */
    float   *grad;      /* rows*cols accumulated dL/d(this)                    */
    int      op;        /* how this tensor was produced (ng__op)              */
    tensor  *a, *b;     /* parents (operands) for the backward pass           */
    const int *targets; /* class indices, for the softmax+cross-entropy op    */
    float   *cache;     /* op scratch kept from forward (softmax probs)        */
    int      param;     /* 1 = persistent weight, 0 = transient graph node    */
};

/* Parameters (persist across steps); leaves you own and update yourself. */
tensor *t_param(int rows, int cols);

/* Differentiable ops — each builds a new tensor and records itself. */
tensor *t_matmul(tensor *a, tensor *b);   /* (m,k)x(k,n) -> (m,n)             */
tensor *t_add_bias(tensor *x, tensor *b); /* x(m,n) + b(1,n) broadcast        */
tensor *t_relu(tensor *x);
tensor *t_softmax_cross_entropy(tensor *logits, const int *targets); /* ->(1,1)*/

void  ng_backward(tensor *loss);          /* seed 1.0, propagate to all grads */
void  ng_zero_grad(tensor **params, int n);
void  ng_sgd_step(tensor **params, int n, float lr);
void  ng_end_step(void);                  /* free this step's graph nodes     */

#endif /* NERVE_GRAD_H */

/* ========================================================================= */
#ifdef NERVE_GRAD_IMPLEMENTATION
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

enum ng__op { OP_LEAF, OP_MATMUL, OP_ADD_BIAS, OP_RELU, OP_SOFTMAX_CE };

#ifndef NG_MAX_NODES
#define NG_MAX_NODES 200000
#endif
static tensor *ng__tape[NG_MAX_NODES];
static int     ng__tape_n = 0;

static tensor *ng__alloc(int rows, int cols)
{
    tensor *t = (tensor *)calloc(1, sizeof *t);
    assert(t);
    t->rows = rows; t->cols = cols;
    t->data = (float *)calloc((size_t)rows * cols, sizeof(float));
    t->grad = (float *)calloc((size_t)rows * cols, sizeof(float));
    t->op   = OP_LEAF;
    return t;
}

/* Transient node: lives on the tape, freed at end of step. */
static tensor *ng__node(int rows, int cols, int op, tensor *a, tensor *b)
{
    tensor *t = ng__alloc(rows, cols);
    t->op = op; t->a = a; t->b = b;
    assert(ng__tape_n < NG_MAX_NODES);
    ng__tape[ng__tape_n++] = t;
    return t;
}

tensor *t_param(int rows, int cols)
{
    tensor *t = ng__alloc(rows, cols);
    t->param = 1;
    return t;
}

/* C(m,n) = A(m,k) * B(k,n) */
tensor *t_matmul(tensor *a, tensor *b)
{
    int m = a->rows, k = a->cols, n = b->cols, i, j, p;
    tensor *c;
    assert(a->cols == b->rows);
    c = ng__node(m, n, OP_MATMUL, a, b);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            float s = 0.0f;
            for (p = 0; p < k; p++) s += a->data[i*k+p] * b->data[p*n+j];
            c->data[i*n+j] = s;
        }
    return c;
}

/* y(m,n) = x(m,n) + b(1,n), bias broadcast over rows */
tensor *t_add_bias(tensor *x, tensor *bias)
{
    int m = x->rows, n = x->cols, i, j;
    tensor *y;
    assert(bias->rows == 1 && bias->cols == n);
    y = ng__node(m, n, OP_ADD_BIAS, x, bias);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            y->data[i*n+j] = x->data[i*n+j] + bias->data[j];
    return y;
}

tensor *t_relu(tensor *x)
{
    int i, sz = x->rows * x->cols;
    tensor *y = ng__node(x->rows, x->cols, OP_RELU, x, NULL);
    for (i = 0; i < sz; i++) y->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
    return y;
}

/* Fused softmax + categorical cross-entropy over a batch of `m` rows.
 * Returns mean loss as a (1,1) tensor; stashes the per-row softmax probs in
 * `cache` so backward is the clean (probs - onehot)/m. */
tensor *t_softmax_cross_entropy(tensor *logits, const int *targets)
{
    int m = logits->rows, n = logits->cols, i, j;
    tensor *loss = ng__node(1, 1, OP_SOFTMAX_CE, logits, NULL);
    float total = 0.0f;
    loss->targets = targets;
    loss->cache   = (float *)calloc((size_t)m * n, sizeof(float));
    for (i = 0; i < m; i++) {
        float maxv = logits->data[i*n], sum = 0.0f;
        for (j = 1; j < n; j++)
            if (logits->data[i*n+j] > maxv) maxv = logits->data[i*n+j];
        for (j = 0; j < n; j++) {
            float e = (float)exp((double)(logits->data[i*n+j] - maxv));
            loss->cache[i*n+j] = e; sum += e;
        }
        for (j = 0; j < n; j++) loss->cache[i*n+j] /= sum;
        {
            float p = loss->cache[i*n + targets[i]];
            total -= (float)log((double)(p > 1e-12f ? p : 1e-12f));
        }
    }
    loss->data[0] = total / (float)m;
    return loss;
}

/* ── Backward: walk the tape in reverse, push grads into parents ─────────── */
void ng_backward(tensor *loss)
{
    int t, i, j, p;
    loss->grad[0] = 1.0f;
    for (t = ng__tape_n - 1; t >= 0; t--) {
        tensor *c = ng__tape[t];
        switch (c->op) {
        case OP_MATMUL: {            /* C=A*B : dA+=dC*B^T, dB+=A^T*dC */
            tensor *a = c->a, *b = c->b;
            int m = a->rows, k = a->cols, n = b->cols;
            for (i = 0; i < m; i++)
                for (p = 0; p < k; p++) {
                    float s = 0.0f;
                    for (j = 0; j < n; j++) s += c->grad[i*n+j] * b->data[p*n+j];
                    a->grad[i*k+p] += s;
                }
            for (p = 0; p < k; p++)
                for (j = 0; j < n; j++) {
                    float s = 0.0f;
                    for (i = 0; i < m; i++) s += a->data[i*k+p] * c->grad[i*n+j];
                    b->grad[p*n+j] += s;
                }
            break;
        }
        case OP_ADD_BIAS: {          /* y=x+b : dx+=dy, db[j]+=sum_i dy[i,j] */
            tensor *x = c->a, *b = c->b;
            int m = x->rows, n = x->cols;
            for (i = 0; i < m; i++)
                for (j = 0; j < n; j++) {
                    x->grad[i*n+j] += c->grad[i*n+j];
                    b->grad[j]     += c->grad[i*n+j];
                }
            break;
        }
        case OP_RELU: {              /* y=relu(x) : dx += (x>0)?dy:0 */
            tensor *x = c->a;
            int sz = x->rows * x->cols;
            for (i = 0; i < sz; i++)
                if (x->data[i] > 0.0f) x->grad[i] += c->grad[i];
            break;
        }
        case OP_SOFTMAX_CE: {        /* dlogits += (probs - onehot)/m */
            tensor *z = c->a;
            int m = z->rows, n = z->cols;
            float scale = c->grad[0] / (float)m;
            for (i = 0; i < m; i++)
                for (j = 0; j < n; j++) {
                    float g = c->cache[i*n+j] - (j == c->targets[i] ? 1.0f : 0.0f);
                    z->grad[i*n+j] += scale * g;
                }
            break;
        }
        default: break;
        }
    }
}

void ng_zero_grad(tensor **params, int n)
{
    int i;
    for (i = 0; i < n; i++)
        memset(params[i]->grad, 0, (size_t)params[i]->rows * params[i]->cols * sizeof(float));
}

void ng_sgd_step(tensor **params, int n, float lr)
{
    int i, j, sz;
    for (i = 0; i < n; i++) {
        sz = params[i]->rows * params[i]->cols;
        for (j = 0; j < sz; j++)
            params[i]->data[j] -= lr * params[i]->grad[j];
    }
}

/* Free all transient nodes built this step; parameters persist. */
void ng_end_step(void)
{
    int i;
    for (i = 0; i < ng__tape_n; i++) {
        free(ng__tape[i]->data);
        free(ng__tape[i]->grad);
        if (ng__tape[i]->cache) free(ng__tape[i]->cache);
        free(ng__tape[i]);
    }
    ng__tape_n = 0;
}

#endif /* NERVE_GRAD_IMPLEMENTATION */
