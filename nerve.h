/*
 * nerve.h — Single-header neural network library
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
 * ─────────────────────────────────────────────────────────────────────────
 *
 *  NERVE  —  Zero-dependency multilayer perceptron in ANSI C
 *
 *  Features
 *    • SGD + momentum  and  Adam optimizer
 *    • Sigmoid, Tanh, ReLU, Leaky ReLU hidden activations
 *    • Xavier (Glorot) and He weight initialisation
 *    • L2 regularisation  •  Dropout
 *    • Mini-batch training helpers
 *    • Text and binary model save / load
 *    • Classification accuracy  •  Confusion matrix
 *
 *  References
 *    [1] Rumelhart et al. (1986). Learning representations by back-
 *        propagating errors. Nature 323, 533–536.
 *    [2] Glorot & Bengio (2010). Understanding the difficulty of training
 *        deep feedforward networks. AISTATS.
 *    [3] He et al. (2015). Delving deep into rectifiers. ICCV.
 *    [4] Kingma & Ba (2014). Adam: A method for stochastic optimization.
 *        ICLR 2015.
 *    [5] Srivastava et al. (2014). Dropout: A simple way to prevent neural
 *        networks from overfitting. JMLR 15, 1929–1958.
 *
 * ─────────────────────────────────────────────────────────────────────────
 *
 *  QUICK START
 *
 *    In exactly ONE .c / .cpp file before including this header:
 *
 *        #define NERVE_IMPLEMENTATION
 *        #include "nerve.h"
 *
 *    In all other translation units just:
 *
 *        #include "nerve.h"
 *
 *  EXAMPLE — XOR in < 20 lines
 *
 *        #define NERVE_IMPLEMENTATION
 *        #include "nerve.h"
 *        #include <stdlib.h>
 *        #include <time.h>
 *
 *        int main(void) {
 *            float in[]  = {1,1, 1,0, 0,1, 0,0};
 *            float tgt[] = {0,   1,   1,   0  };
 *            network_t *net = net_allocate(3, 2, 4, 1);
 *            net_set_optimizer(net, NERVENET_OPTIMIZER_ADAM);
 *            net_initialize_xavier(net);
 *            srand(42);
 *            for (int i = 0; i < 5000; i++) {
 *                int j = i % 4;
 *                net_compute(net, in + j*2, NULL);
 *                net_compute_output_error(net, tgt + j);
 *                net_train(net);
 *            }
 *            net_free(net);
 *            return 0;
 *        }
 *
 *    Compile: gcc -O2 xor.c -o xor -lm
 *
 * ─────────────────────────────────────────────────────────────────────────
 */

#ifndef NERVE_H
#define NERVE_H

#include <stdio.h>

/* ── Version ──────────────────────────────────────────────────────────── */
#define NERVENET_VERSION_MAJOR 2
#define NERVENET_VERSION_MINOR 1
#define NERVENET_VERSION_PATCH 0
#define NERVENET_VERSION       "2.1.0"

/* ── Defaults ─────────────────────────────────────────────────────────── */
#define NERVENET_DEFAULT_MOMENTUM      0.1f
#define NERVENET_DEFAULT_LEARNING_RATE 0.25f
#define NERVENET_DEFAULT_WEIGHT_RANGE  1.0f
#define NERVENET_MAX_LAYERS            256
#define NERVENET_MAX_NEURONS           65536

#define NERVENET_ADAM_BETA1    0.9f
#define NERVENET_ADAM_BETA2    0.999f
#define NERVENET_ADAM_EPSILON  1e-8f

/* ── Enumerations ─────────────────────────────────────────────────────── */
typedef enum
{
    NERVENET_ACTIVATION_SIGMOID    = 0,
    NERVENET_ACTIVATION_TANH       = 1,
    NERVENET_ACTIVATION_RELU       = 2,
    NERVENET_ACTIVATION_LEAKY_RELU = 3,
    NERVENET_ACTIVATION_SOFTMAX    = 4   /* output layer only */
} nervenet_activation_t;

typedef enum
{
    NERVENET_LOSS_MSE           = 0,   /* mean squared error (default)        */
    NERVENET_LOSS_CROSS_ENTROPY = 1    /* categorical cross-entropy           */
} nervenet_loss_t;

typedef enum
{
    NERVENET_OPTIMIZER_SGD  = 0,
    NERVENET_OPTIMIZER_ADAM = 1
} nervenet_optimizer_t;

typedef enum
{
    NERVENET_INIT_UNIFORM = 0,
    NERVENET_INIT_XAVIER  = 1,
    NERVENET_INIT_HE      = 2
} nervenet_init_t;

typedef enum
{
    NERVENET_SUCCESS             =  0,
    NERVENET_ERROR_NULL_POINTER  = -1,
    NERVENET_ERROR_INVALID_PARAM = -2,
    NERVENET_ERROR_MEMORY        = -3,
    NERVENET_ERROR_IO            = -4,
    NERVENET_ERROR_CORRUPT       = -5
} nervenet_error_t;

/* ── Core Structures ──────────────────────────────────────────────────── */
typedef struct neuron_s
{
    float  output;
    float  error;
    float *weight;
    float *delta;
} neuron_t;

typedef struct layer_s
{
    int       no_of_neurons;
    neuron_t *neuron;
} layer_t;

typedef struct network_s
{
    int     no_of_layers;
    int     no_of_patterns;
    float   momentum;
    float   learning_rate;
    float   global_error;
    layer_t *layer;
    layer_t *input_layer;
    layer_t *output_layer;

    int     activation;        /* hidden-layer activation                    */
    int     output_activation; /* output-layer activation (sigmoid/softmax/…)*/
    int     loss;              /* nervenet_loss_t: MSE or cross-entropy       */
    int     optimizer;
    float   l2_lambda;
    float   dropout_rate;   /* fraction of hidden neurons dropped per step */

    float  *adam_m;
    float  *adam_v;
    int     adam_t;
    float   adam_beta1;
    float   adam_beta2;
    float   adam_epsilon;
} network_t;

/* ── Public API ───────────────────────────────────────────────────────── */
#ifdef __cplusplus
extern "C" {
#endif

/* Allocation */
network_t *net_allocate(int no_of_layers, ...);
network_t *net_allocate_l(int no_of_layers, const int *arglist);
void       net_free(network_t *net);

/* Initialisation */
void net_randomize(network_t *net, float range);
void net_initialize_xavier(network_t *net);
void net_initialize_he(network_t *net);
void net_reset_deltas(network_t *net);

/* Configuration */
void  net_set_momentum(network_t *net, float momentum);
void  net_set_learning_rate(network_t *net, float learning_rate);
void  net_use_bias(network_t *net, int use_bias);
void  net_set_activation(network_t *net, nervenet_activation_t act);
void  net_set_output_activation(network_t *net, nervenet_activation_t act);
void  net_set_loss(network_t *net, nervenet_loss_t loss);
void  net_set_classification(network_t *net); /* softmax + cross-entropy combo */
void  net_set_optimizer(network_t *net, nervenet_optimizer_t opt);
void  net_set_l2_lambda(network_t *net, float lambda);
void  net_set_dropout(network_t *net, float rate);

/* Query */
float net_get_momentum(const network_t *net);
float net_get_learning_rate(const network_t *net);
int   net_get_no_of_inputs(const network_t *net);
int   net_get_no_of_outputs(const network_t *net);
int   net_get_no_of_layers(const network_t *net);
int   net_get_no_of_weights(const network_t *net);

/* Weight access */
void  net_set_weight(network_t *net, int layer_index, int nl, int nu, float w);
float net_get_weight(const network_t *net, int layer_index, int nl, int nu);
void  net_set_bias(network_t *net, int layer_index, int neuron_index, float w);
float net_get_bias(const network_t *net, int layer_index, int neuron_index);

/* I/O */
int        net_fprint(FILE *file, const network_t *net);
network_t *net_fscan(FILE *file);
int        net_print(const network_t *net);
int        net_save(const char *filename, const network_t *net);
network_t *net_load(const char *filename);
int        net_fbprint(FILE *file, const network_t *net);
network_t *net_fbscan(FILE *file);
int        net_bsave(const char *filename, const network_t *net);
network_t *net_bload(const char *filename);

/* Inference */
void  net_compute(network_t *net, const float *input, float *output);
float net_compute_output_error(network_t *net, const float *target);
float net_get_output_error(const network_t *net);

/* Online training (one sample) */
void net_train(network_t *net);

/* Batch training */
void net_begin_batch(network_t *net);
void net_train_batch(network_t *net);
void net_end_batch(network_t *net);

/**
 * Mini-batch helper: train on a shuffled epoch, batch_size samples at a time.
 * Pass batch_size = n_pairs for full-batch, 1 for online.
 * Returns the mean MSE over the epoch.
 */
float net_train_epoch(network_t *net,
                      const float *inputs, const float *targets,
                      int n_pairs, int n_inputs, int n_outputs,
                      int batch_size);

/* Metrics */
int   net_classify(network_t *net, const float *input);
float net_compute_accuracy(network_t *net,
                           const float *inputs, const float *targets,
                           int n_pairs, int n_inputs, int n_outputs);

/**
 * Fill confusion matrix (n_classes × n_classes) for a classification dataset.
 * matrix[true_class * n_classes + predicted_class] is incremented.
 * The matrix must be zeroed by the caller before the first call.
 */
void net_confusion_matrix(network_t *net,
                          const float *inputs, const float *targets,
                          int n_pairs, int n_inputs, int n_outputs,
                          int n_classes, int *matrix);

/* Structural modification */
void       net_jolt(network_t *net, float factor, float range);
void       net_add_neurons(network_t *net, int layer, int neuron, int n, float range);
void       net_remove_neurons(network_t *net, int layer, int neuron, int n);
network_t *net_copy(const network_t *net);
void       net_overwrite(network_t *dest, const network_t *src);

/* Utility */
int         net_validate(const network_t *net);
const char *net_get_version(void);

#ifdef __cplusplus
}
#endif

/* ═══════════════════════════════════════════════════════════════════════
 *  EASY API  —  high-level convenience wrapper
 *
 *  Topology string format:  "2->8->1"  or  "784->128->64->10"
 *
 *  Minimal example (XOR in five lines):
 *
 *      float X[] = {1,1, 1,0, 0,1, 0,0};
 *      float y[] = {  0,   1,   1,   0};
 *      nerve_t *net = nerve_new("2->4->1");
 *      nerve_fit(net, X, y, 4, 5000);
 *      nerve_free(net);
 *
 *  All nerve_* names are thin wrappers / macros over the core net_* API.
 *  Both APIs can be mixed freely within the same translation unit.
 * ═══════════════════════════════════════════════════════════════════════ */

/* nerve_t is identical to network_t — just a shorter alias */
typedef network_t nerve_t;

/* Optional configuration passed to nerve_new_ex() */
typedef struct {
    int   optimizer;  /* NERVENET_OPTIMIZER_*    default: ADAM  */
    int   activation; /* NERVENET_ACTIVATION_*   default: TANH  */
    float lr;         /* learning rate; 0 -> 0.01               */
    float l2;         /* L2 lambda;     0 = disabled            */
    float dropout;    /* dropout rate;  0 = disabled            */
    int   seed;       /* RNG seed; -1 = auto from time()        */
} nerve_config_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Create a network from a topology string, e.g. "2->8->1" */
nerve_t       *nerve_new(const char *topology);

/* Same as nerve_new but with explicit configuration */
nerve_t       *nerve_new_ex(const char *topology, const nerve_config_t *cfg);

/* Return the default configuration (Adam, Tanh, lr=0.01) */
nerve_config_t nerve_default_config(void);

/* Train for `epochs` passes over the dataset (silent) */
void  nerve_fit(nerve_t *net, const float *X, const float *y,
                int n_samples, int epochs);

/* Train and print progress every `print_every` epochs (0 = silent) */
void  nerve_fit_verbose(nerve_t *net, const float *X, const float *y,
                        int n_samples, int epochs, int print_every);

/* Classification accuracy over the dataset; targets must be one-hot */
float nerve_score(nerve_t *net, const float *X, const float *y, int n_samples);

#ifdef __cplusplus
}
#endif

/* Thin macro aliases for the remaining core functions */
#define nerve_predict(net, x, out)  net_compute((net), (x), (out))
#define nerve_classify(net, x)      net_classify((net), (x))
#define nerve_save(net, path)       net_save((path), (net))
#define nerve_load(path)            net_load(path)
#define nerve_free(net)             net_free(net)

/* ═══════════════════════════════════════════════════════════════════════
 *  IMPLEMENTATION
 *  (compiled only when NERVE_IMPLEMENTATION is defined)
 * ═══════════════════════════════════════════════════════════════════════ */
#ifdef NERVE_IMPLEMENTATION

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Sigmoid ──────────────────────────────────────────────────────────── */
/* Direct computation — avoids the generated lookup table dependency.
 * For peak throughput use the src/ build which includes interpolation.c.  */
static float nerve__sigma(float x)
{
    return 1.0f / (1.0f + (float)exp(-(double)x));
}

/* ── Activation helpers ───────────────────────────────────────────────── */
static float nerve__activate(float x, int type)
{
    switch (type)
    {
    case NERVENET_ACTIVATION_TANH:
        return (float)tanh((double)x);
    case NERVENET_ACTIVATION_RELU:
        return x > 0.0f ? x : 0.0f;
    case NERVENET_ACTIVATION_LEAKY_RELU:
        return x > 0.0f ? x : 0.01f * x;
    default:
        return nerve__sigma(x);
    }
}

static float nerve__activate_deriv(float y, int type)
{
    switch (type)
    {
    case NERVENET_ACTIVATION_TANH:
        return 1.0f - y * y;
    case NERVENET_ACTIVATION_RELU:
        return y > 0.0f ? 1.0f : 0.0f;
    case NERVENET_ACTIVATION_LEAKY_RELU:
        return y > 0.0f ? 1.0f : 0.01f;
    default:
        return y * (1.0f - y);
    }
}

/* ── Adam offset ──────────────────────────────────────────────────────── */
static int nerve__adam_offset(const network_t *net, int l, int nu, int nl)
{
    int i, off = 0;
    for (i = 1; i < l; i++)
        off += net->layer[i].no_of_neurons *
               (net->layer[i - 1].no_of_neurons + 1);
    return off + nu * (net->layer[l - 1].no_of_neurons + 1) + nl;
}

/* ── Allocation helpers ───────────────────────────────────────────────── */
static void nerve__alloc_layer(layer_t *layer, int n)
{
    layer->no_of_neurons = n;
    layer->neuron = (neuron_t *)calloc((size_t)(n + 1), sizeof(neuron_t));
}

static void nerve__alloc_weights(layer_t *lower, layer_t *upper)
{
    int n;
    for (n = 0; n < upper->no_of_neurons; n++)
    {
        upper->neuron[n].weight =
            (float *)calloc((size_t)(lower->no_of_neurons + 1), sizeof(float));
        upper->neuron[n].delta =
            (float *)calloc((size_t)(lower->no_of_neurons + 1), sizeof(float));
    }
    upper->neuron[n].weight = NULL;
    upper->neuron[n].delta  = NULL;
}

/* ── Public: Allocation ───────────────────────────────────────────────── */
network_t *net_allocate_l(int no_of_layers, const int *arglist)
{
    int l;
    network_t *net;
    assert(no_of_layers >= 2 && arglist != NULL);

    net = (network_t *)malloc(sizeof(network_t));
    if (!net) return NULL;

    net->no_of_layers = no_of_layers;
    net->layer = (layer_t *)calloc((size_t)no_of_layers, sizeof(layer_t));
    if (!net->layer) { free(net); return NULL; }

    for (l = 0; l < no_of_layers; l++)
    {
        assert(arglist[l] > 0);
        nerve__alloc_layer(&net->layer[l], arglist[l]);
    }
    for (l = 1; l < no_of_layers; l++)
        nerve__alloc_weights(&net->layer[l - 1], &net->layer[l]);

    net->input_layer   = &net->layer[0];
    net->output_layer  = &net->layer[no_of_layers - 1];
    net->momentum      = NERVENET_DEFAULT_MOMENTUM;
    net->learning_rate = NERVENET_DEFAULT_LEARNING_RATE;
    net->global_error  = 0.0f;
    net->no_of_patterns = 0;
    net->activation        = NERVENET_ACTIVATION_SIGMOID;
    net->output_activation = NERVENET_ACTIVATION_SIGMOID;
    net->loss              = NERVENET_LOSS_MSE;
    net->optimizer     = NERVENET_OPTIMIZER_SGD;
    net->l2_lambda     = 0.0f;
    net->dropout_rate  = 0.0f;
    net->adam_m        = NULL;
    net->adam_v        = NULL;
    net->adam_t        = 0;
    net->adam_beta1    = NERVENET_ADAM_BETA1;
    net->adam_beta2    = NERVENET_ADAM_BETA2;
    net->adam_epsilon  = NERVENET_ADAM_EPSILON;

    net_randomize(net, NERVENET_DEFAULT_WEIGHT_RANGE);
    net_reset_deltas(net);
    net_use_bias(net, 1);
    return net;
}

network_t *net_allocate(int no_of_layers, ...)
{
    int l, *a;
    va_list args;
    network_t *net;
    assert(no_of_layers >= 2);
    a = (int *)calloc((size_t)no_of_layers, sizeof(int));
    va_start(args, no_of_layers);
    for (l = 0; l < no_of_layers; l++) a[l] = va_arg(args, int);
    va_end(args);
    net = net_allocate_l(no_of_layers, a);
    free(a);
    return net;
}

void net_free(network_t *net)
{
    int l, n;
    assert(net != NULL);
    for (l = 0; l < net->no_of_layers; l++)
    {
        if (l != 0)
            for (n = 0; n < net->layer[l].no_of_neurons; n++)
            {
                free(net->layer[l].neuron[n].weight);
                free(net->layer[l].neuron[n].delta);
            }
        free(net->layer[l].neuron);
    }
    free(net->layer);
    if (net->adam_m) free(net->adam_m);
    if (net->adam_v) free(net->adam_v);
    free(net);
}

/* ── Initialisation ───────────────────────────────────────────────────── */
void net_randomize(network_t *net, float range)
{
    int l, nu, nl;
    assert(net && range >= 0.0f);
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] =
                    2.0f * range * ((float)rand() / (float)RAND_MAX - 0.5f);
}

void net_initialize_xavier(network_t *net)
{
    int l, nu, nl, fi, fo;
    float r;
    assert(net != NULL);
    for (l = 1; l < net->no_of_layers; l++)
    {
        fi = net->layer[l - 1].no_of_neurons;
        fo = net->layer[l].no_of_neurons;
        r  = (float)sqrt(6.0 / (double)(fi + fo));
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] =
                    2.0f * r * ((float)rand() / (float)RAND_MAX - 0.5f);
    }
}

void net_initialize_he(network_t *net)
{
    int l, nu, nl, fi;
    float r;
    assert(net != NULL);
    for (l = 1; l < net->no_of_layers; l++)
    {
        fi = net->layer[l - 1].no_of_neurons;
        r  = (float)sqrt(6.0 / (double)fi);
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] =
                    2.0f * r * ((float)rand() / (float)RAND_MAX - 0.5f);
    }
}

void net_reset_deltas(network_t *net)
{
    int l, nu, nl;
    assert(net != NULL);
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].delta[nl] = 0.0f;
}

/* ── Configuration ────────────────────────────────────────────────────── */
void net_use_bias(network_t *net, int flag)
{
    int l;
    assert(net != NULL);
    for (l = 0; l < net->no_of_layers; l++)
        net->layer[l].neuron[net->layer[l].no_of_neurons].output =
            flag ? 1.0f : 0.0f;
}

void net_set_momentum(network_t *net, float v)
{ assert(net && v >= 0.0f); net->momentum = v; }

void net_set_learning_rate(network_t *net, float v)
{ assert(net && v >= 0.0f); net->learning_rate = v; }

void net_set_activation(network_t *net, nervenet_activation_t a)
{ assert(net != NULL); net->activation = (int)a; }

void net_set_output_activation(network_t *net, nervenet_activation_t a)
{ assert(net != NULL); net->output_activation = (int)a; }

void net_set_loss(network_t *net, nervenet_loss_t loss)
{ assert(net != NULL); net->loss = (int)loss; }

/* Convenience: configure the network for multi-class classification —
 * softmax output + cross-entropy loss, the standard pairing. */
void net_set_classification(network_t *net)
{
    assert(net != NULL);
    net->output_activation = NERVENET_ACTIVATION_SOFTMAX;
    net->loss              = NERVENET_LOSS_CROSS_ENTROPY;
}

void net_set_l2_lambda(network_t *net, float v)
{ assert(net && v >= 0.0f); net->l2_lambda = v; }

void net_set_dropout(network_t *net, float rate)
{ assert(net && rate >= 0.0f && rate < 1.0f); net->dropout_rate = rate; }

void net_set_optimizer(network_t *net, nervenet_optimizer_t opt)
{
    int nw;
    assert(net != NULL);
    if (net->adam_m) { free(net->adam_m); net->adam_m = NULL; }
    if (net->adam_v) { free(net->adam_v); net->adam_v = NULL; }
    net->optimizer = (int)opt;
    if (opt == NERVENET_OPTIMIZER_ADAM)
    {
        nw = net_get_no_of_weights(net);
        net->adam_m      = (float *)calloc((size_t)nw, sizeof(float));
        net->adam_v      = (float *)calloc((size_t)nw, sizeof(float));
        net->adam_t      = 0;
        net->adam_beta1  = NERVENET_ADAM_BETA1;
        net->adam_beta2  = NERVENET_ADAM_BETA2;
        net->adam_epsilon = NERVENET_ADAM_EPSILON;
    }
}

/* ── Query ────────────────────────────────────────────────────────────── */
float net_get_momentum(const network_t *n)      { assert(n); return n->momentum; }
float net_get_learning_rate(const network_t *n) { assert(n); return n->learning_rate; }
int   net_get_no_of_inputs(const network_t *n)  { assert(n); return n->input_layer->no_of_neurons; }
int   net_get_no_of_outputs(const network_t *n) { assert(n); return n->output_layer->no_of_neurons; }
int   net_get_no_of_layers(const network_t *n)  { assert(n); return n->no_of_layers; }

int net_get_no_of_weights(const network_t *net)
{
    int l, r = 0;
    assert(net != NULL);
    for (l = 1; l < net->no_of_layers; l++)
        r += (net->layer[l - 1].no_of_neurons + 1) * net->layer[l].no_of_neurons;
    return r;
}

/* ── Weight access ────────────────────────────────────────────────────── */
void net_set_weight(network_t *net, int l, int nl, int nu, float w)
{
    assert(net && 0 <= l && l < net->no_of_layers);
    net->layer[l].neuron[nu].weight[nl] = w;
}
float net_get_weight(const network_t *net, int l, int nl, int nu)
{
    assert(net && 0 <= l && l < net->no_of_layers - 1);
    return net->layer[l].neuron[nu].weight[nl];
}
float net_get_bias(const network_t *net, int l, int nu)
{ return net_get_weight(net, l - 1, net->layer[l - 1].no_of_neurons, nu); }
void net_set_bias(network_t *net, int l, int nu, float w)
{ net_set_weight(net, l - 1, net->layer[l - 1].no_of_neurons, nu, w); }

/* ── I/O ──────────────────────────────────────────────────────────────── */
int net_fprint(FILE *file, const network_t *net)
{
    int l, nu, nl;
    assert(file && net);
    if (fprintf(file, "%i\n", net->no_of_layers) < 0) return -1;
    for (l = 0; l < net->no_of_layers; l++)
        if (fprintf(file, "%i\n", net->layer[l].no_of_neurons) < 0) return -1;
    if (fprintf(file, "%f\n%f\n%f\n",
                net->momentum, net->learning_rate, net->global_error) < 0)
        return -1;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                if (fprintf(file, "%f\n", net->layer[l].neuron[nu].weight[nl]) < 0)
                    return -1;
    return 0;
}

network_t *net_fscan(FILE *file)
{
    int nl2, l, nu, nl, *a;
    network_t *net;
    assert(file != NULL);
    if (fscanf(file, "%i", &nl2) <= 0) return NULL;
    a = (int *)calloc((size_t)nl2, sizeof(int));
    if (!a) return NULL;
    for (l = 0; l < nl2; l++)
        if (fscanf(file, "%i", &a[l]) <= 0) { free(a); return NULL; }
    net = net_allocate_l(nl2, a);
    free(a);
    if (!net) return NULL;
    if (fscanf(file, "%f%f%f",
               &net->momentum, &net->learning_rate, &net->global_error) < 3)
    { net_free(net); return NULL; }
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                if (fscanf(file, "%f", &net->layer[l].neuron[nu].weight[nl]) <= 0)
                { net_free(net); return NULL; }
    return net;
}

int net_print(const network_t *net) { return net_fprint(stdout, net); }

int net_save(const char *fn, const network_t *net)
{
    FILE *f = fopen(fn, "w");
    int r;
    if (!f) return EOF;
    r = net_fprint(f, net);
    fclose(f);
    return r;
}

network_t *net_load(const char *fn)
{
    FILE *f = fopen(fn, "r");
    network_t *net;
    if (!f) return NULL;
    net = net_fscan(f);
    fclose(f);
    return net;
}

int net_fbprint(FILE *file, const network_t *net)
{
    int l, nu, *info;
    size_t dim;
    float c[3];
    assert(file && net);
    dim  = (size_t)(net->no_of_layers + 1);
    info = (int *)malloc(dim * sizeof(int));
    if (!info) return -1;
    info[0] = net->no_of_layers;
    for (l = 0; l < net->no_of_layers; l++) info[l + 1] = net->layer[l].no_of_neurons;
    if (fwrite(info, sizeof(int), dim, file) < dim) { free(info); return -1; }
    free(info);
    c[0] = net->momentum; c[1] = net->learning_rate; c[2] = net->global_error;
    if (fwrite(c, sizeof(float), 3, file) < 3) return -1;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            fwrite(net->layer[l].neuron[nu].weight, sizeof(float),
                   (size_t)(net->layer[l - 1].no_of_neurons + 1), file);
    return 0;
}

network_t *net_fbscan(FILE *file)
{
    int nl2, l, nu, *a;
    size_t rd;
    network_t *net;
    assert(file != NULL);
    if (fread(&nl2, sizeof(int), 1, file) < 1) return NULL;
    a = (int *)calloc((size_t)nl2, sizeof(int));
    if (!a) return NULL;
    if (fread(a, sizeof(int), (size_t)nl2, file) < (size_t)nl2) { free(a); return NULL; }
    net = net_allocate_l(nl2, a);
    free(a);
    if (!net) return NULL;
    if (fread(&net->momentum,      sizeof(float), 1, file) < 1) { net_free(net); return NULL; }
    if (fread(&net->learning_rate, sizeof(float), 1, file) < 1) { net_free(net); return NULL; }
    if (fread(&net->global_error,  sizeof(float), 1, file) < 1) { net_free(net); return NULL; }
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            rd = fread(net->layer[l].neuron[nu].weight, sizeof(float),
                       (size_t)(net->layer[l - 1].no_of_neurons + 1), file);
            if (rd < (size_t)(net->layer[l - 1].no_of_neurons + 1))
            { net_free(net); return NULL; }
        }
    return net;
}

int net_bsave(const char *fn, const network_t *net)
{
    FILE *f = fopen(fn, "wb");
    int r;
    if (!f) return EOF;
    r = net_fbprint(f, net);
    fclose(f);
    return r;
}

network_t *net_bload(const char *fn)
{
    FILE *f = fopen(fn, "rb");
    network_t *net;
    if (!f) return NULL;
    net = net_fbscan(f);
    fclose(f);
    return net;
}

/* ── Forward pass ─────────────────────────────────────────────────────── */
static void nerve__set_input(network_t *net, const float *in)
{
    int n;
    for (n = 0; n < net->input_layer->no_of_neurons; n++)
        net->input_layer->neuron[n].output = in[n];
}

static void nerve__get_output(const network_t *net, float *out)
{
    int n;
    for (n = 0; n < net->output_layer->no_of_neurons; n++)
        out[n] = net->output_layer->neuron[n].output;
}

static void nerve__propagate(layer_t *lower, layer_t *upper, int act,
                             float dropout)
{
    int nu, nl;
    float v;
    for (nu = 0; nu < upper->no_of_neurons; nu++)
    {
        /* Dropout: randomly silence hidden neurons during training */
        if (dropout > 0.0f &&
            ((float)rand() / (float)RAND_MAX) < dropout)
        {
            upper->neuron[nu].output = 0.0f;
            continue;
        }
        v = 0.0f;
        for (nl = 0; nl <= lower->no_of_neurons; nl++)
            v += upper->neuron[nu].weight[nl] * lower->neuron[nl].output;
        upper->neuron[nu].output = nerve__activate(v, act);
    }
}

/* Output layer forward pass. For softmax the whole layer is normalised
 * together, so it cannot go through the per-neuron nerve__propagate path;
 * every other activation does. */
static void nerve__forward_output(network_t *net)
{
    layer_t *lower = &net->layer[net->no_of_layers - 2];
    layer_t *out   = net->output_layer;
    int nu, nl;
    float v;

    if (net->output_activation != NERVENET_ACTIVATION_SOFTMAX)
    {
        nerve__propagate(lower, out, net->output_activation, 0.0f);
        return;
    }

    /* Softmax: raw logits, then exp/normalise with max-subtraction for
     * numerical stability. */
    {
        float maxv, sum = 0.0f;
        for (nu = 0; nu < out->no_of_neurons; nu++)
        {
            v = 0.0f;
            for (nl = 0; nl <= lower->no_of_neurons; nl++)
                v += out->neuron[nu].weight[nl] * lower->neuron[nl].output;
            out->neuron[nu].output = v;
        }
        maxv = out->neuron[0].output;
        for (nu = 1; nu < out->no_of_neurons; nu++)
            if (out->neuron[nu].output > maxv) maxv = out->neuron[nu].output;
        for (nu = 0; nu < out->no_of_neurons; nu++)
        {
            v = (float)exp((double)(out->neuron[nu].output - maxv));
            out->neuron[nu].output = v;
            sum += v;
        }
        for (nu = 0; nu < out->no_of_neurons; nu++)
            out->neuron[nu].output /= sum;
    }
}

static void nerve__forward(network_t *net, int training)
{
    int l;
    float drop = training ? net->dropout_rate : 0.0f;
    for (l = 1; l < net->no_of_layers - 1; l++)
        nerve__propagate(&net->layer[l - 1], &net->layer[l],
                         net->activation, drop);
    if (net->no_of_layers > 1)
        nerve__forward_output(net);
}

void net_compute(network_t *net, const float *input, float *output)
{
    assert(net && input);
    nerve__set_input(net, input);
    nerve__forward(net, 0);
    if (output) nerve__get_output(net, output);
}

/* ── Error & backward ─────────────────────────────────────────────────── */
float net_compute_output_error(network_t *net, const float *target)
{
    int n;
    float y, e, t;
    assert(net && target);
    net->global_error = 0.0f;

    if (net->loss == NERVENET_LOSS_CROSS_ENTROPY)
    {
        /* Cross-entropy. With a softmax (or sigmoid) output the gradient of
         * the loss w.r.t. each logit collapses to the clean term (t - y),
         * so the rest of the backward pass is unchanged. */
        for (n = 0; n < net->output_layer->no_of_neurons; n++)
        {
            y = net->output_layer->neuron[n].output;
            t = target[n];
            net->output_layer->neuron[n].error = t - y;
            if (t > 0.0f)
                net->global_error -=
                    t * (float)log((double)(y > 1e-12f ? y : 1e-12f));
        }
        return net->global_error;
    }

    /* Mean squared error. The output-layer delta carries the activation
     * derivative; for the default sigmoid output this is exactly y*(1-y)*e. */
    for (n = 0; n < net->output_layer->no_of_neurons; n++)
    {
        y = net->output_layer->neuron[n].output;
        e = target[n] - y;
        net->output_layer->neuron[n].error =
            nerve__activate_deriv(y, net->output_activation) * e;
        net->global_error += e * e;
    }
    net->global_error *= 0.5f;
    return net->global_error;
}

float net_get_output_error(const network_t *net)
{ assert(net); return net->global_error; }

static void nerve__backprop_layer(layer_t *lower, layer_t *upper, int act)
{
    int nl, nu;
    float e, y;
    for (nl = 0; nl <= lower->no_of_neurons; nl++)
    {
        e = 0.0f;
        for (nu = 0; nu < upper->no_of_neurons; nu++)
            e += upper->neuron[nu].weight[nl] * upper->neuron[nu].error;
        y = lower->neuron[nl].output;
        lower->neuron[nl].error = nerve__activate_deriv(y, act) * e;
    }
}

static void nerve__backward(network_t *net)
{
    int l;
    for (l = net->no_of_layers - 1; l > 1; l--)
        nerve__backprop_layer(&net->layer[l - 1], &net->layer[l],
                              net->activation);
}

/* ── Weight update ────────────────────────────────────────────────────── */
static void nerve__adjust(network_t *net)
{
    int l, nu, nl, idx, adam = (net->optimizer == NERVENET_OPTIMIZER_ADAM);
    float lr = net->learning_rate, grad, delta;
    float b1 = 0, b2 = 0, eps = 0, bc1 = 0, bc2 = 0, mh, vh;

    if (adam)
    {
        net->adam_t++;
        b1  = net->adam_beta1;  b2  = net->adam_beta2;
        eps = net->adam_epsilon;
        bc1 = 1.0f - (float)pow((double)b1, (double)net->adam_t);
        bc2 = 1.0f - (float)pow((double)b2, (double)net->adam_t);
    }

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            float err = net->layer[l].neuron[nu].error;
            /* Skip dropped neurons (output is 0 and error is 0) */
            if (net->dropout_rate > 0.0f && err == 0.0f &&
                net->layer[l].neuron[nu].output == 0.0f) continue;

            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                grad = err * net->layer[l - 1].neuron[nl].output;
                if (net->l2_lambda > 0.0f)
                    grad -= net->l2_lambda *
                            net->layer[l].neuron[nu].weight[nl];

                if (adam)
                {
                    idx = nerve__adam_offset(net, l, nu, nl);
                    net->adam_m[idx] = b1 * net->adam_m[idx] + (1.0f - b1) * grad;
                    net->adam_v[idx] = b2 * net->adam_v[idx] + (1.0f - b2) * grad * grad;
                    mh = net->adam_m[idx] / bc1;
                    vh = net->adam_v[idx] / bc2;
                    delta = lr * mh / ((float)sqrt((double)vh) + eps);
                }
                else
                {
                    delta = lr * grad +
                            net->momentum * net->layer[l].neuron[nu].delta[nl];
                }
                net->layer[l].neuron[nu].weight[nl] += delta;
                net->layer[l].neuron[nu].delta[nl]   = delta;
            }
        }
}

/* ── Online training ──────────────────────────────────────────────────── */
static void nerve__forward_train(network_t *net)
{
    nerve__forward(net, 1);
}

void net_train(network_t *net)
{
    assert(net != NULL);
    nerve__backward(net);
    nerve__adjust(net);
}

/* ── Batch training ───────────────────────────────────────────────────── */
static void nerve__accum_deltas(network_t *net)
{
    int l, nu, nl;
    float err;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            err = net->layer[l].neuron[nu].error;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].delta[nl] +=
                    net->learning_rate * err *
                    net->layer[l - 1].neuron[nl].output;
        }
}

static void nerve__apply_deltas(network_t *net)
{
    int l, nu, nl;
    float d = (net->no_of_patterns > 0) ? (float)net->no_of_patterns : 1.0f;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] +=
                    net->layer[l].neuron[nu].delta[nl] / d;
}

void net_begin_batch(network_t *net)
{ assert(net); net->no_of_patterns = 0; net_reset_deltas(net); }

void net_train_batch(network_t *net)
{ assert(net); net->no_of_patterns++; nerve__backward(net); nerve__accum_deltas(net); }

void net_end_batch(network_t *net)
{ assert(net); nerve__apply_deltas(net); }

/* ── Mini-batch / shuffled-online epoch helper ────────────────────────── */
/*
 * Shuffles the dataset, then iterates in mini-batches of `batch_size`.
 *
 * When optimizer == ADAM  (recommended): each sample triggers one Adam step
 *   — fully online with shuffle, which is the standard stochastic approach.
 *   The batch_size parameter is ignored in this mode.
 *
 * When optimizer == SGD: accumulates gradients over each mini-batch before
 *   applying the update (classic mini-batch SGD).
 *
 * Returns the mean MSE over all samples in the epoch.
 */
float net_train_epoch(network_t *net,
                      const float *inputs, const float *targets,
                      int n_pairs, int n_inputs, int n_outputs,
                      int batch_size)
{
    int *order, i, j, tmp, b_start, b_end, k;
    float total_err = 0.0f;
    int use_adam = (net->optimizer == NERVENET_OPTIMIZER_ADAM);
    assert(net && inputs && targets && n_pairs > 0 && batch_size > 0);

    order = (int *)malloc((size_t)n_pairs * sizeof(int));
    for (i = 0; i < n_pairs; i++) order[i] = i;
    for (i = n_pairs - 1; i > 0; i--)
    {
        j = rand() % (i + 1);
        tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    if (use_adam)
    {
        /* One Adam step per sample (stochastic Adam) */
        for (k = 0; k < n_pairs; k++)
        {
            i = order[k];
            nerve__set_input(net, inputs + i * n_inputs);
            nerve__forward_train(net);
            total_err += net_compute_output_error(net, targets + i * n_outputs);
            nerve__backward(net);
            nerve__adjust(net);
        }
    }
    else
    {
        /* Mini-batch SGD gradient accumulation */
        for (b_start = 0; b_start < n_pairs; b_start += batch_size)
        {
            b_end = b_start + batch_size;
            if (b_end > n_pairs) b_end = n_pairs;
            net_begin_batch(net);
            for (k = b_start; k < b_end; k++)
            {
                i = order[k];
                nerve__set_input(net, inputs + i * n_inputs);
                nerve__forward_train(net);
                total_err += net_compute_output_error(net, targets + i * n_outputs);
                net_train_batch(net);
            }
            net_end_batch(net);
        }
    }

    free(order);
    return total_err / (float)n_pairs;
}

/* ── Metrics ──────────────────────────────────────────────────────────── */
int net_classify(network_t *net, const float *input)
{
    int n, best = 0;
    float bv;
    assert(net && input);
    net_compute(net, input, NULL);
    bv = net->output_layer->neuron[0].output;
    for (n = 1; n < net->output_layer->no_of_neurons; n++)
        if (net->output_layer->neuron[n].output > bv)
        { bv = net->output_layer->neuron[n].output; best = n; }
    return best;
}

float net_compute_accuracy(network_t *net,
                           const float *inputs, const float *targets,
                           int n_pairs, int n_inputs, int n_outputs)
{
    int i, n, ok = 0, pred, tc;
    float bv;
    assert(net && inputs && targets && n_pairs > 0);
    for (i = 0; i < n_pairs; i++)
    {
        pred = net_classify(net, inputs + i * n_inputs);
        tc = 0; bv = targets[i * n_outputs];
        for (n = 1; n < n_outputs; n++)
            if (targets[i * n_outputs + n] > bv)
            { bv = targets[i * n_outputs + n]; tc = n; }
        if (pred == tc) ok++;
    }
    return (float)ok / (float)n_pairs;
}

void net_confusion_matrix(network_t *net,
                          const float *inputs, const float *targets,
                          int n_pairs, int n_inputs, int n_outputs,
                          int n_classes, int *matrix)
{
    int i, n, pred, tc;
    float bv;
    assert(net && inputs && targets && matrix && n_pairs > 0);
    for (i = 0; i < n_pairs; i++)
    {
        pred = net_classify(net, inputs + i * n_inputs);
        tc = 0; bv = targets[i * n_outputs];
        for (n = 1; n < n_outputs; n++)
            if (targets[i * n_outputs + n] > bv)
            { bv = targets[i * n_outputs + n]; tc = n; }
        if (pred < n_classes && tc < n_classes)
            matrix[tc * n_classes + pred]++;
    }
}

/* ── Structural modification ──────────────────────────────────────────── */
void net_jolt(network_t *net, float factor, float range)
{
    int l, nu, nl;
    assert(net && factor >= 0.0f && range >= 0.0f);
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                float *w = &net->layer[l].neuron[nu].weight[nl];
                if (fabs(*w) < (double)range)
                    *w = 2.0f * range * ((float)rand()/(float)RAND_MAX - 0.5f);
                else
                    *w *= 1.0f + 2.0f*factor*((float)rand()/(float)RAND_MAX - 0.5f);
            }
}

network_t *net_copy(const network_t *net)
{
    int l, nu, nl, nw, *a;
    network_t *n2;
    assert(net != NULL);
    a = (int *)calloc((size_t)net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++) a[l] = net->layer[l].no_of_neurons;
    n2 = net_allocate_l(net->no_of_layers, a);
    free(a);
    if (!n2) return NULL;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                n2->layer[l].neuron[nu].weight[nl] = net->layer[l].neuron[nu].weight[nl];
                n2->layer[l].neuron[nu].delta[nl]  = net->layer[l].neuron[nu].delta[nl];
            }
    n2->momentum       = net->momentum;
    n2->learning_rate  = net->learning_rate;
    n2->global_error   = net->global_error;
    n2->no_of_patterns = net->no_of_patterns;
    n2->activation        = net->activation;
    n2->output_activation = net->output_activation;
    n2->loss              = net->loss;
    n2->optimizer      = net->optimizer;
    n2->l2_lambda      = net->l2_lambda;
    n2->dropout_rate   = net->dropout_rate;
    n2->adam_t         = net->adam_t;
    n2->adam_beta1     = net->adam_beta1;
    n2->adam_beta2     = net->adam_beta2;
    n2->adam_epsilon   = net->adam_epsilon;
    if (net->adam_m && net->adam_v)
    {
        nw = net_get_no_of_weights(net);
        n2->adam_m = (float *)malloc((size_t)nw * sizeof(float));
        n2->adam_v = (float *)malloc((size_t)nw * sizeof(float));
        if (n2->adam_m) memcpy(n2->adam_m, net->adam_m, (size_t)nw * sizeof(float));
        if (n2->adam_v) memcpy(n2->adam_v, net->adam_v, (size_t)nw * sizeof(float));
    }
    return n2;
}

void net_overwrite(network_t *dest, const network_t *src)
{
    network_t *n2, *tmp;
    assert(dest && src);
    n2  = net_copy(src);
    tmp = (network_t *)malloc(sizeof(network_t));
    memcpy(tmp, n2,   sizeof(network_t));
    memcpy(n2,  dest, sizeof(network_t));
    memcpy(dest, tmp, sizeof(network_t));
    free(tmp);
    net_free(n2);
}

void net_add_neurons(network_t *net, int layer, int neuron, int number, float range)
{
    int l, nu, nl, nnu, nnl, *a;
    network_t *n2, *tmp;
    assert(net && 0 <= layer && layer < net->no_of_layers && number >= 0);
    if (neuron == -1) neuron = net->layer[layer].no_of_neurons;
    a = (int *)calloc((size_t)net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++) a[l] = net->layer[l].no_of_neurons;
    a[layer] += number;
    n2 = net_allocate_l(net->no_of_layers, a);
    free(a);
    net_randomize(n2, range);
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            nnu = (l == layer && nu >= neuron) ? nu + number : nu;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                nnl = (l == layer + 1 && nl >= neuron) ? nl + number : nl;
                n2->layer[l].neuron[nnu].weight[nnl] = net->layer[l].neuron[nu].weight[nl];
                n2->layer[l].neuron[nnu].delta[nnl]  = net->layer[l].neuron[nu].delta[nl];
            }
        }
    n2->momentum = net->momentum; n2->learning_rate = net->learning_rate;
    n2->activation = net->activation; n2->optimizer = net->optimizer;
    n2->output_activation = net->output_activation; n2->loss = net->loss;
    n2->l2_lambda = net->l2_lambda;
    tmp = (network_t *)malloc(sizeof(network_t));
    memcpy(tmp, n2, sizeof(network_t)); memcpy(n2, net, sizeof(network_t));
    memcpy(net, tmp, sizeof(network_t)); free(tmp); net_free(n2);
}

void net_remove_neurons(network_t *net, int layer, int neuron, int number)
{
    int l, nu, nl, onu, onl, *a;
    network_t *n2, *tmp;
    assert(net && 0 <= layer && layer < net->no_of_layers && number >= 0);
    a = (int *)calloc((size_t)net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++) a[l] = net->layer[l].no_of_neurons;
    a[layer] -= number;
    n2 = net_allocate_l(net->no_of_layers, a); free(a);
    for (l = 1; l < n2->no_of_layers; l++)
        for (nu = 0; nu < n2->layer[l].no_of_neurons; nu++)
        {
            onu = (l == layer && nu >= neuron) ? nu + number : nu;
            for (nl = 0; nl <= n2->layer[l - 1].no_of_neurons; nl++)
            {
                onl = (l == layer + 1 && nl >= neuron) ? nl + number : nl;
                n2->layer[l].neuron[nu].weight[nl] = net->layer[l].neuron[onu].weight[onl];
                n2->layer[l].neuron[nu].delta[nl]  = net->layer[l].neuron[onu].delta[onl];
            }
        }
    n2->momentum = net->momentum; n2->learning_rate = net->learning_rate;
    n2->activation = net->activation; n2->optimizer = net->optimizer;
    n2->output_activation = net->output_activation; n2->loss = net->loss;
    n2->l2_lambda = net->l2_lambda;
    tmp = (network_t *)malloc(sizeof(network_t));
    memcpy(tmp, n2, sizeof(network_t)); memcpy(n2, net, sizeof(network_t));
    memcpy(net, tmp, sizeof(network_t)); free(tmp); net_free(n2);
}

/* ── Utility ──────────────────────────────────────────────────────────── */
int net_validate(const network_t *net)
{
    int l, n;
    if (!net || net->no_of_layers < 2 || !net->layer) return 0;
    if (net->learning_rate <= 0.0f || net->momentum < 0.0f) return 0;
    for (l = 0; l < net->no_of_layers; l++)
    {
        if (net->layer[l].no_of_neurons <= 0 || !net->layer[l].neuron) return 0;
        if (l > 0)
            for (n = 0; n < net->layer[l].no_of_neurons; n++)
                if (!net->layer[l].neuron[n].weight ||
                    !net->layer[l].neuron[n].delta) return 0;
    }
    if (net->optimizer == NERVENET_OPTIMIZER_ADAM &&
        (!net->adam_m || !net->adam_v)) return 0;
    return 1;
}

const char *net_get_version(void) { return NERVENET_VERSION; }

/* ── Easy API implementation ──────────────────────────────────────────── */

/* Parse "2->8->1" or "784 128 10" into an int array; return layer count */
static int nerve__parse_topo(const char *topo, int *sizes, int max)
{
    int n = 0, v;
    const char *p = topo;
    char *end;
    while (*p) {
        while (*p == ' ' || *p == '-' || *p == '>') p++;
        if (!*p) break;
        v = (int)strtol(p, &end, 10);
        if (end == p) break;
        if (n < max) sizes[n] = v;
        n++;
        p = end;
    }
    return n;
}

nerve_config_t nerve_default_config(void)
{
    nerve_config_t c;
    c.optimizer  = NERVENET_OPTIMIZER_ADAM;
    c.activation = NERVENET_ACTIVATION_TANH;
    c.lr         = 0.01f;
    c.l2         = 0.0f;
    c.dropout    = 0.0f;
    c.seed       = -1;
    return c;
}

nerve_t *nerve_new_ex(const char *topology, const nerve_config_t *cfg)
{
    int sizes[NERVENET_MAX_LAYERS];
    int n;
    nerve_t *net;
    nerve_config_t def;
    if (!cfg) { def = nerve_default_config(); cfg = &def; }
    n = nerve__parse_topo(topology, sizes, NERVENET_MAX_LAYERS);
    if (n < 2) return NULL;
    if (cfg->seed >= 0) srand((unsigned int)cfg->seed);
    else                srand((unsigned int)time(NULL));
    net = net_allocate_l(n, sizes);
    if (!net) return NULL;
    net_set_optimizer(net,  (nervenet_optimizer_t)cfg->optimizer);
    net_set_activation(net, (nervenet_activation_t)cfg->activation);
    net_set_learning_rate(net, cfg->lr > 0.0f ? cfg->lr : 0.01f);
    net_set_l2_lambda(net, cfg->l2);
    if (cfg->dropout > 0.0f) net_set_dropout(net, cfg->dropout);
    if (cfg->activation == NERVENET_ACTIVATION_RELU ||
        cfg->activation == NERVENET_ACTIVATION_LEAKY_RELU)
        net_initialize_he(net);
    else
        net_initialize_xavier(net);
    return net;
}

nerve_t *nerve_new(const char *topology)
{
    return nerve_new_ex(topology, NULL);
}

void nerve_fit_verbose(nerve_t *net, const float *X, const float *y,
                       int n_samples, int epochs, int print_every)
{
    int n_in, n_out, e;
    float mse;
    assert(net && X && y && n_samples > 0 && epochs > 0);
    n_in  = net_get_no_of_inputs(net);
    n_out = net_get_no_of_outputs(net);
    for (e = 1; e <= epochs; e++) {
        mse = net_train_epoch(net, X, y, n_samples, n_in, n_out, 1);
        if (print_every > 0 && e % print_every == 0)
            printf("  epoch %5d / %d   %s: %.6f\n", e, epochs,
                   net->loss == NERVENET_LOSS_CROSS_ENTROPY ? "loss" : "MSE",
                   mse);
    }
}

void nerve_fit(nerve_t *net, const float *X, const float *y,
               int n_samples, int epochs)
{
    nerve_fit_verbose(net, X, y, n_samples, epochs, 0);
}

float nerve_score(nerve_t *net, const float *X, const float *y, int n_samples)
{
    assert(net && X && y && n_samples > 0);
    return net_compute_accuracy(net, X, y, n_samples,
                                net_get_no_of_inputs(net),
                                net_get_no_of_outputs(net));
}

#endif /* NERVE_IMPLEMENTATION */
#endif /* NERVE_H */
