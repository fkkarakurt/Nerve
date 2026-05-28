/*
 * Nerve — Neural Network Library
 * Copyright (c) 2022 Fatih Küçükkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Nerve — Neural Network Library  (v2.0.0)
 *
 * ANSI C89/C90 compliant; zero external dependencies.
 *
 * References
 *   [1] Rumelhart, Hinton & Williams (1986). Learning representations by
 *       back-propagating errors. Nature 323, 533–536.
 *   [2] Glorot & Bengio (2010). Understanding the difficulty of training deep
 *       feedforward neural networks. AISTATS.
 *   [3] He et al. (2015). Delving deep into rectifiers. ICCV.
 *   [4] Kingma & Ba (2014). Adam: A method for stochastic optimization. ICLR.
 */

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nerveNet.h"

/* ── Internal Defaults ───────────────────────────────────────────────── */
#define DEFAULT_MOMENTUM      0.1
#define DEFAULT_LEARNING_RATE 0.25
#define DEFAULT_WEIGHT_RANGE  1.0

/* ── Sigmoid Look-up Table ───────────────────────────────────────────── */
#include "interpolation.c"

static float sigma(float x)
{
    int index = (int)((x - min_entry) / interval);
    if (index <= 0)            return interpolation[0];
    if (index >= num_entries)  return interpolation[num_entries - 1];
    return interpolation[index];
}

/* ── Activation Functions ────────────────────────────────────────────── */

/* Forward activation — applied during the forward pass */
static float activate(float x, int type)
{
    switch (type)
    {
    case NERVENET_ACTIVATION_TANH:
        return (float)tanh((double)x);
    case NERVENET_ACTIVATION_RELU:
        return x > 0.0f ? x : 0.0f;
    case NERVENET_ACTIVATION_LEAKY_RELU:
        return x > 0.0f ? x : 0.01f * x;
    default: /* SIGMOID */
        return sigma(x);
    }
}

/* Derivative of the activation at a neuron whose OUTPUT is y.
 * Works because:
 *   σ'(x)  = σ(x)(1−σ(x))   = y(1−y)
 *   tanh'  = 1−tanh²(x)      = 1−y²
 *   ReLU'  = 1{x>0}          ≡ 1{y>0}
 *   LReLU' = 1{y>0} + 0.01·1{y≤0}
 */
static float activate_deriv(float y, int type)
{
    switch (type)
    {
    case NERVENET_ACTIVATION_TANH:
        return 1.0f - y * y;
    case NERVENET_ACTIVATION_RELU:
        return y > 0.0f ? 1.0f : 0.0f;
    case NERVENET_ACTIVATION_LEAKY_RELU:
        return y > 0.0f ? 1.0f : 0.01f;
    default: /* SIGMOID */
        return y * (1.0f - y);
    }
}

/* ── Internal Helpers ────────────────────────────────────────────────── */

/* Flat index into the Adam moment arrays for weight [l][nu][nl]. */
static int adam_offset(const network_t *net, int l, int nu, int nl)
{
    int i, offset = 0;
    for (i = 1; i < l; i++)
        offset += net->layer[i].no_of_neurons *
                  (net->layer[i - 1].no_of_neurons + 1);
    offset += nu * (net->layer[l - 1].no_of_neurons + 1) + nl;
    return offset;
}

static void allocate_layer(layer_t *layer, int no_of_neurons)
{
    assert(layer != NULL);
    assert(no_of_neurons > 0);
    layer->no_of_neurons = no_of_neurons;
    /* +1 for the bias unit */
    layer->neuron = (neuron_t *)calloc(no_of_neurons + 1, sizeof(neuron_t));
}

static void allocate_weights(layer_t *lower, layer_t *upper)
{
    int n;
    assert(lower != NULL);
    assert(upper != NULL);
    for (n = 0; n < upper->no_of_neurons; n++)
    {
        /* +1 for the bias weight */
        upper->neuron[n].weight =
            (float *)calloc(lower->no_of_neurons + 1, sizeof(float));
        upper->neuron[n].delta =
            (float *)calloc(lower->no_of_neurons + 1, sizeof(float));
    }
    upper->neuron[n].weight = NULL;
    upper->neuron[n].delta  = NULL;
}

/* ── Allocation ──────────────────────────────────────────────────────── */

network_t *net_allocate_l(int no_of_layers, const int *arglist)
{
    int l;
    network_t *net;

    assert(no_of_layers >= 2);
    assert(arglist != NULL);

    net = (network_t *)malloc(sizeof(network_t));
    if (net == NULL) return NULL;

    net->no_of_layers = no_of_layers;
    net->layer = (layer_t *)calloc(no_of_layers, sizeof(layer_t));
    if (net->layer == NULL) { free(net); return NULL; }

    for (l = 0; l < no_of_layers; l++)
    {
        assert(arglist[l] > 0);
        allocate_layer(&net->layer[l], arglist[l]);
    }
    for (l = 1; l < no_of_layers; l++)
        allocate_weights(&net->layer[l - 1], &net->layer[l]);

    net->input_layer  = &net->layer[0];
    net->output_layer = &net->layer[no_of_layers - 1];

    net->momentum      = (float)DEFAULT_MOMENTUM;
    net->learning_rate = (float)DEFAULT_LEARNING_RATE;
    net->global_error  = 0.0f;
    net->no_of_patterns = 0;

    /* v2 defaults */
    net->activation  = NERVENET_ACTIVATION_SIGMOID;
    net->optimizer   = NERVENET_OPTIMIZER_SGD;
    net->l2_lambda   = 0.0f;
    net->adam_m      = NULL;
    net->adam_v      = NULL;
    net->adam_t      = 0;
    net->adam_beta1  = NERVENET_ADAM_BETA1;
    net->adam_beta2  = NERVENET_ADAM_BETA2;
    net->adam_epsilon = NERVENET_ADAM_EPSILON;

    net_randomize(net, (float)DEFAULT_WEIGHT_RANGE);
    net_reset_deltas(net);
    net_use_bias(net, 1);

    return net;
}

network_t *net_allocate(int no_of_layers, ...)
{
    int l, *arglist;
    va_list args;
    network_t *net;

    assert(no_of_layers >= 2);

    arglist = (int *)calloc(no_of_layers, sizeof(int));
    va_start(args, no_of_layers);
    for (l = 0; l < no_of_layers; l++)
        arglist[l] = va_arg(args, int);
    va_end(args);

    net = net_allocate_l(no_of_layers, arglist);
    free(arglist);
    return net;
}

void net_free(network_t *net)
{
    int l, n;
    assert(net != NULL);

    for (l = 0; l < net->no_of_layers; l++)
    {
        if (l != 0)
        {
            for (n = 0; n < net->layer[l].no_of_neurons; n++)
            {
                free(net->layer[l].neuron[n].weight);
                free(net->layer[l].neuron[n].delta);
            }
        }
        free(net->layer[l].neuron);
    }
    free(net->layer);

    if (net->adam_m != NULL) free(net->adam_m);
    if (net->adam_v != NULL) free(net->adam_v);

    free(net);
}

/* ── Weight Initialisation ───────────────────────────────────────────── */

void net_randomize(network_t *net, float range)
{
    int l, nu, nl;
    assert(net != NULL);
    assert(range >= 0.0f);

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] =
                    2.0f * range * ((float)rand() / (float)RAND_MAX - 0.5f);
}

void net_initialize_xavier(network_t *net)
{
    int l, nu, nl, fan_in, fan_out;
    float range;
    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
    {
        fan_in  = net->layer[l - 1].no_of_neurons;
        fan_out = net->layer[l].no_of_neurons;
        /* Glorot uniform: U[-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))] */
        range = (float)sqrt(6.0 / (double)(fan_in + fan_out));
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] =
                    2.0f * range *
                    ((float)rand() / (float)RAND_MAX - 0.5f);
    }
}

void net_initialize_he(network_t *net)
{
    int l, nu, nl, fan_in;
    float range;
    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
    {
        fan_in = net->layer[l - 1].no_of_neurons;
        /* He uniform: U[-√(6/fan_in), +√(6/fan_in)] */
        range = (float)sqrt(6.0 / (double)fan_in);
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] =
                    2.0f * range *
                    ((float)rand() / (float)RAND_MAX - 0.5f);
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

/* ── Configuration ───────────────────────────────────────────────────── */

void net_use_bias(network_t *net, int flag)
{
    int l;
    assert(net != NULL);
    for (l = 0; l < net->no_of_layers; l++)
        net->layer[l].neuron[net->layer[l].no_of_neurons].output =
            flag ? 1.0f : 0.0f;
}

void net_set_momentum(network_t *net, float momentum)
{
    assert(net != NULL);
    assert(momentum >= 0.0f);
    net->momentum = momentum;
}

void net_set_learning_rate(network_t *net, float learning_rate)
{
    assert(net != NULL);
    assert(learning_rate >= 0.0f);
    net->learning_rate = learning_rate;
}

void net_set_activation(network_t *net, nervenet_activation_t act)
{
    assert(net != NULL);
    net->activation = (int)act;
}

void net_set_optimizer(network_t *net, nervenet_optimizer_t opt)
{
    int n_weights;
    assert(net != NULL);

    /* Release any existing Adam state */
    if (net->adam_m != NULL) { free(net->adam_m); net->adam_m = NULL; }
    if (net->adam_v != NULL) { free(net->adam_v); net->adam_v = NULL; }

    net->optimizer = (int)opt;

    if (opt == NERVENET_OPTIMIZER_ADAM)
    {
        n_weights    = net_get_no_of_weights(net);
        net->adam_m  = (float *)calloc(n_weights, sizeof(float));
        net->adam_v  = (float *)calloc(n_weights, sizeof(float));
        net->adam_t  = 0;
        net->adam_beta1   = NERVENET_ADAM_BETA1;
        net->adam_beta2   = NERVENET_ADAM_BETA2;
        net->adam_epsilon = NERVENET_ADAM_EPSILON;
    }
}

void net_set_l2_lambda(network_t *net, float lambda)
{
    assert(net != NULL);
    assert(lambda >= 0.0f);
    net->l2_lambda = lambda;
}

/* ── Query ───────────────────────────────────────────────────────────── */

float net_get_momentum(const network_t *net)
{
    assert(net != NULL);
    return net->momentum;
}

float net_get_learning_rate(const network_t *net)
{
    assert(net != NULL);
    return net->learning_rate;
}

int net_get_no_of_inputs(const network_t *net)
{
    assert(net != NULL);
    return net->input_layer->no_of_neurons;
}

int net_get_no_of_outputs(const network_t *net)
{
    assert(net != NULL);
    return net->output_layer->no_of_neurons;
}

int net_get_no_of_layers(const network_t *net)
{
    assert(net != NULL);
    return net->no_of_layers;
}

int net_get_no_of_weights(const network_t *net)
{
    int l, result = 0;
    assert(net != NULL);
    for (l = 1; l < net->no_of_layers; l++)
        result += (net->layer[l - 1].no_of_neurons + 1) *
                   net->layer[l].no_of_neurons;
    return result;
}

/* ── Weight Access ───────────────────────────────────────────────────── */

void net_set_weight(network_t *net, int l, int nl, int nu, float weight)
{
    assert(net != NULL);
    assert(0 <= l  && l  < net->no_of_layers);
    assert(0 <= nl && nl <= net->layer[l].no_of_neurons);
    assert(0 <= nu && nu <  net->layer[l + 1].no_of_neurons);
    net->layer[l].neuron[nu].weight[nl] = weight;
}

float net_get_weight(const network_t *net, int l, int nl, int nu)
{
    assert(net != NULL);
    assert(0 <= l  && l  < net->no_of_layers - 1);
    assert(0 <= nl && nl <= net->layer[l].no_of_neurons);
    assert(0 <= nu && nu <  net->layer[l + 1].no_of_neurons);
    return net->layer[l].neuron[nu].weight[nl];
}

float net_get_bias(const network_t *net, int l, int nu)
{
    assert(net != NULL);
    assert(0 < l && l < net->no_of_layers);
    assert(0 <= nu && nu < net->layer[l].no_of_neurons);
    return net_get_weight(net, l - 1, net->layer[l - 1].no_of_neurons, nu);
}

void net_set_bias(network_t *net, int l, int nu, float weight)
{
    assert(net != NULL);
    assert(0 < l && l < net->no_of_layers);
    assert(0 <= nu && nu < net->layer[l].no_of_neurons);
    net_set_weight(net, l - 1, net->layer[l - 1].no_of_neurons, nu, weight);
}

/* ── Text I/O ────────────────────────────────────────────────────────── */

int net_fprint(FILE *file, const network_t *net)
{
    int l, nu, nl, result;
    assert(file != NULL);
    assert(net != NULL);

    result = fprintf(file, "%i\n", net->no_of_layers);
    if (result < 0) return result;

    for (l = 0; l < net->no_of_layers; l++)
    {
        result = fprintf(file, "%i\n", net->layer[l].no_of_neurons);
        if (result < 0) return result;
    }

    if (fprintf(file, "%f\n", net->momentum)      < 0) return -1;
    if (fprintf(file, "%f\n", net->learning_rate) < 0) return -1;
    if (fprintf(file, "%f\n", net->global_error)  < 0) return -1;

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                result = fprintf(file, "%f\n",
                                 net->layer[l].neuron[nu].weight[nl]);
                if (result < 0) return result;
            }

    return 0;
}

network_t *net_fscan(FILE *file)
{
    int no_of_layers, l, nu, nl, *arglist, result;
    network_t *net;

    assert(file != NULL);

    result = fscanf(file, "%i", &no_of_layers);
    if (result <= 0) return NULL;

    arglist = (int *)calloc(no_of_layers, sizeof(int));
    if (arglist == NULL) return NULL;

    for (l = 0; l < no_of_layers; l++)
    {
        result = fscanf(file, "%i", &arglist[l]);
        if (result <= 0) { free(arglist); return NULL; }
    }

    net = net_allocate_l(no_of_layers, arglist);
    free(arglist);
    if (net == NULL) return NULL;

    if (fscanf(file, "%f", &net->momentum)      <= 0) { net_free(net); return NULL; }
    if (fscanf(file, "%f", &net->learning_rate) <= 0) { net_free(net); return NULL; }
    if (fscanf(file, "%f", &net->global_error)  <= 0) { net_free(net); return NULL; }

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                result = fscanf(file, "%f",
                                &net->layer[l].neuron[nu].weight[nl]);
                if (result <= 0) { net_free(net); return NULL; }
            }

    return net;
}

int net_print(const network_t *net)
{
    assert(net != NULL);
    return net_fprint(stdout, net);
}

int net_save(const char *filename, const network_t *net)
{
    int result;
    FILE *file;
    assert(filename != NULL);
    assert(net != NULL);

    file = fopen(filename, "w");
    if (file == NULL) return EOF;

    result = net_fprint(file, net);
    fclose(file);
    return result;
}

network_t *net_load(const char *filename)
{
    FILE *file;
    network_t *net;
    assert(filename != NULL);

    file = fopen(filename, "r");
    if (file == NULL) return NULL;

    net = net_fscan(file);
    fclose(file);
    return net;
}

/* ── Binary I/O ──────────────────────────────────────────────────────── */

int net_fbprint(FILE *file, const network_t *net)
{
    int l, nu;
    int *info;
    float constants[3];
    size_t info_dim;

    assert(file != NULL);
    assert(net != NULL);

    info_dim = (size_t)(net->no_of_layers + 1);
    info = (int *)malloc(info_dim * sizeof(int));
    if (info == NULL) return -1;

    info[0] = net->no_of_layers;
    for (l = 0; l < net->no_of_layers; l++)
        info[l + 1] = net->layer[l].no_of_neurons;

    if (fwrite(info, sizeof(int), info_dim, file) < info_dim)
    {
        free(info);
        return -1;
    }
    free(info);

    constants[0] = net->momentum;
    constants[1] = net->learning_rate;
    constants[2] = net->global_error;
    if (fwrite(constants, sizeof(float), 3, file) < 3) return -1;

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            fwrite(net->layer[l].neuron[nu].weight, sizeof(float),
                   (size_t)(net->layer[l - 1].no_of_neurons + 1), file);

    return 0;
}

network_t *net_fbscan(FILE *file)
{
    int no_of_layers, l, nu, *arglist;
    size_t read_items;
    network_t *net;

    assert(file != NULL);

    if (fread(&no_of_layers, sizeof(int), 1, file) < 1) return NULL;

    arglist = (int *)calloc((size_t)no_of_layers, sizeof(int));
    if (arglist == NULL) return NULL;

    if (fread(arglist, sizeof(int), (size_t)no_of_layers, file) <
        (size_t)no_of_layers)
    {
        free(arglist);
        return NULL;
    }

    net = net_allocate_l(no_of_layers, arglist);
    free(arglist);
    if (net == NULL) return NULL;

    if (fread(&net->momentum,      sizeof(float), 1, file) < 1) { net_free(net); return NULL; }
    if (fread(&net->learning_rate, sizeof(float), 1, file) < 1) { net_free(net); return NULL; }
    if (fread(&net->global_error,  sizeof(float), 1, file) < 1) { net_free(net); return NULL; }

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            read_items = fread(net->layer[l].neuron[nu].weight,
                               sizeof(float),
                               (size_t)(net->layer[l - 1].no_of_neurons + 1),
                               file);
            if (read_items < (size_t)(net->layer[l - 1].no_of_neurons + 1))
            {
                net_free(net);
                return NULL;
            }
        }

    return net;
}

int net_bsave(const char *filename, const network_t *net)
{
    FILE *file;
    int result;
    assert(filename != NULL);
    assert(net != NULL);

    file = fopen(filename, "wb");
    if (file == NULL) return EOF;

    result = net_fbprint(file, net);
    fclose(file);
    return result;
}

network_t *net_bload(const char *filename)
{
    FILE *file;
    network_t *net;
    assert(filename != NULL);

    file = fopen(filename, "rb");
    if (file == NULL) return NULL;

    net = net_fbscan(file);
    fclose(file);
    return net;
}

/* ── Forward Pass ────────────────────────────────────────────────────── */

static void set_input(network_t *net, const float *input)
{
    int n;
    assert(net != NULL);
    assert(input != NULL);
    for (n = 0; n < net->input_layer->no_of_neurons; n++)
        net->input_layer->neuron[n].output = input[n];
}

static void get_output(const network_t *net, float *output)
{
    int n;
    assert(net != NULL);
    assert(output != NULL);
    for (n = 0; n < net->output_layer->no_of_neurons; n++)
        output[n] = net->output_layer->neuron[n].output;
}

/* Propagate one layer.  activation_type is applied to hidden layers;
 * the output layer always uses sigmoid (called separately below).      */
static void propagate_layer(layer_t *lower, layer_t *upper, int act)
{
    int nu, nl;
    float value;
    assert(lower != NULL);
    assert(upper != NULL);

    for (nu = 0; nu < upper->no_of_neurons; nu++)
    {
        value = 0.0f;
        for (nl = 0; nl <= lower->no_of_neurons; nl++)
            value += upper->neuron[nu].weight[nl] * lower->neuron[nl].output;
        upper->neuron[nu].output = activate(value, act);
    }
}

static void forward_pass(network_t *net)
{
    int l;
    assert(net != NULL);

    /* Hidden layers: use the configured activation */
    for (l = 1; l < net->no_of_layers - 1; l++)
        propagate_layer(&net->layer[l - 1], &net->layer[l], net->activation);

    /* Output layer: always sigmoid for backward compatibility */
    if (net->no_of_layers > 1)
        propagate_layer(&net->layer[net->no_of_layers - 2],
                        &net->layer[net->no_of_layers - 1],
                        NERVENET_ACTIVATION_SIGMOID);
}

void net_compute(network_t *net, const float *input, float *output)
{
    assert(net != NULL);
    assert(input != NULL);
    set_input(net, input);
    forward_pass(net);
    if (output != NULL) get_output(net, output);
}

/* ── Error & Backward Pass ───────────────────────────────────────────── */

float net_compute_output_error(network_t *net, const float *target)
{
    int n;
    float output, error;
    assert(net != NULL);
    assert(target != NULL);

    net->global_error = 0.0f;
    for (n = 0; n < net->output_layer->no_of_neurons; n++)
    {
        output = net->output_layer->neuron[n].output;
        error  = target[n] - output;
        /* δ = σ'(x) · (t - y) = y(1-y)(t-y) for sigmoid output */
        net->output_layer->neuron[n].error = output * (1.0f - output) * error;
        net->global_error += error * error;
    }
    net->global_error *= 0.5f;
    return net->global_error;
}

float net_get_output_error(const network_t *net)
{
    assert(net != NULL);
    return net->global_error;
}

/* Backpropagate error from upper to lower.
 * lower's activation derivative uses the hidden activation.            */
static void backpropagate_layer(layer_t *lower, layer_t *upper, int act)
{
    int nl, nu;
    float error, output;
    assert(lower != NULL);
    assert(upper != NULL);

    for (nl = 0; nl <= lower->no_of_neurons; nl++)
    {
        error = 0.0f;
        for (nu = 0; nu < upper->no_of_neurons; nu++)
            error += upper->neuron[nu].weight[nl] * upper->neuron[nu].error;
        output = lower->neuron[nl].output;
        lower->neuron[nl].error = activate_deriv(output, act) * error;
    }
}

static void backward_pass(network_t *net)
{
    int l;
    assert(net != NULL);
    for (l = net->no_of_layers - 1; l > 1; l--)
        backpropagate_layer(&net->layer[l - 1], &net->layer[l],
                            net->activation);
}

/* ── Weight Update ───────────────────────────────────────────────────── */

static void adjust_weights(network_t *net)
{
    int l, nu, nl, idx;
    float error, grad, delta;
    int use_adam = (net->optimizer == NERVENET_OPTIMIZER_ADAM);
    float lr, b1, b2, eps, bc1, bc2, m_hat, v_hat;

    assert(net != NULL);

    lr = net->learning_rate;

    if (use_adam)
    {
        net->adam_t++;
        b1  = net->adam_beta1;
        b2  = net->adam_beta2;
        eps = net->adam_epsilon;
        /* Bias-correction denominators (precomputed once per step) */
        bc1 = 1.0f - (float)pow((double)b1, (double)net->adam_t);
        bc2 = 1.0f - (float)pow((double)b2, (double)net->adam_t);
    }
    else
    {
        /* silence "may be used uninitialised" warnings */
        b1 = b2 = eps = bc1 = bc2 = 0.0f;
    }

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            error = net->layer[l].neuron[nu].error;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                /* Raw gradient (positive = improve) */
                grad = error * net->layer[l - 1].neuron[nl].output;

                /* L2 weight decay: pull weights toward zero */
                if (net->l2_lambda > 0.0f)
                    grad -= net->l2_lambda *
                            net->layer[l].neuron[nu].weight[nl];

                if (use_adam)
                {
                    idx = adam_offset(net, l, nu, nl);
                    net->adam_m[idx] = b1 * net->adam_m[idx] +
                                       (1.0f - b1) * grad;
                    net->adam_v[idx] = b2 * net->adam_v[idx] +
                                       (1.0f - b2) * grad * grad;
                    m_hat = net->adam_m[idx] / bc1;
                    v_hat = net->adam_v[idx] / bc2;
                    delta = lr * m_hat /
                            ((float)sqrt((double)v_hat) + eps);
                    net->layer[l].neuron[nu].weight[nl] += delta;
                    net->layer[l].neuron[nu].delta[nl]   = delta;
                }
                else
                {
                    /* SGD + momentum */
                    delta = lr * grad +
                            net->momentum *
                            net->layer[l].neuron[nu].delta[nl];
                    net->layer[l].neuron[nu].weight[nl] += delta;
                    net->layer[l].neuron[nu].delta[nl]   = delta;
                }
            }
        }
}

/* ── Online Training ─────────────────────────────────────────────────── */

void net_train(network_t *net)
{
    assert(net != NULL);
    backward_pass(net);
    adjust_weights(net);
}

/* ── Batch Training ──────────────────────────────────────────────────── */

static void adjust_deltas_batch(network_t *net)
{
    int l, nu, nl;
    float error;
    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            error = net->layer[l].neuron[nu].error;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].delta[nl] +=
                    net->learning_rate * error *
                    net->layer[l - 1].neuron[nl].output;
        }
}

static void adjust_weights_batch(network_t *net)
{
    int l, nu, nl;
    float denom;
    assert(net != NULL);

    denom = (net->no_of_patterns > 0) ? (float)net->no_of_patterns : 1.0f;

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
                net->layer[l].neuron[nu].weight[nl] +=
                    net->layer[l].neuron[nu].delta[nl] / denom;
}

void net_begin_batch(network_t *net)
{
    assert(net != NULL);
    net->no_of_patterns = 0;
    net_reset_deltas(net);
}

void net_train_batch(network_t *net)
{
    assert(net != NULL);
    net->no_of_patterns++;
    backward_pass(net);
    adjust_deltas_batch(net);
}

void net_end_batch(network_t *net)
{
    assert(net != NULL);
    adjust_weights_batch(net);
}

/* ── Metrics ─────────────────────────────────────────────────────────── */

int net_classify(network_t *net, const float *input)
{
    int n, best = 0;
    float best_val;
    assert(net != NULL);
    assert(input != NULL);

    net_compute(net, input, NULL);
    best_val = net->output_layer->neuron[0].output;
    for (n = 1; n < net->output_layer->no_of_neurons; n++)
        if (net->output_layer->neuron[n].output > best_val)
        {
            best_val = net->output_layer->neuron[n].output;
            best = n;
        }
    return best;
}

float net_compute_accuracy(network_t *net,
                           const float *inputs, const float *targets,
                           int n_pairs, int n_inputs, int n_outputs)
{
    int i, n, correct, pred, true_class;
    float best_val;
    assert(net != NULL);
    assert(inputs  != NULL);
    assert(targets != NULL);
    assert(n_pairs > 0);

    correct = 0;
    for (i = 0; i < n_pairs; i++)
    {
        pred = net_classify(net, inputs + i * n_inputs);

        /* Argmax of target vector */
        true_class = 0;
        best_val   = targets[i * n_outputs];
        for (n = 1; n < n_outputs; n++)
            if (targets[i * n_outputs + n] > best_val)
            {
                best_val   = targets[i * n_outputs + n];
                true_class = n;
            }

        if (pred == true_class) correct++;
    }
    return (float)correct / (float)n_pairs;
}

/* ── Structural Modification ─────────────────────────────────────────── */

void net_jolt(network_t *net, float factor, float range)
{
    int l, nu, nl;
    assert(net != NULL);
    assert(factor >= 0.0f);
    assert(range  >= 0.0f);

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                if (fabs(net->layer[l].neuron[nu].weight[nl]) < (double)range)
                    net->layer[l].neuron[nu].weight[nl] =
                        2.0f * range *
                        ((float)rand() / (float)RAND_MAX - 0.5f);
                else
                    net->layer[l].neuron[nu].weight[nl] *=
                        1.0f + 2.0f * factor *
                        ((float)rand() / (float)RAND_MAX - 0.5f);
            }
}

void net_add_neurons(network_t *net, int layer, int neuron, int number,
                     float range)
{
    int l, nu, nl, new_nu, new_nl, *arglist;
    network_t *new_net, *tmp_net;

    assert(net != NULL);
    assert(0 <= layer && layer < net->no_of_layers);
    assert(0 <= neuron);
    assert(number >= 0);
    assert(range  >= 0.0f);

    if (neuron == -1)
        neuron = net->layer[layer].no_of_neurons;

    arglist = (int *)calloc(net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++)
        arglist[l] = net->layer[l].no_of_neurons;
    arglist[layer] += number;

    new_net = net_allocate_l(net->no_of_layers, arglist);
    free(arglist);

    net_randomize(new_net, range);

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            new_nu = (l == layer) && (nu >= neuron) ? nu + number : nu;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                new_nl = (l == layer + 1) && (nl >= neuron) ?
                         nl + number : nl;
                new_net->layer[l].neuron[new_nu].weight[new_nl] =
                    net->layer[l].neuron[nu].weight[nl];
                new_net->layer[l].neuron[new_nu].delta[new_nl] =
                    net->layer[l].neuron[nu].delta[nl];
            }
        }

    new_net->momentum      = net->momentum;
    new_net->learning_rate = net->learning_rate;
    new_net->activation    = net->activation;
    new_net->optimizer     = net->optimizer;
    new_net->l2_lambda     = net->l2_lambda;

    tmp_net = (network_t *)malloc(sizeof(network_t));
    memcpy(tmp_net, new_net, sizeof(network_t));
    memcpy(new_net, net,     sizeof(network_t));
    memcpy(net,     tmp_net, sizeof(network_t));
    free(tmp_net);
    net_free(new_net);
}

void net_remove_neurons(network_t *net, int layer, int neuron, int number)
{
    int l, nu, nl, orig_nu, orig_nl, *arglist;
    network_t *new_net, *tmp_net;

    assert(net != NULL);
    assert(0 <= layer && layer < net->no_of_layers);
    assert(0 <= neuron);
    assert(number >= 0);

    arglist = (int *)calloc(net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++)
        arglist[l] = net->layer[l].no_of_neurons;
    arglist[layer] -= number;

    new_net = net_allocate_l(net->no_of_layers, arglist);
    free(arglist);

    for (l = 1; l < new_net->no_of_layers; l++)
        for (nu = 0; nu < new_net->layer[l].no_of_neurons; nu++)
        {
            orig_nu = (l == layer) && (nu >= neuron) ? nu + number : nu;
            for (nl = 0; nl <= new_net->layer[l - 1].no_of_neurons; nl++)
            {
                orig_nl = (l == layer + 1) && (nl >= neuron) ?
                          nl + number : nl;
                new_net->layer[l].neuron[nu].weight[nl] =
                    net->layer[l].neuron[orig_nu].weight[orig_nl];
                new_net->layer[l].neuron[nu].delta[nl] =
                    net->layer[l].neuron[orig_nu].delta[orig_nl];
            }
        }

    new_net->momentum      = net->momentum;
    new_net->learning_rate = net->learning_rate;
    new_net->activation    = net->activation;
    new_net->optimizer     = net->optimizer;
    new_net->l2_lambda     = net->l2_lambda;

    tmp_net = (network_t *)malloc(sizeof(network_t));
    memcpy(tmp_net, new_net, sizeof(network_t));
    memcpy(new_net, net,     sizeof(network_t));
    memcpy(net,     tmp_net, sizeof(network_t));
    free(tmp_net);
    net_free(new_net);
}

network_t *net_copy(const network_t *net)
{
    int l, nu, nl, n_weights, *arglist;
    network_t *new_net;
    assert(net != NULL);

    arglist = (int *)calloc(net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++)
        arglist[l] = net->layer[l].no_of_neurons;

    new_net = net_allocate_l(net->no_of_layers, arglist);
    free(arglist);
    if (new_net == NULL) return NULL;

    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                new_net->layer[l].neuron[nu].weight[nl] =
                    net->layer[l].neuron[nu].weight[nl];
                new_net->layer[l].neuron[nu].delta[nl] =
                    net->layer[l].neuron[nu].delta[nl];
            }

    new_net->momentum       = net->momentum;
    new_net->learning_rate  = net->learning_rate;
    new_net->no_of_patterns = net->no_of_patterns;
    new_net->global_error   = net->global_error;
    new_net->activation     = net->activation;
    new_net->optimizer      = net->optimizer;
    new_net->l2_lambda      = net->l2_lambda;
    new_net->adam_t         = net->adam_t;
    new_net->adam_beta1     = net->adam_beta1;
    new_net->adam_beta2     = net->adam_beta2;
    new_net->adam_epsilon   = net->adam_epsilon;

    if (net->adam_m != NULL && net->adam_v != NULL)
    {
        n_weights       = net_get_no_of_weights(net);
        new_net->adam_m = (float *)malloc((size_t)n_weights * sizeof(float));
        new_net->adam_v = (float *)malloc((size_t)n_weights * sizeof(float));
        if (new_net->adam_m) memcpy(new_net->adam_m, net->adam_m,
                                    (size_t)n_weights * sizeof(float));
        if (new_net->adam_v) memcpy(new_net->adam_v, net->adam_v,
                                    (size_t)n_weights * sizeof(float));
    }

    return new_net;
}

void net_overwrite(network_t *dest, const network_t *src)
{
    network_t *new_net, *tmp_net;
    assert(dest != NULL);
    assert(src  != NULL);

    new_net = net_copy(src);
    tmp_net = (network_t *)malloc(sizeof(network_t));
    memcpy(tmp_net, new_net, sizeof(network_t));
    memcpy(new_net, dest,    sizeof(network_t));
    memcpy(dest,    tmp_net, sizeof(network_t));
    free(tmp_net);
    net_free(new_net);
}

/* ── Utility ─────────────────────────────────────────────────────────── */

int net_validate(const network_t *net)
{
    int l, n;
    if (net == NULL)                      return 0;
    if (net->no_of_layers < 2)            return 0;
    if (net->layer == NULL)               return 0;
    if (net->learning_rate <= 0.0f)       return 0;
    if (net->momentum < 0.0f)             return 0;
    if (net->l2_lambda < 0.0f)            return 0;

    for (l = 0; l < net->no_of_layers; l++)
    {
        if (net->layer[l].no_of_neurons <= 0) return 0;
        if (net->layer[l].neuron == NULL)     return 0;
        if (l > 0)
            for (n = 0; n < net->layer[l].no_of_neurons; n++)
            {
                if (net->layer[l].neuron[n].weight == NULL) return 0;
                if (net->layer[l].neuron[n].delta  == NULL) return 0;
            }
    }

    if (net->optimizer == NERVENET_OPTIMIZER_ADAM)
        if (net->adam_m == NULL || net->adam_v == NULL) return 0;

    return 1;
}

const char *net_get_version(void)
{
    return NERVENET_VERSION;
}
