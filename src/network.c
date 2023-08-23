#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "nerveNet.h"

#define DEFAULT_MOMENTUM 0.1
#define DEFAULT_LEARNING_RATE 0.25
#define DEFAULT_WEIGHT_RANGE 1.0

void net_randomize(network_t *net, float range)
{
    int l, nu, nl;

    assert(net != NULL);
    assert(range >= 0.0);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                net->layer[l].neuron[nu].weight[nl] =
                    2.0 * range * ((float)random() / RAND_MAX - 0.5);
            }
        }
    }
}

#if 0
#endif

void net_reset_deltas(network_t *net)
{
    int l, nu, nl;

    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                net->layer[l].neuron[nu].delta[nl] = 0.0;
            }
        }
    }
}

void net_use_bias(network_t *net, int flag)
{
    int l;

    assert(net != NULL);

    if (flag != 0)
    {
        for (l = 0; l < net->no_of_layers; l++)
        {
            net->layer[l].neuron[net->layer[l].no_of_neurons].output = 1.0;
        }
    }
    else
    {
        for (l = 0; l < net->no_of_layers; l++)
        {
            net->layer[l].neuron[net->layer[l].no_of_neurons].output = 0.0;
        }
    }
}

static void
allocate_layer(layer_t *layer, int no_of_neurons)
{
    assert(layer != NULL);
    assert(no_of_neurons > 0);

    layer->no_of_neurons = no_of_neurons;
    layer->neuron = (neuron_t *)calloc(no_of_neurons + 1, sizeof(neuron_t));
}

static void
allocate_weights(layer_t *lower, layer_t *upper)
{
    int n;

    assert(lower != NULL);
    assert(upper != NULL);

    for (n = 0; n < upper->no_of_neurons; n++)
    {
        upper->neuron[n].weight =
            (float *)calloc(lower->no_of_neurons + 1, sizeof(float));
        upper->neuron[n].delta =
            (float *)calloc(lower->no_of_neurons + 1, sizeof(float));
    }

    upper->neuron[n].weight = NULL;
    upper->neuron[n].delta = NULL;
}

network_t *
net_allocate_l(int no_of_layers, const int *arglist)
{
    int l;
    network_t *net;

    assert(no_of_layers >= 2);
    assert(arglist != NULL);

    net = (network_t *)malloc(sizeof(network_t));
    net->no_of_layers = no_of_layers;
    net->layer = (layer_t *)calloc(no_of_layers, sizeof(layer_t));
    for (l = 0; l < no_of_layers; l++)
    {
        assert(arglist[l] > 0);
        allocate_layer(&net->layer[l], arglist[l]);
    }
    for (l = 1; l < no_of_layers; l++)
    {
        allocate_weights(&net->layer[l - 1], &net->layer[l]);
    }

    net->input_layer = &net->layer[0];
    net->output_layer = &net->layer[no_of_layers - 1];

    net->momentum = DEFAULT_MOMENTUM;
    net->learning_rate = DEFAULT_LEARNING_RATE;

    net_randomize(net, DEFAULT_WEIGHT_RANGE);
    net_reset_deltas(net);

    net_use_bias(net, 1);

    return net;
}

network_t *
net_allocate(int no_of_layers, ...)
{
    int l, *arglist;
    va_list args;
    network_t *net;

    assert(no_of_layers >= 2);

    arglist = calloc(no_of_layers, sizeof(int));
    va_start(args, no_of_layers);
    for (l = 0; l < no_of_layers; l++)
    {
        arglist[l] = va_arg(args, int);
    }
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
    free(net);
}

void net_set_momentum(network_t *net, float momentum)
{
    assert(net != NULL);
    assert(momentum >= 0.0);

    net->momentum = momentum;
}

float net_get_momentum(const network_t *net)
{
    assert(net != NULL);
    assert(net->momentum >= 0.0);

    return net->momentum;
}

void net_set_learning_rate(network_t *net, float learning_rate)
{
    assert(net != NULL);
    assert(learning_rate >= 0.0);

    net->learning_rate = learning_rate;
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
    int l, result;

    assert(net != NULL);

    result = 0;
    for (l = 1; l < net->no_of_layers; l++)
    {
        result += (net->layer[l - 1].no_of_neurons + 1) * net->layer[l].no_of_neurons;
    }

    return result;
}

void net_set_weight(network_t *net, int l, int nl, int nu, float weight)
{
    assert(net != NULL);
    assert(0 <= l && l < net->no_of_layers);
    assert(0 <= nl && nl <= net->layer[l].no_of_neurons);
    assert(0 <= nu && nu < net->layer[l + 1].no_of_neurons);

    net->layer[l].neuron[nu].weight[nl] = weight;
}

float net_get_weight(const network_t *net, int l, int nl, int nu)
{
    assert(net != NULL);
    assert(0 <= l && l < net->no_of_layers - 1);
    assert(0 <= nl && nl <= net->layer[l].no_of_neurons);
    assert(0 <= nu && nu < net->layer[l + 1].no_of_neurons);

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

int net_fprint(FILE *file, const network_t *net)
{
    int l, nu, nl, result;

    assert(file != NULL);
    assert(net != NULL);

    result = fprintf(file, "%i\n", net->no_of_layers);
    if (result < 0)
    {
        return result;
    }
    for (l = 0; l < net->no_of_layers; l++)
    {
        result = fprintf(file, "%i\n", net->layer[l].no_of_neurons);
        if (result < 0)
        {
            return result;
        }
    }

    result = fprintf(file, "%f\n", net->momentum);
    if (result < 0)
    {
        return result;
    }
    result = fprintf(file, "%f\n", net->learning_rate);
    if (result < 0)
    {
        return result;
    }
    result = fprintf(file, "%f\n", net->global_error);
    if (result < 0)
    {
        return result;
    }

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                result = fprintf(file, "%f\n", net->layer[l].neuron[nu].weight[nl]);
                if (result < 0)
                {
                    return result;
                }
            }
        }
    }

    return 0;
}

network_t *
net_fscan(FILE *file)
{
    int no_of_layers, l, nu, nl, *arglist, result;
    network_t *net;

    assert(file != NULL);

    result = fscanf(file, "%i", &no_of_layers);
    if (result <= 0)
    {
        return NULL;
    }
    arglist = calloc(no_of_layers, sizeof(int));
    if (arglist == NULL)
    {
        return NULL;
    }
    for (l = 0; l < no_of_layers; l++)
    {
        result = fscanf(file, "%i", &arglist[l]);
        if (result <= 0)
        {
            return NULL;
        }
    }

    net = net_allocate_l(no_of_layers, arglist);
    free(arglist);
    if (net == NULL)
    {
        return NULL;
    }

    result = fscanf(file, "%f", &net->momentum);
    if (result <= 0)
    {
        net_free(net);
        return NULL;
    }
    result = fscanf(file, "%f", &net->learning_rate);
    if (result <= 0)
    {
        net_free(net);
        return NULL;
    }
    result = fscanf(file, "%f", &net->global_error);
    if (result <= 0)
    {
        net_free(net);
        return NULL;
    }

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                result = fscanf(file, "%f", &net->layer[l].neuron[nu].weight[nl]);
                if (result <= 0)
                {
                    net_free(net);
                    return NULL;
                }
            }
        }
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
    if (file == NULL)
    {
        return EOF;
    }
    result = net_fprint(file, net);
    if (result < 0)
    {
        fclose(file);
        return result;
    }
    return fclose(file);
}

network_t *
net_load(const char *filename)
{
    FILE *file;
    network_t *net;

    assert(filename != NULL);

    file = fopen(filename, "r");
    if (file == NULL)
    {
        return NULL;
    }
    net = net_fscan(file);
    fclose(file);

    return net;
}

int net_fbprint(FILE *file, const network_t *net)
{
    int l, nu;
    size_t info_dim = net->no_of_layers + 1;
    int info[info_dim];
    float constants[3];
    assert(file != NULL);
    assert(net != NULL);
    info[0] = net->no_of_layers;
    for (l = 0; l < net->no_of_layers; l++)
    {
        info[l + 1] = net->layer[l].no_of_neurons;
    }
    if (fwrite(info, sizeof(int), info_dim, file) < info_dim)
    {
        return -1;
    }

    constants[0] = net->momentum;
    constants[1] = net->learning_rate;
    constants[2] = net->global_error;
    fwrite(constants, sizeof(float), 3, file);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            fwrite(net->layer[l].neuron[nu].weight, sizeof(float),
                   net->layer[l - 1].no_of_neurons + 1, file);
        }
    }

    return 0;
}

network_t *
net_fbscan(FILE *file)
{
    int no_of_layers, l, nu, *arglist;
    network_t *net;

    // A variable to store the value returned by the fread function
    size_t read_items;

    assert(file != NULL);

    if (fread(&no_of_layers, sizeof(int), 1, file) < 1)
    {
        return NULL;
    }
    arglist = calloc(no_of_layers, sizeof(int));
    if (fread(arglist, sizeof(int), no_of_layers, file) < (size_t)no_of_layers)
    {
        free(arglist);
        return NULL;
    }

    net = net_allocate_l(no_of_layers, arglist);
    free(arglist);

    read_items = fread(&net->momentum, sizeof(float), 1, file);
    if (read_items < 1)
        return NULL;

    read_items = fread(&net->learning_rate, sizeof(float), 1, file);
    if (read_items < 1)
        return NULL;

    read_items = fread(&net->global_error, sizeof(float), 1, file);
    if (read_items < 1)
        return NULL;

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            read_items = fread(net->layer[l].neuron[nu].weight, sizeof(float),
                               net->layer[l - 1].no_of_neurons + 1, file);
            if (read_items < (size_t)(net->layer[l - 1].no_of_neurons + 1))
            {
                net_free(net);
                return NULL;
            }
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
    result = net_fbprint(file, net);
    if (result < 0)
    {
        fclose(file);
        return result;
    }
    return fclose(file);
}

network_t *
net_bload(const char *filename)
{
    FILE *file;
    network_t *net;

    assert(filename != NULL);

    file = fopen(filename, "rb");
    net = net_fbscan(file);
    fclose(file);

    return net;
}

// Input and Output

static inline void
set_input(network_t *net, const float *input)
{
    int n;

    assert(net != NULL);
    assert(input != NULL);

    for (n = 0; n < net->input_layer->no_of_neurons; n++)
    {
        net->input_layer->neuron[n].output = input[n];
    }
}

static inline void
get_output(const network_t *net, float *output)
{
    int n;

    assert(net != NULL);
    assert(output != NULL);

    for (n = 0; n < net->output_layer->no_of_neurons; n++)
    {
        output[n] = net->output_layer->neuron[n].output;
    }
}

// Sigmoidal function
#if 1
#include "interpolation.c"

static inline float
sigma(float x)
{
    int index;

    index = (int)((x - min_entry) / interval);

    if (index <= 0)
    {
        return interpolation[0];
    }
    else if (index >= num_entries)
    {
        return interpolation[num_entries - 1];
    }
    else
    {
        return interpolation[index];
    }
}

#else

#endif

// Forward and Backward Propagation

static inline void
propagate_layer(layer_t *lower, layer_t *upper)
{
    int nu, nl;
    float value;

    assert(lower != NULL);
    assert(upper != NULL);

    for (nu = 0; nu < upper->no_of_neurons; nu++)
    {
        value = 0.0;
        for (nl = 0; nl <= lower->no_of_neurons; nl++)
        {
            value += upper->neuron[nu].weight[nl] * lower->neuron[nl].output;
        }
#if 0
    upper->neuron[nu].activation = value;
#endif
        upper->neuron[nu].output = sigma(value);
    }
}

static inline void
forward_pass(network_t *net)
{
    int l;

    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
    {
        propagate_layer(&net->layer[l - 1], &net->layer[l]);
    }
}

float net_compute_output_error(network_t *net, const float *target)
{
    int n;
    float output, error;

    assert(net != NULL);
    assert(target != NULL);

    net->global_error = 0.0;
    for (n = 0; n < net->output_layer->no_of_neurons; n++)
    {
        output = net->output_layer->neuron[n].output;
        error = target[n] - output;
        net->output_layer->neuron[n].error = output * (1.0 - output) * error;
        net->global_error += error * error;
    }
    net->global_error *= 0.5;

    return net->global_error;
}

float net_get_output_error(const network_t *net)
{
    assert(net != NULL);

    return net->global_error;
}

static inline void
backpropagate_layer(layer_t *lower, layer_t *upper)
{
    int nl, nu;
    float output, error;

    assert(lower != NULL);
    assert(upper != NULL);

    for (nl = 0; nl <= lower->no_of_neurons; nl++)
    {
        error = 0.0;
        for (nu = 0; nu < upper->no_of_neurons; nu++)
        {
            error += upper->neuron[nu].weight[nl] * upper->neuron[nu].error;
        }
        output = lower->neuron[nl].output;
        lower->neuron[nl].error = output * (1.0 - output) * error;
    }
}

static inline void
backward_pass(network_t *net)
{
    int l;

    assert(net != NULL);

    for (l = net->no_of_layers - 1; l > 1; l--)
    {
        backpropagate_layer(&net->layer[l - 1], &net->layer[l]);
    }
}

static inline void
adjust_weights(network_t *net)
{
    int l, nu, nl;
    float error, delta;

    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            error = net->layer[l].neuron[nu].error;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
#if 1
                delta =
                    net->learning_rate * error * net->layer[l - 1].neuron[nl].output +
                    net->momentum * net->layer[l].neuron[nu].delta[nl];
                net->layer[l].neuron[nu].weight[nl] += delta;
                net->layer[l].neuron[nu].delta[nl] = delta;
#else
                net->layer[l].neuron[nu].weight[nl] +=
                    net->learning_rate * error * net->layer[l - 1].neuron[nl].output;
#endif
            }
        }
    }
}

// Evaluation and Training
void net_compute(network_t *net, const float *input, float *output)
{
    assert(net != NULL);
    assert(input != NULL);

    set_input(net, input);
    forward_pass(net);
    if (output != NULL)
    {
        get_output(net, output);
    }
}

void net_train(network_t *net)
{
    assert(net != NULL);

    backward_pass(net);
    adjust_weights(net);
}

// Batch Training
static inline void
adjust_deltas_batch(network_t *net)
{
    int l, nu, nl;
    float error;

    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            error = net->layer[l].neuron[nu].error;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                net->layer[l].neuron[nu].delta[nl] +=
                    net->learning_rate * error * net->layer[l - 1].neuron[nl].output;
            }
        }
    }
}

static inline void
adjust_weights_batch(network_t *net)
{
    int l, nu, nl;

    assert(net != NULL);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                net->layer[l].neuron[nu].weight[nl] +=
                    net->layer[l].neuron[nu].delta[nl] / net->no_of_patterns;
            }
        }
    }
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

// Modification

void net_jolt(network_t *net, float factor, float range)
{
    int l, nu, nl;

    assert(net != NULL);
    assert(factor >= 0.0);
    assert(range >= 0.0);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                if (fabs(net->layer[l].neuron[nu].weight[nl]) < range)
                {
                    net->layer[l].neuron[nu].weight[nl] =
                        2.0 * range * ((float)random() / RAND_MAX - 0.5);
                }
                else
                {
                    net->layer[l].neuron[nu].weight[nl] *=
                        1.0 + 2.0 * factor * ((float)random() / RAND_MAX - 0.5);
                }
            }
        }
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
    assert(range >= 0.0);

    if (neuron == -1)
    {
        neuron = net->layer[layer].no_of_neurons;
    }

    arglist = calloc(net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++)
    {
        arglist[l] = net->layer[l].no_of_neurons;
    }
    arglist[layer] += number;
    new_net = net_allocate_l(net->no_of_layers, arglist);
    free(arglist);

    net_randomize(net, range);

    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            new_nu = (l == layer) && (nu >= neuron) ? nu + number : nu;
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                new_nl = (l == layer + 1) && (nl >= neuron) ? nl + number : nl;
                new_net->layer[l].neuron[new_nu].weight[new_nl] =
                    net->layer[l].neuron[nu].weight[nl];
                new_net->layer[l].neuron[new_nu].delta[new_nl] =
                    net->layer[l].neuron[nu].delta[nl];
            }
        }
    }
    new_net->momentum = net->momentum;
    new_net->learning_rate = net->learning_rate;
    tmp_net = malloc(sizeof(network_t));
    memcpy(tmp_net, new_net, sizeof(network_t));
    memcpy(new_net, net, sizeof(network_t));
    memcpy(net, tmp_net, sizeof(network_t));
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
    arglist = calloc(net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++)
    {
        arglist[l] = net->layer[l].no_of_neurons;
    }
    arglist[layer] -= number;
    new_net = net_allocate_l(net->no_of_layers, arglist);
    free(arglist);
    for (l = 1; l < new_net->no_of_layers; l++)
    {
        for (nu = 0; nu < new_net->layer[l].no_of_neurons; nu++)
        {
            orig_nu = (l == layer) && (nu >= neuron) ? nu + number : nu;
            for (nl = 0; nl <= new_net->layer[l - 1].no_of_neurons; nl++)
            {
                orig_nl = (l == layer + 1) && (nl >= neuron) ? nl + number : nl;
                new_net->layer[l].neuron[nu].weight[nl] =
                    net->layer[l].neuron[orig_nu].weight[orig_nl];
                new_net->layer[l].neuron[nu].delta[nl] =
                    net->layer[l].neuron[orig_nu].delta[orig_nl];
            }
        }
    }
    new_net->momentum = net->momentum;
    new_net->learning_rate = net->learning_rate;
    tmp_net = malloc(sizeof(network_t));
    memcpy(tmp_net, new_net, sizeof(network_t));
    memcpy(new_net, net, sizeof(network_t));
    memcpy(net, tmp_net, sizeof(network_t));
    free(tmp_net);
    net_free(new_net);
}

network_t *
net_copy(const network_t *net)
{
    int l, nu, nl, *arglist;
    network_t *new_net;
    assert(net != NULL);
    arglist = calloc(net->no_of_layers, sizeof(int));
    for (l = 0; l < net->no_of_layers; l++)
    {
        arglist[l] = net->layer[l].no_of_neurons;
    }
    new_net = net_allocate_l(net->no_of_layers, arglist);
    free(arglist);
    for (l = 1; l < net->no_of_layers; l++)
    {
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
        {
            for (nl = 0; nl <= net->layer[l - 1].no_of_neurons; nl++)
            {
                new_net->layer[l].neuron[nu].weight[nl] =
                    net->layer[l].neuron[nu].weight[nl];
                new_net->layer[l].neuron[nu].delta[nl] =
                    net->layer[l].neuron[nu].delta[nl];
            }
        }
    }
    new_net->momentum = net->momentum;
    new_net->learning_rate = net->learning_rate;
    new_net->no_of_patterns = net->no_of_patterns;

    return new_net;
}

void net_overwrite(network_t *dest, const network_t *src)
{
    network_t *new_net, *tmp_net;
    assert(dest != NULL);
    assert(src != NULL);
    new_net = net_copy(src);
    tmp_net = malloc(sizeof(network_t));
    memcpy(tmp_net, new_net, sizeof(network_t));
    memcpy(new_net, dest, sizeof(network_t));
    memcpy(dest, tmp_net, sizeof(network_t));
    free(tmp_net);
    net_free(new_net);
}
