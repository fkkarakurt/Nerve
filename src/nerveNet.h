/*
 * Nerve — Neural Network Library
 * Copyright (c) 2022 Fatih Küçükkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * Nerve — Neural Network Library
 *
 * Lightweight, zero-dependency multi-layer perceptron in ANSI C.
 * Supports SGD with momentum and the Adam optimizer, multiple activation
 * functions, Xavier / He weight initialisation, and L2 regularisation.
 *
 * @file  nerveNet.h
 * @version 2.0.0
 */

#ifndef NERVENET_H
#define NERVENET_H

#include <stdio.h>

/* ── Version ──────────────────────────────────────────────────────────── */
#define NERVENET_VERSION_MAJOR 2
#define NERVENET_VERSION_MINOR 0
#define NERVENET_VERSION_PATCH 0
#define NERVENET_VERSION       "2.0.0"

/* ── Defaults ─────────────────────────────────────────────────────────── */
#define NERVENET_DEFAULT_MOMENTUM      0.1f
#define NERVENET_DEFAULT_LEARNING_RATE 0.25f
#define NERVENET_DEFAULT_WEIGHT_RANGE  1.0f
#define NERVENET_MAX_LAYERS            256
#define NERVENET_MAX_NEURONS           65536

/* Adam hyper-parameter defaults (Kingma & Ba, 2014) */
#define NERVENET_ADAM_BETA1    0.9f
#define NERVENET_ADAM_BETA2    0.999f
#define NERVENET_ADAM_EPSILON  1e-8f
#define NERVENET_ADAM_LR       0.001f

/* ── Error Codes ──────────────────────────────────────────────────────── */
typedef enum
{
    NERVENET_SUCCESS             =  0,
    NERVENET_ERROR_NULL_POINTER  = -1,
    NERVENET_ERROR_INVALID_PARAM = -2,
    NERVENET_ERROR_MEMORY        = -3,
    NERVENET_ERROR_IO            = -4,
    NERVENET_ERROR_CORRUPT       = -5,
    NERVENET_ERROR_LAYER_COUNT   = -6,
    NERVENET_ERROR_NEURON_COUNT  = -7
} nervenet_error_t;

/* ── Activation Functions ─────────────────────────────────────────────── */
/**
 * Activation function applied to every hidden layer.
 * The output layer always uses the logistic sigmoid for backward
 * compatibility with the MSE error signal.
 */
typedef enum
{
    NERVENET_ACTIVATION_SIGMOID    = 0, /**< 1/(1+exp(-x))  — default        */
    NERVENET_ACTIVATION_TANH       = 1, /**< tanh(x)        — often converges faster */
    NERVENET_ACTIVATION_RELU       = 2, /**< max(0,x)       — deep networks  */
    NERVENET_ACTIVATION_LEAKY_RELU = 3  /**< max(0.01x, x)  — avoids dead neurons */
} nervenet_activation_t;

/* ── Optimizers ───────────────────────────────────────────────────────── */
/**
 * Weight-update rule used during training.
 */
typedef enum
{
    NERVENET_OPTIMIZER_SGD  = 0, /**< SGD + momentum (Rumelhart et al., 1986) — default */
    NERVENET_OPTIMIZER_ADAM = 1  /**< Adam           (Kingma & Ba, 2014)               */
} nervenet_optimizer_t;

/* ── Weight Initialisation ────────────────────────────────────────────── */
typedef enum
{
    NERVENET_INIT_UNIFORM = 0, /**< Uniform [-range, +range]  — default           */
    NERVENET_INIT_XAVIER  = 1, /**< Glorot & Bengio (2010)    — sigmoid / tanh    */
    NERVENET_INIT_HE      = 2  /**< He et al. (2015)          — ReLU / Leaky ReLU */
} nervenet_init_t;

/* ── Core Data Structures ─────────────────────────────────────────────── */

typedef struct neuron_s
{
    float  output;  /**< Post-activation value                              */
    float  error;   /**< Local gradient δ used in backpropagation           */
    float *weight;  /**< Incoming weights; index n_lower_neurons == bias     */
    float *delta;   /**< Weight deltas for momentum / Adam first moment      */
} neuron_t;

typedef struct layer_s
{
    int       no_of_neurons; /**< Number of neurons (excluding bias unit)   */
    neuron_t *neuron;        /**< Neuron array; neuron[n] is the bias unit  */
} layer_t;

typedef struct network_s
{
    /* ── Core ── */
    int     no_of_layers;   /**< Total layers (input + hidden… + output)    */
    int     no_of_patterns; /**< Patterns accumulated in current batch      */
    float   momentum;       /**< SGD momentum factor                        */
    float   learning_rate;  /**< Step size                                  */
    float   global_error;   /**< MSE from last net_compute_output_error()   */
    layer_t *layer;         /**< All layers [0..no_of_layers-1]             */
    layer_t *input_layer;   /**< Convenience pointer → layer[0]            */
    layer_t *output_layer;  /**< Convenience pointer → layer[n-1]          */

    /* ── Algorithm Selection ── */
    int activation; /**< nervenet_activation_t for all hidden layers        */
    int optimizer;  /**< nervenet_optimizer_t                               */

    /* ── Regularisation ── */
    float l2_lambda; /**< L2 weight-decay coefficient (0 = disabled)        */

    /* ── Adam State (NULL when SGD is active) ── */
    float *adam_m;      /**< First-moment vectors, flattened over all weights */
    float *adam_v;      /**< Second-moment vectors                           */
    int    adam_t;      /**< Time step (incremented each net_train() call)   */
    float  adam_beta1;
    float  adam_beta2;
    float  adam_epsilon;
} network_t;

/* ── C++ Guard ────────────────────────────────────────────────────────── */
#ifdef __cplusplus
extern "C"
{
#endif

    /* ── Allocation ── */

    /** Allocate a network; layers sizes passed as variadic ints. */
    network_t *net_allocate(int no_of_layers, ...);

    /** Allocate a network; layer sizes given as an array. */
    network_t *net_allocate_l(int no_of_layers, const int *arglist);

    /** Free all memory associated with a network. */
    void net_free(network_t *net);

    /* ── Weight Initialisation ── */

    /** Uniform random in [-range, +range]. */
    void net_randomize(network_t *net, float range);

    /**
     * Xavier / Glorot uniform initialisation (Glorot & Bengio, 2010).
     * Optimal for sigmoid and tanh activations.
     */
    void net_initialize_xavier(network_t *net);

    /**
     * He uniform initialisation (He et al., 2015).
     * Optimal for ReLU and Leaky ReLU activations.
     */
    void net_initialize_he(network_t *net);

    /** Zero all weight deltas. */
    void net_reset_deltas(network_t *net);

    /* ── Configuration ── */

    void net_set_momentum(network_t *net, float momentum);
    void net_set_learning_rate(network_t *net, float learning_rate);

    /**
     * Enable or disable the bias neuron for every layer.
     * Bias is enabled by default.
     */
    void net_use_bias(network_t *net, int use_bias);

    /**
     * Set the hidden-layer activation function.
     * The output layer always uses the logistic sigmoid.
     */
    void net_set_activation(network_t *net, nervenet_activation_t act);

    /**
     * Switch between SGD+momentum and Adam.
     * Switching to Adam allocates the moment arrays and resets the time
     * step; switching back to SGD frees them.
     */
    void net_set_optimizer(network_t *net, nervenet_optimizer_t opt);

    /**
     * Set the L2 regularisation coefficient λ.
     * Pass 0.0 to disable (default).
     */
    void net_set_l2_lambda(network_t *net, float lambda);

    /* ── Query ── */

    float net_get_momentum(const network_t *net);
    float net_get_learning_rate(const network_t *net);
    int   net_get_no_of_inputs(const network_t *net);
    int   net_get_no_of_outputs(const network_t *net);
    int   net_get_no_of_layers(const network_t *net);

    /** Returns the total number of trainable weights (including biases). */
    int   net_get_no_of_weights(const network_t *net);

    /* ── Weight Access ── */

    void  net_set_weight(network_t *net, int layer_index,
                         int neuron_lower, int neuron_upper, float weight);
    float net_get_weight(const network_t *net, int layer_index,
                         int neuron_lower, int neuron_upper);
    void  net_set_bias(network_t *net, int layer_index,
                       int neuron_index, float weight);
    float net_get_bias(const network_t *net, int layer_index,
                       int neuron_index);

    /* ── Text I/O ── */

    int        net_fprint(FILE *file, const network_t *net);
    network_t *net_fscan(FILE *file);
    int        net_print(const network_t *net);
    int        net_save(const char *filename, const network_t *net);
    network_t *net_load(const char *filename);

    /* ── Binary I/O ── */

    int        net_fbprint(FILE *file, const network_t *net);
    network_t *net_fbscan(FILE *file);
    int        net_bsave(const char *filename, const network_t *net);
    network_t *net_bload(const char *filename);

    /* ── Forward Pass & Error ── */

    /** Run a forward pass; if output != NULL, copy results there. */
    void  net_compute(network_t *net, const float *input, float *output);

    /**
     * Compute the MSE output error and store local gradients in output
     * neurons.  Must be called before net_train() / net_train_batch().
     */
    float net_compute_output_error(network_t *net, const float *target);

    float net_get_output_error(const network_t *net);

    /* ── Online Training ── */

    /**
     * Perform one backpropagation + weight-update step.
     * Call net_compute() and net_compute_output_error() first.
     */
    void net_train(network_t *net);

    /* ── Batch Training ── */

    void net_begin_batch(network_t *net);
    void net_train_batch(network_t *net);
    void net_end_batch(network_t *net);

    /* ── Metrics ── */

    /**
     * Returns the index of the output neuron with the highest activation
     * (argmax classification).
     */
    int net_classify(network_t *net, const float *input);

    /**
     * Compute classification accuracy over a dataset stored as flat arrays.
     *
     * @param inputs   Flat array of size n_pairs × n_inputs
     * @param targets  Flat array of size n_pairs × n_outputs (one-hot)
     * @return Fraction of correctly classified samples in [0, 1]
     */
    float net_compute_accuracy(network_t *net,
                               const float *inputs, const float *targets,
                               int n_pairs, int n_inputs, int n_outputs);

    /* ── Structural Modification ── */

    void       net_jolt(network_t *net, float factor, float range);
    void       net_add_neurons(network_t *net, int layer_index,
                               int neuron_index, int number, float range);
    void       net_remove_neurons(network_t *net, int layer_index,
                                  int neuron_index, int number);
    network_t *net_copy(const network_t *net);
    void       net_overwrite(network_t *dest, const network_t *src);

    /* ── Utility ── */

    /** Returns 1 if the network is internally consistent, 0 otherwise. */
    int         net_validate(const network_t *net);

    /** Returns the library version string, e.g. "2.0.0". */
    const char *net_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NERVENET_H */
