#ifndef NERVENET_H
#define NERVENET_H

#include <stdio.h>

typedef struct
{
    float output;
    float error;
    float *weight;
    float *delta;
} neuron_t;

typedef struct
{
    int no_of_neurons;
    neuron_t *neuron;
} layer_t;

typedef struct
{
    int no_of_layers;
    int no_of_patterns;
    float momentum;
    float learning_rate;
    float global_error;
    layer_t *layer;
    layer_t *input_layer;
    layer_t *output_layer;
} network_t;

#ifdef __cplusplus
extern "C"
{
#endif

    network_t *net_allocate(int, ...);
    network_t *net_allocate_l(int, const int *);
    void net_free(network_t *);
    void net_randomize(network_t *, float);
    void net_reset_deltas(network_t *);
    void net_set_momentum(network_t *, float);
    void net_set_learning_rate(network_t *, float);
    float net_get_momentum(const network_t *);
    float net_get_learning_rate(const network_t *);
    int net_get_no_of_inputs(const network_t *);
    int net_get_no_of_outputs(const network_t *);
    int net_get_no_of_layers(const network_t *);
    int net_get_no_of_weights(const network_t *);
    void net_set_weight(network_t *, int, int, int, float);
    float net_get_weight(const network_t *, int, int, int);
    void net_use_bias(network_t *, int);
    void net_set_bias(network_t *, int, int, float);
    float net_get_bias(const network_t *, int, int);
    int net_fprint(FILE *, const network_t *);
    network_t *net_fscan(FILE *);
    int net_print(const network_t *);
    int net_save(const char *, const network_t *);
    network_t *net_load(const char *);
    int net_fbprint(FILE *, const network_t *);
    network_t *net_fbscan(FILE *);
    int net_fbprint(FILE *, const network_t *);
    int net_bsave(const char *, const network_t *);
    network_t *net_bload(const char *);
    void net_compute(network_t *, const float *, float *);
    float net_compute_output_error(network_t *, const float *);
    float net_get_output_error(const network_t *);
    void net_train(network_t *);
    void net_begin_batch(network_t *);
    void net_train_batch(network_t *);
    void net_end_batch(network_t *);
    void net_jolt(network_t *, float, float);
    void net_add_neurons(network_t *, int, int, int, float);
    void net_remove_neurons(network_t *, int, int, int);
    network_t *net_copy(const network_t *);
    void net_overwrite(network_t *, const network_t *);

#ifdef __cplusplus
}
#endif

#endif
