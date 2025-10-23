/**
 * Highly Optimized Neural Network Library Header
 * ANSI C Compliant - No External Dependencies
 * Professional Grade - Maximum Performance & Safety
 *
 * @file nerveNet.h
 * @brief Neural network implementation with forward/backward propagation
 */

#ifndef NERVENET_H
#define NERVENET_H

#include <stdio.h>

/* Configuration Constants */
#define NERVENET_DEFAULT_MOMENTUM 0.1f
#define NERVENET_DEFAULT_LEARNING_RATE 0.25f
#define NERVENET_DEFAULT_WEIGHT_RANGE 1.0f
#define NERVENET_MAX_LAYERS 256
#define NERVENET_MAX_NEURONS 65536

/* Memory Alignment for Cache Optimization */
#ifndef NERVENET_MEMORY_ALIGNMENT
#define NERVENET_MEMORY_ALIGNMENT 64
#endif

/* Error Codes */
typedef enum
{
    NERVENET_SUCCESS = 0,
    NERVENET_ERROR_NULL_POINTER = -1,
    NERVENET_ERROR_INVALID_PARAMETER = -2,
    NERVENET_ERROR_MEMORY_ALLOCATION = -3,
    NERVENET_ERROR_IO_OPERATION = -4,
    NERVENET_ERROR_NETWORK_CORRUPT = -5,
    NERVENET_ERROR_LAYER_COUNT = -6,
    NERVENET_ERROR_NEURON_COUNT = -7
} nervenet_error_t;

/* Neural Network Structures */
typedef struct neuron_s
{
    float output;  /**< Neuron output value */
    float error;   /**< Error term for backpropagation */
    float *weight; /**< Connection weights to previous layer */
    float *delta;  /**< Weight deltas for momentum */
} neuron_t;

typedef struct layer_s
{
    int no_of_neurons; /**< Number of neurons in this layer */
    neuron_t *neuron;  /**< Array of neurons */
} layer_t;

typedef struct network_s
{
    int no_of_layers;      /**< Total number of layers in network */
    int no_of_patterns;    /**< Number of training patterns in batch */
    float momentum;        /**< Momentum factor for training */
    float learning_rate;   /**< Learning rate for weight updates */
    float global_error;    /**< Global error for last computation */
    layer_t *layer;        /**< Array of network layers */
    layer_t *input_layer;  /**< Pointer to input layer for fast access */
    layer_t *output_layer; /**< Pointer to output layer for fast access */
} network_t;

/* C++ Compatibility */
#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Allocate a new neural network with variable arguments
     *
     * @param no_of_layers Number of layers in the network (must be >= 2)
     * @param ... Variable number of integers specifying neurons per layer
     * @return network_t* Pointer to allocated network, NULL on failure
     */
    network_t *net_allocate(int no_of_layers, ...);

    /**
     * @brief Allocate a new neural network with array argument
     *
     * @param no_of_layers Number of layers in the network
     * @param arglist Array of integers specifying neurons per layer
     * @return network_t* Pointer to allocated network, NULL on failure
     */
    network_t *net_allocate_l(int no_of_layers, const int *arglist);

    /**
     * @brief Free all memory associated with a neural network
     *
     * @param net Pointer to network to free
     */
    void net_free(network_t *net);

    /**
     * @brief Randomize network weights within specified range
     *
     * @param net Network to randomize
     * @param range Weight range (-range to +range)
     */
    void net_randomize(network_t *net, float range);

    /**
     * @brief Reset all weight deltas to zero
     *
     * @param net Network to reset
     */
    void net_reset_deltas(network_t *net);

    /**
     * @brief Set momentum factor for training
     *
     * @param net Network to configure
     * @param momentum Momentum value (0.0 to 1.0)
     */
    void net_set_momentum(network_t *net, float momentum);

    /**
     * @brief Set learning rate for training
     *
     * @param net Network to configure
     * @param learning_rate Learning rate value (> 0.0)
     */
    void net_set_learning_rate(network_t *net, float learning_rate);

    /**
     * @brief Get current momentum factor
     *
     * @param net Network to query
     * @return float Current momentum value
     */
    float net_get_momentum(const network_t *net);

    /**
     * @brief Get current learning rate
     *
     * @param net Network to query
     * @return float Current learning rate
     */
    float net_get_learning_rate(const network_t *net);

    /**
     * @brief Get number of input neurons
     *
     * @param net Network to query
     * @return int Number of input neurons
     */
    int net_get_no_of_inputs(const network_t *net);

    /**
     * @brief Get number of output neurons
     *
     * @param net Network to query
     * @return int Number of output neurons
     */
    int net_get_no_of_outputs(const network_t *net);

    /**
     * @brief Get total number of layers
     *
     * @param net Network to query
     * @return int Number of layers
     */
    int net_get_no_of_layers(const network_t *net);

    /**
     * @brief Get total number of weights in network
     *
     * @param net Network to query
     * @return int Total number of weights
     */
    int net_get_no_of_weights(const network_t *net);

    /**
     * @brief Set specific weight value
     *
     * @param net Network to modify
     * @param layer_index Layer index (0 to no_of_layers-2)
     * @param neuron_lower Lower layer neuron index
     * @param neuron_upper Upper layer neuron index
     * @param weight New weight value
     */
    void net_set_weight(network_t *net, int layer_index, int neuron_lower,
                        int neuron_upper, float weight);

    /**
     * @brief Get specific weight value
     *
     * @param net Network to query
     * @param layer_index Layer index (0 to no_of_layers-2)
     * @param neuron_lower Lower layer neuron index
     * @param neuron_upper Upper layer neuron index
     * @return float Current weight value
     */
    float net_get_weight(const network_t *net, int layer_index, int neuron_lower,
                         int neuron_upper);

    /**
     * @brief Enable or disable bias neurons
     *
     * @param net Network to configure
     * @param use_bias Non-zero to enable bias, zero to disable
     */
    void net_use_bias(network_t *net, int use_bias);

    /**
     * @brief Set bias weight for specific neuron
     *
     * @param net Network to modify
     * @param layer_index Layer index (1 to no_of_layers-1)
     * @param neuron_index Neuron index within layer
     * @param weight New bias weight value
     */
    void net_set_bias(network_t *net, int layer_index, int neuron_index, float weight);

    /**
     * @brief Get bias weight for specific neuron
     *
     * @param net Network to query
     * @param layer_index Layer index (1 to no_of_layers-1)
     * @param neuron_index Neuron index within layer
     * @return float Current bias weight value
     */
    float net_get_bias(const network_t *net, int layer_index, int neuron_index);

    /* Text File I/O Operations */

    /**
     * @brief Print network to file in text format
     *
     * @param file File pointer for output
     * @param net Network to print
     * @return int 0 on success, error code on failure
     */
    int net_fprint(FILE *file, const network_t *net);

    /**
     * @brief Load network from file in text format
     *
     * @param file File pointer for input
     * @return network_t* Loaded network, NULL on failure
     */
    network_t *net_fscan(FILE *file);

    /**
     * @brief Print network to stdout in text format
     *
     * @param net Network to print
     * @return int 0 on success, error code on failure
     */
    int net_print(const network_t *net);

    /**
     * @brief Save network to file in text format
     *
     * @param filename Name of output file
     * @param net Network to save
     * @return int 0 on success, error code on failure
     */
    int net_save(const char *filename, const network_t *net);

    /**
     * @brief Load network from file in text format
     *
     * @param filename Name of input file
     * @return network_t* Loaded network, NULL on failure
     */
    network_t *net_load(const char *filename);

    /* Binary File I/O Operations */

    /**
     * @brief Print network to file in binary format
     *
     * @param file File pointer for output
     * @param net Network to print
     * @return int 0 on success, error code on failure
     */
    int net_fbprint(FILE *file, const network_t *net);

    /**
     * @brief Load network from file in binary format
     *
     * @param file File pointer for input
     * @return network_t* Loaded network, NULL on failure
     */
    network_t *net_fbscan(FILE *file);

    /**
     * @brief Save network to file in binary format
     *
     * @param filename Name of output file
     * @param net Network to save
     * @return int 0 on success, error code on failure
     */
    int net_bsave(const char *filename, const network_t *net);

    /**
     * @brief Load network from file in binary format
     *
     * @param filename Name of input file
     * @return network_t* Loaded network, NULL on failure
     */
    network_t *net_bload(const char *filename);

    /* Network Computation Operations */

    /**
     * @brief Compute network output for given input
     *
     * @param net Network to use for computation
     * @param input Input vector
     * @param output Output vector (can be NULL if not needed)
     */
    void net_compute(network_t *net, const float *input, float *output);

    /**
     * @brief Compute output error and return global error
     *
     * @param net Network to evaluate
     * @param target Target output vector
     * @return float Global error value
     */
    float net_compute_output_error(network_t *net, const float *target);

    /**
     * @brief Get global error from last computation
     *
     * @param net Network to query
     * @return float Global error value
     */
    float net_get_output_error(const network_t *net);

    /**
     * @brief Perform one training iteration (backpropagation)
     *
     * @param net Network to train
     */
    void net_train(network_t *net);

    /* Batch Training Operations */

    /**
     * @brief Initialize batch training session
     *
     * @param net Network to prepare for batch training
     */
    void net_begin_batch(network_t *net);

    /**
     * @brief Accumulate training deltas for one pattern
     *
     * @param net Network being trained
     */
    void net_train_batch(network_t *net);

    /**
     * @brief Finalize batch training and update weights
     *
     * @param net Network being trained
     */
    void net_end_batch(network_t *net);

    /* Network Modification Operations */

    /**
     * @brief Randomly perturb network weights
     *
     * @param net Network to modify
     * @param factor Perturbation factor for large weights
     * @param range Range for reinitializing small weights
     */
    void net_jolt(network_t *net, float factor, float range);

    /**
     * @brief Add neurons to specified layer
     *
     * @param net Network to modify
     * @param layer_index Layer index to modify
     * @param neuron_index Insertion position (-1 for end)
     * @param number Number of neurons to add
     * @param range Weight range for new connections
     */
    void net_add_neurons(network_t *net, int layer_index, int neuron_index,
                         int number, float range);

    /**
     * @brief Remove neurons from specified layer
     *
     * @param net Network to modify
     * @param layer_index Layer index to modify
     * @param neuron_index Starting position for removal
     * @param number Number of neurons to remove
     */
    void net_remove_neurons(network_t *net, int layer_index, int neuron_index,
                            int number);

    /**
     * @brief Create a deep copy of network
     *
     * @param net Network to copy
     * @return network_t* New network copy, NULL on failure
     */
    network_t *net_copy(const network_t *net);

    /**
     * @brief Overwrite destination network with source network
     *
     * @param dest Destination network (will be modified)
     * @param src Source network (will be preserved)
     */
    void net_overwrite(network_t *dest, const network_t *src);

    /* Utility Functions */

    /**
     * @brief Validate network structure and parameters
     *
     * @param net Network to validate
     * @return int 1 if valid, 0 if invalid
     */
    int net_validate(const network_t *net);

    /**
     * @brief Get library version information
     *
     * @return const char* Version string
     */
    const char *net_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NERVENET_H */