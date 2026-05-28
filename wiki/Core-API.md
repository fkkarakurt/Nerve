# Core API Reference

All functions use the `net_` prefix. Include `nerve.h` in every translation unit; define `NERVE_IMPLEMENTATION` in exactly one.

**Contents**
- [Enumerations](#enumerations)
- [Allocation](#allocation)
- [Weight Initialization](#weight-initialization)
- [Configuration](#configuration)
- [Query](#query)
- [Weight Access](#weight-access)
- [Forward Pass](#forward-pass)
- [Training](#training)
- [Metrics](#metrics)
- [Persistence](#persistence)
- [Structural Modification](#structural-modification)
- [Utility](#utility)

---

## Enumerations

### `nervenet_activation_t`

Hidden-layer activation function. The output layer always uses the logistic sigmoid.

| Constant | Formula | Notes |
|----------|---------|-------|
| `NERVENET_ACTIVATION_SIGMOID` | 1/(1+e^−x) | Default |
| `NERVENET_ACTIVATION_TANH` | tanh(x) | Often converges faster than sigmoid |
| `NERVENET_ACTIVATION_RELU` | max(0, x) | Pair with He initialization |
| `NERVENET_ACTIVATION_LEAKY_RELU` | max(0.01x, x) | Avoids dead neurons |

### `nervenet_optimizer_t`

| Constant | Algorithm | Notes |
|----------|-----------|-------|
| `NERVENET_OPTIMIZER_SGD` | SGD + momentum | Default; momentum=0.1 |
| `NERVENET_OPTIMIZER_ADAM` | Adam (Kingma & Ba, 2015) | Recommended; lr=0.001–0.01 |

### `nervenet_init_t`

| Constant | Distribution | Best for |
|----------|-------------|----------|
| `NERVENET_INIT_UNIFORM` | U[−1, +1] | Prototyping |
| `NERVENET_INIT_XAVIER` | Glorot 2010 | Sigmoid / Tanh |
| `NERVENET_INIT_HE` | He 2015 | ReLU / Leaky ReLU |

### `nervenet_error_t`

Return codes: `NERVENET_SUCCESS` (0), `NERVENET_ERROR_NULL_POINTER` (−1), `NERVENET_ERROR_INVALID_PARAM` (−2), `NERVENET_ERROR_MEMORY` (−3), `NERVENET_ERROR_IO` (−4), `NERVENET_ERROR_CORRUPT` (−5).

---

## Allocation

### `net_allocate`

```c
network_t *net_allocate(int no_of_layers, ...);
```

Allocate a network; layer sizes passed as variadic `int` arguments.

```c
/* 3-layer network: 64 inputs → 128 hidden → 10 outputs */
network_t *net = net_allocate(3, 64, 128, 10);
```

Returns a pointer to the newly allocated network, or `NULL` on allocation failure. The network is initialized with uniform random weights in [−1, +1] and SGD with momentum.

### `net_allocate_l`

```c
network_t *net_allocate_l(int no_of_layers, const int *arglist);
```

Same as `net_allocate` but reads layer sizes from an array.

```c
int sizes[] = {64, 128, 10};
network_t *net = net_allocate_l(3, sizes);
```

### `net_free`

```c
void net_free(network_t *net);
```

Release all memory owned by the network, including Adam moment arrays if Adam was active.

---

## Weight Initialization

Call one of these after `net_allocate` and before training.

### `net_randomize`

```c
void net_randomize(network_t *net, float range);
```

Assign weights uniformly at random from [−`range`, +`range`].

### `net_initialize_xavier`

```c
void net_initialize_xavier(network_t *net);
```

Xavier / Glorot uniform initialization (Glorot & Bengio, 2010). Recommended for **Sigmoid** and **Tanh** activations.

Each weight is drawn from U[−√(6/(n_in+n_out)), +√(6/(n_in+n_out))].

### `net_initialize_he`

```c
void net_initialize_he(network_t *net);
```

He uniform initialization (He et al., 2015). Recommended for **ReLU** and **Leaky ReLU** activations.

Each weight is drawn from U[−√(6/n_in), +√(6/n_in)].

### `net_reset_deltas`

```c
void net_reset_deltas(network_t *net);
```

Zero all weight deltas (momentum accumulators). Called automatically by `net_begin_batch`.

---

## Configuration

### `net_set_activation`

```c
void net_set_activation(network_t *net, nervenet_activation_t act);
```

Set the hidden-layer activation function. Must be called before training. The output layer always uses sigmoid regardless of this setting.

### `net_set_optimizer`

```c
void net_set_optimizer(network_t *net, nervenet_optimizer_t opt);
```

Switch between SGD and Adam. Switching to Adam allocates the first- and second-moment arrays and resets the time step. Switching back to SGD frees them.

### `net_set_learning_rate`

```c
void net_set_learning_rate(network_t *net, float lr);
```

Set the learning rate η. Typical values: 0.001–0.01 for Adam; 0.1–0.5 for SGD.

### `net_set_momentum`

```c
void net_set_momentum(network_t *net, float momentum);
```

Set the SGD momentum coefficient μ (default 0.1). Has no effect when Adam is active.

### `net_set_l2_lambda`

```c
void net_set_l2_lambda(network_t *net, float lambda);
```

Set the L2 weight-decay coefficient λ. Pass 0.0f to disable (default). Typical range: 1e-5 to 1e-3.

### `net_set_dropout`

```c
void net_set_dropout(network_t *net, float rate);
```

Set the dropout probability p for hidden neurons during training (0 = disabled, default). Nerve uses inverted dropout so inference requires no scaling.

### `net_use_bias`

```c
void net_use_bias(network_t *net, int flag);
```

Enable (flag ≠ 0) or disable (flag = 0) bias units. Bias is enabled by default.

---

## Query

### `net_get_learning_rate` / `net_get_momentum`

```c
float net_get_learning_rate(const network_t *net);
float net_get_momentum(const network_t *net);
```

### `net_get_no_of_inputs` / `net_get_no_of_outputs` / `net_get_no_of_layers`

```c
int net_get_no_of_inputs(const network_t *net);
int net_get_no_of_outputs(const network_t *net);
int net_get_no_of_layers(const network_t *net);
```

### `net_get_no_of_weights`

```c
int net_get_no_of_weights(const network_t *net);
```

Returns the total number of trainable parameters (weights + biases).

---

## Weight Access

### `net_get_weight` / `net_set_weight`

```c
float net_get_weight(const network_t *net, int layer, int from_neuron, int to_neuron);
void  net_set_weight(network_t *net,       int layer, int from_neuron, int to_neuron, float w);
```

`layer` is the index of the **lower** layer (0 = input). Returns/sets the weight connecting `from_neuron` in layer `layer` to `to_neuron` in layer `layer+1`.

### `net_get_bias` / `net_set_bias`

```c
float net_get_bias(const network_t *net, int layer, int neuron);
void  net_set_bias(network_t *net,       int layer, int neuron, float w);
```

Bias is implemented as the weight from the extra bias unit in `layer-1` to `neuron` in `layer`.

---

## Forward Pass

### `net_compute`

```c
void net_compute(network_t *net, const float *input, float *output);
```

Run a forward pass. If `output` is non-`NULL`, the output activations are copied there. Always call this before `net_compute_output_error`.

```c
float out[10];
net_compute(net, input_vector, out);
```

### `net_compute_output_error`

```c
float net_compute_output_error(network_t *net, const float *target);
```

Compute the MSE output error ½ Σ(t_k − a_k)² and store local gradients for backpropagation. Must be called after `net_compute`. Returns the scalar error.

### `net_get_output_error`

```c
float net_get_output_error(const network_t *net);
```

Return the output error computed by the most recent call to `net_compute_output_error`.

---

## Training

### `net_train` — online (one sample)

```c
void net_train(network_t *net);
```

Perform one backpropagation + weight-update step. Call `net_compute` and `net_compute_output_error` first.

```c
net_compute(net, input, NULL);
net_compute_output_error(net, target);
net_train(net);
```

### `net_train_epoch` — shuffled epoch helper

```c
float net_train_epoch(network_t *net,
                      const float *inputs, const float *targets,
                      int n_pairs, int n_inputs, int n_outputs,
                      int batch_size);
```

Train over a full epoch with Fisher-Yates shuffling. Returns the mean MSE over all samples.

- **Adam mode**: ignores `batch_size`; one Adam step per sample (stochastic).
- **SGD mode**: accumulates gradients over `batch_size` samples then updates (mini-batch).

### Batch training (SGD only)

```c
void net_begin_batch(network_t *net);    /* zero deltas */
void net_train_batch(network_t *net);   /* accumulate (call after compute+error) */
void net_end_batch(network_t *net);     /* apply averaged update */
```

Note: batch training always uses SGD. Adam operates online.

---

## Metrics

### `net_classify`

```c
int net_classify(network_t *net, const float *input);
```

Run a forward pass and return the index of the output neuron with the highest activation (argmax). Suitable for one-hot classification.

### `net_compute_accuracy`

```c
float net_compute_accuracy(network_t *net,
                           const float *inputs, const float *targets,
                           int n_pairs, int n_inputs, int n_outputs);
```

Compute classification accuracy over a dataset stored as flat arrays. `targets` must be one-hot encoded. Returns a fraction in [0, 1].

```c
float acc = net_compute_accuracy(net, X_test, y_test, 30, 4, 3);
printf("Test accuracy: %.1f%%\n", acc * 100.f);
```

### `net_confusion_matrix`

```c
void net_confusion_matrix(network_t *net,
                          const float *inputs, const float *targets,
                          int n_pairs, int n_inputs, int n_outputs,
                          int n_classes, int *matrix);
```

Fill a pre-zeroed `n_classes × n_classes` confusion matrix. `matrix[true_class * n_classes + predicted_class]` is incremented for each sample.

```c
int cm[3*3] = {0};
net_confusion_matrix(net, X, y, n, 4, 3, 3, cm);
```

---

## Persistence

> **Important:** `net_save` and `net_bsave` store weights, momentum, learning rate, and global error — but **not** the activation type or optimizer. After loading, call `net_set_activation()` and `net_set_optimizer()` to restore these settings.

### Text format (`.net`)

```c
int        net_save(const char *filename, const network_t *net);
network_t *net_load(const char *filename);
int        net_fprint(FILE *file, const network_t *net);
network_t *net_fscan(FILE *file);
int        net_print(const network_t *net);   /* → stdout */
```

Human-readable; portable across platforms. Each weight is written as an ASCII decimal float.

### Binary format (`.bin`)

```c
int        net_bsave(const char *filename, const network_t *net);
network_t *net_bload(const char *filename);
int        net_fbprint(FILE *file, const network_t *net);
network_t *net_fbscan(FILE *file);
```

Compact IEEE 754 32-bit float representation. Approximately 4× smaller than text. May not be portable across platforms with different endianness.

```c
/* Save */
net_save(net, "model.net");
net_bsave(net, "model.bin");

/* Load + restore settings */
network_t *net = net_load("model.net");
net_set_activation(net, NERVENET_ACTIVATION_TANH);
net_set_optimizer(net,  NERVENET_OPTIMIZER_ADAM);
```

---

## Structural Modification

### `net_jolt`

```c
void net_jolt(network_t *net, float factor, float range);
```

Randomly perturb weights. Weights with |w| < `range` are replaced by U[−range, +range]. Larger weights are multiplied by U[1−factor, 1+factor]. Useful for escaping local minima in neuroevolution.

### `net_copy`

```c
network_t *net_copy(const network_t *net);
```

Deep copy of a network including all weights, Adam state, and configuration. Caller must free with `net_free`.

### `net_overwrite`

```c
void net_overwrite(network_t *dest, const network_t *src);
```

Replace `dest` with a copy of `src`. The original contents of `dest` are freed.

### `net_add_neurons`

```c
void net_add_neurons(network_t *net, int layer, int neuron, int number, float range);
```

Insert `number` neurons before position `neuron` in `layer`. Pass `neuron = -1` to append. New connections are initialized with weights from U[−range, +range].

### `net_remove_neurons`

```c
void net_remove_neurons(network_t *net, int layer, int neuron, int number);
```

Remove `number` neurons starting at position `neuron` in `layer`.

---

## Utility

### `net_validate`

```c
int net_validate(const network_t *net);
```

Returns 1 if the network is internally consistent (no null pointers, valid layer counts, Adam arrays present if Adam active), 0 otherwise.

### `net_get_version`

```c
const char *net_get_version(void);
```

Returns the library version string, e.g. `"2.0.0"`.
