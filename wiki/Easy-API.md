# Easy API Reference

The Easy API wraps the [[Core API]] in a higher-level interface optimized for minimal boilerplate. Every `nerve_*` function or macro delegates to the corresponding `net_*` function — there is no overhead.

`nerve_t` is a typedef alias for `network_t`; the two types are interchangeable.

---

## nerve_config_t

Configuration struct passed to `nerve_new_ex`. All fields have sensible defaults available via `nerve_default_config()`.

```c
typedef struct {
    int   optimizer;   /* NERVENET_OPTIMIZER_*    default: ADAM  */
    int   activation;  /* NERVENET_ACTIVATION_*   default: TANH  */
    float lr;          /* learning rate; 0 = auto (0.01)         */
    float l2;          /* L2 lambda;     0 = disabled            */
    float dropout;     /* dropout rate;  0 = disabled            */
    int   seed;        /* RNG seed; -1 = seeded from time()      */
} nerve_config_t;
```

---

## nerve_default_config

```c
nerve_config_t nerve_default_config(void);
```

Returns a `nerve_config_t` with the default settings:

| Field | Default |
|-------|---------|
| `optimizer` | `NERVENET_OPTIMIZER_ADAM` |
| `activation` | `NERVENET_ACTIVATION_TANH` |
| `lr` | 0.01 |
| `l2` | 0.0 (disabled) |
| `dropout` | 0.0 (disabled) |
| `seed` | -1 (auto from `time()`) |

---

## nerve_new

```c
nerve_t *nerve_new(const char *topology);
```

Create a network from a topology string using default configuration.

**Topology string format:** layer sizes separated by `->`. Any number of layers ≥ 2.

```c
nerve_t *net = nerve_new("2->4->1");          /* 2-4-1 MLP     */
nerve_t *net = nerve_new("784->256->128->10"); /* 4-layer       */
nerve_t *net = nerve_new("4 12 8 3");          /* spaces work   */
```

Defaults applied automatically:
- Optimizer: **Adam**
- Activation: **Tanh**
- Init: **Xavier** (or He if ReLU/Leaky ReLU is selected)
- Learning rate: **0.01**
- RNG seed: `time(NULL)`

Returns `NULL` if the topology string is invalid or if fewer than 2 layers are specified.

---

## nerve_new_ex

```c
nerve_t *nerve_new_ex(const char *topology, const nerve_config_t *cfg);
```

Same as `nerve_new` with explicit configuration. Pass `NULL` for `cfg` to use defaults.

```c
nerve_config_t cfg = nerve_default_config();
cfg.optimizer  = NERVENET_OPTIMIZER_ADAM;
cfg.activation = NERVENET_ACTIVATION_RELU;
cfg.lr         = 0.005f;
cfg.l2         = 1e-4f;
cfg.dropout    = 0.2f;
cfg.seed       = 42;

nerve_t *net = nerve_new_ex("4->64->64->3", &cfg);
```

When `activation` is `RELU` or `LEAKY_RELU`, He initialization is selected automatically. Xavier is used otherwise.

---

## nerve_fit

```c
void nerve_fit(nerve_t *net,
               const float *X, const float *y,
               int n_samples, int epochs);
```

Train for `epochs` full passes over the dataset. No output is printed.

- `X` — flat array of size `n_samples × n_inputs`
- `y` — flat array of size `n_samples × n_outputs`
- `n_inputs` and `n_outputs` are inferred from the network topology

Each epoch shuffles the dataset and calls `net_train_epoch` internally.

```c
/* XOR — 4 samples, 2 inputs, 1 output */
float X[] = {1,1,  1,0,  0,1,  0,0};
float y[] = {0,    1,    1,    0   };
nerve_fit(net, X, y, 4, 5000);
```

---

## nerve_fit_verbose

```c
void nerve_fit_verbose(nerve_t *net,
                       const float *X, const float *y,
                       int n_samples, int epochs,
                       int print_every);
```

Same as `nerve_fit` but prints the MSE every `print_every` epochs to `stdout`. Pass `print_every = 0` for silent training.

```c
nerve_fit_verbose(net, X, y, 120, 1000, 100);
```

```
  epoch   100 / 1000   MSE: 0.045321
  epoch   200 / 1000   MSE: 0.021047
  ...
  epoch  1000 / 1000   MSE: 0.002183
```

---

## nerve_score

```c
float nerve_score(nerve_t *net,
                  const float *X, const float *y,
                  int n_samples);
```

Compute classification accuracy over a dataset. `y` must be **one-hot encoded**. Returns a fraction in [0, 1].

```c
float acc = nerve_score(net, X_test, y_test, 30);
printf("Test accuracy: %.1f%%\n", acc * 100.f);
```

---

## Macro aliases

These macros delegate directly to the corresponding `net_*` functions. They exist solely to provide a consistent `nerve_` prefix for code that uses only the Easy API.

| Macro | Expands to |
|-------|-----------|
| `nerve_predict(net, x, out)` | `net_compute(net, x, out)` |
| `nerve_classify(net, x)` | `net_classify(net, x)` |
| `nerve_save(net, path)` | `net_save(path, net)` *(note: argument order flipped)* |
| `nerve_load(path)` | `net_load(path)` |
| `nerve_free(net)` | `net_free(net)` |

> `nerve_save` reverses the argument order from `net_save` so that the network comes first, matching the other `nerve_*` conventions.

---

## Complete Example — Iris Classification

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"
#include <stdio.h>

/* Iris data: 6 training samples (truncated for brevity) */
static float X_train[] = {
    5.1f,3.5f,1.4f,0.2f,   /* setosa     */
    4.9f,3.0f,1.4f,0.2f,
    6.3f,3.3f,4.7f,1.6f,   /* versicolor */
    5.8f,2.7f,4.1f,1.0f,
    6.3f,3.3f,6.0f,2.5f,   /* virginica  */
    5.8f,2.7f,5.1f,1.9f
};
static float y_train[] = {
    1,0,0,  1,0,0,
    0,1,0,  0,1,0,
    0,0,1,  0,0,1
};

int main(void)
{
    nerve_config_t cfg = nerve_default_config();
    cfg.l2   = 1e-3f;
    cfg.seed = 42;

    nerve_t *net = nerve_new_ex("4->12->8->3", &cfg);
    nerve_fit_verbose(net, X_train, y_train, 6, 2000, 500);

    float acc = nerve_score(net, X_train, y_train, 6);
    printf("Train accuracy: %.1f%%\n", acc * 100.f);

    /* classify one sample */
    float sample[] = {5.1f, 3.5f, 1.4f, 0.2f};
    int cls = nerve_classify(net, sample);
    printf("Predicted class: %d (0=setosa, 1=versicolor, 2=virginica)\n", cls);

    nerve_save(net, "iris.net");
    nerve_free(net);
    return 0;
}
```
