# Getting Started

## Installation

Nerve is a single-header library. There is no build system to configure and no package to install.

1. Copy `nerve.h` from the repository into your project directory.
2. In **exactly one** `.c` or `.cpp` file, define `NERVE_IMPLEMENTATION` before the include:

```c
/* main.c — implementation compiled here */
#define NERVE_IMPLEMENTATION
#include "nerve.h"
```

3. In every other file that needs the API, include the header without the define:

```c
/* other.c — declarations only */
#include "nerve.h"
```

4. Compile and link with the math library:

```
gcc -O2 main.c -o program -lm
```

That is everything. No CMake. No vcpkg. No `apt install`.

---

## Your First Program — Easy API

The [[Easy API]] provides a high-level interface with sensible defaults (Adam optimizer, Tanh activation, Xavier initialization).

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"
#include <stdio.h>

int main(void)
{
    /* XOR truth table: inputs and targets as flat arrays */
    float X[] = {1,1,  1,0,  0,1,  0,0};
    float y[] = {0,    1,    1,    0   };

    nerve_t *net = nerve_new("2->4->1");   /* parse topology string */
    nerve_fit(net, X, y, 4, 5000);         /* 5 000 epochs          */

    float out;
    nerve_predict(net, X + 0*2, &out);  printf("[1,1] -> %.4f\n", out);
    nerve_predict(net, X + 1*2, &out);  printf("[1,0] -> %.4f\n", out);
    nerve_predict(net, X + 2*2, &out);  printf("[0,1] -> %.4f\n", out);
    nerve_predict(net, X + 3*2, &out);  printf("[0,0] -> %.4f\n", out);

    nerve_free(net);
    return 0;
}
```

```
gcc -O2 xor.c -o xor -lm && ./xor
[1,1] -> 0.0194
[1,0] -> 0.9828
[0,1] -> 0.9920
[0,0] -> 0.0080
```

---

## Your First Program — Core API

The [[Core API]] gives you full control over every hyper-parameter.

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    float inputs[4][2] = {{1,1},{1,0},{0,1},{0,0}};
    float targets[4]   = {0, 1, 1, 0};
    float output;
    int   i, epoch;
    float mse;

    srand((unsigned)time(NULL));

    network_t *net = net_allocate(3, 2, 4, 1);   /* 2-4-1 MLP */
    net_set_optimizer(net,  NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net, NERVENET_ACTIVATION_TANH);
    net_initialize_xavier(net);
    net_set_learning_rate(net, 0.01f);

    for (epoch = 1; epoch <= 10000; epoch++) {
        mse = 0.0f;
        for (i = 0; i < 4; i++) {
            net_compute(net, inputs[i], NULL);
            mse += net_compute_output_error(net, &targets[i]);
            net_train(net);
        }
        mse /= 4.0f;
        if (mse < 1e-4f) break;
    }

    printf("Converged in %d epochs (MSE=%.6f)\n", epoch, mse);
    for (i = 0; i < 4; i++) {
        net_compute(net, inputs[i], &output);
        printf("[%.0f,%.0f] -> %.4f\n", inputs[i][0], inputs[i][1], output);
    }

    net_free(net);
    return 0;
}
```

---

## Recommended Configurations

| Task | Architecture hint | Optimizer | Activation | Init |
|------|------------------|-----------|------------|------|
| Classification (shallow) | input→32→32→classes | Adam | Tanh | Xavier |
| Classification (deep) | input→64→64→64→classes | Adam | ReLU | He |
| Regression | input→16→16→1 | Adam | Tanh | Xavier |
| Embedded / tiny | input→8→output | SGD | Sigmoid | Xavier |

**Rule of thumb:** start with `Adam + Tanh + Xavier`, switch to `Adam + ReLU + He` if you add more than two hidden layers.

---

## Building the Included Examples

```bash
# All core examples
make -C examples

# Terminal AI games (Snake, Pong, Flappy Bird)
make -C examples games

# Single file
gcc -O2 examples/01_xor.c -o xor -lm
```

CMake:

```bash
cmake -B build
cmake --build build
```
