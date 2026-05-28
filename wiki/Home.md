# Nerve — Neural Network Library

**Zero-dependency multilayer perceptron in a single C header.**

Copy `nerve.h` into your project. Define `NERVE_IMPLEMENTATION` in one translation unit. Compile with `-lm`. That is the entire integration.

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"

nerve_t *net = nerve_new("2->8->1");
nerve_fit(net, X, y, n_samples, 5000);
printf("accuracy: %.1f%%\n", nerve_score(net, X, y, n_samples) * 100.f);
nerve_free(net);
```

```
gcc -O2 main.c -o main -lm
```

---

## Features

| | |
|---|---|
| **Optimizers** | SGD + momentum · Adam (Kingma & Ba, 2015) |
| **Activations** | Sigmoid · Tanh · ReLU · Leaky ReLU |
| **Init** | Uniform · Xavier/Glorot (2010) · He (2015) |
| **Regularisation** | L2 weight decay · Inverted Dropout |
| **Training** | Online · Mini-batch · Shuffled epoch |
| **Persistence** | Text (`.net`) · Binary (`.bin`) |
| **Metrics** | Accuracy · Confusion matrix · MSE |
| **API levels** | Easy API (`nerve_*`) · Core API (`net_*`) |

---

## Pages

| Page | Contents |
|------|----------|
| [[Getting Started]] | Installation, first program, compilation |
| [[Core API]] | All `net_*` functions — full reference |
| [[Easy API]] | `nerve_*` convenience layer |
| [[Algorithms]] | Activation functions, optimizers, initializers, regularization |

---

## Quick benchmark

Adam vs SGD on a 4-class identity problem (4-6-4 network, 7 runs):

| Optimizer | Avg iterations | Speedup |
|-----------|---------------:|--------:|
| SGD / Uniform / Sigmoid | 22 689 | 1× |
| **Adam / He / ReLU** | **1 214** | **18.7×** |

---

## License

Copyright (C) 2026 Fatih Küçükkarakurt — [GPL-3.0](https://github.com/fkkarakurt/nerve/blob/main/LICENSE)
