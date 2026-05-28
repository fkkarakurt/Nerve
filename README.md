<div align="center">

# Nerve

**Zero-dependency neural network library in pure C**

[![CI](https://github.com/fkkarakurt/nerve/actions/workflows/ci.yml/badge.svg)](https://github.com/fkkarakurt/nerve/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Language-C99-orange.svg)]()
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)]()
[![Single Header](https://img.shields.io/badge/Single%20Header-nerve.h-purple.svg)]()

*Multilayer perceptron · Backpropagation · Adam · Xavier / He init · Dropout · No dependencies*

</div>

---

## Why Nerve?

Most neural network libraries are either massive frameworks or toy implementations.
Nerve sits in between: small enough to read in an afternoon, correct enough for real work,
and portable enough to run anywhere C compiles.

- **One file.** Copy `nerve.h` into your project. That's the entire library.
- **Zero dependencies.** Any ANSI C compiler. Just `math.h`.
- **Modern algorithms.** Adam, Xavier/He init, Dropout, L2 — not just vanilla SGD.
- **Proven results.** Iris 96.7%, sin(x) MSE < 0.00002, XOR in < 200 iterations with Adam.

---

## Quick Start

### The 3-line API

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"

nerve_t *net = nerve_new("2->4->1");        /* topology string */
nerve_fit(net, X, y, 4, 5000);             /* train           */
nerve_free(net);
```

```
gcc -O2 main.c -o main -lm
```

No CMake. No vcpkg. No `apt install`. Just `gcc` and `nerve.h`.

### The full-control API

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"

network_t *net = net_allocate(3, 2, 4, 1);
net_set_optimizer(net, NERVENET_OPTIMIZER_ADAM);
net_initialize_xavier(net);
net_set_learning_rate(net, 0.01f);

for (int i = 0; i < 5000; i++) {
    int j = i % 4;
    net_compute(net, in + j*2, NULL);
    net_compute_output_error(net, tgt + j);
    net_train(net);
}
net_free(net);
```

---

## Benchmark — SGD vs Adam

Results on boolean functions (7 independent runs each):

**XOR (2-4-1 network)**

| Configuration           | Converged | Avg Iters | Speedup |
|-------------------------|:---------:|----------:|--------:|
| SGD / Uniform / Sigmoid | 7 / 7     |    22 285 | 1×      |
| SGD / Xavier  / Tanh    | 7 / 7     |    13 647 | 1.6×    |
| Adam / Xavier / Sigmoid | 7 / 7     |     2 601 | **8.6×**  |
| Adam / Xavier / Tanh    | 7 / 7     |     2 183 | **10.2×** |
| Adam / He / ReLU        | 5 / 7     |     2 064 | **10.8×** |

**4-Class Identity (4-6-4 network)**

| Configuration           | Converged | Avg Iters | Speedup   |
|-------------------------|:---------:|----------:|----------:|
| SGD / Uniform / Sigmoid | 7 / 7     |    22 689 | 1×        |
| Adam / He / ReLU        | 7 / 7     |     1 214 | **18.7×** |

---

## Examples

Eleven standalone `.c` files, each compiling with a single `gcc` command.

| # | File | Topic |
|---|------|-------|
| 01 | `01_xor.c` | XOR — the Hello World of neural nets |
| 02 | `02_sine.c` | Universal approximation with ASCII plot |
| 03 | `03_iris.c` | Multi-class classification + confusion matrix |
| 05 | `05_regression.c` | Continuous regression on Auto MPG dataset |
| 06 | `06_dropout.c` | Dropout vs memorisation of label noise |
| 07 | `07_spiral.c` | Non-linear spiral + ASCII decision boundary |
| 08 | `08_model_io.c` | Checkpoint save / load / fine-tune workflow |
| 09 | `09_predictive_maintenance.c` | Industrial sensor monitor |
| 10 | `10_snake_ai.c` | **Snake AI** via neuroevolution |
| 11 | `11_pong_ai.c` | **Pong AI** — neural net vs rule-based bot |
| 12 | `12_flappy_ai.c` | **Flappy Bird AI** — 20 birds evolving simultaneously |

---

### Example 01 — XOR

```
$ gcc -O2 examples/01_xor.c -o xor -lm && ./xor

Nerve 2.0.0 — XOR Example
Architecture: 2-4-1 | Adam | Xavier Init

  Epoch  1000 | MSE: 0.000043
  ...

Results after 1847 epochs:
  Input         Target        Output
  ------        ------        ------
  [1, 1]        0.0           0.0031  OK
  [1, 0]        1.0           0.9968  OK
  [0, 1]        1.0           0.9967  OK
  [0, 0]        0.0           0.0028  OK
```

---

### Example 03 — Iris Classification

```
$ gcc -O2 examples/03_iris.c -o iris -lm && ./iris

Nerve 2.0.0 — Iris Classification
Architecture: 4-12-8-3 | Adam + Tanh + L2 | Xavier Init

  Epoch  400 | MSE: 0.0209 | Test Acc: 96.7%

  Final test accuracy: 96.7%

  Confusion Matrix:
                       setosa  versicolor  virginica
  setosa                    8           0          0
  versicolor                0          13          0
  virginica                 0           0          9
```

---

### Example 07 — 3-Class Spiral

```
$ gcc -O2 examples/07_spiral.c -o spiral -lm && ./spiral

Nerve 2.0.0 -- 3-Class Spiral Classification
Architecture: 2-32-32-3 | Adam + ReLU | He Init

  Final Test Accuracy: 98.3%  (59 / 60)

  +--------------------------------------------------------------+
  |XXXXXXXXXXXX..................................................|
  |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX........................|
  |XXXXXXXXXXXXXXXXXXXXXXXXXOOOOOOOOOOXXXXXXXXXXX................|
  |XXXXXXXXXXOOOOOOO........XXXX.......OOOOOOXXXXXXXXX...........|
  |OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO|
  +--------------------------------------------------------------+
```

---

### Example 10-12 — Terminal AI Games

Neural networks learning to play games in real-time, displayed in your terminal.
Uses neuroevolution (population-based training, no backprop).

```
$ gcc -O2 examples/10_snake_ai.c -o snake_ai -lm && ./snake_ai

  Nerve Snake AI  generation 42

  +---------------------------------------------------------+
  |                                 <>                      |   Score:   8
  |            @@##########                                 |   Steps:   341
  |                                                         |   Best:    12
  +---------------------------------------------------------+

  Net: 11->16->3  |  Pop: 100  |  Elite: 5
```

Build all games at once:

```bash
make -C examples games
# or individually:
gcc -O2 examples/10_snake_ai.c  -o snake_ai  -lm
gcc -O2 examples/11_pong_ai.c   -o pong_ai   -lm
gcc -O2 examples/12_flappy_ai.c -o flappy_ai -lm
```

---

## API Reference

### Easy API

```c
/* Create from topology string */
nerve_t *net = nerve_new("2->8->1");            /* Adam + Tanh + Xavier defaults */

/* With custom config */
nerve_config_t cfg = nerve_default_config();
cfg.lr       = 0.005f;
cfg.dropout  = 0.2f;
nerve_t *net = nerve_new_ex("4->32->3", &cfg);

/* Train */
nerve_fit(net, X, y, n_samples, epochs);
nerve_fit_verbose(net, X, y, n_samples, epochs, /*print_every=*/100);

/* Evaluate */
float acc = nerve_score(net, X, y, n_samples);   /* classification accuracy */
nerve_predict(net, x, output);                   /* single forward pass     */
int   cls = nerve_classify(net, x);              /* argmax                  */

/* Save / Load */
nerve_save(net, "model.net");
nerve_t *loaded = nerve_load("model.net");

nerve_free(net);
```

### Core API

```c
/* Allocate */
network_t *net = net_allocate(3, 64, 128, 10);    /* variadic layer sizes */
int sizes[] = {64, 128, 10};
network_t *net = net_allocate_l(3, sizes);         /* from array           */
net_free(net);

/* Configure */
net_set_activation(net, NERVENET_ACTIVATION_TANH);
net_set_optimizer(net,  NERVENET_OPTIMIZER_ADAM);
net_initialize_xavier(net);
net_set_learning_rate(net, 0.01f);
net_set_momentum(net,  0.9f);
net_set_l2_lambda(net, 1e-4f);
net_set_dropout(net,   0.2f);

/* Train — one sample */
net_compute(net, input, NULL);
net_compute_output_error(net, target);
net_train(net);

/* Train — shuffled epoch */
float mse = net_train_epoch(net, inputs, targets,
                             n_samples, n_inputs, n_outputs, batch_size);

/* Inference */
float out[10];
net_compute(net, input, out);
int   label = net_classify(net, input);
float acc   = net_compute_accuracy(net, inputs, targets, n, n_in, n_out);

int cm[9] = {0};
net_confusion_matrix(net, inputs, targets, n, n_in, n_out, 3, cm);

/* Persist */
net_save(net, "model.net");       network_t *net = net_load("model.net");
net_bsave(net, "model.bin");      network_t *net = net_bload("model.bin");
```

> **After loading**, re-apply `net_set_activation()` and `net_set_optimizer()` —
> the serialized format stores weights but not optimizer/activation settings.

---

## Algorithm Reference

| Component | Options | Reference |
|-----------|---------|-----------|
| Optimizer | SGD + momentum | Rumelhart et al., 1986 |
| Optimizer | **Adam** | Kingma & Ba, 2015 |
| Init | Xavier / Glorot | Glorot & Bengio, 2010 |
| Init | **He** | He et al., 2015 |
| Activation | Sigmoid, Tanh, **ReLU**, Leaky ReLU | — |
| Regularisation | L2 weight decay | — |
| Regularisation | **Dropout** (inverted) | Srivastava et al., 2014 |

---

## Documentation

Generate HTML + LaTeX reference docs (requires [Doxygen](https://www.doxygen.nl/)):

```bash
doxygen docs/Doxyfile
# HTML → open docs/doxygen/html/index.html
```

A formal scientific manual with full mathematical derivations and benchmarks is
available as LaTeX source:

```bash
cd docs/latex
pdflatex nerve_manual.tex
bibtex   nerve_manual
pdflatex nerve_manual.tex
pdflatex nerve_manual.tex
# → nerve_manual.pdf
```

---

## Building

```bash
# All examples
make -C examples

# Terminal AI games only
make -C examples games

# CMake (Linux / macOS / Windows)
cmake -B build
cmake --build build
```

---

## Project Structure

```
nerve.h           ← the entire library (single file)
examples/
  01_xor.c        ← XOR — 15-line quick start
  02_sine.c       ← sin(x) approximation + ASCII plot
  03_iris.c       ← Iris classification + confusion matrix
  05_regression.c ← Auto MPG regression (RMSE ≈ 3.6 mpg)
  06_dropout.c    ← Dropout vs memorisation of label noise
  07_spiral.c     ← 3-class spiral + ASCII decision boundary
  08_model_io.c   ← Checkpoint save / load / fine-tune
  09_predictive_maintenance.c  ← Live industrial sensor monitor
  10_snake_ai.c   ← Snake AI via neuroevolution
  11_pong_ai.c    ← Pong: neural net vs rule-based bot
  12_flappy_ai.c  ← Flappy Bird: 20 birds evolving in parallel
  Makefile
  CMakeLists.txt
docs/
  Doxyfile        ← Doxygen config (output → docs/doxygen/)
  mathematics.dox ← LaTeX formula reference page
  latex/
    nerve_manual.tex   ← Nerve 1.0 Scientific Manual
    nerve_refs.bib     ← bibliography
CMakeLists.txt
Makefile
LICENSE
```

---

## License

Copyright (C) 2022 Fatih Küçükkarakurt

Released under the [GNU General Public License v3.0](LICENSE).
