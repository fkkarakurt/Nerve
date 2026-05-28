![Social](https://raw.githubusercontent.com/fkkarakurt/Nerve/main/social.png)

<div align="center">

# Nerve

**Zero-dependency neural network library in pure ANSI C**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/Language-ANSI%20C-orange.svg)]()
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)]()
[![Single Header](https://img.shields.io/badge/Single%20Header-nerve.h-purple.svg)]()

*Multilayer perceptron · Backpropagation · Adam optimizer · Xavier / He init · Dropout · No dependencies*

</div>

---

## Why Nerve?

Most neural network libraries are either massive frameworks or toy implementations.
Nerve sits in the middle: it is small enough to read in an afternoon, correct enough to use in production, and fast enough to train real classifiers.

- **One file.** Drop `nerve.h` into your project and you are done.
- **Zero dependencies.** Compiles with any C89/C90 compiler. Just needs `math.h`.
- **Modern algorithms.** Adam optimizer, Xavier / He initialisation, Dropout, L2 regularisation — not just vanilla SGD.
- **Proven results.** Iris 96 %+, sin(x) MSE < 0.00002, XOR < 200 iterations with Adam.

---

## Quick Start — 15 lines

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"
#include <stdlib.h>
#include <time.h>

int main(void) {
    float in[]  = {1,1, 1,0, 0,1, 0,0};
    float tgt[] = {0,   1,   1,   0  };
    network_t *net = net_allocate(3, 2, 4, 1);   /* 2-4-1 MLP */
    net_set_optimizer(net, NERVENET_OPTIMIZER_ADAM);
    net_initialize_xavier(net);
    srand(42);
    for (int i = 0; i < 5000; i++) {
        int j = i % 4;
        net_compute(net, in + j*2, NULL);
        net_compute_output_error(net, tgt + j);
        net_train(net);
    }
    net_free(net);
    return 0;
}
```

```
gcc -O2 xor.c -o xor -lm
```

No CMake. No vcpkg. No `apt install`. Just gcc and `nerve.h`.

---

## Benchmark — SGD vs Adam

Results on an XOR problem (2-4-1 network, 7 independent runs):

| Configuration           | Conv  | Avg Iters | Speedup |
|-------------------------|-------|----------:|--------:|
| SGD / Uniform / Sigmoid | 7 / 7 |    22 285 | 1×      |
| SGD / Xavier  / Sigmoid | 7 / 7 |    23 900 | —       |
| SGD / Xavier  / Tanh    | 7 / 7 |    13 647 | 1.6×    |
| Adam / Xavier / Sigmoid | 7 / 7 |     2 601 | **8.6×**  |
| Adam / Xavier / Tanh    | 7 / 7 |     2 183 | **10.2×** |
| **Adam / He / ReLU**    | 5 / 7 |     2 064 | **10.8×** |

4-Class Identity problem (4-6-4 network):

| Configuration          | Conv  | Avg Iters | Speedup  |
|------------------------|-------|----------:|---------:|
| SGD / Uniform / Sigmoid | 7 / 7 |   22 689 | 1×       |
| Adam / He / ReLU       | 7 / 7 |    1 214 | **18.7×** |

---

## Examples

Nine progressive examples are included — each a standalone `.c` file that compiles with a single `gcc` command.

| # | File | Level | Topic |
|---|------|-------|-------|
| 01 | `01_xor.c` | Beginner | XOR — the Hello World of neural nets |
| 02 | `02_sine.c` | Beginner | Universal approximation with ASCII plot |
| 03 | `03_iris.c` | Beginner | Multi-class classification + confusion matrix |
| 04 | `04_ctypes_demo.py` | Beginner | Python binding via ctypes |
| 05 | `05_regression.c` | Intermediate | Continuous regression on Auto MPG dataset |
| 06 | `06_dropout.c` | Intermediate | Dropout vs memorisation of label noise |
| 07 | `07_spiral.c` | Advanced | Non-linear spiral with ASCII decision boundary |
| 08 | `08_model_io.c` | Advanced | Checkpoint save / load / fine-tune workflow |
| 09 | `09_predictive_maintenance.c` | Real-World | Industrial sensor monitoring with live dashboard |

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

### Example 02 — sin(x) Approximation

```
$ gcc -O2 examples/02_sine.c -o sine -lm && ./sine

Nerve 2.0.0 — sin(x) Approximation
Architecture: 1-16-16-1 | Adam + Tanh | Xavier Init

  sin(x) approximation — Nerve 2.0.0
  Legend:  | truth    * network    # overlap    - zero
  +1 ----------------------------------------------------------------
     |                                          ##########            |
     |                                        ###        ##|          |
     |                                      *##            *##        |
     ...
  -1 |                                                                |
     +----------------------------------------------------------------+
       -pi                                                             +pi

  Best MSE: 0.000019
```

---

### Example 03 — Iris Classification

```
$ gcc -O2 examples/03_iris.c -o iris -lm && ./iris

Nerve 2.0.0 — Iris Classification
Architecture: 4-12-8-3 | Adam + Tanh + L2 | Xavier Init
Dataset: 120 train / 30 test (Fisher 1936)

  Epoch  400 | MSE: 0.0209 | Test Acc: 96.7%
  ...
  Final test accuracy: 96.7%

  Confusion Matrix:
                       setosa  versicolor  virginica
  setosa                    8           0          0
  versicolor                0          13          0
  virginica                 0           0          9
```

---

### Example 05 — Fuel Efficiency Regression

Predicts miles-per-gallon from engine specs. Demonstrates continuous-output regression on the UCI Auto MPG dataset.

```
$ gcc -O2 examples/05_regression.c -o regression -lm && ./regression

Nerve 2.0.0 -- Fuel Efficiency Regression
Architecture: 4-8-8-1 | Adam + Tanh | Xavier Init
Dataset: UCI Auto MPG subset (32 train / 8 test)

Training...
  Epoch  200 | MSE: 0.0019
  ...
  Epoch 2000 | MSE: 0.0017

Predictions on test set:
  Cyl  Disp   HP   Weight  |  Actual  Predicted   Error
  ---------------------------------------------------------
    8   302  140   3449   |   17.0      15.8     -1.2
    8   390  190   3850   |   15.0      15.0     +0.0
    4   113   95   2372   |   24.0      26.2     +2.2
    6   199   90   2648   |   24.0      20.9     -3.1
    4    97   46   1835   |   26.0      34.4     +8.4
    4    97   88   2130   |   28.0      29.2     +1.2
    4    85   70   1990   |   36.0      31.8     -4.2
    6   225  105   3121   |   18.0      18.0     -0.0

  Test RMSE: 3.63 mpg
```

---

### Example 06 — Dropout vs Memorisation

Trains two identical networks on 30 correct + 10 mislabelled examples. Without dropout, the network memorises the corrupted labels; with dropout it learns the true boundary.

```
$ gcc -O2 examples/06_dropout.c -o dropout -lm && ./dropout

Nerve 2.0.0 -- Dropout vs No Dropout
Task: binary classification — circular decision boundary
Network: 2-128-128-2 (17154 params) | Adam + ReLU | He Init
Training: 30 correct + 10 mislabelled  |  Test: 400 clean

  Epoch |  No Dropout             |  Dropout p=0.5
        |  Train Acc  Test Acc    |  Train Acc  Test Acc
  ---------------------------------------------------------
   1000 |   100.0%      72.5%    |    90.0%      78.3%
   2000 |   100.0%      74.3%    |    95.0%      75.5%
   3000 |   100.0%      70.3%    |    97.5%      77.5%
   4000 |   100.0%      69.8%    |    97.5%      78.5%
   5000 |   100.0%      71.0%    |    97.5%      78.0%

  No-dropout test accuracy : 71.0%
  Dropout    test accuracy : 78.0%
  Dropout improved generalisation by 7.0 pp
```

---

### Example 07 — 3-Class Spiral Classification

Canonical non-linear benchmark that no linear classifier can solve. Renders an ASCII decision boundary after training.

```
$ gcc -O2 examples/07_spiral.c -o spiral -lm && ./spiral

Nerve 2.0.0 -- 3-Class Spiral Classification
Architecture: 2-32-32-3 | Adam + ReLU | He Init
Dataset: 180 train + 60 test  (3 classes x 60/20 points)

Training...
  Epoch  500 | MSE: 0.0151 | Test Acc: 98.3%
  Epoch 1000 | MSE: 0.0153 | Test Acc: 98.3%
  Epoch 1500 | MSE: 0.0125 | Test Acc: 98.3%
  Epoch 2000 | MSE: 0.0302 | Test Acc: 96.7%
  Epoch 2500 | MSE: 0.0136 | Test Acc: 98.3%

Final Test Accuracy: 98.3%  (59 / 60)

Decision Boundary (. = Class 0  X = Class 1  O = Class 2):
  +--------------------------------------------------------------+
  |XXXXXXXXXXXX..................................................|
  |XXXXXXXXXXXXXXXXXXXXX.........................................|
  |XXXXXXXXXXXXXXXXXXXXXXXXXXX...................................|
  |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX........................|
  |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX...................|
  |XXXXXXXXXXXXXXXXXXXXXXXXXOOOOOOOOOOXXXXXXXXXXX................|
  |XXXXXXXXXXXXXXXXXXOOOOOOOOOOOOOOOOOOOOXXXXXXXXXX..............|
  |XXXXXXXXXXXXXXXOOOOOOOOOOOOOOOOOOOOOOOOOXXXXXXXXXX............|
  |XXXXXXXXXXXXOOOOOOOOOO...........OOOOOOOOXXXXXXXXXX...........|
  |XXXXXXXXXXXOOOOOOOO................OOOOOOOOXXXXXXXX...........|
  |XXXXXXXXXXOOOOOOO........XXXX.......OOOOOOXXXXXXXXX...........|
  |XXXXXXXXXXOOOOOOO......XXXXXXOOOOOOOOOOOOXXXXXXXXXX...........|
  |XXXXXXXXXXOOOOOOOO.....XXXXXXOOOOOOOOOOOXXXXXXXXX.............|
  |XXXXXXXXXXXOOOOOOO......XXXXXXXXXXXXXXXXXXXXXX................|
  |XXXXXXXXXXXXOOOOOOOO.......XXXXXXXXXXXXXXX...........OOOOOOOOO|
  |XXXXXXXXXXXXOOOOOOOOOO..........XXXXXX.........OOOOOOOOOOOOOOO|
  |XXXXXXXXOOOOOOOOOOOOOOOOOO.................OOOOOOOOOOOOOOOOOOO|
  |XXOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO|
  |OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO|
  |OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO|
  |OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO|
  |OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO|
  +--------------------------------------------------------------+
```

---

### Example 08 — Model Save / Load / Fine-Tune

Demonstrates the checkpoint workflow: train partially, save to disk, load from disk, verify identical results, fine-tune to higher accuracy.

```
$ gcc -O2 examples/08_model_io.c -o model_io -lm && ./model_io

Nerve 2.0.0 -- Model Save / Load / Fine-Tune
Architecture: 4-12-8-3 | Adam + Tanh | Xavier Init
Dataset: Iris (30 train / 30 test)

Phase 1: early training (100 epochs, lr=0.0008)...
  Epoch  25 | MSE: 0.1752 | Test Acc: 73.3%
  Epoch  50 | MSE: 0.1409 | Test Acc: 96.7%
  Epoch  75 | MSE: 0.0927 | Test Acc: 96.7%
  Epoch 100 | MSE: 0.0506 | Test Acc: 96.7%

Checkpoint accuracy (early stop): 96.7%  (29 / 30)

Saving checkpoint to "iris_model.net"... done  (2.0 KB)
Loading  "iris_model.net"... done
Loaded model accuracy: 96.7%  (identical to checkpoint)

Phase 2: fine-tuning (600 epochs, lr=0.006)...
  Epoch 250 | MSE: 0.0006 | Test Acc: 100.0%
  ...

Final accuracy: 100.0%  (+3.3 pp from checkpoint)
```

---

### Example 09 — Predictive Maintenance ★

A complete real-world pipeline: classify rotating machinery health from four live sensor readings, then simulate a machine degrading from Normal to Fault.

**Architecture:** 4-16-8-4 | Adam + ReLU | He Init  
**Input:** Vibration (Hz), Temperature (°C), Pressure (bar), Current (A)  
**Output:** Normal / Warning / Critical / Fault

```
$ gcc -O2 examples/09_predictive_maintenance.c -o maintenance -lm && ./maintenance

Nerve 2.0.0 -- Predictive Maintenance
Dataset: 60 train / 20 test

Test accuracy: 100%  (20 / 20)

Confusion Matrix:
              Normal  Warning  Critical   Fault
  NORMAL           5          0          0          0
  WARNING          0          5          0          0
  CRITICAL         0          0          5          0
  FAULT            0          0          0          5

================================================================
  LIVE MONITORING SIMULATION
  Simulating machine degradation over 12 readings...
================================================================

  Reading  1
    Vibration     14.0 Hz   [==                      ]
    Temperature   58.0 C    [=======                 ]
    Pressure       4.5 bar  [=====                   ]
    Current        9.8 A    [======                  ]
    [  OK  ] NORMAL    Confidence: 99.9%

  Reading  5
    Vibration     42.0 Hz   [=====                   ]
    Temperature   84.0 C    [==========              ]
    Pressure       8.4 bar  [==========              ]
    Current       15.8 A    [=========               ]
    [ WARN ] WARNING   Confidence: 98.8%

  Reading  8
    Vibration     74.0 Hz   [=========               ]
    Temperature  109.0 C    [=============           ]
    Pressure      11.3 bar  [==============          ]
    Current       22.8 A    [==============          ]
    [ CRIT ] CRITICAL  Confidence: 97.3%

  Reading 11
    Vibration    118.0 Hz   [==============          ]
    Temperature  148.0 C    [==================      ]
    Pressure      15.8 bar  [===================     ]
    Current       33.2 A    [====================    ]
    [FAULT ] FAULT     Confidence: 99.5%

================================================================
  Simulation complete.  Machine degradation detected:
  Normal -> Warning -> Critical -> FAULT
================================================================
```

---

## Installation

### Option 1 — Single Header (recommended)

Copy `nerve.h` to your project. In exactly **one** `.c` file:

```c
#define NERVE_IMPLEMENTATION
#include "nerve.h"
```

In all other files, just `#include "nerve.h"`.

### Option 2 — Static Library

```bash
cd src && make          # produces libnerveNET.a
```

Link with `-L../src -lnerveNET -lm`.

---

## API Reference

### Create & Destroy

```c
/* Layers specified as variadic ints */
network_t *net = net_allocate(3, 64, 128, 10);
net_free(net);

/* Or from an array */
int layers[] = {64, 128, 10};
network_t *net = net_allocate_l(3, layers);
```

### Configure

```c
/* Activation (hidden layers; output is always sigmoid) */
net_set_activation(net, NERVENET_ACTIVATION_SIGMOID);    /* default */
net_set_activation(net, NERVENET_ACTIVATION_TANH);       /* often faster */
net_set_activation(net, NERVENET_ACTIVATION_RELU);       /* deep nets */
net_set_activation(net, NERVENET_ACTIVATION_LEAKY_RELU); /* avoids dying ReLU */

/* Optimizer */
net_set_optimizer(net, NERVENET_OPTIMIZER_SGD);   /* default — with momentum */
net_set_optimizer(net, NERVENET_OPTIMIZER_ADAM);  /* usually 5–20× faster */

/* Weight initialisation */
net_randomize(net, 1.0f);        /* Uniform [-1, +1] — default */
net_initialize_xavier(net);      /* Glorot 2010 — for sigmoid / tanh */
net_initialize_he(net);          /* He 2015    — for ReLU */

/* Hyper-parameters */
net_set_learning_rate(net, 0.01f);   /* Adam default: 0.001–0.01 */
net_set_momentum(net, 0.9f);         /* SGD momentum */
net_set_l2_lambda(net, 1e-4f);       /* L2 weight decay  (0 = off) */
net_set_dropout(net, 0.2f);          /* 20 % dropout during training (0 = off) */
```

### Train

```c
/* One-sample online training */
net_compute(net, input, NULL);
net_compute_output_error(net, target);
net_train(net);

/* Epoch helper — shuffles data, works with Adam and SGD */
float mse = net_train_epoch(net, inputs, targets,
                             n_samples, n_inputs, n_outputs,
                             batch_size);   /* ignored when using Adam */
```

### Inference & Metrics

```c
float output[10];
net_compute(net, input, output);              /* forward pass */

int label = net_classify(net, input);         /* argmax */

float acc = net_compute_accuracy(net, inputs, targets,
                                 n_samples, n_inputs, n_outputs);

int cm[10 * 10] = {0};
net_confusion_matrix(net, inputs, targets,
                     n_samples, n_inputs, n_outputs, 10, cm);
```

### Save & Load

```c
net_save(net, "model.net");          /* text — human-readable  */
network_t *net = net_load("model.net");

net_bsave(net, "model.bin");         /* binary — compact        */
network_t *net = net_bload("model.bin");
```

> **Note:** `net_save` stores weights, momentum, learning rate, and global error.
> Re-apply `net_set_activation()` and `net_set_optimizer()` after loading.

---

## Algorithm Reference

| Component | Options | Reference |
|-----------|---------|-----------|
| Optimizer | SGD + momentum | Rumelhart et al., Nature 1986 |
| Optimizer | **Adam** | Kingma & Ba, ICLR 2015 |
| Init | Xavier / Glorot | Glorot & Bengio, AISTATS 2010 |
| Init | **He** | He et al., ICCV 2015 |
| Activation | Sigmoid, Tanh, **ReLU**, Leaky ReLU | — |
| Regularisation | L2 weight decay | — |
| Regularisation | **Dropout** | Srivastava et al., JMLR 2014 |

---

## Python Binding

Nerve exposes a clean C ABI, making it straightforward to call from Python via `ctypes`:

```python
# Build shared library first:
# gcc -O2 -shared -fPIC -DNERVE_IMPLEMENTATION -o libnerve.so nerve.h -lm

import ctypes
lib = ctypes.CDLL("./libnerve.so")
# See examples/04_ctypes_demo.py for the full wrapper
```

Full example: [`examples/04_ctypes_demo.py`](examples/04_ctypes_demo.py)

---

## Project Structure

```
nerve.h                       ← single-header distribution (start here)
src/
  nerveNet.h                  ← public API header
  network.c                   ← implementation
  generate_sigma.c            ← pre-generates sigmoid lookup table
examples/
  01_xor.c                    ← XOR — 15-line quick start
  02_sine.c                   ← sin(x) approximation + ASCII plot
  03_iris.c                   ← Iris classification + confusion matrix
  04_ctypes_demo.py           ← Python binding via ctypes
  05_regression.c             ← Auto MPG regression (continuous output)
  06_dropout.c                ← Dropout vs memorisation of label noise
  07_spiral.c                 ← 3-class spiral + ASCII decision boundary
  08_model_io.c               ← Save / load / fine-tune checkpoint workflow
  09_predictive_maintenance.c ← Live industrial sensor monitor (real-world)
tests/
  benchmark.c                 ← SGD vs Adam convergence benchmark
  train_network.c             ← CLI training tool
ocr/                          ← handwritten digit recognition (requires zlib)
trainingData/                 ← Alpaydin & Kaynak 1995 digit dataset
```

---

## Building the Examples

```bash
cd examples

# Individual
gcc -O2 examples/01_xor.c  -o xor  -lm && ./xor
gcc -O2 examples/02_sine.c -o sine -lm && ./sine
gcc -O2 examples/03_iris.c -o iris -lm && ./iris
gcc -O2 examples/05_regression.c           -o regression  -lm && ./regression
gcc -O2 examples/06_dropout.c              -o dropout     -lm && ./dropout
gcc -O2 examples/07_spiral.c               -o spiral      -lm && ./spiral
gcc -O2 examples/08_model_io.c             -o model_io    -lm && ./model_io
gcc -O2 examples/09_predictive_maintenance.c -o maintenance -lm && ./maintenance

# All at once
make -C examples

# Benchmark
cd tests && gcc -O2 -I../src benchmark.c -o benchmark \
    ../src/libnerveNET.a -lm && ./benchmark
```

---

## Special Thanks

- [Ethem Alpaydın](https://www.cmpe.boun.edu.tr/~ethem/) — datasets, books, and inspiration
- [Cenk Kaynak](https://www.linkedin.com/in/cenk-kaynak-phd-631aa4101/) — datasets and articles
- [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) — benchmark datasets
- [Tom Mitchell — *Machine Learning*](http://www.cs.cmu.edu/~tom/mlbook.html) — foundational text
- [comp.ai.neural-nets FAQ](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/) — original reference

---

## License

Copyright © 2022 Fatih Küçükkarakurt

[MIT License](LICENSE) — do whatever you want, just keep the copyright notice.
