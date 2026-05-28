# Nerve — Neural Network Library

## Project Overview

**Nerve** is a lightweight, zero-external-dependency multi-layer perceptron
library written in ANSI C89/C90.  The primary value proposition is that it
embeds directly into any C or C++ project with a single `#include`.

Current version: **2.0.0**

## Repository Layout

```
nerve.h               Single-header distribution (start here; no deps)
src/
  nerveNet.h          Public API header (include this in your project)
  network.c           Core implementation
  generate_sigma.c    Generates interpolation.c (sigmoid lookup table)
  Makefile            Builds libnerveNET.a

examples/             All use nerve.h; compile with: gcc file.c -o out -lm
  01_xor.c            XOR — 15-line quick start
  02_sine.c           sin(x) approximation + ASCII plot
  03_iris.c           Iris classification + confusion matrix
  04_ctypes_demo.py   Python binding via ctypes
  05_regression.c     Auto MPG regression (RMSE ~3.6 mpg)
  06_dropout.c        Dropout vs memorisation (mislabelled data)
  07_spiral.c         3-class spiral + ASCII decision boundary
  08_model_io.c       Checkpoint save / load / fine-tune
  09_predictive_maintenance.c  HERO: live industrial sensor monitor
  Makefile            Builds all examples

tests/
  benchmark.c         Convergence benchmark: SGD vs Adam (18.7× speedup)
  train_network.c     CLI tool: train a network from a .spec file
  create_network.c    CLI tool: create and save an empty network
  show_network.c      CLI tool: print network weights
  copy_network.c      CLI tool: copy a saved network
  jolt_network.c      CLI tool: randomly perturb a network
  spec.c / spec.h     .spec file reader used by train_network / show_network
  *.spec              Boolean function training specs (XOR, AND, Parity…)
  Makefile

ocr/
  ocr_train.c         Train a digit-recognition network (requires zlib)
  ocr_validate.c      Validate a digit-recognition network
  lib/                OCR helper library (preprocessing, file I/O via zlib)
  Makefile

trainingData/
  optdigits.*         Handwritten digit datasets (Alpaydin & Kaynak, 1995)
                      Custom 32×32 bitmap format, gzip-compressed
```

## Build

### Prerequisites
- GCC (or any ANSI C-compatible compiler)
- GNU Make
- `zlib` — only for the `ocr/` module; the core library has **no dependencies**

### Commands
```bash
make          # build everything
make test     # build + run tests (xornet convergence check)
make -C tests bench   # run the full convergence benchmark
make clean    # remove object files
make clean-all
```

### Build just the library
```bash
cd src && make
# produces: src/libnerveNET.a
```

## Core API (nerveNet.h)

### 1. Create / Destroy
```c
/* 3-layer network: 64 inputs → 32 hidden → 10 outputs */
network_t *net = net_allocate(3, 64, 32, 10);
net_free(net);
```

### 2. Configure (call before training)
```c
net_set_activation(net, NERVENET_ACTIVATION_TANH);      /* hidden layers */
net_set_optimizer(net, NERVENET_OPTIMIZER_ADAM);
net_initialize_xavier(net);                              /* weight init   */
net_set_learning_rate(net, 0.005f);
net_set_l2_lambda(net, 1e-4f);                          /* regularisation */
```

### 3. Train (one sample at a time)
```c
net_compute(net, input_vector, NULL);
net_compute_output_error(net, target_vector);
net_train(net);
```

### 4. Inference
```c
float output[10];
net_compute(net, input_vector, output);

int cls = net_classify(net, input_vector);   /* argmax */

float acc = net_compute_accuracy(net, all_inputs, all_targets,
                                 n_samples, n_inputs, n_outputs);
```

### 5. Save / Load
```c
net_save(net, "mynet.net");          /* text format  */
network_t *net = net_load("mynet.net");

net_bsave(net, "mynet.bin");         /* binary format */
network_t *net = net_bload("mynet.bin");
```

## Algorithm Notes

### Activation Functions
| Enum | Formula | Best for |
|------|---------|----------|
| `NERVENET_ACTIVATION_SIGMOID` | 1/(1+e^−x) | Default; output layer always sigmoid |
| `NERVENET_ACTIVATION_TANH` | tanh(x) | Hidden layers; often faster convergence than sigmoid |
| `NERVENET_ACTIVATION_RELU` | max(0,x) | Deep networks; use He init |
| `NERVENET_ACTIVATION_LEAKY_RELU` | max(0.01x,x) | Avoids dying ReLU units |

The **output layer always uses sigmoid** regardless of the `activation` setting.
This keeps the MSE error signal backward-compatible.

### Weight Initialisation
| Enum | Formula | Reference |
|------|---------|-----------|
| `NERVENET_INIT_UNIFORM` | U[−1, +1] | Classic |
| `NERVENET_INIT_XAVIER` | U[−√(6/(in+out)), +√(6/(in+out))] | Glorot & Bengio 2010 |
| `NERVENET_INIT_HE` | U[−√(6/in), +√(6/in)] | He et al. 2015 |

### Optimizers
| Enum | Update rule | Reference |
|------|------------|-----------|
| `NERVENET_OPTIMIZER_SGD` | w += lr·δ + mom·Δw | Rumelhart et al. 1986 |
| `NERVENET_OPTIMIZER_ADAM` | Adaptive moments with bias correction | Kingma & Ba 2014 |

Adam default hyper-parameters: β₁=0.9, β₂=0.999, ε=10⁻⁸.
Typical learning rate for Adam: 0.001–0.01.

### L2 Regularisation
Applied to the weight gradient: ∇w ← ∇w − λ·w.  
Default λ=0 (disabled).  Typical values: 1e-5 to 1e-3.

## .spec File Format

Used by `train_network` and `show_network`:

```
# number of inputs
2
# number of outputs
1
# input lines followed by target lines (pairs)
1.0 1.0
0.0
1.0 0.0
1.0
```

Comments start with `#`.

## GÜNCELLENMESİ GEREKENLER

1. Projenin tüm kodları ve projeye dair her şey GNU GPLv3 lisansı ile lisanslansın MIT değil. 
2. Örneklere göre , örneklerin sonuçlarını biz mi oluşturuyoruz yoksa yapay zeka kendisi mi buluyor, yani DL
   çalışıyor mu anlamadım? 
3. İnsanları daha da heyecanlandırmak için terminal üzerinde çalışan birkaç oyun yapabiliriz belki. Basit bir flappy bird klonu, araba sürme oyunu, snake, ping-pong oyunu veya senin
   önerdiğin başka şeylerde olabilir. 
4. Makefile yerine, CMake tercih edelim. 
5. Github yıldızları kazanmak ve trend olmak için gereken her şeyi yapalım. 
6. Nerve API bence çok kolay değil. Bu API çok kolay ve anlaşılır olmalı.
7. Tüm örneklerin çıktıları README.md dosyasına eklenmeli.
8. Dökümantasyon, örnekler ve API çok ileri seviye, detaylı ve akademik bir dil ile Tezmiş gibi yazılmalı. Kodlardaki tüm matematiksel yapılmak istenenler, LaTeX formatta ve çok daha ayrıntılı bir şekilde, bir dökümantasyon generator ile /docs klasörüne oluşturulmalı.
9. Kullanılmayan eğitim verileri silinmeli ve örneklerin çıktıları gitignore dosyasına eklenmeli. Asla Github'a push edilmemeli.

## Known Issues / Limitations

- Batch training (`net_train_batch`) always uses SGD; Adam is online-only.
- The OCR module requires `zlib` (not part of the core library).
- `net_save` / `net_load` (text) and `net_bsave` / `net_bload` (binary) save
  weights, momentum, learning_rate, and global_error but NOT the activation or
  optimizer settings. After loading, always call `net_set_activation()` and
  `net_set_optimizer()` to restore the correct hyper-parameters.

## References

1. Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986).
   *Learning representations by back-propagating errors.* Nature 323, 533–536.
2. Glorot, X. & Bengio, Y. (2010).
   *Understanding the difficulty of training deep feedforward neural networks.*
   AISTATS.
3. He, K. et al. (2015).
   *Delving deep into rectifiers.* ICCV.
4. Kingma, D.P. & Ba, J. (2014).
   *Adam: A method for stochastic optimization.* ICLR 2015.
5. Alpaydin, E. & Kaynak, C. (1995).
   *Optical recognition of handwritten digits dataset.*
   UCI Machine Learning Repository.
