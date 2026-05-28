## Nerve 1.0

Zero-dependency multilayer perceptron in a single C header.  
Copy `nerve.h`, define `NERVE_IMPLEMENTATION`, compile with `-lm`. Done.

```c
nerve_t *net = nerve_new("2->8->1");
nerve_fit(net, X, y, n_samples, 5000);
float acc = nerve_score(net, X, y, n_samples);
nerve_free(net);
```

---

### Highlights

- **Single header** — the entire library is `nerve.h` (~1 250 lines). No build system, no package manager, no CMake required for the common case.
- **Zero dependencies** — only `math.h`. Compiles on Linux, macOS, Windows, and bare-metal.
- **Adam optimizer** — converges up to **18.7× faster** than SGD on benchmark tasks.
- **Easy API** — `nerve_new("2->8->1")` parses topology strings; `nerve_fit` trains in one call.
- **Terminal AI games** — Snake, Pong, and Flappy Bird AIs that learn live in the terminal via neuroevolution.
- **CI** — Linux (GCC), macOS (Clang), Windows (MinGW) on every push.

---

### Algorithms

| Component | Implementation | Reference |
|-----------|---------------|-----------|
| Optimizer | SGD + momentum | Rumelhart et al., 1986 |
| Optimizer | **Adam** | Kingma & Ba, 2015 |
| Init | Xavier / Glorot | Glorot & Bengio, 2010 |
| Init | He | He et al., 2015 |
| Activation | Sigmoid · Tanh · ReLU · Leaky ReLU | — |
| Regularisation | L2 weight decay | — |
| Regularisation | Dropout (inverted) | Srivastava et al., 2014 |

---

### Benchmark

4-class identity problem (4-6-4 network, 7 runs):

| Configuration | Avg iterations | Speedup |
|---------------|---------------:|--------:|
| SGD / Uniform / Sigmoid | 22 689 | 1× |
| **Adam / He / ReLU** | **1 214** | **18.7×** |

---

### Examples

| | File | Result |
|---|------|--------|
| 01 | `01_xor.c` | XOR — 2 000 iterations (Adam) |
| 02 | `02_sine.c` | sin(x), MSE < 0.00002 |
| 03 | `03_iris.c` | Iris classification **96.7%** |
| 05 | `05_regression.c` | Auto MPG regression RMSE **3.63 mpg** |
| 06 | `06_dropout.c` | Dropout +7 pp generalisation |
| 07 | `07_spiral.c` | 3-class spiral **98.3%** |
| 08 | `08_model_io.c` | Save / load / fine-tune |
| 09 | `09_predictive_maintenance.c` | Live sensor monitor 100% |
| 10 | `10_snake_ai.c` | Snake AI via neuroevolution |
| 11 | `11_pong_ai.c` | Pong: net vs rule-based bot |
| 12 | `12_flappy_ai.c` | Flappy Bird: 20 birds evolving |

---

### Installation

```bash
# Option 1 — copy the header (recommended)
cp nerve.h /your/project/
```

```cmake
# Option 2 — CMake FetchContent
include(FetchContent)
FetchContent_Declare(nerve
    GIT_REPOSITORY https://github.com/fkkarakurt/nerve.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(nerve)
target_link_libraries(myapp PRIVATE nerve::nerve)
```

---

### Documentation

- **[Wiki](https://github.com/fkkarakurt/nerve/wiki)** — Getting Started, Core API, Easy API, Algorithms
- **Scientific manual** — full mathematical derivations in `docs/latex/nerve_manual.tex` (pdflatex + bibtex)

---

### License

Copyright (C) 2026 Fatih Küçükkarakurt — GNU General Public License v3.0
