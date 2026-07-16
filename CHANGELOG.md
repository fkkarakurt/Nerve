# Changelog


---

## [Unreleased] — in progress

### Licence — relicensed from GPL-3.0 to Apache-2.0
- **Nerve is now Apache-2.0.** The previous GPL-3.0 terms made the library
  unusable for its actual purpose: an embeddable, single-header engine that
  developers drop into their own products. Apache-2.0 is permissive and carries
  an express patent grant, which is what commercial and embedded adopters
  require. Relicensing was clean — Nerve has always been single-author, with no
  outside contributions to re-license.
- Every source file now carries an SPDX `Apache-2.0` identifier. Eighteen files
  previously carried **no licence header at all** — including the inference
  engine (`nerve_infer.h`), the autodiff engine (`nerve_grad.h`), the sentence
  encoder (`nerve_embed.h`) and the WebAssembly entry points (`nerve_web.c`).
  Absent a licence grant those files were legally "all rights reserved"; they
  are now covered explicitly.
- Added `NOTICE`, stating that Nerve's source is independent and from-scratch,
  and that model weights are user data governed by their own licences.
- `npm` package (`@fkkarakurt/nerve`) relicensed to Apache-2.0 to match.

### Added — Nerve is reproducible on every platform
- **Nerve no longer calls libc's `rand()`.** It could not: `rand()` is
  implementation-defined and `RAND_MAX` is **32 767** on the Microsoft C runtime
  against **2 147 483 647** on glibc, so the same seed produced different
  weights, at different granularity, on Windows and Linux. The library advertised
  reproducibility while resting on the one facility in the C standard library
  that cannot provide it — and the Linux and Windows CI jobs were, in effect,
  testing two different libraries.
- `nerve.h` now carries **xoshiro128\*\*** (Blackman & Vigna, 2021, *ACM TOMS*
  47(4)), seeded through SplitMix32. State and arithmetic are 32-bit throughout,
  emulated in `unsigned long` with explicit masking, so the stream is identical
  on LP64 Unix and on Windows while the core stays ANSI C89 — which has neither
  `<stdint.h>` nor a guaranteed 64-bit type.
- New API: `nerve_seed()`, `nerve_rand_u32()`, `nerve_rand_float()`,
  `nerve_rand_below()`. Nerve is now **deterministic by default** — the state
  starts fixed and you opt into variation with `nerve_seed(time(NULL))`.
- `nerve_rand_below(n)` draws by rejection, correcting the modulo bias that
  `rand() % n` carried in the internal Fisher-Yates shuffle: 2^32 is not a
  multiple of most bounds, so low residues came up slightly more often.
- Examples now seed with `nerve_seed()`. The teaching examples (01-03) use a
  fixed seed so their printed output reproduces anywhere; `01_xor` accepts a seed
  argument. The terminal games still vary per run.

### Added — the first test suite
- **The project had no tests.** `tests/` was removed in `3b44e6b` and never
  replaced; CI asserted correctness by grepping an example's standard output.
- `tests/test_nerve.c` — one dependency-free translation unit, no framework,
  built by the same single `gcc` line as everything else. It:
  - checks the generator against an **exact-width reference implementation**
    over 120 000 draws across six seeds, which is what turns "identical on every
    platform" from a hope into a fact, and pins a **golden vector** so the stream
    cannot drift silently;
  - tests `nerve_rand_below` for bias with a chi-square over 700 000 draws;
  - **verifies the backprop gradients against central finite differences** — the
    central correctness property of the library, and one that had never been
    checked;
  - round-trips both persistence formats and covers `net_copy`, softmax
    normalisation, and Xavier/He reproducibility and bounds.
- Runnable three ways: `make test`, `ctest`, or one `gcc` invocation. Added a
  CTest target to CMake and `test` / `check-c89` targets to the Makefile.

### Fixed — CI was red, and red at random
- The smoke test compiled `examples/01_xor.c` — which seeded from `time(NULL)` —
  and counted `OK` in its output. A **CHANGELOG-only commit turned the build
  red**, because the assertion depended on a random draw from a platform-specific
  generator. Replaced with the real suite.
- CI now runs on Linux (GCC **and** Clang), macOS and Windows; under **ASan +
  UBSan**; and enforces the README's **ANSI C89** claim as a build failure rather
  than a promise. It also builds `studies/` and the WebAssembly target for the
  first time — the transformer, autodiff and encoder had never been compiled by
  CI at all.
- `-Werror` is on for the compiler that is reproducible locally (GCC/Linux);
  Clang and macOS run `-Wall -Wextra` until a green run proves their warning set
  is empty.

### Fixed — documentation and build defects
- **`net_save`/`net_bsave` were documented with their arguments reversed.** The
  README showed `net_save(net, "model.net")`; the real signature takes the
  filename first. Anyone copying the documented call got a compile error.
- `CMakeLists.txt` declared version 2.0.0 while `nerve.h` said 2.1.0.
- `studies/mnist/export_mlp.c` ignored every `malloc` and `fread` result and
  tripped `-Wmisleading-indentation`; rewritten to check them.
- Added `web/build.sh` so CI and Unix developers build the WebAssembly demo with
  the same flags as `build.ps1`, instead of the two drifting apart.

### Fixed — repository and packaging
- **The documentation site was silently deleted from version control.** A bare
  `web`, `site` and `*.html` rule in `.gitignore` removed `site/index.html` from
  the repository and ignored the entire WebAssembly demo *source* — the demo and
  docs pages survived only on one developer machine, and could never have been
  deployed from a clone. The generated artifacts those rules were meant to
  exclude (`nerve.js`, `nerve.wasm`, `nerve.data`, `*.nrv`, `*.tok`, `*.nre`)
  were already covered by precise rules; the broad ones only destroyed sources.
  Restored the docs site to `web/site/index.html` and removed the bad rules.

### Transformer INFERENCE — the "own your AI, no token tax" artifact
- `studies/infer/nerve_infer.h` — a single-header, zero-dependency
  **Transformer inference library**: token embeddings, RMSNorm, rotary position
  encoding (RoPE), causal multi-head attention (grouped/multi-query) with a KV
  cache, SwiGLU feed-forward, temperature / top-p sampling, and a BPE tokenizer.
  Implemented from the published math; exposed as a reusable
  `load / forward / generate / free` API you embed in your own program, not a
  standalone binary. ggml-class capability, stb-class integration.
- **Fully independent — Nerve's own formats, no third-party file layouts:**
  - `model.nrv` (magic `NRV1`): a self-describing, versioned 64-byte header
    (dims, flags, `rope_theta`) then float32 weights. Leaner than typical dumps
    — the precomputed RoPE tables are dropped (computed on the fly).
  - `nerve.tok` (magic `NTK1`): Nerve's own tokenizer/vocabulary format.
  - The runtime library reads ONLY these formats — it depends on no one else's.
  - `convert.c` — a clearly-labeled one-time tool to import externally trained
    weights into Nerve format ("bring your own model"; the engine and format are
    Nerve's, the weights are user data).
- `generate.c` — proof: a small (~15M-param) model writes complete, coherent
  stories at ~50 tokens/sec on a laptop CPU. No GPU, no Python, no cloud, no
  per-token bill.
- **Runs a real 1.1B-parameter LLM.** `convert_hf.py` (a build-time tool, pure
  numpy + stdlib — no torch, no safetensors lib) imports any Hugging Face
  Llama-architecture model into Nerve's int8 `.nrv` format, including the
  HF→interleaved RoPE un-permutation of the q/k projections and GQA. TinyLlama
  1.1B-Chat (2.2 GB bf16 → 1.05 GB int8) generates coherent, on-topic English in
  the engine **unchanged** at ~3 tokens/sec on a 2017 laptop CPU (i7-7700HQ),
  offline, zero runtime dependencies. This is the substrate Nerve learns on.

### Nerve LEARNS on-device — the flagship differentiator
- `learn.c` — proof that Nerve doesn't just *run* a model, it *adapts* to you.
  The frozen 1.1B base is used as a feature extractor (its final hidden state);
  a tiny head — a single linear layer trained with Nerve's own autodiff engine
  (`nerve_grad.h`) — learns the user's personal categories from a handful of
  their own sentences, entirely on-device. Reaches 6/6 on held-out test
  sentences after 300 epochs on 12 examples, in seconds, on a laptop CPU. No
  data leaves the machine, no cloud, no tokens, no GPU.
- This is the per-node capability the whole Nerve direction is built on:
  *owned, private, personalized AI that learns from you* — the axis the cloud
  giants structurally can't follow. (First rung: frozen-backbone / linear-probe
  personalization. Next: LoRA / adapters that change the generative model's
  behavior, then federated sharing of those adapters.)
- `search.c` — local **semantic search**: mean-pooled, mean-centered embeddings
  matched by cosine similarity. Queries worded completely differently from the
  notes still retrieve the right one (6/6 on the demo set) — meaning, not
  keywords. Private, offline "ask your own notes" with no embeddings API.
- Strategic direction: Nerve's flagship lane is private, on-device, owned
  inference of small models — the niche the big engines (llama.cpp/ggml) can't
  reach on embeddability, auditability and portability.

### Performance — pure portable C, no intrinsics
- Rewrote the core matvec with eight independent accumulators (+ `restrict`)
  so `-O3 -march=native` auto-vectorises it into SIMD FMAs on whatever ISA the
  target has — without writing a single intrinsic, staying single-header and
  fully portable. Measured ~**3x** end-to-end on the 15M model (≈67 → ≈230
  tokens/sec, single core, CPU).
- `restrict` is guarded by a macro so the core still builds as ANSI C89.
- Optional OpenMP multithreading behind `-fopenmp` (pragmas are ignored without
  it; zero mandatory dependency). Note: threading currently *slows* very small
  models (per-matmul fork/join overhead exceeds the work); it is opt-in and
  pays off as model size grows.
- **int8 quantization** (`quantize.c`): per-row symmetric int8 weights, a new
  native quantized `.nrv` variant, and an int8 matvec in the engine. Cuts the
  model ~**4x** (15M model: 58 MB → 14.7 MB) with output quality intact
  (embedding rows match float to ~4e-4). On tiny cache-resident models speed is
  roughly flat (the int8→float conversion offsets the bandwidth saving); the
  speed win appears as models grow past cache. The size cut is the real enabler:
  useful-sized models fit in the limited RAM of ordinary/old hardware.

### Runs in the browser (WebAssembly)
- `web/` — the same engine compiled to WebAssembly (`nerve.wasm` is ~65 KB),
  running a real model **entirely in the browser**: click a link, type, watch it
  generate. No server, no GPU, no data leaving the page. The small int8 stories
  model (~15 MB) is preloaded; the engine source is reused unchanged. Modular,
  ready to graduate into its own repo / GitHub Pages.

### Roadmap Phase 2 spike (training prototype — not shipped in the core yet)
- `studies/autograd/nerve_grad.h` — a minimal, single-header, ~200-line
  reverse-mode **autodiff** engine over 2D tensors (`t_matmul`, `t_add_bias`,
  `t_relu`, `t_softmax_cross_entropy`). The spine that will make convolutions,
  attention and training *expressible* instead of hand-coded.
- `demo_mlp.c` — trains an MLP to 12/12 with **zero hand-written gradients**.
- `gradcheck.c` — analytic gradients vs central finite differences agree to the
  float floor (~1e-4): **the autodiff is proven correct**.

---

## [2.1.0] — 2026-05-31

First step of the long-game roadmap: Nerve learns to *speak probability
properly*. Fully backward compatible — existing code keeps its sigmoid + MSE
behaviour unchanged.

### Added
- **Softmax output activation** (`NERVENET_ACTIVATION_SOFTMAX`), computed with
  max-subtraction for numerical stability.
- **Cross-entropy loss** (`nervenet_loss_t`: `NERVENET_LOSS_MSE`,
  `NERVENET_LOSS_CROSS_ENTROPY`).
- **Decoupled output activation** — the output layer is no longer hardcoded to
  sigmoid. New field `output_activation`.
- New API:
  - `net_set_output_activation(net, act)`
  - `net_set_loss(net, loss)`
  - `net_set_classification(net)` — one call for the standard softmax +
    cross-entropy classification setup.

### Changed
- Output-layer error now carries the correct activation derivative for MSE
  (reproduces the previous `y(1-y)e` exactly for sigmoid) and collapses to the
  clean `(t - y)` term for softmax/sigmoid + cross-entropy.
- Verbose training reports `loss` instead of `MSE` when cross-entropy is active.
- MNIST study (`studies/mnist`) retrained with softmax + cross-entropy.

### Notes
- `output_activation` and `loss` are not serialized (same convention as
  activation/optimizer): re-apply with the setters after `net_load`.

### Verified
- XOR sigmoid + MSE back-compat: pass.
- 3-class softmax + cross-entropy: calibrated probabilities (sum to 1).
- Builds clean under `-std=c89 -Wall -Wextra`.

---

## [2.0.0] — 2026-05-28

Major modernisation of the library.

### Added
- Adam optimizer; Xavier / He initialisation.
- Activations: sigmoid, tanh, ReLU, leaky-ReLU.
- L2 regularisation, dropout.
- `net_train_epoch`, confusion matrix, `net_classify`,
  `net_compute_accuracy`, `net_validate`, `net_get_version`.
- Single-header distribution (`nerve.h`, stb-style
  `#define NERVE_IMPLEMENTATION`).
- Examples 01–09 (XOR, sine, Iris, ctypes, regression, dropout, spiral,
  model I/O, predictive maintenance) and the MNIST / geotechnical studies.

### Fixed
- `net_train_epoch` with Adam now actually uses the Adam update path.

[2.1.0]: #210--2026-05-31
[2.0.0]: #200--2026-05-28
