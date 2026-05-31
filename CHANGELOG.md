# Changelog

All notable changes to **Nerve** are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning is
[Semantic Versioning](https://semver.org/).

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

## [Unreleased] — in progress

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

### Roadmap Phase 2 spike (training prototype — not shipped in the core yet)
- `studies/autograd/nerve_grad.h` — a minimal, single-header, ~200-line
  reverse-mode **autodiff** engine over 2D tensors (`t_matmul`, `t_add_bias`,
  `t_relu`, `t_softmax_cross_entropy`). The spine that will make convolutions,
  attention and training *expressible* instead of hand-coded.
- `demo_mlp.c` — trains an MLP to 12/12 with **zero hand-written gradients**.
- `gradcheck.c` — analytic gradients vs central finite differences agree to the
  float floor (~1e-4): **the autodiff is proven correct**.

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
