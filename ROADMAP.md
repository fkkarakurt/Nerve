# Nerve — Roadmap

> *"Hey, I'm here too."*
>
> A neural network you can read end-to-end, that depends on nothing, runs
> everywhere, and proves — at a size that fits in your head — that it belongs
> in the same conversation as the giants.

This document is the long-game plan for Nerve: how a single-header, zero-
dependency C neural network becomes **small but genuinely powerful**, stays
**usable and alive**, fills real gaps the giants ignore, and earns respect not
by competing on scale but by doing things people didn't think a library this
small could do.

---

## 1. The thesis

The AI world is optimizing one axis almost exclusively: **scale**. Bigger
models, bigger clusters, bigger bills. That race is real, but it leaves a wide
gap behind it — and that gap is Nerve's home:

- Frameworks are **enormous and opaque**. PyTorch is millions of lines; nobody
  reads it. Nerve is one header you can understand in an afternoon.
- Real AI needs **dependencies, GPUs, Python, the cloud**. Nerve needs a C
  compiler. It runs on a laptop, a browser tab, a microcontroller, a 20-year-old
  machine, a satellite.
- "AI" has become **someone else's API**. Nerve is *yours* — every weight update
  is plain code you own.

Nerve does not win by becoming a small GPT. It wins by being **the most
understandable, most portable, most embeddable real neural network in
existence**, and by proving — with shippable artifacts — that this lane is not a
toy lane. Power through elegance, not through size.

---

## 2. Invariants — the rules that keep Nerve *Nerve* as it grows

Growth is dangerous: most projects become powerful by becoming bloated, and that
bloat is exactly what would bury Nerve on a dusty shelf. These constraints are
load-bearing. Every new feature must respect them.

1. **Zero mandatory dependencies.** The core builds with `cc file.c -lm`.
   Acceleration (SIMD, threads, GPU) is always *optional*, behind a flag, never
   required to run.
2. **Readable over clever.** If a contributor can't follow a function in one
   sitting, it's wrong. The code is documentation.
3. **It must run on a laptop.** Every feature ships with a demo that trains or
   runs in minutes on a normal CPU. Nothing requires a datacenter.
4. **Portable C first.** Nerve targets anything with a C compiler — desktop,
   WASM, embedded, bare metal. Platform-specific code is additive, never the
   trunk.
5. **Every capability ships with proof.** No feature is "done" until there is a
   runnable example that demonstrates it doing something real.

Smallness is not a stage Nerve is trying to outgrow. It is the **product**.

---

## 3. The gap Nerve fills

These are real, underserved needs the giants structurally cannot address. Each
is a place Nerve can be genuinely *useful*, not just impressive:

- **Education that's real, not hand-wavy.** Students read the actual backprop,
  not a slide. Nerve is how a generation can learn how a neural net truly works.
- **Private, on-device inference.** Ship trained weights; data never leaves the
  device. Health, biometrics, keystrokes — inferred locally, no server, no leak.
- **Edge / embedded / TinyML.** A trained model on a `$3` microcontroller with no
  OS. Gesture recognition, anomaly detection, wake-words — offline, instant.
- **Reproducible research.** A model defined in one auditable file, with no
  dependency drift, that will still compile and produce identical results in ten
  years.
- **Zero-infra deployment.** Inference compiled into the app itself — browser,
  CLI, game, firmware — with no Python runtime and no cloud bill.

---

## 4. The roadmap

Each phase pairs a **technical addition** with a **proof point** — the concrete,
runnable artifact that says "hey, I'm here." Phases are ordered so each unlocks
the next.

### Phase 0 — Foundation *(done)*
The core MLP, SGD + momentum, Adam, regularization, model I/O, and — crucially —
evidence it learns *real* things: it recovered slope-stability physics from data
(`studies/geotechnical`), reads handwritten digits at **96.97 %** on MNIST with a
plain MLP (`studies/mnist`), and exhibits live behavior in the game agents
(`examples/10–12`). The bet is already paying off at small scale.

### Phase 1 — Speak probability properly
**Add:** softmax output + cross-entropy loss; decouple the output activation from
the hardcoded sigmoid.
**Why:** the current output is always sigmoid + MSE — fine for the demos, wrong
for serious multi-class and a hard blocker for language. This is the smallest
change with the largest downstream payoff.
**Proof point:** calibrated multi-class probabilities; MNIST retrained with
proper cross-entropy.

### Phase 2 — Become a framework, stay a library
**Add:** reverse-mode automatic differentiation over a minimal n-dimensional
tensor type, with a small, cache-friendly `matmul` (GEMM) at the core.
**Why:** today every layer's gradient is hand-derived. Autodiff is the gateway —
with it, convolutions, attention, and normalization become *expressible* instead
of *hand-coded*. This is the pivot from "an MLP" to "a tiny framework."
**Identity guard:** micrograd (~100 lines) and tinygrad prove a readable autodiff
engine is possible. Keep it that small. This is the phase most likely to bloat —
guard it hardest.
**Proof point:** define a brand-new layer type without writing a single gradient
by hand.

### Phase 3 — Train deep
**Add:** residual connections + LayerNorm (and optionally BatchNorm).
**Why:** without these, networks past ~10 layers simply don't train (vanishing/
exploding gradients). They are cheap to add and are the reason modern depth works
at all.
**Proof point:** a 20+ layer network that trains stably from scratch.

### Phase 4 — Enter the modern era
**Add:** the Transformer block — scaled dot-product attention, multi-head,
positional encoding (everything else is already in place by now).
**Why:** attention is the engine of the current era. This is the literal doorway
to "GPT-like."
**Proof point — the flagship moment:** a **readable, pure-C, character-level
Transformer that generates coherent text** (a tiny "GPT-nano"), trainable on a
laptop in minutes. This is Nerve looking the giants in the eye. Same idea they
use — small enough to read in one file.

### Phase 5 — Be everywhere
**Add:** a WASM build target; an embedded/TinyML inference profile (fixed-point
optional); optional SIMD + multithreading (and later a GPU backend) — all behind
flags, per invariant #1.
**Why:** portability *is* the product. The same trained model should run in a
browser tab, in a CLI, and on a microcontroller.
**Proof point:** one trained model demonstrated running in (a) a shareable
browser page with no server, and (b) a `$3` chip with no OS.

### Phase 6 — Be useful, daily
**Add:** a small model zoo (pretrained weights for common tasks), a stable
versioned API, real docs, and a frictionless install story (CMake FetchContent
already exists; add a single-command path).
**Why:** respect is earned not by a demo but by people *shipping* with it.
**Proof point:** a real product or research result that uses Nerve for private,
on-device, dependency-free inference.

---

## 5. Non-goals (on purpose)

Saying no is how the project stays alive and respected:

- **No chasing GPT-4-scale "power."** That is not an algorithm you add; it is
  billions of parameters, terabytes of data, and thousands of GPU-hours — a
  factory, not a library. Aiming there would waste the project. Nerve targets
  modern *architecture*, not industrial *scale*.
- **No mandatory heavyweight dependencies.** The moment Nerve *requires* CUDA,
  Python, or a 200 MB toolchain, it stops being Nerve.
- **No feature without a runnable proof.** Capability claims without a demo are
  vapor.

---

## 6. How respect is actually earned

Not with benchmarks nobody can reproduce. With **undeniable small artifacts**:

- a neural net that learns MNIST in a file you can read;
- a Transformer that writes text, in pure C, on a laptop;
- the same model running in a browser link and on a thumbnail-sized chip;
- a student who finally *gets* backprop because they read Nerve.

Each one is a quiet, concrete "hey, I'm here too." Stack enough of them and the
giants don't get to ignore the small thing that does what they said needed a
datacenter.

Small is the whole point. Keep it small. Make it undeniable.
