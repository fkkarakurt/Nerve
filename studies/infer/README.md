# Nerve — on-device inference & learning

A single-header, zero-dependency engine that **runs a real Transformer and
learns from your data — on your own machine.** No GPU, no Python at runtime, no
cloud, no per-token bill. The library is one header you drop into any C/C++
project; everything here builds with `cc file.c -lm`.

> Model weights and tokenizers are **not** committed (they are large, and they
> are *your* data). Download or convert them locally — see below. The engine
> reads Nerve's own formats: `*.nrv` (model) and `*.tok` (tokenizer).

## Files

| File | What it is |
|------|------------|
| `nerve_infer.h` | The inference engine: RMSNorm, RoPE, causal multi-head attention (GQA) + KV cache, SwiGLU, int8 weights, temperature/top-p sampling, BPE tokenizer. Single header. |
| `generate.c` | Text generation demo. |
| `learn.c` | **On-device learning**: uses a frozen base model as a feature extractor and trains a tiny head (with `../autograd/nerve_grad.h`) on your own labelled sentences. |
| `search.c` | **Local semantic search**: turns notes into embeddings and matches a query by *meaning* (cosine similarity, mean-centered), fully on-device — the core of "ask your own notes" / local RAG. |
| `convert.c` | Import a flat float32 checkpoint + SentencePiece vocab into Nerve's native `.nrv` / `.tok`. |
| `convert_hf.py` | Build-time tool: import a Hugging Face Llama-architecture model (safetensors) into int8 `.nrv`. Pure numpy + stdlib (no torch). Handles bf16, GQA, and the HF→interleaved RoPE un-permutation. |
| `quantize.c` | Convert a float32 `.nrv` to a 4×-smaller int8 `.nrv` (per-row symmetric). |

## Quick start — a tiny model that writes stories

```sh
# 1. get a small checkpoint + tokenizer in the classic flat format
curl -L -o stories15M.bin   https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o tokenizer.bin    https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
# 2. convert into Nerve's native formats, then (optionally) shrink to int8
gcc -O2 convert.c  -o convert  -lm && ./convert  stories15M.bin tokenizer.bin   # -> model.nrv, nerve.tok
gcc -O2 quantize.c -o quantize -lm && ./quantize                                 # -> model_q8.nrv (4x smaller)
# 3. generate
gcc -O3 -march=native -funroll-loops generate.c -o generate -lm
NERVE_MODEL=model_q8.nrv ./generate "Once upon a time" 200 0.8 0.9 42
```

## Run a real 1.1B LLM (TinyLlama)

```sh
python convert_hf.py tinyllama.safetensors tinyllama_config.json tinyllama_q8.nrv
NERVE_MODEL=tinyllama_q8.nrv ./generate "The meaning of life is" 80 0.7 0.9 42
```
~3 tokens/sec on a 2017 laptop CPU (i7-7700HQ), offline, zero runtime deps.
The Llama tokenizer matches `nerve.tok`, so it is reused as-is.

## Nerve learns on-device

```sh
gcc -O3 -march=native -funroll-loops learn.c -o learn -lm
NERVE_MODEL=tinyllama_q8.nrv ./learn
```
The frozen base model turns each sentence into a feature vector; a tiny linear
head, trained with Nerve's own autodiff on a handful of *your* examples, learns
*your* categories — in seconds, on the CPU, with no data leaving the machine.

## Local semantic search

```sh
gcc -O3 -march=native -funroll-loops search.c -o search -lm
NERVE_MODEL=tinyllama_q8.nrv ./search            # built-in demo queries
echo "how do plants make food?" | NERVE_MODEL=tinyllama_q8.nrv ./search -i
```
Matches a query to the closest notes by *meaning*, not keywords — queries
worded completely differently from the notes still find the right one. The core
of private, offline "ask your own notes" with no embeddings API and no per-call
bill.

## Native formats

- **`.nrv`** — `"NRV1"` magic, a self-describing 64-byte header (dims, flags,
  `rope_theta`), then weights. float32, or int8 (per-row symmetric) when the
  quantized flag is set. RoPE is computed on the fly (no wasted frequency
  tables).
- **`.tok`** — `"NTK1"` magic, then the BPE vocabulary (scores + pieces).
