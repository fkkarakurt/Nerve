# Nerve — sentence-embedding encoder

A single-header BERT/MiniLM-style **encoder** (`nerve_embed.h`) that turns a
sentence into a meaning vector — the right tool for understanding text:
teaching categories, semantic search, clustering. Zero dependencies beyond libm.

Unlike a decoder LM (which predicts the next word), an encoder reads the whole
sentence bidirectionally and is trained specifically for embeddings — so it is
*genuinely good* at "what does this mean?". On a quick probe it correctly places
`cola → food`, `running shoes → fitness`, `what time is my appointment →
calendar`, and leaves unrelated text with low scores everywhere.

> Model + vocab are **not** committed (downloaded/converted locally). The engine
> reads Nerve's `.nre` model and a plain WordPiece `vocab.txt`.

## Files

| File | What it is |
|------|------------|
| `nerve_embed.h`     | The encoder: WordPiece tokenizer, token/position/type embeddings, LayerNorm, bidirectional multi-head attention, GELU FFN, mean-pooling + L2-norm. Single header. |
| `convert_minilm.py` | Build-time tool: import `all-MiniLM-L6-v2` (HF safetensors) into Nerve's `.nre` format (pure numpy). |
| `embed_test.c`      | Quality probe: classify tricky inputs by nearest category centroid. |

## Build & run

```sh
# 1. fetch the model + vocab (sentence-transformers/all-MiniLM-L6-v2)
curl -L -o minilm.safetensors  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors
curl -L -o minilm_config.json  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
curl -L -o vocab.txt           https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
# 2. convert to Nerve's format, then probe
python convert_minilm.py minilm.safetensors minilm_config.json minilm.nre
gcc -O3 -march=native -funroll-loops embed_test.c -o embed_test -lm && ./embed_test
```

## Status & next

The fp32 model is ~86 MB (great on the desktop). Next: int8 quantization
(~22 MB) so it can power the browser demo's "teach" and "ask your notes"
features with genuinely smart embeddings.
