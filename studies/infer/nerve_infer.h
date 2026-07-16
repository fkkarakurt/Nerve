/*
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * nerve_infer.h — single-header Transformer INFERENCE for Nerve
 * ===========================================================================
 * Run a decoder-only Transformer and generate text from a single, dependency
 * -free C header you drop into any project. No build system, no Python, no
 * cloud, no per-token bill — the model runs where you run, and the weights are
 * yours.
 *
 * Architecture (all standard, published building blocks):
 *   - token embedding lookup
 *   - pre-norm blocks with RMSNorm           (Zhang & Sennrich, 2019)
 *   - rotary position embeddings (RoPE)       (Su et al., 2021)
 *   - causal multi-head self-attention        (Vaswani et al., 2017)
 *     with grouped/multi-query support and a KV cache
 *   - SwiGLU feed-forward                      (Shazeer, 2020)
 *   - tied or separate output classifier
 *   - temperature / top-p (nucleus) sampling
 *   - a SentencePiece-style BPE tokenizer
 *
 * Everything reads Nerve's OWN self-describing formats:
 *   model.nrv  — magic "NRV1", a 64-byte versioned header then float32 weights
 *   nerve.tok  — magic "NTK1", the BPE vocabulary
 * (Use the separate `convert` tool to import externally trained weights.)
 *
 *   #define NERVE_INFER_IMPLEMENTATION
 *   #include "nerve_infer.h"
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef NERVE_INFER_H
#define NERVE_INFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NERVE_NRV_MAGIC  "NRV1"
#define NERVE_NTK_MAGIC  "NTK1"
#define NERVE_NRV_HEADER 64        /* bytes before the weight blob */

/* ── Model configuration (read straight from the .nrv header) ────────────── */
typedef struct {
    int   dim;         /* transformer hidden width                            */
    int   hidden_dim;  /* FFN inner width                                     */
    int   n_layers;    /* number of transformer blocks                        */
    int   n_heads;     /* number of query heads                               */
    int   n_kv_heads;  /* number of key/value heads (< n_heads => grouped)    */
    int   vocab_size;  /* token vocabulary size                               */
    int   seq_len;     /* maximum context length                              */
    int   shared_cls;  /* 1 => output classifier reuses the embedding matrix  */
    int   quantized;   /* 1 => weights are int8 (per-row symmetric)           */
    float rope_theta;  /* RoPE base frequency (self-describing in the format) */
} nerve_config;

/* ── Pointers into the loaded weight blob ────────────────────────────────── */
typedef struct {
    float *token_embedding;  /* (vocab, dim)                                  */
    float *rms_att;          /* (layer, dim)                                  */
    float *rms_ffn;          /* (layer, dim)                                  */
    float *wq;               /* (layer, dim, n_heads    * head_size)          */
    float *wk;               /* (layer, dim, n_kv_heads * head_size)          */
    float *wv;               /* (layer, dim, n_kv_heads * head_size)          */
    float *wo;               /* (layer, n_heads * head_size, dim)             */
    float *w1, *w2, *w3;     /* SwiGLU feed-forward weights                   */
    float *rms_final;        /* (dim)                                         */
    float *wcls;             /* classifier (vocab, dim); may alias embedding  */

    /* int8 path — non-NULL only when config.quantized. Per-row symmetric:
     * each output row has one float scale, weights are signed 8-bit. The float
     * pointers above are NULL in this mode (and vice-versa). RMSNorm weights
     * stay float either way (tiny and precision-sensitive). */
    signed char *q_tok, *q_wq, *q_wk, *q_wv, *q_wo, *q_w1, *q_w2, *q_w3, *q_wcls;
    float       *s_tok, *s_wq, *s_wk, *s_wv, *s_wo, *s_w1, *s_w2, *s_w3, *s_wcls;
} nerve_weights;

/* ── Scratch buffers reused every forward step ───────────────────────────── */
typedef struct {
    float *x, *xb, *xb2;     /* residual stream + norm scratch (dim)          */
    float *hb, *hb2;         /* FFN scratch (hidden_dim)                      */
    float *q;                /* query (dim)                                   */
    float *att;              /* attention scores (n_heads, seq_len)           */
    float *logits;           /* output logits (vocab)                         */
    float *key_cache;        /* (layer, seq_len, kv_dim)                      */
    float *value_cache;      /* (layer, seq_len, kv_dim)                      */
} nerve_runstate;

typedef struct {
    nerve_config   config;
    nerve_weights  weights;
    nerve_runstate state;
    float         *data;       /* the weight blob, owned                      */
    size_t         data_size;
} nerve_transformer;

/* ── Tokenizer (BPE) ─────────────────────────────────────────────────────── */
typedef struct {
    char         **vocab;
    float         *scores;
    int            vocab_size;
    unsigned int   max_token_length;
    unsigned char  byte_pieces[512];   /* raw single-byte fallbacks           */
} nerve_tokenizer;

/* ── Sampler ─────────────────────────────────────────────────────────────── */
typedef struct {
    int                vocab_size;
    float              temperature;   /* 0 => greedy argmax                    */
    float              topp;          /* nucleus threshold (>=1 => off)        */
    unsigned long long rng_state;     /* xorshift state                        */
    void              *probindex;     /* scratch for top-p                     */
} nerve_sampler;

/* Returns 0 on success, non-zero on failure. */
int  nerve_infer_load(nerve_transformer *t, const char *model_path);
void nerve_infer_free(nerve_transformer *t);

int  nerve_tokenizer_load(nerve_tokenizer *tk, const char *path);
void nerve_tokenizer_free(nerve_tokenizer *tk);

void nerve_sampler_init(nerve_sampler *s, int vocab_size,
                        float temperature, float topp, unsigned long long seed);
void nerve_sampler_free(nerve_sampler *s);

/* Run one forward step for `token` at position `pos`; returns logits (vocab). */
float *nerve_infer_forward(nerve_transformer *t, int token, int pos);

/* The final hidden state (dim floats) from the most recent forward step — the
 * model's learned representation, used as a frozen feature for on-device
 * learning (train a small head on top without touching the base). */
float *nerve_infer_hidden(nerve_transformer *t);

/* Public tokenizer access: encode `text` into `tokens`, returns token count. */
int nerve_tokenizer_encode(nerve_tokenizer *tk, const char *text,
                           int bos, int eos, int *tokens);

/* High level: tokenize `prompt`, then autoregressively generate up to `steps`
 * tokens, calling `on_piece(piece, user)` for each decoded text fragment. If
 * `on_piece` is NULL the pieces are written to stdout. */
void nerve_generate(nerve_transformer *t, nerve_tokenizer *tk, nerve_sampler *s,
                    const char *prompt, int steps,
                    void (*on_piece)(const char *piece, void *user), void *user);

#ifdef __cplusplus
}
#endif
#endif /* NERVE_INFER_H */

/* ========================================================================= */
#ifdef NERVE_INFER_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* `restrict` is C99+; fall back cleanly so the core still builds as ANSI C89.
 * (gcc/clang also accept __restrict__ in C89 mode.) */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#  define NERVE_RESTRICT restrict
#elif defined(__GNUC__)
#  define NERVE_RESTRICT __restrict__
#else
#  define NERVE_RESTRICT
#endif

/* Optional multithreading: enabled only when the program is compiled with
 * -fopenmp. Without it these pragmas are ignored and the core stays pure,
 * single-threaded ANSI C with zero mandatory dependencies. */
#if defined(_OPENMP)
#  include <omp.h>
#endif

/* ── Math kernels ────────────────────────────────────────────────────────── */

/* RMSNorm: scale each element by the inverse root-mean-square of the row,
 * then by a learned per-element weight. No mean subtraction (that is the whole
 * point — cheaper and just as effective as LayerNorm for transformers). */
static void nerve_i__rmsnorm(float *o, const float *x, const float *w, int n)
{
    int   j;
    float ss = 0.0f;
    for (j = 0; j < n; j++) ss += x[j] * x[j];
    ss = 1.0f / (float)sqrt((double)(ss / (float)n) + 1e-5);
    for (j = 0; j < n; j++) o[j] = w[j] * (ss * x[j]);
}

/* Numerically stable softmax over the first n elements, in place. */
static void nerve_i__softmax(float *x, int n)
{
    int   i;
    float maxv = x[0], sum = 0.0f;
    for (i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    for (i = 0; i < n; i++) { x[i] = (float)exp((double)(x[i] - maxv)); sum += x[i]; }
    for (i = 0; i < n; i++) x[i] /= sum;
}

/* out(d) = W(d,n) . x(n): one dot product per output row — where ~all the time
 * goes. Two tricks, both pure portable C (no intrinsics):
 *   1. Eight independent accumulators. A single running sum can't be
 *      auto-vectorised (float addition isn't associative, so the compiler
 *      won't reorder it). Eight lanes summed in fixed order ARE reorder-free
 *      yet map straight onto one 256-bit SIMD register, so `-O3 -march=native`
 *      emits vector FMAs for us — portably, on whatever ISA the target has.
 *   2. `restrict` lets the compiler assume out/x/w don't alias, so it can keep
 *      everything in registers across the loop.
 * Outer loop is independent per row, so it parallelises across cores when the
 * caller opts into OpenMP. */
static void nerve_i__matmul(float *NERVE_RESTRICT out, const float *NERVE_RESTRICT x,
                            const float *NERVE_RESTRICT w, int n, int d)
{
    int i;
#if defined(_OPENMP)
#   pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < d; i++) {
        const float *NERVE_RESTRICT row = w + (long)i * n;
        float acc[8];
        float v;
        int   j, k;
        for (k = 0; k < 8; k++) acc[k] = 0.0f;
        for (j = 0; j + 8 <= n; j += 8)
            for (k = 0; k < 8; k++) acc[k] += row[j + k] * x[j + k];
        v = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
            ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; j < n; j++) v += row[j] * x[j];
        out[i] = v;
    }
}

/* Quantized matvec: weights are int8 with one float scale per output row.
 * out[i] = scale[i] * sum_j (q[i][j] * x[j]). Only 1/4 the weight bytes are
 * read from memory — and on CPU inference, memory bandwidth, not arithmetic,
 * is the real bottleneck, so this is the lever that lets useful-sized models
 * run on modest hardware. The dot product keeps the same eight-lane shape so
 * it still auto-vectorises. */
static void nerve_i__qmatmul(float *NERVE_RESTRICT out, const float *NERVE_RESTRICT x,
                             const signed char *NERVE_RESTRICT q,
                             const float *NERVE_RESTRICT scale, int n, int d)
{
    int i;
#if defined(_OPENMP)
#   pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < d; i++) {
        const signed char *NERVE_RESTRICT row = q + (long)i * n;
        float acc[8];
        float v;
        int   j, k;
        for (k = 0; k < 8; k++) acc[k] = 0.0f;
        for (j = 0; j + 8 <= n; j += 8)
            for (k = 0; k < 8; k++) acc[k] += (float)row[j + k] * x[j + k];
        v = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
            ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; j < n; j++) v += (float)row[j] * x[j];
        out[i] = v * scale[i];
    }
}

/* Pick the right kernel: int8 when a quantized weight is supplied (q != NULL),
 * float otherwise. Lets the forward pass stay written once for both models. */
static void nerve_i__mm(float *out, const float *x,
                        const float *wf, const signed char *wq, const float *ws,
                        int n, int d)
{
    if (wq) nerve_i__qmatmul(out, x, wq, ws, n, d);
    else    nerve_i__matmul(out, x, wf, n, d);
}

/* ── Loading the native .nrv model ───────────────────────────────────────── */
static int nerve_i__read_i32(FILE *f, int *v) { return fread(v, sizeof(int), 1, f) == 1; }

static void nerve_i__map_weights(nerve_weights *w, const nerve_config *p, float *ptr)
{
    int  head_size = p->dim / p->n_heads;
    long L = p->n_layers;
    w->token_embedding = ptr; ptr += (long)p->vocab_size * p->dim;
    w->rms_att = ptr;         ptr += L * p->dim;
    w->wq = ptr;              ptr += L * p->dim * (p->n_heads    * head_size);
    w->wk = ptr;              ptr += L * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;              ptr += L * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;              ptr += L * (p->n_heads * head_size) * p->dim;
    w->rms_ffn = ptr;         ptr += L * p->dim;
    w->w1 = ptr;              ptr += L * p->dim * p->hidden_dim;
    w->w2 = ptr;              ptr += L * p->hidden_dim * p->dim;
    w->w3 = ptr;              ptr += L * p->dim * p->hidden_dim;
    w->rms_final = ptr;       ptr += p->dim;
    w->wcls = p->shared_cls ? w->token_embedding : ptr;
    /* float model: no int8 weights */
    w->q_tok = w->q_wq = w->q_wk = w->q_wv = w->q_wo =
        w->q_w1 = w->q_w2 = w->q_w3 = w->q_wcls = NULL;
    w->s_tok = w->s_wq = w->s_wk = w->s_wv = w->s_wo =
        w->s_w1 = w->s_w2 = w->s_w3 = w->s_wcls = NULL;
}

/* int8 model layout. Each quantized tensor is stored as R float scales (one
 * per output row) followed by R*C signed bytes. RMSNorm vectors stay float. */
static void nerve_i__map_weights_q(nerve_weights *w, const nerve_config *p, void *base)
{
    int  hs  = p->dim / p->n_heads;
    long L   = p->n_layers, dim = p->dim, hid = p->hidden_dim, voc = p->vocab_size;
    long kvd = (long)p->n_kv_heads * hs;
    char *c  = (char *)base;

    w->s_tok = (float *)c; c += voc * sizeof(float);
    w->q_tok = (signed char *)c; c += voc * dim;
    w->rms_att = (float *)c; c += L * dim * sizeof(float);
    w->s_wq = (float *)c; c += L * dim * sizeof(float);  w->q_wq = (signed char *)c; c += L * dim * dim;
    w->s_wk = (float *)c; c += L * kvd * sizeof(float);  w->q_wk = (signed char *)c; c += L * kvd * dim;
    w->s_wv = (float *)c; c += L * kvd * sizeof(float);  w->q_wv = (signed char *)c; c += L * kvd * dim;
    w->s_wo = (float *)c; c += L * dim * sizeof(float);  w->q_wo = (signed char *)c; c += L * dim * dim;
    w->rms_ffn = (float *)c; c += L * dim * sizeof(float);
    w->s_w1 = (float *)c; c += L * hid * sizeof(float);  w->q_w1 = (signed char *)c; c += L * hid * dim;
    w->s_w2 = (float *)c; c += L * dim * sizeof(float);  w->q_w2 = (signed char *)c; c += L * dim * hid;
    w->s_w3 = (float *)c; c += L * hid * sizeof(float);  w->q_w3 = (signed char *)c; c += L * hid * dim;
    w->rms_final = (float *)c; c += dim * sizeof(float);
    if (p->shared_cls) { w->q_wcls = w->q_tok; w->s_wcls = w->s_tok; }
    else { w->s_wcls = (float *)c; c += voc * sizeof(float); w->q_wcls = (signed char *)c; c += voc * dim; }

    w->token_embedding = w->wq = w->wk = w->wv = w->wo = NULL;
    w->w1 = w->w2 = w->w3 = w->wcls = NULL;
}

static int nerve_i__alloc_state(nerve_runstate *s, const nerve_config *p)
{
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x      = (float *)calloc(p->dim, sizeof(float));
    s->xb     = (float *)calloc(p->dim, sizeof(float));
    s->xb2    = (float *)calloc(p->dim, sizeof(float));
    s->hb     = (float *)calloc(p->hidden_dim, sizeof(float));
    s->hb2    = (float *)calloc(p->hidden_dim, sizeof(float));
    s->q      = (float *)calloc(p->dim, sizeof(float));
    s->att    = (float *)calloc((long)p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    s->key_cache   = (float *)calloc((long)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc((long)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    return (s->x && s->xb && s->xb2 && s->hb && s->hb2 && s->q && s->att &&
            s->logits && s->key_cache && s->value_cache) ? 0 : -1;
}

int nerve_infer_load(nerve_transformer *t, const char *path)
{
    FILE *f = fopen(path, "rb");
    char  magic[4];
    int   version, flags;
    long  blob_bytes;
    nerve_config *p = &t->config;
    if (!f) return -1;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, NERVE_NRV_MAGIC, 4) != 0) { fclose(f); return -2; }
    if (!nerve_i__read_i32(f, &version)) { fclose(f); return -3; }
    if (!nerve_i__read_i32(f, &p->dim)        || !nerve_i__read_i32(f, &p->hidden_dim) ||
        !nerve_i__read_i32(f, &p->n_layers)   || !nerve_i__read_i32(f, &p->n_heads)    ||
        !nerve_i__read_i32(f, &p->n_kv_heads) || !nerve_i__read_i32(f, &p->vocab_size) ||
        !nerve_i__read_i32(f, &p->seq_len)    || !nerve_i__read_i32(f, &flags)) { fclose(f); return -4; }
    if (fread(&p->rope_theta, sizeof(float), 1, f) != 1) { fclose(f); return -5; }
    p->shared_cls = flags & 1;
    p->quantized  = (flags >> 1) & 1;

    /* the weight blob starts at a fixed 64-byte offset */
    fseek(f, 0, SEEK_END);
    blob_bytes = ftell(f) - NERVE_NRV_HEADER;
    fseek(f, NERVE_NRV_HEADER, SEEK_SET);
    t->data_size = (size_t)blob_bytes;
    t->data = (float *)malloc((size_t)blob_bytes);
    if (!t->data) { fclose(f); return -6; }
    if (fread(t->data, 1, (size_t)blob_bytes, f) != (size_t)blob_bytes) { fclose(f); free(t->data); return -7; }
    fclose(f);

    if (p->quantized) nerve_i__map_weights_q(&t->weights, p, t->data);
    else              nerve_i__map_weights(&t->weights, p, t->data);
    return nerve_i__alloc_state(&t->state, p);
}

void nerve_infer_free(nerve_transformer *t)
{
    nerve_runstate *s = &t->state;
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(t->data);
}

/* ── The forward pass (one token at position `pos`) ──────────────────────── */
float *nerve_infer_forward(nerve_transformer *t, int token, int pos)
{
    nerve_config   *p = &t->config;
    nerve_weights  *w = &t->weights;
    nerve_runstate *s = &t->state;
    int   dim       = p->dim;
    int   kv_dim    = (p->dim * p->n_kv_heads) / p->n_heads;
    int   kv_mul    = p->n_heads / p->n_kv_heads;     /* query heads per kv head */
    int   hidden    = p->hidden_dim;
    int   head_size = dim / p->n_heads;
    float scale     = 1.0f / (float)sqrt((double)head_size);
    float *x = s->x;
    int   l, h, i, tt;

    /* start the residual stream from the token's embedding row */
    if (p->quantized) {
        const signed char *row = w->q_tok + (long)token * dim;
        float sc = w->s_tok[token];
        for (i = 0; i < dim; i++) x[i] = sc * (float)row[i];
    } else {
        memcpy(x, w->token_embedding + (long)token * dim, dim * sizeof(float));
    }

    for (l = 0; l < p->n_layers; l++) {
        long   loff = (long)l * p->seq_len * kv_dim;
        float *krow = s->key_cache   + loff + (long)pos * kv_dim;
        float *vrow = s->value_cache + loff + (long)pos * kv_dim;

        /* --- attention --- */
        nerve_i__rmsnorm(s->xb, x, w->rms_att + (long)l * dim, dim);
        nerve_i__mm(s->q, s->xb,
                    w->wq   ? w->wq   + (long)l * dim * dim : NULL,
                    w->q_wq ? w->q_wq + (long)l * dim * dim : NULL,
                    w->s_wq ? w->s_wq + (long)l * dim       : NULL, dim, dim);
        nerve_i__mm(krow, s->xb,
                    w->wk   ? w->wk   + (long)l * dim * kv_dim : NULL,
                    w->q_wk ? w->q_wk + (long)l * dim * kv_dim : NULL,
                    w->s_wk ? w->s_wk + (long)l * kv_dim       : NULL, dim, kv_dim);
        nerve_i__mm(vrow, s->xb,
                    w->wv   ? w->wv   + (long)l * dim * kv_dim : NULL,
                    w->q_wv ? w->q_wv + (long)l * dim * kv_dim : NULL,
                    w->s_wv ? w->s_wv + (long)l * kv_dim       : NULL, dim, kv_dim);

        /* RoPE: rotate each adjacent (even,odd) pair by an angle that grows
         * with position and shrinks with depth-in-head. This injects relative
         * position directly into Q and K. */
        for (i = 0; i < dim; i += 2) {
            int   hd   = i % head_size;
            float freq = 1.0f / (float)pow((double)p->rope_theta, (double)hd / (double)head_size);
            float ang  = (float)pos * freq;
            float c    = (float)cos((double)ang), sn = (float)sin((double)ang);
            int   pairs = (i < kv_dim) ? 2 : 1;   /* rotate K only within kv_dim */
            int   r;
            for (r = 0; r < pairs; r++) {
                float *vec = (r == 0) ? s->q : krow;
                float a = vec[i], b = vec[i + 1];
                vec[i]     = a * c - b * sn;
                vec[i + 1] = a * sn + b * c;
            }
        }

        /* causal self-attention, per head */
        for (h = 0; h < p->n_heads; h++) {
            float *q   = s->q   + h * head_size;
            float *att = s->att + (long)h * p->seq_len;
            float *out = s->xb  + h * head_size;
            for (tt = 0; tt <= pos; tt++) {
                float *k = s->key_cache + loff + (long)tt * kv_dim + (h / kv_mul) * head_size;
                float dot = 0.0f;
                for (i = 0; i < head_size; i++) dot += q[i] * k[i];
                att[tt] = dot * scale;
            }
            nerve_i__softmax(att, pos + 1);
            for (i = 0; i < head_size; i++) out[i] = 0.0f;
            for (tt = 0; tt <= pos; tt++) {
                float *v = s->value_cache + loff + (long)tt * kv_dim + (h / kv_mul) * head_size;
                float a = att[tt];
                for (i = 0; i < head_size; i++) out[i] += a * v[i];
            }
        }

        nerve_i__mm(s->xb2, s->xb,
                    w->wo   ? w->wo   + (long)l * dim * dim : NULL,
                    w->q_wo ? w->q_wo + (long)l * dim * dim : NULL,
                    w->s_wo ? w->s_wo + (long)l * dim       : NULL, dim, dim);
        for (i = 0; i < dim; i++) x[i] += s->xb2[i];        /* residual */

        /* --- SwiGLU feed-forward:  w2( silu(w1 x) (*) w3 x ) --- */
        nerve_i__rmsnorm(s->xb, x, w->rms_ffn + (long)l * dim, dim);
        nerve_i__mm(s->hb, s->xb,
                    w->w1   ? w->w1   + (long)l * dim * hidden : NULL,
                    w->q_w1 ? w->q_w1 + (long)l * hidden * dim : NULL,
                    w->s_w1 ? w->s_w1 + (long)l * hidden       : NULL, dim, hidden);
        nerve_i__mm(s->hb2, s->xb,
                    w->w3   ? w->w3   + (long)l * dim * hidden : NULL,
                    w->q_w3 ? w->q_w3 + (long)l * hidden * dim : NULL,
                    w->s_w3 ? w->s_w3 + (long)l * hidden       : NULL, dim, hidden);
        for (i = 0; i < hidden; i++) {
            float v = s->hb[i];
            v *= 1.0f / (1.0f + (float)exp(-(double)v));    /* SiLU / swish */
            s->hb[i] = v * s->hb2[i];
        }
        nerve_i__mm(s->xb, s->hb,
                    w->w2   ? w->w2   + (long)l * dim * hidden : NULL,
                    w->q_w2 ? w->q_w2 + (long)l * dim * hidden : NULL,
                    w->s_w2 ? w->s_w2 + (long)l * dim          : NULL, hidden, dim);
        for (i = 0; i < dim; i++) x[i] += s->xb[i];         /* residual */
    }

    nerve_i__rmsnorm(x, x, w->rms_final, dim);
    nerve_i__mm(s->logits, x, w->wcls, w->q_wcls, w->s_wcls, dim, p->vocab_size);
    return s->logits;
}

float *nerve_infer_hidden(nerve_transformer *t) { return t->state.x; }

/* ── Tokenizer (native .tok) ─────────────────────────────────────────────── */
int nerve_tokenizer_load(nerve_tokenizer *tk, const char *path)
{
    FILE *f = fopen(path, "rb");
    char  magic[4];
    int   version, i, len;
    if (!f) return -1;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, NERVE_NTK_MAGIC, 4) != 0) { fclose(f); return -2; }
    if (fread(&version, sizeof(int), 1, f) != 1) { fclose(f); return -3; }
    if (fread(&tk->vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return -4; }
    if (fread(&tk->max_token_length, sizeof(int), 1, f) != 1) { fclose(f); return -5; }
    tk->vocab  = (char **)malloc((size_t)tk->vocab_size * sizeof(char *));
    tk->scores = (float *)malloc((size_t)tk->vocab_size * sizeof(float));
    for (i = 0; i < 256; i++) { tk->byte_pieces[i*2] = (unsigned char)i; tk->byte_pieces[i*2+1] = 0; }
    for (i = 0; i < tk->vocab_size; i++) {
        if (fread(tk->scores + i, sizeof(float), 1, f) != 1) { fclose(f); return -6; }
        if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return -7; }
        tk->vocab[i] = (char *)malloc((size_t)len + 1);
        if (fread(tk->vocab[i], 1, (size_t)len, f) != (size_t)len) { fclose(f); return -8; }
        tk->vocab[i][len] = '\0';
    }
    fclose(f);
    return 0;
}

void nerve_tokenizer_free(nerve_tokenizer *tk)
{
    int i;
    for (i = 0; i < tk->vocab_size; i++) free(tk->vocab[i]);
    free(tk->vocab); free(tk->scores);
}

static int nerve_i__lookup(const char *str, char **vocab, int n)
{
    int i;
    for (i = 0; i < n; i++) if (strcmp(str, vocab[i]) == 0) return i;
    return -1;
}

/* BPE encode: split into UTF-8 pieces (byte fallback when unknown), then keep
 * merging the highest-scoring adjacent pair until none can merge. Token 1 is
 * BOS, token 2 is EOS; the leading space is the usual word-boundary marker. */
static void nerve_i__encode(nerve_tokenizer *tk, const char *text,
                            int bos, int eos, int *tokens, int *n_tokens)
{
    char  *buf = (char *)malloc((size_t)tk->max_token_length * 2 + 3);
    size_t blen = 0;
    const char *c;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int sp = nerve_i__lookup(" ", tk->vocab, tk->vocab_size);
        if (sp != -1) tokens[(*n_tokens)++] = sp;
    }
    for (c = text; *c != '\0'; c++) {
        int id;
        if (((unsigned char)*c & 0xC0) != 0x80) blen = 0;   /* new codepoint */
        buf[blen++] = *c; buf[blen] = '\0';
        if (((unsigned char)*(c + 1) & 0xC0) == 0x80 && blen < 4) continue;
        id = nerve_i__lookup(buf, tk->vocab, tk->vocab_size);
        if (id != -1) tokens[(*n_tokens)++] = id;
        else { size_t k; for (k = 0; k < blen; k++) tokens[(*n_tokens)++] = (unsigned char)buf[k] + 3; }
        blen = 0;
    }
    for (;;) {
        float best = -1e10f; int best_id = -1, best_idx = -1, i;
        for (i = 0; i < (*n_tokens) - 1; i++) {
            sprintf(buf, "%s%s", tk->vocab[tokens[i]], tk->vocab[tokens[i + 1]]);
            int id = nerve_i__lookup(buf, tk->vocab, tk->vocab_size);
            if (id != -1 && tk->scores[id] > best) { best = tk->scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (i = best_idx + 1; i < (*n_tokens) - 1; i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(buf);
}

static const char *nerve_i__decode(nerve_tokenizer *tk, int prev, int token)
{
    char *piece = tk->vocab[token];
    unsigned int byte_val;
    if (prev == 1 && piece[0] == ' ') piece++;   /* drop the BOS-leading space */
    if (sscanf(piece, "<0x%02X>", &byte_val) == 1)
        piece = (char *)tk->byte_pieces + byte_val * 2;
    return piece;
}

int nerve_tokenizer_encode(nerve_tokenizer *tk, const char *text,
                           int bos, int eos, int *tokens)
{
    int n = 0;
    nerve_i__encode(tk, text, bos, eos, tokens, &n);
    return n;
}

/* ── Sampler ─────────────────────────────────────────────────────────────── */
typedef struct { float prob; int index; } nerve_i__pi;

void nerve_sampler_init(nerve_sampler *s, int vocab_size,
                        float temperature, float topp, unsigned long long seed)
{
    s->vocab_size  = vocab_size;
    s->temperature = temperature;
    s->topp        = topp;
    s->rng_state   = seed ? seed : 1ULL;
    s->probindex   = malloc((size_t)vocab_size * sizeof(nerve_i__pi));
}

void nerve_sampler_free(nerve_sampler *s) { free(s->probindex); }

static unsigned int nerve_i__rand_u32(unsigned long long *st)
{
    *st ^= *st >> 12; *st ^= *st << 25; *st ^= *st >> 27;
    return (unsigned int)((*st * 0x2545F4914F6CDD1DULL) >> 32);
}
static float nerve_i__rand_f32(unsigned long long *st)
{ return (nerve_i__rand_u32(st) >> 8) / 16777216.0f; }

static int nerve_i__argmax(const float *p, int n)
{ int i, mi = 0; for (i = 1; i < n; i++) if (p[i] > p[mi]) mi = i; return mi; }

static int nerve_i__sample_mult(const float *p, int n, float coin)
{
    int   i; float cdf = 0.0f;
    for (i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}

static int nerve_i__cmp_pi(const void *a, const void *b)
{
    float pa = ((const nerve_i__pi *)a)->prob, pb = ((const nerve_i__pi *)b)->prob;
    return (pa < pb) - (pa > pb);   /* descending */
}

static int nerve_i__sample_topp(const float *p, int n, float topp,
                                nerve_i__pi *order, float coin)
{
    int   i, n0 = 0, last;
    float cutoff = (1.0f - topp) / (float)(n - 1);
    float cum = 0.0f, r, cdf = 0.0f;
    for (i = 0; i < n; i++)
        if (p[i] >= cutoff) { order[n0].index = i; order[n0].prob = p[i]; n0++; }
    qsort(order, (size_t)n0, sizeof(nerve_i__pi), nerve_i__cmp_pi);
    last = n0 - 1;
    for (i = 0; i < n0; i++) { cum += order[i].prob; if (cum > topp) { last = i; break; } }
    r = coin * cum;
    for (i = 0; i <= last; i++) { cdf += order[i].prob; if (r < cdf) return order[i].index; }
    return order[last].index;
}

static int nerve_i__sample(nerve_sampler *s, float *logits)
{
    int i;
    if (s->temperature == 0.0f) return nerve_i__argmax(logits, s->vocab_size);
    for (i = 0; i < s->vocab_size; i++) logits[i] /= s->temperature;
    nerve_i__softmax(logits, s->vocab_size);
    {
        float coin = nerve_i__rand_f32(&s->rng_state);
        if (s->topp <= 0.0f || s->topp >= 1.0f)
            return nerve_i__sample_mult(logits, s->vocab_size, coin);
        return nerve_i__sample_topp(logits, s->vocab_size, s->topp,
                                    (nerve_i__pi *)s->probindex, coin);
    }
}

/* ── Generation loop ─────────────────────────────────────────────────────── */
void nerve_generate(nerve_transformer *t, nerve_tokenizer *tk, nerve_sampler *s,
                    const char *prompt, int steps,
                    void (*on_piece)(const char *piece, void *user), void *user)
{
    int *ptoks;
    int  n_prompt = 0, pos = 0, token, next;
    if (prompt == NULL) prompt = "";
    ptoks = (int *)malloc((size_t)(strlen(prompt) + 3) * sizeof(int));
    nerve_i__encode(tk, prompt, 1 /*bos*/, 0 /*eos*/, ptoks, &n_prompt);
    if (n_prompt < 1) { free(ptoks); return; }

    token = ptoks[0];
    while (pos < steps) {
        float *logits = nerve_infer_forward(t, token, pos);
        if (pos < n_prompt - 1) next = ptoks[pos + 1];     /* still reading prompt */
        else                    next = nerve_i__sample(s, logits);
        pos++;
        if (next == 1) break;                              /* BOS marks end */
        {
            const char *piece = nerve_i__decode(tk, token, next);
            if (piece && piece[0] != '\0') {
                if (on_piece) on_piece(piece, user);
                else { fputs(piece, stdout); fflush(stdout); }
            }
        }
        token = next;
    }
    free(ptoks);
}

#endif /* NERVE_INFER_IMPLEMENTATION */
