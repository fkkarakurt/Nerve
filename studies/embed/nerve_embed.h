/*
 * nerve_embed.h — single-header sentence-embedding ENCODER for Nerve.
 * ===========================================================================
 * A BERT/MiniLM-style bidirectional Transformer encoder that turns a sentence
 * into a meaning vector — the right tool for "understand my text": teaching
 * categories, semantic search, clustering. Zero dependencies beyond libm.
 *
 * Pipeline: WordPiece tokenize -> token+position+type embeddings -> LayerNorm
 * -> N encoder layers (bidirectional multi-head attention + GELU feed-forward,
 * pre-bias, post-LayerNorm) -> mean-pool over tokens -> L2-normalise.
 *
 * Reads Nerve's own .nre model (see convert_minilm.py) plus a plain WordPiece
 * vocab.txt (one token per line, line number = id).
 *
 *   #define NERVE_EMBED_IMPLEMENTATION
 *   #include "nerve_embed.h"
 * SPDX-License-Identifier: MIT
 */
#ifndef NERVE_EMBED_H
#define NERVE_EMBED_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int hidden, n_layers, n_heads, intermediate, vocab, max_pos;
} nerve_embed_config;

typedef struct {
    float *qw,*qb,*kw,*kb,*vw,*vb,*ow,*ob, *aln_w,*aln_b,
          *iw,*ib,*dw,*db, *oln_w,*oln_b;
} nerve_embed_layer;

typedef struct {
    nerve_embed_config cfg;
    float  *data;                 /* the weight blob */
    float  *word, *pos, *type, *eln_w, *eln_b;
    nerve_embed_layer *layer;
    /* WordPiece vocab + hash */
    char  **vocab;
    int    *hbucket, hsize;
    /* scratch */
    float  *x, *q, *k, *v, *ctx, *att, *h1, *xb;
    int     maxT;
} nerve_embed_t;

int  nerve_embed_load(nerve_embed_t *m, const char *model_path, const char *vocab_path);
void nerve_embed_free(nerve_embed_t *m);
int  nerve_embed_dim(const nerve_embed_t *m);
/* Encode text into out[hidden] (mean-pooled, L2-normalised). */
void nerve_embed_text(nerve_embed_t *m, const char *text, float *out);

#ifdef __cplusplus
}
#endif
#endif /* NERVE_EMBED_H */

/* ========================================================================= */
#ifdef NERVE_EMBED_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NERVE_EMB_CLS 101
#define NERVE_EMB_SEP 102
#define NERVE_EMB_UNK 100
#define NERVE_EMB_MAXT 256

/* ── small kernels ───────────────────────────────────────────────────────── */
static void nerve_e__linear(float *out, const float *x, const float *w,
                            const float *b, int in, int d)
{   /* out[d] = W[d,in] x[in] + b[d] */
    int i, j;
    for (i = 0; i < d; i++) {
        const float *r = w + (long)i * in;
        float s = b ? b[i] : 0.0f;
        for (j = 0; j < in; j++) s += r[j] * x[j];
        out[i] = s;
    }
}
static void nerve_e__layernorm(float *x, const float *w, const float *b, int n)
{
    int i; float mean = 0, var = 0, inv;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    inv = 1.0f / (float)sqrt((double)var + 1e-12);
    for (i = 0; i < n; i++) x[i] = w[i] * ((x[i] - mean) * inv) + b[i];
}
static float nerve_e__gelu(float x)
{ return 0.5f * x * (1.0f + (float)erf((double)x * 0.7071067811865476)); }

static void nerve_e__softmax(float *x, int n)
{
    int i; float mx = x[0], s = 0;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) { x[i] = (float)exp((double)(x[i] - mx)); s += x[i]; }
    for (i = 0; i < n; i++) x[i] /= s;
}

/* ── WordPiece ───────────────────────────────────────────────────────────── */
static unsigned nerve_e__hash(const char *s)
{ unsigned h = 2166136261u; while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; } return h; }

static int nerve_e__lookup(nerve_embed_t *m, const char *s)
{
    unsigned h = nerve_e__hash(s) & (m->hsize - 1);
    while (m->hbucket[h]) {
        int id = m->hbucket[h] - 1;
        if (strcmp(m->vocab[id], s) == 0) return id;
        h = (h + 1) & (m->hsize - 1);
    }
    return -1;
}

/* Tokenize lowercased text into WordPiece ids, with [CLS] … [SEP]. */
static int nerve_e__encode(nerve_embed_t *m, const char *text, int *out, int maxn)
{
    char word[512]; int n = 0, wl = 0; const char *c;
    out[n++] = NERVE_EMB_CLS;

    for (c = text; ; c++) {
        int ch = (unsigned char)*c;
        int is_alnum = (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9');
        int lo = (ch >= 'A' && ch <= 'Z') ? ch + 32 : ch;
        int lo_alnum = (lo >= 'a' && lo <= 'z') || (lo >= '0' && lo <= '9');
        (void)is_alnum;
        if (lo_alnum) { if (wl < 500) word[wl++] = (char)lo; continue; }

        /* flush accumulated word via greedy WordPiece */
        if (wl > 0) {
            word[wl] = '\0';
            if (wl > 100) { if (n < maxn) out[n++] = NERVE_EMB_UNK; }
            else {
                int start = 0, bad = 0, tmpn = n;
                while (start < wl) {
                    int end = wl, found = -1;
                    char buf[520];
                    while (end > start) {
                        int len = end - start, off = 0;
                        if (start > 0) { buf[0] = '#'; buf[1] = '#'; off = 2; }
                        memcpy(buf + off, word + start, (size_t)len); buf[off + len] = '\0';
                        found = nerve_e__lookup(m, buf);
                        if (found >= 0) break;
                        end--;
                    }
                    if (found < 0) { bad = 1; break; }
                    if (n < maxn) out[n++] = found;
                    start = end;
                }
                if (bad) { n = tmpn; if (n < maxn) out[n++] = NERVE_EMB_UNK; }
            }
            wl = 0;
        }
        if (ch == '\0') break;
        /* a punctuation/symbol char becomes its own token */
        if (lo != ' ' && lo != '\t' && lo != '\n' && lo != '\r') {
            char p[2]; int id; p[0] = (char)lo; p[1] = '\0';
            id = nerve_e__lookup(m, p);
            if (n < maxn) out[n++] = (id >= 0 ? id : NERVE_EMB_UNK);
        }
        if (n >= maxn - 1) break;
    }
    if (n < maxn) out[n++] = NERVE_EMB_SEP;
    return n;
}

/* ── load ────────────────────────────────────────────────────────────────── */
static char **nerve_e__load_vocab(const char *path, int vocab)
{
    FILE *f = fopen(path, "rb");
    char **v, line[512]; int i = 0;
    if (!f) return 0;
    v = (char **)calloc((size_t)vocab, sizeof(char *));
    while (i < vocab && fgets(line, sizeof(line), f)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = '\0';
        v[i] = (char *)malloc(L + 1); memcpy(v[i], line, L + 1); i++;
    }
    fclose(f);
    return (i == vocab) ? v : (free(v), (char **)0);
}

int nerve_embed_load(nerve_embed_t *m, const char *model_path, const char *vocab_path)
{
    FILE *f = fopen(model_path, "rb");
    char magic[4]; int ver, i; long bytes; float *p;
    nerve_embed_config *c = &m->cfg;
    if (!f) return -1;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "NEMB", 4)) { fclose(f); return -2; }
    fread(&ver, 4, 1, f);
    fread(&c->hidden, 4, 1, f); fread(&c->n_layers, 4, 1, f); fread(&c->n_heads, 4, 1, f);
    fread(&c->intermediate, 4, 1, f); fread(&c->vocab, 4, 1, f); fread(&c->max_pos, 4, 1, f);
    fseek(f, 0, SEEK_END); bytes = ftell(f) - 64; fseek(f, 64, SEEK_SET);
    m->data = (float *)malloc((size_t)bytes);
    if (fread(m->data, 1, (size_t)bytes, f) != (size_t)bytes) { fclose(f); return -3; }
    fclose(f);

    /* map weight pointers (canonical order from convert_minilm.py) */
    {
        int H = c->hidden, I = c->intermediate, l;
        p = m->data;
        m->word = p; p += (long)c->vocab * H;
        m->pos  = p; p += (long)c->max_pos * H;
        m->type = p; p += 2 * H;
        m->eln_w = p; p += H; m->eln_b = p; p += H;
        m->layer = (nerve_embed_layer *)calloc((size_t)c->n_layers, sizeof(nerve_embed_layer));
        for (l = 0; l < c->n_layers; l++) {
            nerve_embed_layer *L = &m->layer[l];
            L->qw=p;p+=H*H; L->qb=p;p+=H; L->kw=p;p+=H*H; L->kb=p;p+=H;
            L->vw=p;p+=H*H; L->vb=p;p+=H; L->ow=p;p+=H*H; L->ob=p;p+=H;
            L->aln_w=p;p+=H; L->aln_b=p;p+=H;
            L->iw=p;p+=(long)I*H; L->ib=p;p+=I; L->dw=p;p+=(long)H*I; L->db=p;p+=H;
            L->oln_w=p;p+=H; L->oln_b=p;p+=H;
        }
    }

    /* vocab + hash table */
    m->vocab = nerve_e__load_vocab(vocab_path, c->vocab);
    if (!m->vocab) return -4;
    m->hsize = 1; while (m->hsize < c->vocab * 4) m->hsize <<= 1;
    m->hbucket = (int *)calloc((size_t)m->hsize, sizeof(int));
    for (i = 0; i < c->vocab; i++) {
        unsigned h = nerve_e__hash(m->vocab[i]) & (m->hsize - 1);
        while (m->hbucket[h]) h = (h + 1) & (m->hsize - 1);
        m->hbucket[h] = i + 1;
    }

    /* scratch */
    m->maxT = NERVE_EMB_MAXT;
    {
        int H = c->hidden, I = c->intermediate, T = m->maxT;
        m->x   = (float *)malloc((size_t)T * H * sizeof(float));
        m->xb  = (float *)malloc((size_t)T * H * sizeof(float));
        m->q   = (float *)malloc((size_t)T * H * sizeof(float));
        m->k   = (float *)malloc((size_t)T * H * sizeof(float));
        m->v   = (float *)malloc((size_t)T * H * sizeof(float));
        m->ctx = (float *)malloc((size_t)T * H * sizeof(float));
        m->att = (float *)malloc((size_t)T * sizeof(float));
        m->h1  = (float *)malloc((size_t)I * sizeof(float));
    }
    return 0;
}

void nerve_embed_free(nerve_embed_t *m)
{
    int i;
    if (m->vocab) { for (i = 0; i < m->cfg.vocab; i++) free(m->vocab[i]); free(m->vocab); }
    free(m->hbucket); free(m->layer); free(m->data);
    free(m->x); free(m->xb); free(m->q); free(m->k); free(m->v);
    free(m->ctx); free(m->att); free(m->h1);
}

int nerve_embed_dim(const nerve_embed_t *m) { return m->cfg.hidden; }

/* ── forward ─────────────────────────────────────────────────────────────── */
void nerve_embed_text(nerve_embed_t *m, const char *text, float *out)
{
    int toks[NERVE_EMB_MAXT];
    int H = m->cfg.hidden, NH = m->cfg.n_heads, hd = H / NH;
    int n = nerve_e__encode(m, text, toks, m->maxT);
    int t, l, i, j, h;
    float scale = 1.0f / (float)sqrt((double)hd), nrm = 0.0f;

    /* embeddings + LayerNorm */
    for (t = 0; t < n; t++) {
        float *xt = m->x + (long)t * H;
        const float *we = m->word + (long)toks[t] * H, *pe = m->pos + (long)t * H;
        for (i = 0; i < H; i++) xt[i] = we[i] + pe[i] + m->type[i];   /* type 0 */
        nerve_e__layernorm(xt, m->eln_w, m->eln_b, H);
    }

    for (l = 0; l < m->cfg.n_layers; l++) {
        nerve_embed_layer *Ly = &m->layer[l];
        /* Q,K,V for all tokens */
        for (t = 0; t < n; t++) {
            float *xt = m->x + (long)t * H;
            nerve_e__linear(m->q + (long)t*H, xt, Ly->qw, Ly->qb, H, H);
            nerve_e__linear(m->k + (long)t*H, xt, Ly->kw, Ly->kb, H, H);
            nerve_e__linear(m->v + (long)t*H, xt, Ly->vw, Ly->vb, H, H);
        }
        /* bidirectional attention -> ctx */
        for (t = 0; t < n; t++)
            for (h = 0; h < NH; h++) {
                float *qh = m->q + (long)t*H + h*hd;
                for (j = 0; j < n; j++) {
                    float *kh = m->k + (long)j*H + h*hd, s = 0;
                    for (i = 0; i < hd; i++) s += qh[i] * kh[i];
                    m->att[j] = s * scale;
                }
                nerve_e__softmax(m->att, n);
                {
                    float *ch = m->ctx + (long)t*H + h*hd;
                    for (i = 0; i < hd; i++) ch[i] = 0;
                    for (j = 0; j < n; j++) {
                        float *vh = m->v + (long)j*H + h*hd, a = m->att[j];
                        for (i = 0; i < hd; i++) ch[i] += a * vh[i];
                    }
                }
            }
        /* attention output + residual + LayerNorm */
        for (t = 0; t < n; t++) {
            float *xt = m->x + (long)t*H, *xb = m->xb + (long)t*H;
            nerve_e__linear(xb, m->ctx + (long)t*H, Ly->ow, Ly->ob, H, H);
            for (i = 0; i < H; i++) xt[i] += xb[i];
            nerve_e__layernorm(xt, Ly->aln_w, Ly->aln_b, H);
        }
        /* FFN + residual + LayerNorm */
        for (t = 0; t < n; t++) {
            float *xt = m->x + (long)t*H, *xb = m->xb + (long)t*H;
            nerve_e__linear(m->h1, xt, Ly->iw, Ly->ib, H, m->cfg.intermediate);
            for (i = 0; i < m->cfg.intermediate; i++) m->h1[i] = nerve_e__gelu(m->h1[i]);
            nerve_e__linear(xb, m->h1, Ly->dw, Ly->db, m->cfg.intermediate, H);
            for (i = 0; i < H; i++) xt[i] += xb[i];
            nerve_e__layernorm(xt, Ly->oln_w, Ly->oln_b, H);
        }
    }

    /* mean-pool over all tokens, then L2-normalise */
    for (i = 0; i < H; i++) out[i] = 0;
    for (t = 0; t < n; t++) { float *xt = m->x + (long)t*H; for (i = 0; i < H; i++) out[i] += xt[i]; }
    for (i = 0; i < H; i++) out[i] /= (float)n;
    for (i = 0; i < H; i++) nrm += out[i] * out[i];
    nrm = 1.0f / ((float)sqrt((double)nrm) + 1e-12f);
    for (i = 0; i < H; i++) out[i] *= nrm;
}

#endif /* NERVE_EMBED_IMPLEMENTATION */
