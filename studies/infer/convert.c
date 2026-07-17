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
 * convert.c — one-time import tool: external weights -> Nerve's native format.
 *
 * This utility is NOT part of the Nerve library. It exists only to bring
 * externally trained weights INTO Nerve's own self-describing format, so the
 * runtime library (nerve_infer.h) depends on no one else's file layout. The
 * weights themselves are user-supplied data ("bring your own model"); Nerve
 * owns the engine and the format, not the data.
 *
 * It reads a flat float32 weight dump (Config header of 7 int32, then weights
 * in the standard decoder-only-Transformer order) plus a SentencePiece-style
 * vocab dump, and writes:
 *     model.nrv   — Nerve native model   (magic "NRV1")
 *     nerve.tok   — Nerve native vocab    (magic "NTK1")
 *
 * Build:  gcc -O2 convert.c -o convert
 * Run:    ./convert <weights.bin> <vocab.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NRV_MAGIC "NRV1"
#define NTK_MAGIC "NTK1"
#define NRV_HEADER_BYTES 64

static long seg(long count) { return count; }   /* element count helper */

int main(int argc, char **argv)
{
    const char *win = (argc > 1) ? argv[1] : "stories15M.bin";
    const char *tin = (argc > 2) ? argv[2] : "tokenizer.bin";
    FILE *f, *o;
    int   cfg[7], shared, i;
    int   dim, hidden, layers, heads, kv_heads, vocab, seq_len, head_size;
    float rope_theta = 10000.0f;
    long  n_emb, n_attn_w, n_q, n_kv, n_ffn_w, n_w13, n_w2, total_floats, fpos;
    float *W;
    int   flags, version = 1, reserved = 0;

    /* ---- read external weight dump ---- */
    f = fopen(win, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", win); return 1; }
    if (fread(cfg, sizeof(int), 7, f) != 7) { fprintf(stderr, "bad header\n"); return 1; }
    dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
    kv_heads = cfg[4]; vocab = cfg[5]; seq_len = cfg[6];
    shared = vocab > 0 ? 1 : 0; if (vocab < 0) vocab = -vocab;
    head_size = dim / heads;

    /* element counts of each weight segment (canonical Transformer order) */
    n_emb    = (long)vocab * dim;
    n_attn_w = (long)layers * dim;                       /* rms_att   */
    n_q      = (long)layers * dim * (heads * head_size);
    n_kv     = (long)layers * dim * (kv_heads * head_size);
    n_ffn_w  = (long)layers * dim;                       /* rms_ffn   */
    n_w13    = (long)layers * dim * hidden;
    n_w2     = (long)layers * hidden * dim;

    /* everything Nerve keeps (note: we DROP the legacy precomputed RoPE tables —
     * Nerve computes RoPE on the fly, so our format is smaller and cleaner) */
    total_floats =
        n_emb + n_attn_w + n_q + n_kv + n_kv +
        (long)layers * (heads * head_size) * dim +       /* wo        */
        n_ffn_w + n_w13 + n_w2 + n_w13 + dim +           /* w1 w2 w3 rms_final */
        (shared ? 0 : n_emb);                            /* wcls if not shared */

    W = (float *)malloc((size_t)total_floats * sizeof(float));
    if (!W) { fprintf(stderr, "oom\n"); return 1; }

    fpos = 0;
    /* read the segments we keep, in order, skipping the freq tables */
    {
        long keep_before_freq =
            n_emb + n_attn_w + n_q + n_kv + n_kv +
            (long)layers * (heads * head_size) * dim +
            n_ffn_w + n_w13 + n_w2 + n_w13 + dim;
        if (fread(W, sizeof(float), (size_t)keep_before_freq, f) != (size_t)keep_before_freq) {
            fprintf(stderr, "short read (body)\n"); return 1;
        }
        fpos = keep_before_freq;
        /* skip legacy freq_cis_real + freq_cis_imag */
        fseek(f, (long)(seq_len * head_size / 2) * 2 * (long)sizeof(float), SEEK_CUR);
        if (!shared) {
            if (fread(W + fpos, sizeof(float), (size_t)n_emb, f) != (size_t)n_emb) {
                fprintf(stderr, "short read (wcls)\n"); return 1;
            }
            fpos += n_emb;
        }
    }
    fclose(f);
    (void)seg;

    /* ---- write Nerve native model ---- */
    o = fopen("model.nrv", "wb");
    if (!o) { fprintf(stderr, "cannot write model.nrv\n"); return 1; }
    flags = shared ? 1 : 0;
    {
        char pad[NRV_HEADER_BYTES];
        memset(pad, 0, sizeof pad);
        fwrite(NRV_MAGIC, 1, 4, o);
        fwrite(&version, sizeof(int), 1, o);
        fwrite(&dim, sizeof(int), 1, o);
        fwrite(&hidden, sizeof(int), 1, o);
        fwrite(&layers, sizeof(int), 1, o);
        fwrite(&heads, sizeof(int), 1, o);
        fwrite(&kv_heads, sizeof(int), 1, o);
        fwrite(&vocab, sizeof(int), 1, o);
        fwrite(&seq_len, sizeof(int), 1, o);
        fwrite(&flags, sizeof(int), 1, o);
        fwrite(&rope_theta, sizeof(float), 1, o);
        fwrite(&reserved, sizeof(int), 1, o);     /* [44..47] */
        /* pad to 64 bytes */
        fwrite(pad, 1, NRV_HEADER_BYTES - 48, o); /* 48 bytes written so far */
    }
    fwrite(W, sizeof(float), (size_t)fpos, o);
    fclose(o);
    free(W);
    printf("wrote model.nrv  (%d-dim, %d layers, %d heads, vocab %d, ctx %d, %ld floats)\n",
           dim, layers, heads, vocab, seq_len, fpos);

    /* ---- convert vocab to Nerve native tokenizer ---- */
    f = fopen(tin, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", tin); return 1; }
    o = fopen("nerve.tok", "wb");
    if (!o) { fprintf(stderr, "cannot write nerve.tok\n"); return 1; }
    {
        int max_len, len;
        float score;
        char *sbuf;
        if (fread(&max_len, sizeof(int), 1, f) != 1) { fprintf(stderr, "bad tokenizer header\n"); return 1; }
        fwrite(NTK_MAGIC, 1, 4, o);
        fwrite(&version, sizeof(int), 1, o);
        fwrite(&vocab, sizeof(int), 1, o);
        fwrite(&max_len, sizeof(int), 1, o);
        sbuf = (char *)malloc((size_t)max_len + 1);
        for (i = 0; i < vocab; i++) {
            if (fread(&score, sizeof(float), 1, f) != 1 ||
                fread(&len, sizeof(int), 1, f) != 1) { fprintf(stderr, "bad vocab\n"); return 1; }
            if (fread(sbuf, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "bad vocab\n"); return 1; }
            fwrite(&score, sizeof(float), 1, o);
            fwrite(&len, sizeof(int), 1, o);
            fwrite(sbuf, 1, (size_t)len, o);
        }
        free(sbuf);
    }
    fclose(f); fclose(o);
    printf("wrote nerve.tok  (%d tokens)\n", vocab);
    return 0;
}
