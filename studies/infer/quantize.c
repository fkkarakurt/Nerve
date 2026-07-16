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
 * quantize.c — convert a float32 Nerve model into an int8 Nerve model.
 *
 * Per-row symmetric quantization: every output row of every big matrix gets one
 * float scale, and its weights become signed bytes (-127..127). This cuts the
 * weight memory ~4x — and since CPU inference is bandwidth-bound, that is the
 * lever that lets useful models run on modest hardware. RMSNorm vectors stay
 * float (tiny, precision-sensitive).
 *
 * Build:  gcc -O2 quantize.c -o quantize -lm
 * Run:    ./quantize            (reads model.nrv, writes model_q8.nrv)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HDR 64

/* quantize R rows of C cols. Layout: ALL R float scales first (contiguous),
 * THEN all R*C int8 weights — exactly what the loader maps. */
static void write_q(FILE *o, const float *W, long R, long C)
{
    long r, j;
    float       *scales = (float *)calloc((size_t)R, sizeof(float));
    signed char *q      = (signed char *)malloc((size_t)C);

    for (r = 0; r < R; r++) {
        const float *row = W + r * C;
        float amax = 0.0f;
        for (j = 0; j < C; j++) { float a = (float)fabs(row[j]); if (a > amax) amax = a; }
        scales[r] = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    }
    fwrite(scales, sizeof(float), (size_t)R, o);       /* all scales first */

    for (r = 0; r < R; r++) {
        const float *row = W + r * C;
        for (j = 0; j < C; j++) {
            int v = (int)lroundf(row[j] / scales[r]);
            if (v < -127) v = -127;
            if (v >  127) v =  127;
            q[j] = (signed char)v;
        }
        fwrite(q, 1, (size_t)C, o);                    /* then all int8 rows */
    }
    free(scales); free(q);
}

int main(int argc, char **argv)
{
    const char *in  = (argc > 1) ? argv[1] : "model.nrv";
    const char *out = (argc > 2) ? argv[2] : "model_q8.nrv";
    FILE *f, *o;
    char  magic[4];
    int   version, dim, hid, L, heads, kvh, voc, seq, flags, reserved = 0;
    float rope;
    long  hs, kvd, off, nfloats;
    float *W;
    int   shared;

    f = fopen(in, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", in); return 1; }
    fread(magic, 1, 4, f);
    if (memcmp(magic, "NRV1", 4) != 0) { fprintf(stderr, "not a NRV1 model\n"); return 1; }
    fread(&version, 4, 1, f);
    fread(&dim, 4, 1, f); fread(&hid, 4, 1, f); fread(&L, 4, 1, f);
    fread(&heads, 4, 1, f); fread(&kvh, 4, 1, f); fread(&voc, 4, 1, f);
    fread(&seq, 4, 1, f); fread(&flags, 4, 1, f); fread(&rope, 4, 1, f);
    shared = flags & 1;
    if (flags & 2) { fprintf(stderr, "already quantized\n"); return 1; }
    hs = dim / heads; kvd = (long)kvh * hs;

    /* read the whole float weight blob */
    fseek(f, 0, SEEK_END);
    nfloats = (ftell(f) - HDR) / (long)sizeof(float);
    fseek(f, HDR, SEEK_SET);
    W = (float *)malloc((size_t)nfloats * sizeof(float));
    if (fread(W, sizeof(float), (size_t)nfloats, f) != (size_t)nfloats) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    /* float-model segment pointers (same canonical order as the loader) */
    {
        float *emb, *rms_att, *wq, *wk, *wv, *wo, *rms_ffn, *w1, *w2, *w3, *rms_final, *wcls;
        off = 0;
        emb = W + off;       off += (long)voc * dim;
        rms_att = W + off;   off += (long)L * dim;
        wq = W + off;        off += (long)L * dim * dim;
        wk = W + off;        off += (long)L * dim * kvd;
        wv = W + off;        off += (long)L * dim * kvd;
        wo = W + off;        off += (long)L * dim * dim;
        rms_ffn = W + off;   off += (long)L * dim;
        w1 = W + off;        off += (long)L * dim * hid;
        w2 = W + off;        off += (long)L * hid * dim;
        w3 = W + off;        off += (long)L * dim * hid;
        rms_final = W + off; off += dim;
        wcls = shared ? emb : (W + off);

        o = fopen(out, "wb");
        if (!o) { fprintf(stderr, "cannot write %s\n", out); return 1; }
        flags |= 2;                                  /* mark quantized */
        {
            char pad[HDR]; memset(pad, 0, sizeof pad);
            fwrite("NRV1", 1, 4, o);
            fwrite(&version, 4, 1, o);
            fwrite(&dim, 4, 1, o); fwrite(&hid, 4, 1, o); fwrite(&L, 4, 1, o);
            fwrite(&heads, 4, 1, o); fwrite(&kvh, 4, 1, o); fwrite(&voc, 4, 1, o);
            fwrite(&seq, 4, 1, o); fwrite(&flags, 4, 1, o); fwrite(&rope, 4, 1, o);
            fwrite(&reserved, 4, 1, o);
            fwrite(pad, 1, HDR - 48, o);
        }
        /* weights, in the loader's expected order */
        write_q(o, emb, voc, dim);
        fwrite(rms_att, sizeof(float), (size_t)((long)L * dim), o);
        write_q(o, wq, (long)L * dim, dim);
        write_q(o, wk, (long)L * kvd, dim);
        write_q(o, wv, (long)L * kvd, dim);
        write_q(o, wo, (long)L * dim, dim);
        fwrite(rms_ffn, sizeof(float), (size_t)((long)L * dim), o);
        write_q(o, w1, (long)L * hid, dim);
        write_q(o, w2, (long)L * dim, hid);
        write_q(o, w3, (long)L * hid, dim);
        fwrite(rms_final, sizeof(float), (size_t)dim, o);
        if (!shared) write_q(o, wcls, voc, dim);
        fclose(o);
    }
    free(W);
    printf("wrote %s  (int8, per-row; was float32)\n", out);
    return 0;
}
