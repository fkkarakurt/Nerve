/*
 * search.c — PROOF: private, local SEMANTIC search. Nerve understands *meaning*,
 * not keywords. Each note is turned into a vector by the model; a query is
 * matched to the closest notes by cosine similarity — entirely on your machine,
 * nothing leaving it. This is the core of "ask your own notes" / local RAG, with
 * no cloud, no embeddings API, no per-call bill.
 *
 * Build (from studies/infer):
 *   gcc -O3 -march=native -funroll-loops search.c -o search -lm
 * Run:
 *   NERVE_MODEL=tinyllama_q8.nrv ./search          # built-in demo queries
 *   echo "how do plants make food?" | NERVE_MODEL=tinyllama_q8.nrv ./search -i
 */
#define NERVE_INFER_IMPLEMENTATION
#include "nerve_infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* A small personal "knowledge base". */
static const char *NOTES[] = {
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Plants make their own food from sunlight through photosynthesis.",
    "Python is a popular programming language for data science.",
    "Regular aerobic exercise strengthens the heart and lungs.",
    "The Great Wall of China is over thirteen thousand kilometres long.",
    "A balanced budget means spending no more than your income.",
    "Honey never spoils if it is stored properly in a sealed jar."
};
#define NNOTES ((int)(sizeof(NOTES)/sizeof(NOTES[0])))

/* Queries phrased DIFFERENTLY from the notes — to show it matches meaning. */
static const char *QUERIES[] = {
    "Which city is the French capital?",
    "At what temperature does water turn into steam?",
    "How do plants get their energy?",
    "What language should I learn for machine learning?",
    "How can I keep my heart healthy?",
    "How should I manage my money sensibly?"
};
#define NQ ((int)(sizeof(QUERIES)/sizeof(QUERIES[0])))

/* Encode text -> mean-pooled (raw) sentence embedding. */
static void embed(nerve_transformer *m, nerve_tokenizer *tk, const char *text,
                  float *out, int dim)
{
    int toks[1024];
    int n = nerve_tokenizer_encode(tk, text, 1, 0, toks), p, i;
    for (i = 0; i < dim; i++) out[i] = 0.0f;
    for (p = 0; p < n; p++) {
        float *h = (nerve_infer_forward(m, toks[p], p), nerve_infer_hidden(m));
        for (i = 0; i < dim; i++) out[i] += h[i];          /* mean-pool */
    }
    for (i = 0; i < dim; i++) out[i] /= (float)(n > 0 ? n : 1);
}

/* Subtract a reference vector then L2-normalise (removes the dominant common
 * component — the well-known anisotropy fix for language-model embeddings). */
static void center_norm(float *v, const float *mu, int dim)
{
    int i; float nrm = 0.0f;
    for (i = 0; i < dim; i++) v[i] -= mu[i];
    for (i = 0; i < dim; i++) nrm += v[i] * v[i];
    nrm = 1.0f / ((float)sqrt((double)nrm) + 1e-8f);
    for (i = 0; i < dim; i++) v[i] *= nrm;
}

static float cosine(const float *a, const float *b, int dim)
{ int i; float s = 0.0f; for (i = 0; i < dim; i++) s += a[i] * b[i]; return s; }

int main(int argc, char **argv)
{
    nerve_transformer m;
    nerve_tokenizer   tk;
    const char *mp = getenv("NERVE_MODEL"); if (!mp) mp = "model.nrv";
    int dim, i, j, q, interactive = (argc > 1 && strcmp(argv[1], "-i") == 0), rc;
    float *DB, *qv, *mu;

    if ((rc = nerve_infer_load(&m, mp)))            { fprintf(stderr,"model %d\n",rc); return 1; }
    if ((rc = nerve_tokenizer_load(&tk, "nerve.tok"))) { fprintf(stderr,"tok %d\n",rc); return 1; }
    dim = m.config.dim;

    printf("Indexing %d notes with %s (dim=%d)...\n\n", NNOTES, mp, dim);
    DB = (float *)malloc((size_t)NNOTES * dim * sizeof(float));
    qv = (float *)malloc((size_t)dim * sizeof(float));
    mu = (float *)calloc((size_t)dim, sizeof(float));
    for (i = 0; i < NNOTES; i++) embed(&m, &tk, NOTES[i], DB + (long)i*dim, dim);

    /* reference = mean of the note embeddings; center + normalise everything */
    for (i = 0; i < NNOTES; i++) for (j = 0; j < dim; j++) mu[j] += DB[(long)i*dim + j];
    for (j = 0; j < dim; j++) mu[j] /= (float)NNOTES;
    for (i = 0; i < NNOTES; i++) center_norm(DB + (long)i*dim, mu, dim);

    if (!interactive) {
        for (q = 0; q < NQ; q++) {
            int best = 0, second = 0; float bs = -2, ss = -2;
            embed(&m, &tk, QUERIES[q], qv, dim);
            center_norm(qv, mu, dim);
            for (i = 0; i < NNOTES; i++) {
                float c = cosine(qv, DB + (long)i*dim, dim);
                if (c > bs) { ss=bs; second=best; bs=c; best=i; }
                else if (c > ss) { ss=c; second=i; }
            }
            printf("Q: %s\n", QUERIES[q]);
            printf("   -> %.3f  %s\n", bs, NOTES[best]);
            printf("      %.3f  %s\n\n", ss, NOTES[second]);
        }
    } else {
        char line[1024];
        printf("Type a query (ctrl-D to quit):\n");
        while (fgets(line, sizeof(line), stdin)) {
            int best = 0; float bs = -2;
            size_t L = strlen(line); if (L && line[L-1]=='\n') line[L-1]='\0';
            if (line[0] == '\0') continue;
            embed(&m, &tk, line, qv, dim);
            center_norm(qv, mu, dim);
            for (i = 0; i < NNOTES; i++) {
                float c = cosine(qv, DB + (long)i*dim, dim);
                if (c > bs) { bs = c; best = i; }
            }
            printf("   -> %.3f  %s\n", bs, NOTES[best]);
        }
    }

    free(DB); free(qv); free(mu);
    nerve_tokenizer_free(&tk); nerve_infer_free(&m);
    return 0;
}
