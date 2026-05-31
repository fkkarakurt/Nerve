/*
 * learn.c — PROOF: Nerve doesn't just run a model, it LEARNS on-device.
 *
 * The big base model (TinyLlama, frozen) is used only as a feature extractor:
 * we take its final hidden state as a representation of a sentence. Then a tiny
 * head — a single linear layer trained with Nerve's own autodiff engine
 * (nerve_grad.h) — learns YOUR personal categories from a handful of your own
 * examples, entirely on your device. No data leaves the machine, no cloud, no
 * tokens, no GPU. This is the per-node capability the whole Nerve vision is
 * built on: a model that adapts to you.
 *
 * Build (from studies/infer):
 *   gcc -O3 -march=native -funroll-loops learn.c -o learn -lm
 * Run:
 *   NERVE_MODEL=tinyllama_q8.nrv ./learn      (model.nrv default; nerve.tok)
 */
#define NERVE_INFER_IMPLEMENTATION
#include "nerve_infer.h"
#define NERVE_GRAD_IMPLEMENTATION
#include "../autograd/nerve_grad.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCLS 3

/* A tiny PERSONAL dataset: short sentences in three categories the user cares
 * about. (Imagine these came from the user's own notes.) */
static const char *train_txt[] = {
    "Schedule a meeting with the team tomorrow morning",
    "Move my dentist appointment to next Tuesday",
    "Remind me about the project deadline on Friday",
    "Block two hours on my calendar for deep work",
    "Add tomatoes and olive oil to the shopping list",
    "How long should I roast the chicken for dinner",
    "I want to try a new pasta recipe tonight",
    "We are out of coffee and milk again",
    "I went for a five kilometer run this morning",
    "My legs are sore after yesterday's workout",
    "Plan a gym session focused on upper body",
    "I hit a new personal best on the bench press"
};
static const int train_y[] = { 0,0,0,0, 1,1,1,1, 2,2,2,2 };
#define NTRAIN 12

static const char *test_txt[] = {
    "Set up a call with the client next week",       /* 0 calendar */
    "Put eggs and bread on the grocery list",        /* 1 food     */
    "I am going cycling for an hour after work",      /* 2 fitness  */
    "Reschedule the standup to the afternoon",        /* 0 calendar */
    "What temperature should the oven be for bread",  /* 1 food     */
    "Time for some stretching and a light jog"        /* 2 fitness  */
};
static const int test_y[] = { 0, 1, 2, 0, 1, 2 };
#define NTEST 6

static const char *LABELS[NCLS] = { "calendar", "food", "fitness" };

/* Encode a sentence -> the frozen base model's final hidden state, L2-normalised. */
static void embed(nerve_transformer *m, nerve_tokenizer *tk, const char *text,
                  float *out, int dim)
{
    int toks[512];
    int n = nerve_tokenizer_encode(tk, text, 1 /*bos*/, 0, toks), p, i;
    float *h, nrm = 0.0f;
    for (p = 0; p < n; p++) nerve_infer_forward(m, toks[p], p);  /* run the base */
    h = nerve_infer_hidden(m);                                   /* last-token state */
    for (i = 0; i < dim; i++) nrm += h[i] * h[i];
    nrm = 1.0f / ((float)sqrt((double)nrm) + 1e-8f);
    for (i = 0; i < dim; i++) out[i] = h[i] * nrm;
}

static float frand(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }

/* head prediction for one feature vector using current W,b */
static int predict(const float *x, tensor *W, tensor *b, int dim)
{
    int c, j, best = 0;
    float logit[NCLS];
    for (c = 0; c < NCLS; c++) {
        float v = b->data[c];
        for (j = 0; j < dim; j++) v += W->data[j * NCLS + c] * x[j];
        logit[c] = v;
        if (v > logit[best]) best = c;
    }
    return best;
}

static int accuracy(float *X, const int *y, int n, tensor *W, tensor *b, int dim)
{
    int i, ok = 0;
    for (i = 0; i < n; i++) ok += (predict(X + (long)i * dim, W, b, dim) == y[i]);
    return ok;
}

int main(void)
{
    nerve_transformer m;
    nerve_tokenizer tk;
    const char *mp = getenv("NERVE_MODEL"); if (!mp) mp = "model.nrv";
    int dim, i, e, rc;
    float *Xtr, *Xte;
    tensor *W, *b, *params[2];

    if ((rc = nerve_infer_load(&m, mp))) { fprintf(stderr, "model load %d\n", rc); return 1; }
    if ((rc = nerve_tokenizer_load(&tk, "nerve.tok"))) { fprintf(stderr, "tok %d\n", rc); return 1; }
    dim = m.config.dim;
    printf("Base model: %s  dim=%d layers=%d  (FROZEN feature extractor)\n",
           mp, dim, m.config.n_layers);

    /* --- extract features once (the slow part: runs the base model) --- */
    printf("Extracting features for %d train + %d test sentences...\n", NTRAIN, NTEST);
    Xtr = (float *)malloc((size_t)NTRAIN * dim * sizeof(float));
    Xte = (float *)malloc((size_t)NTEST  * dim * sizeof(float));
    for (i = 0; i < NTRAIN; i++) embed(&m, &tk, train_txt[i], Xtr + (long)i * dim, dim);
    for (i = 0; i < NTEST;  i++) embed(&m, &tk, test_txt[i],  Xte + (long)i * dim, dim);

    /* --- the trainable head (single linear layer, our autodiff) --- */
    srand(1);
    W = t_param(dim, NCLS); b = t_param(1, NCLS);
    for (i = 0; i < dim * NCLS; i++) W->data[i] = 0.05f * frand();
    params[0] = W; params[1] = b;

    printf("\nBEFORE training (random head):  test accuracy = %d/%d\n",
           accuracy(Xte, test_y, NTEST, W, b, dim), NTEST);

    /* --- train the head on YOUR examples, on-device --- */
    for (e = 1; e <= 300; e++) {
        tensor *xin, *logits, *loss;
        ng_zero_grad(params, 2);
        xin = t_param(NTRAIN, dim);
        memcpy(xin->data, Xtr, (size_t)NTRAIN * dim * sizeof(float));
        logits = t_add_bias(t_matmul(xin, W), b);
        loss   = t_softmax_cross_entropy(logits, train_y);
        ng_backward(loss);
        ng_sgd_step(params, 2, 0.2f);
        if (e % 100 == 0) printf("  epoch %3d  loss=%.4f\n", e, loss->data[0]);
        free(xin->data); free(xin->grad); free(xin);
        ng_end_step();
    }

    printf("\nAFTER training (learned from %d of your examples):\n", NTRAIN);
    for (i = 0; i < NTEST; i++) {
        int pred = predict(Xte + (long)i * dim, W, b, dim);
        printf("  [%-8s] (true %-8s) \"%s\"\n",
               LABELS[pred], LABELS[test_y[i]], test_txt[i]);
    }
    printf("\ntest accuracy = %d/%d   <- Nerve learned your categories on-device\n",
           accuracy(Xte, test_y, NTEST, W, b, dim), NTEST);

    free(Xtr); free(Xte);
    nerve_tokenizer_free(&tk);
    nerve_infer_free(&m);
    return 0;
}
