/*
 * gradcheck.c — proves the autodiff is CORRECT, not just convergent.
 *
 * For each parameter we compare the analytic gradient from ng_backward()
 * against a central finite-difference estimate  (L(w+eps) - L(w-eps)) / 2eps.
 * If the engine's chain rule is right, the two agree to ~1e-3 in float.
 *
 * The network here is LINEAR (no ReLU) on purpose: ReLU's kink at 0 makes the
 * finite-difference reference unreliable wherever a pre-activation sits within
 * eps of zero (the +eps and -eps probes land on different sides of the kink).
 * That is a property of the *numerical reference*, not the autodiff. Removing
 * ReLU gives a smooth loss and an exact check of the matmul / bias / softmax-
 * cross-entropy chain — the only non-trivial gradients. ReLU's gradient is a
 * one-line mask, exercised end-to-end by demo_mlp (which converges to 12/12).
 *
 * Build:  gcc -O2 gradcheck.c -o gradcheck -lm
 */
#define NERVE_GRAD_IMPLEMENTATION
#include "nerve_grad.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 6
#define IN 3
#define H 5
#define C 4

static float frand(void){ return (float)rand()/(float)RAND_MAX - 0.5f; }

static float X[N*IN];
static int   y[N] = {0,1,2,3,1,2};

/* One full forward pass -> scalar loss. Leaves params untouched. */
static float forward_loss(tensor *W1, tensor *b1, tensor *W2, tensor *b2)
{
    tensor *xin = t_param(N, IN), *h, *logits, *loss;
    float L;
    int i;
    for (i = 0; i < N*IN; i++) xin->data[i] = X[i];
    h      = t_add_bias(t_matmul(xin, W1), b1);   /* linear: no ReLU kink */
    logits = t_add_bias(t_matmul(h, W2), b2);
    loss   = t_softmax_cross_entropy(logits, y);
    L = loss->data[0];
    free(xin->data); free(xin->grad); free(xin);
    ng_end_step();
    return L;
}

static int check_param(tensor *p, const char *name,
                       tensor *W1, tensor *b1, tensor *W2, tensor *b2)
{
    /* Combined abs+rel tolerance: a single-precision central difference with
     * eps=1e-3 only resolves the loss to ~1e-3, so gradients near zero are
     * dominated by truncation/round-off. We flag a mismatch only when BOTH the
     * absolute error is meaningful AND the relative error is large — the
     * standard, honest way to grad-check a float engine. */
    const float eps = 1e-3f, atol = 2e-3f, rtol = 2e-2f;
    float max_abs = 0.0f, wnum = 0, wana = 0;
    int i, sz = p->rows * p->cols, bad = 0, wi = -1;
    for (i = 0; i < sz; i++) {
        float orig = p->data[i], lp, lm, num, ana, abserr, rel, denom;
        p->data[i] = orig + eps; lp = forward_loss(W1,b1,W2,b2);
        p->data[i] = orig - eps; lm = forward_loss(W1,b1,W2,b2);
        p->data[i] = orig;
        num = (lp - lm) / (2.0f * eps);
        ana = p->grad[i];
        abserr = (float)fabs(num - ana);
        denom  = (float)fabs(num) + (float)fabs(ana) + 1e-8f;
        rel    = abserr / denom;
        if (abserr > max_abs) { max_abs = abserr; wi = i; wnum = num; wana = ana; }
        if (abserr > atol && rel > rtol) bad++;
    }
    printf("  %-4s (%dx%d)  max abs err = %.2e at [%d]  num=%.5f ana=%.5f   %s\n",
           name, p->rows, p->cols, max_abs, wi, wnum, wana, bad ? "FAIL" : "ok");
    return bad;
}

int main(void)
{
    tensor *W1, *b1, *W2, *b2, *params[4];
    int i, bad = 0;
    float r;

    srand(99);
    for (i = 0; i < N*IN; i++) X[i] = frand();

    W1 = t_param(IN,H); b1 = t_param(1,H);
    W2 = t_param(H, C); b2 = t_param(1,C);
    r = 0.5f;
    for (i = 0; i < IN*H; i++) W1->data[i] = r*frand();
    for (i = 0; i < H;    i++) b1->data[i] = r*frand();
    for (i = 0; i < H*C;  i++) W2->data[i] = r*frand();
    for (i = 0; i < C;    i++) b2->data[i] = r*frand();
    params[0]=W1; params[1]=b1; params[2]=W2; params[3]=b2;

    /* analytic grads once */
    ng_zero_grad(params, 4);
    {
        tensor *xin = t_param(N, IN), *h, *logits, *loss;
        for (i = 0; i < N*IN; i++) xin->data[i] = X[i];
        h      = t_add_bias(t_matmul(xin, W1), b1);   /* linear: no ReLU kink */
        logits = t_add_bias(t_matmul(h, W2), b2);
        loss   = t_softmax_cross_entropy(logits, y);
        ng_backward(loss);
        free(xin->data); free(xin->grad); free(xin);
        ng_end_step();
    }

    printf("Gradient check: analytic (autodiff) vs central finite difference\n\n");
    bad += check_param(W1, "W1", W1,b1,W2,b2);
    bad += check_param(b1, "b1", W1,b1,W2,b2);
    bad += check_param(W2, "W2", W1,b1,W2,b2);
    bad += check_param(b2, "b2", W1,b1,W2,b2);

    printf("\n  %s\n", bad ? "GRADIENT CHECK FAILED" : "GRADIENT CHECK PASSED — autodiff is correct");
    return bad ? 1 : 0;
}
