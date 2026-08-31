/* nerve_discover.h - symbolic regression / equation discovery in one header.
 *
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Give it data, get back a compact human-readable EQUATION. No Python, no
 * dependencies (only libm), no build system - drop this file in and #include it.
 * Runs anywhere C compiles: a laptop, a browser (WebAssembly), an old machine.
 *
 * Where a neural network answers "what is y?" with a million opaque weights,
 * this answers with a formula you can read, check against theory, differentiate
 * by hand, put in a paper, and evaluate in nanoseconds forever after.
 *
 * HOW IT WORKS
 *   Genetic programming over expression trees, with the pieces that make the
 *   difference between a toy and an engine:
 *
 *   - Linear scaling (Keijzer 2003). Every candidate f is scored as its best
 *     a*f(x)+b, solved in closed form. Structure is searched; the two constants
 *     that usually dominate the error are never searched for at all.
 *   - An island model. Sub-populations evolve independently and exchange
 *     migrants, which is what keeps a GP run from collapsing onto one lineage.
 *     Islands also carry different parsimony pressures, so some hunt accuracy
 *     and others hunt simplicity - and between them they populate the front.
 *   - A Pareto front, not one answer. The result is the best equation found at
 *     every complexity, so you can see the accuracy you buy with each extra
 *     node and pick the knee yourself. A law should look like a law.
 *   - Algebraic simplification. x*1, x+0, x-x and constant subtrees are folded
 *     away, which both shortens the answer and stops the bloat that otherwise
 *     eats a GP run alive.
 *   - Derivative-free constant polish (compass search), which never accepts a
 *     step that makes the fit worse, so it recovers real constants without the
 *     divergence a fixed-step gradient descent suffers on stiff expressions.
 *   - A flattened postfix evaluator. Trees are compiled to a linear instruction
 *     array once per fitness evaluation and run over the rows with an explicit
 *     stack - the inner loop is contiguous memory, not pointer chasing.
 *
 * Part of the Nerve family of single-header libraries (see nerve.h).
 *
 * ---------------------------------------------------------------------------
 * USAGE:  in exactly ONE .c file, define the implementation before including:
 *
 *     #define NERVE_DISCOVER_IMPLEMENTATION
 *     #include "nerve_discover.h"
 *
 *     double X[N*NV], y[N];              // your data, X row-major
 *     char *names[NV] = { "m1", "m2", "r" };
 *
 *     nd_options o = nd_defaults();
 *     nd_model  *m = nd_fit(X, y, N, NV, o);
 *
 *     nd_print(nd_knee(m, 0.05), names, stdout);  // the one you want
 *     nd_print(nd_best(m), names, stdout);        // the most accurate one
 *
 *     for (int i = 0; i < nd_count(m); i++) {     // the whole Pareto front
 *         const nd_expr *e = nd_at(m, i);
 *         printf("%2d nodes  R2=%.5f  ", nd_complexity(e),
 *                nd_r2(e, Xtest, ytest, NTEST, NV));
 *         nd_print(e, names, stdout); putchar('\n');
 *     }
 *     nd_free(m);
 *
 * Other files that need the API just #include "nerve_discover.h" (no macro).
 * Build:  cc -O2 -std=c99 yourfile.c -lm
 *
 * License: Apache-2.0 (same as the rest of Nerve).
 * ------------------------------------------------------------------------- */

#ifndef NERVE_DISCOVER_H
#define NERVE_DISCOVER_H

#include <stdio.h>

#define ND_VERSION_MAJOR 1
#define ND_VERSION_MINOR 0
#define ND_VERSION_PATCH 0
#define ND_VERSION       "1.0.0"

#ifdef __cplusplus
extern "C" {
#endif

/* A discovered equation: an expression tree plus the linear scale a*f(x)+b
 * that was solved for it. Opaque - owned by the nd_model it came from. */
typedef struct nd_expr  nd_expr;

/* The result of a search: a Pareto front of equations, ascending in
 * complexity and strictly improving in accuracy. */
typedef struct nd_model nd_model;

/* ---- which operators the search may use -------------------------------- */
/* Restricting the set is the single most effective way to steer a search.
 * An equation meant for a design code should probably stay algebraic; a
 * wave or a decay needs sin or exp. Fewer operators, smaller haystack. */
enum {
    ND_OP_ADD  = 1u << 0,   /* a + b                                       */
    ND_OP_SUB  = 1u << 1,   /* a - b                                       */
    ND_OP_MUL  = 1u << 2,   /* a * b                                       */
    ND_OP_DIV  = 1u << 3,   /* a / b   (protected: |b| ~ 0 yields 1)        */
    ND_OP_NEG  = 1u << 4,   /* -a                                          */
    ND_OP_INV  = 1u << 5,   /* 1 / a   (protected)                         */
    ND_OP_SQR  = 1u << 6,   /* a^2     (one node, not three)               */
    ND_OP_SQRT = 1u << 7,   /* sqrt(|a|)                                   */
    ND_OP_EXP  = 1u << 8,   /* exp(a)  (argument clamped)                  */
    ND_OP_LOG  = 1u << 9,   /* log(|a|)                                    */
    ND_OP_SIN  = 1u << 10,  /* sin(a)                                      */
    ND_OP_COS  = 1u << 11,  /* cos(a)                                      */
    ND_OP_TANH = 1u << 12   /* tanh(a)                                     */
};

/* Ready-made sets. */
#define ND_OPS_ARITH   (ND_OP_ADD | ND_OP_SUB | ND_OP_MUL | ND_OP_DIV)
#define ND_OPS_ALGEBRA (ND_OPS_ARITH | ND_OP_INV | ND_OP_SQR | ND_OP_SQRT)
#define ND_OPS_DEFAULT (ND_OPS_ALGEBRA | ND_OP_SIN | ND_OP_COS | ND_OP_EXP | ND_OP_LOG)
#define ND_OPS_ALL     (ND_OPS_DEFAULT | ND_OP_NEG | ND_OP_TANH)

/* ---- search settings --------------------------------------------------- */
typedef struct {
    int          population;   /* candidates per island              (400)   */
    int          islands;      /* independent sub-populations          (5)   */
    int          generations;  /* search iterations                  (300)   */
    int          migration;    /* migrate every N generations, 0=off  (25)   */
    int          max_nodes;    /* complexity ceiling                  (28)   */
    double       parsimony;    /* size penalty, in units of nrmse  (0.002)   */
    unsigned int ops;          /* operator set        (ND_OPS_DEFAULT)       */
    int          tune_iters;   /* constant-polish effort, 0=off       (80)   */
    double       tolerance;    /* stop early once nrmse <= this     (1e-9)   */
    unsigned long long seed;   /* deterministic: same seed, same run   (1)   */

    /* Called once per generation with the best equation so far. Return
     * non-zero to stop the search early. `best` is borrowed - do not free it,
     * and do not keep it past the callback. Both may be NULL. */
    int (*progress)(int gen, int generations, double nrmse,
                    const nd_expr *best, void *user);
    void *user;
} nd_options;

/* Sensible defaults; override fields as needed. */
nd_options nd_defaults(void);

/* Discover y ~ f(x0..x[nvars-1]) from n rows (X row-major: X[i*nvars + j]).
 * Returns a new nd_model (free with nd_free), or NULL on bad input / OOM. */
nd_model *nd_fit(const double *X, const double *y, int n, int nvars,
                 nd_options opt);

void nd_free(nd_model *m);

/* ---- reading the result ------------------------------------------------ */

/* How many equations are on the front. */
int             nd_count(const nd_model *m);

/* The i-th equation, 0 <= i < nd_count: ascending complexity, descending
 * training error. NULL if i is out of range. */
const nd_expr  *nd_at(const nd_model *m, int i);

/* The most accurate equation found (the last point on the front). */
const nd_expr  *nd_best(const nd_model *m);

/* The knee of the front: the SIMPLEST equation that gives up no more than
 * `slack` of the accuracy the front has to offer. The front spans some range
 * of error, from its crudest one-node entry down to its best; `slack` is the
 * fraction of that span you are willing to concede for a shorter formula.
 * 0.05 is a good default; 0 returns the most accurate entry, 1 the simplest.
 *
 * This is usually the equation you actually want. Past the knee the front is
 * buying six more nodes for a fourth decimal place, which on noisy data is
 * just memorising the noise. Being a fraction of the front's own span, it
 * needs no prior knowledge of the noise level and stays meaningful on exact
 * data, where the best error is zero and any relative tolerance collapses. */
const nd_expr  *nd_knee(const nd_model *m, double slack);

/* ---- using an equation ------------------------------------------------- */

double nd_eval(const nd_expr *e, const double *x);   /* one input row        */
double nd_r2  (const nd_expr *e, const double *X, const double *y,
               int n, int nvars);                    /* use held-out data    */
double nd_rmse(const nd_expr *e, const double *X, const double *y,
               int n, int nvars);
int    nd_complexity(const nd_expr *e);              /* nodes in the tree    */

/* Write the equation as text, snprintf-style: always NUL-terminates when
 * cap > 0, returns the length it would have written. var_names may be NULL
 * (then x0, x1, ...). */
int    nd_format(const nd_expr *e, char *const *var_names, char *buf, int cap);

/* Same, straight to a stream. */
void   nd_print(const nd_expr *e, char *const *var_names, FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* NERVE_DISCOVER_H */

/* ======================================================================== */
/*                             IMPLEMENTATION                               */
/* ======================================================================== */
#ifdef NERVE_DISCOVER_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Hard ceilings. A closed-form law that needs more than this many nodes is
 * not a law, and these let every scratch buffer be a fixed size. */
#define ND__CAP      192   /* absolute max nodes in a tree                  */
#define ND__MAXCONST  24   /* constants the polish step will touch          */
#define ND__BIG      1e30  /* anything larger is treated as a blown-up fit  */

/* ---- node types (the first two are terminals) -------------------------- */
enum {
    ND__CONST, ND__VAR,
    ND__ADD, ND__SUB, ND__MUL, ND__DIV,                    /* arity 2 */
    ND__NEG, ND__INV, ND__SQR, ND__SQRT,                   /* arity 1 */
    ND__EXP, ND__LOG, ND__SIN, ND__COS, ND__TANH
};

static int nd__arity(int t){
    if (t <= ND__VAR)  return 0;
    if (t <= ND__DIV)  return 2;
    return 1;
}

typedef struct nd__node {
    int    type;
    int    var;               /* ND__VAR: which input column               */
    double val;               /* ND__CONST: its value                      */
    struct nd__node *l, *r;
} nd__node;

/* One postfix instruction. Trees are compiled to an array of these before
 * they are evaluated, so the hot loop walks contiguous memory. */
typedef struct { int type; int var; double val; } nd__ins;

struct nd_expr {
    nd__node *tree;
    double    a, b;           /* the solved linear scale a*f(x)+b          */
    double    err;            /* training nrmse when it was archived       */
    int       nodes;
    int       nvars;
    nd__ins  *code;           /* compiled once, when the model is built    */
    int       ncode;
};

struct nd_model {
    nd_expr *front;
    int      count;
    int      nvars;
};

/* ======================================================================== */
/*  Deterministic RNG - xorshift64 with a multiply finaliser.               */
/*  State lives in the search context, so two fits never contaminate each   */
/*  other and the same seed always replays the same run.                    */
/* ======================================================================== */
typedef struct {
    unsigned long long rng;
    int          nvars;
    int          bin[4],  nbin;   /* enabled binary op types               */
    int          un [10], nun;    /* enabled unary op types                */
    double       unary_p;         /* chance a new internal node is unary   */
} nd__ctx;

static unsigned long long nd__u64(nd__ctx *c){
    unsigned long long x = c->rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    c->rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static double nd__unit(nd__ctx *c){
    return (double)(nd__u64(c) >> 11) * (1.0 / 9007199254740992.0);
}
static int nd__pick(nd__ctx *c, int n){
    return n > 0 ? (int)(nd__unit(c) * n) % n : 0;
}
static double nd__range(nd__ctx *c, double lo, double hi){
    return lo + (hi - lo) * nd__unit(c);
}
static double nd__gauss(nd__ctx *c){
    double u1 = nd__unit(c), u2 = nd__unit(c);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ======================================================================== */
/*  Trees                                                                   */
/* ======================================================================== */
static nd__node *nd__new(int t){
    nd__node *n = (nd__node *)malloc(sizeof(nd__node));
    if (!n) return NULL;
    n->type = t; n->val = 0.0; n->var = 0; n->l = n->r = NULL;
    return n;
}
static void nd__drop(nd__node *n){
    if (!n) return;
    nd__drop(n->l); nd__drop(n->r); free(n);
}
static nd__node *nd__clone(const nd__node *n){
    nd__node *m;
    if (!n) return NULL;
    m = nd__new(n->type);
    if (!m) return NULL;
    m->val = n->val; m->var = n->var;
    m->l = nd__clone(n->l); m->r = nd__clone(n->r);
    return m;
}
static int nd__size(const nd__node *n){
    return n ? 1 + nd__size(n->l) + nd__size(n->r) : 0;
}
/* Structural equality - the basis of the x-x and x/x simplifications. */
static int nd__same(const nd__node *a, const nd__node *b){
    if (a == b)             return 1;
    if (!a || !b)           return 0;
    if (a->type != b->type) return 0;
    if (a->type == ND__CONST) return a->val == b->val;
    if (a->type == ND__VAR)   return a->var == b->var;
    return nd__same(a->l, b->l) && nd__same(a->r, b->r);
}

/* A fresh leaf: a variable, or a constant drawn from a mixture that favours
 * the small integers and halves real laws are actually built from. */
static nd__node *nd__leaf(nd__ctx *c){
    nd__node *n;
    if (nd__unit(c) < 0.65){
        n = nd__new(ND__VAR);
        if (n) n->var = nd__pick(c, c->nvars);
        return n;
    }
    n = nd__new(ND__CONST);
    if (n){
        double r = nd__unit(c);
        if      (r < 0.40) n->val = (double)(1 + nd__pick(c, 4));
        else if (r < 0.55) n->val = 0.5;
        else if (r < 0.65) n->val = 2.0;
        else               n->val = nd__range(c, -3.0, 3.0);
    }
    return n;
}

/* Ramped grow. `full` forces branching down to maxd, which is what gives an
 * initial population a spread of shapes rather than a pile of stumps. */
static nd__node *nd__grow(nd__ctx *c, int depth, int maxd, int full){
    nd__node *n;
    if (depth >= maxd) return nd__leaf(c);
    if (!full && depth > 0 && nd__unit(c) < 0.30) return nd__leaf(c);

    if (c->nun > 0 && nd__unit(c) < c->unary_p){
        n = nd__new(c->un[nd__pick(c, c->nun)]);
        if (!n) return NULL;
        n->l = nd__grow(c, depth + 1, maxd, full);
        return n;
    }
    if (c->nbin == 0) return nd__leaf(c);
    n = nd__new(c->bin[nd__pick(c, c->nbin)]);
    if (!n) return NULL;
    n->l = nd__grow(c, depth + 1, maxd, full);
    n->r = nd__grow(c, depth + 1, maxd, full);
    return n;
}

/* ======================================================================== */
/*  Compiled evaluation                                                     */
/*                                                                          */
/*  A tree is flattened to postfix once per fitness evaluation and then run  */
/*  over every row from that flat array. The hot loop touches contiguous     */
/*  memory and a small stack instead of chasing child pointers n times.      */
/* ======================================================================== */
static void nd__emit(const nd__node *t, nd__ins *code, int *k){
    if (!t || *k >= ND__CAP) return;
    nd__emit(t->l, code, k);
    nd__emit(t->r, code, k);
    if (*k >= ND__CAP) return;
    code[*k].type = t->type;
    code[*k].var  = t->var;
    code[*k].val  = t->val;
    (*k)++;
}

/* Protected primitives. Every one of these is total: no domain error, no
 * NaN, no trap - a candidate that strays out of a function's domain is
 * simply a poor candidate, not a crash. */
static double nd__apply1(int type, double a){
    switch (type){
        case ND__NEG:  return -a;
        case ND__INV:  return fabs(a) < 1e-12 ? 1.0 : 1.0 / a;
        case ND__SQR:  return a * a;
        case ND__SQRT: return sqrt(fabs(a));
        case ND__EXP:  return exp(a > 40.0 ? 40.0 : (a < -40.0 ? -40.0 : a));
        case ND__LOG:  return log(fabs(a) + 1e-12);
        case ND__SIN:  return sin(a);
        case ND__COS:  return cos(a);
        default:       return tanh(a);
    }
}
static double nd__apply2(int type, double a, double b){
    switch (type){
        case ND__ADD: return a + b;
        case ND__SUB: return a - b;
        case ND__MUL: return a * b;
        default:      return fabs(b) < 1e-12 ? 1.0 : a / b;   /* ND__DIV */
    }
}

/* Run the compiled expression on one row. Returns 0 if the value blew up. */
static int nd__run(const nd__ins *code, int nc, const double *x,
                   double *stk, double *out){
    int i, sp = 0;
    for (i = 0; i < nc; i++){
        int t = code[i].type;
        if      (t == ND__CONST) stk[sp++] = code[i].val;
        else if (t == ND__VAR)   stk[sp++] = x[code[i].var];
        else if (nd__arity(t) == 1){
            if (sp < 1) return 0;
            stk[sp-1] = nd__apply1(t, stk[sp-1]);
        } else {
            if (sp < 2) return 0;
            stk[sp-2] = nd__apply2(t, stk[sp-2], stk[sp-1]);
            sp--;
        }
    }
    if (sp != 1) return 0;
    *out = stk[0];
    return (*out == *out) && *out < ND__BIG && *out > -ND__BIG;
}

/* ======================================================================== */
/*  Fitness: linear scaling (Keijzer 2003) + a parsimony pressure           */
/* ======================================================================== */
typedef struct {
    const double *X, *y;
    int     n, nv;
    double  ystd, ymean;
    double *pred;              /* n scratch predictions                    */
    double *stk;               /* evaluation stack                         */
    nd__ins code[ND__CAP];     /* compiled candidate                       */
    int     ncode;             /* how much of it the last predict used     */
} nd__data;

/* Predict every row into d->pred. Returns 0 if any row blew up. */
static int nd__predict(const nd__node *t, nd__data *d){
    int nc = 0, i;
    nd__emit(t, d->code, &nc);
    d->ncode = nc;
    if (nc <= 0 || nc >= ND__CAP) return 0;
    for (i = 0; i < d->n; i++)
        if (!nd__run(d->code, nc, &d->X[(size_t)i * d->nv], d->stk, &d->pred[i]))
            return 0;
    return 1;
}

/* Solve the least-squares a, b for the predictions already in d->pred and
 * return the resulting RMSE. This is the whole trick: two of the constants
 * in every candidate are computed, never searched for. */
static double nd__scale_rmse(nd__data *d, double *a, double *b){
    double sp = 0, spp = 0, spy = 0, mp, cov, var, se = 0;
    int i, n = d->n;
    for (i = 0; i < n; i++){ sp += d->pred[i]; }
    mp = sp / n;
    for (i = 0; i < n; i++){
        double dp = d->pred[i] - mp;
        spp += dp * dp;
        spy += dp * (d->y[i] - d->ymean);
    }
    var = spp / n; cov = spy / n;
    if (var < 1e-14){ *a = 0.0; *b = d->ymean; }
    else            { *a = cov / var; *b = d->ymean - (*a) * mp; }
    for (i = 0; i < n; i++){
        double e = (*a) * d->pred[i] + (*b) - d->y[i];
        se += e * e;
    }
    return sqrt(se / n);
}

/* Normalised RMSE: error in units of the target's own spread, so a threshold
 * means the same thing whether y is in millimetres or megapascals. */
static double nd__nrmse(const nd__node *t, nd__data *d, double *a, double *b){
    double aa, bb;
    if (!nd__predict(t, d)){
        if (a) *a = 0.0;
        if (b) *b = d->ymean;
        return ND__BIG;
    }
    { double r = nd__scale_rmse(d, &aa, &bb) / d->ystd;
      if (a) *a = aa;
      if (b) *b = bb;
      return (r == r) ? r : ND__BIG; }
}

/* ======================================================================== */
/*  Algebraic simplification                                                */
/*                                                                          */
/*  Folds constants and removes identities. Two things fall out: the answer  */
/*  reads like an equation instead of a transcript, and the parsimony        */
/*  pressure starts measuring real complexity rather than accumulated junk.  */
/* ======================================================================== */
static nd__node *nd__constant(double v){
    nd__node *n = nd__new(ND__CONST);
    if (n) n->val = v;
    return n;
}
static int nd__is_const(const nd__node *n, double v){
    return n && n->type == ND__CONST && n->val == v;
}

static nd__node *nd__simplify(nd__node *n){
    int t;
    if (!n) return NULL;
    n->l = nd__simplify(n->l);
    n->r = nd__simplify(n->r);
    t = n->type;
    if (t == ND__CONST || t == ND__VAR) return n;

    /* constant folding - the protected primitives make this exact */
    if (n->l && n->l->type == ND__CONST &&
        (nd__arity(t) == 1 || (n->r && n->r->type == ND__CONST))){
        double v = (nd__arity(t) == 1) ? nd__apply1(t, n->l->val)
                                       : nd__apply2(t, n->l->val, n->r->val);
        if (v == v && v < ND__BIG && v > -ND__BIG){
            nd__node *k = nd__constant(v);
            if (k){ nd__drop(n); return k; }
        }
    }

    switch (t){
        case ND__ADD:
            if (nd__is_const(n->r, 0.0)){ nd__node *k = n->l; n->l = NULL; nd__drop(n); return k; }
            if (nd__is_const(n->l, 0.0)){ nd__node *k = n->r; n->r = NULL; nd__drop(n); return k; }
            break;
        case ND__SUB:
            if (nd__is_const(n->r, 0.0)){ nd__node *k = n->l; n->l = NULL; nd__drop(n); return k; }
            if (nd__same(n->l, n->r)){ nd__node *k = nd__constant(0.0); if (k){ nd__drop(n); return k; } }
            break;
        case ND__MUL:
            if (nd__is_const(n->r, 1.0)){ nd__node *k = n->l; n->l = NULL; nd__drop(n); return k; }
            if (nd__is_const(n->l, 1.0)){ nd__node *k = n->r; n->r = NULL; nd__drop(n); return k; }
            if (nd__is_const(n->l, 0.0) || nd__is_const(n->r, 0.0)){
                nd__node *k = nd__constant(0.0); if (k){ nd__drop(n); return k; }
            }
            if (nd__same(n->l, n->r)){                 /* x*x -> x^2, if we have it */
                nd__node *k = nd__new(ND__SQR);
                if (k){ k->l = n->l; n->l = NULL; nd__drop(n); return k; }
            }
            break;
        case ND__DIV:
            if (nd__is_const(n->r, 1.0)){ nd__node *k = n->l; n->l = NULL; nd__drop(n); return k; }
            if (nd__same(n->l, n->r)){ nd__node *k = nd__constant(1.0); if (k){ nd__drop(n); return k; } }
            if (nd__is_const(n->l, 1.0)){              /* 1/x -> inv(x) */
                nd__node *k = nd__new(ND__INV);
                if (k){ k->l = n->r; n->r = NULL; nd__drop(n); return k; }
            }
            break;
        default: break;
    }
    return n;
}

/* The reported equation is a*f(x)+b, so a constant sitting at the ROOT of f
 * is already accounted for twice. Peeling it off changes nothing but the node
 * count and how the answer reads: ((n*hbar) - 1) + 1 becomes n*hbar. */
static nd__node *nd__canon_root(nd__node *n){
    int guard;
    for (guard = 0; n && guard < ND__CAP; guard++){
        nd__node *keep = NULL;
        int t = n->type;
        if (t == ND__ADD || t == ND__SUB || t == ND__MUL){
            /* a+c, c+a, a-c, c-a, a*c, c*a - the scale absorbs every one,
             * sign flips included */
            if (n->r && n->r->type == ND__CONST &&
                !(t == ND__MUL && n->r->val == 0.0)) keep = n->l;
            else if (n->l && n->l->type == ND__CONST &&
                     !(t == ND__MUL && n->l->val == 0.0)) keep = n->r;
        } else if (t == ND__DIV){
            /* a/c only: c/a is not an affine rescaling of a */
            if (n->r && n->r->type == ND__CONST && n->r->val != 0.0) keep = n->l;
        }
        if (!keep || keep->type == ND__CONST) break;
        if (n->l == keep) n->l = NULL; else n->r = NULL;
        nd__drop(n);
        n = keep;
    }
    return n;
}

/* ======================================================================== */
/*  Constant polish - compass search                                        */
/*                                                                          */
/*  Coordinate-wise probing with an adaptive step. It is derivative-free,    */
/*  so no stiff subexpression can make a numerical gradient explode, and it  */
/*  only ever accepts a strict improvement, so it cannot damage a champion.  */
/* ======================================================================== */
static void nd__consts(nd__node *n, double **arr, int *c){
    if (!n || *c >= ND__MAXCONST) return;
    if (n->type == ND__CONST) arr[(*c)++] = &n->val;
    nd__consts(n->l, arr, c);
    nd__consts(n->r, arr, c);
}

static void nd__polish(nd__node *t, nd__data *d, int iters){
    double *cs[ND__MAXCONST];
    int nc = 0, it, k;
    double best, step = 0.5;

    if (iters <= 0) return;
    nd__consts(t, cs, &nc);
    if (nc == 0) return;
    best = nd__nrmse(t, d, NULL, NULL);
    if (best >= ND__BIG) return;

    /* One reflection sweep before the local search starts. A constant born
     * with the wrong SIGN is the one case a compass search cannot repair on
     * its own: walking c from +2 to -0.5 has to pass through c = 0, where
     * exp(c*x^2) collapses to a constant and the error is at a local maximum.
     * Testing -c outright steps over that wall. */
    for (k = 0; k < nc; k++){
        double old = *cs[k], trial;
        if (old == 0.0) continue;
        *cs[k] = -old;
        trial  = nd__nrmse(t, d, NULL, NULL);
        if (trial < best - 1e-15) best = trial; else *cs[k] = old;
    }

    for (it = 0; it < iters && step > 1e-9; it++){
        int improved = 0;
        for (k = 0; k < nc; k++){
            double old = *cs[k], trial;
            int s;
            for (s = 0; s < 2; s++){
                *cs[k] = old + (s ? -step : step) * (fabs(old) > 1.0 ? fabs(old) : 1.0);
                trial = nd__nrmse(t, d, NULL, NULL);
                if (trial < best - 1e-15){ best = trial; old = *cs[k]; improved = 1; break; }
            }
            *cs[k] = old;
        }
        if (!improved) step *= 0.5;
    }
}

/* ======================================================================== */
/*  Genetic operators                                                       */
/* ======================================================================== */
static void nd__slots(nd__node **np, nd__node ***arr, int *cnt){
    if (!*np || *cnt >= ND__CAP) return;
    arr[(*cnt)++] = np;
    nd__slots(&(*np)->l, arr, cnt);
    nd__slots(&(*np)->r, arr, cnt);
}

/* Subtree crossover, biased towards grafting SMALL donors. Uniform choice on
 * the donor is how GP bloats: big subtrees get copied in far more often than
 * they earn their place. */
static nd__node *nd__cross(nd__ctx *c, const nd__node *a, const nd__node *b){
    nd__node *ca = nd__clone(a), *cb = nd__clone(b);
    nd__node **sa[ND__CAP]; int na = 0;
    nd__node **sb[ND__CAP]; int nb = 0;
    nd__node **dst, **src, *graft;
    int tries;

    if (!ca || !cb){ nd__drop(ca); nd__drop(cb); return NULL; }
    nd__slots(&ca, sa, &na);
    nd__slots(&cb, sb, &nb);
    if (na == 0 || nb == 0){ nd__drop(cb); return ca; }

    dst = sa[nd__pick(c, na)];
    src = sb[nd__pick(c, nb)];
    for (tries = 0; tries < 3 && nd__size(*src) > 8; tries++)
        src = sb[nd__pick(c, nb)];

    graft = nd__clone(*src);
    if (graft){ nd__drop(*dst); *dst = graft; }
    nd__drop(cb);
    return ca;
}

static void nd__mutate(nd__ctx *c, nd__node **root){
    nd__node **s[ND__CAP]; int n = 0;
    nd__node **pk;
    double r;

    nd__slots(root, s, &n);
    if (n == 0) return;
    pk = s[nd__pick(c, n)];
    r  = nd__unit(c);

    if ((*pk)->type == ND__CONST && r < 0.40){
        /* jitter a constant, occasionally by a lot */
        (*pk)->val += nd__gauss(c) * (nd__unit(c) < 0.2 ? 2.0 : 0.35);
    } else if (r < 0.55 && (*pk)->type == ND__VAR){
        (*pk)->var = nd__pick(c, c->nvars);            /* swap the variable  */
    } else if (r < 0.72 && nd__arity((*pk)->type) == 2 && c->nbin > 1){
        (*pk)->type = c->bin[nd__pick(c, c->nbin)];    /* swap the operator  */
    } else if (r < 0.85 && (*pk)->l && (*pk)->r){
        nd__node *keep = nd__unit(c) < 0.5 ? (*pk)->l : (*pk)->r;   /* hoist */
        nd__node *lift = nd__clone(keep);
        if (lift){ nd__drop(*pk); *pk = lift; }
    } else {
        nd__node *sub = nd__grow(c, 0, 2, 0);          /* fresh subtree      */
        if (sub){ nd__drop(*pk); *pk = sub; }
    }
}

/* ======================================================================== */
/*  The Pareto archive                                                      */
/*                                                                          */
/*  Best raw error seen at each complexity. Keeping it by size (rather than  */
/*  keeping one champion) is what turns a single answer into a front - and   */
/*  it costs one comparison per evaluation.                                  */
/* ======================================================================== */
typedef struct {
    nd__node *tree[ND__CAP + 1];
    double    err [ND__CAP + 1];
    int       cap;
} nd__archive;

static void nd__archive_init(nd__archive *ar, int cap){
    int i;
    ar->cap = cap;
    for (i = 0; i <= cap; i++){ ar->tree[i] = NULL; ar->err[i] = ND__BIG; }
}
/* Admission polishes before it judges. A candidate that has just found the
 * right SHAPE usually still carries the random constants it was born with,
 * and is thrown away one generation later for an error that a few coordinate
 * steps would have fixed. Tuning on admission is what lets exp(c*x^2) survive
 * long enough to become exp(-0.5*x^2). Admissions get rarer every generation,
 * so the cost is front-loaded and small. */
static void nd__archive_offer(nd__archive *ar, const nd__node *t, int size,
                              double err, nd__data *d, int polish_iters){
    nd__node *cp;
    if (size < 1 || size > ar->cap || err >= ND__BIG) return;
    if (err >= ar->err[size] - 1e-12) return;

    cp = nd__clone(t);
    if (!cp) return;
    if (polish_iters > 0){
        nd__polish(cp, d, polish_iters);
        err = nd__nrmse(cp, d, NULL, NULL);
        if (err >= ar->err[size] - 1e-12){ nd__drop(cp); return; }
    }
    nd__drop(ar->tree[size]);
    ar->tree[size] = cp;
    ar->err[size]  = err;
}
static void nd__archive_clear(nd__archive *ar){
    int i;
    for (i = 0; i <= ar->cap; i++){ nd__drop(ar->tree[i]); ar->tree[i] = NULL; }
}

/* ======================================================================== */
/*  Public API                                                              */
/* ======================================================================== */
nd_options nd_defaults(void){
    nd_options o;
    o.population  = 400;
    o.islands     = 5;
    o.generations = 300;
    o.migration   = 25;
    o.max_nodes   = 28;
    o.parsimony   = 0.002;
    o.ops         = ND_OPS_DEFAULT;
    o.tune_iters  = 80;
    o.tolerance   = 1e-9;
    o.seed        = 1;
    o.progress    = NULL;
    o.user        = NULL;
    return o;
}

/* Expand the operator bitmask into the two lists the tree builder samples. */
static void nd__ops(nd__ctx *c, unsigned int ops){
    static const int bin_t[4] = { ND__ADD, ND__SUB, ND__MUL, ND__DIV };
    static const unsigned int bin_f[4] = { ND_OP_ADD, ND_OP_SUB, ND_OP_MUL, ND_OP_DIV };
    static const int un_t[9]  = { ND__NEG, ND__INV, ND__SQR, ND__SQRT,
                                  ND__EXP, ND__LOG, ND__SIN, ND__COS, ND__TANH };
    static const unsigned int un_f[9] = { ND_OP_NEG, ND_OP_INV, ND_OP_SQR, ND_OP_SQRT,
                                          ND_OP_EXP, ND_OP_LOG, ND_OP_SIN, ND_OP_COS,
                                          ND_OP_TANH };
    int i;
    c->nbin = c->nun = 0;
    for (i = 0; i < 4; i++) if (ops & bin_f[i]) c->bin[c->nbin++] = bin_t[i];
    for (i = 0; i < 9; i++) if (ops & un_f[i])  c->un [c->nun++]  = un_t[i];
    if (c->nbin == 0){ c->bin[c->nbin++] = ND__ADD; c->bin[c->nbin++] = ND__MUL; }
    /* Unary nodes are cheap in size but expensive in search space, so their
     * share of new internal nodes scales with how many are actually enabled. */
    c->unary_p = c->nun ? 0.12 + 0.02 * (double)c->nun : 0.0;
    if (c->unary_p > 0.35) c->unary_p = 0.35;
}

nd_model *nd_fit(const double *X, const double *y, int n, int nvars,
                 nd_options opt){
    nd__ctx      ctx;
    nd__data     d;
    nd__archive  ar;
    nd_model    *model;
    nd__node   **pop = NULL;
    double      *sc  = NULL, *pars = NULL;
    nd__node   **next = NULL;
    int          POP, ISL, TOT, g, i, k, stop = 0;
    double       best_err = ND__BIG;
    const int    ELITE = 2, TOUR = 5;

    if (!X || !y || n < 4 || nvars < 1) return NULL;

    POP = opt.population < 16 ? 16 : opt.population;
    ISL = opt.islands    < 1  ? 1  : opt.islands;
    if (ISL > 16) ISL = 16;
    TOT = POP * ISL;
    if (opt.max_nodes < 3)        opt.max_nodes = 3;
    if (opt.max_nodes > ND__CAP)  opt.max_nodes = ND__CAP;
    if (opt.generations < 1)      opt.generations = 1;

    ctx.rng   = opt.seed ? opt.seed : 1;
    ctx.nvars = nvars;
    nd__ops(&ctx, opt.ops);

    /* ---- data + scratch ------------------------------------------------ */
    d.X = X; d.y = y; d.n = n; d.nv = nvars;
    d.pred = (double *)malloc(sizeof(double) * (size_t)n);
    d.stk  = (double *)malloc(sizeof(double) * (ND__CAP + 2));
    if (!d.pred || !d.stk){ free(d.pred); free(d.stk); return NULL; }
    {   double m = 0, v = 0;
        for (i = 0; i < n; i++) m += y[i];
        m /= n;
        for (i = 0; i < n; i++){ double e = y[i] - m; v += e * e; }
        d.ymean = m;
        d.ystd  = sqrt(v / n) + 1e-12;
    }

    pop  = (nd__node **)malloc(sizeof(nd__node *) * (size_t)TOT);
    next = (nd__node **)malloc(sizeof(nd__node *) * (size_t)POP);
    sc   = (double *)   malloc(sizeof(double)     * (size_t)TOT);
    pars = (double *)   malloc(sizeof(double)     * (size_t)ISL);
    if (!pop || !next || !sc || !pars){
        free(pop); free(next); free(sc); free(pars);
        free(d.pred); free(d.stk);
        return NULL;
    }
    nd__archive_init(&ar, opt.max_nodes);

    /* Islands differ in how hard they punish size. The accuracy-hungry ones
     * find the structure; the frugal ones compress it. Migration lets each
     * benefit from the other, and between them they fill the front. */
    for (k = 0; k < ISL; k++)
        pars[k] = opt.parsimony * (ISL == 1 ? 1.0
                                 : 0.25 + 1.75 * (double)k / (double)(ISL - 1));

    /* ---- ramped half-and-half initialisation ---------------------------
     * Alternating depths, alternately grown and full, so generation zero
     * holds a spread of shapes rather than a pile of stumps.
     * Individuals over the node ceiling are deliberately left in - they cannot
     * be archived, but their subtrees are building blocks, and one round of
     * breeding prunes them anyway. */
    for (i = 0; i < TOT; i++){
        int depth = 2 + (i % 4);
        pop[i] = nd__grow(&ctx, 0, depth, (i % 2) == 0);
        if (!pop[i]) pop[i] = nd__leaf(&ctx);
        pop[i] = nd__simplify(pop[i]);
    }

    /* ---- evolve -------------------------------------------------------- */
    for (g = 0; g <= opt.generations && !stop; g++){
        int isl;

        /* score everyone, and offer every candidate to the archive */
        for (i = 0; i < TOT; i++){
            int sz = nd__size(pop[i]);
            double raw = nd__nrmse(pop[i], &d, NULL, NULL);
            sc[i] = (raw >= ND__BIG) ? ND__BIG : raw + pars[i / POP] * (double)sz;
            nd__archive_offer(&ar, pop[i], sz, raw, &d, opt.tune_iters / 5);
            if (raw < best_err) best_err = raw;
        }

        if (opt.progress){
            nd_expr view;
            int bi = -1;
            for (i = 1; i <= ar.cap; i++)
                if (ar.tree[i] && (bi < 0 || ar.err[i] < ar.err[bi])) bi = i;
            view.tree = bi > 0 ? ar.tree[bi] : NULL;
            view.nodes = bi > 0 ? bi : 0;
            view.err = bi > 0 ? ar.err[bi] : ND__BIG;
            view.nvars = nvars;
            view.a = 1.0; view.b = 0.0;
            view.code = NULL; view.ncode = 0;
            if (view.tree){
                /* nd__nrmse leaves the compiled form in the scratch buffer;
                 * lending it to the view is what makes nd_eval work inside a
                 * progress callback instead of silently returning zero. */
                nd__nrmse(view.tree, &d, &view.a, &view.b);
                view.code  = d.code;
                view.ncode = d.ncode;
            }
            if (opt.progress(g, opt.generations, best_err,
                             view.tree ? &view : NULL, opt.user)) stop = 1;
        }
        if (best_err <= opt.tolerance) stop = 1;
        if (g == opt.generations || stop) break;

        /* ---- one generation, island by island -------------------------- */
        for (isl = 0; isl < ISL; isl++){
            nd__node **P = pop + (size_t)isl * POP;
            double    *S = sc  + (size_t)isl * POP;
            int        e, chosen[8];

            /* elites: a partial selection, not a full sort - we only ever
             * need the top few, and sorting POP every generation was the
             * single largest avoidable cost in this loop. */
            for (e = 0; e < ELITE; e++){
                int bi = -1;
                for (i = 0; i < POP; i++){
                    int already = 0, j;
                    for (j = 0; j < e; j++) if (chosen[j] == i) already = 1;
                    if (already) continue;
                    if (bi < 0 || S[i] < S[bi]) bi = i;
                }
                chosen[e] = bi;
                next[e] = bi >= 0 ? nd__clone(P[bi]) : nd__leaf(&ctx);
                if (!next[e]) next[e] = nd__leaf(&ctx);
            }

            for (i = ELITE; i < POP; i++){
                int pa = nd__pick(&ctx, POP), pb = nd__pick(&ctx, POP), t;
                nd__node *ch;
                for (t = 1; t < TOUR; t++){
                    int c1 = nd__pick(&ctx, POP); if (S[c1] < S[pa]) pa = c1;
                    c1 = nd__pick(&ctx, POP);     if (S[c1] < S[pb]) pb = c1;
                }
                ch = (nd__unit(&ctx) < 0.85) ? nd__cross(&ctx, P[pa], P[pb])
                                             : nd__clone(P[pa]);
                if (!ch) ch = nd__leaf(&ctx);
                if (nd__unit(&ctx) < 0.30) nd__mutate(&ctx, &ch);
                ch = nd__simplify(ch);
                if (nd__size(ch) > opt.max_nodes){
                    nd__node *fallback = nd__clone(P[pa]);
                    nd__drop(ch);
                    ch = fallback ? fallback : nd__leaf(&ctx);
                    if (nd__size(ch) > opt.max_nodes){
                        nd__drop(ch); ch = nd__leaf(&ctx);
                    }
                }
                next[i] = ch ? ch : nd__leaf(&ctx);
            }

            /* polish the island champion now and then: constants that are
             * nearly right steer the search, constants left random do not */
            if (opt.tune_iters > 0 && g > 0 && (g % 20) == 0)
                nd__polish(next[0], &d, opt.tune_iters / 4);

            for (i = 0; i < POP; i++) nd__drop(P[i]);
            for (i = 0; i < POP; i++) P[i] = next[i];
        }

        /* ---- migration: each island's best joins its neighbour ---------- */
        if (ISL > 1 && opt.migration > 0 && g > 0 && (g % opt.migration) == 0){
            for (isl = 0; isl < ISL; isl++){
                nd__node **src = pop + (size_t)isl * POP;
                nd__node **dst = pop + (size_t)((isl + 1) % ISL) * POP;
                nd__node  *cp  = nd__clone(src[0]);
                int slot = ELITE + nd__pick(&ctx, POP - ELITE);
                if (cp){ nd__drop(dst[slot]); dst[slot] = cp; }
            }
        }
    }

    for (i = 0; i < TOT; i++) nd__drop(pop[i]);
    free(pop); free(next); free(sc); free(pars);

    /* ---- build the front ----------------------------------------------- */
    /* Polish every archived structure, then keep only the sizes that are
     * genuinely worth their nodes: walking up in complexity, an entry stays
     * only if it beats everything simpler. That is the Pareto front. */
    model = (nd_model *)malloc(sizeof(nd_model));
    if (!model){ nd__archive_clear(&ar); free(d.pred); free(d.stk); return NULL; }
    model->front = (nd_expr *)malloc(sizeof(nd_expr) * (size_t)(ar.cap + 1));
    model->count = 0;
    model->nvars = nvars;
    if (!model->front){ free(model); nd__archive_clear(&ar); free(d.pred); free(d.stk); return NULL; }

    /* Polish, simplify, and take ownership of every archived structure. */
    for (i = 1; i <= ar.cap; i++){
        nd_expr *e;
        if (!ar.tree[i]) continue;
        nd__polish(ar.tree[i], &d, opt.tune_iters);
        ar.tree[i] = nd__canon_root(nd__simplify(ar.tree[i]));
        e = &model->front[model->count++];
        e->tree    = ar.tree[i];
        ar.tree[i] = NULL;                     /* ownership moves to the model */
        e->err     = nd__nrmse(e->tree, &d, &e->a, &e->b);
        e->nodes   = nd__size(e->tree);
        e->nvars   = nvars;
        e->code    = NULL;
        e->ncode   = 0;
    }
    nd__archive_clear(&ar);

    /* Simplification can shrink an entry below the size it was archived at,
     * so re-sort by true complexity before thinning. */
    for (i = 1; i < model->count; i++){
        nd_expr key = model->front[i];
        int j = i - 1;
        while (j >= 0 && (model->front[j].nodes > key.nodes ||
                         (model->front[j].nodes == key.nodes &&
                          model->front[j].err   > key.err))){
            model->front[j + 1] = model->front[j];
            j--;
        }
        model->front[j + 1] = key;
    }

    /* Thin to the front proper: walking up in complexity, an equation earns
     * its place only by beating everything simpler than it. */
    {
        double floor_err = ND__BIG;
        int kept = 0;
        for (i = 0; i < model->count; i++){
            if (model->front[i].err < floor_err - 1e-12){
                floor_err = model->front[i].err;
                model->front[kept++] = model->front[i];
            } else {
                nd__drop(model->front[i].tree);
            }
        }
        model->count = kept;
    }
    free(d.pred); free(d.stk);

    if (model->count == 0){ free(model->front); free(model); return NULL; }

    /* Compile each survivor once. Callers evaluate a discovered equation over
     * whole datasets; re-flattening the tree per row would make that cost the
     * traversal it was written to avoid. */
    for (i = 0; i < model->count; i++){
        nd_expr *e = &model->front[i];
        int nc = 0;
        e->code = (nd__ins *)malloc(sizeof(nd__ins) * (size_t)(e->nodes + 1));
        if (!e->code) continue;             /* nd_eval falls back to 0.0 */
        nd__emit(e->tree, e->code, &nc);
        e->ncode = nc;
    }
    return model;
}

void nd_free(nd_model *m){
    int i;
    if (!m) return;
    for (i = 0; i < m->count; i++){
        nd__drop(m->front[i].tree);
        free(m->front[i].code);
    }
    free(m->front);
    free(m);
}

int nd_count(const nd_model *m){ return m ? m->count : 0; }

const nd_expr *nd_at(const nd_model *m, int i){
    if (!m || i < 0 || i >= m->count) return NULL;
    return &m->front[i];
}
const nd_expr *nd_best(const nd_model *m){
    return (m && m->count) ? &m->front[m->count - 1] : NULL;
}
const nd_expr *nd_knee(const nd_model *m, double slack){
    double best, worst, budget;
    int i;
    if (!m || m->count == 0) return NULL;
    if (slack < 0.0) slack = 0.0;
    /* the front is sorted simplest-first and strictly improving, so its two
     * ends are the whole range of accuracy on offer */
    worst = m->front[0].err;
    best  = m->front[m->count - 1].err;
    budget = best + slack * (worst - best) + 1e-15;
    for (i = 0; i < m->count; i++)
        if (m->front[i].err <= budget) return &m->front[i];
    return &m->front[m->count - 1];
}

double nd_eval(const nd_expr *e, const double *x){
    double stk[ND__CAP + 2], v = 0.0;
    if (!e || !e->code) return 0.0;
    if (!nd__run(e->code, e->ncode, x, stk, &v)) return e->b;
    return e->a * v + e->b;
}

double nd_rmse(const nd_expr *e, const double *X, const double *y,
               int n, int nvars){
    double se = 0; int i;
    if (!e || n < 1) return ND__BIG;
    for (i = 0; i < n; i++){
        double d = nd_eval(e, &X[(size_t)i * nvars]) - y[i];
        se += d * d;
    }
    return sqrt(se / n);
}

double nd_r2(const nd_expr *e, const double *X, const double *y,
             int n, int nvars){
    double m = 0, sr = 0, st = 0; int i;
    if (!e || n < 2) return 0.0;
    for (i = 0; i < n; i++) m += y[i];
    m /= n;
    for (i = 0; i < n; i++){
        double er = nd_eval(e, &X[(size_t)i * nvars]) - y[i];
        double et = y[i] - m;
        sr += er * er; st += et * et;
    }
    if (st < 1e-30) return 0.0;
    return 1.0 - sr / st;
}

int nd_complexity(const nd_expr *e){ return e ? e->nodes : 0; }

/* ---- formatting -------------------------------------------------------- */
typedef struct { char *buf; int cap, len; } nd__sb;

static void nd__put(nd__sb *s, const char *txt){
    while (*txt){
        if (s->len + 1 < s->cap) s->buf[s->len] = *txt;
        s->len++; txt++;
    }
}
static void nd__putf(nd__sb *s, double v){
    char tmp[40];
    sprintf(tmp, "%.6g", v);
    nd__put(s, tmp);
}

static const char *nd__opname(int t){
    switch (t){
        case ND__SQRT: return "sqrt";
        case ND__EXP:  return "exp";
        case ND__LOG:  return "log";
        case ND__SIN:  return "sin";
        case ND__COS:  return "cos";
        case ND__TANH: return "tanh";
        default:       return "";
    }
}

/* A node that already prints as one indivisible token: a leaf, or a function
 * call, which carries its own brackets. Such a node never needs wrapping. */
static int nd__atomic(const nd__node *n){
    return n && (n->type == ND__CONST || n->type == ND__VAR ||
                 nd__arity(n->type) == 1);
}

/* `bare` asks for the outermost parentheses of a binary node to be omitted.
 * It is set only where dropping them cannot change how the result reads:
 * inside a function's own brackets, and at the root - see nd_format. */
static void nd__write(const nd__node *n, char *const *names, nd__sb *s, int bare){
    if (!n) return;
    switch (n->type){
        case ND__CONST: nd__putf(s, n->val); return;
        case ND__VAR:
            if (names && names[n->var]) nd__put(s, names[n->var]);
            else { char tmp[16]; sprintf(tmp, "x%d", n->var); nd__put(s, tmp); }
            return;
        case ND__SQR:
            /* a^2 rather than (a)^2 when the operand is already one token -
             * except another square, because a^2^2 is a reader's trap even
             * where it happens to evaluate the same */
            if (nd__atomic(n->l) && n->l->type != ND__SQR){
                nd__write(n->l, names, s, 0); nd__put(s, "^2");
            } else {
                nd__put(s, "("); nd__write(n->l, names, s, 1); nd__put(s, ")^2");
            }
            return;
        case ND__NEG: case ND__INV:
            /* These two must come out FULLY bracketed - (1/x), not 1/x. An
             * unbracketed reciprocal is read wrong the moment it lands on the
             * right of a division: a/1/x is (a/1)/x, which is not a/(1/x). */
            nd__put(s, n->type == ND__NEG ? "(-" : "(1/");
            nd__write(n->l, names, s, 0);
            nd__put(s, ")");
            return;
        case ND__SQRT: case ND__EXP: case ND__LOG:
        case ND__SIN:  case ND__COS: case ND__TANH:
            /* the brackets belong to the function, so the argument needs none
             * of its own: sqrt(L/g), not sqrt((L/g)) */
            nd__put(s, nd__opname(n->type));
            nd__put(s, "("); nd__write(n->l, names, s, 1); nd__put(s, ")");
            return;
        default: {
            const char *op = n->type == ND__ADD ? " + " :
                             n->type == ND__SUB ? " - " :
                             n->type == ND__MUL ? "*"   : "/";
            if (!bare) nd__put(s, "(");
            nd__write(n->l, names, s, 0);
            nd__put(s, op);
            nd__write(n->r, names, s, 0);
            if (!bare) nd__put(s, ")");
            return;
        }
    }
}

int nd_format(const nd_expr *e, char *const *var_names, char *buf, int cap){
    nd__sb s;
    int scaled = 0, bare;
    s.buf = buf; s.cap = cap; s.len = 0;
    if (!e){ if (cap > 0) buf[0] = '\0'; return 0; }

    nd__put(&s, "y = ");
    if (fabs(e->a - 1.0) > 1e-9){ nd__putf(&s, e->a); nd__put(&s, "*"); scaled = 1; }
    /* The outermost parentheses are noise - "y = mu*Nn" beats "y = (mu*Nn)" -
     * but only where dropping them is harmless. A leading scale in front of a
     * SUM is exactly where it is not: 2*(a + b) is not 2*a + b. A trailing
     * offset is always safe, since + and - are read left to right anyway. */
    bare = e->tree && (e->tree->type == ND__ADD || e->tree->type == ND__SUB ||
                       e->tree->type == ND__MUL || e->tree->type == ND__DIV) &&
           (!scaled || e->tree->type == ND__MUL || e->tree->type == ND__DIV);
    nd__write(e->tree, var_names, &s, bare);
    if (fabs(e->b) > 1e-9){
        nd__put(&s, e->b < 0 ? " - " : " + ");
        nd__putf(&s, fabs(e->b));
    }
    if (cap > 0) buf[s.len < cap ? s.len : cap - 1] = '\0';
    return s.len;
}

void nd_print(const nd_expr *e, char *const *var_names, FILE *out){
    char buf[4096];
    nd_format(e, var_names, buf, (int)sizeof buf);
    fputs(buf, out);
}

#endif /* NERVE_DISCOVER_IMPLEMENTATION */
