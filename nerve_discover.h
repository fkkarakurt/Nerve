/* nerve_discover.h - symbolic regression / equation discovery in one header.
 *
 * Give it data, get back a compact human-readable EQUATION. No Python, no
 * dependencies (only libm), no build system - drop this file in and #include it.
 * Runs anywhere C compiles: a laptop, a browser (WebAssembly), an old machine.
 *
 * It searches the space of closed-form expressions (genetic programming over
 * expression trees) for the SIMPLEST formula that fits your data - a parsimony
 * pressure keeps the answer as small as a real law should be. Linear scaling
 * (Keijzer 2003) lets each candidate find its best a*f(x)+b for free, and a
 * numerical-gradient polish recovers real constants (2*pi, 1/sqrt(g), ...).
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
 *     nd_options o = nd_defaults();
 *     nd_expr *e = nd_fit(X, y, N, NV, o);
 *     char *names[NV] = { "x", ... };
 *     nd_print(e, names, stdout);        // e.g.  y = 1.00*((x*x*x) - x) + 0.00
 *     printf("R2 = %.4f\n", nd_r2(e, X, y, N, NV));
 *     nd_free(e);
 *
 * Other files that need the API just #include "nerve_discover.h" (no macro).
 * Build:  cc -O2 -std=c99 yourfile.c -lm
 *
 * License: Apache-2.0 (same as the rest of Nerve).
 * ------------------------------------------------------------------------- */

#ifndef NERVE_DISCOVER_H
#define NERVE_DISCOVER_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nd_expr nd_expr;   /* opaque: a discovered equation */

typedef struct {
    int                population;  /* candidate formulas per generation (2000) */
    int                generations; /* search iterations                 (200)  */
    int                max_nodes;   /* complexity ceiling / parsimony cap (30)   */
    double             parsimony;   /* size penalty weight             (0.0015)  */
    unsigned long long seed;        /* deterministic: same seed, same run (1)    */
} nd_options;

/* Sensible defaults; override fields as needed. */
nd_options nd_defaults(void);

/* Discover y ~ f(x0..x[nvars-1]) from n rows (X row-major: X[i*nvars + j]).
 * Returns a new nd_expr (free with nd_free), or NULL on allocation failure. */
nd_expr *nd_fit(const double *X, const double *y, int n, int nvars, nd_options opt);

/* Evaluate the discovered equation at one input row (length nvars). */
double nd_eval(const nd_expr *e, const double *x);

/* Coefficient of determination R^2 of e over a dataset (use held-out data to
 * measure generalization). */
double nd_r2(const nd_expr *e, const double *X, const double *y, int n, int nvars);

/* Number of nodes in the expression tree (a complexity measure). */
int nd_num_nodes(const nd_expr *e);

/* Print as "y = a*(expr) + b". var_names may be NULL (then x0, x1, ...). */
void nd_print(const nd_expr *e, char *const *var_names, FILE *out);

void nd_free(nd_expr *e);

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

/* ---- deterministic RNG (xorshift + splitmix mix) ----------------------- */
static unsigned long long nd__rng = 1;
static void   nd__seed(unsigned long long s){ nd__rng = s ? s : 1; }
static unsigned long long nd__u64(void){
    unsigned long long x = nd__rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    nd__rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static double nd__unit(void){ return (double)(nd__u64() >> 11) * (1.0/9007199254740992.0); }
static int    nd__int(int n){ return (int)(nd__unit() * n); }
static double nd__range(double a, double b){ return a + (b - a) * nd__unit(); }
static double nd__gauss(void){
    double u1 = nd__unit(), u2 = nd__unit();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ---- expression tree --------------------------------------------------- */
enum { ND_CONST, ND_VAR, ND_ADD, ND_SUB, ND_MUL, ND_DIV, ND_SQRT, ND_SIN, ND_EXP, ND_LOG };

typedef struct nd__node {
    int type; double val; int var;
    struct nd__node *l, *r;
} nd__node;

struct nd_expr { nd__node *tree; double a, b; int nvars; };

#define ND__MAXSLOTS 512

static int nd__gvars = 1;

static nd__node *nd__new(int t){
    nd__node *n = (nd__node *)malloc(sizeof(nd__node));
    n->type = t; n->val = 0.0; n->var = 0; n->l = n->r = NULL;
    return n;
}
static void nd__free_node(nd__node *n){ if (!n) return; nd__free_node(n->l); nd__free_node(n->r); free(n); }
static nd__node *nd__clone(const nd__node *n){
    nd__node *m;
    if (!n) return NULL;
    m = nd__new(n->type); m->val = n->val; m->var = n->var;
    m->l = nd__clone(n->l); m->r = nd__clone(n->r);
    return m;
}
static int nd__size(const nd__node *n){ return n ? 1 + nd__size(n->l) + nd__size(n->r) : 0; }

static nd__node *nd__terminal(void){
    if (nd__unit() < 0.55){ nd__node *n = nd__new(ND_VAR); n->var = nd__int(nd__gvars); return n; }
    else {
        nd__node *n = nd__new(ND_CONST); double r = nd__unit();
        if      (r < 0.45) n->val = (double)nd__int(5);
        else if (r < 0.60) n->val = 0.5;
        else               n->val = nd__range(-3.0, 3.0);
        return n;
    }
}
static nd__node *nd__grow(int depth, int maxd){
    nd__node *n;
    if (depth >= maxd || (depth > 0 && nd__unit() < 0.30)) return nd__terminal();
    if (nd__unit() < 0.22){
        int u = nd__int(4);
        int t = (u == 0) ? ND_SQRT : (u == 1) ? ND_SIN : (u == 2) ? ND_EXP : ND_LOG;
        n = nd__new(t); n->l = nd__grow(depth + 1, maxd);
        return n;
    } else {
        static const int ops[4] = { ND_ADD, ND_SUB, ND_MUL, ND_DIV };
        n = nd__new(ops[nd__int(4)]);
        n->l = nd__grow(depth + 1, maxd);
        n->r = nd__grow(depth + 1, maxd);
        return n;
    }
}
static double nd__evaltree(const nd__node *n, const double *x){
    double a, b;
    switch (n->type){
        case ND_CONST: return n->val;
        case ND_VAR:   return x[n->var];
        case ND_ADD:   return nd__evaltree(n->l,x) + nd__evaltree(n->r,x);
        case ND_SUB:   return nd__evaltree(n->l,x) - nd__evaltree(n->r,x);
        case ND_MUL:   return nd__evaltree(n->l,x) * nd__evaltree(n->r,x);
        case ND_DIV:   b = nd__evaltree(n->r,x); if (fabs(b) < 1e-9) return 1.0; a = nd__evaltree(n->l,x); return a / b;
        case ND_SQRT:  return sqrt(fabs(nd__evaltree(n->l,x)));
        case ND_SIN:   return sin(nd__evaltree(n->l,x));
        case ND_EXP:   a = nd__evaltree(n->l,x); if (a > 40.0) a = 40.0; if (a < -40.0) a = -40.0; return exp(a);
        case ND_LOG:   return log(fabs(nd__evaltree(n->l,x)) + 1e-9);
    }
    return 0.0;
}

/* ---- fitness: linear scaling + parsimony ------------------------------- */
static void nd__scale(const nd__node *t, const double *X, const double *y, int n, int nv, double *a, double *b){
    double sp = 0, sy = 0, spp = 0, spy = 0; int i;
    for (i = 0; i < n; i++){
        double p = nd__evaltree(t, &X[i*nv]), yy = y[i];
        if (!(p == p) || p > 1e15 || p < -1e15){ *a = 0; *b = yy; return; }
        sp += p; sy += yy; spp += p*p; spy += p*yy;
    }
    { double mp = sp/n, my = sy/n, cov = spy/n - mp*my, var = spp/n - mp*mp;
      if (var < 1e-12){ *a = 0; *b = my; } else { *a = cov/var; *b = my - (*a)*mp; } }
}
static double nd__rmse(const nd__node *t, const double *X, const double *y, int n, int nv, double a, double b){
    double se = 0; int i;
    for (i = 0; i < n; i++){
        double p = nd__evaltree(t, &X[i*nv]), e;
        if (!(p == p) || p > 1e15 || p < -1e15) return 1e30;
        e = (a*p + b) - y[i]; se += e*e;
    }
    return sqrt(se / n);
}
static double nd__ystd(const double *y, int n){
    double m = 0, v = 0; int i;
    for (i = 0; i < n; i++) m += y[i];
    m /= n;
    for (i = 0; i < n; i++){ double e = y[i] - m; v += e*e; } v /= n;
    return sqrt(v) + 1e-12;
}

typedef struct { const double *X, *y; int n, nv; double ystd, parsimony; } nd__data;

static double nd__err(const nd__node *t, const nd__data *d){
    double a, b; nd__scale(t, d->X, d->y, d->n, d->nv, &a, &b);
    return nd__rmse(t, d->X, d->y, d->n, d->nv, a, b);
}
static double nd__score(const nd__node *t, const nd__data *d){
    double e = nd__err(t, d);
    if (e >= 1e30) return 1e30;
    return e / d->ystd + d->parsimony * (double)nd__size(t);
}

static void nd__consts(nd__node *n, double **arr, int *c){
    if (!n) return;
    if (n->type == ND_CONST && *c < 128) arr[(*c)++] = &n->val;
    nd__consts(n->l, arr, c); nd__consts(n->r, arr, c);
}
static void nd__tune(nd__node *t, const nd__data *d){
    double *cs[128]; double save[128]; int nc = 0, it, k; double before, after;
    nd__consts(t, cs, &nc);
    if (!nc) return;
    before = nd__err(t, d);
    for (k = 0; k < nc; k++) save[k] = *cs[k];
    for (it = 0; it < 60; it++){
        double lr = 0.05;
        for (k = 0; k < nc; k++){
            double old = *cs[k], h = 1e-4, ep, em, g;
            *cs[k] = old + h; ep = nd__err(t, d);
            *cs[k] = old - h; em = nd__err(t, d);
            *cs[k] = old; g = (ep - em) / (2*h);
            if (g == g) *cs[k] = old - lr * g;
        }
    }
    after = nd__err(t, d);
    if (after > before) for (k = 0; k < nc; k++) *cs[k] = save[k];
}

/* ---- genetic operators ------------------------------------------------- */
static void nd__collect(nd__node **np, nd__node ***arr, int *cnt){
    if (!*np || *cnt >= ND__MAXSLOTS) return;
    arr[(*cnt)++] = np;
    nd__collect(&(*np)->l, arr, cnt);
    nd__collect(&(*np)->r, arr, cnt);
}
static nd__node *nd__crossover(const nd__node *a, const nd__node *b){
    nd__node *ca = nd__clone(a), *cb = nd__clone(b);
    nd__node **sa[ND__MAXSLOTS]; int na = 0;
    nd__node **sb[ND__MAXSLOTS]; int nb = 0;
    nd__collect(&ca, sa, &na);
    nd__collect(&cb, sb, &nb);
    { nd__node **da = sa[nd__int(na)]; nd__node **db = sb[nd__int(nb)];
      nd__node *sub = nd__clone(*db); nd__free_node(*da); *da = sub; }
    nd__free_node(cb);
    return ca;
}
static void nd__mutate(nd__node **root){
    nd__node **s[ND__MAXSLOTS]; int n = 0;
    nd__collect(root, s, &n);
    { nd__node **pk = s[nd__int(n)];
      if (nd__unit() < 0.35 && (*pk)->type == ND_CONST) (*pk)->val += nd__gauss() * 0.6;
      else { nd__free_node(*pk); *pk = nd__grow(0, 2); } }
}

/* ---- public API -------------------------------------------------------- */
nd_options nd_defaults(void){
    nd_options o;
    o.population = 2000; o.generations = 200; o.max_nodes = 30;
    o.parsimony = 0.0015; o.seed = 1;
    return o;
}

nd_expr *nd_fit(const double *X, const double *y, int n, int nvars, nd_options opt){
    nd__data d; nd_expr *out; nd__node *champ = NULL; double champ_sc = 1e30;
    nd__node **pop; double *sc; int *idx; int g, i, POP = opt.population, GEN = opt.generations;
    const int ELITE = 3, TOUR = 6;
    if (n < 2 || nvars < 1 || POP < 8) return NULL;

    nd__seed(opt.seed);
    nd__gvars = nvars;
    d.X = X; d.y = y; d.n = n; d.nv = nvars; d.ystd = nd__ystd(y, n); d.parsimony = opt.parsimony;

    pop = (nd__node **)malloc(sizeof(nd__node*) * POP);
    sc  = (double *)malloc(sizeof(double) * POP);
    idx = (int *)malloc(sizeof(int) * POP);
    for (i = 0; i < POP; i++){ pop[i] = nd__grow(0, 4); sc[i] = nd__score(pop[i], &d); }

    for (g = 0; g <= GEN; g++){
        for (i = 0; i < POP; i++) idx[i] = i;
        for (i = 1; i < POP; i++){ int k = idx[i], j = i - 1;
            while (j >= 0 && sc[idx[j]] > sc[k]){ idx[j+1] = idx[j]; j--; } idx[j+1] = k; }
        if (sc[idx[0]] < champ_sc){ champ_sc = sc[idx[0]]; nd__free_node(champ); champ = nd__clone(pop[idx[0]]); }
        if (g == GEN) break;
        {
            nd__node **np = (nd__node **)malloc(sizeof(nd__node*) * POP);
            for (i = 0; i < ELITE; i++) np[i] = nd__clone(pop[idx[i]]);
            for (i = ELITE; i < POP; i++){
                int pa = nd__int(POP), pb = nd__int(POP), t2;
                for (t2 = 1; t2 < TOUR; t2++){ int c = nd__int(POP); if (sc[c] < sc[pa]) pa = c; c = nd__int(POP); if (sc[c] < sc[pb]) pb = c; }
                {
                    nd__node *ch = (nd__unit() < 0.90) ? nd__crossover(pop[pa], pop[pb]) : nd__clone(pop[pa]);
                    if (nd__unit() < 0.35) nd__mutate(&ch);
                    if (nd__size(ch) >= opt.max_nodes){ nd__free_node(ch); ch = nd__clone(pop[pa]); }
                    np[i] = ch;
                }
            }
            for (i = 0; i < POP; i++) nd__free_node(pop[i]);
            free(pop); pop = np;
            for (i = 0; i < POP; i++) sc[i] = nd__score(pop[i], &d);
        }
    }
    for (i = 0; i < POP; i++) nd__free_node(pop[i]);
    free(pop); free(sc); free(idx);

    nd__tune(champ, &d);

    out = (nd_expr *)malloc(sizeof(nd_expr));
    out->tree = champ; out->nvars = nvars;
    nd__scale(champ, X, y, n, nvars, &out->a, &out->b);
    return out;
}

double nd_eval(const nd_expr *e, const double *x){
    return e->a * nd__evaltree(e->tree, x) + e->b;
}
double nd_r2(const nd_expr *e, const double *X, const double *y, int n, int nvars){
    double m = 0, sr = 0, st = 0; int i;
    for (i = 0; i < n; i++) m += y[i];
    m /= n;
    for (i = 0; i < n; i++){
        double p = nd_eval(e, &X[i*nvars]);
        double er = p - y[i], et = y[i] - m;
        sr += er*er; st += et*et;
    }
    return (st < 1e-30) ? 0.0 : 1.0 - sr/st;
}
int nd_num_nodes(const nd_expr *e){ return nd__size(e->tree); }

static void nd__print_tree(const nd__node *n, char *const *names, FILE *out){
    switch (n->type){
        case ND_CONST: fprintf(out, "%.4g", n->val); break;
        case ND_VAR:   if (names) fprintf(out, "%s", names[n->var]); else fprintf(out, "x%d", n->var); break;
        case ND_SQRT:  fprintf(out, "sqrt("); nd__print_tree(n->l, names, out); fprintf(out, ")"); break;
        case ND_SIN:   fprintf(out, "sin(");  nd__print_tree(n->l, names, out); fprintf(out, ")"); break;
        case ND_EXP:   fprintf(out, "exp(");  nd__print_tree(n->l, names, out); fprintf(out, ")"); break;
        case ND_LOG:   fprintf(out, "log(");  nd__print_tree(n->l, names, out); fprintf(out, ")"); break;
        default: {
            char op = n->type==ND_ADD?'+':n->type==ND_SUB?'-':n->type==ND_MUL?'*':'/';
            fprintf(out, "("); nd__print_tree(n->l, names, out);
            fprintf(out, " %c ", op); nd__print_tree(n->r, names, out); fprintf(out, ")");
        }
    }
}
void nd_print(const nd_expr *e, char *const *var_names, FILE *out){
    fprintf(out, "y = %.4g*(", e->a);
    nd__print_tree(e->tree, var_names, out);
    fprintf(out, ") + %.4g", e->b);
}
void nd_free(nd_expr *e){ if (!e) return; nd__free_node(e->tree); free(e); }

#endif /* NERVE_DISCOVER_IMPLEMENTATION */
