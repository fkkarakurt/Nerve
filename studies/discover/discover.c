/* studies/discover/discover.c
 *
 * Nerve — symbolic regression spike.
 *
 * Question this file answers: can a tiny, zero-dependency C engine rediscover a
 * KNOWN closed-form law from nothing but noisy (x, y) samples? If it can, then
 * "discover the equation, then never pay a token again" is a real capability and
 * not a slogan.
 *
 * Method: genetic programming over expression trees, with a parsimony pressure
 * so the population collapses toward the SIMPLEST expression that fits — because
 * the whole point of a law is that it is compact. This is the same evolutionary
 * machinery Nerve already uses for neuroevolution, pointed at equations instead
 * of network weights.
 *
 * Zero dependencies beyond libm.  Build:  gcc -O2 -std=c99 discover.c -o discover -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------- RNG ----- */
/* Deterministic: same seed -> same run, everywhere. (Nerve values this.)     */
static unsigned long long g_rng = 0x9E3779B97F4A7C15ULL;
static void   rng_seed(unsigned long long s){ g_rng = s ? s : 1; }
static unsigned long long rng_u64(void){
    unsigned long long x = g_rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static double rng_unit(void){ return (double)(rng_u64() >> 11) * (1.0/9007199254740992.0); } /* [0,1) */
static int    rng_int(int n){ return (int)(rng_unit() * n); }
static double rng_range(double a, double b){ return a + (b - a) * rng_unit(); }
static double rng_gauss(void){
    double u1 = rng_unit(), u2 = rng_unit();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ------------------------------------------------------------- Node -------- */
enum { N_CONST, N_VAR, N_ADD, N_SUB, N_MUL, N_DIV };

typedef struct Node {
    int type;
    double val;          /* N_CONST */
    int    var;          /* N_VAR   */
    struct Node *l, *r;  /* operator children */
} Node;

#define MAX_SLOTS 512    /* hard cap on nodes per tree (also our parsimony ceiling) */

static int    g_nvars = 1;
static char **g_names = NULL;

static Node *node_new(int type){
    Node *n = (Node *)malloc(sizeof(Node));
    n->type = type; n->val = 0.0; n->var = 0; n->l = n->r = NULL;
    return n;
}
static void node_free(Node *n){ if (!n) return; node_free(n->l); node_free(n->r); free(n); }
static Node *node_clone(const Node *n){
    Node *m;
    if (!n) return NULL;
    m = node_new(n->type);
    m->val = n->val; m->var = n->var;
    m->l = node_clone(n->l); m->r = node_clone(n->r);
    return m;
}
static int node_size(const Node *n){ return n ? 1 + node_size(n->l) + node_size(n->r) : 0; }

/* ------------------------------------------------- random tree building ---- */
static Node *rand_terminal(void){
    if (rng_unit() < 0.55){                       /* variable */
        Node *n = node_new(N_VAR); n->var = rng_int(g_nvars); return n;
    } else {                                       /* constant, biased to simple */
        Node *n = node_new(N_CONST);
        double r = rng_unit();
        if      (r < 0.45) n->val = (double)rng_int(5);      /* 0,1,2,3,4     */
        else if (r < 0.60) n->val = 0.5;                     /* the ubiquitous 1/2 */
        else               n->val = rng_range(-3.0, 3.0);
        return n;
    }
}
static int rand_op(void){ static const int ops[4] = {N_ADD,N_SUB,N_MUL,N_DIV}; return ops[rng_int(4)]; }

static Node *rand_tree(int depth, int maxd){
    Node *n;
    if (depth >= maxd || (depth > 0 && rng_unit() < 0.30)) return rand_terminal();
    n = node_new(rand_op());
    n->l = rand_tree(depth + 1, maxd);
    n->r = rand_tree(depth + 1, maxd);
    return n;
}

/* ------------------------------------------------------------ evaluate ----- */
static double node_eval(const Node *n, const double *x){
    double a, b;
    switch (n->type){
        case N_CONST: return n->val;
        case N_VAR:   return x[n->var];
        case N_ADD:   return node_eval(n->l,x) + node_eval(n->r,x);
        case N_SUB:   return node_eval(n->l,x) - node_eval(n->r,x);
        case N_MUL:   return node_eval(n->l,x) * node_eval(n->r,x);
        case N_DIV:
            b = node_eval(n->r,x);
            if (fabs(b) < 1e-9) return 1.0;          /* protected division */
            a = node_eval(n->l,x);
            return a / b;
    }
    return 0.0;
}

/* --------------------------------------------------------------- data ------ */
typedef struct { double *X; double *y; int n; int nv; double ystd; } Data;

/* score = normalized RMSE + a small parsimony penalty on size.  Lower is better. */
static double score_of(const Node *t, const Data *d){
    double se = 0.0; int i;
    for (i = 0; i < d->n; i++){
        double p = node_eval(t, &d->X[i * d->nv]);
        double e;
        if (!(p == p) || p > 1e15 || p < -1e15) return 1e30;   /* NaN / inf */
        e = p - d->y[i];
        se += e * e;
    }
    {
        double nrmse = sqrt(se / d->n) / d->ystd;
        return nrmse + 0.0015 * (double)node_size(t);
    }
}

/* ---------------------------------------------- subtree pointer collection - */
static void collect(Node **np, Node ***arr, int *cnt){
    if (!*np || *cnt >= MAX_SLOTS) return;
    arr[(*cnt)++] = np;
    collect(&(*np)->l, arr, cnt);
    collect(&(*np)->r, arr, cnt);
}

/* child = clone(a) with a random subtree replaced by a random subtree of b */
static Node *crossover(const Node *a, const Node *b){
    Node *ca = node_clone(a);
    Node *cb = node_clone(b);
    Node **sa[MAX_SLOTS]; int na = 0;
    Node **sb[MAX_SLOTS]; int nb = 0;
    collect(&ca, sa, &na);
    collect(&cb, sb, &nb);
    {
        Node **da = sa[rng_int(na)];
        Node **db = sb[rng_int(nb)];
        Node  *sub = node_clone(*db);
        node_free(*da);
        *da = sub;
    }
    node_free(cb);
    return ca;
}

static void mutate(Node **root){
    Node **s[MAX_SLOTS]; int n = 0;
    collect(root, s, &n);
    {
        Node **pick = s[rng_int(n)];
        if (rng_unit() < 0.35 && (*pick)->type == N_CONST){
            (*pick)->val += rng_gauss() * 0.6;              /* nudge a constant */
        } else {
            node_free(*pick);
            *pick = rand_tree(0, 2);                        /* graft a small twig */
        }
    }
}

/* --------------------------------------------------------------- print ----- */
static void print_expr(const Node *n){
    switch (n->type){
        case N_CONST: printf("%.4g", n->val); break;
        case N_VAR:   printf("%s", g_names[n->var]); break;
        default: {
            char op = n->type==N_ADD?'+':n->type==N_SUB?'-':n->type==N_MUL?'*':'/';
            printf("("); print_expr(n->l); printf(" %c ", op); print_expr(n->r); printf(")");
        }
    }
}

/* ----------------------------------------------------------- the search ---- */
#define POP   2500
#define GEN   250
#define ELITE 3
#define TOUR  6

static int tournament(const double *sc){
    int best = rng_int(POP), i;
    for (i = 1; i < TOUR; i++){ int c = rng_int(POP); if (sc[c] < sc[best]) best = c; }
    return best;
}

static Node *evolve(const Data *d){
    Node  **pop = (Node **)malloc(sizeof(Node*) * POP);
    double *sc  = (double *)malloc(sizeof(double) * POP);
    int    *idx = (int *)malloc(sizeof(int) * POP);
    Node   *champ = NULL; double champ_sc = 1e30;
    int g, i;

    for (i = 0; i < POP; i++){ pop[i] = rand_tree(0, 4); sc[i] = score_of(pop[i], d); }

    for (g = 0; g <= GEN; g++){
        /* rank (simple insertion of indices by score) */
        for (i = 0; i < POP; i++) idx[i] = i;
        for (i = 1; i < POP; i++){ int k = idx[i], j = i - 1;
            while (j >= 0 && sc[idx[j]] > sc[k]){ idx[j+1] = idx[j]; j--; } idx[j+1] = k; }

        if (sc[idx[0]] < champ_sc){ champ_sc = sc[idx[0]]; node_free(champ); champ = node_clone(pop[idx[0]]); }

        if (g % 50 == 0 || g == GEN){
            double rmse;
            { double se = 0; int t; for (t = 0; t < d->n; t++){ double e = node_eval(champ,&d->X[t*d->nv]) - d->y[t]; se += e*e; } rmse = sqrt(se/d->n); }
            printf("  gen %3d   nrmse=%.5f  size=%2d   ", g, rmse/d->ystd, node_size(champ));
            print_expr(champ); printf("\n");
        }
        if (g == GEN) break;

        {
            Node **np = (Node **)malloc(sizeof(Node*) * POP);
            for (i = 0; i < ELITE; i++) np[i] = node_clone(pop[idx[i]]);
            for (i = ELITE; i < POP; i++){
                int pa = tournament(sc), pb = tournament(sc);
                Node *child = (rng_unit() < 0.90) ? crossover(pop[pa], pop[pb]) : node_clone(pop[pa]);
                if (rng_unit() < 0.35) mutate(&child);
                if (node_size(child) >= MAX_SLOTS){ node_free(child); child = node_clone(pop[pa]); }
                np[i] = child;
            }
            for (i = 0; i < POP; i++) node_free(pop[i]);
            free(pop); pop = np;
            for (i = 0; i < POP; i++) sc[i] = score_of(pop[i], d);
        }
    }

    for (i = 0; i < POP; i++) node_free(pop[i]);
    free(pop); free(sc); free(idx);
    return champ;
}

/* --------------------------------------------------------------- demos ----- */
static Data make_data(int n, int nv, double noise,
                      double (*law)(const double*), double (*sampler)(int)){
    Data d; int i, j; double mean = 0, var = 0;
    d.n = n; d.nv = nv;
    d.X = (double *)malloc(sizeof(double) * n * nv);
    d.y = (double *)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++){
        for (j = 0; j < nv; j++) d.X[i*nv + j] = sampler(j);
        d.y[i] = law(&d.X[i*nv]) * (1.0 + noise * rng_gauss());
    }
    for (i = 0; i < n; i++) mean += d.y[i];
    mean /= n;
    for (i = 0; i < n; i++){ double e = d.y[i] - mean; var += e*e; }
    var /= n;
    d.ystd = sqrt(var) + 1e-12;
    return d;
}
static void free_data(Data *d){ free(d->X); free(d->y); }

/* target 1: a cubic,  y = x^3 - x           (one variable, nonlinear) */
static double law_cubic(const double *x){ return x[0]*x[0]*x[0] - x[0]; }
static double samp_cubic(int j){ (void)j; return rng_range(-2.0, 2.0); }

/* target 2: Newton / Coulomb inverse-square,  F = m1*m2 / r^2  (G = 1) */
static double law_invsq(const double *x){ return x[0]*x[1] / (x[2]*x[2]); }
static double samp_invsq(int j){ return (j == 2) ? rng_range(1.0, 3.0) : rng_range(1.0, 5.0); }

int main(void){
    /* ---- demo 1: rediscover a cubic ---- */
    {
        static char *names[] = { "x" };
        Data d; Node *best;
        rng_seed(20260717ULL);
        g_nvars = 1; g_names = names;
        d = make_data(80, 1, 0.0, law_cubic, samp_cubic);
        printf("Target 1 (hidden law):  y = x^3 - x     [80 samples, no noise]\n");
        best = evolve(&d);
        printf("  DISCOVERED:  y = "); print_expr(best); printf("\n\n");
        node_free(best); free_data(&d);
    }

    /* ---- demo 2: rediscover the inverse-square law ---- */
    {
        static char *names[] = { "m1", "m2", "r" };
        Data d; Node *best;
        rng_seed(424242ULL);
        g_nvars = 3; g_names = names;
        d = make_data(160, 3, 0.02, law_invsq, samp_invsq);
        printf("Target 2 (hidden law):  F = m1*m2 / r^2   [160 samples, 2%% noise]\n");
        best = evolve(&d);
        printf("  DISCOVERED:  F = "); print_expr(best); printf("\n");
        node_free(best); free_data(&d);
    }
    return 0;
}
