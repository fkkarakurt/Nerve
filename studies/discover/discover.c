/* studies/discover/discover.c
 *
 * Nerve — symbolic regression engine (spike).
 *
 * Discover a closed-form equation from noisy (x, y) samples. If a problem has
 * real underlying structure, the discovered equation replaces a giant opaque
 * model with a compact, local, zero-cost, auditable formula.
 *
 * Design:
 *   - Genetic programming over expression trees finds the STRUCTURE.
 *   - Linear scaling (Keijzer 2003) gives every candidate its best global
 *     a*f + b for free, so the search sees the shape, not the scale.
 *   - Numerical-gradient constant tuning then refines the INTERNAL constants,
 *     so real irrational constants (2*pi, 1/sqrt(g), ...) are recovered.
 *   - Parsimony pressure keeps the answer as simple as a law should be.
 *   - A held-out TEST split reports generalization: on unknown data, matching
 *     held-out points is the only thing that separates a law from an overfit.
 *
 * Rediscovering known laws is the INSTRUMENT CALIBRATION. Discovering an unknown
 * law needs real-world data whose law is not known in advance.
 *
 * Zero dependencies beyond libm.  gcc -O2 -std=c99 discover.c -o discover -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PI 3.14159265358979323846

/* ---------------------------------------------------------------- RNG ----- */
static unsigned long long g_rng = 0x9E3779B97F4A7C15ULL;
static void   rng_seed(unsigned long long s){ g_rng = s ? s : 1; }
static unsigned long long rng_u64(void){
    unsigned long long x = g_rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static double rng_unit(void){ return (double)(rng_u64() >> 11) * (1.0/9007199254740992.0); }
static int    rng_int(int n){ return (int)(rng_unit() * n); }
static double rng_range(double a, double b){ return a + (b - a) * rng_unit(); }
static double rng_gauss(void){
    double u1 = rng_unit(), u2 = rng_unit();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ------------------------------------------------------------- Node -------- */
enum { N_CONST, N_VAR, N_ADD, N_SUB, N_MUL, N_DIV, N_SQRT, N_SIN };

typedef struct Node {
    int type;
    double val;          /* N_CONST */
    int    var;          /* N_VAR   */
    struct Node *l, *r;  /* r is NULL for unary ops */
} Node;

#define MAX_SLOTS 512

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
    if (rng_unit() < 0.55){
        Node *n = node_new(N_VAR); n->var = rng_int(g_nvars); return n;
    } else {
        Node *n = node_new(N_CONST);
        double r = rng_unit();
        if      (r < 0.45) n->val = (double)rng_int(5);
        else if (r < 0.60) n->val = 0.5;
        else               n->val = rng_range(-3.0, 3.0);
        return n;
    }
}
static Node *rand_tree(int depth, int maxd){
    Node *n;
    if (depth >= maxd || (depth > 0 && rng_unit() < 0.30)) return rand_terminal();
    if (rng_unit() < 0.22){                              /* unary op */
        n = node_new(rng_unit() < 0.7 ? N_SQRT : N_SIN);
        n->l = rand_tree(depth + 1, maxd);
        return n;
    } else {                                             /* binary op */
        static const int ops[4] = { N_ADD, N_SUB, N_MUL, N_DIV };
        n = node_new(ops[rng_int(4)]);
        n->l = rand_tree(depth + 1, maxd);
        n->r = rand_tree(depth + 1, maxd);
        return n;
    }
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
            if (fabs(b) < 1e-9) return 1.0;
            a = node_eval(n->l,x);
            return a / b;
        case N_SQRT:  return sqrt(fabs(node_eval(n->l,x)));   /* protected */
        case N_SIN:   return sin(node_eval(n->l,x));
    }
    return 0.0;
}

/* --------------------------------------------------------------- data ------ */
typedef struct {
    double *X; double *y; int n, nv;
    int *tr; int ntr;      /* train indices */
    int *te; int nte;      /* held-out test indices */
    double ystd_tr;
} Data;

/* best-fit a,b for a*f(x)+b over an index set (linear scaling). */
static void fit_scale(const Node *t, const Data *d, const int *idx, int m, double *a, double *b){
    double sp=0, sy=0, spp=0, spy=0; int i;
    for (i = 0; i < m; i++){
        int k = idx[i];
        double p = node_eval(t, &d->X[k*d->nv]);
        double y = d->y[k];
        if (!(p==p) || p>1e15 || p<-1e15){ *a = 0; *b = y; return; }
        sp += p; sy += y; spp += p*p; spy += p*y;
    }
    {
        double mp = sp/m, my = sy/m;
        double cov = spy/m - mp*my, var = spp/m - mp*mp;
        if (var < 1e-12){ *a = 0; *b = my; } else { *a = cov/var; *b = my - (*a)*mp; }
    }
}
static double rmse_scaled(const Node *t, const Data *d, const int *idx, int m, double a, double b){
    double se = 0; int i;
    for (i = 0; i < m; i++){
        int k = idx[i];
        double p = node_eval(t, &d->X[k*d->nv]);
        double e;
        if (!(p==p) || p>1e15 || p<-1e15) return 1e30;
        e = (a*p + b) - d->y[k];
        se += e*e;
    }
    return sqrt(se/m);
}
static double err_train(const Node *t, const Data *d){
    double a, b; fit_scale(t, d, d->tr, d->ntr, &a, &b);
    return rmse_scaled(t, d, d->tr, d->ntr, a, b);
}
static double score_train(const Node *t, const Data *d){
    double e = err_train(t, d);
    if (e >= 1e30) return 1e30;
    return e / d->ystd_tr + 0.0015 * (double)node_size(t);
}

/* -------------------------------------------------- constant refinement ---- */
static void collect_consts(Node *n, double **arr, int *c){
    if (!n) return;
    if (n->type == N_CONST && *c < 128) arr[(*c)++] = &n->val;
    collect_consts(n->l, arr, c);
    collect_consts(n->r, arr, c);
}
static void tune_constants(Node *t, const Data *d){
    double *cs[128]; int nc = 0, it, k;
    collect_consts(t, cs, &nc);
    if (!nc) return;
    for (it = 0; it < 60; it++){
        double lr = 0.05;
        for (k = 0; k < nc; k++){
            double old = *cs[k], h = 1e-4, ep, em, g;
            *cs[k] = old + h; ep = err_train(t, d);
            *cs[k] = old - h; em = err_train(t, d);
            *cs[k] = old;
            g = (ep - em) / (2*h);
            if (g == g) *cs[k] = old - lr * g;
        }
    }
}

/* -------------------------------------------- subtree crossover / mutate --- */
static void collect(Node **np, Node ***arr, int *cnt){
    if (!*np || *cnt >= MAX_SLOTS) return;
    arr[(*cnt)++] = np;
    collect(&(*np)->l, arr, cnt);
    collect(&(*np)->r, arr, cnt);
}
static Node *crossover(const Node *a, const Node *b){
    Node *ca = node_clone(a), *cb = node_clone(b);
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
            (*pick)->val += rng_gauss() * 0.6;
        } else {
            node_free(*pick);
            *pick = rand_tree(0, 2);
        }
    }
}

/* --------------------------------------------------------------- print ----- */
static void print_expr(const Node *n){
    switch (n->type){
        case N_CONST: printf("%.4g", n->val); break;
        case N_VAR:   printf("%s", g_names[n->var]); break;
        case N_SQRT:  printf("sqrt("); print_expr(n->l); printf(")"); break;
        case N_SIN:   printf("sin(");  print_expr(n->l); printf(")"); break;
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

    for (i = 0; i < POP; i++){ pop[i] = rand_tree(0, 4); sc[i] = score_train(pop[i], d); }

    for (g = 0; g <= GEN; g++){
        for (i = 0; i < POP; i++) idx[i] = i;
        for (i = 1; i < POP; i++){ int k = idx[i], j = i - 1;
            while (j >= 0 && sc[idx[j]] > sc[k]){ idx[j+1] = idx[j]; j--; } idx[j+1] = k; }

        if (sc[idx[0]] < champ_sc){ champ_sc = sc[idx[0]]; node_free(champ); champ = node_clone(pop[idx[0]]); }
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
            for (i = 0; i < POP; i++) sc[i] = score_train(pop[i], d);
        }
    }
    for (i = 0; i < POP; i++) node_free(pop[i]);
    free(pop); free(sc); free(idx);

    tune_constants(champ, d);      /* final polish of internal constants */
    return champ;
}

static void report(Node *best, const Data *d){
    double a, b, tr, te;
    fit_scale(best, d, d->tr, d->ntr, &a, &b);
    tr = rmse_scaled(best, d, d->tr, d->ntr, a, b);
    te = rmse_scaled(best, d, d->te, d->nte, a, b);
    printf("  DISCOVERED:  y = %.4g * (", a); print_expr(best); printf(") + %.4g\n", b);
    /* A real law lands held-out error near the noise floor and comparable to
       train. An overfit curve blows up on unseen points: high absolute test
       error, and test >> train. (The boolean is a heuristic; the numbers are
       the truth -- read them.) */
    {
        double trn = tr/d->ystd_tr, ten = te/d->ystd_tr;
        int ok = (ten < 0.10) && (te < tr * 2.5);
        printf("  train nrmse = %.5f   |   held-out TEST nrmse = %.5f   (generalizes: %s)\n\n",
               trn, ten, ok ? "yes" : "no (overfit)");
    }
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
    /* deterministic shuffle, 70/30 train/test split */
    { int *perm = (int *)malloc(sizeof(int)*n);
      for (i = 0; i < n; i++) perm[i] = i;
      for (i = n-1; i > 0; i--){ int k = rng_int(i+1); int t = perm[i]; perm[i] = perm[k]; perm[k] = t; }
      d.ntr = (int)(n * 0.7); d.nte = n - d.ntr;
      d.tr = (int *)malloc(sizeof(int)*d.ntr);
      d.te = (int *)malloc(sizeof(int)*d.nte);
      for (i = 0; i < d.ntr; i++) d.tr[i] = perm[i];
      for (i = 0; i < d.nte; i++) d.te[i] = perm[d.ntr + i];
      free(perm);
    }
    for (i = 0; i < d.ntr; i++) mean += d.y[d.tr[i]];
    mean /= d.ntr;
    for (i = 0; i < d.ntr; i++){ double e = d.y[d.tr[i]] - mean; var += e*e; }
    var /= d.ntr;
    d.ystd_tr = sqrt(var) + 1e-12;
    return d;
}
static void free_data(Data *d){ free(d->X); free(d->y); free(d->tr); free(d->te); }

/* pendulum period:  T = 2*pi*sqrt(L/g),  g = 9.81  (real irrational constant) */
static double law_pendulum(const double *x){ return 2.0*PI*sqrt(x[0]/9.81); }
static double samp_pendulum(int j){ (void)j; return rng_range(0.1, 4.0); }

/* Kepler's third law:  T = sqrt(a^3) */
static double law_kepler(const double *x){ return sqrt(x[0]*x[0]*x[0]); }
static double samp_kepler(int j){ (void)j; return rng_range(0.5, 4.0); }

/* Newton / Coulomb inverse-square:  F = m1*m2 / r^2  (multivariate) */
static double law_invsq(const double *x){ return x[0]*x[1] / (x[2]*x[2]); }
static double samp_invsq(int j){ return (j == 2) ? rng_range(1.0, 3.0) : rng_range(1.0, 5.0); }

static void run(const char *title, unsigned long long seed, int nv, char **names,
                double noise, double (*law)(const double*), double (*samp)(int)){
    Data d;
    Node *best;
    rng_seed(seed);
    g_nvars = nv; g_names = names;
    d = make_data(200, nv, noise, law, samp);
    printf("%s   [200 samples, %.0f%% noise, 70/30 train/test]\n", title, noise*100.0);
    best = evolve(&d);
    report(best, &d);
    node_free(best); free_data(&d);
}

int main(void){
    { static char *nm[] = { "L" };        run("Target: T = 2*pi*sqrt(L/g)  (pendulum, g=9.81)", 111ULL, 1, nm, 0.02, law_pendulum, samp_pendulum); }
    { static char *nm[] = { "a" };        run("Target: T = sqrt(a^3)       (Kepler)",            222ULL, 1, nm, 0.02, law_kepler,   samp_kepler);   }
    { static char *nm[] = { "m1","m2","r" }; run("Target: F = m1*m2 / r^2     (inverse-square)",    333ULL, 3, nm, 0.02, law_invsq,    samp_invsq);    }
    return 0;
}
