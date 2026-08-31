# nerve_discover — symbolic regression in one header

Give it data, get back an **equation**. `nerve_discover.h` (at the repository
root, next to `nerve.h`) searches the space of closed-form expressions for the
simplest formula that fits your samples — and hands you something you can read,
check against theory, and evaluate in nanoseconds, rather than a set of weights.

No Python, no dependencies beyond libm, no build system. Drop the header in and
`#include` it. It runs wherever C compiles: a laptop, an old machine, a
microcontroller, or a **browser tab** via WebAssembly.

## How it works

- **Genetic programming** over expression trees evolves candidate formulas.
- **Linear scaling** (Keijzer 2003) solves every candidate's best `a·f(x)+b` in
  closed form, so the search spends itself on structure and never on the two
  constants that usually dominate the error.
- An **island model** — several sub-populations evolving separately and trading
  migrants — is what keeps a run from collapsing onto a single lineage. The
  islands also carry different parsimony pressures, so some hunt accuracy and
  others hunt brevity.
- The result is a **Pareto front**, not one answer: the best equation at every
  complexity, so you can see what each extra node bought and pick the knee.
- **Algebraic simplification** folds away `x·1`, `x+0`, `x−x` and constant
  subtrees, which both shortens the answer and stops the bloat that otherwise
  eats a GP run alive.
- A **derivative-free constant polish** recovers real constants — it is what
  turns `exp(c·θ²)` into `exp(−0.5·θ²)` and puts `1/√(2π)` in front of it.
- Deterministic: same seed, same result, everywhere.

## Use it

```c
#define NERVE_DISCOVER_IMPLEMENTATION
#include "nerve_discover.h"

double X[N*NV], y[N];               /* your data, X row-major */
char  *names[NV] = { "m1", "m2", "r" };

nd_options o = nd_defaults();
nd_model  *m = nd_fit(X, y, N, NV, o);

nd_print(nd_knee(m, 0.05), names, stdout);   /* the equation you want    */

for (int i = 0; i < nd_count(m); i++) {      /* or the whole front       */
    const nd_expr *e = nd_at(m, i);
    printf("%2d nodes  R2=%.5f  ", nd_complexity(e),
           nd_r2(e, Xtest, ytest, NTEST, NV));
    nd_print(e, names, stdout); putchar('\n');
}
nd_free(m);
```

Build: `cc -O2 -std=c99 yourfile.c -lm`

Restricting the operator set is the single most effective way to steer a
search — `o.ops = ND_OPS_ALGEBRA` keeps the answer to `+ − × ÷ x² √x 1/x`,
which is usually what you want for anything meant to end up in a design code.

## The demo

`discover.c` hands the engine only noisy samples from three hidden laws and
prints the front it recovers. Abridged:

```
$ cc -O2 -std=c99 discover.c -o discover -lm && ./discover

Newton:   F = m1*m2 / r^2   [200 samples, 2% noise]
     3 nodes   R2  0.65124   y = 2.66989*m1/r - 1.3702
     4 nodes   R2  0.76090   y = 3.18775*m1/r^2 - 0.0290851
     5 nodes   R2  0.91747   y = 1.81265*(m1 + m2)/r - 3.00765
     6 nodes   R2  0.99934   y = 0.996748*(m2*m1)/r^2 + 0.00943491
  -> the knee: the simplest equation worth its nodes
     y = 0.996748*(m2*m1)/r^2 + 0.00943491

Pendulum: T = 2*pi*sqrt(L/g) [200 samples, 1% noise]
  -> y = 6.27712*sqrt(L/g) - 0.000430896
```

The inverse square from the numbers alone, and 2π recovered to within 0.1%
from measurements that were 1% noisy. Note the rows above the answer: the
search passes through `m1/r`, then `m1/r²`, then `(m1+m2)/r` before the product
falls into place — and the front records every step of it.

## How good is it, really?

Measured, not asserted: see [`bench/feynman/`](../../bench/feynman/), which runs
the engine against the 100 equations of the Feynman Lectures — the standard
yardstick for this field — and reports the score with the protocol written down.

## Honest notes

- A high R² is a good *model*, not automatically a *law*. Judge a result by
  whether it is **simple**, **generalises to held-out data**, and is
  **physically meaningful** — never by accuracy alone.
- Rediscovering a known law validates the engine. Discovering a genuinely new
  one needs real data whose relationship is not known in advance.
- The search is stochastic. It is deterministic for a given seed, but a
  different seed is a different run; on a hard problem, try a few.
