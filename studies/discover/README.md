# nerve_discover — symbolic regression in one header

Give it data, get back an **equation**. `nerve_discover.h` (at the repository
root, next to `nerve.h`) searches the space of closed-form expressions for the
*simplest* formula that fits your `(x, y)` samples — and hands you a readable
equation, not a black box.

No Python, no dependencies beyond libm, no build system. Drop the header in and
`#include` it. It runs wherever C compiles: a laptop, an old machine, a
microcontroller, or a **browser tab** via WebAssembly.

## How it works

- **Genetic programming** over expression trees (`+ − × ÷ √ sin exp log`) evolves
  candidate formulas.
- A **parsimony pressure** keeps the answer compact — the whole point of a law is
  that it is small.
- **Linear scaling** (Keijzer 2003) gives every candidate its best `a·f(x)+b` for
  free, so the search sees the *shape*, not the scale.
- A **numerical-gradient polish** recovers real constants (`2π`, `1/√g`, …).
- Deterministic: same seed → same result, everywhere.

## Use it

```c
#define NERVE_DISCOVER_IMPLEMENTATION
#include "nerve_discover.h"

double X[N*NV], y[N];               /* your data, X row-major  */
nd_options o = nd_defaults();
nd_expr *e = nd_fit(X, y, N, NV, o);

char *names[NV] = { "x", /* ... */ };
nd_print(e, names, stdout);         /* y = 1.00*((x*x*x) - x) + 0.00 */
printf("R2 = %.4f\n", nd_r2(e, X, y, N, NV));
nd_free(e);
```

Build: `cc -O2 -std=c99 yourfile.c -lm`

## The demo

`discover.c` hands the engine only noisy samples from a hidden law and prints the
equation it recovers:

```
$ cc -O2 -std=c99 discover.c -o discover -lm && ./discover
Hidden law:  F = m1*m2 / r^2     [160 samples, 2% noise]
  y = 0.9987*(((m1 / r) / (r / m2))) + 0.006128
  R2 = 0.99849,  7 nodes
Hidden law:  T = sqrt(a^3)       [120 samples, 2% noise]
  y = 1.004*((sqrt(a) * a)) + -0.007327
  R2 = 0.99839,  4 nodes
```

The inverse-square law and Kepler's third law, recovered from raw numbers.

## Honest notes

- A high R² fit is a good *model*, not automatically a *law*. Judge a result by
  whether it is **simple**, **generalizes to held-out data**, and is
  **physically meaningful** — not by accuracy alone.
- Rediscovering a known law validates the engine. Discovering a genuinely new one
  needs real data whose relationship is not known in advance.
