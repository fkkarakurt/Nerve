# The Feynman benchmark

The standard yardstick for symbolic regression, run against
[`nerve_discover.h`](../../nerve_discover.h).

One hundred equations from the Feynman Lectures, catalogued by Udrescu &
Tegmark for AI Feynman (*Sci. Adv.* **6**, eaay2631, 2020) and used as the
accuracy track of SRBench (La Cava et al., NeurIPS 2021 Datasets & Benchmarks).
For each one the engine is handed a table of sampled inputs and a single output
column. It is told nothing else: not the operators that built the equation, not
the constants, not which of the nine columns actually matter. It has to write
the formula.

```sh
make quick     # 12 equations, about ten seconds
make full      # all 100, and a results.csv
make front EQ=I.6.2a
```

No download, no Python, no dataset directory. The equations and their sampling
intervals are compiled in — see [`equations.c`](equations.c), where every one
of them is a readable line of C next to its ground-truth formula.

## Protocol

Written down here rather than left implicit, because a benchmark whose rules
are adjustable is not a benchmark.

- **300 training rows and 300 held-out rows** per equation. Variables are drawn
  uniformly and independently from the intervals in `equations.c`, which are
  $(1,5)$ unless the physics forces otherwise — a speed must stay below $c$, a
  resonant denominator must not straddle zero, an arcsine argument must stay
  inside $[-1,1]$. Every such exception is a comment on its own line. None of
  them is there to flatter the engine; they exist to keep an equation
  well-posed.
- **Noise-free targets**, as in the reference benchmark. What is being measured
  is whether the search finds the right closed form at all.
- **One run per equation**, at a fixed per-equation seed. No restarts, no
  best-of-*n*, and no per-equation tuning: every equation gets the identical
  budget and the identical operator set.
- **The reported answer is the knee of the Pareto front** — the simplest
  equation conceding at most 1% of the accuracy the front spans — never the
  largest candidate on it. An engine that reports its most over-fitted
  candidate is grading its own homework.
- **Solved means held-out $R^2 \geq 0.999$**, the accuracy criterion used by
  SRBench, computed on rows the search never saw.
- **The operator set is `+ − × ÷ x² √x 1/x exp log sin cos tanh`.** Note what is
  missing: there is no arcsine, so `I.26.2` and `I.30.5` cannot be written
  exactly at all — the engine can only approximate them over the sampled
  interval. They stay in, scored on the same criterion as everything else,
  rather than being quietly dropped for being inconvenient.

## Results

Measured on this machine, from a clean `make full`:

```
CPU              Intel Core i7-7700HQ @ 2.80GHz, single core
compiler         gcc -O2 -std=c99
budget           400 candidates x 5 islands x 400 generations, max 26 nodes
```

| | |
|---|---|
| **solved** (held-out R² ≥ 0.999) | **82 / 100** |
| close (held-out R² ≥ 0.99) | 91 / 100 |
| median R² | 1.000000 |
| mean R² | 0.995736 |
| total search time | 854.4 s (8.5 s per equation) |

Some of what it recovers, verbatim from the run:

| Feynman | the law | what the engine wrote | R² |
|---------|---------|-----------------------|-----|
| `I.6.2a` | `exp(-theta^2/2)/sqrt(2*pi)` | `y = 0.398942*exp(-0.5*theta^2)` | 1.000000 |
| `I.14.4` | `0.5*k_spring*x^2` | `y = 0.5*k_spring*x^2` | 1.000000 |
| `I.47.23` | `sqrt(gamma*pr/rho)` | `y = sqrt(pr/(rho/gamma))` | 1.000000 |
| `III.8.54` | `sin(E_n*t/hbar)^2` | `y = -0.5*cos((t + t)*(E_n/hbar)) + 0.5` | 1.000000 |
| `III.15.27` | `2*pi*alpha/(n*d)` | `y = 6.28319*alpha/(d*n)` | 1.000000 |
| `II.38.14` | `Y/(2*(1+sigma))` | `y = 0.5*Y/(sigma + 1)` | 1.000000 |

Worth reading that table slowly. `I.14.4` came back character-for-character.
`I.6.2a` and `III.15.27` are the same law with the constant recovered
numerically — 0.398942 is 1/√(2π), and 6.28319 is 2π. And `III.8.54` did not
find `sin(x)²` at all: it found `(1 − cos 2x)/2`, which is the half-angle
identity, and is exactly correct. None of it was told any of this.

The full per-equation table, including every failure, is what `make full`
prints; `results.csv` holds the same data in a form you can plot.

## Reading this honestly

**This is a compact engine on a modest budget, on one core.** It is not a
distributed search, and it is not tuned per problem. The established tools
surveyed in SRBench — PySR, Operon, and the rest — reach higher recovery rates
when given the compute they are built for, and they should. The claim being
made here is narrower and, for a lot of people, more useful: this much search
quality is available with **no dependencies whatsoever**, from a single header,
in a form that compiles into an instrument, a microcontroller, a desktop
binary, or a browser tab.

**A solved equation is not always the textbook form.** The criterion is
predictive accuracy on held-out data, which is what SRBench measures and what
is mechanically checkable. An equation can reach R² ≥ 0.999 with an expression
that is algebraically equivalent to the original, or with one that is merely a
very good approximation over the sampled interval. The per-equation table
prints what was actually found so you can judge for yourself — and in the
clean cases the judgement is easy, as when `I.6.2a` comes back as
`0.398942*exp(-0.5*theta^2)` and that leading constant is $1/\sqrt{2\pi}$ to
six figures.

**The hard ones are hard for a reason.** The misses cluster where an exponential
sits in a denominator, where nine variables have to be sorted out at once, or
where the target is a difference of two nearly equal terms. They are listed
with their R² so the shape of the failure is visible rather than averaged away.
