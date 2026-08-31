---
title: 'nerve_discover: symbolic regression in a single dependency-free C header'
tags:
  - C
  - symbolic regression
  - equation discovery
  - genetic programming
  - interpretable machine learning
  - WebAssembly
  - embedded systems
authors:
  - name: Fatih Küçükkarakurt
    orcid: 0009-0009-0865-3958
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 1 September 2026
bibliography: paper.bib
---

# Summary

`nerve_discover.h` is a symbolic regression engine: given a table of measured
inputs and one measured output, it searches the space of closed-form
expressions and returns the simplest formula that explains the data. Where a
regression model answers a question with parameters, this answers it with an
equation — something a reader can check against theory, differentiate by hand,
argue with, put in a paper, and afterwards evaluate in nanoseconds.

The distinguishing property of the implementation is what it needs in order to
run: a C compiler. It is one header file of roughly 1,000 lines, depends on
nothing but the C standard library and `libm`, requires no build system, and
compiles as C99 on any platform that has a compiler at all. Dropping the file
into a project and writing `#include` is the entire integration step. The same
source compiles to a WebAssembly module of about 70 KB, which — because an
equation discoverer carries no trained weights — is the complete artifact,
with no model download of any kind.

It is part of Nerve, a family of single-header C libraries covering the modern
machine-learning stack.

# Statement of need

Symbolic regression is one of the few machine-learning methods whose output is
directly interpretable, and it has a long record in the physical sciences
[@schmidt2009; @brunton2016; @udrescu2020]. Its available implementations,
however, all assume a scientific Python or Julia installation. PySR
[@cranmer2023], the current reference tool, is a Python package over a Julia
backend; AI Feynman [@udrescu2020] is Python with a neural-network dependency;
the methods surveyed in SRBench [@lacava2021] are almost uniformly Python
libraries. Each is a capable research tool. None can be embedded.

That gap matters wherever the interpreter cannot go. An instrument that fits
its own calibration curve in the field, a microcontroller distilling a lookup
table into an expression, an engineering desktop application that must ship as
one binary with no runtime, a data-processing pipeline in C or C++, a
regulated environment where every dependency must be audited, a browser page
that must not upload the data it is analysing — in all of these the barrier is
not the algorithm but the stack it arrives in. A dependency-free C header
removes the barrier entirely, and it does so without giving up the search
quality that makes the method worth using.

There is a second, quieter need. Symbolic regression is taught far less than
it deserves partly because its implementations are large and layered. An
engine that a graduate student can read end to end in an afternoon, and modify
without touching a build system, is useful as an object of study as well as a
tool.

# Implementation

The search is genetic programming over expression trees [@koza1992], with the
components that separate a working engine from a demonstration:

- **Linear scaling** [@keijzer2003]. Every candidate $f$ is scored as its
  best-fitting $af(x)+b$, solved in closed form from the residuals. The search
  therefore spends itself entirely on structure and never on the two
  parameters that usually dominate the error.
- **An island model.** Several sub-populations evolve independently and
  exchange migrants periodically, which is what prevents the collapse onto a
  single lineage that limits naive genetic programming. The islands also carry
  graduated parsimony pressures, so some pursue accuracy and others brevity;
  between them they populate the accuracy–complexity trade-off.
- **A Pareto front as the result.** `nd_fit` returns the best equation found at
  every complexity, not one answer. The caller can read down the front to see
  exactly what each additional node bought, and `nd_knee` selects the simplest
  equation that concedes no more than a given fraction of the accuracy the
  front spans. Reporting a single most-accurate candidate is how a symbolic
  regressor quietly turns into an unreadable one.
- **Algebraic simplification.** Constant subtrees and the identities
  $x\cdot 1$, $x+0$, $x-x$, $x/x$ are folded away as candidates are formed,
  which shortens the answers and blunts the bloat that otherwise dominates a
  long run.
- **Derivative-free constant polish.** A compass search over a candidate's
  constants, run on admission to the archive, recovers real constants without
  the divergence a numerical-gradient step suffers on stiff expressions. A
  reflection probe precedes it, because a constant born with the wrong sign is
  the one case coordinate descent cannot repair: moving $c$ from $+2$ to
  $-0.5$ in $\exp(cx^2)$ has to cross $c=0$, where the expression collapses to
  a constant and the error is at a local maximum.
- **A compiled evaluator.** Trees are flattened to a postfix instruction array
  once per fitness evaluation and executed over the rows with an explicit
  stack, so the inner loop walks contiguous memory rather than chasing child
  pointers once per row.

Every protected primitive is total — no domain error, no NaN, no trap — so a
candidate that strays outside a function's domain is merely a poor candidate.
The search is deterministic: a given seed reproduces a given run exactly, on
every platform.

# Performance

The engine is evaluated on the 100 equations of the Feynman Lectures
[@feynman1963] catalogued by @udrescu2020, which is the accuracy track of
SRBench [@lacava2021]. Each equation is sampled at 300 training and 300
held-out rows, noise-free; every equation receives an identical budget, an
identical operator set, and a single run with no restarts. The reported answer
is the knee of the front, never the largest candidate on it. Following
SRBench, an equation counts as **solved** when its held-out $R^2 \geq 0.999$.

| | |
|---|---|
| solved ($R^2 \geq 0.999$) | **82 / 100** |
| close ($R^2 \geq 0.99$) | 91 / 100 |
| median $R^2$ | 1.000000 |
| mean search time per equation | 8.5 s (single core, Intel Core i7-7700HQ) |

The benchmark is in the repository, runs from a single `make`, needs no
download, and reports the full per-equation table. Two of the equations require
an arcsine, which is absent from the operator set and so cannot be written
exactly; they remain in the set, scored on the same criterion as the rest,
rather than being excluded.

The accuracy criterion measures prediction on held-out data, not algebraic
identity with the textbook form — which is what SRBench measures, and what is
mechanically checkable. In the clean cases the two coincide and visibly so, as
when `I.6.2a` returns `0.398942*exp(-0.5*theta^2)` and that leading constant is
$1/\sqrt{2\pi}$ to six figures; in others the engine has found a close
approximation over the sampled interval. The per-equation table prints what was
actually recovered, so the distinction is visible rather than averaged away.

These numbers should be read for what they are. This is a compact engine given
a modest single-run budget on one core, not a distributed search; the
established Python and C++ tools surveyed in SRBench reach higher recovery
rates when given the compute they are designed for. The claim here is that
respectable search quality is available with no dependencies at all, in a form
that runs where those tools cannot.

# Availability

`nerve_discover.h` is released under the Apache-2.0 licence as part of Nerve,
at <https://github.com/fkkarakurt/nerve>. The repository carries the benchmark,
a test suite exercised under AddressSanitizer and UndefinedBehaviorSanitizer,
and a browser demonstration in which the engine — with no data file to fetch —
recovers Kepler's third law from the nine measured planetary orbits in under a
second, entirely on the visitor's own machine.

# References
