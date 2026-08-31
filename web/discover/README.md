# Nerve Discover — in the browser

Paste a table of numbers, get back an equation. The whole symbolic-regression
engine compiled to WebAssembly, running on the visitor's own machine.

```
nerve_discover.wasm    ~70 KB
nerve_discover.js      ~12 KB
data files             none
```

That last line is the point. Every other in-browser AI demo begins with a
download — tens or hundreds of megabytes of weights before anything can happen.
Equation discovery has no weights. The algorithm *is* the product, so the page
is usable the moment it loads, works offline afterwards, and never sends a
single row of your data anywhere.

## Running it

```sh
# from the repository root
python -m http.server 8000
# then open http://localhost:8000/web/discover/
```

It has to be served over `http://`. Opening `index.html` from the filesystem
leaves the browser unable to start the worker, and the page will say so.

## Building

```sh
./build.sh          # needs an activated emsdk
```

or on Windows:

```powershell
$env:EMSDK_ROOT = "path\to\emsdk"
.\build.ps1
```

Both produce `nerve_discover.js` and `nerve_discover.wasm` next to
`index.html`. They are build outputs and are not committed.

## How it fits together

| file | what it is |
|------|-----------|
| `discover_web.c` | the WebAssembly entry points: CSV parsing, and a thin wrapper over `nd_fit`. Progress is pushed to JavaScript from inside the search loop via `EM_JS`. |
| `worker.js` | starts the module and runs the search. |
| `index.html` | the page: sample datasets, the operator picker, the live equation, and the Pareto front. |

The search is a blocking loop over hundreds of generations, which on the main
thread would be a frozen tab. It runs in a Web Worker instead, posting a
message every few generations, so the equation visibly sharpens while the page
stays responsive.

## The sample datasets

Five are built in. The first is the interesting one:

- **Kepler · real planets** — the nine bodies of the solar system as actually
  measured: semi-major axis in AU against orbital period in years. Nine rows.
  The engine is given no theory and no hint, and returns `T ≈ a·√a` — the third
  law — in under a second.
- **Newton · gravitation**, **Pendulum · period**, **Air resistance** —
  synthetic samples from known laws with 1–3% noise added, to show what the
  engine does when the measurements are not clean.
- **Planck · blackbody** — deliberately included as a *hard* one. It has an
  exponential in the denominator, and the engine gets to R² ≈ 0.9999 with a
  close but not exact form. A demo that only shows its wins is a brochure.

Or paste your own: inputs in the first columns, the value to explain in the
last, one row per line. A header row names the variables in the answer.

## What to look at

Not the single answer — the **front**. Every row is the best equation found at
that size, so reading down it you can see exactly what each additional term
bought. For the planets it goes 0.978 → 0.988 → 1.000, and then five more rows
that spend seventeen extra nodes to move the eighth decimal place. The
highlighted row is where the buying stops, and that is almost always the row
that is actually a law.
