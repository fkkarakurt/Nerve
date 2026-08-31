# Nerve (JS/TS) — on-device AI in WebAssembly

An **equation discoverer**, a real Transformer text generator, and a MiniLM
sentence-embedding model, running entirely in WebAssembly — in the browser or
Node. No server, no GPU, no API key, no per-token cost. The data never leaves
the machine.

Built from [Nerve](https://github.com/fkkarakurt/nerve): zero-dependency AI in
pure C.

## Install

```sh
npm install @fkkarakurt/nerve
```

## Use

```js
import Nerve from "@fkkarakurt/nerve";

const nerve = await Nerve.load();

// 1. generate text (streams)
const story = nerve.generate("Once upon a time", {
  steps: 120,
  onToken: t => process.stdout.write(t),
});

// 2. understand meaning
nerve.similarity("a puppy on the grass", "a young dog in the park"); // ~0.7

// 3. learn your own categories (on-device, in ms)
nerve.teach([
  { text: "schedule a meeting",     label: "calendar" },
  { text: "i want a hamburger",     label: "food" },
  { text: "go for a run",           label: "fitness" },
]);
nerve.classify("i'm hungry for pizza"); // { label: "food", confidence: 0.8, scores: {...} }

// 4. semantic search over your own notes
nerve.index(["The capital of France is Paris.", "Coffee contains caffeine."]);
nerve.search("what keeps me awake?"); // [{ text: "Coffee contains caffeine.", score: 0.4 }, ...]
```

## API

| Method | Description |
|--------|-------------|
| `Nerve.load()` | Load the models; returns a ready instance. |
| `embed(text)` | Sentence → L2-normalised `Float32Array`. |
| `similarity(a, b)` | Cosine similarity (-1..1). |
| `generate(prompt, opts)` | Generate text; `opts.onToken` streams. |
| `teach(examples)` | Train a classifier on `{text, label}` examples. |
| `classify(text)` | `{ label, confidence, scores }`. |
| `index(notes)` / `search(query, k)` | Semantic search over your notes. |

Generation is synchronous (runs to completion); for long outputs in the browser,
call it from a Web Worker to keep the UI responsive.

## Give it numbers, get back an equation

A separate entry point, and the interesting one: symbolic regression. Hand it a
table and it returns a **formula** — not a prediction, not weights, but
something you can read, check against theory, and evaluate in nanoseconds
forever after.

```js
import { load, Ops } from "@fkkarakurt/nerve/discover";

const d = await load();

// The nine planets, as actually measured: distance in AU, period in years.
const { knee, front } = d.fromTable(`a,T
0.387,0.2408
0.723,0.6152
1.000,1.0000
1.524,1.8808
5.204,11.862
9.583,29.457
19.191,84.016
30.070,164.79
39.482,247.94`, { ops: Ops.ALGEBRA });

console.log(knee.equation);   // y = 0.999498*sqrt(a)*a - 0.0274583
console.log(knee.r2);         // 1.0        <- Kepler's third law
```

`front` is the whole accuracy/complexity trade-off — the best equation the
search found at *every* size, simplest first — so you can see exactly what each
extra term bought. `knee` is the point where it stops buying anything, which is
almost always the one that is actually a law.

Arrays work too, if the data is already in memory:

```js
d.fit({ X, y, names: ["width", "height"] },   // X row-major
      { ops: Ops.ARITH, generations: 250, seed: 1 });
// -> y = 2*height*width + 5
```

| | |
|--|--|
| `load()` | Load the engine. **Nothing to download** — an equation discoverer has no weights, so this is a ~62 KB module and it starts instantly. |
| `fromTable(csv, opts?)` | Discover from a CSV string: inputs first, the value to explain last, optional header row for names. |
| `fit({X, y, names}, opts?)` | Discover from arrays. |
| `parseTable(csv)` | Just the parser, if you want the arrays yourself. |
| `Ops.ARITH` / `ALGEBRA` / `DEFAULT` / `ALL` | Which operators the search may use. Fewer operators, smaller haystack — keep it `ALGEBRA` unless the thing really oscillates or decays. |

Options: `ops`, `generations` (250), `maxNodes` (24), `population` (300),
`seed` (1 — same seed, same answer), `kneeSlack` (0.02).

The search is synchronous and CPU-bound. In a browser, run it in a Web Worker.

How good is it? Measured on the 100 Feynman equations, the standard benchmark
for this field — [the numbers and the protocol are in the
repository](https://github.com/fkkarakurt/nerve/tree/main/bench/feynman).

## License

Apache-2.0 — see the main repository.
