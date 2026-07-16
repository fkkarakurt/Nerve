# Nerve (JS/TS) — on-device AI in WebAssembly

A real Transformer text generator **and** a MiniLM sentence-embedding model,
running entirely in WebAssembly — in the browser or Node. No server, no GPU, no
API key, no per-token cost. The data never leaves the machine.

Built from [Nerve](https://github.com/fkkarakurt/nerve): zero-dependency neural
networks in pure C, compiled to a ~65 KB WASM engine.

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

## License

Apache-2.0 — see the main repository.
