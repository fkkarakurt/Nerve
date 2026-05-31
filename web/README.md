# Nerve — in the browser (WebAssembly)

The same zero-dependency C engine (`studies/infer/nerve_infer.h`), compiled to
WebAssembly, running a real language model **entirely in your browser** — no
server, no GPU, no data leaving the page. Click, type, watch it write.

This is a *showcase*, kept modular so it can graduate into its own repo / a
GitHub Pages site without touching the core library.

## Build

Requires the [Emscripten SDK](https://emscripten.org/) installed and activated.

```powershell
# from web/
./build.ps1            # set $env:EMSDK_ROOT if emsdk isn't the repo's sibling
```

This produces `nerve.js`, `nerve.wasm`, and `nerve.data` (the preloaded ~15 MB
int8 model + tokenizer). They are generated artifacts and are gitignored.

## Run

A browser can't `fetch` the `.data` file over `file://`, so serve over HTTP:

```sh
python -m http.server 8000
# open http://localhost:8000/
```

## Files

| File | What it is |
|------|------------|
| `nerve_web.c`  | WASM entry points; reuses the engine unchanged, streams tokens to JS. |
| `index.html`   | The demo page (prompt → in-browser generation). |
| `build.ps1`    | Copies the model in and runs `emcc`. |

The model used here is the small `stories15M` (int8, ~15 MB) so the page loads
fast. The full 1.1B model runs on the desktop/CLI, not in a browser tab.
