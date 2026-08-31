#!/bin/sh
# Build the Nerve Discover WebAssembly demo.
#
# Unlike the other demo in this repo there is nothing to preload: an equation
# discoverer carries no weights, so the whole deliverable is one .js shim and
# one .wasm module of a few tens of kilobytes.
#
# Prereq: an activated Emscripten SDK (`source /path/to/emsdk/emsdk_env.sh`).
set -eu
cd "$(dirname "$0")"

emcc discover_web.c -O3 -o nerve_discover.js \
  -sMODULARIZE=1 \
  -sEXPORT_NAME=createNerveDiscover \
  -sENVIRONMENT=worker \
  -sEXPORTED_FUNCTIONS=_main,_ndw_load_csv,_ndw_run,_ndw_error,_ndw_rows,_ndw_vars,_ndw_var_name \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString \
  -sALLOW_MEMORY_GROWTH=1 \
  -sINITIAL_MEMORY=33554432 \
  -lm

ls -l nerve_discover.js nerve_discover.wasm
echo
echo "serve with:  python -m http.server 8000"
echo "then open :  http://localhost:8000/web/discover/   (from the repo root)"
