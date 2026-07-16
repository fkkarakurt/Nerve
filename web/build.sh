#!/bin/sh
# Build the Nerve WebAssembly demo (Linux / macOS / CI).
#
# This is the single source of truth for the Emscripten flags; build.ps1 is the
# Windows twin and must be kept in step with it. CI calls this script, so the
# page that ships is built exactly the way a developer builds it locally.
#
# The model blobs are not in the repository -- they are tens of megabytes of
# converted weights. Supply them one of two ways:
#   * put them in this directory (CI downloads them from the models release), or
#   * produce them under studies/ and let this script copy them across.
#
# Prereq: emcc on PATH (see https://emscripten.org).

set -eu

cd "$(dirname "$0")"

MODELS="model_q8.nrv nerve.tok minilm_q8.nre vocab.txt"

# Fall back to the studies/ working copies when the blobs are not already here.
[ -f model_q8.nrv  ] || cp ../studies/infer/model_q8.nrv  . 2>/dev/null || true
[ -f nerve.tok     ] || cp ../studies/infer/nerve.tok     . 2>/dev/null || true
[ -f minilm_q8.nre ] || cp ../studies/embed/minilm_q8.nre . 2>/dev/null || true
[ -f vocab.txt     ] || cp ../studies/embed/vocab.txt     . 2>/dev/null || true

missing=
for f in $MODELS; do
    [ -f "$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
    echo "error: missing model blobs:$missing" >&2
    echo "       fetch them from the models release, or build them under studies/." >&2
    exit 1
fi

emcc nerve_web.c -O3 -msimd128 -o nerve.js \
  -sMODULARIZE=1 -sEXPORT_NAME=createNerve -sENVIRONMENT=web \
  -sEXPORTED_FUNCTIONS=_nerve_web_init,_nerve_web_generate,_nerve_web_ctx,_nerve_web_dim,_nerve_web_embed,_nerve_web_gen_start,_nerve_web_gen_step,_nerve_web_gen_done \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPF32 \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=134217728 \
  --preload-file model_q8.nrv --preload-file nerve.tok \
  --preload-file minilm_q8.nre --preload-file vocab.txt \
  -lm

echo "built nerve.js / nerve.wasm / nerve.data"
echo "serve with:  python3 -m http.server 8000   then open http://localhost:8000/"
