# Build the npm package's WASM artifacts (nerve.mjs / .wasm / .data).
# These are generated (gitignored) but ARE shipped on `npm publish` via the
# package.json "files" field. Run this before publishing.
#
# Prereq: Emscripten SDK (see ../emsdk). Models come from ../studies (gitignored).
$ErrorActionPreference = "Stop"
$emsdk = if ($env:EMSDK_ROOT) { $env:EMSDK_ROOT } else { (Resolve-Path "$PSScriptRoot\..\..\emsdk").Path }
. (Join-Path $emsdk "emsdk_env.ps1") | Out-Null
$emcc = Join-Path $emsdk "upstream\emscripten\emcc.bat"

# stage models next to nerve_web.c (in web/), build ES6 module into npm/
Push-Location "$PSScriptRoot\..\web"
Copy-Item ..\studies\infer\model_q8.nrv  . -Force
Copy-Item ..\studies\infer\nerve.tok     . -Force
Copy-Item ..\studies\embed\minilm_q8.nre . -Force
Copy-Item ..\studies\embed\vocab.txt     . -Force
$EF = "-sEXPORTED_FUNCTIONS=_nerve_web_init,_nerve_web_dim,_nerve_web_embed,_nerve_web_gen_start,_nerve_web_gen_step,_nerve_web_gen_done"
& $emcc nerve_web.c -O3 -msimd128 -o ..\npm\nerve.mjs `
  '-sMODULARIZE=1' '-sEXPORT_ES6=1' '-sEXPORT_NAME=createNerve' '-sENVIRONMENT=web,node' `
  $EF '-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPF32' '-sALLOW_MEMORY_GROWTH=1' `
  --preload-file model_q8.nrv --preload-file nerve.tok `
  --preload-file minilm_q8.nre --preload-file vocab.txt -lm
Pop-Location
Write-Output "built npm/nerve.mjs (+ .wasm, .data). Now: cd npm && npm publish"
