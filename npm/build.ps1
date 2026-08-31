# Build the npm package's WASM artifacts:
#   nerve.mjs / nerve.wasm       the LLM + sentence encoder (models embedded)
#   discover.mjs / discover.wasm the equation discoverer (no models at all)
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
# --embed-file bakes the data INTO the .wasm (no separate .data, no path lookup),
# so it works after `npm install` from any directory, in Node and the browser.
& $emcc nerve_web.c -O3 -msimd128 -o ..\npm\nerve.mjs `
  '-sMODULARIZE=1' '-sEXPORT_ES6=1' '-sEXPORT_NAME=createNerve' '-sENVIRONMENT=web,node' `
  $EF '-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPF32' '-sALLOW_MEMORY_GROWTH=1' `
  --embed-file model_q8.nrv --embed-file nerve.tok `
  --embed-file minilm_q8.nre --embed-file vocab.txt -lm
Pop-Location

# The discoverer has no weights, so it needs no staging and no --embed-file.
& $emcc (Join-Path $PSScriptRoot "nerve_discover.c") -O3 -Wall -Wextra -o (Join-Path $PSScriptRoot "discover.mjs") `
  '-sMODULARIZE=1' '-sEXPORT_ES6=1' '-sEXPORT_NAME=createDiscover' '-sENVIRONMENT=web,node' `
  '-sEXPORTED_FUNCTIONS=_main,_malloc,_free,_nd_json_fit,_nd_json_version' `
  '-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPF64,lengthBytesUTF8,stringToUTF8' `
  '-sALLOW_MEMORY_GROWTH=1' -lm

Get-ChildItem (Join-Path $PSScriptRoot "nerve.wasm") (Join-Path $PSScriptRoot "discover.wasm") |
  Select-Object Name, Length
Write-Output "built npm/nerve.mjs and npm/discover.mjs. Now: cd npm && npm publish"
