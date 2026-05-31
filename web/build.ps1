# Build the Nerve WebAssembly demo.
#   - copies the small int8 model + tokenizer into web/ (preloaded into the page)
#   - compiles nerve_web.c -> nerve.js / nerve.wasm / nerve.data with Emscripten
#
# Prereq: Emscripten SDK installed & activated (see emsdk). Override its location
# with $env:EMSDK_ROOT; defaults to a sibling 'emsdk' folder next to the repo.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$emsdk = if ($env:EMSDK_ROOT) { $env:EMSDK_ROOT } else { (Resolve-Path "$PSScriptRoot\..\..\emsdk").Path }
$envps = Join-Path $emsdk "emsdk_env.ps1"
if (-not (Test-Path $envps)) { throw "emsdk_env.ps1 not found at $envps (set `$env:EMSDK_ROOT)" }
. $envps

# bring the model + tokenizer in for preloading (gitignored; produced under studies/infer)
Copy-Item ..\studies\infer\model_q8.nrv .\model_q8.nrv -Force
Copy-Item ..\studies\infer\nerve.tok    .\nerve.tok    -Force

emcc nerve_web.c -O3 -msimd128 -o nerve.js `
  -s "EXPORTED_FUNCTIONS=['_nerve_web_init','_nerve_web_generate','_nerve_web_ctx']" `
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']" `
  -s ALLOW_MEMORY_GROWTH=1 `
  -s INITIAL_MEMORY=67108864 `
  --preload-file model_q8.nrv --preload-file nerve.tok `
  -lm

Write-Output "built nerve.js / nerve.wasm / nerve.data"
Write-Output "serve with:  python -m http.server 8000   then open http://localhost:8000/"
