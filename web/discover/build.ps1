# Build the Nerve Discover WebAssembly demo.
#
# Unlike the other demo in this repo there is nothing to preload: an equation
# discoverer carries no weights, so the whole deliverable is one .js shim and
# one .wasm module of a few tens of kilobytes.
#
# Prereq: Emscripten SDK. Override its location with $env:EMSDK_ROOT; defaults
# to a sibling 'emsdk' folder next to the repo.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$emsdk = if ($env:EMSDK_ROOT) { $env:EMSDK_ROOT } else { (Resolve-Path "$PSScriptRoot\..\..\..\emsdk").Path }
$envps = Join-Path $emsdk "emsdk_env.ps1"
if (-not (Test-Path $envps)) { throw "emsdk_env.ps1 not found at $envps (set `$env:EMSDK_ROOT)" }
. $envps

# emsdk_env.ps1 rewrites PATH into an MSYS-style string, so `emcc` alone is not
# resolvable afterwards; call the batch shim by its full path instead.
$emcc = Join-Path $emsdk "upstream\emscripten\emcc.bat"

& $emcc discover_web.c -O3 -o nerve_discover.js `
  '-sMODULARIZE=1' `
  '-sEXPORT_NAME=createNerveDiscover' `
  '-sENVIRONMENT=worker' `
  '-sEXPORTED_FUNCTIONS=_main,_ndw_load_csv,_ndw_run,_ndw_error,_ndw_rows,_ndw_vars,_ndw_var_name' `
  '-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString' `
  '-sALLOW_MEMORY_GROWTH=1' `
  '-sINITIAL_MEMORY=33554432' `
  -lm

Get-ChildItem nerve_discover.js, nerve_discover.wasm | Select-Object Name, Length
Write-Output ""
Write-Output "serve with:  python -m http.server 8000"
Write-Output "then open :  http://localhost:8000/web/discover/   (from the repo root)"
