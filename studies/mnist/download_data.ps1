# Fetch the MNIST IDX files into ./data  (Windows / PowerShell)
# Usage:  ./download_data.ps1
$ErrorActionPreference = "Stop"
$data = Join-Path $PSScriptRoot "data"
$base = "https://ossci-datasets.s3.amazonaws.com/mnist"
$files = @(
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte"
)
New-Item -ItemType Directory -Force $data | Out-Null
$ProgressPreference = "SilentlyContinue"
foreach ($f in $files) {
    if (Test-Path "$data\$f") { "skip $f (already present)"; continue }
    $gz = "$data\$f.gz"
    Invoke-WebRequest -Uri "$base/$f.gz" -OutFile $gz -UseBasicParsing
    $in  = [System.IO.File]::OpenRead($gz)
    $out = [System.IO.File]::Create("$data\$f")
    $gzs = New-Object System.IO.Compression.GzipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
    $gzs.CopyTo($out); $gzs.Close(); $out.Close(); $in.Close()
    Remove-Item $gz
    "{0,-26} {1,12:N0} bytes" -f $f, (Get-Item "$data\$f").Length
}
"MNIST ready in $data"
