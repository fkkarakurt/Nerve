#!/bin/sh
# Fetch the MNIST IDX files into ./data  (Linux / macOS)
# Usage:  ./download_data.sh
set -e
cd "$(dirname "$0")"
mkdir -p data
base="https://ossci-datasets.s3.amazonaws.com/mnist"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
    if [ -f "data/$f" ]; then echo "skip $f (already present)"; continue; fi
    echo "downloading $f ..."
    curl -fsSL "$base/$f.gz" -o "data/$f.gz"
    gunzip -f "data/$f.gz"
done
echo "MNIST ready in ./data"
