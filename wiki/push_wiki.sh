#!/usr/bin/env bash
# Run this script once to push all wiki pages to GitHub.
# Usage: bash wiki/push_wiki.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WIKI_DIR="$REPO_DIR/wiki"
WIKI_REMOTE="https://github.com/fkkarakurt/nerve.wiki.git"
TMP_DIR="$(mktemp -d)"

echo "Cloning wiki repo..."
git clone "$WIKI_REMOTE" "$TMP_DIR"

echo "Copying pages..."
cp "$WIKI_DIR/Home.md"            "$TMP_DIR/Home.md"
cp "$WIKI_DIR/Getting-Started.md" "$TMP_DIR/Getting-Started.md"
cp "$WIKI_DIR/Core-API.md"        "$TMP_DIR/Core-API.md"
cp "$WIKI_DIR/Easy-API.md"        "$TMP_DIR/Easy-API.md"
cp "$WIKI_DIR/Algorithms.md"      "$TMP_DIR/Algorithms.md"
cp "$WIKI_DIR/_Footer.md"         "$TMP_DIR/_Footer.md"

echo "Removing stale pages..."
rm -f "$TMP_DIR/Description.md"
rm -f "$TMP_DIR/File-Reference.md"
rm -f "$TMP_DIR/Synopsis.md"

echo "Committing..."
cd "$TMP_DIR"
git add -A
git commit -m "docs: complete wiki rewrite for Nerve 2.0.0

- Home: overview, feature table, benchmark, quick links
- Getting-Started: install, easy API + core API first program, config table
- Core-API: all net_* functions grouped by category (enums, alloc,
  init, config, query, weight access, forward, train, metrics,
  persistence, structural modification, utility)
- Easy-API: nerve_config_t, nerve_new/new_ex, nerve_fit/verbose,
  nerve_score, macro aliases, complete Iris example
- Algorithms: forward pass, backprop, activations, SGD, Adam,
  Xavier/He init, L2, dropout, universal approximation, 8 references
- Removed stale Description / File-Reference / Synopsis pages
  (all content superseded by new pages)"

echo "Pushing..."
git push origin master

echo "Done. Wiki updated at https://github.com/fkkarakurt/nerve/wiki"
rm -rf "$TMP_DIR"
