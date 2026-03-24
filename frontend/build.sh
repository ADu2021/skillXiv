#!/bin/bash
# SkillXiv full build: index skills → build frontend → copy to dist/
# Run from the frontend/ directory: ./build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TMP_BUILD="/tmp/skillxiv-build"

echo "=== Step 1: Build skills index ==="
python3 "$SCRIPT_DIR/build_index.py"

echo ""
echo "=== Step 2: Build frontend ==="
# Use a temp dir for vite build to avoid mounted-fs permission issues
rm -rf "$TMP_BUILD"
mkdir -p "$TMP_BUILD"

# Copy source files to temp
cp -r "$SCRIPT_DIR/src" "$TMP_BUILD/"
cp -r "$SCRIPT_DIR/public" "$TMP_BUILD/"
cp "$SCRIPT_DIR/index.html" "$TMP_BUILD/"
cp "$SCRIPT_DIR/vite.config.js" "$TMP_BUILD/"
cp "$SCRIPT_DIR/package.json" "$TMP_BUILD/"

# Install deps and build in temp
cd "$TMP_BUILD"
npm install --silent 2>/dev/null
npx vite build 2>&1

echo ""
echo "=== Step 3: Copy dist to workspace ==="
mkdir -p "$SCRIPT_DIR/dist"
cp -r "$TMP_BUILD/dist/"* "$SCRIPT_DIR/dist/"

echo ""
echo "=== Build complete ==="
echo "Output: $SCRIPT_DIR/dist/"
