#!/bin/bash
# Batch convert all SVG files in current directory to EPS

for file in *.svg; do
    out="${file%.svg}.eps"
    inkscape "$file" --export-type=eps -o "$out"
    echo "Converted $file to $out"
done
