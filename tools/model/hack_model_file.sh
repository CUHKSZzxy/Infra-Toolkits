#!/usr/bin/env bash
set -euo pipefail

# Configuration
model_path="${model_path:-/path/to/source}"
new_model_path="${new_model_path:-/path/to/target}"
exclude_file="${exclude_file:-config.json}"  # File to copy (not symlink)

# Create target directory
mkdir -p "$new_model_path"

# Copy excluded file if present
if [[ -e "$model_path/$exclude_file" ]]; then
    cp -f "$model_path/$exclude_file" "$new_model_path/"
fi

# Symlink remaining files/dirs
find "$model_path" -mindepth 1 -maxdepth 1 ! -name "$exclude_file" \
    -exec ln -sf {} "$new_model_path/" \;
