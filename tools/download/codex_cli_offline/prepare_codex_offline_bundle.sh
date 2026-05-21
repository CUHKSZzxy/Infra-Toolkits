#!/usr/bin/env bash
set -euo pipefail

PACKAGE_SPEC="@openai/codex"
OUTPUT_DIR="$PWD/codex-offline-artifacts"
NODE_ARCHIVE=""
KEEP_WORK=0
FROM_EXISTING=0
CODEX_PACKAGE_DIR=""

usage() {
  cat <<'EOF'
Usage:
  prepare_codex_offline_bundle.sh [options]

Run this script on a machine with internet access, or use --from-existing
when @openai/codex is already installed locally. The machine should have the
same OS/CPU architecture as the restricted dev environment.

Options:
  --package SPEC        npm package spec to install. Default: @openai/codex
                        Example: @openai/codex@latest
  --output-dir DIR      Directory for the final upload bundle.
                        Default: ./codex-offline-artifacts
  --node-archive PATH   Optional Node runtime archive to include.
                        Supported: .tar.xz, .tar.gz, .tgz
  --from-existing       Build from the locally installed @openai/codex package
                        instead of running npm install. Useful on restricted
                        machines when Codex is already present.
  --codex-package-dir DIR
                        Package directory to use with --from-existing.
                        Default: $(npm root -g)/@openai/codex
  --keep-work           Keep the temporary work directory for inspection.
  -h, --help            Show this help.

Typical:
  ./prepare_codex_offline_bundle.sh \
    --node-archive ~/Downloads/node-v*-linux-x64.tar.xz

Output:
  A codex-cli-offline-bundle-*.tar.gz file. Upload that file to the restricted
  dev environment and run install_codex_offline_bundle.sh there.
EOF
}

say() {
  printf '[prepare] %s\n' "$*"
}

die() {
  printf '[prepare] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

cmd_version() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    "$cmd" --version 2>/dev/null || printf 'unknown\n'
  else
    printf 'unknown\n'
  fi
}

warn() {
  printf '[prepare] warning: %s\n' "$*" >&2
}

node_major() {
  local version
  version="$(node --version 2>/dev/null || true)"
  version="${version#v}"
  printf '%s\n' "${version%%.*}"
}

detect_existing_package_dir() {
  if [[ -n "$CODEX_PACKAGE_DIR" ]]; then
    printf '%s\n' "$CODEX_PACKAGE_DIR"
    return
  fi

  need_cmd npm
  local npm_root
  npm_root="$(npm root -g)"
  printf '%s\n' "$npm_root/@openai/codex"
}

find_first() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -path "$pattern" -print -quit 2>/dev/null || true
}

codex_version_line() {
  local output="$1"
  local line
  while IFS= read -r line; do
    if [[ "$line" == *"codex-cli "* ]]; then
      printf '%s\n' "$line"
      return 0
    fi
  done <<< "$output"
  return 1
}

is_codex_version_output() {
  codex_version_line "$1" >/dev/null
}

copy_existing_codex_prefix() {
  local package_dir
  package_dir="$(detect_existing_package_dir)"
  [[ -d "$package_dir" ]] || die "Codex package directory not found: $package_dir"
  [[ -f "$package_dir/package.json" ]] || die "missing package.json in $package_dir"

  say "copying existing Codex package from $package_dir"
  mkdir -p "$prefix_dir/lib/node_modules/@openai" "$prefix_dir/bin"
  cp -a "$package_dir" "$prefix_dir/lib/node_modules/@openai/codex"
  ln -s ../lib/node_modules/@openai/codex/bin/codex.js "$prefix_dir/bin/codex"
}

check_prepared_codex() {
  local codex_version=""
  local native_bin
  native_bin="$(find_first "$prefix_dir/lib/node_modules/@openai/codex" '*/vendor/*/codex/codex')"

  if [[ -x "$prefix_dir/bin/codex" ]]; then
    codex_version="$("$prefix_dir/bin/codex" --version 2>&1 || true)"
  fi

  if ! is_codex_version_output "$codex_version" && [[ -n "$native_bin" ]]; then
    codex_version="$("$native_bin" --version 2>&1 || true)"
  fi

  if ! is_codex_version_output "$codex_version"; then
    if [[ -n "$native_bin" ]]; then
      warn "native Codex binary exists, but version check did not produce expected output"
      codex_version="unknown"
    else
      printf '%s\n' "$codex_version" >&2
      die "Codex did not run from the prepared prefix and no native binary fallback was found"
    fi
  fi

  codex_version_line "$codex_version"
}

copy_node_archive() {
  local src="$1"
  local dest_dir="$2"
  local dest

  case "$src" in
    *.tar.xz) dest="$dest_dir/node-runtime.tar.xz" ;;
    *.tar.gz) dest="$dest_dir/node-runtime.tar.gz" ;;
    *.tgz) dest="$dest_dir/node-runtime.tgz" ;;
    *) die "--node-archive must be .tar.xz, .tar.gz, or .tgz: $src" ;;
  esac

  cp "$src" "$dest"
  printf '%s\n' "$(basename "$dest")"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package)
      [[ $# -ge 2 ]] || die "--package requires a value"
      PACKAGE_SPEC="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --node-archive)
      [[ $# -ge 2 ]] || die "--node-archive requires a value"
      NODE_ARCHIVE="$2"
      shift 2
      ;;
    --from-existing)
      FROM_EXISTING=1
      shift
      ;;
    --codex-package-dir)
      [[ $# -ge 2 ]] || die "--codex-package-dir requires a value"
      CODEX_PACKAGE_DIR="$2"
      shift 2
      ;;
    --keep-work)
      KEEP_WORK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

need_cmd tar
need_cmd uname
need_cmd date
need_cmd mktemp
need_cmd find

if [[ -n "$NODE_ARCHIVE" && ! -f "$NODE_ARCHIVE" ]]; then
  die "node archive not found: $NODE_ARCHIVE"
fi

if [[ "$FROM_EXISTING" -eq 0 ]]; then
  need_cmd npm
  need_cmd node
  major="$(node_major)"
  if [[ -z "$major" || "$major" -lt 16 ]]; then
    die "Node >=16 is required for npm-install mode; found: $(node --version 2>/dev/null || echo missing). Use --from-existing if Codex is already installed locally."
  fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
work_dir="$(mktemp -d)"
cleanup() {
  if [[ "$KEEP_WORK" -eq 1 ]]; then
    say "kept temporary work directory: $work_dir"
  else
    rm -rf "$work_dir"
  fi
}
trap cleanup EXIT

prefix_dir="$work_dir/codex-prefix"
bundle_root="$work_dir/bundle-root"
payload_dir="$bundle_root/payload"

mkdir -p "$prefix_dir" "$payload_dir" "$OUTPUT_DIR"

if [[ "$FROM_EXISTING" -eq 1 ]]; then
  copy_existing_codex_prefix
else
  say "installing $PACKAGE_SPEC into an isolated npm prefix"
  NO_UPDATE_NOTIFIER=1 \
    NPM_CONFIG_AUDIT=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_PROGRESS=false \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    npm install --prefix "$prefix_dir" -g "$PACKAGE_SPEC" --no-audit --no-fund
fi

say "checking bundled Codex CLI"
codex_version="$(check_prepared_codex)"
say "$codex_version"

say "packing Codex prefix"
tar -C "$prefix_dir" -czf "$payload_dir/codex-prefix.tar.gz" .

node_payload=""
if [[ -n "$NODE_ARCHIVE" ]]; then
  say "including Node runtime archive"
  node_payload="$(copy_node_archive "$NODE_ARCHIVE" "$payload_dir")"
fi

if [[ -f "$script_dir/install_codex_offline_bundle.sh" ]]; then
  cp "$script_dir/install_codex_offline_bundle.sh" "$bundle_root/install_codex_offline_bundle.sh"
  chmod +x "$bundle_root/install_codex_offline_bundle.sh"
else
  say "warning: install_codex_offline_bundle.sh not found next to prepare script"
fi
cp "${BASH_SOURCE[0]}" "$bundle_root/prepare_codex_offline_bundle.sh"
chmod +x "$bundle_root/prepare_codex_offline_bundle.sh"

created_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
platform="$(uname -s)"
arch="$(uname -m)"
cat > "$bundle_root/manifest.env" <<EOF
CREATED_UTC=$created_utc
PACKAGE_SPEC=$PACKAGE_SPEC
PREPARE_PLATFORM=$platform
PREPARE_ARCH=$arch
NODE_VERSION=$(cmd_version node)
NPM_VERSION=$(cmd_version npm)
CODEX_VERSION=$codex_version
FROM_EXISTING=$FROM_EXISTING
NODE_PAYLOAD=$node_payload
BUNDLE_KIND=codex-cli
EOF

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
safe_platform="$(printf '%s' "$platform" | tr '[:upper:]' '[:lower:]')"
safe_arch="$(printf '%s' "$arch" | tr '/ ' '__')"
bundle_name="codex-cli-offline-bundle-${safe_platform}-${safe_arch}-${stamp}.tar.gz"
bundle_path="$OUTPUT_DIR/$bundle_name"

say "creating upload bundle: $bundle_path"
tar -C "$bundle_root" -czf "$bundle_path" .

say "done"
say "upload this file to the restricted dev environment:"
say "  $bundle_path"
say "then run:"
say "  ./install_codex_offline_bundle.sh --bundle $bundle_name"
