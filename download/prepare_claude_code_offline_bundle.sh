#!/usr/bin/env bash
set -euo pipefail

PACKAGE_SPEC="@anthropic-ai/claude-code"
OUTPUT_DIR="$PWD/claude-code-offline-artifacts"
KEEP_WORK=0
FROM_EXISTING=0
CLAUDE_PACKAGE_DIR=""
NODE_BIN_DIR=""

usage() {
  cat <<'EOF'
Usage:
  prepare_claude_code_offline_bundle.sh [options]

Run this script on a machine with internet access, or use --from-existing
when @anthropic-ai/claude-code is already installed with npm. The machine
should have the same OS/CPU architecture as the restricted dev environment,
because Claude Code uses a platform-specific native binary.

Options:
  --package SPEC        npm package spec to install.
                        Default: @anthropic-ai/claude-code
                        Example: @anthropic-ai/claude-code@latest
  --output-dir DIR      Directory for the final upload bundle.
                        Default: ./claude-code-offline-artifacts
  --from-existing       Build from the locally installed npm package instead
                        of running npm install.
  --claude-package-dir DIR
                        Package directory to use with --from-existing.
                        Default: $(npm root -g)/@anthropic-ai/claude-code
  --node-bin-dir DIR    Directory containing Node >=18 and npm. Prepended to
                        PATH for npm-install mode when default node is old.
  --keep-work           Keep the temporary work directory for inspection.
  -h, --help            Show this help.

Typical:
  ./prepare_claude_code_offline_bundle.sh \
    --package @anthropic-ai/claude-code@latest

  ./prepare_claude_code_offline_bundle.sh \
    --node-bin-dir ~/.nvm/versions/node/v24.15.0/bin

Output:
  A claude-code-offline-bundle-*.tar.gz file. Upload that file to the
  restricted dev environment and run install_claude_code_offline_bundle.sh.
EOF
}

say() {
  printf '[prepare] %s\n' "$*"
}

warn() {
  printf '[prepare] warning: %s\n' "$*" >&2
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

node_major() {
  local version
  version="$(node --version 2>/dev/null || true)"
  version="${version#v}"
  printf '%s\n' "${version%%.*}"
}

detect_existing_package_dir() {
  if [[ -n "$CLAUDE_PACKAGE_DIR" ]]; then
    printf '%s\n' "$CLAUDE_PACKAGE_DIR"
    return
  fi

  need_cmd npm
  local npm_root
  npm_root="$(npm root -g)"
  printf '%s\n' "$npm_root/@anthropic-ai/claude-code"
}

find_first() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -path "$pattern" -print -quit 2>/dev/null || true
}

find_prepared_claude() {
  local candidate
  for candidate in \
    "$prefix_dir/bin/claude" \
    "$prefix_dir/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  find "$prefix_dir/lib/node_modules/@anthropic-ai" \
    -maxdepth 3 -type f -name claude -perm -111 -print -quit 2>/dev/null || true
}

copy_existing_claude_prefix() {
  local package_dir
  package_dir="$(detect_existing_package_dir)"
  [[ -d "$package_dir" ]] || die "Claude Code package directory not found: $package_dir"
  [[ -f "$package_dir/package.json" ]] || die "missing package.json in $package_dir"

  local source_scope
  source_scope="$(cd "$package_dir/.." && pwd -P)"

  say "copying existing Claude Code npm packages from $source_scope"
  mkdir -p "$prefix_dir/lib/node_modules/@anthropic-ai" "$prefix_dir/bin"

  local copied=0
  local dir
  while IFS= read -r -d '' dir; do
    case "$(basename "$dir")" in
      claude-code|claude-code-*)
        cp -a "$dir" "$prefix_dir/lib/node_modules/@anthropic-ai/"
        copied=$((copied + 1))
        ;;
    esac
  done < <(find "$source_scope" -maxdepth 1 -mindepth 1 -type d -print0)

  [[ "$copied" -gt 0 ]] || die "no Claude Code npm packages found under $source_scope"
  chmod +x "$prefix_dir/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe" 2>/dev/null || true
  ln -s ../lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe "$prefix_dir/bin/claude"
}

check_prepared_claude() {
  local claude_bin
  claude_bin="$(find_prepared_claude)"
  [[ -n "$claude_bin" ]] || die "could not find a prepared Claude Code binary"

  local output
  output="$("$claude_bin" --version 2>&1 || true)"
  if [[ -z "$output" || "$output" != *[0-9]* ]]; then
    printf '%s\n' "$output" >&2
    die "Claude Code did not produce a version from the prepared prefix"
  fi

  printf '%s\n' "$output" | sed -n '1p'
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
    --from-existing)
      FROM_EXISTING=1
      shift
      ;;
    --claude-package-dir)
      [[ $# -ge 2 ]] || die "--claude-package-dir requires a value"
      CLAUDE_PACKAGE_DIR="$2"
      shift 2
      ;;
    --node-bin-dir)
      [[ $# -ge 2 ]] || die "--node-bin-dir requires a value"
      NODE_BIN_DIR="$2"
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

if [[ "$FROM_EXISTING" -eq 0 ]]; then
  if [[ -n "$NODE_BIN_DIR" ]]; then
    [[ -d "$NODE_BIN_DIR" ]] || die "node bin dir not found: $NODE_BIN_DIR"
    [[ -x "$NODE_BIN_DIR/node" ]] || die "node not executable in: $NODE_BIN_DIR"
    [[ -x "$NODE_BIN_DIR/npm" ]] || die "npm not executable in: $NODE_BIN_DIR"
    export PATH="$NODE_BIN_DIR:$PATH"
    hash -r
  fi

  need_cmd npm
  need_cmd node
  major="$(node_major)"
  if [[ -z "$major" || "$major" -lt 18 ]]; then
    die "Node >=18 is required for npm-install mode; found: $(node --version 2>/dev/null || echo missing). Use --node-bin-dir or --from-existing."
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

prefix_dir="$work_dir/claude-code-prefix"
bundle_root="$work_dir/bundle-root"
payload_dir="$bundle_root/payload"

mkdir -p "$prefix_dir" "$payload_dir" "$OUTPUT_DIR"

if [[ "$FROM_EXISTING" -eq 1 ]]; then
  copy_existing_claude_prefix
else
  say "installing $PACKAGE_SPEC into an isolated npm prefix"
  NO_UPDATE_NOTIFIER=1 \
    NPM_CONFIG_AUDIT=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_PROGRESS=false \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_OPTIONAL=true \
    NPM_CONFIG_IGNORE_SCRIPTS=false \
    npm install --prefix "$prefix_dir" -g "$PACKAGE_SPEC" --no-audit --no-fund
fi

say "checking bundled Claude Code"
claude_version="$(check_prepared_claude)"
say "$claude_version"

say "packing Claude Code prefix"
tar -C "$prefix_dir" -czf "$payload_dir/claude-code-prefix.tar.gz" .

if [[ -f "$script_dir/install_claude_code_offline_bundle.sh" ]]; then
  cp "$script_dir/install_claude_code_offline_bundle.sh" "$bundle_root/install_claude_code_offline_bundle.sh"
  chmod +x "$bundle_root/install_claude_code_offline_bundle.sh"
else
  warn "install_claude_code_offline_bundle.sh not found next to prepare script"
fi
cp "${BASH_SOURCE[0]}" "$bundle_root/prepare_claude_code_offline_bundle.sh"
chmod +x "$bundle_root/prepare_claude_code_offline_bundle.sh"

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
CLAUDE_VERSION=$claude_version
FROM_EXISTING=$FROM_EXISTING
BUNDLE_KIND=claude-code-cli
EOF

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
safe_platform="$(printf '%s' "$platform" | tr '[:upper:]' '[:lower:]')"
safe_arch="$(printf '%s' "$arch" | tr '/ ' '__')"
bundle_name="claude-code-offline-bundle-${safe_platform}-${safe_arch}-${stamp}.tar.gz"
bundle_path="$OUTPUT_DIR/$bundle_name"

say "creating upload bundle: $bundle_path"
tar -C "$bundle_root" -czf "$bundle_path" .

say "done"
say "upload this file to the restricted dev environment:"
say "  $bundle_path"
say "then run:"
say "  ./install_claude_code_offline_bundle.sh --bundle $bundle_name"
