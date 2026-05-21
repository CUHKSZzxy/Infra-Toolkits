#!/usr/bin/env bash
set -euo pipefail

PACKAGE_SPEC="@anthropic-ai/claude-code@latest"
INSTALL_ROOT="${CLAUDE_CODE_ONLINE_INSTALL_ROOT:-$HOME/.local/claude-code-online}"
NODE_BIN_DIR=""
UPDATE_PROFILE=1
PROFILE_PATH=""

usage() {
  cat <<'EOF'
Usage:
  install_claude_code_online.sh [options]

Install Claude Code directly from npm into a user-local prefix.

Options:
  --package SPEC      Claude Code npm package.
                      Default: @anthropic-ai/claude-code@latest
  --install-root DIR  Install root. Default: ~/.local/claude-code-online
  --node-bin-dir DIR  Directory containing Node >=18 and npm.
  --profile PATH      Shell profile to update. Default: ~/.zshrc or ~/.bashrc.
  --no-profile        Do not update a shell profile; print the export instead.
  -h, --help          Show this help.

Typical:
  ./install_claude_code_online.sh --node-bin-dir ~/.nvm/versions/node/v24.15.0/bin
EOF
}

say() {
  printf '[claude-online] %s\n' "$*"
}

die() {
  printf '[claude-online] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

choose_profile() {
  if [[ -n "$PROFILE_PATH" ]]; then
    printf '%s\n' "$PROFILE_PATH"
    return
  fi

  case "${SHELL:-}" in
    */zsh) printf '%s\n' "$HOME/.zshrc" ;;
    */bash) printf '%s\n' "$HOME/.bashrc" ;;
    *) printf '%s\n' "$HOME/.profile" ;;
  esac
}

node_major() {
  local version
  version="$(node --version 2>/dev/null || true)"
  version="${version#v}"
  printf '%s\n' "${version%%.*}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package)
      [[ $# -ge 2 ]] || die "--package requires a value"
      PACKAGE_SPEC="$2"
      shift 2
      ;;
    --install-root)
      [[ $# -ge 2 ]] || die "--install-root requires a value"
      INSTALL_ROOT="$2"
      shift 2
      ;;
    --node-bin-dir)
      [[ $# -ge 2 ]] || die "--node-bin-dir requires a value"
      NODE_BIN_DIR="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      PROFILE_PATH="$2"
      shift 2
      ;;
    --no-profile)
      UPDATE_PROFILE=0
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

if [[ -n "$NODE_BIN_DIR" ]]; then
  [[ -d "$NODE_BIN_DIR" ]] || die "node bin dir not found: $NODE_BIN_DIR"
  [[ -x "$NODE_BIN_DIR/node" ]] || die "node not executable in: $NODE_BIN_DIR"
  [[ -x "$NODE_BIN_DIR/npm" ]] || die "npm not executable in: $NODE_BIN_DIR"
  export PATH="$NODE_BIN_DIR:$PATH"
  hash -r
fi

need_cmd node
need_cmd npm

major="$(node_major)"
if [[ -z "$major" || "$major" -lt 18 ]]; then
  die "Node >=18 is required; found: $(node --version 2>/dev/null || echo missing). Use --node-bin-dir."
fi

NPM_PREFIX="$INSTALL_ROOT/npm-prefix"
SHIM_DIR="$INSTALL_ROOT/bin"
SHIM="$SHIM_DIR/claude"
NODE_RUNTIME_DIR="$(dirname "$(command -v node)")"

say "install root: $INSTALL_ROOT"
say "using node: $(command -v node) ($(node --version))"
say "using npm: $(command -v npm) ($(npm --version))"
say "installing Claude Code: $PACKAGE_SPEC"

mkdir -p "$NPM_PREFIX" "$SHIM_DIR"
NO_UPDATE_NOTIFIER=1 \
  NPM_CONFIG_AUDIT=false \
  NPM_CONFIG_FUND=false \
  NPM_CONFIG_PROGRESS=false \
  NPM_CONFIG_UPDATE_NOTIFIER=false \
  NPM_CONFIG_OPTIONAL=true \
  npm install --prefix "$NPM_PREFIX" -g "$PACKAGE_SPEC" --no-audit --no-fund

[[ -x "$NPM_PREFIX/bin/claude" ]] || die "expected npm binary not found: $NPM_PREFIX/bin/claude"
cat > "$SHIM" <<EOF
#!/usr/bin/env bash
set -e
export PATH="$NODE_RUNTIME_DIR:$NPM_PREFIX/bin:\$PATH"
exec "$NPM_PREFIX/bin/claude" "\$@"
EOF
chmod +x "$SHIM"

profile_export="export PATH=\"$SHIM_DIR:\$PATH\""
if [[ "$UPDATE_PROFILE" -eq 1 ]]; then
  profile="$(choose_profile)"
  marker_begin="# >>> claude code online install >>>"
  marker_end="# <<< claude code online install <<<"
  mkdir -p "$(dirname "$profile")"
  touch "$profile"
  if grep -Fq "$marker_begin" "$profile"; then
    say "profile already contains Claude Code online install block: $profile"
  else
    say "adding PATH entry to $profile"
    cat >> "$profile" <<EOF

$marker_begin
$profile_export
$marker_end
EOF
  fi
else
  say "profile update skipped; run this in your shell:"
  say "  $profile_export"
fi

say "verifying Claude Code"
PATH="$SHIM_DIR:$PATH" "$SHIM" --version

say "done"
say "open a new shell, or run:"
say "  export PATH=\"$SHIM_DIR:\$PATH\""
