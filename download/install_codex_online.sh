#!/usr/bin/env bash
set -euo pipefail

PACKAGE_SPEC="@openai/codex@latest"
INSTALL_ROOT="${CODEX_ONLINE_INSTALL_ROOT:-$HOME/.local/codex-online}"
NODE_BIN_DIR=""
CODEX_HOME_DIR="${CODEX_HOME:-}"
UPDATE_PROFILE=1
PROFILE_PATH=""

usage() {
  cat <<'EOF'
Usage:
  install_codex_online.sh [options]

Install Codex CLI directly from npm into a user-local prefix.

Options:
  --package SPEC      Codex npm package. Default: @openai/codex@latest
  --install-root DIR  Install root. Default: ~/.local/codex-online
  --node-bin-dir DIR  Directory containing Node >=16 and npm.
  --codex-home DIR    Optional CODEX_HOME to write into your profile.
                      Default Codex home is ~/.codex.
  --profile PATH      Shell profile to update. Default: ~/.zshrc or ~/.bashrc.
  --no-profile        Do not update a shell profile; print the export instead.
  -h, --help          Show this help.

Typical:
  ./install_codex_online.sh --node-bin-dir ~/.nvm/versions/node/v24.15.0/bin
EOF
}

say() {
  printf '[codex-online] %s\n' "$*"
}

die() {
  printf '[codex-online] ERROR: %s\n' "$*" >&2
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

profile_exports() {
  printf 'export PATH="%s:$PATH"\n' "$SHIM_DIR"
  if [[ -n "$CODEX_HOME_DIR" ]]; then
    printf 'export CODEX_HOME="%s"\n' "$CODEX_HOME_DIR"
  fi
}

update_profile() {
  local exports
  exports="$(profile_exports)"

  if [[ "$UPDATE_PROFILE" -eq 0 ]]; then
    say "profile update skipped; run this in your shell:"
    printf '%s\n' "$exports" | sed 's/^/[codex-online]   /'
    return
  fi

  local profile
  profile="$(choose_profile)"
  local marker_begin="# >>> codex online install >>>"
  local marker_end="# <<< codex online install <<<"

  mkdir -p "$(dirname "$profile")"
  touch "$profile"
  if grep -Fq "$marker_begin" "$profile"; then
    say "profile already contains Codex online install block: $profile"
  else
    say "adding PATH entry to $profile"
    cat >> "$profile" <<EOF

$marker_begin
$exports
$marker_end
EOF
  fi
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
    --codex-home)
      [[ $# -ge 2 ]] || die "--codex-home requires a value"
      CODEX_HOME_DIR="$2"
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
if [[ -z "$major" || "$major" -lt 16 ]]; then
  die "Node >=16 is required; found: $(node --version 2>/dev/null || echo missing). Use --node-bin-dir."
fi

NPM_PREFIX="$INSTALL_ROOT/npm-prefix"
SHIM_DIR="$INSTALL_ROOT/bin"
SHIM="$SHIM_DIR/codex"
NODE_RUNTIME_DIR="$(dirname "$(command -v node)")"

say "install root: $INSTALL_ROOT"
say "using node: $(command -v node) ($(node --version))"
say "using npm: $(command -v npm) ($(npm --version))"
say "installing Codex CLI: $PACKAGE_SPEC"

mkdir -p "$NPM_PREFIX" "$SHIM_DIR"
NO_UPDATE_NOTIFIER=1 \
  NPM_CONFIG_AUDIT=false \
  NPM_CONFIG_FUND=false \
  NPM_CONFIG_PROGRESS=false \
  NPM_CONFIG_UPDATE_NOTIFIER=false \
  NPM_CONFIG_OPTIONAL=true \
  npm install --prefix "$NPM_PREFIX" -g "$PACKAGE_SPEC" --no-audit --no-fund

[[ -x "$NPM_PREFIX/bin/codex" ]] || die "expected npm binary not found: $NPM_PREFIX/bin/codex"
cat > "$SHIM" <<EOF
#!/usr/bin/env bash
set -e
export PATH="$NODE_RUNTIME_DIR:$NPM_PREFIX/bin:\$PATH"
exec "$NPM_PREFIX/bin/codex" "\$@"
EOF
chmod +x "$SHIM"

update_profile

codex_home="${CODEX_HOME_DIR:-$HOME/.codex}"
mkdir -p "$codex_home/skills"

say "verifying Codex"
CODEX_HOME="$codex_home" PATH="$SHIM_DIR:$PATH" "$SHIM" --version

say "Codex home: $codex_home"
say "config file: $codex_home/config.toml"
say "skills dir: $codex_home/skills/<skill-name>/SKILL.md"
say "done"
say "open a new shell, or run:"
say "  export PATH=\"$SHIM_DIR:\$PATH\""
if [[ -n "$CODEX_HOME_DIR" ]]; then
  say "  export CODEX_HOME=\"$CODEX_HOME_DIR\""
fi
