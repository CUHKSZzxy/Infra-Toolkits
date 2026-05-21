#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${CLAUDE_CODE_OFFLINE_INSTALL_ROOT:-$HOME/.local/claude-code-offline}"
BUNDLE=""
SOURCE_DIR=""
UPDATE_PROFILE=1
PROFILE_PATH=""

usage() {
  cat <<'EOF'
Usage:
  install_claude_code_offline_bundle.sh [options]

Run this script inside the restricted dev environment after uploading the
bundle created by prepare_claude_code_offline_bundle.sh.

Options:
  --bundle PATH       Install from a claude-code-offline-bundle-*.tar.gz file.
  --source-dir DIR    Install from an already extracted bundle directory.
  --install-root DIR  Install root. Default: ~/.local/claude-code-offline
  --profile PATH      Shell profile to update. Default: ~/.zshrc or ~/.bashrc.
  --no-profile        Do not update a shell profile; print the export instead.
  -h, --help          Show this help.

Typical:
  ./install_claude_code_offline_bundle.sh --bundle claude-code-offline-bundle-linux-x86_64-*.tar.gz

After install:
  source ~/.zshrc
  claude --version
EOF
}

say() {
  printf '[install] %s\n' "$*"
}

warn() {
  printf '[install] warning: %s\n' "$*" >&2
}

die() {
  printf '[install] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

extract_tar() {
  local archive="$1"
  local dest="$2"
  local strip="${3:-0}"
  local strip_arg=()

  if [[ "$strip" -gt 0 ]]; then
    strip_arg=(--strip-components "$strip")
  fi

  case "$archive" in
    *.tar.xz) tar -C "$dest" -xJf "$archive" "${strip_arg[@]}" ;;
    *.tar.gz|*.tgz) tar -C "$dest" -xzf "$archive" "${strip_arg[@]}" ;;
    *) die "unsupported archive type: $archive" ;;
  esac
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

find_claude_binary() {
  local candidate
  for candidate in \
    "$claude_prefix/bin/claude" \
    "$claude_prefix/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  find "$claude_prefix/lib/node_modules/@anthropic-ai" \
    -maxdepth 3 -type f -name claude -perm -111 -print -quit 2>/dev/null || true
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      [[ $# -ge 2 ]] || die "--bundle requires a value"
      BUNDLE="$2"
      shift 2
      ;;
    --source-dir)
      [[ $# -ge 2 ]] || die "--source-dir requires a value"
      SOURCE_DIR="$2"
      shift 2
      ;;
    --install-root)
      [[ $# -ge 2 ]] || die "--install-root requires a value"
      INSTALL_ROOT="$2"
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

need_cmd tar
need_cmd date
need_cmd find
need_cmd mkdir

tmp_dir=""
cleanup() {
  if [[ -n "$tmp_dir" ]]; then
    rm -rf "$tmp_dir"
  fi
}
trap cleanup EXIT

if [[ -n "$BUNDLE" ]]; then
  [[ -f "$BUNDLE" ]] || die "bundle not found: $BUNDLE"
  tmp_dir="$(mktemp -d)"
  say "extracting bundle"
  tar -C "$tmp_dir" -xzf "$BUNDLE"
  SOURCE_DIR="$tmp_dir"
fi

if [[ -z "$SOURCE_DIR" ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  if [[ -f "$script_dir/payload/claude-code-prefix.tar.gz" ]]; then
    SOURCE_DIR="$script_dir"
  elif [[ -f "$PWD/payload/claude-code-prefix.tar.gz" ]]; then
    SOURCE_DIR="$PWD"
  else
    die "no bundle source found; pass --bundle or --source-dir"
  fi
fi

[[ -d "$SOURCE_DIR" ]] || die "source directory not found: $SOURCE_DIR"
claude_payload="$SOURCE_DIR/payload/claude-code-prefix.tar.gz"
[[ -f "$claude_payload" ]] || die "missing payload/claude-code-prefix.tar.gz in $SOURCE_DIR"

release_id="$(date -u +%Y%m%dT%H%M%SZ)"
release_dir="$INSTALL_ROOT/releases/$release_id"
claude_prefix="$release_dir/claude-code-prefix"
shim_dir="$INSTALL_ROOT/bin"
shim="$shim_dir/claude"

say "install root: $INSTALL_ROOT"
mkdir -p "$claude_prefix" "$shim_dir"

say "installing Claude Code payload"
extract_tar "$claude_payload" "$claude_prefix" 0

claude_bin="$(find_claude_binary)"
[[ -n "$claude_bin" ]] || die "could not find a runnable Claude Code binary in the payload"
chmod +x "$claude_bin" 2>/dev/null || true

say "creating Claude Code shim"
cat > "$shim" <<EOF
#!/usr/bin/env bash
set -e
exec "$claude_bin" "\$@"
EOF
chmod +x "$shim"

ln -sfn "$release_dir" "$INSTALL_ROOT/current"

profile_export="export PATH=\"$shim_dir:\$PATH\""
if [[ "$UPDATE_PROFILE" -eq 1 ]]; then
  profile="$(choose_profile)"
  marker_begin="# >>> claude code offline install >>>"
  marker_end="# <<< claude code offline install <<<"
  mkdir -p "$(dirname "$profile")"
  touch "$profile"
  if grep -Fq "$marker_begin" "$profile"; then
    say "profile already contains Claude Code offline install block: $profile"
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
PATH="$shim_dir:$PATH" "$shim" --version

say "done"
say "open a new shell, or run:"
say "  export PATH=\"$shim_dir:\$PATH\""
say "then:"
say "  claude --version"
