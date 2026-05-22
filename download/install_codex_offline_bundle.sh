#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${CODEX_OFFLINE_INSTALL_ROOT:-$HOME/.local/codex-offline}"
BUNDLE=""
SOURCE_DIR=""
UPDATE_PROFILE=1
PROFILE_PATH=""

usage() {
  cat <<'EOF'
Usage:
  install_codex_offline_bundle.sh [options]

Run this script inside the restricted dev environment after uploading the
bundle created by prepare_codex_offline_bundle.sh.

Options:
  --bundle PATH       Install from a codex-cli-offline-bundle-*.tar.gz file.
  --source-dir DIR    Install from an already extracted bundle directory.
  --install-root DIR  Install root. Default: ~/.local/codex-offline
  --profile PATH      Shell profile to update. Default: ~/.zshrc and ~/.bashrc.
  --no-profile        Do not update a shell profile; print the export instead.
  -h, --help          Show this help.

Typical:
  ./install_codex_offline_bundle.sh --bundle codex-cli-offline-bundle-linux-x86_64-*.tar.gz

After install:
  source ~/.zshrc
  # or: source ~/.bashrc
  codex --version
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

node_major_from_cmd() {
  local cmd="$1"
  local version
  version="$("$cmd" --version 2>/dev/null || true)"
  version="${version#v}"
  printf '%s\n' "${version%%.*}"
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

choose_profiles() {
  if [[ -n "$PROFILE_PATH" ]]; then
    printf '%s\n' "$PROFILE_PATH"
    return
  fi

  printf '%s\n' "$HOME/.zshrc"
  printf '%s\n' "$HOME/.bashrc"
}

find_first() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -path "$pattern" -print -quit 2>/dev/null || true
}

update_profile_block() {
  local profile="$1"
  local marker_begin="$2"
  local marker_end="$3"
  local profile_export="$4"
  local tmp_profile

  mkdir -p "$(dirname "$profile")"
  touch "$profile"
  tmp_profile="$(mktemp)"

  if ! awk -v begin="$marker_begin" -v end="$marker_end" '
    $0 == begin { in_block = 1; next }
    $0 == end && in_block { in_block = 0; next }
    !in_block { print }
    END { if (in_block) exit 2 }
  ' "$profile" > "$tmp_profile"; then
    rm -f "$tmp_profile"
    die "failed to update profile block in $profile"
  fi

  {
    awk '
      NF {
        for (i = 1; i <= blanks; i++) print ""
        blanks = 0
        print
        seen = 1
        next
      }
      seen { blanks++ }
    ' "$tmp_profile"
    printf '\n%s\n%s\n%s\n' "$marker_begin" "$profile_export" "$marker_end"
  } > "$profile"
  rm -f "$tmp_profile"
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
need_cmd mktemp
need_cmd awk

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
  if [[ -f "$script_dir/payload/codex-prefix.tar.gz" ]]; then
    SOURCE_DIR="$script_dir"
  elif [[ -f "$PWD/payload/codex-prefix.tar.gz" ]]; then
    SOURCE_DIR="$PWD"
  else
    die "no bundle source found; pass --bundle or --source-dir"
  fi
fi

[[ -d "$SOURCE_DIR" ]] || die "source directory not found: $SOURCE_DIR"
codex_payload="$SOURCE_DIR/payload/codex-prefix.tar.gz"
[[ -f "$codex_payload" ]] || die "missing payload/codex-prefix.tar.gz in $SOURCE_DIR"

release_id="$(date -u +%Y%m%dT%H%M%SZ)"
release_dir="$INSTALL_ROOT/releases/$release_id"
codex_prefix="$release_dir/codex-prefix"
node_dir="$release_dir/node"
shim_dir="$INSTALL_ROOT/bin"
shim="$shim_dir/codex"

say "install root: $INSTALL_ROOT"
mkdir -p "$codex_prefix" "$node_dir" "$shim_dir"

say "installing Codex payload"
extract_tar "$codex_payload" "$codex_prefix" 0

node_payload=""
for candidate in \
  "$SOURCE_DIR/payload/node-runtime.tar.xz" \
  "$SOURCE_DIR/payload/node-runtime.tar.gz" \
  "$SOURCE_DIR/payload/node-runtime.tgz"; do
  if [[ -f "$candidate" ]]; then
    node_payload="$candidate"
    break
  fi
done

node_cmd=""
if [[ -n "$node_payload" ]]; then
  say "installing bundled Node runtime"
  extract_tar "$node_payload" "$node_dir" 1
  if [[ -x "$node_dir/bin/node" ]]; then
    node_cmd="$node_dir/bin/node"
  else
    warn "bundled Node archive did not produce bin/node"
  fi
fi

if [[ -z "$node_cmd" && "$(command -v node || true)" != "" ]]; then
  system_node="$(command -v node)"
  major="$(node_major_from_cmd "$system_node")"
  if [[ -n "$major" && "$major" -ge 16 ]]; then
    node_cmd="$system_node"
  else
    warn "system Node is too old for the npm shim: $("$system_node" --version 2>/dev/null || true)"
  fi
fi

js_entry="$codex_prefix/lib/node_modules/@openai/codex/bin/codex.js"
native_bin="$(find_first "$codex_prefix/lib/node_modules/@openai/codex" '*/vendor/*/codex/codex')"
vendor_path=""
if [[ -n "$native_bin" ]]; then
  native_codex_dir="$(dirname "$native_bin")"
  arch_root="$(dirname "$native_codex_dir")"
  if [[ -d "$arch_root/path" ]]; then
    vendor_path="$arch_root/path"
  fi
fi

if [[ -n "$node_cmd" && -f "$js_entry" ]]; then
  say "creating Codex shim using Node: $node_cmd"
  node_bin_dir="$(dirname "$node_cmd")"
  cat > "$shim" <<EOF
#!/usr/bin/env bash
set -e
export PATH="$node_bin_dir:$vendor_path:\$PATH"
exec "$node_cmd" "$js_entry" "\$@"
EOF
elif [[ -n "$native_bin" ]]; then
  say "creating Codex shim using native binary fallback"
  cat > "$shim" <<EOF
#!/usr/bin/env bash
set -e
export PATH="$vendor_path:\$PATH"
exec "$native_bin" "\$@"
EOF
else
  die "could not find a runnable Codex entrypoint"
fi
chmod +x "$shim"

ln -sfn "$release_dir" "$INSTALL_ROOT/current"

profile_export="export PATH=\"$shim_dir:\$PATH\""
if [[ "$UPDATE_PROFILE" -eq 1 ]]; then
  marker_begin="# >>> codex offline install >>>"
  marker_end="# <<< codex offline install <<<"
  while IFS= read -r profile; do
    [[ -n "$profile" ]] || continue
    if [[ -f "$profile" ]] && grep -Fq "$marker_begin" "$profile"; then
      say "updating existing PATH entry in $profile"
    else
      say "adding PATH entry to $profile"
    fi
    update_profile_block "$profile" "$marker_begin" "$marker_end" "$profile_export"
  done < <(choose_profiles)
else
  say "profile update skipped; run this in your shell:"
  say "  $profile_export"
fi

say "verifying Codex"
PATH="$shim_dir:$PATH" "$shim" --version

say "done"
say "open a new shell, or run:"
say "  export PATH=\"$shim_dir:\$PATH\""
say "then:"
say "  codex --version"
