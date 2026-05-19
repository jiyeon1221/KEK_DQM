#!/bin/bash
# ── Full DQM rebuild: dylib + every standalone executable ──────────────────
#
# Run this whenever a header (DQM/include/*.h) changes. Changing a class's
# data members alters its memory layout, and any executable that does
# `new T(...)` on that class needs to be relinked or it will corrupt
# memory at runtime (typical symptom: segfault inside yaml-cpp/ROOT/etc.
# on the first non-trivial pointer access after construction).
#
# Steps:
#   1. buildNinstall.sh  → rebuild + install libdrcTB.dylib
#   2. envset.sh         → set DYLD paths, ROOT, yaml-cpp
#   3. compile.sh <file> → relink each standalone *.cc in DQM/
#
# Usage:
#   bash rebuild_all.sh                # rebuild everything
#   bash rebuild_all.sh monit          # rebuild only the listed targets
#   bash rebuild_all.sh --lib-only     # only step 1 (dylib, no executables)
#   bash rebuild_all.sh --bins-only    # only steps 2–3 (executables only)
#
# Exit code:
#   0 if every requested step succeeded; 1 otherwise. The script never
#   stops early on a single failure — it reports a summary at the end so
#   you can see all problems at once.

set -u  # unset variable = error; we intentionally do NOT set -e (see above)

# ── Locate DQM root regardless of where the script is invoked from ────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ───────────────────────────────────────────────────────────
DO_LIB=1
DO_BINS=1
EXPLICIT_TARGETS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --lib-only)  DO_BINS=0 ;;
    --bins-only) DO_LIB=0  ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    -*)
      echo "rebuild_all.sh: unknown flag '$1'" >&2
      exit 2
      ;;
    *)
      EXPLICIT_TARGETS+=("$1")
      ;;
  esac
  shift
done

# If the user named explicit targets, switch off the dylib rebuild unless
# they also passed --lib-only (selective re-link only).
if [ ${#EXPLICIT_TARGETS[@]} -gt 0 ] && [ "$DO_LIB" -eq 1 ] && [ "$DO_BINS" -eq 1 ]; then
  DO_LIB=0
fi

# Pretty header
hr() { printf '═%.0s' {1..70}; echo; }
banner() { hr; echo "  $1"; hr; }

OK_LIST=()
FAIL_LIST=()

# ── Step 1: rebuild the dylib ─────────────────────────────────────────────
if [ "$DO_LIB" -eq 1 ]; then
  banner "1/2  building libdrcTB.dylib (buildNinstall.sh)"
  if bash buildNinstall.sh; then
    OK_LIST+=("libdrcTB.dylib")
  else
    FAIL_LIST+=("libdrcTB.dylib")
    echo "FAIL: dylib build failed — executables will be skipped"
    DO_BINS=0
  fi
fi

# ── Step 2: source env, then re-link every standalone *.cc ─────────────────
if [ "$DO_BINS" -eq 1 ]; then
  banner "2/2  re-linking standalone executables (compile.sh)"

  # Source env vars (ROOT, yaml-cpp, DYLD paths). compile.sh needs these.
  # shellcheck disable=SC1091
  source ./envset.sh

  # Decide which sources to recompile.
  if [ ${#EXPLICIT_TARGETS[@]} -gt 0 ]; then
    TARGETS=()
    for t in "${EXPLICIT_TARGETS[@]}"; do
      # Accept either "monit" or "monit.cc"
      src="${t%.cc}.cc"
      if [ -f "$src" ]; then
        TARGETS+=("$src")
      else
        FAIL_LIST+=("$t (no such .cc file)")
      fi
    done
  else
    # Glob every standalone .cc in DQM/ (src/*.cc belongs to the library).
    shopt -s nullglob
    TARGETS=( *.cc )
    shopt -u nullglob
  fi

  if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "(nothing to compile)"
  fi

  for src in "${TARGETS[@]}"; do
    name="${src%.cc}"
    echo
    echo "──  $src  ──"
    if bash compile.sh "$src"; then
      OK_LIST+=("$name")
    else
      FAIL_LIST+=("$name")
    fi
  done
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo
banner "summary"
if [ ${#OK_LIST[@]} -gt 0 ]; then
  echo "OK   (${#OK_LIST[@]}):"
  for t in "${OK_LIST[@]}"; do echo "  ✓ $t"; done
fi
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
  echo "FAIL (${#FAIL_LIST[@]}):"
  for t in "${FAIL_LIST[@]}"; do echo "  ✗ $t"; done
  exit 1
fi
exit 0
