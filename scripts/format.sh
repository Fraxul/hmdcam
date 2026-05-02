#!/usr/bin/env bash
# Format C/C++/CUDA source files with clang-format, then post-process to
# restore the space between named-argument block comments and their values:
#   FunctionCall(/*name=*/value)  ->  FunctionCall(/*name=*/ value)
#
# clang-format has no option to control spacing after a /* */ block comment
# in mid-expression position, so a sed pass restores it.
#
# Idempotent: the sed pattern only matches =*/ followed by a non-space char,
# so re-running is a no-op.
#
# Usage:
#   scripts/format.sh <file_or_directory> [...]
#
# Files are formatted directly. Directories are searched recursively for
# .h .hpp .c .cc .cpp .cu source files.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <file_or_directory> [...]" >&2
  exit 1
fi

files=()
for arg in "$@"; do
  if [[ -d "$arg" ]]; then
    while IFS= read -r f; do
      files+=("$f")
    done < <(find "$arg" -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cc' -o -name '*.c' -o -name '*.cu' \))
  elif [[ -f "$arg" ]]; then
    files+=("$arg")
  else
    echo "skipping (not a file or directory): $arg" >&2
  fi
done

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no source files found" >&2
  exit 1
fi

clang-format -i "${files[@]}"
sed -i 's|=\*/\([^ ]\)|=*/ \1|g' "${files[@]}"

echo "formatted ${#files[@]} file(s)"
