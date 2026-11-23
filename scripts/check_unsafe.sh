#!/usr/bin/env sh
set -eu

LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
allowlist="$repo_root/scripts/unsafe_allowlist.txt"
cd "$repo_root"
status=0

directives=$(git grep -nE '^[[:space:]]*//go:(nocheckptr|linkname)' -- '*.go' || true)
if [ -n "$directives" ]; then
	echo "forbidden //go:nocheckptr or //go:linkname directives:" >&2
	echo "$directives" >&2
	status=1
fi

if ! sort -c "$allowlist" 2>/dev/null; then
	echo "$allowlist is not sorted" >&2
	status=1
fi

actual=$(git grep -lE '^[[:space:]]*(import[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*[[:space:]]+)?"unsafe"' -- '*.go' ':!*_test.go' | sort)
unlisted=$(printf '%s\n' "$actual" | comm -23 - "$allowlist")
stale=$(printf '%s\n' "$actual" | comm -13 - "$allowlist")
if [ -n "$unlisted" ]; then
	echo "non-test files import unsafe but are not in scripts/unsafe_allowlist.txt:" >&2
	echo "$unlisted" >&2
	status=1
fi
if [ -n "$stale" ]; then
	echo "scripts/unsafe_allowlist.txt lists files that no longer import unsafe:" >&2
	echo "$stale" >&2
	status=1
fi
exit $status
