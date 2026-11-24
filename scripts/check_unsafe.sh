#!/usr/bin/env sh
set -eu

# unsafe lives in one audited package, internal/unsafeutil. This check fails
# when:
#   - a //go:nocheckptr or //go:linkname directive appears anywhere;
#   - a non-test file imports unsafe and is not listed in
#     scripts/unsafe_allowlist.txt, or a listed file no longer imports it;
#   - the allowlist names a file outside internal/unsafeutil without a
#     "pending:" justification (the motion/loopfilter kernels still being
#     moved), or any entry lacks a justification;
#   - a _test.go file uses unsafe for anything but Sizeof, Alignof or
#     Offsetof (struct-size pins);
#   - any file outside internal/unsafeutil reaches raw memory through
#     reflect (Value.UnsafePointer, Value.UnsafeAddr, reflect.NewAt).

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

# Allowlist lines are "path  # justification"; blank lines and lines
# starting with # are comments.
entries=$(grep -vE '^[[:space:]]*(#|$)' "$allowlist" || true)
listed=$(printf '%s\n' "$entries" | sed -E 's/[[:space:]]*#.*$//' | sed '/^$/d')
if ! printf '%s\n' "$listed" | sort -c 2>/dev/null; then
	echo "$allowlist entries are not sorted" >&2
	status=1
fi
unjustified=$(printf '%s\n' "$entries" | grep -vE '[[:space:]]#[[:space:]]*[^[:space:]]' || true)
if [ -n "$unjustified" ]; then
	echo "scripts/unsafe_allowlist.txt entries without a justification:" >&2
	echo "$unjustified" >&2
	status=1
fi
outside=$(printf '%s\n' "$entries" | grep -vE '^internal/unsafeutil/' | grep -vE '#[[:space:]]*pending:' || true)
if [ -n "$outside" ]; then
	echo "only internal/unsafeutil may import unsafe; move these sites behind it:" >&2
	echo "$outside" >&2
	status=1
fi

actual=$(git grep -lE '^[[:space:]]*(import[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*[[:space:]]+)?"unsafe"' -- '*.go' ':!*_test.go' | sort)
listed_file=$(mktemp "${TMPDIR:-/tmp}/goav1-unsafe-listed.XXXXXX")
trap 'rm -f "$listed_file"' EXIT INT HUP TERM
printf '%s\n' "$listed" >"$listed_file"
unlisted=$(printf '%s\n' "$actual" | comm -23 - "$listed_file")
stale=$(printf '%s\n' "$actual" | comm -13 - "$listed_file")
if [ -n "$unlisted" ]; then
	echo "non-test files import unsafe outside internal/unsafeutil (or are not in scripts/unsafe_allowlist.txt):" >&2
	echo "$unlisted" >&2
	status=1
fi
if [ -n "$stale" ]; then
	echo "scripts/unsafe_allowlist.txt lists files that no longer import unsafe:" >&2
	echo "$stale" >&2
	status=1
fi

# grep -o reports each use on its own (git grep -o stops at the first match
# of a line), so a line mixing an allowed and a forbidden use is caught.
tests=$(git grep -lE 'unsafe\.' -- '*_test.go' | xargs grep -HnoE '\bunsafe\.[A-Za-z_]+' | grep -vE ':unsafe\.(Sizeof|Alignof|Offsetof)$' || true)
if [ -n "$tests" ]; then
	echo "_test.go files may use only unsafe.Sizeof, unsafe.Alignof and unsafe.Offsetof:" >&2
	echo "$tests" >&2
	status=1
fi

reflectraw=$(git grep -nE '\.UnsafePointer\(|\.UnsafeAddr\(|reflect\.NewAt\(' -- '*.go' ':!internal/unsafeutil' || true)
if [ -n "$reflectraw" ]; then
	echo "raw memory through reflect is unsafe; use internal/unsafeutil:" >&2
	echo "$reflectraw" >&2
	status=1
fi
exit $status
