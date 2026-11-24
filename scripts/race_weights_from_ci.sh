#!/usr/bin/env bash
# race_weights_from_ci.sh [run-id] rewrites .github/scripts/race-weights.txt
# from the race-timings-* artifacts of a ci run (default: the latest successful
# run on main), so race-ci.sh balances its split shards on CI's own -race
# times. Tests under 5 s are left out; race-ci.sh counts them as 1 s.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
out="$repo_root/.github/scripts/race-weights.txt"
run_id="${1:-$(gh run list --workflow ci --branch main --status success --limit 1 --json databaseId -q '.[0].databaseId')}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

gh run download "$run_id" --pattern 'race-timings-*' --dir "$tmp"
{
  printf '# Measured -race seconds per test (CI ci run %s) for tests of 5 s or\n' "$run_id"
  printf '# more; race-ci.sh balances the split shards with them. Regenerate with\n'
  printf '# scripts/race_weights_from_ci.sh after adding a slow test.\n'
  cat "$tmp"/race-timings-*/*.txt | awk '$3 >= 5 { printf "%s %s %d\n", $1, $2, $3 + 0.5 }' | sort -k1,1 -k3,3nr
} >"$out"
printf 'wrote %s from run %s (%d tests)\n' "$out" "$run_id" "$(grep -vc '^#' "$out")"
