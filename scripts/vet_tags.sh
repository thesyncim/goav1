#!/usr/bin/env sh
# Vet the module once per project build tag. The files behind goav1_oracle, the
# refmvs oracle and the trace, poison and debug builds are compiled by nothing
# else, so a change that breaks one of them goes unnoticed without this.
set -eu

cd "$(dirname "$0")/.."
tags=$(git grep -h '^//go:build' -- '*.go' | grep -o 'goav1_[a-z_0-9]*' | sort -u)
[ -n "$tags" ] || { echo "vet_tags: no project build tags found" >&2; exit 1; }
for tag in $tags; do
	echo "go vet -tags $tag ./..."
	go vet -tags "$tag" ./...
done
