#!/bin/bash
# gates.sh tag "race pkgs" [decoder] -> runs the gate set concurrently; logs in $S/fable-novel/gates/<tag>-*.log
S=/private/tmp/claude-501/-Users-thesyncim-GolandProjects-goav1/800d3de7-31a0-4b85-a8cc-659f6d99e59e/scratchpad
W=/Users/thesyncim/GolandProjects/goav1/.claude/worktrees/fable-novel
cd $W || exit 1
export GOCACHE=$S/fable-novel/gocache
tag=$1; racepkgs=$2; dec=$3
L=$S/fable-novel/gates/$tag
run() { n=$1; shift; ( "$@" > $L-$n.log 2>&1; echo "exit $?" >> $L-$n.log ) & }
run internal        go test -count=1 -timeout 60m ./internal/...
run internal-simd   env GOEXPERIMENT=simd go test -count=1 -timeout 60m ./internal/...
run root-enc        env GOEXPERIMENT=simd go test -count=1 -timeout 60m -run 'Encode|Encoder|RealC|Recon|RTC|Roundtrip' .
run race            env GOEXPERIMENT=simd go test -count=1 -timeout 60m -race $racepkgs
run vet-amd64       sh -c "GOOS=linux GOARCH=amd64 GOEXPERIMENT=simd go vet ./... && GOARCH=amd64 GOEXPERIMENT=simd go build ./... && go vet -tags goav1_oracle ./..."
run compiler        make compiler-reports
run alloc           make alloc
if [ -n "$dec" ]; then
  run md5-simd      env GOAV1_PUBLIC_FULLCLIP=1 GOEXPERIMENT=simd go test -count=1 -timeout 60m -run 'MD5|Workers|SteadyState' .
  run md5-default   env GOAV1_PUBLIC_FULLCLIP=1 go test -count=1 -timeout 60m -run 'MD5|Workers|SteadyState' .
  run profiles      sh -c "make dryrun-profiles && GOEXPERIMENT=simd make dryrun-profiles"
  run testvectors   make testvectors-fast
  run conformance   make dryrun-fast
fi
wait
for f in $L-*.log; do echo "$(basename $f): $(tail -1 $f) $(grep -c -- '--- FAIL' $f) fails"; done
