#!/bin/bash
# gates.sh <tag> "<race pkgs>" [decoder] -> the gate set, run concurrently; logs in $ITX_WORK/gates/<tag>-*.log
. "$(dirname "$0")/env.sh"
cd "$root" || exit 1
tag=$1; racepkgs=$2; dec=$3
L=$work/gates/$tag
run() { n=$1; shift; ( "$@" > "$L-$n.log" 2>&1; echo "exit $?" >> "$L-$n.log" ) & }
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
  run conformance-pkg sh -c "go test -count=1 ./conformance && GOEXPERIMENT=simd go test -count=1 ./conformance"
fi
wait
for f in "$L"-*.log; do echo "$(basename "$f"): $(tail -1 "$f") $(grep -c -- '--- FAIL' "$f") fails"; done
