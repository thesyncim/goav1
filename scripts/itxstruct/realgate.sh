#!/bin/bash
# realgate.sh <decoder-binary> [workers] -> MD5 of every realcorpus stream (including aq0/) vs its dav1d .md5
# The binary is cmd/aom-go-dec: go build -o $ITX_WORK/bin/dec ./cmd/aom-go-dec
. "$(dirname "$0")/env.sh"
bin=$1; workers=${2:-1}; pass=0; fail=0
for ivf in "$corpus"/streams/*.ivf "$corpus"/aq0/streams/*.ivf; do
  md5f=${ivf%.ivf}.md5
  want=$(tr -d ' \n' < "$md5f")
  got=$(GOMAXPROCS=8 "$bin" -workers "$workers" -quiet "$ivf" 2>/dev/null | md5 -q)
  if [ "$got" = "$want" ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL $(basename "$ivf") got=$got want=$want"; fi
done
echo "realcorpus: pass=$pass fail=$fail (workers=$workers)"
[ $fail -eq 0 ]
