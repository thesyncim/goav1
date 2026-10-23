#!/bin/bash
# realgate.sh <decoder-binary> [workers] -> MD5 of every realcorpus stream vs its dav1d .md5
S=/private/tmp/claude-501/-Users-thesyncim-GolandProjects-goav1/800d3de7-31a0-4b85-a8cc-659f6d99e59e/scratchpad
R=/Users/thesyncim/GolandProjects/goav1/testdata/realcorpus
bin=$1; workers=${2:-1}; pass=0; fail=0
for ivf in $R/streams/*.ivf $R/aq0/streams/*.ivf; do
  md5f=${ivf%.ivf}.md5
  want=$(tr -d ' \n' < "$md5f")
  got=$(GOMAXPROCS=8 $bin -workers $workers -quiet "$ivf" 2>/dev/null | md5 -q)
  if [ "$got" = "$want" ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL $(basename $ivf) got=$got want=$want"; fi
done
echo "realcorpus: pass=$pass fail=$fail (workers=$workers)"
[ $fail -eq 0 ]
