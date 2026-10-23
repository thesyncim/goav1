#!/bin/bash
# abenc.sh <bitrate> <runs> <binA> <binB> -> min instructions/cycles of interleaved single-thread encodes
br=$1; runs=$2; A=$3; B=$4
S=/private/tmp/claude-501/-Users-thesyncim-GolandProjects-goav1/800d3de7-31a0-4b85-a8cc-659f6d99e59e/scratchpad
cd $S/clips || exit 1
minA_i=; minA_c=; minB_i=; minB_c=
meas() { GOMAXPROCS=1 /usr/bin/time -l $S/fable-novel/bin/$1 -input clip.yuv -frames 240 -width 1280 -height 720 -fps 30 -bitrate $br -threads 0 -nopsnr -o /dev/null 2>&1 >/dev/null | awk '/instructions retired/{i=$1} /cycles elapsed/{c=$1} END{print i, c}'; }
for r in $(seq $runs); do
  set -- $(meas $A); ia=$1; ca=$2
  set -- $(meas $B); ib=$1; cb=$2
  [ -z "$minA_i" ] || [ $ia -lt $minA_i ] && minA_i=$ia
  [ -z "$minA_c" ] || [ $ca -lt $minA_c ] && minA_c=$ca
  [ -z "$minB_i" ] || [ $ib -lt $minB_i ] && minB_i=$ib
  [ -z "$minB_c" ] || [ $cb -lt $minB_c ] && minB_c=$cb
done
awk -v n="encode 720p ${br}bps" -v ai=$minA_i -v ac=$minA_c -v bi=$minB_i -v bc=$minB_c 'BEGIN{printf "%-28s instr %.3fG -> %.3fG (%+.2f%%)  cycles %.3fG -> %.3fG (%+.2f%%)\n", n, ai/1e9, bi/1e9, (bi-ai)*100/ai, ac/1e9, bc/1e9, (bc-ac)*100/ac}'
