#!/bin/bash
# ident.sh base new -> IVF byte-identity at 0.5/1/2 Mbps and 1 Mbps -threads 4
S=/private/tmp/claude-501/-Users-thesyncim-GolandProjects-goav1/800d3de7-31a0-4b85-a8cc-659f6d99e59e/scratchpad
cd $S/clips || exit 1
base=$1; new=$2; rc=0
for cfg in "500000 0" "1000000 0" "2000000 0" "1000000 4"; do
  br=${cfg% *}; th=${cfg#* }
  for b in $base $new; do
    $S/fable-novel/bin/$b -input clip.yuv -frames 240 -width 1280 -height 720 -fps 30 -bitrate $br -threads $th -nopsnr -o $S/fable-novel/$b-$br-t$th.ivf >/dev/null 2>&1
  done
  a=$(md5 -q $S/fable-novel/$base-$br-t$th.ivf); c=$(md5 -q $S/fable-novel/$new-$br-t$th.ivf); sz=$(stat -f %z $S/fable-novel/$base-$br-t$th.ivf)
  if [ "$a" = "$c" ]; then echo "IDENTICAL bitrate=$br threads=$th md5=$a size=$sz"; else echo "DIFF bitrate=$br threads=$th base=$a new=$c"; rc=1; fi
  rm -f $S/fable-novel/$base-$br-t$th.ivf $S/fable-novel/$new-$br-t$th.ivf
done
exit $rc
