#!/bin/bash
# ident.sh <baseBin> <newBin> -> encoder IVF byte-identity at 0.5/1/2 Mbps and 1 Mbps -threads 4
# The binaries are cmd/encbench builds; ITX_CLIP is the raw 1280x720 I420 input.
. "$(dirname "$0")/env.sh"
clip=${ITX_CLIP:?set ITX_CLIP to a raw 1280x720 I420 clip}
base=$1; new=$2; rc=0
for cfg in "500000 0" "1000000 0" "2000000 0" "1000000 4"; do
  br=${cfg% *}; th=${cfg#* }
  "$base" -input "$clip" -frames 240 -width 1280 -height 720 -fps 30 -bitrate "$br" -threads "$th" -nopsnr -o "$work/ident-base.ivf" >/dev/null 2>&1
  "$new" -input "$clip" -frames 240 -width 1280 -height 720 -fps 30 -bitrate "$br" -threads "$th" -nopsnr -o "$work/ident-new.ivf" >/dev/null 2>&1
  a=$(md5 -q "$work/ident-base.ivf"); c=$(md5 -q "$work/ident-new.ivf"); sz=$(stat -f %z "$work/ident-base.ivf")
  if [ "$a" = "$c" ]; then echo "IDENTICAL bitrate=$br threads=$th md5=$a size=$sz"; else echo "DIFF bitrate=$br threads=$th base=$a new=$c"; rc=1; fi
  rm -f "$work/ident-base.ivf" "$work/ident-new.ivf"
done
exit $rc
