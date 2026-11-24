#!/bin/bash
# ident.sh <baseBin> <newBin> -> encoder IVF byte-identity at 0.5/1/2 Mbps and 1 Mbps -threads 4
# The binaries are cmd/encbench builds; ITX_CLIP is the raw 1280x720 I420 input.
# Exit status: 0 all identical, 1 a byte difference, 2 an encode failed (the
# failing command and its stderr are printed; a failed run is never a DIFF).
. "$(dirname "$0")/env.sh"
clip=${ITX_CLIP:?set ITX_CLIP to a raw 1280x720 I420 clip}
base=$1; new=$2; rc=0
# Per-invocation outputs: concurrent ident.sh runs sharing ITX_WORK must not
# overwrite each other's streams.
tmp=$(mktemp -d "$work/ident.XXXXXX") || exit 2
trap 'rm -rf "$tmp"' EXIT
# enc <bin> <bitrate> <threads> <out>: one encode; a non-zero exit or a missing
# output reports FAIL with the command and its stderr.
enc() {
  local cmd=("$1" -input "$clip" -frames 240 -width 1280 -height 720 -fps 30 -bitrate "$2" -threads "$3" -nopsnr -o "$4")
  "${cmd[@]}" >/dev/null 2>"$tmp/stderr"
  local st=$?
  if [ $st -ne 0 ]; then
    echo "FAIL exit=$st bitrate=$2 threads=$3: ${cmd[*]}" >&2
    sed 's/^/  /' "$tmp/stderr" >&2
    return 1
  fi
  if [ ! -s "$4" ]; then
    echo "FAIL empty output bitrate=$2 threads=$3: ${cmd[*]}" >&2
    return 1
  fi
}
for cfg in "500000 0" "1000000 0" "2000000 0" "1000000 4"; do
  br=${cfg% *}; th=${cfg#* }
  enc "$base" "$br" "$th" "$tmp/base.ivf" || exit 2
  enc "$new" "$br" "$th" "$tmp/new.ivf" || exit 2
  a=$(md5 -q "$tmp/base.ivf"); c=$(md5 -q "$tmp/new.ivf"); sz=$(stat -f %z "$tmp/base.ivf")
  if [ "$a" = "$c" ]; then echo "IDENTICAL bitrate=$br threads=$th md5=$a size=$sz"; else echo "DIFF bitrate=$br threads=$th base=$a new=$c"; rc=1; fi
  rm -f "$tmp/base.ivf" "$tmp/new.ivf"
done
exit $rc
