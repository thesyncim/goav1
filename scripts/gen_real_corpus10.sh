#!/usr/bin/env bash
#
# gen_real_corpus10.sh -- build the 10-bit (high bit depth) real-content
# decode corpus.
#
# testdata/realcorpus (scripts/gen_real_corpus.sh) is 8-bit only. This script
# derives yuv420p10le versions of a few of its sources and encodes them with
# the same encoder settings, at --input-depth 10 / --bit-depth=10 (profile 0,
# 4:2:0):
#
#   aomrt10_*   libaom realtime with libwebrtc's settings (gen_real_corpus.sh)
#   aomgood10_* libaom good mode (all intra tools, altrefs, LR), frame-capped
#   svtrtc10_*  SVT-AV1 rtc low-delay CBR
#   svtra10_*   SVT-AV1 preset 8 random access (one with film grain)
#
# Nothing generated here is committed: OUTDIR is gitignored. Layout:
#
#   OUTDIR/streams/<name>.ivf + <name>.md5   decode streams + MD5 sidecars
#                                            (16-bit little-endian samples)
#   OUTDIR/streams.tsv                       stream manifest
#
# The MD5 is aomdec's; the script cross-checks it against dav1d and records
# the result in streams.tsv (dav1d is the arbiter: on disagreement the .md5
# holds the dav1d MD5).
#
# Usage:
#   scripts/gen_real_corpus10.sh [OUTDIR]
#
# OUTDIR defaults to $GOAV1_REAL_CORPUS10_DIR, then testdata/realcorpus10.
# The 8-bit sources come from $GOAV1_REAL_CORPUS_DIR/src (default
# testdata/realcorpus/src; run `scripts/gen_real_corpus.sh sources` first).
#
# Tools: ffmpeg, aomenc, aomdec, dav1d, SvtAv1EncApp (v4.1.0; SVTENC).

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTDIR=${1:-${GOAV1_REAL_CORPUS10_DIR:-$REPO_ROOT/testdata/realcorpus10}}
SRCDIR=${GOAV1_REAL_CORPUS_DIR:-$REPO_ROOT/testdata/realcorpus}/src
mkdir -p "$OUTDIR/streams"
OUTDIR=$(cd "$OUTDIR" && pwd)

FFMPEG=${FFMPEG:-$(command -v ffmpeg || true)}
AOMENC=${AOMENC:-$(command -v aomenc || true)}
AOMDEC=${AOMDEC:-$(command -v aomdec || true)}
DAV1D=${DAV1D:-$(command -v dav1d || true)}
SVTENC=${SVTENC:-$HOME/bin/svt410/SvtAv1EncApp}
for tool_var in FFMPEG AOMENC AOMDEC DAV1D; do
  if [ -z "${!tool_var}" ]; then
    echo "ERROR: $tool_var not found on PATH" >&2
    exit 1
  fi
done
if [ ! -x "$SVTENC" ]; then
  echo "ERROR: SvtAv1EncApp not found at $SVTENC (set SVTENC)" >&2
  exit 1
fi

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

WORK=$(mktemp -d "$OUTDIR/work.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# scaled10 SRC_ID W H FRAMES FMT(y4m|yuv) -> yuv420p10le source at WxH.
scaled10() {
  local id=$1 w=$2 h=$3 frames=$4 fmt=$5
  local src="$SRCDIR/$id.y4m"
  local out="$WORK/${id}_${w}x${h}_${frames}.$fmt"
  if [ ! -f "$src" ]; then
    echo "ERROR: missing source $src (run scripts/gen_real_corpus.sh sources)" >&2
    exit 1
  fi
  if [ ! -f "$out" ]; then
    local f=()
    [ "$fmt" = "yuv" ] && f=(-f rawvideo)
    "$FFMPEG" -v error -y -i "$src" -frames:v "$frames" -strict -1 \
      -vf "scale=${w}:${h}:flags=bicubic,format=yuv420p10le" -pix_fmt yuv420p10le "${f[@]}" "$out"
  fi
  printf '%s' "$out"
}

# libwebrtc LibaomAv1Encoder::InitEncode settings (gen_real_corpus.sh).
aom_webrtc_args() { # w h speed threads
  local w=$1 h=$2 speed=$3 threads=$4
  local sb=dynamic
  if [ "$threads" -ge 4 ] && [ $((w * h)) -ge $((960 * 540)) ] && [ $((w * h)) -lt $((1920 * 1080)) ]; then
    sb=64
  fi
  printf '%s\n' \
    --rt --cpu-used="$speed" --threads="$threads" --end-usage=cbr --lag-in-frames=0 \
    --min-q=10 --max-q=56 --undershoot-pct=50 --overshoot-pct=50 \
    --buf-initial-sz=600 --buf-optimal-sz=600 --buf-sz=1000 --disable-kf \
    --error-resilient=0 --enable-cdef=1 --enable-tpl-model=0 --deltaq-mode=0 \
    --enable-order-hint=0 --aq-mode=3 --max-intra-rate=300 \
    --coeff-cost-upd-freq=3 --mode-cost-upd-freq=3 --mv-cost-upd-freq=3 \
    --auto-tiles=1 --row-mt=1 --enable-obmc=0 --noise-sensitivity=0 \
    --enable-warped-motion=0 --enable-global-motion=0 --enable-ref-frame-mvs=0 \
    --sb-size="$sb" --enable-cfl-intra=0 --enable-smooth-intra=0 \
    --enable-angle-delta=0 --enable-filter-intra=0 --use-intra-default-tx-only=1 \
    --disable-trellis-quant=1 --enable-dist-wtd-comp=0 --enable-diff-wtd-comp=0 \
    --enable-dual-filter=0 --enable-interintra-comp=0 --enable-interintra-wedge=0 \
    --enable-intra-edge-filter=0 --enable-intrabc=0 --enable-masked-comp=0 \
    --enable-paeth-intra=0 --enable-qm=0 --enable-rect-partitions=0 \
    --enable-restoration=0 --enable-smooth-interintra=0 --enable-tx64=0 \
    --max-reference-frames=3 --enable-palette=0
}

MANIFEST="$OUTDIR/streams.tsv"
{
  printf '# goav1_real_corpus10_streams_v1 (yuv420p10le, profile 0)\n'
  printf '# aomenc=%s\n' "$("$AOMENC" --help 2>&1 | grep -m1 -o 'v[0-9][^ ]*' || true)"
  printf '# svt=%s\n' "$("$SVTENC" --version 2>&1 | awk 'NF{print; exit}')"
  printf '# dav1d=%s\n' "$("$DAV1D" --version 2>&1 | awk 'NF{print; exit}')"
  printf 'name\tencoder\tsource_id\twidth\theight\tframes\tivf_bytes\tivf_sha256\tmd5\tdav1d_check\targs\n'
} > "$MANIFEST"

record() { # name encoder src w h frames args...
  local name=$1 encoder=$2 id=$3 w=$4 h=$5 frames=$6; shift 6
  local ivf="$OUTDIR/streams/$name.ivf"
  local ref d check md5
  ref=$("$AOMDEC" --rawvideo --md5 "$ivf" 2>/dev/null | awk 'NR==1{print $1}')
  d=$("$DAV1D" -q --filmgrain 1 --muxer md5 -o - -i "$ivf" 2>/dev/null || true)
  md5=$d
  if [ -n "$ref" ] && [ "$d" = "$ref" ]; then
    check="dav1d=OK"
  else
    check="dav1d=MISMATCH(aomdec=$ref)"
    echo "WARNING: aomdec/dav1d disagree on $name: aomdec=$ref dav1d=$d" >&2
  fi
  printf '%s\n' "$md5" > "$OUTDIR/streams/$name.md5"
  local bytes; bytes=$(wc -c < "$ivf" | tr -d ' ')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$encoder" "$id" "$w" "$h" "$frames" "$bytes" "$(sha256_file "$ivf")" \
    "$md5" "$check" "$*" >> "$MANIFEST"
  printf '  %-40s %-9s %9s bytes  md5=%s  %s\n' "$name" "${w}x${h}" "$bytes" "$md5" "$check"
}

stream_aomrt() { # name src w h kbps speed threads frames
  local name=$1 id=$2 w=$3 h=$4 kbps=$5 speed=$6 threads=$7 frames=$8
  local in; in=$(scaled10 "$id" "$w" "$h" "$frames" y4m)
  local args=()
  while IFS= read -r a; do args+=("$a"); done < <(aom_webrtc_args "$w" "$h" "$speed" "$threads")
  args+=(--bit-depth=10 --input-bit-depth=10 --target-bitrate="$kbps" --limit="$frames")
  "$AOMENC" --quiet "${args[@]}" --ivf -o "$OUTDIR/streams/$name.ivf" "$in" 2>/dev/null
  record "$name" "aomenc-rt" "$id" "$w" "$h" "$frames" "${args[*]}"
}

stream_aomgood() { # name src w h kbps speed frames extra...
  local name=$1 id=$2 w=$3 h=$4 kbps=$5 speed=$6 frames=$7; shift 7
  local in; in=$(scaled10 "$id" "$w" "$h" "$frames" y4m)
  local args=(--good --cpu-used="$speed" --end-usage=vbr --target-bitrate="$kbps" --limit="$frames"
    --bit-depth=10 --input-bit-depth=10 "$@")
  "$AOMENC" --quiet "${args[@]}" --ivf -o "$OUTDIR/streams/$name.ivf" "$in" 2>/dev/null
  record "$name" "aomenc-good" "$id" "$w" "$h" "$frames" "${args[*]}"
}

stream_svtrtc() { # name src w h kbps preset frames extra...
  local name=$1 id=$2 w=$3 h=$4 kbps=$5 preset=$6 frames=$7; shift 7
  local in; in=$(scaled10 "$id" "$w" "$h" "$frames" yuv)
  local args=(
    -w "$w" -h "$h" --fps-num 30 --fps-denom 1 --frames "$frames" --input-depth 10
    --color-format 1 --profile 0 --level 0 --preset "$preset" --rc 2 --tbr "$kbps"
    --buf-sz 1000 --buf-initial-sz 500 --buf-optimal-sz 600 --lookahead 0
    --pred-struct 1 --rtc 1 --scd 0 --enable-tf 0 --irefresh-type 2 --keyint -1
    --progress 0 "$@"
  )
  "$SVTENC" -i "$in" -b "$OUTDIR/streams/$name.ivf" "${args[@]}" >/dev/null 2>&1
  record "$name" "svt-rtc" "$id" "$w" "$h" "$frames" "${args[*]}"
}

stream_svtra() { # name src w h crf frames extra...
  local name=$1 id=$2 w=$3 h=$4 crf=$5 frames=$6; shift 6
  local in; in=$(scaled10 "$id" "$w" "$h" "$frames" yuv)
  local args=(
    -w "$w" -h "$h" --fps-num 30 --fps-denom 1 --frames "$frames" --input-depth 10
    --color-format 1 --profile 0 --preset 8 --rc 0 --crf "$crf" --pred-struct 2
    --keyint 150 --progress 0 "$@"
  )
  "$SVTENC" -i "$in" -b "$OUTDIR/streams/$name.ivf" "${args[@]}" >/dev/null 2>&1
  record "$name" "svt-ra" "$id" "$w" "$h" "$frames" "${args[*]}"
}

echo "libaom realtime (libwebrtc settings), 10-bit:"
stream_aomrt aomrt10_fourpeople_720p_1000k_s9 fourpeople 1280 720 1000 9 4 300

echo "libaom good mode, 10-bit:"
stream_aomgood aomgood10_mtbike_1080p_2500k_c6 mtbike 1920 1080 2500 6 40

echo "SVT-AV1 rtc, 10-bit:"
stream_svtrtc svtrtc10_crowdrun_1080p_3000k_p11 crowdrun 1920 1080 3000 11 300

echo "SVT-AV1 preset 8 random access, 10-bit:"
stream_svtra svtra10_parkjoy_720p_crf35_p8 parkjoy 1280 720 35 300
stream_svtra svtra10_oldtown_720p_crf35_p8_fgs8 oldtown 1280 720 35 300 --film-grain 8

echo "streams: $MANIFEST"
