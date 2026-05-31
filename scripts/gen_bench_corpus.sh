#!/usr/bin/env bash
#
# gen_bench_corpus.sh -- regenerate the multi-config AV1 decode benchmark corpus.
#
# The cross-decoder throughput benchmark (TestCrossDecoderCorpus in
# internal/av1/testvector/cross_decoder_corpus_bench_test.go) needs clips that
# are long enough (~30-60 frames) that steady-state decode dominates process
# startup. The bundled libaom conformance vectors are only a couple of frames
# each, so this script synthesizes a representative matrix from a single small
# source clip by scaling/length-extending with ffmpeg and encoding with aomenc.
#
# The generated .ivf clips are NOT committed to git (the output dir is in
# .gitignore). Run this script to (re)materialize them; the benchmark skips
# gracefully when they are absent.
#
# Each emitted clip is paired with a libaom-format stream MD5 (one digest over
# the concatenated visible-frame planes, no stride padding) produced by aomdec
# and cross-checked against dav1d's md5 muxer. The Go benchmark recomputes the
# same digest from goav1's in-process decode and fails any clip that does not
# match -- so the corpus doubles as a conformance probe on real content.
#
# Axes covered: resolution (256x144, 640x360, 1280x720), rate/quality
# (cq-level 20/32/55), coding tools (all-intra vs inter GOP, single vs 2 tile
# columns), bit depth (8-bit primary plus 10/12-bit profile coverage), and
# chroma sampling (4:2:0 primary plus a profile-2 4:2:2 probe).
#
# Usage:
#   scripts/gen_bench_corpus.sh [OUTDIR]
#
# OUTDIR defaults to $GOAV1_BENCH_CORPUS_DIR, then to testdata/benchcorpus
# under the repo root.

set -euo pipefail

# --- locate tools -----------------------------------------------------------
AOMENC=${AOMENC:-$(command -v aomenc || true)}
AOMDEC=${AOMDEC:-$(command -v aomdec || true)}
DAV1D=${DAV1D:-$(command -v dav1d || true)}
FFMPEG=${FFMPEG:-$(command -v ffmpeg || true)}

for tool_var in AOMENC AOMDEC FFMPEG; do
  if [ -z "${!tool_var}" ]; then
    echo "ERROR: $tool_var not found on PATH" >&2
    exit 1
  fi
done

# --- locate source + output dir ---------------------------------------------
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SRC=${GOAV1_BENCH_SOURCE:-/Users/thesyncim/GolandProjects/govpx/internal/coracle/build/test-data/encoder/park_joy_90p_8_420.y4m}
if [ ! -f "$SRC" ]; then
  echo "ERROR: source y4m not found: $SRC" >&2
  echo "       set GOAV1_BENCH_SOURCE to a 4:2:0 8-bit y4m" >&2
  exit 1
fi

OUTDIR=${1:-${GOAV1_BENCH_CORPUS_DIR:-$REPO_ROOT/testdata/benchcorpus}}
mkdir -p "$OUTDIR"

# Number of frames each clip is length-extended to (steady-state dominates
# startup at this length). The source clip is short, so ffmpeg loops it.
FRAMES=${GOAV1_BENCH_FRAMES:-48}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "source : $SRC"
echo "outdir : $OUTDIR"
echo "frames : $FRAMES"
echo

# scaled_source WxH -> path to a length-extended y4m at that resolution.
# Cached per resolution within one run.
scaled_source() {
  local w=$1 h=$2 depth=${3:-8} chroma=${4:-420}
  local fmt
  case "$chroma:$depth" in
    420:8) fmt=yuv420p ;;
    420:10) fmt=yuv420p10le ;;
    420:12) fmt=yuv420p12le ;;
    422:8) fmt=yuv422p ;;
    422:10) fmt=yuv422p10le ;;
    422:12) fmt=yuv422p12le ;;
    444:8) fmt=yuv444p ;;
    444:10) fmt=yuv444p10le ;;
    444:12) fmt=yuv444p12le ;;
    *) echo "ERROR: unsupported generated corpus format chroma=$chroma depth=$depth" >&2; exit 1 ;;
  esac
  local key="$w"x"$h"_"$chroma"_"$depth"
  local out="$WORK/src_${key}.y4m"
  if [ -f "$out" ]; then
    echo "$out"; return
  fi
  local strict=()
  if [ "$depth" != "8" ]; then
    # ffmpeg's y4m muxer treats high-bit-depth Y4M as non-standard; -strict -1
    # lets it write the header. aomenc reads it with --input-bit-depth.
    strict=(-strict -1)
  fi
  # -stream_loop large enough to exceed FRAMES after the 10-frame source.
  "$FFMPEG" -v error -y -stream_loop 200 -i "$SRC" -frames:v "$FRAMES" \
    -vf "scale=${w}:${h}" -pix_fmt "$fmt" "${strict[@]}" "$out"
  echo "$out"
}

# encode NAME WxH CQ DEPTH [CHROMA] EXTRA_AOMENC_ARGS...
# Produces $OUTDIR/NAME.ivf plus $OUTDIR/NAME.md5 (libaom stream-md5 format),
# and cross-checks aomdec vs dav1d.
encode() {
  local name=$1 w=$2 h=$3 cq=$4 depth=$5; shift 5
  local chroma=420
  case "${1:-}" in
    420|422|444) chroma=$1; shift ;;
  esac
  local extra=("$@")
  local src; src=$(scaled_source "$w" "$h" "$depth" "$chroma")
  local ivf="$OUTDIR/$name.ivf"
  local md5="$OUTDIR/$name.md5"

  local depth_args=(--bit-depth="$depth")
  if [ "$depth" != "8" ]; then
    depth_args+=(--input-bit-depth="$depth")
  fi
  local profile_args=(--profile=0)
  if [ "$chroma" = "422" ] || [ "$depth" = "12" ]; then
    profile_args=(--profile=2)
  elif [ "$chroma" = "444" ]; then
    profile_args=(--profile=1)
  fi
  if [ "$chroma" = "444" ] && [ "$depth" = "12" ]; then
    profile_args=(--profile=2)
  fi
  if [ "$depth" = "10" ] && [ "$chroma" = "420" ]; then
    profile_args=(--profile=0)
  fi

  "$AOMENC" --quiet --cpu-used=6 --end-usage=q --cq-level="$cq" \
    "${depth_args[@]}" "${profile_args[@]}" "${extra[@]}" \
    --ivf -o "$ivf" "$src"

  # libaom stream MD5: md5 over concatenated visible-frame planes (no padding).
  # aomdec --i420 forces 8-bit 4:2:0 YUV; for high bit depth or non-4:2:0 use
  # native rawvideo so the digest matches goav1's FrameMD5 plane layout.
  local ref
  if [ "$depth" = "8" ] && [ "$chroma" = "420" ]; then
    ref=$("$AOMDEC" --i420 --md5 "$ivf" 2>/dev/null | awk 'NR==1{print $1}')
  else
    ref=$("$AOMDEC" --rawvideo --md5 "$ivf" 2>/dev/null | awk 'NR==1{print $1}')
  fi
  printf '%s\n' "$ref" > "$md5"

  # Cross-check against dav1d's md5 muxer (8-bit 4:2:0 only; dav1d emits the
  # same stream digest as aomdec there). High-bit-depth and non-4:2:0 raw
  # output ordering can differ, so we trust aomdec and the Go bench is the
  # final arbiter for those.
  local x="(dav1d skipped)"
  if [ -n "$DAV1D" ] && [ "$depth" = "8" ] && [ "$chroma" = "420" ]; then
    local d
    d=$("$DAV1D" --muxer md5 -o - -i "$ivf" 2>/dev/null || true)
    if [ "$d" = "$ref" ]; then
      x="dav1d=OK"
    else
      x="dav1d=MISMATCH($d)"
    fi
  fi

  local bytes; bytes=$(wc -c < "$ivf" | tr -d ' ')
  printf '  %-30s %-9s cq=%-2s d=%-2s c=%-3s %8s bytes  md5=%s  %s\n' \
    "$name" "${w}x${h}" "$cq" "$depth" "$chroma" "$bytes" "$ref" "$x"
}

echo "encoding corpus..."

# ---- 256x144 (low res) -----------------------------------------------------
encode p144_intra_q32  256 144 32 8 --kf-min-dist=0 --kf-max-dist=0
encode p144_inter_q20  256 144 20 8
encode p144_inter_q32  256 144 32 8
encode p144_inter_q55  256 144 55 8

# ---- 512x288 (mid-low res) -------------------------------------------------
encode p288_intra_q32  512 288 32 8 --kf-min-dist=0 --kf-max-dist=0
encode p288_inter_q20  512 288 20 8
encode p288_inter_q32  512 288 32 8
encode p288_inter_q55  512 288 55 8

# ---- 640x360 (mid res) -----------------------------------------------------
encode p360_intra_q32  640 360 32 8 --kf-min-dist=0 --kf-max-dist=0
encode p360_inter_q20  640 360 20 8
encode p360_inter_q32  640 360 32 8
encode p360_inter_q55  640 360 55 8
encode p360_inter_q32_2tiles 640 360 32 8 --tile-columns=1
encode p360_inter_q32_10bit  640 360 32 10
encode p360_intra_q32_12bit  640 360 32 12 --kf-min-dist=0 --kf-max-dist=0
encode p360_inter_q32_12bit  640 360 32 12
encode p360_inter_q32_12bit_2tiles 640 360 32 12 --tile-columns=1
encode p360_inter_q55_12bit  640 360 55 12
encode p360_inter_q32_422_10bit 640 360 32 10 422

# ---- 1280x720 (high res) ---------------------------------------------------
encode p720_inter_q20  1280 720 20 8
encode p720_inter_q32  1280 720 32 8
encode p720_inter_q55  1280 720 55 8
encode p720_inter_q32_2tiles 1280 720 32 8 --tile-columns=1

echo
echo "done. $(ls "$OUTDIR"/*.ivf 2>/dev/null | wc -l | tr -d ' ') clips in $OUTDIR"
echo "run the benchmark with:"
echo "  GOAV1_BENCH_CORPUS=1 go test -tags goav1_oracle -run TestCrossDecoderCorpus -timeout 30m ./internal/av1/testvector/"
