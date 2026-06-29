#!/usr/bin/env bash
#
# gen_bench_corpus.sh -- regenerate the multi-config AV1 decode benchmark corpus.
#
# The cross-decoder throughput benchmark (TestCrossDecoderCorpus in
# internal/av1/testvector/cross_decoder_corpus_bench_test.go) needs clips that
# are long enough (~30-60 frames) that steady-state decode dominates process
# startup. The bundled libaom conformance vectors are only a couple of frames
# each, so this script synthesizes a representative matrix from one explicitly
# supplied source clip by scaling/length-extending with ffmpeg and encoding
# with aomenc.
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
# Axes covered: resolution (256x144, 512x288, 640x360, 1280x720), rate/quality
# (cq-level 20/32/55), coding tools (all-intra vs inter GOP, single vs 2 tile
# columns), bit depth (8-bit primary plus 10/12-bit profile coverage), and
# chroma sampling (4:2:0 primary plus profile-1 4:4:4 and profile-2 4:2:2
# probes).
#
# Usage:
#   scripts/gen_bench_corpus.sh [OUTDIR]
#
# OUTDIR defaults to $GOAV1_BENCH_CORPUS_DIR, then to testdata/benchcorpus
# under the repo root.
#
# Required input:
#   GOAV1_BENCH_SOURCE=/path/to/source_8bit_420.y4m
#   GOAV1_BENCH_SOURCE_SHA256=<sha256 of that source>

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

SRC=${GOAV1_BENCH_SOURCE:-}
SRC_EXPECTED_SHA=${GOAV1_BENCH_SOURCE_SHA256:-}
if [ -z "$SRC" ]; then
  echo "ERROR: set GOAV1_BENCH_SOURCE to a 4:2:0 8-bit y4m source clip" >&2
  exit 1
fi
if [ ! -f "$SRC" ]; then
  echo "ERROR: source y4m not found: $SRC" >&2
  echo "       set GOAV1_BENCH_SOURCE to a 4:2:0 8-bit y4m source clip" >&2
  exit 1
fi
if [ -z "$SRC_EXPECTED_SHA" ]; then
  echo "ERROR: set GOAV1_BENCH_SOURCE_SHA256 to pin the source clip content" >&2
  exit 1
fi

OUTDIR=${1:-${GOAV1_BENCH_CORPUS_DIR:-$REPO_ROOT/testdata/benchcorpus}}
mkdir -p "$OUTDIR"

# Number of frames each clip is length-extended to (steady-state dominates
# startup at this length). The source clip is short, so ffmpeg loops it.
FRAMES=${GOAV1_BENCH_FRAMES:-48}
FPS=${GOAV1_BENCH_FPS:-30}
EXPECTED_CLIPS=25
AOM_THREADS=${GOAV1_BENCH_AOM_THREADS:-1}
AOM_ROW_MT=${GOAV1_BENCH_AOM_ROW_MT:-1}
MANIFEST="$OUTDIR/manifest.tsv"

case "$AOM_THREADS" in
  ''|*[!0-9]*|0)
    echo "ERROR: GOAV1_BENCH_AOM_THREADS must be a positive integer" >&2
    exit 1
    ;;
esac
case "$FRAMES" in
  ''|*[!0-9]*|0)
    echo "ERROR: GOAV1_BENCH_FRAMES must be a positive integer" >&2
    exit 1
    ;;
esac
case "$FPS" in
  ''|*[!0-9]*|0)
    echo "ERROR: GOAV1_BENCH_FPS must be a positive integer" >&2
    exit 1
    ;;
esac
case "$AOM_ROW_MT" in
  0|1) ;;
  *)
    echo "ERROR: GOAV1_BENCH_AOM_ROW_MT must be 0 or 1" >&2
    exit 1
    ;;
esac

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "source : $SRC"
echo "outdir : $OUTDIR"
echo "frames : $FRAMES"
echo "fps    : $FPS"
echo "aomenc : threads=$AOM_THREADS row-mt=$AOM_ROW_MT"
echo

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

SRC_ACTUAL_SHA=$(sha256_file "$SRC")
if [ "$SRC_ACTUAL_SHA" != "$SRC_EXPECTED_SHA" ]; then
  echo "ERROR: source sha256 mismatch for $SRC" >&2
  echo "       got  $SRC_ACTUAL_SHA" >&2
  echo "       want $SRC_EXPECTED_SHA" >&2
  exit 1
fi

tool_sha256() {
  local tool=$1
  if [ -n "$tool" ] && [ -f "$tool" ]; then
    sha256_file "$tool"
  fi
}

quoted_args() {
  local out="" arg
  for arg in "$@"; do
    printf -v arg '%q' "$arg"
    out+="${out:+ }$arg"
  done
  printf '%s' "$out"
}

write_manifest_header() {
  local generated_at
  generated_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  {
    printf '# goav1_bench_corpus_manifest_v1\n'
    printf '# generated_at_utc=%s\n' "$generated_at"
    printf '# source_path=%s\n' "$SRC"
    printf '# source_sha256=%s\n' "$SRC_ACTUAL_SHA"
    printf '# frames=%s\n' "$FRAMES"
    printf '# fps=%s\n' "$FPS"
    printf '# expected_clips=%s\n' "$EXPECTED_CLIPS"
    printf '# aomenc_path=%s\n' "$AOMENC"
    printf '# aomenc_sha256=%s\n' "$(tool_sha256 "$AOMENC")"
    printf '# aomenc_threads=%s\n' "$AOM_THREADS"
    printf '# aomenc_row_mt=%s\n' "$AOM_ROW_MT"
    printf '# aomdec_path=%s\n' "$AOMDEC"
    printf '# aomdec_sha256=%s\n' "$(tool_sha256 "$AOMDEC")"
    printf '# dav1d_path=%s\n' "${DAV1D:-}"
    printf '# dav1d_sha256=%s\n' "$(tool_sha256 "${DAV1D:-}")"
    printf '# ffmpeg_path=%s\n' "$FFMPEG"
    printf '# ffmpeg_sha256=%s\n' "$(tool_sha256 "$FFMPEG")"
    printf 'name\twidth\theight\tframes\tcq\tdepth\tchroma\tprofile\tivf_bytes\tivf_sha256\tmd5\tmd5_sha256\tdav1d_check\taomenc_args\n'
  } > "$MANIFEST"
}

write_manifest_header

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
    -vf "fps=${FPS},scale=${w}:${h}:flags=bicubic,format=${fmt}" \
    -pix_fmt "$fmt" "${strict[@]}" "$out"
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

  local aom_args=(
    --quiet --cpu-used=6 --end-usage=q --cq-level="$cq"
    --threads="$AOM_THREADS" --row-mt="$AOM_ROW_MT"
    "${depth_args[@]}" "${profile_args[@]}" "${extra[@]}"
    --ivf -o "$ivf" "$src"
  )
  "$AOMENC" "${aom_args[@]}"

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
      echo "ERROR: dav1d md5 mismatch for $name" >&2
      echo "       dav1d: $d" >&2
      echo "       aomdec: $ref" >&2
      exit 1
    fi
  fi

  local bytes ivf_sha md5_sha profile aom_args_text
  bytes=$(wc -c < "$ivf" | tr -d ' ')
  ivf_sha=$(sha256_file "$ivf")
  md5_sha=$(sha256_file "$md5")
  profile=${profile_args[0]#--profile=}
  aom_args_text=$(quoted_args "${aom_args[@]}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$w" "$h" "$FRAMES" "$cq" "$depth" "$chroma" "$profile" \
    "$bytes" "$ivf_sha" "$ref" "$md5_sha" "$x" "$aom_args_text" >> "$MANIFEST"
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
encode p360_inter_q32_444_8bit  640 360 32 8 444
encode p360_inter_q32_444_10bit 640 360 32 10 444

# ---- 1280x720 (high res) ---------------------------------------------------
encode p720_inter_q20  1280 720 20 8
encode p720_inter_q32  1280 720 32 8
encode p720_inter_q55  1280 720 55 8
encode p720_inter_q32_2tiles 1280 720 32 8 --tile-columns=1

echo
clip_count=$(ls "$OUTDIR"/*.ivf 2>/dev/null | wc -l | tr -d ' ')
if [ "$clip_count" != "$EXPECTED_CLIPS" ]; then
  echo "ERROR: generated $clip_count clips, expected $EXPECTED_CLIPS" >&2
  exit 1
fi
echo "done. $clip_count clips in $OUTDIR"
echo "manifest: $MANIFEST"
echo "run an exploratory benchmark with:"
echo "  make bench-corpus"
echo "run a publish benchmark with manifest/hash/reference-decoder checks:"
echo "  make bench-corpus-publish"
