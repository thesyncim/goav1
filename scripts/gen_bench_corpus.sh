#!/usr/bin/env bash
#
# gen_bench_corpus.sh -- regenerate the multi-config AV1 decode benchmark corpus.
#
# The cross-decoder throughput benchmark (TestCrossDecoderCorpus in
# internal/av1/testvector/cross_decoder_corpus_bench_test.go) needs clips that
# are long enough (~30-60 frames) that steady-state decode dominates process
# startup. The bundled libaom conformance vectors are only a couple of frames
# each, so this script synthesizes a representative matrix from explicitly
# supplied source clips by scaling/length-extending with ffmpeg and encoding
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
# Publishable multi-source input:
#   GOAV1_BENCH_SOURCES_TSV=/path/to/sources.tsv
#
# sources.tsv is tab-separated with one source per line:
#   path sha256 source_id source_url source_license source_category
#
# Blank lines and lines beginning with # are ignored. Multi-source mode emits a
# v2 manifest with row-level source provenance and prefixes each generated clip
# with a sanitized source_id.
#
# Backward-compatible single-source input:
#   GOAV1_BENCH_SOURCE=/path/to/source_8bit_420.y4m
#   GOAV1_BENCH_SOURCE_SHA256=<sha256 of that source>
#   GOAV1_BENCH_SOURCE_ID=<stable source identifier>
#   GOAV1_BENCH_SOURCE_URL=<source URL or internal provenance URI>
#   GOAV1_BENCH_SOURCE_LICENSE=<license or usage grant>
#   GOAV1_BENCH_SOURCE_CATEGORY=<content category>
#
# dav1d is required by default because publishable corpus manifests must record
# generator-time dav1d MD5 agreement for 8-bit 4:2:0 rows. Set
# GOAV1_BENCH_CORPUS_ALLOW_MISSING_DAV1D=1 only for exploratory local corpora
# that will never be used for publishable goav1-vs-dav1d tables.
#
# Publishable corpus generation requires absolute tool paths and SHA-256 pins:
#   GOAV1_BENCH_AOMENC_SHA256=<sha256 of $AOMENC>
#   GOAV1_BENCH_AOMDEC_SHA256=<sha256 of $AOMDEC>
#   GOAV1_BENCH_FFMPEG_SHA256=<sha256 of $FFMPEG>
#   GOAV1_BENCH_DAV1D_SHA256=<sha256 of $DAV1D>
# Set GOAV1_BENCH_CORPUS_ALLOW_UNPINNED_TOOLS=1 only for exploratory local
# corpora that will never be used for publishable benchmark tables.

set -euo pipefail

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

canonical_sha256() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

verify_tool_pin() {
  local tool_name=$1 tool_path=$2 expected_sha=$3 env_name=$4
  if [ "${GOAV1_BENCH_CORPUS_ALLOW_UNPINNED_TOOLS:-}" = "1" ]; then
    return
  fi
  if [ -z "$expected_sha" ]; then
    echo "ERROR: set $env_name to pin $tool_name for publishable corpus generation" >&2
    echo "       set GOAV1_BENCH_CORPUS_ALLOW_UNPINNED_TOOLS=1 only for exploratory non-publishable corpora" >&2
    exit 1
  fi
  case "$tool_path" in
    /*) ;;
    *)
      echo "ERROR: $tool_name path must be absolute for publishable corpus generation: $tool_path" >&2
      exit 1
      ;;
  esac
  if [ ! -f "$tool_path" ]; then
    echo "ERROR: $tool_name path is not a file: $tool_path" >&2
    exit 1
  fi
  local want got
  want=$(canonical_sha256 "$expected_sha")
  if [[ ! "$want" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: $env_name must be a 64-hex SHA-256, got: $expected_sha" >&2
    exit 1
  fi
  got=$(sha256_file "$tool_path")
  if [ "$got" != "$want" ]; then
    echo "ERROR: $tool_name sha256 mismatch for $tool_path" >&2
    echo "       got  $got" >&2
    echo "       want $want" >&2
    exit 1
  fi
}

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
if [ -z "$DAV1D" ] && [ "${GOAV1_BENCH_CORPUS_ALLOW_MISSING_DAV1D:-}" != "1" ]; then
  echo "ERROR: DAV1D not found on PATH" >&2
  echo "       install dav1d or set GOAV1_BENCH_CORPUS_ALLOW_MISSING_DAV1D=1 for exploratory non-publishable corpus generation" >&2
  exit 1
fi
verify_tool_pin "aomenc" "$AOMENC" "${GOAV1_BENCH_AOMENC_SHA256:-}" "GOAV1_BENCH_AOMENC_SHA256"
verify_tool_pin "aomdec" "$AOMDEC" "${GOAV1_BENCH_AOMDEC_SHA256:-}" "GOAV1_BENCH_AOMDEC_SHA256"
verify_tool_pin "ffmpeg" "$FFMPEG" "${GOAV1_BENCH_FFMPEG_SHA256:-}" "GOAV1_BENCH_FFMPEG_SHA256"
if [ -n "$DAV1D" ]; then
  verify_tool_pin "dav1d" "$DAV1D" "${GOAV1_BENCH_DAV1D_SHA256:-}" "GOAV1_BENCH_DAV1D_SHA256"
fi

# --- locate source + output dir ---------------------------------------------
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SOURCE_PATHS=()
SOURCE_SHAS=()
SOURCE_IDS=()
SOURCE_URLS=()
SOURCE_LICENSES=()
SOURCE_CATEGORIES=()
SOURCE_SAFE_IDS=()
MANIFEST_VERSION=1

sanitize_source_id() {
  local id=$1 safe
  safe=$(printf '%s' "$id" | LC_ALL=C tr -c '[:alnum:]_.-' '_' | sed 's/^_*//; s/_*$//; s/__*/_/g')
  if [ -z "$safe" ]; then
    echo "ERROR: source_id $id does not contain any filename-safe characters" >&2
    exit 1
  fi
  printf '%s' "$safe"
}

add_source() {
  local source_path=$1 expected_sha=$2 source_id=$3 source_url=$4 source_license=$5 source_category=$6 origin=$7
  if [ -z "$source_path" ] || [ -z "$expected_sha" ] || [ -z "$source_id" ] || [ -z "$source_url" ] || [ -z "$source_license" ] || [ -z "$source_category" ]; then
    echo "ERROR: $origin must provide path, sha256, source_id, source_url, source_license, and source_category" >&2
    exit 1
  fi
  if [ ! -f "$source_path" ]; then
    echo "ERROR: source y4m not found for $origin: $source_path" >&2
    exit 1
  fi
  expected_sha=$(canonical_sha256 "$expected_sha")
  if [[ ! "$expected_sha" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: source sha256 for $origin must be a 64-hex SHA-256, got: $expected_sha" >&2
    exit 1
  fi
  local actual_sha
  actual_sha=$(sha256_file "$source_path")
  if [ "$actual_sha" != "$expected_sha" ]; then
    echo "ERROR: source sha256 mismatch for $source_path" >&2
    echo "       got  $actual_sha" >&2
    echo "       want $expected_sha" >&2
    exit 1
  fi
  local safe_id existing
  safe_id=$(sanitize_source_id "$source_id")
  for existing in "${SOURCE_IDS[@]}"; do
    if [ "$existing" = "$source_id" ]; then
      echo "ERROR: duplicate source_id $source_id in benchmark corpus sources" >&2
      exit 1
    fi
  done
  for existing in "${SOURCE_SAFE_IDS[@]}"; do
    if [ "$existing" = "$safe_id" ]; then
      echo "ERROR: source_id $source_id sanitizes to duplicate clip prefix $safe_id" >&2
      exit 1
    fi
  done
  SOURCE_PATHS+=("$source_path")
  SOURCE_SHAS+=("$actual_sha")
  SOURCE_IDS+=("$source_id")
  SOURCE_URLS+=("$source_url")
  SOURCE_LICENSES+=("$source_license")
  SOURCE_CATEGORIES+=("$source_category")
  SOURCE_SAFE_IDS+=("$safe_id")
}

source_category_count() {
  local categories=() category existing found count=0
  for category in "${SOURCE_CATEGORIES[@]}"; do
    found=0
    for existing in "${categories[@]}"; do
      if [ "$existing" = "$category" ]; then
        found=1
        break
      fi
    done
    if [ "$found" = "0" ]; then
      categories+=("$category")
      count=$((count + 1))
    fi
  done
  printf '%s' "$count"
}

if [ -n "${GOAV1_BENCH_SOURCES_TSV:-}" ]; then
  MANIFEST_VERSION=2
  if [ ! -f "$GOAV1_BENCH_SOURCES_TSV" ]; then
    echo "ERROR: GOAV1_BENCH_SOURCES_TSV not found: $GOAV1_BENCH_SOURCES_TSV" >&2
    exit 1
  fi
  line_no=0
  while IFS= read -r line || [ -n "$line" ]; do
    line_no=$((line_no + 1))
    case "$line" in
      ''|\#*) continue ;;
    esac
    IFS=$'\t' read -r source_path expected_sha source_id source_url source_license source_category extra <<< "$line"
    if [ -n "${extra:-}" ]; then
      echo "ERROR: $GOAV1_BENCH_SOURCES_TSV:$line_no has more than 6 tab-separated fields" >&2
      exit 1
    fi
    add_source "$source_path" "$expected_sha" "$source_id" "$source_url" "$source_license" "$source_category" "$GOAV1_BENCH_SOURCES_TSV:$line_no"
  done < "$GOAV1_BENCH_SOURCES_TSV"
  if [ "${#SOURCE_PATHS[@]}" -lt 2 ]; then
    echo "ERROR: GOAV1_BENCH_SOURCES_TSV must list at least two sources for publishable v2 corpus generation" >&2
    exit 1
  fi
  if [ "$(source_category_count)" -lt 2 ]; then
    echo "ERROR: GOAV1_BENCH_SOURCES_TSV must list at least two source categories for publishable v2 corpus generation" >&2
    exit 1
  fi
else
  add_source "${GOAV1_BENCH_SOURCE:-}" "${GOAV1_BENCH_SOURCE_SHA256:-}" "${GOAV1_BENCH_SOURCE_ID:-}" \
    "${GOAV1_BENCH_SOURCE_URL:-}" "${GOAV1_BENCH_SOURCE_LICENSE:-}" "${GOAV1_BENCH_SOURCE_CATEGORY:-}" \
    "GOAV1_BENCH_SOURCE* environment"
fi

OUTDIR=${1:-${GOAV1_BENCH_CORPUS_DIR:-$REPO_ROOT/testdata/benchcorpus}}
mkdir -p "$OUTDIR"

# Number of frames each clip is length-extended to (steady-state dominates
# startup at this length). The source clip is short, so ffmpeg loops it.
FRAMES=${GOAV1_BENCH_FRAMES:-48}
FPS=${GOAV1_BENCH_FPS:-30}
SOURCE_COUNT=${#SOURCE_PATHS[@]}
EXPECTED_CLIPS=$((25 * SOURCE_COUNT))
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

echo "sources: $SOURCE_COUNT"
for i in "${!SOURCE_PATHS[@]}"; do
  printf '  %-20s category=%-18s sha256=%s\n' "${SOURCE_IDS[$i]}" "${SOURCE_CATEGORIES[$i]}" "${SOURCE_SHAS[$i]}"
done
echo "outdir : $OUTDIR"
echo "frames : $FRAMES"
echo "fps    : $FPS"
echo "aomenc : threads=$AOM_THREADS row-mt=$AOM_ROW_MT"
echo

tool_sha256() {
  local tool=$1
  if [ -n "$tool" ] && [ -f "$tool" ]; then
    sha256_file "$tool"
  fi
}

tool_version() {
  local tool=$1; shift
  if [ -n "$tool" ] && [ -f "$tool" ]; then
    "$tool" "$@" 2>&1 | awk 'NF { print; exit }' || true
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
  {
    if [ "$MANIFEST_VERSION" = "2" ]; then
      printf '# goav1_bench_corpus_manifest_v2\n'
      printf '# source_count=%s\n' "$SOURCE_COUNT"
    else
      printf '# goav1_bench_corpus_manifest_v1\n'
      printf '# source_sha256=%s\n' "${SOURCE_SHAS[0]}"
      printf '# source_id=%s\n' "${SOURCE_IDS[0]}"
      printf '# source_url=%s\n' "${SOURCE_URLS[0]}"
      printf '# source_license=%s\n' "${SOURCE_LICENSES[0]}"
      printf '# source_category=%s\n' "${SOURCE_CATEGORIES[0]}"
    fi
    if [ -n "${GOAV1_BENCH_GENERATED_AT_UTC:-}" ]; then
      printf '# generated_at_utc=%s\n' "$GOAV1_BENCH_GENERATED_AT_UTC"
    fi
    printf '# frames=%s\n' "$FRAMES"
    printf '# fps=%s\n' "$FPS"
    printf '# expected_clips=%s\n' "$EXPECTED_CLIPS"
    printf '# aomenc_sha256=%s\n' "$(tool_sha256 "$AOMENC")"
    printf '# aomenc_version=%s\n' "$(tool_version "$AOMENC" --version)"
    printf '# aomenc_threads=%s\n' "$AOM_THREADS"
    printf '# aomenc_row_mt=%s\n' "$AOM_ROW_MT"
    printf '# aomdec_sha256=%s\n' "$(tool_sha256 "$AOMDEC")"
    printf '# aomdec_version=%s\n' "$(tool_version "$AOMDEC" --help)"
    printf '# dav1d_sha256=%s\n' "$(tool_sha256 "${DAV1D:-}")"
    printf '# dav1d_version=%s\n' "$(tool_version "${DAV1D:-}" --version)"
    printf '# ffmpeg_sha256=%s\n' "$(tool_sha256 "$FFMPEG")"
    printf '# ffmpeg_version=%s\n' "$(tool_version "$FFMPEG" -hide_banner -version)"
    if [ "$MANIFEST_VERSION" = "2" ]; then
      printf 'name\twidth\theight\tframes\tcq\tdepth\tchroma\tprofile\tivf_bytes\tivf_sha256\tmd5\tmd5_sha256\tdav1d_check\taomenc_args\tsource_id\tsource_sha256\tsource_url\tsource_license\tsource_category\n'
    else
      printf 'name\twidth\theight\tframes\tcq\tdepth\tchroma\tprofile\tivf_bytes\tivf_sha256\tmd5\tmd5_sha256\tdav1d_check\taomenc_args\n'
    fi
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
  local key="${CURRENT_SRC_SAFE}_${w}x${h}_${chroma}_${depth}"
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
  "$FFMPEG" -v error -y -stream_loop 200 -i "$CURRENT_SRC" -frames:v "$FRAMES" \
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
  local clip_name=$name
  if [ -n "${CURRENT_NAME_PREFIX:-}" ]; then
    clip_name="${CURRENT_NAME_PREFIX}_${name}"
  fi
  local src; src=$(scaled_source "$w" "$h" "$depth" "$chroma")
  local ivf="$OUTDIR/$clip_name.ivf"
  local md5="$OUTDIR/$clip_name.md5"

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
  local aom_record_args=(
    --cpu-used=6 --end-usage=q --cq-level="$cq"
    --threads="$AOM_THREADS" --row-mt="$AOM_ROW_MT"
    "${depth_args[@]}" "${profile_args[@]}" "${extra[@]}"
    --ivf
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
      echo "ERROR: dav1d md5 mismatch for $clip_name" >&2
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
  aom_args_text=$(quoted_args "${aom_record_args[@]}")
  if [ "$MANIFEST_VERSION" = "2" ]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$clip_name" "$w" "$h" "$FRAMES" "$cq" "$depth" "$chroma" "$profile" \
      "$bytes" "$ivf_sha" "$ref" "$md5_sha" "$x" "$aom_args_text" \
      "$CURRENT_SRC_ID" "$CURRENT_SRC_SHA" "$CURRENT_SRC_URL" "$CURRENT_SRC_LICENSE" "$CURRENT_SRC_CATEGORY" >> "$MANIFEST"
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$clip_name" "$w" "$h" "$FRAMES" "$cq" "$depth" "$chroma" "$profile" \
      "$bytes" "$ivf_sha" "$ref" "$md5_sha" "$x" "$aom_args_text" >> "$MANIFEST"
  fi
  printf '  %-30s %-9s cq=%-2s d=%-2s c=%-3s %8s bytes  md5=%s  %s\n' \
    "$clip_name" "${w}x${h}" "$cq" "$depth" "$chroma" "$bytes" "$ref" "$x"
}

encode_matrix_for_current_source() {
  # ---- 256x144 (low res) ---------------------------------------------------
  encode p144_intra_q32  256 144 32 8 --kf-min-dist=0 --kf-max-dist=0
  encode p144_inter_q20  256 144 20 8
  encode p144_inter_q32  256 144 32 8
  encode p144_inter_q55  256 144 55 8

  # ---- 512x288 (mid-low res) -----------------------------------------------
  encode p288_intra_q32  512 288 32 8 --kf-min-dist=0 --kf-max-dist=0
  encode p288_inter_q20  512 288 20 8
  encode p288_inter_q32  512 288 32 8
  encode p288_inter_q55  512 288 55 8

  # ---- 640x360 (mid res) ---------------------------------------------------
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

  # ---- 1280x720 (high res) -------------------------------------------------
  encode p720_inter_q20  1280 720 20 8
  encode p720_inter_q32  1280 720 32 8
  encode p720_inter_q55  1280 720 55 8
  encode p720_inter_q32_2tiles 1280 720 32 8 --tile-columns=1
}

echo "encoding corpus..."
for i in "${!SOURCE_PATHS[@]}"; do
  CURRENT_SRC=${SOURCE_PATHS[$i]}
  CURRENT_SRC_SHA=${SOURCE_SHAS[$i]}
  CURRENT_SRC_ID=${SOURCE_IDS[$i]}
  CURRENT_SRC_URL=${SOURCE_URLS[$i]}
  CURRENT_SRC_LICENSE=${SOURCE_LICENSES[$i]}
  CURRENT_SRC_CATEGORY=${SOURCE_CATEGORIES[$i]}
  CURRENT_SRC_SAFE=${SOURCE_SAFE_IDS[$i]}
  CURRENT_NAME_PREFIX=""
  if [ "$MANIFEST_VERSION" = "2" ]; then
    CURRENT_NAME_PREFIX=$CURRENT_SRC_SAFE
  fi
  printf '\nsource %s/%s: %s (%s)\n' "$((i + 1))" "$SOURCE_COUNT" "$CURRENT_SRC_ID" "$CURRENT_SRC_CATEGORY"
  encode_matrix_for_current_source
done

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
