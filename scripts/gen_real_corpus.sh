#!/usr/bin/env bash
#
# gen_real_corpus.sh -- build the real-content AV1 benchmark corpus.
#
# testdata/benchcorpus (scripts/gen_bench_corpus.sh) is one source clip scaled
# to several sizes and encoded only by aomenc. This script instead pulls a
# small, diverse set of real camera / film / screen-capture sources from
# Xiph's media server and encodes them with the encoders goav1 actually
# receives streams from in production:
#
#   aomrt_*  libaom realtime, configured the way libwebrtc's
#            LibaomAv1Encoder configures it (third_party/upstream/webrtc/
#            modules/video_coding/codecs/av1/libaom_av1_encoder.cc), CBR,
#            screen-content tools on for screen captures
#   svtrtc_* SVT-AV1 rtc low-delay (preset 10-12, CBR)
#   svtra_*  SVT-AV1 preset 8 random access (VOD-style; one with film grain)
#   goav1_*  goav1's own realtime encoder (cmd/encbench)
#
# Nothing generated here is committed: OUTDIR is gitignored. Layout:
#
#   OUTDIR/orig/          downloaded originals (deleted after derivation
#                         unless GOAV1_REAL_CORPUS_KEEP_ORIG=1)
#   OUTDIR/src/<id>.y4m   derived sources: 8-bit 4:2:0, 30 fps, <= 300 frames
#   OUTDIR/sources.tsv    gen_bench_corpus.sh sources.tsv format (sha256 of
#                         the derived y4m), usable as GOAV1_BENCH_SOURCES_TSV
#   OUTDIR/streams/<name>.ivf + <name>.md5
#                         decode streams + libaom stream MD5 sidecars (the
#                         format TestCrossDecoderCorpus reads; point
#                         GOAV1_BENCH_CORPUS_DIR at OUTDIR/streams)
#   OUTDIR/streams.tsv    stream manifest (encoder, args, sizes, MD5s,
#                         aomdec-vs-dav1d agreement)
#
# Usage:
#   scripts/gen_real_corpus.sh [sources|streams|all] [OUTDIR]
#
# OUTDIR defaults to $GOAV1_REAL_CORPUS_DIR, then testdata/realcorpus.
#
# Tools: curl, ffmpeg, aomenc, aomdec, dav1d, SvtAv1EncApp (v4.1.0; set SVTENC
# to its path), go (to build cmd/encbench unless GOAV1_ENCBENCH is set).
#
# Source licences (recorded per row in sources.tsv):
#   * FourPeople, Johnny, vidyo1: Vidyo, listed as public domain on
#     https://media.xiph.org/video/derf/ (AOM CTC copies carry a Vidyo
#     standards-development notice for vidyo3/4).
#   * park_joy, old_town_cross, crowd_run: SVT High Definition Multi Format
#     Test Set; free redistribution without fee, but "may only be used for the
#     purpose of developing, testing and presenting technology standards"
#     (SVT_MultiFormat_v10.pdf). Do not use these rows in published material
#     outside that purpose; mtbike covers high motion under CC BY-SA.
#   * MountainBike: CC BY-SA 4.0, Neil Birkbeck (AOM CTC a2_2k).
#   * Wikipedia: CC0, Thomas Daede (AOM CTC b2_scc).
#   * Debugging: CC BY 4.0, Netflix Inc. (AOM CTC b2_scc).

set -euo pipefail

STAGE=${1:-all}
case "$STAGE" in
  sources|streams|all) ;;
  *) echo "usage: $0 [sources|streams|all] [OUTDIR]" >&2; exit 2 ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTDIR=${2:-${GOAV1_REAL_CORPUS_DIR:-$REPO_ROOT/testdata/realcorpus}}
mkdir -p "$OUTDIR/orig" "$OUTDIR/src" "$OUTDIR/streams"
OUTDIR=$(cd "$OUTDIR" && pwd)

FFMPEG=${FFMPEG:-$(command -v ffmpeg || true)}
AOMENC=${AOMENC:-$(command -v aomenc || true)}
AOMDEC=${AOMDEC:-$(command -v aomdec || true)}
DAV1D=${DAV1D:-$(command -v dav1d || true)}
SVTENC=${SVTENC:-$HOME/bin/svt410/SvtAv1EncApp}
CURL=${CURL:-$(command -v curl || true)}
for tool_var in FFMPEG AOMENC AOMDEC DAV1D CURL; do
  if [ -z "${!tool_var}" ]; then
    echo "ERROR: $tool_var not found on PATH" >&2
    exit 1
  fi
done

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

DERF=https://media.xiph.org/video/derf/y4m
CTC=https://media.xiph.org/video/aomctc/test_set
LIC_VIDYO="public domain (Vidyo; per media.xiph.org/video/derf listing)"
LIC_SVT="SVT HD Multi Format Test Set: free redistribution, no fee; use only for developing/testing/presenting technology standards (SVT_MultiFormat_v10.pdf)"

# id | original URL | original sha256 | licence | category | frames
# Every source is converted to 30 fps by dropping frames (ffmpeg fps=30),
# which keeps capture-rate motion per frame realistic for 30 fps RTC.
SOURCES=(
  "fourpeople|$DERF/FourPeople_1280x720_60.y4m|c8877dab4220847942eba4ab57ac36b4a0509694d25c42fe67a641ddbfa5cfe8|$LIC_VIDYO|conferencing|300"
  "johnny|$DERF/Johnny_1280x720_60.y4m|412eef72f06e22bad30cccc527eddc3ae1cb98f3c145472f13f30844520b2517|$LIC_VIDYO|conferencing|300"
  "vidyo1|$DERF/vidyo1_720p_60fps.y4m|ff0f87880309d1edde7391fdb108adf06ad4aed41e7f08a76d9e73971e941052|$LIC_VIDYO|conferencing|300"
  "parkjoy|$DERF/park_joy_420_720p50.y4m|b1e3784a6c74e07d53dd7077737b5a6edcd53e5ea07e4718c9affbe0ec36536f|$LIC_SVT|high-motion-film|300"
  "oldtown|$DERF/old_town_cross_420_720p50.y4m|0fbc2ec0c552ad693528b2e49409c7449880b9230055a7696346473d1aaf4123|$LIC_SVT|film-grain|300"
  "crowdrun|$DERF/crowd_run_1080p50.y4m|e056e01fc13906dd9a84f5cc686a36888436379476577965a0866821caaf55c8|$LIC_SVT|high-motion-film|300"
  "mtbike|$CTC/a2_2k/MountainBike_1920x1080_30fps_8bit.y4m|7b5f6dd23d8d5cde7623b208d2ab73c0e737150f7615c1e5b313bfcf69018261|CC BY-SA 4.0 (Neil Birkbeck)|high-motion-camera|130"
  "wikipedia|$CTC/b2_scc/Wikipedia_1920x1080p30.y4m|2fd9ac850d5d71f6274e7b8ae5828c3f949210874048f644fc91153bdae87c07|CC0 1.0 (Thomas Daede)|screen-content|130"
  "debugging|$CTC/b2_scc/Debugging_1920x1080_30fps_8bit_420.y4m|8004be81aedb40f9c26db25ff72d8725d832ef3b95747fb4c9fc8a3693eb4e8e|CC BY 4.0 (Netflix Inc.)|screen-content|130"
)

source_field() { # id field_index
  local row
  for row in "${SOURCES[@]}"; do
    IFS='|' read -r -a f <<< "$row"
    if [ "${f[0]}" = "$1" ]; then
      printf '%s' "${f[$2]}"
      return
    fi
  done
  echo "ERROR: unknown source $1" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Stage 1: sources
# ---------------------------------------------------------------------------
derive_sources() {
  local tsv="$OUTDIR/sources.tsv" row
  {
    printf '# goav1 real-content corpus sources (gen_bench_corpus.sh sources.tsv format)\n'
    printf '# derived with: ffmpeg -i ORIG -vf fps=30 -frames:v N -pix_fmt yuv420p (%s)\n' \
      "$("$FFMPEG" -hide_banner -version | awk 'NR==1{print $3}')"
    printf '# path\tsha256\tsource_id\tsource_url\tsource_license\tsource_category\n'
  } > "$tsv"
  for row in "${SOURCES[@]}"; do
    IFS='|' read -r id url want license category frames <<< "$row"
    local src="$OUTDIR/src/$id.y4m"
    if [ ! -f "$src" ]; then
      local orig="$OUTDIR/orig/$(basename "$url")"
      if [ ! -f "$orig" ] || [ "$(sha256_file "$orig")" != "$want" ]; then
        echo "fetch  $url"
        "$CURL" -sSf -o "$orig" "$url"
      fi
      local got; got=$(sha256_file "$orig")
      if [ "$got" != "$want" ]; then
        echo "ERROR: sha256 mismatch for $orig: got $got want $want" >&2
        exit 1
      fi
      echo "derive $id ($frames frames @30fps)"
      "$FFMPEG" -v error -y -i "$orig" -vf fps=30 -frames:v "$frames" -pix_fmt yuv420p "$src.tmp.y4m"
      mv "$src.tmp.y4m" "$src"
      if [ "${GOAV1_REAL_CORPUS_KEEP_ORIG:-}" != "1" ]; then
        rm -f "$orig"
      fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$src" "$(sha256_file "$src")" "$id" "$url" "$license" "$category" >> "$tsv"
  done
  echo "sources: $tsv"
}

# ---------------------------------------------------------------------------
# Stage 2: streams
# ---------------------------------------------------------------------------
WORK=""
cleanup() { if [ -n "$WORK" ]; then rm -rf "$WORK"; fi; }
trap cleanup EXIT

# scaled SRC_ID W H FMT(y4m|yuv) -> path of the source at WxH (cached).
scaled() {
  local id=$1 w=$2 h=$3 fmt=$4
  local src="$OUTDIR/src/$id.y4m"
  local out="$WORK/${id}_${w}x${h}.$fmt"
  if [ ! -f "$out" ]; then
    if [ "$fmt" = "yuv" ]; then
      "$FFMPEG" -v error -y -i "$src" -vf "scale=${w}:${h}:flags=bicubic" -pix_fmt yuv420p -f rawvideo "$out"
    else
      "$FFMPEG" -v error -y -i "$src" -vf "scale=${w}:${h}:flags=bicubic" -pix_fmt yuv420p "$out"
    fi
  fi
  printf '%s' "$out"
}

# libwebrtc LibaomAv1Encoder::InitEncode settings (non-SVC, no frame drop).
aom_webrtc_args() { # w h speed threads screen(0|1)
  local w=$1 h=$2 speed=$3 threads=$4 screen=$5
  local sb=dynamic
  if [ "$threads" -ge 4 ] && [ $((w * h)) -ge $((960 * 540)) ] && [ $((w * h)) -lt $((1920 * 1080)) ]; then
    sb=64
  fi
  local args=(
    --rt --cpu-used="$speed" --threads="$threads" --end-usage=cbr --lag-in-frames=0
    --min-q=10 --max-q=56 --undershoot-pct=50 --overshoot-pct=50
    --buf-initial-sz=600 --buf-optimal-sz=600 --buf-sz=1000 --disable-kf
    --error-resilient=0 --enable-cdef=1 --enable-tpl-model=0 --deltaq-mode=0
    --enable-order-hint=0 --aq-mode=3 --max-intra-rate=300
    --coeff-cost-upd-freq=3 --mode-cost-upd-freq=3 --mv-cost-upd-freq=3
    --auto-tiles=1 --row-mt=1 --enable-obmc=0 --noise-sensitivity=0
    --enable-warped-motion=0 --enable-global-motion=0 --enable-ref-frame-mvs=0
    --sb-size="$sb" --enable-cfl-intra=0 --enable-smooth-intra=0
    --enable-angle-delta=0 --enable-filter-intra=0 --use-intra-default-tx-only=1
    --disable-trellis-quant=1 --enable-dist-wtd-comp=0 --enable-diff-wtd-comp=0
    --enable-dual-filter=0 --enable-interintra-comp=0 --enable-interintra-wedge=0
    --enable-intra-edge-filter=0 --enable-intrabc=0 --enable-masked-comp=0
    --enable-paeth-intra=0 --enable-qm=0 --enable-rect-partitions=0
    --enable-restoration=0 --enable-smooth-interintra=0 --enable-tx64=0
    --max-reference-frames=3
  )
  if [ "$screen" = "1" ]; then
    args+=(--tune-content=screen --enable-palette=1)
  else
    args+=(--enable-palette=0)
  fi
  printf '%s\n' "${args[@]}"
}

# libwebrtc NumberOfThreads() on a >8-core host.
webrtc_threads() {
  local px=$(($1 * $2))
  if [ "$px" -gt $((1280 * 720)) ]; then echo 8
  elif [ "$px" -ge $((640 * 360)) ]; then echo 4
  elif [ "$px" -ge $((320 * 180)) ]; then echo 2
  else echo 1; fi
}

MANIFEST="$OUTDIR/streams.tsv"

record() { # name encoder src w h frames args...
  local name=$1 encoder=$2 id=$3 w=$4 h=$5 frames=$6; shift 6
  local ivf="$OUTDIR/streams/$name.ivf"
  local ref d check
  ref=$("$AOMDEC" --i420 --md5 "$ivf" 2>/dev/null | awk 'NR==1{print $1}')
  d=$("$DAV1D" -q --filmgrain 1 --muxer md5 -o - -i "$ivf" 2>/dev/null || true)
  if [ -n "$ref" ] && [ "$d" = "$ref" ]; then
    check="dav1d=OK"
  else
    check="dav1d=MISMATCH($d)"
    echo "WARNING: aomdec/dav1d disagree on $name: aomdec=$ref dav1d=$d" >&2
  fi
  printf '%s\n' "$ref" > "$OUTDIR/streams/$name.md5"
  local bytes; bytes=$(wc -c < "$ivf" | tr -d ' ')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$encoder" "$id" "$w" "$h" "$frames" "$bytes" "$(sha256_file "$ivf")" \
    "$ref" "$check" "$*" >> "$MANIFEST"
  printf '  %-44s %-6s %9s bytes  md5=%s  %s\n' "$name" "${w}x${h}" "$bytes" "$ref" "$check"
}

stream_aomrt() { # name src w h kbps speed screen
  local name=$1 id=$2 w=$3 h=$4 kbps=$5 speed=$6 screen=$7
  local frames; frames=$(source_field "$id" 5)
  local in; in=$(scaled "$id" "$w" "$h" y4m)
  local args=()
  while IFS= read -r a; do args+=("$a"); done < <(aom_webrtc_args "$w" "$h" "$speed" "$(webrtc_threads "$w" "$h")" "$screen")
  args+=(--target-bitrate="$kbps" --limit="$frames")
  "$AOMENC" --quiet "${args[@]}" --ivf -o "$OUTDIR/streams/$name.ivf" "$in" 2>/dev/null
  record "$name" "aomenc-rt" "$id" "$w" "$h" "$frames" "${args[*]}"
}

stream_svtrtc() { # name src w h kbps preset extra...
  local name=$1 id=$2 w=$3 h=$4 kbps=$5 preset=$6; shift 6
  local frames; frames=$(source_field "$id" 5)
  local in; in=$(scaled "$id" "$w" "$h" yuv)
  local args=(
    -w "$w" -h "$h" --fps-num 30 --fps-denom 1 --frames "$frames" --input-depth 8
    --color-format 1 --profile 0 --level 0 --preset "$preset" --rc 2 --tbr "$kbps"
    --buf-sz 1000 --buf-initial-sz 500 --buf-optimal-sz 600 --lookahead 0
    --pred-struct 1 --rtc 1 --scd 0 --enable-tf 0 --irefresh-type 2 --keyint -1
    --progress 0 "$@"
  )
  "$SVTENC" -i "$in" -b "$OUTDIR/streams/$name.ivf" "${args[@]}" >/dev/null 2>&1
  record "$name" "svt-rtc" "$id" "$w" "$h" "$frames" "${args[*]}"
}

stream_svtra() { # name src w h crf extra...
  local name=$1 id=$2 w=$3 h=$4 crf=$5; shift 5
  local frames; frames=$(source_field "$id" 5)
  local in; in=$(scaled "$id" "$w" "$h" yuv)
  local args=(
    -w "$w" -h "$h" --fps-num 30 --fps-denom 1 --frames "$frames" --input-depth 8
    --color-format 1 --profile 0 --preset 8 --rc 0 --crf "$crf" --pred-struct 2
    --keyint 150 --progress 0 "$@"
  )
  "$SVTENC" -i "$in" -b "$OUTDIR/streams/$name.ivf" "${args[@]}" >/dev/null 2>&1
  record "$name" "svt-ra" "$id" "$w" "$h" "$frames" "${args[*]}"
}

stream_goav1() { # name src w h bps extra...
  local name=$1 id=$2 w=$3 h=$4 bps=$5; shift 5
  local frames; frames=$(source_field "$id" 5)
  local in; in=$(scaled "$id" "$w" "$h" yuv)
  local args=(-width "$w" -height "$h" -fps 30 -frames "$frames" -bitrate "$bps" -nopsnr "$@")
  "$ENCBENCH" -input "$in" -o "$OUTDIR/streams/$name.ivf" "${args[@]}" >/dev/null 2>&1
  record "$name" "goav1" "$id" "$w" "$h" "$frames" "${args[*]}"
}

gen_streams() {
  if [ ! -x "$SVTENC" ]; then
    echo "ERROR: SvtAv1EncApp not found at $SVTENC (set SVTENC)" >&2
    exit 1
  fi
  WORK=$(mktemp -d "$OUTDIR/work.XXXXXX")
  ENCBENCH=${GOAV1_ENCBENCH:-}
  if [ -z "$ENCBENCH" ]; then
    ENCBENCH="$WORK/encbench"
    (cd "$REPO_ROOT" && GOEXPERIMENT=simd go build -pgo=cmd/qualitybench/default.pgo -o "$ENCBENCH" ./cmd/encbench)
  fi
  {
    printf '# goav1_real_corpus_streams_v1\n'
    printf '# aomenc=%s\n' "$("$AOMENC" --help 2>&1 | grep -m1 -o 'v[0-9][^ ]*' || true)"
    printf '# svt=%s\n' "$("$SVTENC" --version 2>&1 | awk 'NF{print; exit}')"
    printf '# dav1d=%s\n' "$("$DAV1D" --version 2>&1 | awk 'NF{print; exit}')"
    printf '# goav1=%s\n' "$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    printf 'name\tencoder\tsource_id\twidth\theight\tframes\tivf_bytes\tivf_sha256\tmd5\tdav1d_check\targs\n'
  } > "$MANIFEST"

  echo "libaom realtime (libwebrtc settings):"
  stream_aomrt aomrt_fourpeople_720p_1000k_s10   fourpeople 1280 720 1000 10 0
  stream_aomrt aomrt_johnny_720p_1500k_s9        johnny     1280 720 1500  9 0
  stream_aomrt aomrt_vidyo1_720p_2500k_s8        vidyo1     1280 720 2500  8 0
  stream_aomrt aomrt_parkjoy_720p_2500k_s10      parkjoy    1280 720 2500 10 0
  stream_aomrt aomrt_oldtown_720p_1500k_s7       oldtown    1280 720 1500  7 0
  stream_aomrt aomrt_crowdrun_1080p_4000k_s10    crowdrun   1920 1080 4000 10 0
  stream_aomrt aomrt_mtbike_1080p_3000k_s9       mtbike     1920 1080 3000  9 0
  stream_aomrt aomrt_wikipedia_1080p_1500k_s11   wikipedia  1920 1080 1500 11 1
  stream_aomrt aomrt_debugging_720p_800k_s11     debugging  1280 720  800 11 1
  stream_aomrt aomrt_fourpeople_360p_500k_s9     fourpeople  640 360  500  9 0
  stream_aomrt aomrt_mtbike_360p_600k_s8         mtbike      640 360  600  8 0

  echo "SVT-AV1 rtc:"
  stream_svtrtc svtrtc_fourpeople_720p_1000k_p12 fourpeople 1280 720 1000 12
  stream_svtrtc svtrtc_parkjoy_720p_2000k_p10    parkjoy    1280 720 2000 10
  stream_svtrtc svtrtc_crowdrun_1080p_3000k_p11  crowdrun   1920 1080 3000 11
  stream_svtrtc svtrtc_wikipedia_1080p_1000k_p12 wikipedia  1920 1080 1000 12 --scm 1
  stream_svtrtc svtrtc_johnny_360p_500k_p12      johnny      640 360  500 12

  echo "SVT-AV1 preset 8 random access:"
  stream_svtra svtra_johnny_720p_crf35_p8        johnny     1280 720 35
  stream_svtra svtra_oldtown_720p_crf35_p8_fgs8  oldtown    1280 720 35 --film-grain 8
  stream_svtra svtra_mtbike_1080p_crf35_p8_2tile mtbike     1920 1080 35 --tile-columns 1

  echo "goav1 realtime:"
  stream_goav1 goav1_fourpeople_720p_1000k       fourpeople 1280 720 1000000
  stream_goav1 goav1_parkjoy_720p_2000k          parkjoy    1280 720 2000000
  stream_goav1 goav1_debugging_720p_1000k        debugging  1280 720 1000000
  stream_goav1 goav1_crowdrun_1080p_3000k        crowdrun   1920 1080 3000000
  stream_goav1 goav1_johnny_360p_500k            johnny      640 360  500000

  echo "streams: $MANIFEST"
}

case "$STAGE" in
  sources) derive_sources ;;
  streams) gen_streams ;;
  all) derive_sources; gen_streams ;;
esac
