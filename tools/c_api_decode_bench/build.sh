#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build_dir=${GOAV1_C_API_BENCH_BUILD_DIR:-"${TMPDIR:-/tmp}/goav1-c-api-decode-bench-${CODEX_AGENT_ID:-local}"}
aom_build=${GOAV1_AOM_BUILD_DIR:-/tmp/goav1-quality-aom-build}
pkg_config=${PKG_CONFIG:-pkg-config}
cc=${CC:-cc}

if [[ -n "${GOAV1_AOM_SOURCE:-}" ]]; then
  aom_source=$GOAV1_AOM_SOURCE
else
  aom_source=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$aom_build/CMakeCache.txt" 2>/dev/null | head -n 1)
  if [[ -z "$aom_source" ]]; then
    aom_source="$root/third_party/upstream/libaom"
  fi
fi

expected_aom_commit=047d8cf6168feafe1300eb6902000dd1a03d5549
expected_dav1d_commit=b546257f770768b2c88258c533da38b91a06f737
expected_dav1d_version=${GOAV1_C_API_EXPECT_DAV1D_VERSION:-1.5.3}
flags=(-O3 -DNDEBUG -std=c11 -Wall -Wextra -Wconversion)
flags_text="-O3 -DNDEBUG -std=c11 -Wall -Wextra -Wconversion"

if [[ ! -d "$aom_source/.git" ]]; then
  echo "libaom source clone not found: $aom_source (set GOAV1_AOM_SOURCE)" >&2
  exit 1
fi
if [[ ! -f "$aom_build/libaom.a" || ! -f "$aom_build/config/aom_config.h" ]]; then
  echo "pinned libaom build is incomplete: $aom_build (set GOAV1_AOM_BUILD_DIR)" >&2
  exit 1
fi
aom_commit=$(git -C "$aom_source" rev-parse HEAD)
if [[ "$aom_commit" != "$expected_aom_commit" ]]; then
  echo "libaom source is $aom_commit; expected pinned $expected_aom_commit" >&2
  exit 1
fi
dav1d_version=$("$pkg_config" --modversion dav1d)
if [[ "$dav1d_version" != "$expected_dav1d_version" ]]; then
  echo "pkg-config dav1d is $dav1d_version; expected $expected_dav1d_version" >&2
  exit 1
fi
if ! command -v "$cc" >/dev/null 2>&1; then
  echo "C compiler not found: $cc" >&2
  exit 1
fi
mkdir -p "$build_dir"

build_flags_define="-DGOAV1_BUILD_FLAGS=\"$flags_text\""
aom_commit_define="-DGOAV1_AOM_SOURCE_COMMIT=\"$aom_commit\""
dav1d_commit_define="-DGOAV1_DAV1D_REFERENCE_COMMIT=\"$expected_dav1d_commit\""
common_includes=(-I"$aom_source" -I"$aom_build" -I"$aom_build/config")

"$cc" "${flags[@]}" "$build_flags_define" "$aom_commit_define" \
  "$dav1d_commit_define" -DGOAV1_DECODER_AOM "${common_includes[@]}" \
  "$root/tools/c_api_decode_bench/bench.c" \
  "$aom_source/common/md5_utils.c" "$aom_source/common/rawenc.c" \
  "$aom_build/libaom.a" -pthread -lm \
  -o "$build_dir/c-api-decode-bench-aom"

read -r -a dav1d_cflags <<<"$("$pkg_config" --cflags dav1d)"
read -r -a dav1d_libs <<<"$("$pkg_config" --libs dav1d)"
"$cc" "${flags[@]}" "$build_flags_define" "$aom_commit_define" \
  "$dav1d_commit_define" -DGOAV1_DECODER_DAV1D "${common_includes[@]}" \
  "${dav1d_cflags[@]}" "$root/tools/c_api_decode_bench/bench.c" \
  "$aom_source/common/md5_utils.c" "${dav1d_libs[@]}" -pthread \
  -o "$build_dir/c-api-decode-bench-dav1d"

printf 'build_dir=%s\n' "$build_dir"
printf 'compiler=%s\n' "$("$cc" --version | head -n 1)"
printf 'compiler_flags=%s\n' "$flags_text"
printf 'libaom_source_commit=%s\n' "$aom_commit"
printf 'libaom_build_dir=%s\n' "$aom_build"
printf 'dav1d_pkg_config_version=%s\n' "$dav1d_version"
printf 'dav1d_reference_commit=%s\n' "$expected_dav1d_commit"
printf 'aom_binary=%s\n' "$build_dir/c-api-decode-bench-aom"
printf 'dav1d_binary=%s\n' "$build_dir/c-api-decode-bench-dav1d"
