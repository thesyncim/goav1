#!/usr/bin/env bash
# Builds the libdav1d validation oracle used by decode_validation_diff_test.go
# and prints the binary path; pass it to the test as GOAV1_DAV1D_ORACLE.
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build_dir=${GOAV1_DAV1D_ORACLE_BUILD_DIR:-"${TMPDIR:-/tmp}/goav1-dav1d-validation-oracle"}
pkg_config=${PKG_CONFIG:-pkg-config}
cc=${CC:-cc}
expected_dav1d_version=${GOAV1_EXPECT_DAV1D_VERSION:-1.5.4}

dav1d_version=$("$pkg_config" --modversion dav1d)
if [[ "$dav1d_version" != "$expected_dav1d_version" ]]; then
  echo "pkg-config dav1d is $dav1d_version; expected $expected_dav1d_version" >&2
  exit 1
fi
mkdir -p "$build_dir"
read -r -a dav1d_cflags <<<"$("$pkg_config" --cflags dav1d)"
read -r -a dav1d_libs <<<"$("$pkg_config" --libs dav1d)"
"$cc" -O2 -std=c11 -Wall -Wextra "${dav1d_cflags[@]}" \
  "$root/tools/dav1d_validation_oracle/oracle.c" "${dav1d_libs[@]}" \
  -o "$build_dir/dav1d-validation-oracle"
echo "$build_dir/dav1d-validation-oracle"
