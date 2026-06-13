#!/usr/bin/env sh
set -eu

base_url="${LIBAOM_TEST_DATA_URL:-https://storage.googleapis.com/aom-test-data}"
fixture_dir="${LIBAOM_TEST_DATA_DIR:-internal/av1/testdata/libaom}"

mkdir -p "$fixture_dir"

for name in \
	av1-1-b8-00-quantizer-00.ivf \
	av1-1-b8-22-svc-L1T2.ivf \
	av1-1-b8-22-svc-L2T1.ivf
do
	path="$fixture_dir/$name"
	if [ ! -f "$path" ]; then
		tmp="$path.tmp"
		curl -fsSL -o "$tmp" "$base_url/$name"
		mv "$tmp" "$path"
	fi
done
