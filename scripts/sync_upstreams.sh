#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream="$root/third_party/upstream"
mkdir -p "$upstream"

sync_repo() {
	name=$1
	url=$2
	ref=$3
	commit=$4
	shift 4

	dir="$upstream/$name"
	if [ ! -d "$dir/.git" ]; then
		git clone --filter=blob:none --no-checkout "$url" "$dir"
	fi

	git -C "$dir" remote set-url origin "$url"
	git -C "$dir" fetch --depth=1 origin "$ref"

	if [ "$#" -gt 0 ]; then
		git -C "$dir" sparse-checkout init --no-cone
		git -C "$dir" sparse-checkout set "$@"
	fi

	git -C "$dir" checkout --detach "$commit"
}

sync_repo \
	dav1d \
	https://code.videolan.org/videolan/dav1d.git \
	refs/tags/1.5.3 \
	b546257f770768b2c88258c533da38b91a06f737 \
	src include tests tools /meson.build /NEWS /COPYING

sync_repo \
	libaom \
	https://aomedia.googlesource.com/aom \
	refs/tags/v3.14.0 \
	047d8cf6168feafe1300eb6902000dd1a03d5549 \
	av1 aom aom_dsp aom_ports aom_util test examples /CHANGELOG /README.md /LICENSE /PATENTS

sync_repo \
	webrtc \
	https://webrtc.googlesource.com/src \
	refs/branch-heads/7848 \
	7974ac00e6e0046950002bda6a38eb515dbe48a5 \
	modules/rtp_rtcp/source modules/video_coding/codecs/av1 api/video_codecs
