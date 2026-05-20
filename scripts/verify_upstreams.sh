#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream="$root/third_party/upstream"

verify_repo() {
	name=$1
	want=$2
	dir="$upstream/$name"

	if [ ! -d "$dir/.git" ]; then
		echo "$name: missing clone at $dir" >&2
		exit 1
	fi

	got=$(git -C "$dir" rev-parse HEAD)
	if [ "$got" != "$want" ]; then
		echo "$name: got $got want $want" >&2
		exit 1
	fi

	echo "$name $got"
}

verify_repo dav1d b546257f770768b2c88258c533da38b91a06f737
verify_repo libaom 047d8cf6168feafe1300eb6902000dd1a03d5549
verify_repo webrtc 7974ac00e6e0046950002bda6a38eb515dbe48a5
