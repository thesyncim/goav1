#!/bin/sh
# Builds the wasm encoder and decoder variants into web/ with TinyGo (>= 0.43
# for Go 1.27), plus stock Go js/wasm builds for comparison. TinyGo 0.43.0
# needs a one-line compiler fix to build goav1 for wasm32 (see README.md);
# ./setup-tinygo.sh builds one into .tinygo/, which is used by default.
#
# Environment:
#   TINYGO  tinygo binary (default .tinygo/bin/tinygo, else tinygo on PATH)
#   WEB     output directory (default web). When it is not web, the pages and
#           scripts are copied there too, so `go run ./serve -dir $WEB` works.
#   ONLY    space-separated build names to build (default: all), for example
#           ONLY="encoder-simd decoder-simd"
#
# Stock Go needs GOTOOLCHAIN=go1.27.0 (or newer) when the go on PATH is older.
set -eu
cd "$(dirname "$0")"
if [ -z "${TINYGO:-}" ] && [ -x .tinygo/bin/tinygo ]; then TINYGO=$(pwd)/.tinygo/bin/tinygo; fi
TINYGO=${TINYGO:-tinygo}
WEB=${WEB:-web}
ONLY=${ONLY:-}
export GOTOOLCHAIN=${GOTOOLCHAIN:-go1.27.0}
mkdir -p "$WEB"
rm -f "$WEB"/wasm_exec.js "$WEB"/wasm_exec_go.js  # the Go copy is read-only
if [ "$WEB" != web ]; then cp web/*.html web/app.js web/worker.js web/decode.js web/decode-worker.js "$WEB"/; fi
ROOT=$("$TINYGO" env TINYGOROOT)
cp "$ROOT/targets/wasm_exec.js" "$WEB/wasm_exec.js"
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" "$WEB/wasm_exec_go.js"
want() { [ -z "$ONLY" ] && return 0; case " $ONLY " in *" $1 "*) return 0 ;; esac; return 1; }
flags="-target=wasm -opt=2 -no-debug -panic=trap"
tinygo() { pkg=$1 name=$2; shift 2; want "$name" || return 0; echo "building $WEB/$name.wasm"; "$TINYGO" build $flags -o "$WEB/$name.wasm" "$@" "./$pkg"; }
stock() { pkg=$1 name=$2; want "$name" || return 0; echo "building $WEB/$name.wasm (stock Go)"; GOOS=js GOARCH=wasm go build -trimpath -ldflags=-s -o "$WEB/$name.wasm" "./$pkg"; }
for pkg in encoder decoder; do
  tinygo $pkg $pkg-simd -llvm-features=+simd128
  tinygo $pkg $pkg-simd-nobounds -llvm-features=+simd128 -nobounds
  tinygo $pkg $pkg-scalar
  stock $pkg $pkg-go
done
ls -l "$WEB"/*.wasm
