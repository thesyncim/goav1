#!/bin/sh
# Installs TinyGo 0.44.0 and Binaryen for building goav1 to wasm (macOS arm64).
# Afterwards `./build.sh` uses it automatically.
#
#   ./setup-tinygo.sh                    # release TinyGo (LLVM 22.1.4)
#   LLVM_TAG=llvm23 ./setup-tinygo.sh    # TinyGo built against LLVM 23
#                                        # (brew install llvm lld)
#
# Environment:
#   OUT       install directory (default ./.tinygo)
#   LLVM_TAG  release (default) or llvm23
#   LLVM      Homebrew LLVM 23 prefix for llvm23 (default /opt/homebrew/opt/llvm)
#   LLD       Homebrew lld prefix for llvm23 (default /opt/homebrew/opt/lld)
#
# Why the steps are needed:
#   - The release tarball has no wasm-opt; Binaryen provides it.
#   - The release is built against LLVM 22. For LLVM 23 the compiler is built
#     from the v0.44.0 source with -tags llvm23, and it drives Homebrew's
#     LLVM 23 clang and lld: the release's LLVM 22 linker cannot read the
#     ThinLTO bitcode an LLVM 23 compiler writes. The release tree still
#     serves as TINYGOROOT (runtime sources and prebuilt libraries).
set -eu
cd "$(dirname "$0")"
EX=$(pwd)
OUT=${OUT:-$EX/.tinygo}
TINYGO_VERSION=0.44.0
BINARYEN_VERSION=133
LLVM_TAG=${LLVM_TAG:-release}
case $LLVM_TAG in release|llvm23) ;; *) echo "LLVM_TAG must be release or llvm23"; exit 1 ;; esac
mkdir -p "$OUT/bin"

TG=$OUT/release/tinygo
if [ "$("$TG/bin/tinygo" version 2>/dev/null | awk '{print $3}')" != "$TINYGO_VERSION" ]; then
  echo "downloading TinyGo $TINYGO_VERSION release"
  rm -rf "$OUT/release" && mkdir -p "$OUT/release"
  curl -fsSL "https://github.com/tinygo-org/tinygo/releases/download/v$TINYGO_VERSION/tinygo$TINYGO_VERSION.darwin-arm64.tar.gz" |
    tar -xz -C "$OUT/release"
fi

if [ ! -x "$OUT/binaryen/bin/wasm-opt" ]; then
  echo "downloading Binaryen version_$BINARYEN_VERSION"
  mkdir -p "$OUT/binaryen"
  curl -fsSL "https://github.com/WebAssembly/binaryen/releases/download/version_$BINARYEN_VERSION/binaryen-version_$BINARYEN_VERSION-arm64-macos.tar.gz" |
    tar -xz -C "$OUT/binaryen" --strip-components 1
fi

COMPILER=$TG/bin/tinygo
PATHS=$OUT/binaryen/bin
if [ "$LLVM_TAG" = llvm23 ]; then
  LLVM=${LLVM:-/opt/homebrew/opt/llvm}
  LLD=${LLD:-/opt/homebrew/opt/lld}
  [ -x "$LLVM/bin/llvm-config" ] || { echo "need Homebrew LLVM 23 at $LLVM (brew install llvm)"; exit 1; }
  case $("$LLVM/bin/llvm-config" --version) in 23.*) ;; *) echo "$LLVM is not LLVM 23"; exit 1 ;; esac
  [ -x "$LLD/bin/wasm-ld" ] || { echo "need Homebrew lld at $LLD (brew install lld)"; exit 1; }
  if [ "$(git -C "$OUT/src" describe --tags 2>/dev/null)" != "v$TINYGO_VERSION" ]; then
    rm -rf "$OUT/src"
    git clone -q --depth 1 --branch "v$TINYGO_VERSION" https://github.com/tinygo-org/tinygo "$OUT/src"
  fi
  echo "building tinygo against $LLVM"
  (cd "$OUT/src" && GOTOOLCHAIN=go1.27.0 go build -tags llvm23 -o "$OUT/bin/tinygo-compiler" .)
  COMPILER=$OUT/bin/tinygo-compiler
  # The source-built compiler looks clang-23, ld.lld and wasm-ld up on PATH.
  mkdir -p "$OUT/shims" && rm -f "$OUT"/shims/*
  printf '#!/bin/sh\nexec "%s/bin/clang" "$@"\n' "$LLVM" > "$OUT/shims/clang-23"
  for tool in ld.lld wasm-ld; do
    printf '#!/bin/sh\nexec "%s/bin/%s" "$@"\n' "$LLD" "$tool" > "$OUT/shims/$tool"
  done
  chmod +x "$OUT"/shims/*
  PATHS=$OUT/shims:$PATHS
fi

# Entry point wiring TINYGOROOT, the tools and wasm-opt together.
cat > "$OUT/bin/tinygo" <<EOS
#!/bin/sh
export PATH="$PATHS:\$PATH"
export TINYGOROOT="$TG"
export WASMOPT="$OUT/binaryen/bin/wasm-opt"
exec "$COMPILER" "\$@"
EOS
chmod +x "$OUT/bin/tinygo"
"$OUT/bin/tinygo" version
