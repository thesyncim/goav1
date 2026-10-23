#!/bin/sh
# Builds a patched TinyGo 0.43.0 that can compile goav1 for wasm32 (macOS
# arm64 + Homebrew LLVM). Afterwards `./build.sh` uses it automatically.
#
#   ./setup-tinygo.sh                    # LLVM 22 (brew install llvm@22)
#   LLVM_TAG=llvm23 ./setup-tinygo.sh    # LLVM 23 (brew install llvm lld)
#
# Environment:
#   OUT       install directory (default ./.tinygo)
#   LLVM_TAG  llvm22 (default) or llvm23
#   LLVM      Homebrew LLVM prefix (default: llvm@22, or the unversioned
#             llvm formula for llvm23)
#   TOOLS     release or brew: which clang/lld the compiler drives (default:
#             release for llvm22; brew for llvm23 when Homebrew lld is installed)
#   LLD       Homebrew lld prefix for TOOLS=brew (default /opt/homebrew/opt/lld)
#
# Why the steps are needed:
#   - TinyGo 0.43.0 rejects make([]T, n) when sizeof(T) > 64 KB on 32-bit
#     targets (tinygo-wasm32-large-slice-elem.patch fixes the check).
#   - Homebrew llvm@22 ships clang but no ld.lld/wasm-ld, so for LLVM 22 the
#     patched compiler shells out to the release binary's built-in clang and
#     linkers (LLVM 22.1.4).
#   - The release tarball has no wasm-opt; Binaryen provides it.
#   - TinyGo 0.43.0 has no LLVM 23 support. tinygo-llvm23.patch bumps go-llvm
#     to a revision with LLVM 23 bindings, adds the cgo header config, reads
#     branch instructions through successors in interp (LLVM 23 split br into
#     condbr/uncondbr) and mangles the llvm.returnaddress import. It is applied
#     only for LLVM_TAG=llvm23. The release's LLVM 22 linker cannot read the
#     ThinLTO bitcode an LLVM 23 compiler writes, hence TOOLS=brew there.
set -eu
cd "$(dirname "$0")"
EX=$(pwd)
OUT=${OUT:-$EX/.tinygo}
TINYGO_VERSION=0.43.0
BINARYEN_VERSION=133
LLVM_TAG=${LLVM_TAG:-llvm22}
case $LLVM_TAG in
  llvm22) LLVM=${LLVM:-/opt/homebrew/opt/llvm@22} ;;
  llvm23) LLVM=${LLVM:-/opt/homebrew/opt/llvm} ;;
  *) echo "LLVM_TAG must be llvm22 or llvm23"; exit 1 ;;
esac

[ -x "$LLVM/bin/llvm-config" ] || { echo "need Homebrew LLVM at $LLVM (brew install llvm@22, or llvm for 23)"; exit 1; }
case $("$LLVM/bin/llvm-config" --version) in
  22.*) [ "$LLVM_TAG" = llvm22 ] || { echo "$LLVM is LLVM 22 but LLVM_TAG=$LLVM_TAG"; exit 1; } ;;
  23.*) [ "$LLVM_TAG" = llvm23 ] || { echo "$LLVM is LLVM 23 but LLVM_TAG=$LLVM_TAG"; exit 1; } ;;
esac
mkdir -p "$OUT"

if [ ! -x "$OUT/release/tinygo/bin/tinygo" ]; then
  echo "downloading TinyGo $TINYGO_VERSION release"
  mkdir -p "$OUT/release"
  curl -fsSL "https://github.com/tinygo-org/tinygo/releases/download/v$TINYGO_VERSION/tinygo$TINYGO_VERSION.darwin-arm64.tar.gz" |
    tar -xz -C "$OUT/release"
fi

if [ ! -x "$OUT/binaryen/bin/wasm-opt" ]; then
  echo "downloading Binaryen version_$BINARYEN_VERSION"
  mkdir -p "$OUT/binaryen"
  curl -fsSL "https://github.com/WebAssembly/binaryen/releases/download/version_$BINARYEN_VERSION/binaryen-version_$BINARYEN_VERSION-arm64-macos.tar.gz" |
    tar -xz -C "$OUT/binaryen" --strip-components 1
fi

if [ ! -d "$OUT/src" ]; then
  git clone -q --depth 1 --branch "v$TINYGO_VERSION" https://github.com/tinygo-org/tinygo "$OUT/src"
  git -C "$OUT/src" apply "$EX/tinygo-wasm32-large-slice-elem.patch"
  if [ "$LLVM_TAG" = llvm23 ]; then
    git -C "$OUT/src" apply "$EX/tinygo-llvm23.patch"
  fi
fi
echo "building patched tinygo against $LLVM ($LLVM_TAG)"
(cd "$OUT/src" && GOTOOLCHAIN=go1.27.0 go build -tags "$LLVM_TAG" -o "$OUT/bin/tinygo-compiler" .)

# Shims: the patched compiler looks clang-NN, ld.lld and wasm-ld up on PATH.
# TinyGo hands its final module to the linker as ThinLTO bitcode and the linker
# runs the last optimisation passes and the code generator, so the LLVM that
# the *linker* comes from decides the generated code as much as the compiler
# does. TOOLS picks which one:
#   release  the tools inside the TinyGo release tarball (LLVM 22.1.4); the
#            default for llvm22
#   brew     Homebrew's clang and lld (`brew install llvm lld`), the same LLVM
#            as the compiler; the default for llvm23 when lld is installed
# The release's clang only finds its headers (float.h, ...) with an explicit
# resource dir, which the compiler does not pass when it uses external tools.
# TinyGo also caches the C libraries it compiles (compiler-rt) under a key that
# includes the TINYGOROOT path, so without the flag a build from a new
# location fails with "'float.h' file not found".
LLD=${LLD:-/opt/homebrew/opt/lld}
if [ -z "${TOOLS:-}" ]; then
  TOOLS=release
  [ "$LLVM_TAG" = llvm23 ] && [ -x "$LLD/bin/wasm-ld" ] && TOOLS=brew
fi
mkdir -p "$OUT/shims"
rm -f "$OUT"/shims/*
TG=$OUT/release/tinygo
case $TOOLS in
  release)
    printf '#!/bin/sh\nexec "%s" clang -resource-dir="%s" "$@"\n' "$TG/bin/tinygo" "$TG/lib/clang" > "$OUT/shims/clang-${LLVM_TAG#llvm}"
    for tool in ld.lld wasm-ld; do
      printf '#!/bin/sh\nexec "%s" %s "$@"\n' "$TG/bin/tinygo" "$tool" > "$OUT/shims/$tool"
    done ;;
  brew)
    [ -x "$LLD/bin/wasm-ld" ] || { echo "TOOLS=brew needs Homebrew lld at $LLD (brew install lld)"; exit 1; }
    printf '#!/bin/sh\nexec "%s/bin/clang" "$@"\n' "$LLVM" > "$OUT/shims/clang-${LLVM_TAG#llvm}"
    for tool in ld.lld wasm-ld; do
      printf '#!/bin/sh\nexec "%s/bin/%s" "$@"\n' "$LLD" "$tool" > "$OUT/shims/$tool"
    done ;;
  *) echo "TOOLS must be release or brew"; exit 1 ;;
esac
chmod +x "$OUT"/shims/*
echo "linker and C compiler: $TOOLS"

# Entry point wiring TINYGOROOT, the shims and wasm-opt together.
cat > "$OUT/bin/tinygo" <<EOS
#!/bin/sh
export PATH="$OUT/shims:$OUT/binaryen/bin:\$PATH"
export TINYGOROOT="$OUT/release/tinygo"
export WASMOPT="$OUT/binaryen/bin/wasm-opt"
exec "$OUT/bin/tinygo-compiler" "\$@"
EOS
chmod +x "$OUT/bin/tinygo"
"$OUT/bin/tinygo" version
