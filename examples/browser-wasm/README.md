# browser-wasm

goav1 running entirely in the browser, as two pages.

**Encoder** (`index.html`):

```
camera ─► MediaStreamTrackProcessor ─► Web Worker: goav1 encoder (TinyGo → wasm + SIMD128)
       ─► AV1 temporal units ─► WebCodecs VideoDecoder (browser's own AV1) ─► <canvas>
```

Decoding with the browser's independent AV1 decoder also checks that the
bitstream is valid.

**Decoder** (`decode.html`):

```
IVF (corpus clip or your file) ─► Web Worker: goav1 decoder (wasm) ─► I420 VideoFrame ─► <canvas>
```

WebCodecs is not used for decoding here. The page can also benchmark every
decoder build and checks each output MD5 against the corpus `.md5` files.

There is no server-side media path. The Go server only serves static files,
the repository's `testdata/benchcorpus` under `/corpus/` and
`testdata/realcorpus` under `/realcorpus/`.

Both pages offer the same set of builds: TinyGo with SIMD128, SIMD128 without
bounds checks, scalar, and stock Go (`GOOS=js GOARCH=wasm`).

## Run (macOS arm64)

**Requires** goav1 with the fix from branch `fix/wasm-coeff-cdf-compare`
(`tile.CoeffCDFs.Equal`). Without it the encoder build makes `wasm-opt` run for
over ten minutes and past 26 GB of RAM without finishing; see
[Toolchain notes](#toolchain-notes).

```sh
brew install llvm@22        # once; TinyGo is built against it
./setup-tinygo.sh           # once: patched TinyGo 0.43.0 + Binaryen into .tinygo/
./build.sh                  # web/{encoder,decoder}-{simd,simd-nobounds,scalar,go}.wasm
go run ./serve              # http://localhost:8090
```

Stock Go needs 1.27 (`GOTOOLCHAIN=go1.27.0` if the `go` on your PATH is older;
`build.sh` sets it by default). Building all eight wasm files takes about 6
minutes; `ONLY="encoder-simd decoder-simd" ./build.sh` builds just those.

### LLVM 23

```sh
brew install llvm lld       # Homebrew's unversioned formulae are LLVM 23.1
LLVM_TAG=llvm23 OUT=$PWD/.tinygo23 ./setup-tinygo.sh
TINYGO=$PWD/.tinygo23/bin/tinygo WEB=$PWD/web-llvm23 ./build.sh
go run ./serve -dir web-llvm23
```

This builds the same TinyGo 0.43.0 with [`tinygo-llvm23.patch`](tinygo-llvm23.patch)
against LLVM 23 and drives Homebrew's LLVM 23 `clang` and `lld` (see
[Toolchain notes](#toolchain-notes) for why the release tarball's linker cannot
be used). It works for the wasm builds and for native builds
(`tinygo build -tags purego -opt=2 ./cmd/aom-go-dec`). Its wasm output passes
the same checks as LLVM 22 and is within 3% of its speed; see
[Measured](#measured-apple-m4-max-chrome-155-headless-2026-10-23) for the numbers.

### Encoder page

Pick **Camera** or **Synthetic pattern**, then press Start. The source label
shows the pixel format the browser delivers:
- **NV12 and I420** are copied straight into wasm memory.
- **RGBA/BGRA**, as delivered by canvas sources, are converted in Go.
- **Anything else** is converted to RGBA by the browser first.

The **Benchmark** button encodes 60 frames of a hard synthetic 640×360
pattern inside the worker. The pattern is `pattern.Fill`, shared with the
native `verify` tool, so the same frames can be encoded natively.

Hash modes for headless runs (the result lands in the tab title):
- `http://localhost:8090/#bench`: best of 3 runs of 60 frames for every build.
- `#verify`: encodes 60 frames of the pattern with every build (640×360, CBR
  1 Mbps, effort 0, one thread), uploads each stream to the server's `/dump/`
  endpoint and writes its SHA-256 into the title.
- `#demo`: runs the whole page pipeline (synthetic canvas → worker → wasm
  encoder → WebCodecs decoder → canvas) for 6 s per build and reports
  `encoded/decoded/decoder errors/mean encode ms/canvas spread`.

### Decoder page

Pick a corpus clip or your own `.ivf`, then a build:
- **TinyGo builds:** SIMD128, SIMD128 without bounds checks, or scalar.
- **Stock Go:** `GOOS=js GOARCH=wasm`.

**Play** decodes and paces the frames to the canvas. 10-bit streams are shown
reduced to 8 bits.

**Benchmark** decodes the clip three times and reports the best pass. A
fourth, untimed pass hashes the output for the MD5 check.

Hash modes:
- `decode.html#bench`: every build on 720p inter q20, 360p inter q32 and 360p
  intra q32.
- `decode.html#verify`: every build on every clip in `testdata/benchcorpus`
  (or `?clips=a,b,real/c` to choose, where `real/` names a stream in
  `testdata/realcorpus/streams`), comparing the output MD5 with the `.md5`
  file. The title reads `VERIFY <runs> <mismatches> <failing runs>`.

The decoder uses `syscall/js` rather than `//go:wasmexport`, so the same
source builds with both TinyGo and stock Go. Each toolchain needs its own
`wasm_exec.js`, and the worker picks the right one. The encoder's
`//go:wasmexport` functions also work under stock Go's js/wasm port (its
memory export is named `mem`, TinyGo's `memory`; the worker handles both).

## Verifying correctness

Decoder: `decode.html#verify` as above. With LLVM 22 and with LLVM 23, all four
builds match the `.md5` of all 18 benchcorpus clips (72 runs) and of eight
realcorpus streams (720p and 1080p, film grain, two tiles; 32 runs).

Encoder: the wasm stream must equal the native encoder's, which dav1d must
decode to the encoder's own reconstruction.

```sh
go run ./serve -dump /tmp/dump &        # then open http://localhost:8090/#verify
go run ./verify -o native.ivf -recon native.yuv
cmp native.ivf /tmp/dump/encoder-simd.ivf            # and -simd-nobounds, -scalar, -go
dav1d -q -i /tmp/dump/encoder-simd.ivf -o wasm.yuv && cmp wasm.yuv native.yuv
```

All four builds, with LLVM 22 and with LLVM 23, produce a stream identical to
the native one (247,334 bytes, SHA-256 `5cef0202…b521ba`), and dav1d 1.5.4
decodes it bit-exactly to the encoder's reconstruction. The native encoder
builds (stock Go with and without SIMD, TinyGo with LLVM 22 and 23) agree too.

## Toolchain notes

- **TinyGo ≥ 0.43.0** is required. It is the first release that supports
  Go 1.27.
- **TinyGo 0.43.0 cannot build goav1 for wasm32 without a fix.** Its
  `make([]T, n)` check compares the element *size in bytes* against the
  maximum element *count*. On a 32-bit target, any element type over 64 KB is
  rejected with `slice element type is too big`.
  [`tinygo-wasm32-large-slice-elem.patch`](tinygo-wasm32-large-slice-elem.patch)
  is the one-line fix, and `setup-tinygo.sh` applies it. Upstream issue:
  [tinygo-org/tinygo#5972](https://github.com/tinygo-org/tinygo/issues/5972).
- **LLVM 23 needs LLVM 23 tools.** TinyGo hands its final module to `wasm-ld`
  (or `ld.lld` natively) as ThinLTO bitcode, and the linker runs the last
  optimisation passes and the code generator. The release tarball's linker is
  LLVM 22.1.4 and rejects bitcode written by LLVM 23 (`wasm-ld: error: Invalid
  summary version 14`), so `setup-tinygo.sh` points the compiler at Homebrew's
  `clang` and `lld` when `LLVM_TAG=llvm23` (`TOOLS=release` forces the release
  tools). The same `clang` also compiles TinyGo's own C sources: `compiler-rt` and
  `wasi-libc` for wasm, plus `bdwgc` in native builds.
- **TinyGo 0.43.0 has no LLVM 23 support.** [`tinygo-llvm23.patch`](tinygo-llvm23.patch)
  makes it build and work: a newer `go-llvm` (LLVM 23 bindings), the cgo header
  config, `interp` reading branches through successors (LLVM 23 split `br`
  into `condbr`/`uncondbr`), and the unmangled `llvm.returnaddress` import
  becoming `llvm.returnaddress.p0` (native builds only). The cgo config and the branch helpers come from TinyGo's `dev`
  branch, which has LLVM 23 support but not yet a release.
- **Do not compare large aggregates with `==` in code that TinyGo builds.**
  TinyGo expands `==` on a struct or array into one comparison per element.
  `st.trialCDFs == *def` in the encoder (a 100 KB struct of arrays) produced a
  626 KB wasm function, and `wasm-opt` then ran for over ten minutes and past
  26 GB without finishing. `tile.CoeffCDFs.Equal` replaces it. If a wasm build
  stalls in `wasm-opt`, look for the largest function with
  `wasm-opt --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int
  --enable-sign-ext --enable-mutable-globals --func-metrics X.wasm -o /dev/null`.
- **TinyGo's built-in `clang` needs `-resource-dir`.** The shims written by
  `setup-tinygo.sh` pass it. TinyGo caches the C libraries it compiles
  (`compiler-rt`) under a key that includes the `TINYGOROOT` path, so a build
  from a new location fails with `'float.h' file not found` without it.
- **The encoder's exports use `//go:wasmexport` with `main` blocked on
  `select {}`.** Legacy `//export` calls made after `main` returns run on a
  stack the asyncify scheduler no longer manages, and the encoder fails with
  a bogus index panic. `//go:wasmexport` gives each call its own goroutine,
  but it needs `main` to still be running.
- **`-llvm-features=+simd128`** lets LLVM auto-vectorise into wasm SIMD.
  Stock Go's wasm port cannot emit SIMD.
- **`-nobounds` is offered as a variant, not the default.** Inside wasm linear
  memory, an out-of-bounds write corrupts the heap silently instead of
  trapping.
- **Encoder content mode.** `VideoEncoderConfig.Content` defaults to screen
  content detection on every keyframe. The page pins `EncoderContentCamera`,
  which is what it did before the field existed. The benchmark pattern is not
  detected as screen content, so the choice does not change its stream.
- **Chrome ignores AV1 `render_size`.** Frames whose size is not a multiple of
  8 decode at the padded coded size, so the encoder page crops the result.
- **wazero 1.11 cannot run the TinyGo decoder builds.** Its arm64 compiler
  panics (`BUG` in `resolveAddressingMode`) while compiling the module. It was
  not retried with newer wazero releases.
- **Benchmark in a tab without DevTools or browser automation attached.** With
  a debugger attached, the same page measured 4–6× slower: 132 instead of 25
  ms/frame for 720p decode. The recipe:

  ```sh
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
      --user-data-dir=$(mktemp -d) --remote-debugging-port=9333 \
      http://localhost:8090/decode.html#bench &
  curl -s localhost:9333/json | grep '"title"'     # poll until it starts with BENCH
  ```

  Kill only that Chrome afterwards.

## Measured (Apple M4 Max, Chrome 155 headless, 2026-10-23)

goav1 at `origin/main` 35e4f943 plus the `fix/wasm-coeff-cdf-compare` change,
Go 1.27.0, TinyGo 0.43.0 (patched), Binaryen 133, LLVM 22.1.8 and 23.1.4. All
runs use one thread, and every build's output is byte-identical to native (see
above). "Old" is the 2026-10-20 run of this example, before about 50 PRs of
decoder and encoder work, so the old/new difference is mostly goav1 itself.
(That run's TinyGo SIMD128 decode of 720p was 25.0 ms/frame; the later
`perf/purego-scalar` work reached 12.0 the same day.)

**Decode in Chrome**, ms/frame, best of 3 passes over 48 frames, minimum of
two page loads. Old → LLVM 22 → LLVM 23.

| build                           | 720p inter q20         | 360p inter q32       | 360p intra q32      |
|---------------------------------|------------------------|----------------------|---------------------|
| stock Go wasm (any LLVM)        | 43.4 → 33.3            | 14.0 → 10.4          | 13.1 → 9.4          |
| TinyGo, scalar                  | 28.7 → 15.0 → 14.5     | 9.9 → 4.6 → 4.6      | 5.8 → 4.1 → 4.0     |
| TinyGo, SIMD128                 | 25.0 → 11.5 → 11.3     | 9.1 → 3.8 → 3.8      | 5.7 → 3.8 → 3.9     |
| TinyGo, SIMD128 + nobounds      | 23.3 → 9.7 → 9.3       | 8.2 → 3.2 → 3.3      | 5.1 → 3.7 → 3.6     |

TinyGo SIMD128 decodes 720p at about 87 fps on one thread, 2.9× faster than
stock Go wasm. The page-load-to-page-load spread is ±0.3 ms.

**Encode in Chrome**, 640×360 hard pattern, CBR 1 Mbps, one thread, ms/frame,
best of 3 runs of 60 frames, minimum of two page loads. Old → LLVM 22 → LLVM 23.

| build                           | ms/frame               |
|---------------------------------|------------------------|
| stock Go wasm (js/wasm)         | not built → 28.7       |
| TinyGo, scalar                  | 36.9 → 7.6 → 7.9       |
| TinyGo, SIMD128                 | 31.9 → 7.2 → 7.3       |
| TinyGo, SIMD128 + nobounds      | 29.4 → 6.3 → 6.6       |

The live page (synthetic source, effort −2, 640×360 at 30 fps) encodes at
about 5.5 ms/frame mean with no drops; the decoded canvas matches the source
and WebCodecs reports no decoder errors (`#demo`).

**Native decode CPU**, `aom-go-dec -workers 1`, user+sys seconds, min of 5
(the `decoded N frames in X ms` loop time in parentheses, ms/frame).

| clip           | Go + SIMD + PGO | Go, no SIMD | TinyGo, LLVM 22 | TinyGo, LLVM 23 | old Go → old TinyGo |
|----------------|-----------------|-------------|-----------------|-----------------|---------------------|
| 720p inter q20 | 0.14 (2.94)     | 0.60 (12.56)| 0.42 (8.78)     | 0.39 (8.28)     | 0.86 → 0.63         |
| 720p inter q32 | 0.11 (2.34)     | 0.53 (11.21)| 0.34 (7.18)     | 0.32 (6.81)     | 0.74 → 0.50         |
| 360p inter q32 | 0.05 (1.06)     | 0.17 (3.62) | 0.12 (2.57)     | 0.11 (2.43)     | n/a                 |
| 360p intra q32 | 0.05 (1.01)     | 0.16 (3.48) | 0.12 (2.57)     | 0.12 (2.55)     | 0.27 → 0.19         |

The LLVM 23 columns are the full Homebrew LLVM 23 toolchain (compiler, `lld`,
`clang`); the LLVM 22 columns are TinyGo's release tools (22.1.4) with a Homebrew
LLVM 22.1.8 compiler. "Go + SIMD + PGO" is `GOEXPERIMENT=simd go build -pgo=cmd/qualitybench/default.pgo`;
"Go, no SIMD" is a plain `go build` (identical to `-tags purego`); TinyGo is
`tinygo build -tags purego -opt=2`, because it has no `simd` experiment. The
time resolution is 10 ms. The old ms/frame figures for 720p q20 / 360p inter /
360p intra were 17.1 / 4.7 / 5.2 (Go) and 12.8 / 4.0 / 3.6 (TinyGo).

**Native realtime encode**, one thread (`GOMAXPROCS=1`, `-threads 1`), min of 5.

| encoder                               | 720p 1 Mbps (`encbench`, 120 frames)  | 360p probe (`verify`, 60 frames) |
|---------------------------------------|---------------------------------------|----------------------------------|
| Go + SIMD + PGO                       | 0.53 s CPU, 4.12 ms/frame             | 1.79 ms/frame                    |
| Go, no SIMD (old: 19.5 ms/frame)      | 6.50 s CPU, 53.45 ms/frame            | 14.85 ms/frame                   |
| TinyGo, LLVM 22                       | 1.75 s CPU, 14.19 ms/frame            | 4.93 ms/frame                    |
| TinyGo, LLVM 23                       | 1.18 s CPU, 9.38 ms/frame             | 3.64 ms/frame                    |

Native TinyGo decode and encode are now slower than stock Go with its SIMD
kernels (2.8× and 2.3× at 720p) but 1.4-1.5× (decode) and 3.8-5.7× (encode)
faster than stock Go without them.

**Where LLVM 23 helps.** Native TinyGo encode is 34% faster with the LLVM 23
toolchain (14.1 to 9.3 ms/frame at 720p, same bitstream), and a four-way
control puts all of it in the linker, which runs TinyGo's ThinLTO pipeline and
the AArch64 code generator: LLVM 22 compiler + Homebrew lld 23 gives 9.40, LLVM
23 compiler + the release's LLVM 22 clang + lld 23 gives 9.58, and LLVM 22
compiler + upstream LLVM 22.1.8 clang and lld gives 14.52 (TinyGo's own
LLVM 22.1.4 linker: 14.08). Wasm builds see no gain (SIMD128 7.2 vs 7.3 ms
encode, 11.5 vs 11.3 decode), and native decode gains 5-6% at 720p.

**Builds**: the encoder takes about 80 s and 3 GB to build per variant, the
decoder about 60 s. Sizes (SIMD128): encoder 2.09 MB, decoder 4.74 MB with
either LLVM.
