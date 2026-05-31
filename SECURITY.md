# Security Policy

`goav1` is a pure-Go AV1 decoder intended for realtime/WebRTC and
batch decoding workloads. This document describes the security posture
of the project, the threat model it targets, the mitigations already in
place, and how to report a vulnerability.

It is meant for production integrators who must reason about the blast
radius of an untrusted bitstream reaching the decoder.

## Reporting a vulnerability

If you believe you have found a security issue in `goav1`, please do
**not** open a public GitHub issue. Instead:

1. Email the maintainer at `thesyncim@gmail.com` with the subject line
   `goav1 security report` and a description of the issue, ideally
   including a minimal reproducer (an `.ivf`, `.obu`, or captured RTP
   payload), the goav1 commit/tag, and the Go toolchain version.
2. If you prefer, use GitHub's [private vulnerability reporting] on
   `https://github.com/thesyncim/goav1` (Security tab > "Report a
   vulnerability") so the disclosure stays inside GitHub.

[private vulnerability reporting]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability

Please give the maintainer a reasonable disclosure window before going
public. As a single-maintainer open-source project, target response
times are best-effort:

| Stage                  | Target                       |
| ---------------------- | ---------------------------- |
| Acknowledgement        | within 5 business days       |
| Initial triage         | within 10 business days      |
| Fix or mitigation plan | within 30 days of triage     |
| Coordinated disclosure | once a fix is available, or 90 days from triage, whichever is sooner |

If a report is declined (out-of-scope, not reproducible, expected
behavior, etc.) the maintainer will explain why and, where appropriate,
suggest a non-security route to land the change.

## Supported versions

`goav1` is pre-1.0 and follows a rolling-`main` release model. Only the
latest commit on `main` and the most recent tagged release receive
security fixes. Older tags will not be back-patched. Integrators are
expected to pin a recent commit and re-pin during routine dependency
updates.

| Version          | Security fixes |
| ---------------- | -------------- |
| `main` (HEAD)    | Yes            |
| Latest tag       | Yes            |
| Older tags       | No             |

## Threat model

### Trust boundary

The fundamental trust boundary is **untrusted compressed AV1 bytes
entering the decoder from a remote or otherwise attacker-controlled
source** (WebRTC peer, RTP relay, IVF/OBU file from disk, network
fetch, etc.). Everything downstream of `DecoderStream` and the public
transport parsers must assume that input bytes are adversarial.

Out of scope:

- Trusted-side bugs (the AV1 encoder, which is not yet implemented).
- The host Go runtime, OS kernel, or hardware vulnerabilities.
- Calling-program misuse of caller-owned scratch buffers in a way the
  documented contract forbids (e.g. mutating a buffer while the
  decoder still holds a reference to it).

### Attack surface

The reachable attack surface from an attacker who controls the input
bytes:

1. **Transport parsers.** IVF (`NewIVFIterator`), Section 5
   temporal-unit OBU streams (`NewLowOverheadIterator`,
   `NewTemporalUnitIterator`), Annex B (`NewAnnexBIterator`), and the
   AV1 RTP aggregation header / payload iterator / frame assembler
   (`NewRTPPayloadIterator`, `AssembleRTPFrame`). These walk
   length-prefixed and LEB128-framed byte slices and must never run off
   the end of a caller-owned buffer or accept lengths that exceed the
   payload.
2. **OBU header parser.** Forbidden-bit and reserved-bit checks,
   size-field presence checks (low-overhead vs. Annex B), temporal
   delimiter ordering. Errors are returned, not panicked — see
   `internal/av1/obu/errors.go`.
3. **Sequence / frame header parser.** Profile, bit depth, chroma
   subsampling, frame size, render size, tile layout, segmentation,
   loop-filter, CDEF, loop-restoration, transform-mode, reference-mode,
   skip-mode, warp/global-motion, and film-grain parameters. Every
   syntax element has a documented numeric domain; out-of-domain values
   produce an `ErrInvalid*` error instead of a panic.
4. **Tile-group parser and tile decode.** Tile sizing, tile-payload
   span splitting, entropy reader setup, coefficient decode. Tile-row
   and tile-column counts are hard-capped at `MaxTileRows = 64` and
   `MaxTileCols = 64` (see `internal/av1/parser/tile.go`).
5. **Residual + reconstruct path.** Dequantization, inverse transform
   dispatch (DCT/ADST/IDTX/WHT and the inverse-qmatrix path), residual
   add into caller-owned planes, intra/inter prediction (directional,
   filter-intra, CfL, OBMC, compound, warp), and the postfilter
   pipeline (loop filter, CDEF, super-res, loop restoration, film
   grain).
6. **Caller-owned-buffer sizing helpers.** The decoder exposes
   public preflight sizing helpers (`*Size`, `*ScratchSize`,
   `Required*` accessors) so callers can compute exact buffer sizes
   before binding them. A malformed header that asks for a buffer
   larger than the caller provides is detected and returned as an
   error before any out-of-bounds write.

### Threats

#### T1. Malformed AV1 bitstream (primary)

An attacker submits bytes that pass `obu` framing but violate AV1
syntax constraints inside a sequence header, frame header, tile group,
or coefficient stream — for example, advertising a tile count or
quantizer level outside the legal range, an unsupported transform
class, or a film-grain parameter table that contradicts the sequence
header.

*Mitigation.* Each parser validates syntax-element domains and returns
`ErrInvalid*` rather than panicking. Recent hardening for
`filmgrain` / `quantize` (commit `744ef85`) and the restoration
bordered-block path (commit `7c893f5`) explicitly converted previously
panicking branches into typed errors.

#### T2. Truncated frames, oversize headers, corrupt OBUs

An attacker truncates an OBU mid-LEB128, claims a size larger than the
remaining buffer, or stuffs the temporal-unit / Annex B framing with
nested oversize lengths.

*Mitigation.* All readers operate on caller-owned byte slices with
bounds-checked indexing. LEB128 reads (`internal/av1/bitstream`) cap
at the AV1-defined byte count and reject overlong encodings. The OBU
iterator surfaces `ErrShortHeader`, `ErrShortPayload`,
`ErrSizeMismatch`, and `ErrInvalidAnnexB` for these conditions.
The RTP depacketizer cross-checks fragment lengths against the
remainder of the payload before reassembly. Fuzz coverage:
`internal/av1/bitstream/fuzz_test.go::FuzzReadLEB128` and the
public-API fuzz harnesses listed below.

#### T3. Resource exhaustion (memory, CPU, file descriptors)

A small adversarial input could in principle trigger a disproportionate
amount of allocation, compute, or descriptor churn (a "decoder bomb").

*Mitigation.*

- **Memory.** The decoder is a zero-allocation-in-steady-state
  design. Buffers (`FramePool`, postfilter scratch, residual scratch,
  loop-restoration frame buffers, side-data arenas) are caller-owned
  and bound up front via public sizing helpers. The decoder cannot
  silently allocate large buffers in response to a hostile header —
  if the header asks for more than the caller has bound, the call
  returns an error.
- **Compute.** Tile counts are capped at 64x64. Per-frame work is
  bounded by the parsed (and validated) frame dimensions. The tile
  worker pool is bounded and caller-owned
  (`NewTileWorkerPool`).
- **File descriptors.** The library opens no files and no sockets. All
  I/O is owned by the caller; the public API consumes `[]byte` or
  iterator state. Goroutine fan-out is bounded by the caller-owned
  worker pool.

#### T4. Side channels (timing, cache)

`goav1` does not handle secret material. There is no key, no MAC, no
session token routed through the decoder. As such, no part of the
decoder claims to be constant-time, and timing of decode operations is
necessarily data-dependent (skip blocks, transform classes, palette
modes, and entropy-coded residuals all branch on bitstream content).
Integrators who multiplex a decoder instance across mutually
distrusting tenants should not infer secrecy properties from decode
timing.

#### T5. Concurrency hazards

Race conditions inside the decoder are treated as correctness bugs
with potential security impact (torn writes into caller planes,
double-release of frame-pool surfaces).

*Mitigation.* The threading model is documented in `ARCHITECTURE.md`.
State that crosses goroutines is bound to caller-owned scratch with
explicit lifecycle helpers (`BeginDecoderFrameWork`,
`RunDecoderFrameWorkEventWithContext`, batch run/finish helpers).
Frame-pool acquire/release is checked. CI runs `make test` (which
includes `-race` coverage for the threading and tile packages).

### Input validation boundaries

| Layer                              | Rejects on                                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| `bitstream` (LEB128, bit reader)   | Overlong / truncated LEB128, reads past end-of-buffer.                                  |
| `obu`                              | Short headers, forbidden/reserved bits, missing size field, size > buffer.              |
| `rtp`                              | Inconsistent fragment lengths, missing aggregation header, oversize OBU spans.          |
| `parser` (sequence/frame/tile)     | Out-of-domain syntax elements, unsupported profile/bitdepth combos, tile-count overflow.|
| `quantize`, `filmgrain`            | Invalid quantizer indices, malformed grain tables (commit `744ef85`).                   |
| `restoration`                      | Bordered-block layout that would read outside the unit (commit `7c893f5`).              |
| `frame` / `memory` (pool, scratch) | Surface size mismatch on acquire; capacity vs. requested-size mismatch on bind.         |
| Decoder lifecycle helpers          | Event ordering (e.g. `BeginDecoderFrameWork` before run/finish), surface double-release.|

Beyond these boundaries the decoder treats bytes as already validated
for the corresponding layer, so all bounds checks must happen at the
boundary above.

## Existing mitigations

- **Panic-to-error hardening.** Bitstream-driven panics in the
  `filmgrain`, `quantize`, and `restoration` paths were converted into
  typed errors so callers can recover gracefully from hostile input.
  See commits `744ef85` and `7c893f5`. There are currently no
  `panic()` calls in non-test code paths in the module.
- **Fuzz harnesses.** The repository ships over 100 `Fuzz*` harnesses
  across the public API and internal packages, including:
  - Public API: `FuzzPublicDecodeAndReconstructDecoderFrameWorkJobResiduals`,
    `FuzzPublicDecodeAndRetainDecoderFrameWorkBatchResiduals`,
    `FuzzPublicDecoderFrameWorkBatchResidualRunnerSideData`,
    `FuzzPublicRunDecoderFrameWorkEventWithResidualRunner`,
    `FuzzPublicDecoderFrameWorkSideDataBinding`,
    `FuzzPublicDecoderFrameWorkPostFilterScratchContext`,
    `FuzzPublicSimpleDecoderIVF`,
    `FuzzPublicDecodeTileBlockCoefficients`,
    `FuzzPublicParseTileListOBU`,
    `FuzzPublicDecodeAndReconstructDecoderFrameWorkBlockCoefficients`.
  - Inverse transforms: WHT, ADST, warp, film grain (commit
    `ff8ba63`).
  - Inverse-qmatrix dequant path (commit `455ddac`).
  - Palette mode and color-map decode (commit `133a448`).
  - `internal/av1/bitstream/fuzz_test.go::FuzzReadLEB128`.
  - Decoder stream dispatch across IVF, low-overhead OBU, Annex B,
    temporal-unit grouping, and single-OBU input (`FuzzStreamPush`).
  - Tile-list OBU parsing (`FuzzParseTileListOBU`).
  - Entropy reader and CDF state harnesses.
  - Prediction harnesses (intra, directional, filter-intra, CFL,
    static intra, intra edges, DC predictor).
  - Frame plane round-trip harnesses (sample plane, bordered sample
    plane).
  Run the short sweep with `make fuzz-smoke`.
- **Allocation regression coverage.** `make alloc` keeps the public
  hot paths zero-allocation, which limits the blast radius of
  adversarial input as a memory-pressure vector. Locked in across the
  postfilter dispatch in commit `e69a428`.
- **Bounds-checked indexing throughout.** Tile-count, tile-row, and
  tile-column counts are capped at 64. Frame and tile geometry helpers
  validate the requested block against the bound plane window before
  any write.
- **Caller-owned buffers and preflight sizing.** Every hot-path
  allocation is replaced by a caller-supplied buffer plus a public
  `*Size` / `Required*` helper. A malformed header that would require
  more memory than bound is rejected at bind time, before any write.
- **CI gates.** `make ci-local` (run in CI) executes `fmt-check`,
  `vet`, `test`, and `alloc`. `make testvectors-fast` runs the
  committed libaom conformance subset on every push.

## Dependency inventory

The decoder is **pure Go with no third-party module dependencies**.

`go.mod`:

```
module github.com/thesyncim/goav1

go 1.26
```

There is no `go.sum` file because the module has no external
`require` directives. The only imports are from the Go standard
library and `github.com/thesyncim/goav1/internal/...`. This is a
deliberate design choice — see the "Pure Go. No CGO wrappers around C
codecs." principle in `README.md` — and substantially reduces the
upstream supply-chain risk.

`third_party/upstream` holds vendored libaom reference data /
specification artifacts pinned per `UPSTREAM.md`; it is not a Go
dependency and is not compiled into binaries that use this module.

## Vulnerability scanning

The module is regularly scanned with [`govulncheck`].

[`govulncheck`]: https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck

To install and run it locally:

```sh
go install golang.org/x/vuln/cmd/govulncheck@latest
govulncheck ./...
```

Last run against `main` (Go 1.26.3, `govulncheck` v1.3.0, DB
`https://vuln.go.dev` updated 2026-05-22):

```
No vulnerabilities found.
```

Because the module has no external dependencies, `govulncheck` only
reports issues against the standard library symbols actually used by
`goav1`. Integrators should run `govulncheck` against their own
binary as well — that scan covers the toolchain version they ship
with.

## Coordinated disclosure history

No security advisories have been issued for `goav1` as of the date
of this document. Future advisories will be published as GitHub
Security Advisories on `https://github.com/thesyncim/goav1` and
referenced here.
