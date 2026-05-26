# Changelog

All notable changes to goav1 are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once a tagged release is cut. Until the first tag the only section is
`[Unreleased]`. Commit SHAs are cited in parentheses where they add
clarity; not every commit is enumerated.

## [Unreleased]

This entry summarises the work landed on `main` ahead of the first
tagged pre-release (`v0.1.0`). It groups the recent commit history into
the categories that matter to integrators: correctness gains against
the libaom conformance vectors, the public API and CLI surface,
performance instrumentation, hardening / fuzz coverage, and CI.

### Added

#### Tooling and public API

- **End-to-end CLI**: `cmd/aom-go-dec` ships an IVF -> raw YUV
  command-line decoder that drives the same public residual stream
  runner used by `BenchmarkDecodeFullVector`. It is the shortest path
  from "I have an IVF" to "I have raw YUV bytes" and serves as
  executable documentation for the public API (`c36d64e`). Build via
  `make build-cmd` / `make install-cmd`. Supports `-o`, `-workers`,
  `-quiet`, with `stderr` reserved for per-frame timing so `stdout`
  remains pipe-clean for `ffplay`/`ffmpeg`.
- **Public API documentation passes**: every long-lived caller-facing
  type and helper now has doc comments and an example test. Documented
  groups include the decoder entry point (`91085d5`), tile decode and
  restoration (`e7fd015`), sequence/frame-header parsers (`dfd7d70`),
  residual decode and reconstruct entry points (`d5c769b`), postfilter
  bind helpers (`2cd7446`), and remaining decoder/tile re-export
  aliases (`b2dbe97`). The README was refreshed to match the surface
  area, status table, and example coverage (`33844b5`).
- **Architecture and conformance docs**:
  - `ARCHITECTURE.md` is the end-to-end map of the decoder for new
    contributors and integrators (`da57bf0`).
  - `CONFORMANCE.md` is the spec-level feature inventory: which AV1
    syntax / prediction / transform / postfilter rows are
    implemented, partial, or missing, plus the libaom fast-suite
    vector pass/fail table (`e259eb7`).

#### Licensing and attribution

- **LICENSE / PATENTS / NOTICE**: BSD 2-Clause `LICENSE`, the verbatim
  Alliance for Open Media Patent License 1.0 in `PATENTS`, and a
  complete upstream attribution list in `NOTICE` (`7637155`). The
  patent grant is reproduced because portions of this repository are
  derived from libaom.
- **Upstream citations**: SPDX-style headers and libaom citations were
  added across the `internal/av1/*` tree, including parser/OBU
  (`17e999b`), quantize/entropy/prediction tables (`d2bc4d7`),
  transform/motion (`9048333`), filter packages (`920438d`), and the
  goav1-original support packages (`16f533e`).

#### Security policy

- **`SECURITY.md`** documents the threat model, the
  caller-provided-scratch contract, the supported versions policy,
  and the reporting process (`5514ef4`).

#### Conformance / dry-run infrastructure

- **Strict per-frame MD5 mode** for the libaom dry-run, gated by
  `GOAV1_STRICT_MD5=1`. The default `make dryrun-fast` keeps the
  lenient first-frame gate; CI emits an informational `strict-md5`
  group for diagnostic snapshots (`5d18016`).
- **End-to-end decoder throughput and allocation benchmarks**
  (`BenchmarkDecodeFullVector`, `BenchmarkDecodeFirstFrameOnly`,
  `BenchmarkDecodeFullVectorAllocs`) wired through the public residual
  stream runner (`f2b1b62`), with `make bench` / `make bench-all` /
  `make bench-public` targets and README documentation
  (`42a12b0`).
- **Post-filter chain benchmarks** that measure loop filter, CDEF,
  and loop restoration both in isolation and as a chain on a shared
  synthetic frame so per-stage cost is directly comparable
  (`8d16be8`). The breakdown is reproduced in `README.md` under
  "Post-filter performance".

#### Fuzz harnesses and smoke targets

- Inverse-transform fuzz coverage for WHT, ADST, warp, and film grain
  (`ff8ba63`).
- Inverse-qmatrix dequant fuzz path (`455ddac`) and inverse-qmatrix
  allocation guard (`c59bbe1`).
- Palette mode and color-map decode fuzz harnesses (`133a448`).
- Public-API fuzz smokes: residual batch decode (`b820136`),
  public coefficient (`c6be58d`), public residual decode (`edfe62b`),
  residual event runner (`6851e77`), residual runner side data
  (`292087e`), and frame-work side-data binding (`5896d8a`).
- RTP sizing cross-checks inside the existing fuzzers (`10766cf`)
  and a `make fuzz-smoke` target that exercises every fuzz harness
  for a short, deterministic run (`030a3f6`).
- Gated debug instrumentation for CDEF, LR, and Wiener stages to
  speed up future bug bisects (`d2254e8`).

### Fixed

This section lists the high-leverage correctness fixes that
unblocked specific libaom fast-suite vectors. The pass set went
from a single vector to seven of eight over the course of this
work.

#### Inter / motion vector decode (`mv` vector)

- Fixed compound `NEW_NEW` MV reference selection (`c3a12c0`).
- Wrapped coefficient decode errors with surrounding block context
  for triage (`a3b0b74`).
- Honored `qindex == 0` for the inter transform type selection
  (`bf85e9f`).
- Honored frame-edge availability in the inter txfm partition
  context (`0961ac1`).
- Restricted the `GLOBALMV` mode-context bit to the (0,0) temporal
  sample (`a2070d7`).
- Silently skip inter prediction beyond the visible frame instead of
  erroring (`a66c0d4`).
- Provisioned a libaom-sized dry-run frame pool (`c41b4ba`) and
  reset retained frame CDF counters (`871a328`).
- Promoted `mv` into the framework dry-run pass set and added a CLI
  smoke (`907479d`).

(See also `4c680b8`, `7a65271`, `0a8d482`, `1bb40cb` for the
`intrabc` neighbor history follow-up that is in flight but does not
yet unblock the `intrabc_extreme_dv` vector.)

#### Entropy and CDF adaptation (`cdf_update` vector)

- Read the post-segment id before the CDEF index (`4264755`).
- Match libaom's 2D BR coefficient context (`ee175e7`).
- Match libaom's selected-TX context availability (`7e10845`).
- Derive intra TX mode from the filter-intra mapping (`dd4e26d`).
- Match libaom's palette cache boundary (`21580d5`).
- Derive the EOB context from the transform class (`4359c1b`).
- Cap transform scale to the libaom maximum (`4014017`).
- Wire inverse qmatrix dequantization, indexing the matrix by the
  libaom row-major position (`d6fb1d2`, `81dc0d6`, `c1787d0`).
- Restored column-major inverse qmatrix indexing where required
  (`9e5d21c`).

#### Monochrome surface decode (`monochrome` vector)

- Match libaom monochrome frame MD5 (`e5c9416`).
- Header-derived frame formats including monochrome surface layout
  flowed in through the postfilter bind work below.

#### All-intra and palette (`all-intra` vector)

- Wire luma palette mode decode (`1cc0425`).
- Wire chroma palette mode decode (`79a5431`).
- Decode UV palette mode even when Y palette is absent (`90250a5`).
- Decode palette tokens after `mbmi` syntax (`59390b3`).
- Cap directional intra top-right and bottom-left edges
  (`747b3d4`).
- Clip frame-edge intra predictions (`e504175`).

#### Inverse transform / dequant correctness

- Clamp inverse transform 1D stage inputs to int16 (`f11f7e2`).
- Restore transform scale 2 for large transforms (`de715f5`).
- Mask the dequantized coefficient product to 24 bits (`1e1e304`).
- Bit-exact libaom comparison tests for inverse ADST8
  (`af49ff4`) and inverse DCT16 (`5755782`).

#### Loop filter, CDEF, restoration, film grain

- Use the 6-tap loop filter for chroma edges (`b40ae2f`).
- Skip the chroma loop-filter edge when the neighbor lacks chroma
  (`c61cdd0`).
- Clamp the loop-filter level after `delta_lf` before `segdelta`
  (`706ef13`).
- Skip all-skipped 8x8 CDEF blocks like libaom (`2941cc5`).
- Wrap SGR projection through int16 like libaom (`06f433b`).
- Save restoration stripe boundary lines before CDEF and restoration
  (`d7b8372`); also save restoration boundaries when CDEF is disabled
  (`bd8aa1b`).
- Harden restoration bordered-block validation (`7c893f5`).

#### Inter-intra and warped motion

- Support the inter-intra prediction path end-to-end (`820c844`).
- Avoid stale bound postfilter maps (`93592a7`).
- Track and provision max frame side-scratch (`1a9a603`).
- Wire framework side data into the oracle dry-run (`e586480`).
- Reject explicitly unsupported intrabc paths until the full neighbor
  history lands (`7e2f748`, `dca70b0`).

### Changed

- **Palette decode order**: palette tokens are now decoded after the
  `mbmi` syntax, matching libaom's actual decode order (`59390b3`).
  This is a behavioral change for any downstream that consumed the
  intermediate palette tokens before they were authoritative.
- **Postfilter chain**: restoration stripe boundary lines are now
  captured before CDEF and restoration, and stay captured even when
  CDEF is disabled (`d7b8372`, `bd8aa1b`).
- **Inverse qmatrix indexing** is now libaom row-major
  (`81dc0d6`, `d6fb1d2`, `c1787d0`).
- **Modernization sweep**: every internal package now uses
  `range`-over-int and `min`/`max` built-ins where appropriate.
  Touched packages: `cdef` (`1467bf1`), `restoration` (`9098e8f`),
  `filmgrain` (`049ce34`), `loopfilter` (`1b54dfa`),
  `transform` (`361e410`), `dsp` (`6b8fda6`), `parser` (`ab1647b`),
  `motion` (`062e6b5`), `quantize` (`a0f0fb3`), `frame` (`abe4f78`),
  `ivf` and `rtp` (`17597f1`), and `entropy` (`0037947`). Behavior
  is unchanged; the diff is mechanical and reviewable.

### Security

- **Bitstream hardening**: `filmgrain` and `quantize` are now hardened
  against malformed bitstreams (`744ef85`). Restoration bordered-block
  validation was tightened (`7c893f5`).
- **Allocation guardrails** locked in across the postfilter dispatch
  paths (`e69a428`) and on the inverse qmatrix dequant path
  (`c59bbe1`). These prevent a malformed stream from triggering an
  unbounded scratch allocation and are paired with corresponding
  zero-allocation regression tests.
- **CI gate**: a regression workflow (`91042e6`) now enforces
  build + vet + test + lint + dry-run on every push. Allocation
  regressions and dry-run pass-set regressions fail CI.

### CI / Infrastructure

- New regression workflow with build, test, lint, and dry-run gates
  (`91042e6`).
- New `make` targets: `build-cmd`, `install-cmd`, `bench`,
  `bench-all`, `bench-public`, `testvectors-fast`,
  `dryrun-fast`, `fuzz-smoke`, and `ci-local`. All targets are
  documented in `README.md`.

---

## Test vector status (libaom `SuiteLevelFast`)

Snapshot of `make dryrun-fast` results at the time of writing.
The lenient gate is the first-frame MD5; the strict gate
(`GOAV1_STRICT_MD5=1`) is every frame.

| Vector                          | Lenient (first-frame MD5) | Strict (every frame) |
|---------------------------------|:-------------------------:|:--------------------:|
| `quantizer_00`                  | PASS                      | (diagnostic)         |
| `16x16_size`                    | PASS                      | PASS                 |
| `all-intra`                     | PASS                      | (diagnostic)         |
| `cdf_update`                    | PASS                      | (diagnostic)         |
| `mv`                            | PASS                      | (diagnostic)         |
| `mfmv`                          | PASS                      | (diagnostic)         |
| `monochrome`                    | PASS                      | (diagnostic)         |
| `intra-only_intrabc_extreme_dv` | FAIL (later frames)       | FAIL                 |

The `testvectors` CI workflow asserts the seven-vector lenient pass
set as a regression gate. The `intra-only_intrabc_extreme_dv` vector
is the next unblock target; the cross-superblock intrabc neighbor
history work is in flight (`4c680b8`, `7a65271`, `0a8d482`,
`1bb40cb`).

The strict-MD5 column is informational only and is emitted as a
diagnostic CI group. It is expected to expand as the per-frame
parity work continues.
