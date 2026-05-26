# goav1 v0.1.0 (pre-release notes)

This document captures the intent of the upcoming `v0.1.0` tag. It
is meant to be read alongside [CHANGELOG.md](CHANGELOG.md). The
changelog is the authoritative ordered list of changes; this file
summarises what is stable, what is experimental, and what is known
broken so that integrators can decide whether to depend on goav1
today.

`v0.1.0` is a **pre-1.0 pre-release**. Public APIs are stable in
shape but the project does not yet make backwards-compatibility
promises. Pin a commit or a tagged release in `go.mod`.

## Highlights

- **Seven of eight libaom `SuiteLevelFast` vectors pass** the
  lenient first-frame MD5 gate. The CI workflow asserts the
  seven-vector pass set as a regression gate on every push.
- **End-to-end CLI** (`cmd/aom-go-dec`) turns an IVF into raw YUV
  using only the public API. It is the canonical example of how
  to wire goav1 into a downstream pipeline.
- **Zero allocations** in the steady-state hot path of the
  residual stream runner, the postfilter dispatch, and the
  inverse qmatrix dequant path, guarded by allocation-regression
  benchmarks that fail CI on regression.
- **Public API documentation** on every long-lived caller-facing
  type and helper, plus a top-level `ARCHITECTURE.md` and
  `CONFORMANCE.md` for new contributors and integrators.
- **Licensing and patent grant** explicit: `LICENSE` (BSD 2-Clause),
  the verbatim Alliance for Open Media Patent License 1.0 in
  `PATENTS`, and a complete upstream attribution list in `NOTICE`.

## What is stable

The following surfaces are considered stable in shape and are
expected to keep their current signatures across `v0.1.x`. Bug
fixes may change semantics in narrow, documented ways.

- **Transport iterators**: IVF, low-overhead OBU, temporal-unit,
  Annex B, RTP payload iteration, RTP frame assembly, and the
  matching size-preflight helpers.
- **Bitstream parsers**: sequence header, uncompressed frame
  header, all per-syntax parameter blocks
  (`QuantizationParams`, `LoopFilterParams`, `CDEFParams`,
  `RestorationParams`, `FilmGrainParams`, etc.), tile-group
  header.
- **Decoder driving**: `DecoderStream`, `DecoderEvent`,
  `DecoderFrameWorkState`, `DecoderSurfaceReferences`,
  `FramePool`, `BeginDecoderFrameWork`,
  `RunDecoderFrameWorkEventWithContext`.
- **Tile work scheduling**: `NewTileWorkerPool`, `TileBatch`,
  `TileJob`, `PlanDecoderTileWork`,
  `ExecuteDecoderFrameWorkStep`.
- **Residual decode + reconstruct**:
  `DecoderFrameWorkBatchResidualRunner` and the per-job
  `DecodeAndReconstructDecoderFrameWorkJobResiduals` /
  `DecodeAndRetainDecoderFrameWorkJobResiduals` helpers.
- **Post-filter pipeline**:
  `BindDecoderFrameWorkPostFilterRequest` composes loop filter,
  CDEF, super-res, loop restoration, and film grain over
  caller-owned scratch.
- **Public DSP primitives**: intra prediction, directional intra,
  filter-intra, CfL, OBMC, inter prediction, residual add, A64
  mask blending, loop-filter edge application, CDEF and loop-
  restoration helpers.
- **Frame pool and frame format**: caller-owned plane layout,
  pool backing sizing, binding primitives, sample-plane
  load/store helpers, including the monochrome (I400) layout.

These are all driven by allocation-regression tests; the
hot-path contract is `0 B/op` per frame.

## What is experimental

These surfaces work but are subject to short-term shape
changes. Depend on them only with a vendored commit pin.

- **End-to-end CLI flags**. `aom-go-dec`'s flag set
  (`-o`, `-workers`, `-quiet`) is unlikely to break, but
  additional flags (post-filter staging, deterministic worker
  ordering, profile gates) are expected to arrive in `v0.2`.
- **Strict per-frame MD5 mode** (`GOAV1_STRICT_MD5=1`). The
  diagnostic group is informational. Today only the `16x16_size`
  vector clears every-frame parity; the goal is to expand
  coverage across `v0.1.x`.
- **Postfilter chain on the CLI**. The CLI currently runs
  residual decode and reconstruction. The full post-filter
  chain (loop filter, CDEF, super-res, loop restoration, film
  grain) is wired into the public API but not yet driven by the
  CLI.
- **Public fuzz harnesses**. The harnesses exist and run under
  `make fuzz-smoke`, but the corpus is intentionally minimal.
  Long-running fuzz on these surfaces is expected to produce
  new findings; please file issues.

## What is known broken

- **`intra-only_intrabc_extreme_dv` libaom vector**. The
  framework dry-run hits a known mismatch in later frames.
  Initial intrabc decode (`865639b`), advancing the intrabc
  vector to MD5 (`0b1fe9d`), resetting stale intra-mode and
  palette context (`4c680b8`), and snapshotting SB-top / SB-left
  / prior-SB-interior neighbor history for the cross-superblock
  intrabc DV scan (`0a8d482`, `7a65271`, `1bb40cb`) are in
  flight. Until the vector flips green, intrabc paths that are
  not yet covered are explicitly rejected (`7e2f748`,
  `dca70b0`) so that downstream callers see a clear error
  rather than a silent miscompare.
- **Encoder**. Not implemented. The repository is decoder-first;
  the encoder foundation is on the roadmap (see `README.md`
  "Principles").
- **SIMD / platform-specific assembly**. Not implemented. The
  pure-Go reference implementations are correct and benchmarked
  but unaccelerated. SIMD parity is on the roadmap.
- **Vectors outside `SuiteLevelFast`**. The full libaom remote
  suite is available via `make testvectors-full` for local
  parity sweeps. It is not part of the default CI gate and is
  expected to surface failures while broader vector coverage is
  built out.

## Reproducing the v0.1.0 state

```sh
git clone https://github.com/thesyncim/goav1
cd goav1
make ci-local            # fmt-check + vet + test + alloc
make testvectors-fast    # committed test-vector suite + oracle MD5 checks
make dryrun-fast         # framework dry-run against the eight fast vectors
make build-cmd           # builds ./bin/aom-go-dec
```

End-to-end smoke:

```sh
./bin/aom-go-dec \
    -o /tmp/quantizer00.yuv \
    internal/av1/testdata/libaom/av1-1-b8-00-quantizer-00.ivf
```

## Performance snapshot

End-to-end decoder throughput on the bundled `av1-1-b8-00-quantizer-00.ivf`
vector (Apple M4 Max, Go 1.23, residual decode + reconstruct only;
post-filter chain not in the publication path on this fixture):

```
BenchmarkDecodeFullVector-16          87  37268384 ns/op   3.46 MB/s    2.000 frames/op    53.66 frames/s    0 B/op    0 allocs/op
BenchmarkDecodeFirstFrameOnly-16      67  17721317 ns/op   4.20 MB/s                                        0 B/op    0 allocs/op
BenchmarkDecodeFullVectorAllocs-16    32  37482921 ns/op                                                    0 B/op    0 allocs/op
```

Post-filter chain breakdown (synthetic 128x128 4:2:0 8-bit fixture,
same machine):

| Stage                   | ms/frame | MB/s  |
|-------------------------|---------:|------:|
| Loop filter             |    0.728 | 33.78 |
| CDEF                    |    1.480 | 16.60 |
| Loop restoration        |    0.342 | 71.87 |
| Full chain (LF+CDEF+LR) |    2.497 |  9.84 |

These numbers exist primarily as a baseline for SIMD acceleration
work; they are not representative of the throughput a SIMD-enabled
release will deliver.

## Upgrade notes

There is no prior tagged release. Callers vendoring the current
`main` should re-vendor at the `v0.1.0` tag once cut. Notable
behavioral changes since the start of this session:

- Palette tokens are decoded after `mbmi` syntax. Any downstream
  that consumed them earlier in the block decode order needs to
  follow the new order.
- Inverse qmatrix indexing is now libaom row-major.
- Restoration stripe boundary lines are captured before CDEF and
  restoration, including when CDEF is disabled.

## Licensing reminder

goav1 is distributed under BSD 2-Clause (`LICENSE`). Portions of
the repository are derived from libaom; the Alliance for Open
Media Patent License 1.0 is reproduced verbatim in `PATENTS` and
must be redistributed alongside any source-form distribution of
goav1, per section 1.2.1 of that grant. The complete upstream
attribution list lives in `NOTICE`.
