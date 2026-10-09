# Final assembly removal — October 9, 2026

Commit `d985372a` removes the final repository-owned assembly sources:
AMD64 DCT32/DCT64 four-row/four-column transforms and CPUID/XGETBV. The
tracked source tree contains zero `.s` files. Upstream reference code and
the Go toolchain's own assembly are outside this inventory.

## Replacement contract

The four-lane transforms use Go `simd/archsimd` with the existing narrow
(±2^17) and wide (±2^19) fixed-point decompositions and stage clamps. DCT32
is generated from the even half of the DCT64 network; regenerating it
produces an identical checked-in file. Out-of-envelope stage bounds fall
back to the independent scalar implementation. Row adapters preserve
short-row behavior and leave extra tail elements untouched. Column length
guards use division to avoid overflowing the stride-length multiplication.

AMD64 dispatch requires the official `archsimd.X86.AVX2()` query, including
its OS-state gate. Non-SIMD and `purego` AMD64 builds now use scalar
transforms rather than the former assembly implementation. This can affect
default-build performance; enable `GOEXPERIMENT=simd` for SIMD measurements.
AVX-512 reporting now requires the toolchain's complete feature bundle,
rather than only the old AVX-512F check. No new dependency or runtime
linkname is introduced.

## Emulated AMD64 timing, not native performance

Apple M4 Max, Rosetta 2, official Go 1.27.2, `GOEXPERIMENT=simd`,
`GOOS=darwin`, `GOARCH=amd64`, `GOAMD64=v1`, `CGO_ENABLED=0`.
The live-dispatch assertion and DCT32/DCT64 differential tests execute
without skipping in this environment. That proves execution under
emulation, not native x86 correctness or throughput.

Six interleaved baseline/candidate samples, one worker, 150 ms per benchmark:

| Four-lane operation | Old assembly ns/op | Go SIMD ns/op | Time change |
| --- | ---: | ---: | ---: |
| DCT32 columns | 2370.0 | 156.2 | -93.41% |
| DCT64 columns | 5017.0 | 379.6 | -92.43% |
| DCT32 rows | 2295.0 | 212.6 | -90.74% |
| DCT64 rows | 4929.5 | 491.1 | -90.04% |

All samples report 0 B/op and 0 allocs/op. Benchstat reports p=0.002 for
each timing comparison (n=6). These unusually large improvements are
**Rosetta-specific observations**, not native AMD64 or whole-decoder gains.
The shared workstation is not an isolated benchmark host.

The baseline directly invokes the old assembly adapters; the candidate
invokes the live dispatch slots, asserted to select Go SIMD. Both use the
same `benchmarkCol4`/`benchmarkRow4` helpers and repeatedly transform bounded
mutable buffers with `testing.B.Loop`. Column stride is six; bounds are
[-32768,32767]. The buffers are initialized once, not reset each iteration.
This measures repeated narrow-envelope transforms, not every codec shape
or wide-envelope performance. Baseline benchmark names are normalized from
`AVX2` to `Dispatch` only for benchstat pairing.

Binary SHA-256:

- Baseline: `d16133d1918e8b874264a599d135e0c56e134b67150127dca427fcefa1a3d5fd`.
- Candidate: `7720622a7bc25f789276d9056d8d8586a2d688bcbc0ac0121416e065aff9d578`.

The baseline was built before the final transform deletion from the old
transform sources present at `d5de4005`, with CPU-detection work in flight.
The direct assembly adapter calls bypass CPU dispatch. The candidate was
built from the replacement before its commit. Binary build metadata
confirms the identical toolchain and architecture flags above.

Raw samples: [baseline](benchmarks/assembly-removal-rosetta-base.txt) and
[candidate](benchmarks/assembly-removal-rosetta-candidate.txt).

Reproduce each of six alternating rounds with the preserved binaries in
`/tmp/pr6/assembly-removal-20261009`:

```sh
./base-transform-amd64.test -test.run '^$' \
  -test.bench '^BenchmarkDCT(32|64)(Col|Row)4AVX2$' \
  -test.benchtime 150ms -test.cpu 1 -test.count 1
./transform-amd64.test -test.run '^$' \
  -test.bench '^BenchmarkDCT(32|64)(Col|Row)4Dispatch$' \
  -test.benchtime 150ms -test.cpu 1 -test.count 1
```

## Validation boundaries

Native ARM64 transform and CPU package tests pass. Differential checks cover
stage bounds, asymmetric clamps, extreme values, sparse inputs, unaligned
buffers, strides, short rows and tails. Normal builds retain a separate
zero-allocation assertion. With `-d=checkptr=2`, pointer instrumentation
forces row staging onto the heap: the normal zero-allocation assertion
fails with one allocation per pair in that instrumented build. Parity and
short-row checks run separately with checkptr enabled; the allocation gate
is not weakened or skipped in normal builds.

Focused race tests pass, as do default and SIMD+purego transform/CPU tests,
SIMD vet for transform/CPU/encoder, and Linux AMD64/ARM64 SIMD cross-builds.
The public decoder MD5 preflight passes all 18 clips and 864 visible frames
against the independent sidecars. These are correctness checks, not timing
samples. The race-shard package audit is updated to remove the deleted
`tools/itxgen/avx2gen` package and passes for all 38 remaining packages.

Rosetta execution is emulated evidence. Linux cross-builds prove compilation,
not execution. The CI workflow now runs focused four-lane parity and live
AMD64 binding checks early on its native AVX2 runner; those results must be
read separately before claiming native AMD64 validation.

The [E2E comparison](e2e-codec-gap-20261009.md) remains pinned to `0824b687`:
about 3.65× Go/dav1d decode time and 3.68× median Go/SVT encode time in its
measured configuration, with unequal quality tradeoffs. Assembly removal
alone does not establish that either gap has closed.
