# Go 1.27 SIMD performance evidence

The final validation for the October 8 follow-up uses official Go 1.27.2.
Earlier measurements below retain their original Go 1.27.1 labels; compiler
versions are not mixed within a before/after comparison. The module minimum
remains 1.27.0 and CI tracks 1.27.x without a toolchain directive.

These arm64 kernel measurements used the official Go 1.27.1 toolchain with
`GOEXPERIMENT=simd` on an Apple M4 Max. The source baseline was `35c0602a`; the
kernel-selection checkpoint was `c45e106e`. Kernel results are medians of five
200 ms samples at `GOMAXPROCS=1`; all reported `0 B/op` and `0 allocs/op`.

The kernel measurements describe specific operations and shapes. They do not
quantify whole-codec throughput, encoder quality, or performance against a C
codec. The exploratory public-path samples later in this note have separate
limits.

## Selected kernel results

The encoder SIMD candidates beat their ARM64 assembly comparators in these
measured cases. The ratio is comparator time divided by Go SIMD time.

| Kernel | Go SIMD | Assembly | Ratio |
| --- | ---: | ---: | ---: |
| SATD, 16 coefficients | 2.489 ns | 9.857 ns | 3.96× |
| SATD, 64 coefficients | 3.992 ns | 8.198 ns | 2.05× |
| SATD, 256 coefficients | 10.57 ns | 29.68 ns | 2.81× |
| SATD, 1024 coefficients | 37.01 ns | 116.2 ns | 3.14× |
| Hadamard 4×4 | 2.814 ns | 2.926 ns | 1.04× |
| Hadamard 8×8 | 5.781 ns | 6.907 ns | 1.19× |
| Hadamard 16×16 | 41.69 ns | 48.27 ns | 1.16× |
| Hadamard 32×32 | 223.2 ns | 246.5 ns | 1.10× |
| SAD 8×8 ×4 | 11.59 ns | 13.63 ns | 1.18× |

Forward DCT 4×4 measured 11.46 ns versus 12.04 ns for NEON and 35.44 ns for
pure Go; IDTX 8×8 measured 5.895 ns. The historical `InverseDCTBlock` and
`InverseBlock` measurements below came from the int32 transform path at the
pre-guard checkpoint `c45e106e`. They do not measure the bitdepth-8 int16
pipeline or the public raw-block path. Bounds guards added at `28678` protect
extreme values, so treat this as an archival pre-guard comparison.

| Inverse int32 block | SIMD | Assembly fallback | Ratio |
| --- | ---: | ---: | ---: |
| 8×8 | 106.9 ns | 135.2 ns | 1.26× |
| 16×16 | 452.6 ns | 538.0 ns | 1.19× |
| 32×32 | 1061 ns | 1536 ns | 1.45× |
| 64×64 | 4973 ns | 5489 ns | 1.10× |

The int16-output transform entry point has a separate
`BenchmarkInverseBlockBitDepth8` benchmark for bounded and high-range coefficients. At candidate checkpoint
`ab546d30`, the DCT8 full-range rotation and input-guard tests were validated.
The paired four-sample, 300 ms comparison on the M4 Max (Go 1.27.1,
`GOEXPERIMENT=simd`) measured these DCT8×8 fixtures:

| Fixture | Baseline at `28678f23` | Candidate at `ab546d30` |
| --- | ---: | ---: |
| Bounded coefficients | 194.1 ns/op | 170.1 ns/op |
| High-range coefficients | 125.0 ns/op | 127.2 ns/op |

Both paths reported zero allocations. The bounded fixture was 12.4% faster in
the candidate; the high-range fixture was 1.8% slower. These synthetic kernel
results do not measure whole-decoder throughput; the decoder uses the separate
int32 raw-transform entry point.

Other selected cases were narrow wins: CDEF direction search measured 18.37 ns
with SIMD versus 19.03 ns with assembly; its 8-bit path measured 18.10 ns
versus 19.38 ns with NEON. HBD filter14 improved on 10-bit horizontal flat and
mixed edges (269.7 vs 311.5 ns; 147.4 vs 306.8 ns) and vertical flat edges
(511.9 vs 568.9 ns). HBD filter6 improved on horizontal 10-bit flat and mixed
edges (157.6 vs 165.8 ns; 138.0 vs 163.0 ns), vertical 10-bit flat and mixed
edges (373.9 vs 409.6 ns; 374.5 vs 405.1 ns), and vertical 12-bit flat and
mixed edges against the fallback (376.2 vs 693.4 ns; 379.5 vs 872.1 ns).
CFL subsampling improved against NEON for 8-bit inputs and against pure Go for
16-bit inputs. The larger CFL apply case did not beat NEON, so that path keeps
its existing implementation.

## Removing repeated prediction work

The resident high-bit-depth warp horizontal pass now evaluates four dot products
with Go SIMD pairwise reductions and rounds/stores them as one vector. It reads
the same 15-by-15 source footprint as the scalar implementation. XOR-centering
and a filter-sum correction preserve the full uint16 input domain; unaligned
sources and odd strides retain the scalar path.

On the same M4 Max and official Go 1.27.1 SIMD build, five 300 ms samples measured
371.8 ns for the scalar 15-by-8 intermediate tile and 104.3 ns for the selected
SIMD kernel (72.0% less time), with zero allocations. The intermediate variant
using scalar reductions and vector rounding measured 111.6 ns. These are kernel
measurements, not a whole-decoder speedup claim.

CfL now prepares the common luma subsampling and AC buffer once for U and V.
The live HBD path also reaches the existing SIMD subsampling primitive through
a checked little-endian uint16 view. Aliasing planes, odd byte strides, unaligned
inputs, and unsupported native byte order retain the appropriate fallback;
U still writes before V geometry errors are reported.

A three-sample, 200 ms CfL block benchmark over 10/12-bit 4:2:0 and 4:4:4
8/16/32-sized blocks measured 14.9–19.4% less time for shared preparation, with
zero allocations. Both sides of this particular comparison already use the
new HBD loader: it isolates preparation reuse rather than measuring the entire
change against the previous decoder.

Against `855b80f4`, six alternating baseline/candidate public warm-reset runs
measured the two changes together on the same M4 Max (one thread, 1 s per
sample). Median time for the 48-frame `p360_inter_q32_10bit` clip decreased
from 101.334 ms to 98.417 ms (2.88%). The two 720p 8-bit clips changed by
−0.35% and −0.96%, too small for a strong improvement claim on this host.
All 18 clips / 864 reconstructed frames match their reference MD5s. The full
Go test suite, focused race/checkptr tests, default and purego fallbacks passed.

A request/CDF pointer experiment removed 792 bytes of repeated block arguments,
but its initial improvement did not survive an ablation in the combined decoder.
It was discarded. A per-cell motion-projection cache similarly removed arithmetic
but increased routine time by 22–25% versus the actual previous implementation;
it was discarded in favor of testing whole-run processing separately.

The whole-run motion-field prototype was also discarded: it halved the
repeated-run microbenchmark but regressed alternating-reference inputs and
showed no convincing public decode gain. A typed scratch reset reduced clearing
from 35,458 to 16,002 bytes per root and measured 560.8 to 167.9 ns. Its initial
standalone result did not translate reliably into the combined decoder; the
manual field-clearing implementation and stale-payload contract were not kept.

## Remaining 8-bit motion work

The retained I8MM single-reference and compound kernels now use six vertical
taps and height+5 intermediate rows when both endpoint coefficients are zero.
Sharp/mixed-endpoint filters retain eight taps; width-four and edge fallbacks
remain unchanged. This optimizes existing assembly rather than replacing it:
the Go 1.27 ARM64 SIMD API does not expose I8MM matrix multiplication.

Five 200 ms samples on the M4 Max with Go 1.27.1 measured 11–15% less time for
single-reference filtering and 14–16% for compound filtering across 8×8 through
128×128 blocks, with zero allocations. A twelve-pair public follow-up against
`855b80f4` measured 230.332 to 227.699 ms for `p720_inter_q20` (1.14% less time)
and 185.468 to 184.247 ms for `p720_inter_q32` (0.66%). An earlier six-pair run
had the same direction; whole-decoder gains remain much smaller than kernel gains.

| Shape | Go single | dav1d put | Go compound | dav1d prep |
| --- | ---: | ---: | ---: | ---: |
| 8×8 | 37.74 ns | 14.98 ns | 35.58 ns | 14.97 ns |
| 32×32 | 283.3 ns | 191.6 ns | 256.0 ns | 191.3 ns |
| 128×128 | 4079 ns | 2925 ns | 3658 ns | 2925 ns |

The Go measurements include resident wrapper checks and coefficient preparation;
the C measurements call specialized symbols directly. Both use matching pixels,
filter phases, source layout and output checks. The remaining structural gap is
that Go writes/rereads a full horizontal intermediate tile while dav1d keeps a
rolling row window in registers. See the [reproducible I8MM harness](../tools/u8_motion_bench/README.md)
for the pinned dav1d revision, archive hashes, exact timing boundary and commands.

## High-bit-depth interpolation follow-up

At `71aa5963`, the ARM64 SIMD dispatch uses six coefficient slots for 2D
10/12-bit interpolation when both endpoint coefficients are zero. This removes
two guaranteed-zero taps per pass and reduces intermediate rows from height+7
to height+5. The single-reference and compound paths retain their separate
rounding rules. Widths below eight and filters with nonzero endpoints still use
the existing assembly; that assembly remains necessary for those fallbacks.

The pre-final-cleanup candidate was measured against the existing NEON dispatch
on the same M4 Max with official Go 1.27.1 SIMD, one thread, and three 200 ms
samples. All 64 compared cases (regular 8-tap and 4-tap families, single and
compound output, shapes from 8×1 through 128×128) used 21.5–45.1% less time,
with zero allocations. For example, regular single-reference 8×1 measured
21.84 ns versus 29.17 ns, and 8×4 measured 32.80 ns versus 47.45 ns. The final
cleanup removes a redundant clamp confirmed in generated code; these numbers
precede that cleanup.

These are routine improvements. Paired public samples were noisy and did not
establish a comparable whole-decoder improvement. Differential
checks cover mixed filter phases, clamped edges, odd byte strides and offsets,
exact source windows, 10/12-bit extrema, and maximum scratch shapes. The full
public corpus matched all 864 reference frames, and normal SIMD and race tests
preserved zero-allocation assertions.

### Direct comparison with dav1d motion kernels

At `5518e364`, expressing both high-half operands explicitly lets Go 1.27 emit
`SMULL2` instead of a shuffle followed by a low-half multiply. This reduced
resident kernel time by roughly 4–12% in ten measured cases. The following
comparison uses the same source bytes, strides, phase pair (regular horizontal
6, smooth vertical 9), and 10-bit range for Go and dav1d. Every output sample
was checked before timing, including the signed-offset conversion for compound
buffers. Go values are medians of three 200 ms samples; dav1d values are medians
of five 200 ms samples after warmup. Both run serially on the M4 Max.

| Shape | Go SIMD single reference | dav1d `put` | Go SIMD compound | dav1d `prep` |
| --- | ---: | ---: | ---: | ---: |
| 8×8 | 42.87 ns | 25.43 ns | 39.65 ns | 24.24 ns |
| 16×16 | 132.4 ns | 85.56 ns | 118.0 ns | 81.46 ns |
| 32×32 | 481.7 ns | 313.1 ns | 429.1 ns | 296.4 ns |
| 64×64 | 1822 ns | 1202 ns | 1626 ns | 1135 ns |
| 128×128 | 7341 ns | 4707 ns | 6424 ns | 4497 ns |

Go still takes about 1.43–1.69× the time of these dav1d NEON kernels. This is
resident kernel cost, excluding decoder setup and Go dispatch checks. The
[standalone harness](../tools/hbd_motion_bench/README.md) reproduces the fixtures,
parity checks, and C timing without linking C into the Go decoder.

Two streaming variants modeled on dav1d removed the intermediate buffer but
lost against the retained Go kernel. The int16 version needed separate widening
multiplies and adds; the int32 version added broadcasts and register rotations.
Both were rejected after parity and timing checks. A similar high-half rewrite
in inverse transforms introduced spills and regressed, so it was rejected too.

## Skipping empty transform rows

At `61731bb4`, trusted reconstruction derives the default-scan row bound even
when it selects full dequantization. Previously that optimization depended on
selecting sparse or nonzero dequantization. The change preserves custom-scan,
lossless, and identity-transform behavior and uses the existing trailing-scratch
clear before the column pass.

Instrumentation on three clips found 327,600 provably empty row samples in the
affected blocks. In five 300 ms samples of full-dequant reconstruction fixtures,
16×16 with seven active rows measured 355.5 ns versus 490.6 ns for the full-row
fallback; 32×32 with fifteen active rows measured 1,189 ns versus 1,368 ns. Both
reported zero allocations. The comparator uses an identical copied scan to
select the full-row path. Six paired public runs per clip did not establish an
end-to-end improvement: medians varied from 0.2% to 1.0% slower amid several
percent of host drift. These kernel results are not decoder speedup claims.

## Limits and scope

Several candidates improved over their earlier Go SIMD versions but remained
slower than assembly: `ConvolveX8` moved from 832.7 to 241.7 ns (3.45×), while
I8MM measured 97.64 ns; Smooth 64×64 moved from 1326 to 700.7 ns (1.89×), while
NEON measured 463.7 ns; MinMax 8×8 moved from 37.68 to 12.20 ns (3.09×), while
assembly measured 6.876 ns. The 8×8 forward DCT reduced vector spills from 18
to 7 but remained slower than NEON (64.37 vs 45.76 ns). Corrected Wiener
measurements found the SIMD variants slower than assembly; an earlier result is
excluded because its NEON comparator fell back to scalar code.
Geometry-cache and compact-grid candidates were rejected because their
measured changes (about 0–3% and 0–1.6%, respectively) stayed within noise.

An earlier internal-oracle comparison reported `aomdec` and `dav1d` as 1.98×
and 3.05× faster than goav1; those ratios are not public API results and lack
corpus provenance.

## Public API and C API comparison

This matched API sample is from source checkpoint `61731bb4`. It
used the same 18 clips, each 48 frames, for 864 visible
frames total. Every Go and C decoder matched its clip's MD5 sidecar before
timing. The corpus has no source manifest, so the results are exploratory and
do not establish corpus provenance. Each path used one thread, one warmup, and
nine whole-clip samples; the table sums each clip's median. File reads and MD5
verification are outside the timed intervals. Go used official Go 1.27.1 with
`GOEXPERIMENT=simd`. The C backends are libaom 3.14.0 and dav1d 1.5.3 through
their APIs, so these results exclude CLI process startup.

| Decoder path | goav1 | libaom C API | dav1d C API |
| --- | ---: | ---: | ---: |
| Cold from IVF | 1681.637 ms | 834.506 ms | 528.357 ms |
| Cold from pre-parsed payloads | 1683.201 ms | — | — |
| Warm `Reset` and decode | 1630.585 ms | — | — |

The cold pre-parsed path parses and copies payloads before timing; its timed
scope is `NewDecoder(payloads)`, all `DecodeNext` calls, and `Close`. It differs
from the cold-IVF aggregate by only 1.564 ms (0.09%), so packet parsing does not
explain the decoder gap. `NewDecoderFromIVF` does IVF parsing and payload copies
during construction. The C helper views preloaded packet bytes without copying,
while parsing IVF packet records during each timed decode. These boundaries
are close but not identical. The cold-IVF Go/libaom and Go/dav1d ratios are
2.02× and 3.18×. The warm Go path reuses a decoder; no matching warm-reset C path
was measured.

These ratios describe this corpus and these API paths. Both C libraries use
architecture-specific optimized code, so the comparison does not isolate Go
SIMD or language speed. This dav1d sample uses a local optimized static build
from pinned commit `b546257f770768b2c88258c533da38b91a06f737`, with ARM64 assembly,
8/16-bit support, `-O3`, debug symbols, no LTO, and no stripping. Its full output
matches the installed dav1d build and all corpus sidecars. The libaom binary's
build provenance remains unattested. For sample boundaries and commands, see
the [C API sampler README](../tools/c_api_decode_bench/README.md).

To collect the Go values, set the corpus directory and run:

~~~sh
export GOAV1_PUBLIC_FULLCLIP_SAMPLES=1 GOAV1_BENCH_CORPUS_DIR=/path/to/benchcorpus
export GOEXPERIMENT=simd GOMAXPROCS=1 GOGC=100 GOTOOLCHAIN=local
scripts/simdtip.sh test -run '^TestPublicDecoderFullClipCorpusSamples$' -count=1 -v .
~~~

## Reproducing the kernel results

Use official Go 1.27.1 for the comparisons below; the module minimum is Go
1.27.0 and the project does not pin a patch release. The checkpoints identify
the available source and benchmark harness containing each candidate and its
comparator. They may differ from the exact build snapshot behind the stored
medians; use the listed benchmark selectors at those checkpoints.

| Results | Checkpoint | Package and benchmark selectors |
| --- | --- | --- |
| Encoder metrics | `60c0a67a` | `encoder`: SATDCoeffs 16/64/256/1024, Hadamard 4×4/8×8/16×16/32×32, SAD 8×8×4 |
| Transforms | `60c0a67a` | `transform`: ForwardDCT4x4Kernels, ForwardBlock8x8IDTXImpl, InverseDCTBlock/InverseBlock int32 8×8–64×64 |
| Motion | `3298f798` | `motion`: ConvolveX8 Go SIMD, I8MM, and NEON comparators |
| Smooth and CFL | `d0344a30` | `prediction`: Smooth and CFL subsampling comparisons |
| DSP and CDEF | `ce77788f` | `dsp`: MinMax8x8; `cdef`: FindDirection SIMD and assembly comparisons |
| Wiener and HBD loop filters | `3298f798` | `restoration`: Wiener horizontal/vertical; `loopfilter`: filter6/filter14 edge cases |

At a checkpoint, run `scripts/simdtip.sh test -run '^$' -cpu=1 -bench '<selector>' -benchmem -benchtime=200ms -count=5 <package>`. Keep `GOMAXPROCS=1`, `GOGC=100`, and `GOFLAGS=`; use `GOTOOLCHAIN=local` with the official Go executable. Run the assembly fallback comparison with `GOEXPERIMENT=nosimd` and the same inverse-DCT selector. The temporary raw logs are not checked into the repository.

These measurements cover one M4 Max and selected kernel shapes. They do not
establish codec speedups, cross-machine results, or superiority to optimized C
codecs. Recheck correctness, dispatch, allocations, and public end-to-end
workloads before making those claims.

## October 9 recovered motion and loop-filter work

PR checkpoint `4936325b` did not include nine completed motion and loop-filter
optimization commits. Checkpoint `4f5d6348` integrates those commits: specialized
8-bit compound filtering, vectorized warp filtering and edge windows, HBD
vertical-row reuse and unrolled compound taps, and loop-filter transposes,
shift reuse, and checked full-width gathers.

An interleaved A/B on one Apple M4 Max used official Go 1.27.2,
`GOEXPERIMENT=simd`, `-cpu=1`, six samples per side, and prebuilt test binaries.
Kernel samples used 150 ms; whole-clip samples used three iterations. The
`BenchmarkGoav1DecodeE2E` fixture reads IVF bytes before timing and observes the
decoded frame count. It includes decoder construction and decoding, not file
reads or CLI startup. Both revisions used the same existing corpus files.

| Whole-clip fixture | `4936325b` | `4f5d6348` | Time change |
| --- | ---: | ---: | ---: |
| `p360_inter_q32_10bit.ivf` | 114.4 ms | 100.6 ms | -12.0% |
| `p720_inter_q32.ivf` | 238.4 ms | 195.2 ms | -18.1% |

Both comparisons have `p=0.002`, `n=6` in benchstat. The geometric mean time
change across these two fixtures is -15.1%; it is not a full-corpus result.
Representative public-entry motion kernels improved by 70.8% for 32×32 8-bit
compound X, 74.3% for compound Y, 22.4% for HBD compound 2D, and 62.3% for HBD
compound Y. The direct 8-bit vertical warp kernel improved by 81.4%. Three
10-bit filter14 fixtures improved by 32.6–72.3%. All of these measured kernels
reported zero bytes and zero allocations per operation. The existing 16-wide
clamped 8-bit convolution benchmark had no significant change.

A separate four-tap, width-four HBD horizontal specialization at `b471b9fa`
avoids computing unused upper lanes. Public-entry 10/12-bit 4×4 and 4×8 measurements against
`4f5d6348` reduced time by 40.0–52.1% (`p=0.002`, six samples, zero allocations).
Exact-minimum input slices, odd strides, output guards, scalar agreement, and
allocation checks cover this path. The two whole-clip fixtures showed no
significant additional gain when this specialization and two other proposed
refinements were tested together. The unproven convolution split and
loop-filter constant-placement refinements were not retained; the HBD gain
is a public-operation result, not an additional decoder speedup claim.

The retained code passes motion/loop-filter SIMD and default tests, race tests,
`purego` tests, strict `checkptr=2` window checks, and focused `go vet`. The
generated corpus passes MD5 conformance for all 18 clips and 864 visible frames.
Linux/AMD64 SIMD motion and loop-filter test binaries cross-compile; this is
build coverage, not native AMD64 correctness or performance evidence.

Reproduce the whole-clip comparison by building each checkpoint's test binary
separately, then alternating executions instead of measuring concurrent runs:

~~~sh
CODEX_AGENT_ID=simd-ab GOTOOLCHAIN=go1.27.2 GOEXPERIMENT=simd \
  /Users/thesyncim/.codex/bin/project-env go test -tags goav1_oracle \
  -c -o /tmp/goav1-e2e.test ./internal/av1/testvector
GOAV1_BENCH_CORPUS_DIR=/path/to/benchcorpus /tmp/goav1-e2e.test \
  -test.run '^$' -test.cpu 1 -test.benchtime 3x \
  -test.bench 'BenchmarkGoav1DecodeE2E/(p360_inter_q32_10bit|p720_inter_q32)\.ivf$'
~~~

The project environment wrapper in this example is local to the measurement
machine; elsewhere use the official Go executable with a project-scoped cache.
Raw local logs are under `/tmp/pr6/recovery-20261009/`, not checked into Git.
These are exploratory same-project comparisons, not a new matched dav1d or
SVT-AV1 comparison. They do not establish superiority to either codec, and the
historical C API and encoder tables retain their original checkpoints.

## Encoder versus SVT-AV1

The [encoder comparison](encoder-svt-performance.md) records the Go 1.27.2
process-inclusive results against SVT-AV1 4.0.1 preset 13, quality metrics,
source hashes, removed prediction/range-scan work, and the diagnostic corpus
limits. Go remains slower in that comparison.
