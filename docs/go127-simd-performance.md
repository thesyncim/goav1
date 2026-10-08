# Go 1.27 SIMD performance evidence

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
