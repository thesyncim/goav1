# Go 1.27 SIMD kernel measurements

These arm64 kernel measurements used the official Go 1.27.1 toolchain with
`GOEXPERIMENT=simd` on an Apple M4 Max. The source baseline was `35c0602a`; the
final selected implementation is `c45e106e`. Each result is the median of five
200 ms samples at `GOMAXPROCS=1`; all reported `0 B/op` and `0 allocs/op`.

The values below describe specific kernels and shapes. They do not quantify
whole-codec throughput or encoder quality, and they are not comparisons against
a C codec.

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
pure Go; IDTX 8×8 measured 5.895 ns. The SIMD-enabled inverse DCT int16 pipeline
beat the assembly fallback at each measured block size:

| Block | SIMD | Assembly fallback | Ratio |
| --- | ---: | ---: | ---: |
| 8×8 | 106.9 ns | 135.2 ns | 1.26× |
| 16×16 | 452.6 ns | 538.0 ns | 1.19× |
| 32×32 | 1061 ns | 1536 ns | 1.45× |
| 64×64 | 4973 ns | 5489 ns | 1.10× |

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

## Limits and scope

Several candidates improved over their earlier Go SIMD versions but remained
slower than assembly: `ConvolveX8` moved from 832.7 to 241.7 ns (3.45×), while
I8MM measured 97.64 ns; Smooth 64×64 moved from 1326 to 700.7 ns (1.89×), while
NEON measured 463.7 ns; MinMax 8×8 moved from 37.68 to 12.20 ns (3.09×), while
assembly measured 6.876 ns. The 8×8 forward DCT reduced vector spills from 18
to 7 but remained slower than NEON (64.37 vs 45.76 ns). Corrected Wiener
measurements found the SIMD variants slower than assembly; an earlier result is
excluded because its NEON comparator fell back to scalar code.

The exploratory cross-decoder report used an internal oracle pipeline, not the
public decoder API. Its corpus lacks a manifest and provenance, so it does not
support a public decode-performance claim; a public-path comparison is pending.

## Reproducing the kernel results

Use official Go 1.27.1 for the comparisons below; the module minimum is Go
1.27.0 and the project does not pin a patch release. The checkpoints identify
the available source and benchmark harness containing each candidate and its
comparator. They may differ from the exact build snapshot behind the stored
medians; use the listed benchmark selectors at those checkpoints.

| Results | Checkpoint | Package and benchmark selectors |
| --- | --- | --- |
| Encoder metrics | `60c0a67a` | `encoder`: SATDCoeffs 16/64/256/1024, Hadamard 4×4/8×8/16×16/32×32, SAD 8×8×4 |
| Transforms | `60c0a67a` | `transform`: ForwardDCT4x4Kernels, ForwardBlock8x8IDTXImpl, InverseDCTBlock 8×8–64×64 |
| Motion | `3298f798` | `motion`: ConvolveX8 Go SIMD, I8MM, and NEON comparators |
| Smooth and CFL | `d0344a30` | `prediction`: Smooth and CFL subsampling comparisons |
| DSP and CDEF | `ce77788f` | `dsp`: MinMax8x8; `cdef`: FindDirection SIMD and assembly comparisons |
| Wiener and HBD loop filters | `3298f798` | `restoration`: Wiener horizontal/vertical; `loopfilter`: filter6/filter14 edge cases |

At a checkpoint, run `scripts/simdtip.sh test -run '^$' -cpu=1 -bench '<selector>' -benchmem -benchtime=200ms -count=5 <package>`. Keep `GOMAXPROCS=1`, `GOGC=100`, and `GOFLAGS=`; use `GOTOOLCHAIN=local` with the official Go executable. Run the assembly fallback comparison with `GOEXPERIMENT=nosimd` and the same inverse-DCT selector. The temporary raw logs are not checked into the repository.

These measurements cover one M4 Max and selected kernel shapes. They do not
establish codec speedups, cross-machine results, or superiority to optimized C
codecs. Recheck correctness, dispatch, allocations, and public end-to-end
workloads before making those claims.
