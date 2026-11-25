//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"testing"
)

// filterIntra16Depths are the high-bit-depth clamps the SIMD kernel must honour:
// 10-bit (max 1023) and 12-bit (max 4095, whose tap sums overflow int16 and so
// exercise the 32-bit widening path).
var filterIntra16Depths = [...]int{1023, 4095}

// filterIntra16Kernels are the arm64 high-bit-depth filter-intra kernels: the
// 10-bit int16 TBL kernel (which hands every other depth to the 32-bit one)
// and the 32-bit kernel itself.
var filterIntra16Kernels = []struct {
	name string
	fn   func(planeBlock, int, int, FilterIntraMode, IntraEdges, int)
}{
	{"tbl10", predictFilterIntraBlockDirect10TBL},
	{"simd32", predictFilterIntraBlockDirect16SIMD},
}

// TestFilterIntra10SumFitsInt16 proves the int16 arithmetic of the 10-bit
// kernel exact: for every mode and output, the rounding bias plus the largest
// and smallest tap sums over neighbours in [0, 1023] stay inside int16.
func TestFilterIntra10SumFitsInt16(t *testing.T) {
	for mode := range filterIntraTaps {
		for out, taps := range filterIntraTaps[mode] {
			pos, neg := 0, 0
			for _, c := range taps {
				if c > 0 {
					pos += int(c)
				} else {
					neg += int(c)
				}
			}
			hi, lo := 1023*pos+8, 1023*neg+8
			if hi > 32767 || lo < -32768 {
				t.Fatalf("mode %d output %d: sums [%d, %d] leave int16", mode, out, lo, hi)
			}
		}
	}
}

func filterIntra16SIMDEdges(width, height int, max int, seed uint32) IntraEdges {
	above, s1 := samplesFor(width, uint16(max), seed)
	left, _ := samplesFor(height, uint16(max), s1)
	return IntraEdges{
		Above:              above,
		Left:               left,
		AboveLeft:          uint16((seed >> 3) % uint32(max+1)),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
}

// TestFilterIntraSIMDBinding asserts the arm64 SIMD build binds the Go-native
// filter-intra kernels, so the differential tests below exercise the production
// dispatch target.
func TestFilterIntraSIMDBinding(t *testing.T) {
	assertDispatchTarget(t, "predictFilterIntra8Impl", predictFilterIntra8Impl, "predictFilterIntraBlockDirect8TBL")
	assertDispatchTarget(t, "predictFilterIntra16Impl", predictFilterIntra16Impl, "predictFilterIntraBlockDirect10TBL")
}

// TestFilterIntra16SIMDMatchesPureGo executes the 16-bit SIMD kernel directly
// against the pure-Go reference for every mode, every valid transform size, both
// high bit depths and a spread of pseudo-random edges.
func TestFilterIntra16SIMDMatchesPureGo(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, max := range filterIntra16Depths {
				for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df, 0xdeadbeef} {
					edges := filterIntra16SIMDEdges(sz.width, sz.height, max, seed+uint32(mode)*131)
					base := makeDispatchBlock(sz.width, sz.height, 2)
					want := cloneBlock(base)
					predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
					for _, k := range filterIntra16Kernels {
						got := cloneBlock(base)
						k.fn(got, sz.width, sz.height, mode, edges, max)
						diffBlocks(t, "filter-intra16-simd/"+k.name, max, sz.width*2, sz.height, got, want)
					}
				}
			}
		}
	}
}

// TestFilterIntra16SIMDEdgeValues stresses the [0,max] clamp corners at both
// high bit depths, where the 32-bit tap sums saturate in both directions.
func TestFilterIntra16SIMDEdgeValues(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, max := range filterIntra16Depths {
				for _, pat := range []uint16{0, uint16(max), uint16(max / 2), uint16(max - 1)} {
					above := make([]uint16, sz.width)
					left := make([]uint16, sz.height)
					for i := range above {
						above[i] = pat
						if i%3 == 0 {
							above[i] = uint16(max) - pat
						}
					}
					for i := range left {
						left[i] = uint16(max) - pat
						if i%2 == 0 {
							left[i] = pat
						}
					}
					for _, al := range []uint16{0, uint16(max)} {
						edges := IntraEdges{
							Above:              above,
							Left:               left,
							AboveLeft:          al,
							AboveAvailable:     true,
							LeftAvailable:      true,
							AboveLeftAvailable: true,
						}
						base := makeDispatchBlock(sz.width, sz.height, 2)
						want := cloneBlock(base)
						predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
						for _, k := range filterIntra16Kernels {
							got := cloneBlock(base)
							k.fn(got, sz.width, sz.height, mode, edges, max)
							diffBlocks(t, "filter-intra16-simd-edge/"+k.name, max, sz.width*2, sz.height, got, want)
						}
					}
				}
			}
		}
	}
}

// TestFilterIntra16SIMDZeroAlloc protects the hot-path contract: the 16-bit SIMD
// predictor must not allocate per call.
func TestFilterIntra16SIMDZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	edges := filterIntra16SIMDEdges(w, h, 1023, 0x24)
	block := makeDispatchBlock(w, h, 2)
	fn := func() {
		predictFilterIntraBlockDirect10TBL(block, w, h, FilterIntraModePaeth, edges, 1023)
	}
	if allocs := testing.AllocsPerRun(1000, fn); allocs != 0 {
		t.Fatalf("filter-intra16 SIMD allocated %f times per call", allocs)
	}
}

func TestFilterIntra8TBLMatchesPureGo(t *testing.T) {
	patterns := []func(i int, seed uint32) uint16{
		func(i int, seed uint32) uint16 { return uint16((uint32(i)*2654435761 + seed) >> 7 & 0xff) },
		func(i int, seed uint32) uint16 { return 0 },
		func(i int, seed uint32) uint16 { return 255 },
		func(i int, seed uint32) uint16 { return uint16(255 * (i & 1)) },
	}
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for pi, ap := range patterns {
				for _, lp := range patterns {
					for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df} {
						above := make([]uint16, sz.width)
						left := make([]uint16, sz.height)
						for i := range above {
							above[i] = ap(i, seed)
						}
						for i := range left {
							left[i] = lp(i, seed*3+uint32(pi))
						}
						edges := IntraEdges{Above: above, Left: left, AboveLeft: uint16(seed & 0xff),
							AboveAvailable: true, LeftAvailable: true, AboveLeftAvailable: true}
						base := makeDispatchBlock(sz.width, sz.height, 1)
						got := cloneBlock(base)
						want := cloneBlock(base)
						predictFilterIntraBlockDirect8TBL(got, sz.width, sz.height, mode, edges, 0xff)
						predictFilterIntraBlockDirect8(want, sz.width, sz.height, mode, edges, 0xff)
						diffBlocks(t, "filter-intra-tbl", 0xff, sz.width, sz.height, got, want)
					}
				}
			}
		}
	}
}

func BenchmarkFilterIntra8Kernels(b *testing.B) {
	edges := filterIntraSIMDEdges(32, 32, 0xabcd)
	blk := makeDispatchBlock(32, 32, 1)
	b.Run("tbl", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8TBL(blk, 32, 32, FilterIntraModeD157, edges, 0xff)
		}
	})
	b.Run("bcast", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8SIMD(blk, 32, 32, FilterIntraModeD157, edges, 0xff)
		}
	})
	edges8 := filterIntraSIMDEdges(8, 8, 0xabcd)
	b.Run("tbl8x8", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8TBL(blk, 8, 8, FilterIntraModeD157, edges8, 0xff)
		}
	})
	b.Run("bcast8x8", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8SIMD(blk, 8, 8, FilterIntraModeD157, edges8, 0xff)
		}
	})
}

func TestGoSIMDDispatchProbe(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("arm64 NEON not detected")
	}
	assertDispatchTarget(t, "applyCFLImpl", applyCFLImpl, "applyCFLSIMD")
	assertDispatchTarget(t, "subsampleLuma8Impl", subsampleLuma8Impl, "subsampleLuma8SIMD")
	assertDispatchTarget(t, "subsampleLuma16Impl", subsampleLuma16Impl, "subsampleLuma16SIMD")
	assertDispatchTarget(t, "subtractCFLAverageImpl", subtractCFLAverageImpl, "subtractCFLAverageSIMD")

	assertDispatchTarget(t, "predictPaethImpl", predictPaethImpl, "predictPaethSIMD")
	assertDispatchTarget(t, "predictSmoothImpl", predictSmoothImpl, "predictSmoothSIMD")
	assertDispatchTarget(t, "predictSmoothVerticalImpl", predictSmoothVerticalImpl, "predictSmoothVerticalSIMD")
	assertDispatchTarget(t, "predictSmoothHorizontalImpl", predictSmoothHorizontalImpl, "predictSmoothHorizontalSIMD")
	assertDispatchTarget(t, "sumSamplesImpl", sumSamplesImpl, "sumSamplesSIMD")
	assertDispatchTarget(t, "dirRowInterp8Impl", dirRowInterp8Impl, "dirRowInterp8SIMD")
	assertDispatchTarget(t, "dirAboveRun8Impl", dirAboveRun8Impl, "dirAboveRun8SIMD")
	assertDispatchTarget(t, "dirLeftCol8Impl", dirLeftCol8Impl, "dirLeftCol8SIMD")
}

func BenchmarkCFLSubsample8Arm64(b *testing.B) {
	input := make([]uint8, 32*32)
	output := make([]uint16, CFLBufSquare)
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for i := range input {
		input[i] = uint8(rnd.pseudoUniform(256))
	}
	for _, mode := range []struct {
		name       string
		subX, subY bool
		outW, outH int
	}{
		{"444", false, false, 32, 32},
		{"422", true, false, 16, 32},
		{"420", true, true, 16, 16},
	} {
		for _, v := range []struct {
			name string
			fn   subsampleLuma8Func
		}{
			{"SIMD", subsampleLuma8SIMD},
			{"PureGo", subsampleLuma8PureGo},
		} {
			b.Run(fmt.Sprintf("%s/%s", mode.name, v.name), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					v.fn(output, input, 32, 32, 32, mode.outW, mode.outH, mode.subX, mode.subY)
				}
			})
		}
	}
}

func BenchmarkCFLSubtractAverageArm64(b *testing.B) {
	src := make([]uint16, CFLBufSquare)
	dst := make([]int16, CFLBufSquare)
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 15))
	}
	log2, _ := log2PowerOfTwoInt(32 * 32)
	for _, v := range []struct {
		name string
		fn   subtractCFLAverageFunc
	}{
		{"SIMD", subtractCFLAverageSIMD},
		{"PureGo", subtractCFLAveragePureGo},
	} {
		b.Run(v.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				v.fn(src, dst, 32, 32, log2)
			}
		})
	}
}

func BenchmarkCFLSubsample16Arm64(b *testing.B) {
	const width, height = 32, 32
	stride := width + 3
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for _, depth := range []uint8{10, 12} {
		max := uint16((1 << depth) - 1)
		input := make([]uint16, stride*height)
		for i := range input {
			input[i] = uint16(rnd.pseudoUniform(int(max) + 1))
		}
		for _, mode := range []struct {
			name       string
			subX, subY bool
			outW, outH int
		}{
			{"444", false, false, width, height},
			{"422", true, false, width / 2, height},
			{"420", true, true, width / 2, height / 2},
		} {
			for _, variant := range []struct {
				name string
				fn   subsampleLuma16Func
			}{
				{"SIMD", subsampleLuma16SIMD},
				{"PureGo", subsampleLuma16PureGo},
			} {
				b.Run(fmt.Sprintf("%dbit/%s/%s", depth, mode.name, variant.name), func(b *testing.B) {
					output := make([]uint16, CFLBufSquare)
					b.ReportAllocs()
					for b.Loop() {
						if err := variant.fn(output, input, stride, width, height, mode.outW, mode.outH, mode.subX, mode.subY, max); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
