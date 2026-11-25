//go:build goexperiment.simd && (arm64 || amd64) && !purego

package transform

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"simd/archsimd"
	"slices"
	"testing"
)

// BenchmarkLanes4 times the four-lane DCT32/DCT64 adapters on the shapes the
// row and column passes hand them: a 64-wide column buffer and four rows.
func BenchmarkLanes4(b *testing.B) {
	const stride = 64
	fill := func(s []int32) {
		for i := range s {
			s[i] = int32((i*1103-4000)&0x3fff) - 0x2000
		}
	}
	col := make([]int32, 63*stride+4)
	rows := make([]int32, 4*dct64Size)
	lo, hi := int32(-1<<17), int32(1<<17-1)
	run := func(name string, fn func()) {
		b.Run(name, func(b *testing.B) {
			fill(col)
			fill(rows)
			for b.Loop() {
				fn()
			}
		})
	}
	r := func(n, i int) []int32 { return rows[i*n : (i+1)*n] }
	run("dct64col", func() { inverseDCT64Col4SIMD(col, stride, lo, hi) })
	run("dct64lowcol", func() { inverseDCT64LowCol4SIMD(col, stride, 32, lo, hi) })
	run("dct64row", func() { inverseDCT64Row4SIMD(r(64, 0), r(64, 1), r(64, 2), r(64, 3), lo, hi) })
	run("dct64rowlow", func() { inverseDCT64Row4LowSIMD(r(64, 0), r(64, 1), r(64, 2), r(64, 3), 32, lo, hi) })
	run("dct32col", func() { inverseDCT32Lanes4ColSIMD(col, stride, lo, hi) })
	run("dct32row", func() { inverseDCT32Lanes4RowSIMD(r(32, 0), r(32, 1), r(32, 2), r(32, 3), lo, hi) })
}

func fourLaneSIMDAvailable() bool {
	return runtime.GOARCH == "arm64" || archsimd.X86.AVX2()
}

func TestNativeDCTLanes4Parity(t *testing.T) {
	if !fourLaneSIMDAvailable() {
		t.Skip("native SIMD unavailable")
	}
	random := rand.New(rand.NewSource(0x32644))
	clamps := append(slices.Clone(stageClampSets),
		[2]int32{-itxNarrowBound, itxNarrowBound - 1},
		[2]int32{-itxNarrowBound - 1, itxNarrowBound},
		[2]int32{-itxWideBound, itxWideBound - 1},
		[2]int32{-3, 5},
		[2]int32{-itxWideBound - 1, itxWideBound},
		[2]int32{math.MinInt32, math.MaxInt32})
	for _, kernel := range []struct {
		name         string
		size         int
		column       func([]int32, int, int32, int32)
		row          func([]int32, []int32, []int32, []int32, int32, int32)
		scalarColumn func([]int32, int, int32, int32)
		scalarRow    func([]int32, []int32, []int32, []int32, int32, int32)
	}{
		{"DCT32", 32, inverseDCT32Lanes4ColSIMD, inverseDCT32Lanes4RowSIMD, inverseDCT32Col4PureGo, inverseDCT32Row4PureGo},
		{"DCT64", 64, inverseDCT64Col4SIMD, inverseDCT64Row4SIMD, inverseDCT64Col4PureGo, inverseDCT64Row4PureGo},
	} {
		t.Run(kernel.name, func(t *testing.T) {
			for _, clamp := range clamps {
				for _, stride := range []int{4, 5, 11, 64} {
					for pattern := 0; pattern < 4; pattern++ {
						length := (kernel.size-1)*stride + 4
						actual := make([]int32, length+6)
						for index := range actual {
							actual[index] = 0x12345678
						}
						for index := 0; index < kernel.size; index++ {
							for lane := 0; lane < 4; lane++ {
								value := clamp[0] + int32(random.Int63n(int64(clamp[1])-int64(clamp[0])+1))
								if pattern == 1 {
									value = clamp[(index+lane)%2]
								} else if pattern == 2 {
									value = 0
									if index == lane {
										value = clamp[1]
									}
								} else if pattern == 3 {
									value = int32((index+lane)%3 - 1)
								}
								actual[3+index*stride+lane] = value
							}
						}
						expected := slices.Clone(actual)
						kernel.scalarColumn(expected[3:3+length], stride, clamp[0], clamp[1])
						kernel.column(actual[3:3+length], stride, clamp[0], clamp[1])
						if !slices.Equal(actual, expected) {
							t.Fatalf("column clamp=%v stride=%d pattern=%d mismatch", clamp, stride, pattern)
						}
						rows := make([]int32, 4*(kernel.size+2))
						for index := range rows {
							rows[index] = clamp[0] + int32(random.Int63n(int64(clamp[1])-int64(clamp[0])+1))
						}
						expectedRows := slices.Clone(rows)
						rowStride := kernel.size + 2
						kernel.scalarRow(expectedRows[1:1+kernel.size], expectedRows[rowStride+1:rowStride+1+kernel.size], expectedRows[2*rowStride+1:2*rowStride+1+kernel.size], expectedRows[3*rowStride+1:3*rowStride+1+kernel.size], clamp[0], clamp[1])
						kernel.row(rows[1:1+kernel.size], rows[rowStride+1:rowStride+1+kernel.size], rows[2*rowStride+1:2*rowStride+1+kernel.size], rows[3*rowStride+1:3*rowStride+1+kernel.size], clamp[0], clamp[1])
						if !slices.Equal(rows, expectedRows) {
							t.Fatalf("row clamp=%v mismatch", clamp)
						}
					}
				}
			}
		})
	}
}

func TestNativeDCTLanes4NoAlloc(t *testing.T) {
	if !fourLaneSIMDAvailable() {
		t.Skip("native SIMD unavailable")
	}
	for _, kernel := range []struct {
		name   string
		size   int
		column func([]int32, int, int32, int32)
		row    func([]int32, []int32, []int32, []int32, int32, int32)
	}{
		{"DCT32", 32, inverseDCT32Lanes4ColSIMD, inverseDCT32Lanes4RowSIMD},
		{"DCT64", 64, inverseDCT64Col4SIMD, inverseDCT64Row4SIMD},
	} {
		t.Run(kernel.name, func(t *testing.T) {
			buffer := make([]int32, kernel.size*4)
			if allocations := testing.AllocsPerRun(20, func() {
				kernel.column(buffer, 4, -itxNarrowBound, itxNarrowBound-1)
				kernel.row(buffer[:kernel.size], buffer[kernel.size:2*kernel.size], buffer[2*kernel.size:3*kernel.size], buffer[3*kernel.size:], -itxWideBound, itxWideBound-1)
			}); allocations != 0 {
				t.Fatalf("allocated %g objects per row/column pair", allocations)
			}
		})
	}
}

func TestAMD64DCTLanes4Dispatch(t *testing.T) {
	if runtime.GOARCH != "amd64" || !archsimd.X86.AVX2() {
		t.Skip("AMD64 AVX2 unavailable")
	}
	for _, binding := range []struct{ actual, expected any }{
		{inverseDCT32Col4Impl, inverseDCT32Lanes4ColSIMD},
		{inverseDCT32Row4Impl, inverseDCT32Lanes4RowSIMD},
		{inverseDCT64Col4Impl, inverseDCT64Col4SIMD},
		{inverseDCT64Row4Impl, inverseDCT64Row4SIMD},
	} {
		if reflect.ValueOf(binding.actual).Pointer() != reflect.ValueOf(binding.expected).Pointer() {
			t.Fatal("four-lane dispatch is not bound to Go SIMD")
		}
	}
}

func TestNativeDCTLanes4ShortRows(t *testing.T) {
	if !fourLaneSIMDAvailable() {
		t.Skip("native SIMD unavailable")
	}
	for _, kernel := range []struct {
		name   string
		size   int
		row    func([]int32, []int32, []int32, []int32, int32, int32)
		scalar func([]int32, []int32, []int32, []int32, int32, int32)
	}{
		{"DCT32", 32, inverseDCT32Lanes4RowSIMD, inverseDCT32Row4PureGo},
		{"DCT64", 64, inverseDCT64Row4SIMD, inverseDCT64Row4PureGo},
	} {
		t.Run(kernel.name, func(t *testing.T) {
			for shortLane := 0; shortLane < 4; shortLane++ {
				for _, length := range []int{0, 1, kernel.size - 1, kernel.size, kernel.size + 3} {
					var actual, expected [4][]int32
					for lane := range actual {
						rowLength := kernel.size + 3
						if lane == shortLane {
							rowLength = length
						}
						actual[lane] = make([]int32, rowLength)
						for index := range actual[lane] {
							actual[lane][index] = int32(index*17 + lane*31 - 100)
						}
						expected[lane] = slices.Clone(actual[lane])
					}
					kernel.scalar(expected[0], expected[1], expected[2], expected[3], -32768, 32767)
					kernel.row(actual[0], actual[1], actual[2], actual[3], -32768, 32767)
					for lane := range actual {
						if !slices.Equal(actual[lane], expected[lane]) {
							t.Fatalf("short lane=%d length=%d output lane=%d mismatch", shortLane, length, lane)
						}
					}
				}
			}
		})
	}
}

func BenchmarkNativeDCT32Lanes4(b *testing.B) {
	if !fourLaneSIMDAvailable() {
		b.Skip("native SIMD unavailable")
	}
	for _, bounds := range [][2]int32{{-itxNarrowBound, itxNarrowBound - 1}, {-itxWideBound, itxWideBound - 1}} {
		b.Run(fmt.Sprint(bounds), func(b *testing.B) {
			buffer := make([]int32, 32*4)
			b.ReportAllocs()
			for b.Loop() {
				for index := range buffer {
					buffer[index] = int32(index%251 - 125)
				}
				inverseDCT32Lanes4ColSIMD(buffer, 4, bounds[0], bounds[1])
			}
		})
	}
}

func fwd64SIMDAvailable() bool { return runtime.GOARCH == "arm64" || archsimd.X86.AVX2() }

// TestForwardDCT64x64SIMDMatchesScalar differential-tests the SVT-shaped
// 64x64 kernel (highbd_fdct64_N2_x4_neon passes) against the libaom scalar
// reference over 8-bit residuals, extremes included.
func TestForwardDCT64x64SIMDMatchesScalar(t *testing.T) {
	if !fwd64SIMDAvailable() {
		t.Skip("no vector backend")
	}
	rng := rand.New(rand.NewSource(64))
	var residual [64 * 64]int16
	for trial := 0; trial < 400; trial++ {
		switch trial % 5 {
		case 0:
			for i := range residual {
				residual[i] = int16(rng.Intn(511) - 255)
			}
		case 1:
			v := int16(255)
			if trial&8 != 0 {
				v = -255
			}
			for i := range residual {
				residual[i] = v
			}
		case 2:
			for i := range residual {
				if (i/64+i%64+trial)&1 == 0 {
					residual[i] = 255
				} else {
					residual[i] = -255
				}
			}
		case 3:
			for i := range residual {
				residual[i] = int16(rng.Intn(21) - 10)
			}
		default:
			for i := range residual {
				if rng.Intn(2) == 0 {
					residual[i] = 255
				} else {
					residual[i] = -255
				}
			}
		}
		var want, got [32 * 32]int32
		for i := range got {
			got[i] = -12345
		}
		forwardDCT64x64PureGo(want[:], 32, residual[:], 64)
		forwardDCT64x64SIMD(got[:], 32, residual[:], 64)
		if want != got {
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("trial %d: coeff[col %d][row %d] = %d, want %d", trial, i/32, i%32, got[i], want[i])
				}
			}
		}
		const wideStride = 35
		var wide [32 * wideStride]int32
		forwardDCT64x64SIMD(wide[:], wideStride, residual[:], 64)
		for k := range 32 {
			for r := range 32 {
				if wide[k*wideStride+r] != want[k*32+r] {
					t.Fatalf("trial %d stride %d: coeff[col %d][row %d] = %d, want %d", trial, wideStride, k, r, wide[k*wideStride+r], want[k*32+r])
				}
			}
		}
	}
}

func BenchmarkForwardDCT64x64SIMD(b *testing.B) {
	if !fwd64SIMDAvailable() {
		b.Skip("no vector backend")
	}
	var residual [64 * 64]int16
	rng := rand.New(rand.NewSource(1))
	for i := range residual {
		residual[i] = int16(rng.Intn(511) - 255)
	}
	var coeff [32 * 32]int32
	b.ResetTimer()
	for b.Loop() {
		forwardDCT64x64SIMD(coeff[:], 32, residual[:], 64)
	}
}

// TestForwardBlock8x8SIMDMatchesPureGo checks the Go SIMD 8x8 kernels, called
// directly, against the scalar oracle over random 8-bit residuals, the
// extreme sign patterns, and non-packed strides.
func TestForwardBlock8x8SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(401))
	const resStride, coeffStride = 19, 13
	kernels := []struct {
		name string
		simd func([]int32, int, []int16, int, []int32)
		pure func([]int32, int, []int16, int, []int32)
	}{
		{"DCT_DCT", func(c []int32, cs int, r []int16, rs int, _ []int32) { forwardDCT8x8SIMD(c, cs, r, rs) },
			func(c []int32, cs int, r []int16, rs int, _ []int32) { forwardDCT8x8PureGo(c, cs, r, rs) }},
		{"ADST_DCT", forwardBlock8x8ADSTDCTSIMD, forwardBlock8x8ADSTDCTPureGo},
		{"DCT_ADST", forwardBlock8x8DCTADSTSIMD, forwardBlock8x8DCTADSTPureGo},
		{"ADST_ADST", forwardBlock8x8ADSTADSTSIMD, forwardBlock8x8ADSTADSTPureGo},
	}
	extremes := []func(r, c int) int16{
		func(r, c int) int16 { return 255 },
		func(r, c int) int16 { return -255 },
		func(r, c int) int16 {
			if (r+c)&1 == 0 {
				return 255
			}
			return -255
		},
		func(r, c int) int16 {
			if r&1 == 0 {
				return 255
			}
			return -255
		},
	}
	for _, k := range kernels {
		residual := make([]int16, resStride*8)
		got := make([]int32, coeffStride*8)
		want := make([]int32, coeffStride*8)
		var scratch [64]int32
		check := func(trial int) {
			t.Helper()
			clear(got)
			clear(want)
			k.simd(got, coeffStride, residual, resStride, scratch[:])
			k.pure(want, coeffStride, residual, resStride, scratch[:])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s trial %d: coeff[%d] simd %d want %d", k.name, trial, i, got[i], want[i])
				}
			}
		}
		for _, pat := range extremes {
			for r := range 8 {
				for c := range 8 {
					residual[r*resStride+c] = pat(r, c)
				}
			}
			check(-1)
		}
		for trial := range 3000 {
			for i := range residual {
				residual[i] = int16(rng.Intn(511)) - 255
			}
			check(trial)
		}
	}
}

// TestForwardDCTSquareSIMDMatchesPureGo checks the Go SIMD 16x16 and 32x32 DCT
// kernels against the scalar oracle over random and extreme 8-bit residuals.
func TestForwardDCTSquareSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(83))
	cases := []struct {
		name string
		side int
		simd func([]int32, int, []int16, int)
		pure func([]int32, int, []int16, int)
	}{
		{"16x16", 16, forwardDCT16x16SIMD, forwardDCT16x16PureGo},
		{"32x32", 32, forwardDCT32x32SIMD, forwardDCT32x32PureGo},
	}
	for _, tc := range cases {
		for _, coeffExtra := range []int{0, 3} {
			testForwardDCTSquareSIMD(t, rng, tc.name, tc.side, coeffExtra, tc.simd, tc.pure)
		}
	}
}

// testForwardDCTSquareSIMD runs one kernel pair at coefficient stride
// side+coeffExtra (0 takes the in-place path, others the copy-out one).
func testForwardDCTSquareSIMD(t *testing.T, rng *rand.Rand, name string, side, coeffExtra int, simd, pure func([]int32, int, []int16, int)) {
	const resExtra = 5
	resStride := side + resExtra
	coeffStride := side + coeffExtra
	residual := make([]int16, resStride*side)
	got := make([]int32, coeffStride*side)
	want := make([]int32, coeffStride*side)
	patterns := []func(r, c int) int16{
		func(r, c int) int16 { return 255 },
		func(r, c int) int16 { return -255 },
		func(r, c int) int16 {
			if (r+c)&1 == 0 {
				return 255
			}
			return -255
		},
	}
	check := func(trial int) {
		t.Helper()
		clear(got)
		clear(want)
		simd(got, coeffStride, residual, resStride)
		pure(want, coeffStride, residual, resStride)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s trial %d: coeff[%d] simd %d want %d", name, trial, i, got[i], want[i])
			}
		}
	}
	for _, pat := range patterns {
		for r := range side {
			for c := range side {
				residual[r*resStride+c] = pat(r, c)
			}
		}
		check(-1)
	}
	for trial := range 400 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		check(trial)
	}
}

// TestForwardDCT8x8SIMDZeroAlloc keeps the 8x8 SIMD driver allocation-free.
func TestForwardDCT8x8SIMDZeroAlloc(t *testing.T) {
	var residual [64]int16
	for i := range residual {
		residual[i] = int16(i%511) - 255
	}
	var coeff [64]int32
	var scratch [64]int32
	allocs := testing.AllocsPerRun(100, func() {
		forwardDCT8x8SIMDGuarded(coeff[:], 8, residual[:], 8)
		forwardBlock8x8ADSTDCTSIMDGuarded(coeff[:], 8, residual[:], 8, scratch[:])
	})
	if allocs != 0 {
		t.Fatalf("SIMD 8x8 forward kernels allocated %v objects/run, want 0", allocs)
	}
}

// invGlueLCG is a tiny deterministic generator so the differential tests are
// reproducible without pulling in math/rand.
type invGlueLCG struct{ s uint64 }

func (g *invGlueLCG) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s >> 11
}

// stage-range-representative signed int32 in [-1<<21, 1<<21).
func (g *invGlueLCG) stageVal() int32 {
	return int32(g.next()%(1<<22)) - (1 << 21)
}

// TestClampRoundSIMDMatchesPureGo runs the Go SIMD mid-pass round-and-clamp and
// the pure-Go reference over identical inputs and asserts lane-for-lane
// equality. It calls clampRoundSIMD directly so the vector path executes
// regardless of the dispatch binding. It sweeps shift 0 (clamp-only) and several
// positive shifts, with [min, max] windows that force both clamps, over
// multiple-of-eight lengths plus lengths that exercise the pure-Go tail.
func TestClampRoundSIMDMatchesPureGo(t *testing.T) {
	shifts := []int{0, 1, 2, 3, 4, 6, 8, 12, 31}
	bounds := []struct{ min, max int32 }{
		{-(1 << 18), (1 << 18) - 1},
		{-(1 << 10), (1 << 10) - 1},
		{-4096, 4095},
		{0, 255},
		{-(1 << 31), (1 << 31) - 1},
	}
	lengths := []int{8, 16, 24, 64, 256, 13, 4, 7}

	for _, shift := range shifts {
		for _, b := range bounds {
			for _, n := range lengths {
				g := invGlueLCG{s: uint64(shift*7919 + int(b.max&0xffff)*31 + n*17 + 1)}
				got := make([]int32, n)
				want := make([]int32, n)
				edges := []int32{math.MaxInt32, math.MinInt32, math.MaxInt32 - 1, math.MinInt32 + 1, -1, 0, 1, 1 << 30}
				for i := range got {
					v := g.stageVal()
					if i%5 == 0 {
						v = int32(g.next()) // full-range int32 values
					}
					if i < len(edges) {
						v = edges[i] // rounding-overflow boundaries, every shift
					}
					got[i] = v
					want[i] = v
				}
				clampRoundSIMD(got, shift, b.min, b.max)
				clampRoundPureGo(want, shift, b.min, b.max)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("shift=%d min=%d max=%d n=%d lane %d: simd=%d ref=%d",
							shift, b.min, b.max, n, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestNarrowStoreSIMDMatchesPureGo runs the Go SIMD final narrowing store and the
// pure-Go reference over identical inputs and asserts element-for-element
// equality (including the past-width dst region, seeded with a sentinel to catch
// stray writes). It sweeps width four and multiples of eight (plus a non-4,
// non-multiple-of-8 width for the tail fallback), with source values large
// enough to force int16 saturation on both sides.
func TestNarrowStoreSIMDMatchesPureGo(t *testing.T) {
	widths := []int{4, 8, 16, 24, 32, 64, 12}
	heights := []int{1, 2, 4, 7}

	for _, width := range widths {
		for _, height := range heights {
			g := invGlueLCG{s: uint64(width*131 + height*29 + 3)}
			dstStride := width + 5
			scratch := make([]int32, width*height)
			for i := range scratch {
				switch g.next() & 3 {
				case 0:
					// Large magnitude to force int16 saturation after >>4.
					scratch[i] = int32(g.next()%(1<<25)) - (1 << 24)
				default:
					scratch[i] = int32(g.next()%(1<<19)) - (1 << 18)
				}
			}

			gotDst := make([]int16, dstStride*height)
			wantDst := make([]int16, dstStride*height)
			for i := range gotDst {
				gotDst[i] = -12345
				wantDst[i] = -12345
			}

			narrowStoreSIMD(gotDst, dstStride, scratch, width, height)
			narrowStorePureGo(wantDst, dstStride, scratch, width, height)

			for i := range gotDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("w=%d h=%d elem %d: simd=%d ref=%d", width, height, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}

// TestInvGlueDispatchBindsSIMD asserts the Go SIMD glue kernels are the bound
// dispatch slots whenever the host has the vector ISA (arm64: always; amd64:
// AVX2).
func TestInvGlueDispatchBindsSIMD(t *testing.T) {
	if !invGlueSIMDAvailable() {
		t.Skip("host lacks the vector ISA the glue kernels need")
	}
	if reflect.ValueOf(clampRoundImpl).Pointer() != reflect.ValueOf(clampRoundSIMD).Pointer() {
		t.Fatal("clampRoundImpl is not bound to clampRoundSIMD")
	}
	if reflect.ValueOf(narrowStoreImpl).Pointer() != reflect.ValueOf(narrowStoreSIMD).Pointer() {
		t.Fatal("narrowStoreImpl is not bound to narrowStoreSIMD")
	}
}

func TestInvGlueSIMDIsZeroAlloc(t *testing.T) {
	scratch := make([]int32, 1024)
	for i := range scratch {
		scratch[i] = int32(i%2048) - 1024
	}
	dst := make([]int16, 32*32)
	if a := testing.AllocsPerRun(1000, func() {
		clampRoundSIMD(scratch, 4, -4096, 4095)
	}); a != 0 {
		t.Fatalf("clampRoundSIMD allocated: %f", a)
	}
	if a := testing.AllocsPerRun(1000, func() {
		narrowStoreSIMD(dst, 32, scratch, 32, 32)
	}); a != 0 {
		t.Fatalf("narrowStoreSIMD allocated: %f", a)
	}
}
