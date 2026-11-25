//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"encoding/binary"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"simd/archsimd"
	"strings"
	"testing"
)

func benchDCT8x4(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 4
	buf := make([]int32, 8*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<13) - (1 << 12))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 14), (1<<14)-1)
	}
}

func BenchmarkDCT8x4_ASMCol4(b *testing.B) {
	benchDCT8x4(b, inverseDCT8Col4SIMDAdapter)
}

func BenchmarkDCT8x4_ASMCol2(b *testing.B) {
	benchDCT8x4(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[2:], s, mn, mx)
	})
}

func benchDCT16x4(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 4
	buf := make([]int32, 16*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<13) - (1 << 12))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 14), (1<<14)-1)
	}
}

func BenchmarkDCT16x4_ASMCol4(b *testing.B) {
	benchDCT16x4(b, inverseDCT16Col4SIMDAdapter)
}

func BenchmarkDCT16x4_ASMCol2(b *testing.B) {
	benchDCT16x4(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[2:], s, mn, mx)
	})
}

func TestInt16ColumnSIMDInputGuardMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16b001))
	for _, height := range []int{8, 16, 32, 64} {
		width := height
		limit := int16ColumnSIMDInputBound(height)
		clamps := [][2]int32{{minInt16, maxInt16}, {0, maxInt16}, {minInt16, 0}, {1, maxInt16}, {minInt16, -1}, {0, 0}}
		for _, clamp := range clamps {
			for _, pattern := range []string{"random", "at-bound", "over-bound", "min-int16", "max-int16"} {
				buf := make([]int16, width*height)
				for i := range buf {
					switch pattern {
					case "random":
						buf[i] = int16(rng.Uint32())
					case "at-bound":
						if i&1 == 0 {
							buf[i] = int16(limit)
						} else {
							buf[i] = int16(-limit)
						}
					case "over-bound":
						if i&1 == 0 {
							buf[i] = int16(limit + 1)
						} else {
							buf[i] = int16(-limit - 1)
						}
					case "min-int16":
						buf[i] = minInt16
					case "max-int16":
						buf[i] = maxInt16
					}
				}
				got := int16ColumnSIMDInputSafeSIMD(buf, width, height, clamp[0], clamp[1])
				want := int16ColumnSIMDInputSafeScalar(buf, width, height, clamp[0], clamp[1])
				if got != want {
					t.Fatalf("height=%d clamp=[%d,%d] pattern=%s SIMD=%t scalar=%t", height, clamp[0], clamp[1], pattern, got, want)
				}
			}
		}
	}

	invalid := []struct {
		buf           []int16
		width, height int
		min, max      int32
	}{
		{nil, 8, 16, minInt16, maxInt16},
		{make([]int16, 8*16), 8, 16, 1, 0},
		{make([]int16, 8*16), 8, 16, -32769, maxInt16},
		{make([]int16, 8*16), 8, 16, minInt16, 32768},
	}
	for _, tc := range invalid {
		got := int16ColumnSIMDInputSafeSIMD(tc.buf, tc.width, tc.height, tc.min, tc.max)
		want := int16ColumnSIMDInputSafeScalar(tc.buf, tc.width, tc.height, tc.min, tc.max)
		if got != want {
			t.Fatalf("invalid input width=%d height=%d clamp=[%d,%d] SIMD=%t scalar=%t", tc.width, tc.height, tc.min, tc.max, got, want)
		}
	}
}

func TestRoundShiftNarrowInt32x4ToInt16x8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x51f7))
	values := []int32{
		-1 << 31, -1<<31 + 1, -1 << 30, -32769, -32768, -3, -2, -1,
		0, 1, 2, 3, 32767, 32768, 1<<30 - 1, 1<<31 - 1,
	}
	for range 1024 {
		values = append(values, int32(rng.Uint32()))
	}
	for _, shift := range []uint8{1, 2, 8, 11, 12, 13, 31} {
		for start := 0; start < len(values); start += 8 {
			var lanes [8]int32
			for i := range lanes {
				if start+i < len(values) {
					lanes[i] = values[start+i]
				}
			}
			lo := archsimd.LoadInt32x4Array((*[4]int32)(lanes[:4]))
			hi := archsimd.LoadInt32x4Array((*[4]int32)(lanes[4:]))
			var got [8]int16
			roundShiftNarrowInt32x4ToInt16x8(lo, hi, shift).StoreArray(&got)
			for i, value := range lanes {
				want := clipInt16(int32(roundShift(int64(value), int(shift))))
				if got[i] != want {
					t.Fatalf("shift=%d value=%d SIMD=%d scalar=%d", shift, value, got[i], want)
				}
			}
		}
	}
}

func benchDCT8x8(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 8
	buf := make([]int32, 8*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 12), (1<<12)-1)
	}
}

func BenchmarkDCT8x8_ASMCol4(b *testing.B) {
	benchDCT8x8(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col4SIMDAdapter(buf, s, mn, mx)
		inverseDCT8Col4SIMDAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT8x8_ASMCol2(b *testing.B) {
	benchDCT8x8(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[2:], s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[4:], s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[6:], s, mn, mx)
	})
}

func TestInverseDCT8Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16b))
	min, max := int32(minInt16), int32(maxInt16)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 8*stride)
		for k := 0; k < 8; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(rng.Uint32())
			}
		}
		if iter == 0 {
			clear(a)
			a[0], a[4*stride] = 30000, 30000
		} else if iter == 1 {
			for k := 0; k < 8; k++ {
				for col := 0; col < 8; col++ {
					if (k+col)&1 == 0 {
						a[k*stride+col] = minInt16
					} else {
						a[k*stride+col] = maxInt16
					}
				}
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT8(a[col:], stride, min, max) // T=int16 via inference
		}
		inverseDCT8Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func TestInverseDCT8Col8SIMD16FullRangeClampIntervals(t *testing.T) {
	clamps := [][2]int32{{minInt16, maxInt16}, {1, maxInt16}, {minInt16, -1}, {-1024, 2047}}
	patterns := [][8]int16{
		{30000, -30000, 12000, -8000, 30000, 24000, -16000, -30000},
		{30000, minInt16, -30000, maxInt16, -30000, maxInt16, 30000, minInt16},
		{minInt16, maxInt16, minInt16, maxInt16, maxInt16, minInt16, maxInt16, minInt16},
	}
	for _, clamp := range clamps {
		for patternIndex, pattern := range patterns {
			const stride = 8
			input := make([]int16, 8*stride)
			for row, value := range pattern {
				for col := 0; col < 8; col++ {
					input[row*stride+col] = value
				}
			}
			want, got := append([]int16(nil), input...), append([]int16(nil), input...)
			for col := 0; col < 8; col++ {
				inverseDCT8(want[col:], stride, clamp[0], clamp[1])
			}
			inverseDCT8Col8SIMD16(got, stride, clamp[0], clamp[1])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("clamp=[%d,%d] pattern=%d index=%d: scalar=%d SIMD=%d", clamp[0], clamp[1], patternIndex, i, want[i], got[i])
				}
			}
		}
	}
}

func BenchmarkDCT8x8_Int16Buf(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	const stride = 8
	buf := make([]int16, 8*stride+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int16, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		inverseDCT8Col8SIMD16(work, stride, -(1 << 12), (1<<12)-1)
	}
}

func TestInverseDCT16Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1616))
	min, max := -int16ColumnSIMDInputBound(16), int16ColumnSIMDInputBound(16)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 16*stride)
		for k := 0; k < 16; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(min + int32(rng.Int63n(int64(max)-int64(min)+1)))
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT16(a[col:], stride, min, max)
		}
		inverseDCT16Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func BenchmarkDCT16x8_Int16Buf(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	const stride = 8
	buf := make([]int16, 16*stride+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int16, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		inverseDCT16Col8SIMD16(work, stride, -(1 << 12), (1<<12)-1)
	}
}
func BenchmarkDCT16x8_ASMCol4(b *testing.B) {
	benchDCTx8ASM(b, 16, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col4SIMDAdapter(buf, s, mn, mx)
		inverseDCT16Col4SIMDAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT16x8_ASMCol2(b *testing.B) {
	benchDCTx8ASM(b, 16, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[2:], s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[4:], s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[6:], s, mn, mx)
	})
}

func TestInverseDCT32Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x3232))
	min, max := -int16ColumnSIMDInputBound(32), int16ColumnSIMDInputBound(32)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 32*stride)
		for k := 0; k < 32; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(min + int32(rng.Int63n(int64(max)-int64(min)+1)))
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT32(a[col:], stride, min, max)
		}
		inverseDCT32Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func TestInverseDCT64Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6464))
	min, max := -int16ColumnSIMDInputBound(64), int16ColumnSIMDInputBound(64)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 64*stride)
		for k := 0; k < 64; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(min + int32(rng.Int63n(int64(max)-int64(min)+1)))
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT64(a[col:], stride, min, max)
		}
		inverseDCT64Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func benchDCTx8Int16(b *testing.B, n int, fn func([]int16, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	stride := 8
	buf := make([]int16, n*stride+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int16, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 12), (1<<12)-1)
	}
}
func BenchmarkDCT32x8_Int16Buf(b *testing.B) { benchDCTx8Int16(b, 32, inverseDCT32Col8SIMD16) }
func BenchmarkDCT64x8_Int16Buf(b *testing.B) { benchDCTx8Int16(b, 64, inverseDCT64Col8SIMD16) }
func benchDCTx8ASM(b *testing.B, n int, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	stride := 8
	buf := make([]int32, n*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 12), (1<<12)-1)
	}
}
func BenchmarkDCT32x8_ASM(b *testing.B) {
	benchDCTx8ASM(b, 32, func(buf []int32, s int, mn, mx int32) {
		inverseDCT32Col4SIMDAdapter(buf, s, mn, mx)
		inverseDCT32Col4SIMDAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT64x8_PureGo(b *testing.B) {
	benchDCTx8ASM(b, 64, func(buf []int32, s int, mn, mx int32) {
		inverseDCT64Col4PureGo(buf, s, mn, mx)
		inverseDCT64Col4PureGo(buf[4:], s, mn, mx)
	})
}

// colPass2TestFuncs returns the arm64 adapters directly so the differential test
// always exercises every arm64 column kernel for bit-exactness.
func colPass2TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT8", dct8Size, inverseDCT8Col2SIMDAdapter, inverseDCT8Col2PureGo},
		{"DCT16", dct16Size, inverseDCT16Col2SIMDAdapter, inverseDCT16Col2PureGo},
		{"DCT32", dct32Size, inverseDCT32Col2SIMDAdapter, inverseDCT32Col2PureGo},
	}
}

// colPass4TestFuncs returns the arm64 four-column adapters directly so the
// differential test always exercises the four-column kernels for
// bit-exactness.
func colPass4TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT4Col4", dct4Size, inverseDCT4Col4SIMDAdapter, inverseDCT4Col4PureGo},
		{"DCT8Col4", dct8Size, inverseDCT8Col4SIMDAdapter, inverseDCT8Col4PureGo},
		{"DCT16Col4", dct16Size, inverseDCT16Col4SIMDAdapter, inverseDCT16Col4PureGo},
		{"DCT32Col4", dct32Size, inverseDCT32Col4SIMDAdapter, inverseDCT32Col4PureGo},
		{"DCT64Col4", dct64Size, inverseDCT64Col4Impl, inverseDCT64Col4PureGo},
		{"ADST16Col4", adst16Size, inverseADST16Col4SIMDAdapter, inverseADST16Col4PureGo},
		{"ADST16Col4Flip", adst16Size, inverseADST16Col4FlipSIMDAdapter, inverseADST16Col4FlipPureGo},
	}
}

// lanes4ClampSets is the clamp ladder the four-lane kernels must be exact
// for: every bit depth's stage bounds plus the edges of both envelopes.
func lanes4ClampSets() [][2]int32 {
	sets := append([][2]int32(nil), stageClampSets...)
	return append(sets,
		[2]int32{-itxNarrowBound, itxNarrowBound - 1},
		[2]int32{-itxWideBound, itxWideBound - 1},
		[2]int32{-itxNarrowBound - 1, itxNarrowBound - 1},
		[2]int32{-100000, 70000},
		[2]int32{-3, 5},
	)
}

// lanes4Fill returns an input generator for one of several value patterns:
// uniform in [min, max], extremes only, sparse, or small.
func lanes4Fill(rng *rand.Rand, pattern int, min, max int32) func() int32 {
	switch pattern {
	case 0:
		return func() int32 { return min + int32(rng.Int63n(int64(max)-int64(min)+1)) }
	case 1:
		ext := []int32{min, max, min + 1, max - 1, 0, 1, -1}
		return func() int32 { return ext[rng.Intn(len(ext))] }
	case 2:
		return func() int32 {
			if rng.Intn(8) != 0 {
				return 0
			}
			if rng.Intn(2) == 0 {
				return min
			}
			return max
		}
	default:
		return func() int32 { return clipRange(int64(rng.Intn(512)-256), min, max) }
	}
}

// dct64Lanes4Kernels lists both generated kernels with the bound each is
// valid for, so the Wide kernel is also exercised at narrow clamp bounds.
var dct64Lanes4Kernels = []struct {
	name  string
	bound int32
	fn    func(p []int32, stride int, min, max int32)
}{
	{"Narrow", itxNarrowBound, inverseDCT64Lanes4Narrow},
	{"Wide", itxWideBound, inverseDCT64Lanes4Wide},
}

func TestDCT64Lanes4MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6464))
	for _, kern := range dct64Lanes4Kernels {
		for _, cs := range lanes4ClampSets() {
			min, max := cs[0], cs[1]
			if min < -kern.bound || max >= kern.bound {
				continue
			}
			for _, rowStride := range []int{4, 5, 7, 64} {
				for iter := 0; iter < 400; iter++ {
					fill := lanes4Fill(rng, iter%4, min, max)
					off := rng.Intn(4)
					buf := make([]int32, off+(dct64Size-1)*rowStride+4+3)
					for i := range buf {
						buf[i] = fill()
					}
					want := append([]int32(nil), buf...)
					for c := 0; c < 4; c++ {
						inverseDCT64(want[off+c:], rowStride, min, max)
					}
					got := append([]int32(nil), buf...)
					kern.fn(got[off:], rowStride, min, max)
					if !eqInt32(got, want) {
						t.Fatalf("%s clamp=[%d,%d] stride=%d iter=%d\n in=%v\n got=%v\nwant=%v",
							kern.name, min, max, rowStride, iter, buf, got, want)
					}
				}
			}
		}
	}
}

func TestDCT64Col4Row4SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4646))
	for _, cs := range lanes4ClampSets() {
		min, max := cs[0], cs[1]
		for iter := 0; iter < 600; iter++ {
			fill := lanes4Fill(rng, iter%4, min, max)
			const rowStride = 6
			buf := make([]int32, (dct64Size-1)*rowStride+4)
			for i := range buf {
				buf[i] = fill()
			}
			want := append([]int32(nil), buf...)
			got := append([]int32(nil), buf...)
			inverseDCT64Col4PureGo(want, rowStride, min, max)
			inverseDCT64Col4SIMD(got, rowStride, min, max)
			if !eqInt32(got, want) {
				t.Fatalf("Col4 clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v", min, max, iter, buf, got, want)
			}

			rows := make([]int32, 4*dct64Size+5)
			for i := range rows {
				rows[i] = fill()
			}
			wantR := append([]int32(nil), rows...)
			gotR := append([]int32(nil), rows...)
			split := func(s []int32) ([]int32, []int32, []int32, []int32) {
				return s[1:65], s[66:130], s[130:194], s[195:259]
			}
			w0, w1, w2, w3 := split(wantR)
			g0, g1, g2, g3 := split(gotR)
			inverseDCT64Row4PureGo(w0, w1, w2, w3, min, max)
			inverseDCT64Row4SIMD(g0, g1, g2, g3, min, max)
			if !eqInt32(gotR, wantR) {
				t.Fatalf("Row4 clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v", min, max, iter, rows, gotR, wantR)
			}
		}
	}
}

func TestDCT64Lanes4SIMDDispatchBound(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON unavailable")
	}
	if reflect.ValueOf(inverseDCT64Col4Impl).Pointer() != reflect.ValueOf(inverseDCT64Col4SIMD).Pointer() {
		t.Fatal("DCT64 column dispatch did not bind the Go SIMD kernel")
	}
	if reflect.ValueOf(inverseDCT64Row4Impl).Pointer() != reflect.ValueOf(inverseDCT64Row4SIMD).Pointer() {
		t.Fatal("DCT64 row dispatch did not bind the Go SIMD kernel")
	}
}

// TestDCT64Lanes4OutOfEnvelopeFallsBack checks bounds beyond the int32-exact
// envelope and short buffers route to the scalar reference.
func TestDCT64Lanes4OutOfEnvelopeFallsBack(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	min, max := int32(-(1 << 22)), int32(1<<22-1)
	buf := make([]int32, (dct64Size-1)*4+4)
	for i := range buf {
		buf[i] = min + int32(rng.Int63n(int64(max)-int64(min)+1))
	}
	want := append([]int32(nil), buf...)
	inverseDCT64Col4PureGo(want, 4, min, max)
	inverseDCT64Col4SIMD(buf, 4, min, max)
	if !eqInt32(buf, want) {
		t.Fatal("out-of-envelope Col4 mismatch")
	}
	short := make([]int32, 3*dct64Size+60)
	wantS := append([]int32(nil), short...)
	inverseDCT64Row4SIMD(short[:60], short[64:128], short[128:192], short[192:], -100, 100)
	inverseDCT64Row4PureGo(wantS[:60], wantS[64:128], wantS[128:192], wantS[192:], -100, 100)
	if !eqInt32(short, wantS) {
		t.Fatal("short-row Row4 mismatch")
	}
}

func TestForwardDCTSIMDBindings(t *testing.T) {
	check := func(name string, got any, want string) {
		t.Helper()
		fn := runtime.FuncForPC(reflect.ValueOf(got).Pointer())
		if fn == nil {
			t.Fatalf("%s binding has no FuncForPC entry", name)
		}
		if !strings.Contains(fn.Name(), want) {
			t.Fatalf("%s bound to %s, want %s", name, fn.Name(), want)
		}
	}
	check("4x4", forwardDCT4x4Impl, "forwardDCT4x4SIMD")
	check("8x8", forwardDCT8x8Impl, "forwardDCT8x8SIMDGuarded")
	check("16x16", forwardDCT16x16Impl, "forwardDCT16x16SIMDGuarded")
	check("32x32", forwardDCT32x32Impl, "forwardDCT32x32SIMDGuarded")
}

func TestForwardDCT4x4SIMDExactResidualLength(t *testing.T) {
	var residual [16]int16
	var want, got [16]int32
	rng := rand.New(rand.NewSource(401))
	for trial := range 1000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(1<<16) - (1 << 15))
		}
		clear(want[:])
		clear(got[:])
		forwardDCT4x4PureGo(want[:], 4, residual[:], 4)
		forwardDCT4x4SIMD(got[:], 4, residual[:], 4)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d coeff[%d] SIMD=%d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestForwardDCT4x4SIMDProvenInt32Range(t *testing.T) {
	var residual [16]int16
	var want, got [16]int32
	rng := rand.New(rand.NewSource(405))
	for trial := range 1000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(4097) - 2048)
		}
		clear(want[:])
		clear(got[:])
		forwardDCT4x4PureGo(want[:], 4, residual[:], 4)
		forwardDCT4x4SIMD(got[:], 4, residual[:], 4)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d coeff[%d] SIMD=%d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestForwardDCTSIMDZeroAlloc(t *testing.T) {
	var residual [16]int16
	for i := range residual {
		residual[i] = int16(i%511) - 255
	}
	var coeff [16]int32
	allocs := testing.AllocsPerRun(100, func() {
		forwardDCT4x4SIMD(coeff[:], 4, residual[:], 4)
	})
	if allocs != 0 {
		t.Fatalf("SIMD forward DCT allocated %v objects/run, want 0", allocs)
	}
}

func BenchmarkForwardDCT4x4Kernels(b *testing.B) {
	var residual [16]int16
	for i := range residual {
		residual[i] = int16(i*29%400) - 200
	}
	benchmarkForwardDCTKernel(b, 4, residual[:], 4, []struct {
		name string
		fn   func([]int32, int, []int16, int)
	}{
		{name: "simd", fn: forwardDCT4x4SIMD},
		{name: "purego", fn: forwardDCT4x4PureGo},
	})
}

func BenchmarkForwardDCT8x8Kernels(b *testing.B) {
	var residual [64]int16
	for i := range residual {
		residual[i] = int16(i*7%400) - 200
	}
	benchmarkForwardDCTKernel(b, 8, residual[:], 8, []struct {
		name string
		fn   func([]int32, int, []int16, int)
	}{
		{name: "simd", fn: benchmarkForwardDCTGuardedNEON(forwardDCT8x8SIMD, forwardDCT8x8PureGo, 8)},
		{name: "purego", fn: forwardDCT8x8PureGo},
	})
}

func BenchmarkForwardDCT16x16Kernels(b *testing.B) {
	var residual [256]int16
	for i := range residual {
		residual[i] = int16(i*11%400) - 200
	}
	benchmarkForwardDCTKernel(b, 16, residual[:], 16, []struct {
		name string
		fn   func([]int32, int, []int16, int)
	}{
		{name: "simd", fn: benchmarkForwardDCTGuardedNEON(forwardDCT16x16SIMD, forwardDCT16x16PureGo, 16)},
		{name: "purego", fn: forwardDCT16x16PureGo},
	})
}

// benchmarkForwardDCTGuardedNEON includes the same range check and scalar
// fallback as the production asm dispatch.
func benchmarkForwardDCTGuardedNEON(neon, pure func([]int32, int, []int16, int), side int) func([]int32, int, []int16, int) {
	return func(coeff []int32, coeffStride int, residual []int16, residualStride int) {
		if !residualFitsMagnitude(residual, residualStride, side, side, 255) {
			pure(coeff, coeffStride, residual, residualStride)
			return
		}
		neon(coeff, coeffStride, residual, residualStride)
	}
}

func benchmarkForwardDCTKernel(b *testing.B, side int, residual []int16, residualStride int, kernels []struct {
	name string
	fn   func([]int32, int, []int16, int)
}) {
	b.Helper()
	for _, k := range kernels {
		b.Run(k.name, func(b *testing.B) {
			coeff := make([]int32, side*side)
			b.ReportAllocs()
			for b.Loop() {
				k.fn(coeff, side, residual, residualStride)
			}
		})
	}
}

// TestStageTransposeClampSIMDDifferential locksteps the SIMD staging kernel
// against stageTransposeClampScalar over every block shape the row pass can
// request, both rect2 variants, adversarial extremes (full int32 range, well
// past the dequant envelope) and random dequant-realistic values.
func TestStageTransposeClampSIMDDifferential(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	extremes := []int32{
		math.MinInt32, math.MinInt32 + 1, -(1 << 22) - 1, -(1 << 22), -(1 << 22) + 1,
		-524288, -181, -1, 0, 1, 181, 524287,
		(1 << 22) - 1, 1 << 22, (1 << 22) + 1, math.MaxInt32 - 1, math.MaxInt32,
	}
	clamps := []struct{ lo, hi int32 }{
		{-32768, 32767},                // 8-bit row clamp
		{-131072, 131071},              // 10-bit
		{-524288, 524287},              // 12-bit
		{math.MinInt32, math.MaxInt32}, // degenerate wide clamp (rect2 falls back)
	}
	shapes := []struct{ rows, cols, width, stride int }{
		{4, 4, 4, 4}, {4, 4, 8, 8}, {8, 8, 8, 8}, {16, 16, 16, 16},
		{32, 32, 32, 32}, {32, 32, 64, 32}, {5, 4, 8, 8}, {4, 5, 8, 8},
		{7, 6, 8, 8}, {13, 9, 16, 16}, {3, 3, 4, 4}, {2, 7, 8, 8},
		{16, 4, 16, 16}, {4, 16, 16, 16}, {31, 17, 32, 32},
		{1, 4, 4, 4}, {1, 8, 8, 8}, {2, 32, 32, 32}, {3, 16, 16, 16}, {2, 6, 8, 8},
		{6, 8, 8, 8}, {10, 12, 16, 16}, {3, 64, 64, 32}, {1, 32, 32, 4},
	}
	for _, sh := range shapes {
		coeff := make([]int32, sh.stride*sh.cols+sh.rows)
		want := make([]int32, sh.width*sh.rows)
		got := make([]int32, sh.width*sh.rows)
		for _, cl := range clamps {
			for _, rect2 := range []bool{false, true} {
				for trial := 0; trial < 8; trial++ {
					for i := range coeff {
						switch trial {
						case 0:
							coeff[i] = extremes[i%len(extremes)]
						case 1:
							coeff[i] = extremes[rng.Intn(len(extremes))]
						default:
							coeff[i] = int32(rng.Intn(1<<20) - 1<<19)
						}
					}
					for i := range want {
						want[i] = math.MaxInt32
						got[i] = math.MaxInt32
					}
					stageTransposeClampScalar(want, sh.width, coeff, sh.stride, sh.rows, sh.cols, rect2, cl.lo, cl.hi)
					stageTransposeClamp(got, sh.width, coeff, sh.stride, sh.rows, sh.cols, rect2, cl.lo, cl.hi)
					for i := range want {
						if want[i] != got[i] {
							t.Fatalf("shape %+v clamp %+v rect2=%v trial=%d: mismatch at %d: scalar %d simd %d",
								sh, cl, rect2, trial, i, want[i], got[i])
						}
					}
				}
			}
		}
	}
}

func BenchmarkStageTransposeClamp32x32(b *testing.B) {
	benchStage(b, 32, 32, 32, false)
}

func BenchmarkStageTransposeClamp32x32Rect2(b *testing.B) {
	benchStage(b, 32, 32, 32, true)
}

func BenchmarkStageTransposeClamp16x16(b *testing.B) {
	benchStage(b, 16, 16, 16, false)
}

func BenchmarkStageTransposeClampScalar32x32(b *testing.B) {
	benchStageScalar(b, 32, 32, 32, false)
}

func BenchmarkStageTransposeClampScalar16x16(b *testing.B) {
	benchStageScalar(b, 16, 16, 16, false)
}

func benchStage(b *testing.B, rows, cols, width int, rect2 bool) {
	coeff := make([]int32, cols*width)
	scratch := make([]int32, rows*width)
	rng := rand.New(rand.NewSource(2))
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<16) - 1<<15)
	}
	b.ReportAllocs()
	for b.Loop() {
		stageTransposeClamp(scratch, width, coeff, width, rows, cols, rect2, -32768, 32767)
	}
}

func benchStageScalar(b *testing.B, rows, cols, width int, rect2 bool) {
	coeff := make([]int32, cols*width)
	scratch := make([]int32, rows*width)
	rng := rand.New(rand.NewSource(2))
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<16) - 1<<15)
	}
	b.ReportAllocs()
	for b.Loop() {
		stageTransposeClampScalar(scratch, width, coeff, width, rows, cols, rect2, -32768, 32767)
	}
}

// Short stagings (EOB-limited active rows) and row counts off the 4-row grid.
func BenchmarkStageTransposeClampShort(b *testing.B) {
	for _, sh := range []struct{ rows, cols int }{{1, 8}, {1, 32}, {2, 8}, {2, 16}, {3, 32}, {2, 32}, {6, 16}} {
		name := fmt.Sprintf("%dx%d", sh.rows, sh.cols)
		b.Run(name+"/simd", func(b *testing.B) { benchStage(b, sh.rows, sh.cols, sh.cols, false) })
		b.Run(name+"/scalar", func(b *testing.B) { benchStageScalar(b, sh.rows, sh.cols, sh.cols, false) })
	}
}

// itx16ColL1OK reports whether every column of the width x height int16 block
// has sum |x| <= bound.
func itx16ColL1OK(buf []int16, width, height int, bound int) bool {
	for c := 0; c < width; c++ {
		l1 := 0
		for r := 0; r < height; r++ {
			v := int(buf[r*width+c])
			if v < 0 {
				v = -v
			}
			l1 += v
		}
		if l1 > bound {
			return false
		}
	}
	return true
}

func itx16FillAdversarial(rng *rand.Rand, buf []int16, width, height, mode int) {
	for i := range buf {
		buf[i] = 0
	}
	switch mode % 6 {
	case 0: // dense uniform, within the guard
		for i := range buf {
			buf[i] = int16(rng.Intn(4081) - 2040)
		}
	case 1: // one full-range row (sparse: eob-like)
		r := rng.Intn(height)
		for c := 0; c < width; c++ {
			buf[r*width+c] = int16(rng.Intn(65536) - 32768)
		}
	case 2: // two rows at +-32767 with random signs
		r0, r1 := rng.Intn(height), rng.Intn(height)
		for c := 0; c < width; c++ {
			buf[r0*width+c] = int16(16383 * (2*rng.Intn(2) - 1))
			buf[r1*width+c] = int16(16383 * (2*rng.Intn(2) - 1))
		}
	case 3: // full-range random everywhere (outside the guard: report only)
		for i := range buf {
			buf[i] = int16(rng.Intn(65536) - 32768)
		}
	case 4: // extremes
		for i := range buf {
			if rng.Intn(2) == 0 {
				buf[i] = -32768
			} else {
				buf[i] = 32767
			}
		}
	default: // mixed magnitudes with zero rows
		for r := 0; r < height; r++ {
			if rng.Intn(3) == 0 {
				continue
			}
			m := 1 << uint(rng.Intn(15))
			for c := 0; c < width; c++ {
				buf[r*width+c] = int16(rng.Intn(2*m+1) - m)
			}
		}
	}
}

func TestInverseDCT16Col8SIMD16v2MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const width, height = 8, 16
	const bound = 32767 - 64
	got := make([]int16, width*height)
	want := make([]int16, width*height)
	inGuard, matchOut, totalOut := 0, 0, 0
	for trial := 0; trial < 4000; trial++ {
		itx16FillAdversarial(rng, want, width, height, trial)
		copy(got, want)
		guard := itx16ColL1OK(want, width, height, bound)
		for c := 0; c < width; c++ {
			inverseDCT16(want[c:], width, minInt16, maxInt16)
		}
		inverseDCT16Col8SIMD16v2(got, width, minInt16, maxInt16)
		equal := true
		for i := range got {
			if got[i] != want[i] {
				equal = false
				break
			}
		}
		if guard {
			inGuard++
			if !equal {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial %d (mode %d) in guard: mismatch at %d: simd %d scalar %d", trial, trial%6, i, got[i], want[i])
					}
				}
			}
		} else {
			totalOut++
			if equal {
				matchOut++
			}
		}
	}
	t.Logf("in-guard trials %d (all exact); out-of-guard trials %d, of which %d still exact", inGuard, totalOut, matchOut)
}

func BenchmarkDCT16x8_Int16v2(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int16, 16*8+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col8SIMD16v2(buf, 8, minInt16, maxInt16)
	}
}

func BenchmarkDCT16x8_Int16Old(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int16, 16*8+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col8SIMD16(buf, 8, minInt16, maxInt16)
	}
}

func BenchmarkDCT16x8_Col4x2(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int32, 16*8+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col4SIMDAdapter(buf, 8, minInt16, maxInt16)
		inverseDCT16Col4SIMDAdapter(buf[4:], 8, minInt16, maxInt16)
	}
}

func TestInverseDCT16Col8RawSIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	const width, height = 8, 16
	const bound = 32767 - 64
	src := make([]int16, width*height)
	got := make([]int32, width*height)
	want := make([]int32, width*height)
	inGuard, rejected := 0, 0
	for trial := 0; trial < 4000; trial++ {
		itx16FillAdversarial(rng, src, width, height, trial)
		for i, v := range src {
			got[i] = int32(v)
			want[i] = int32(v)
		}
		guard := itx16ColL1OK(src, width, height, bound)
		for c := 0; c < width; c++ {
			inverseDCT16(want[c:], width, minInt16, maxInt16)
		}
		ok := inverseDCT16Col8RawSIMD16(got, width)
		if ok != guard {
			t.Fatalf("trial %d: kernel guard %v, reference guard %v", trial, ok, guard)
		}
		if !ok {
			rejected++
			for i, v := range src {
				if got[i] != int32(v) {
					t.Fatalf("trial %d: rejected block modified at %d", trial, i)
				}
			}
			continue
		}
		inGuard++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d): mismatch at %d: simd %d scalar %d", trial, trial%6, i, got[i], want[i])
			}
		}
	}
	t.Logf("in-guard trials %d (all exact), rejected %d (untouched)", inGuard, rejected)
}

func BenchmarkDCT16x8_RawInt16(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int32, 16*8+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col8RawSIMD16(buf, 8)
	}
}

func testDCT16RawLow(t *testing.T, lowN int, kernel func([]int32, int) bool) {
	rng := rand.New(rand.NewSource(int64(100 + lowN)))
	const width, height = 8, 16
	const bound = 32767 - 64
	src := make([]int16, width*height)
	got := make([]int32, width*height)
	want := make([]int32, width*height)
	inGuard, rejected := 0, 0
	for trial := 0; trial < 4000; trial++ {
		itx16FillAdversarial(rng, src, width, height, trial)
		for i := lowN * width; i < len(src); i++ {
			src[i] = 0
		}
		for i, v := range src {
			got[i] = int32(v)
			want[i] = int32(v)
		}
		guard := itx16ColL1OK(src, width, height, bound)
		for c := 0; c < width; c++ {
			inverseDCT16(want[c:], width, minInt16, maxInt16)
		}
		ok := kernel(got, width)
		if ok != guard {
			t.Fatalf("trial %d: kernel guard %v, reference guard %v", trial, ok, guard)
		}
		if !ok {
			rejected++
			for i, v := range src {
				if got[i] != int32(v) {
					t.Fatalf("trial %d: rejected block modified at %d", trial, i)
				}
			}
			continue
		}
		inGuard++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d): mismatch at %d: simd %d scalar %d", trial, trial%6, i, got[i], want[i])
			}
		}
	}
	t.Logf("low%d: in-guard trials %d (all exact), rejected %d (untouched)", lowN, inGuard, rejected)
}

func TestInverseDCT16Col8RawLow4SIMD16MatchesScalar(t *testing.T) {
	testDCT16RawLow(t, 4, inverseDCT16Col8RawLow4SIMD16)
}

func TestInverseDCT16Col8RawLow8SIMD16MatchesScalar(t *testing.T) {
	testDCT16RawLow(t, 8, inverseDCT16Col8RawLow8SIMD16)
}

func benchRawLow(b *testing.B, lowN int, kernel func([]int32, int) bool) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int32, 16*8+8)
	for i := range buf {
		if i/8 < lowN {
			buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kernel(buf, 8)
		for j := lowN * 8; j < 16*8; j++ {
			buf[j] = 0
		}
	}
}

func BenchmarkDCT16x8_RawInt16Low4(b *testing.B) { benchRawLow(b, 4, inverseDCT16Col8RawLow4SIMD16) }
func BenchmarkDCT16x8_RawInt16Low8(b *testing.B) { benchRawLow(b, 8, inverseDCT16Col8RawLow8SIMD16) }

// TestInverseDCT32Col8RawLow8SIMD16MatchesScalar: every column group the
// int16 Low8 kernel accepts equals the scalar inverseDCT32 (the int32 path's
// result on 8-bit column bounds), a declined group is left untouched, and
// every group inside the per-column L1 guard is accepted.
func TestInverseDCT32Col8RawLow8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const width, height = 8, 32
	want := make([]int16, width*height)
	buf := make([]int32, width*height)
	keep := make([]int32, width*height)
	inGuard, accepted, declined := 0, 0, 0
	for trial := 0; trial < 6000; trial++ {
		itx16FillAdversarial(rng, want, width, height, trial)
		for r := 8; r < height; r++ {
			clear(want[r*width : (r+1)*width])
		}
		for i, v := range want {
			buf[i] = int32(v)
		}
		copy(keep, buf)
		guard := itx16ColL1OK(want, width, height, itx16ColumnL1Bound)
		for c := 0; c < width; c++ {
			inverseDCT32(want[c:], width, minInt16, maxInt16)
		}
		if !inverseDCT32Col8RawLow8SIMD16(buf, width) {
			declined++
			for i := range buf {
				if buf[i] != keep[i] {
					t.Fatalf("trial %d: declined group modified at %d", trial, i)
				}
			}
			if guard {
				t.Fatalf("trial %d (mode %d): in-guard group declined", trial, trial%6)
			}
			continue
		}
		accepted++
		if guard {
			inGuard++
		}
		for i := range buf {
			if buf[i] != int32(want[i]) {
				t.Fatalf("trial %d (mode %d, guard %v): mismatch at row %d col %d: simd %d scalar %d",
					trial, trial%6, guard, i/width, i%width, buf[i], want[i])
			}
		}
	}
	t.Logf("accepted %d (in guard %d), declined %d", accepted, inGuard, declined)
}

// TestITX16RawColumnPass32Low8MatchesScalar: the pass over a 16 x 32 block
// with rows 8..31 zero, including column groups that fail the guard (falling
// back to the int32 kernels), equals the scalar column pass.
func TestITX16RawColumnPass32Low8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	const width, height = 16, 32
	want := make([]int16, width*height)
	buf := make([]int32, width*height)
	for trial := 0; trial < 3000; trial++ {
		itx16FillAdversarial(rng, want, width, height, trial)
		for r := 8; r < height; r++ {
			clear(want[r*width : (r+1)*width])
		}
		if trial%7 == 0 {
			for r := 0; r < 8; r++ {
				for c := 0; c < 8; c++ {
					want[r*width+c] = int16(rng.Intn(65536) - 32768)
				}
			}
		}
		for i, v := range want {
			buf[i] = int32(v)
		}
		for c := 0; c < width; c++ {
			inverseDCT32(want[c:], width, minInt16, maxInt16)
		}
		itx16RawColumnPass32Low8SIMD(buf, width, minInt16, maxInt16)
		for i := range buf {
			if buf[i] != int32(want[i]) {
				t.Fatalf("trial %d: mismatch at row %d col %d: pass %d scalar %d", trial, i/width, i%width, buf[i], want[i])
			}
		}
	}
}

func itx32Low8BenchInput() []int32 {
	rng := rand.New(rand.NewSource(9))
	src := make([]int32, 32*8)
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			src[r*8+c] = int32(rng.Intn(1<<12) - (1 << 11))
		}
	}
	return src
}

func BenchmarkDCT32x8Low8_Int16Raw(b *testing.B) {
	src := itx32Low8BenchInput()
	buf := make([]int32, len(src))
	for i := 0; i < b.N; i++ {
		copy(buf, src)
		if !inverseDCT32Col8RawLow8SIMD16(buf, 8) {
			b.Fatal("guard declined the benchmark input")
		}
	}
}

func BenchmarkDCT32x8Low8_Int32Col4x2(b *testing.B) {
	src := itx32Low8BenchInput()
	buf := make([]int32, len(src))
	for i := 0; i < b.N; i++ {
		copy(buf, src)
		inverseDCT32Col4Low8SIMD(buf, 8, minInt16, maxInt16)
		inverseDCT32Col4Low8SIMD(buf[4:], 8, minInt16, maxInt16)
	}
}

func TestInverseDCT32Col4Low8SIMDMatchesFullAndScalar(t *testing.T) {
	const prefix = 3
	const suffix = 4
	const canary = int32(0x51a7c0de)
	rng := rand.New(rand.NewSource(0x32_108))

	check := func(t *testing.T, stride int, min, max int32, values [32]int32) {
		t.Helper()
		span := 31*stride + 4
		newBacking := func() ([]int32, []int32) {
			backing := make([]int32, prefix+span+suffix)
			for i := range backing {
				backing[i] = canary
			}
			view := backing[prefix : prefix+span : prefix+span]
			for row := 0; row < 32; row++ {
				for lane := 0; lane < 4; lane++ {
					if row < 8 {
						view[row*stride+lane] = values[row*4+lane]
					} else {
						view[row*stride+lane] = 0
					}
				}
			}
			return backing, view
		}

		wantFull, fullView := newBacking()
		wantScalar, scalarView := newBacking()
		got, low8View := newBacking()
		inverseDCT32Col4SIMD(fullView, stride, min, max)
		inverseDCT32Col4PureGo(scalarView, stride, min, max)
		inverseDCT32Col4Low8SIMD(low8View, stride, min, max)
		for i := range wantFull {
			if wantFull[i] != wantScalar[i] {
				t.Fatalf("full/scalar mismatch stride=%d clamp=[%d,%d] backing[%d]=%d want %d", stride, min, max, i, wantFull[i], wantScalar[i])
			}
			if got[i] != wantFull[i] {
				t.Fatalf("low8/full mismatch stride=%d clamp=[%d,%d] backing[%d]=%d want %d", stride, min, max, i, got[i], wantFull[i])
			}
		}
		for i := 0; i < prefix; i++ {
			if got[i] != canary {
				t.Fatalf("prefix canary[%d]=%#x want %#x", i, got[i], canary)
			}
		}
		for i := prefix + span; i < len(got); i++ {
			if got[i] != canary {
				t.Fatalf("suffix canary[%d]=%#x want %#x", i, got[i], canary)
			}
		}
	}

	// Every signed-int16 value appears in the active input rows across these
	// batches, covering the kernel's full lane input domain.
	for batch := 0; batch < (1<<16)/32; batch++ {
		var values [32]int32
		for i := range values {
			values[i] = int32(int16(int32(-1<<15) + int32(batch*32+i)))
		}
		check(t, 4, minInt16, maxInt16, values)
	}

	for _, stride := range []int{4, 5, 13} {
		for _, clamp := range [][2]int32{{minInt16, maxInt16}, {-1024, 1023}, {-1, 1}, {0, maxInt16}} {
			for iter := 0; iter < 180; iter++ {
				var values [32]int32
				for i := range values {
					values[i] = clamp[0] + int32(rng.Int63n(int64(clamp[1])-int64(clamp[0])+1))
				}
				check(t, stride, clamp[0], clamp[1], values)
			}
		}
	}

	// Exercise every input row/lane independently at signed and rounding
	// boundaries, including values adjacent to int16 extrema.
	impulses := []int32{minInt16, minInt16 + 1, -16385, -9, -8, -7, -1, 0, 1, 7, 8, 9, 16383, maxInt16 - 1, maxInt16}
	for row := 0; row < 8; row++ {
		for lane := 0; lane < 4; lane++ {
			for _, value := range impulses {
				var values [32]int32
				values[row*4+lane] = value
				check(t, 4, minInt16, maxInt16, values)
			}
		}
	}
}

func TestInverseBlockBitDepthRawDCT32Low8RouteAndFallback(t *testing.T) {
	original := inverseDCT32Low8Col4Impl
	defer func() { inverseDCT32Low8Col4Impl = original }()
	// The int16 column pass takes this shape first; this test covers the
	// int32 route it falls back to.
	pass16 := itx16RawColumnPass32Low8
	itx16RawColumnPass32Low8 = nil
	defer func() { itx16RawColumnPass32Low8 = pass16 }()

	type routeCase struct {
		name       string
		size       Size
		typ        Type
		bitDepth   uint8
		activeRows int
		wantRoute  bool
	}
	cases := []routeCase{
		{name: "width8 rows1", size: Size{Width: 8, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 1, wantRoute: true},
		{name: "width16 rows7", size: Size{Width: 16, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 7, wantRoute: true},
		{name: "width32 rows8", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 8, wantRoute: true},
		{name: "width64 rows8", size: Size{Width: 64, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 8, wantRoute: true},
		{name: "vertical dct horizontal adst", size: Size{Width: 8, Height: 32}, typ: TypeDCTADST, bitDepth: 8, activeRows: 4, wantRoute: true},
		{name: "vertical dct identity horizontal", size: Size{Width: 8, Height: 32}, typ: TypeVDCT, bitDepth: 8, activeRows: 4, wantRoute: true},
		{name: "vertical identity", size: Size{Width: 32, Height: 32}, typ: TypeHDCT, bitDepth: 8, activeRows: 4},
		{name: "ten bit same column envelope", size: Size{Width: 16, Height: 32}, typ: TypeDCTDCT, bitDepth: 10, activeRows: 8, wantRoute: true},
		{name: "full sentinel", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 0},
		{name: "above low rows", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 9},
		{name: "wide adjusted scan above low rows", size: Size{Width: 64, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 9},
		{name: "twelve bit wider column range", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 12, activeRows: 8},
		{name: "short vertical", size: Size{Width: 16, Height: 16}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 4},
		{name: "vertical adst", size: Size{Width: 8, Height: 16}, typ: TypeADSTDCT, bitDepth: 8, activeRows: 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.typ.Supported(tc.size) {
				t.Fatalf("test case uses unsupported type/size: %v %+v", tc.typ, tc.size)
			}
			scan := adjustedScanSize(tc.size)
			coeffStride := int(scan.Height) + 3
			coeff := make([]int32, (int(scan.Width)-1)*coeffStride+int(scan.Height))
			rng := rand.New(rand.NewSource(0x32_500 + int64(tc.activeRows) + int64(tc.size.Width)*7))
			active := tc.activeRows
			if active == 0 || active > int(scan.Height) {
				active = int(scan.Height)
			}
			for col := 0; col < int(scan.Width); col++ {
				for row := 0; row < active; row++ {
					coeff[col*coeffStride+row] = int32(rng.Intn(8191) - 4095)
				}
			}
			want := make([]int32, int(tc.size.Width)*int(tc.size.Height))
			got := make([]int32, len(want))
			baseline := inverseDCT32Low8Col4Impl
			inverseDCT32Low8Col4Impl = nil
			wantErr := InverseBlockBitDepthRaw(want, coeff, coeffStride, tc.size, tc.typ, tc.bitDepth, tc.activeRows)
			inverseDCT32Low8Col4Impl = baseline
			if wantErr != nil {
				t.Fatalf("baseline Raw: %v", wantErr)
			}
			calls := 0
			inverseDCT32Low8Col4Impl = func(buf []int32, rowStride int, min, max int32) {
				calls++
				original(buf, rowStride, min, max)
			}
			gotErr := InverseBlockBitDepthRaw(got, coeff, coeffStride, tc.size, tc.typ, tc.bitDepth, tc.activeRows)
			inverseDCT32Low8Col4Impl = baseline
			if gotErr != nil {
				t.Fatalf("candidate Raw: %v", gotErr)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("Raw mismatch at %d: got %d want %d", i, got[i], want[i])
				}
			}
			wantCalls := 0
			if tc.wantRoute {
				wantCalls = int(tc.size.Width) / 4
			}
			if calls != wantCalls {
				t.Fatalf("specialized kernel calls=%d want %d", calls, wantCalls)
			}
		})
	}

	// The optional slot is nil in pure-Go/other architectures; explicitly
	// verify that an eligible public Raw transform still uses the old path.
	size := Size{Width: 32, Height: 32}
	scan := adjustedScanSize(size)
	coeff := make([]int32, int(scan.Width)*int(scan.Height))
	coeff[0] = 17
	out := make([]int32, 32*32)
	inverseDCT32Low8Col4Impl = nil
	if err := InverseBlockBitDepthRaw(out, coeff, int(scan.Height), size, TypeDCTDCT, 8, 1); err != nil {
		t.Fatalf("unbound optional slot fallback: %v", err)
	}
}

func TestInverseBlockBitDepthRawDCT32Low8ReusesPoisonedDestination(t *testing.T) {
	original := inverseDCT32Low8Col4Impl
	if original == nil {
		t.Fatal("ARM64 low8 implementation is not bound")
	}
	defer func() { inverseDCT32Low8Col4Impl = original }()
	// The int16 column pass takes this shape first; this test covers the
	// int32 route it falls back to.
	pass16 := itx16RawColumnPass32Low8
	itx16RawColumnPass32Low8 = nil
	defer func() { itx16RawColumnPass32Low8 = pass16 }()

	size := Size{Width: 32, Height: 32}
	scan := adjustedScanSize(size)
	stride := int(scan.Height)
	coeff := make([]int32, int(scan.Width)*stride)
	for col := 0; col < int(scan.Width); col++ {
		for row := 0; row < stride; row++ {
			coeff[col*stride+row] = int32((col*31+row*47)%4093 - 2046)
		}
	}
	shared := make([]int32, 32*32)
	if err := InverseBlockBitDepthRaw(shared, coeff, stride, size, TypeDCTDCT, 8, 0); err != nil {
		t.Fatalf("initial full Raw: %v", err)
	}
	poisoned := false
	for _, v := range shared {
		poisoned = poisoned || v != 0
	}
	if !poisoned {
		t.Fatal("full transform left the destination entirely zero")
	}

	low := make([]int32, len(coeff))
	for col := 0; col < int(scan.Width); col++ {
		low[col*stride] = int32(col*23 - 300)
	}
	want := make([]int32, 32*32)
	prev := inverseDCT32Low8Col4Impl
	inverseDCT32Low8Col4Impl = nil
	err := InverseBlockBitDepthRaw(want, low, stride, size, TypeDCTDCT, 8, 0)
	inverseDCT32Low8Col4Impl = prev
	if err != nil {
		t.Fatalf("independent full-sentinel Raw oracle: %v", err)
	}
	calls := 0
	inverseDCT32Low8Col4Impl = func(buf []int32, rowStride int, min, max int32) {
		calls++
		original(buf, rowStride, min, max)
	}
	err = InverseBlockBitDepthRaw(shared, low, stride, size, TypeDCTDCT, 8, 1)
	inverseDCT32Low8Col4Impl = prev
	if err != nil {
		t.Fatalf("low8 Raw after full transform: %v", err)
	}
	if calls != 8 {
		t.Fatalf("low8 kernel calls=%d want 8", calls)
	}
	for i := range want {
		if shared[i] != want[i] {
			t.Fatalf("reused destination[%d]=%d want %d", i, shared[i], want[i])
		}
	}
}

func TestInverseBlockBitDepthRawDCT32Low8NoAlloc(t *testing.T) {
	size := Size{Width: 32, Height: 32}
	scan := adjustedScanSize(size)
	coeff := make([]int32, int(scan.Width)*int(scan.Height))
	for col := 0; col < int(scan.Width); col++ {
		for row := 0; row < 8; row++ {
			coeff[col*int(scan.Height)+row] = int32((col*19+row*7)%511 - 255)
		}
	}
	dst := make([]int32, 32*32)
	allocs := testing.AllocsPerRun(100, func() {
		if err := InverseBlockBitDepthRaw(dst, coeff, int(scan.Height), size, TypeDCTDCT, 8, 8); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Raw DCT32 low8 allocs/op=%g want 0", allocs)
	}
}

func TestInverseDCT32Low8AdapterFallsBackWhenClampMovesZero(t *testing.T) {
	const stride = 5
	const span = 31*stride + 4
	input := make([]int32, span)
	for row := 0; row < 8; row++ {
		for lane := 0; lane < 4; lane++ {
			input[row*stride+lane] = int32(row*11 + lane - 20)
		}
	}
	full := append([]int32(nil), input...)
	low8 := append([]int32(nil), input...)
	inverseDCT32Col4SIMDAdapter(full, stride, 1, 32767)
	inverseDCT32Low8Col4SIMDAdapter(low8, stride, 1, 32767)
	for i := range full {
		if low8[i] != full[i] {
			t.Fatalf("clamp=[1,32767] fallback mismatch at %d: got %d want %d", i, low8[i], full[i])
		}
	}
}

// itxFused16Oracle is the staged path the fused 16x16 kernel replaces.
func itxFused16Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 16*16)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 16, Height: 16}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 16; r++ {
		for c := 0; c < 16; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*16+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT16x16MatchesStagedPath: every (rows, cols) shape up to
// 16x16, every network form pairing, byte-exact against the staged path;
// declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT16x16MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	const stride = 16 + 8
	coeff := make([]int32, 16*16)
	pred := make([]byte, 16*stride)
	got := make([]byte, 16*stride)
	want := make([]byte, 16*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 30000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(16), 1+rng.Intn(16)
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
		case 1:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*16+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused16Oracle(t, want, stride, pred, stride, coeff, 16, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT16x16(got, stride, got, stride, coeff, 16, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT16x16(got, stride, pred, stride, coeff, 16, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 16; r++ {
			for c := 0; c < 16; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 16; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT16x16(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low4", 4, 4}, {"low8", 8, 8}, {"full", 16, 16}} {
		coeff := make([]int32, 16*16)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*16+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 16*16)
		for i := range dst {
			dst[i] = byte(rng.Intn(256))
		}
		raw := make([]int32, 16*16)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT16x16(dst, 16, dst, 16, coeff, 16, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 16, Size{Width: 16, Height: 16}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestInverseAddDCT32x32FullMatchesStagedPath: every (rows, cols) shape up
// to 32x32, every network form pairing, byte-exact against the staged path;
// declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT32x32FullMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		case 1:
			activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT32x32Full(got, stride, got, stride, coeff, 32, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT32x32Full(got, stride, pred, stride, coeff, 32, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 32; r++ {
			for c := 0; c < 32; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 32; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
}

func BenchmarkInverseAddDCT32x32Full(b *testing.B) {
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low16", 16, 16}, {"full", 32, 32}} {
		rng := rand.New(rand.NewSource(5))
		coeff := make([]int32, 32*32)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 32*32)
		raw := make([]int32, 32*32)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT32x32Full(dst, 32, dst, 32, coeff, 32, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// itxFused32Oracle is the staged path the fused kernel replaces: the
// transform into the raw int32 scratch, then Round2(., 4) added to the
// prediction and clipped to bytes.
func itxFused32Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 32*32)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 32; r++ {
		for c := 0; c < 32; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*32+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT32x32Low8MatchesStagedPath: every block the fused path
// accepts reproduces the staged transform plus the raw add byte for byte, a
// declined block leaves dst untouched, and the decline rate at the
// magnitudes real content codes is zero.
func TestInverseAddDCT32x32Low8MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(8), 1+rng.Intn(8)
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		if rng.Intn(4) == 0 {
			coeff[0] = int32(rng.Intn(2*mag+1) - mag)
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		// In place (the decoder: the prediction already sits in dst).
		copy(got, pred)
		ok := InverseAddDCT32x32Low8(got, stride, got, stride, coeff, 32)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		// Separate prediction buffer (the encoder's reconstruction).
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT32x32Low8(got, stride, pred, stride, coeff, 32) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 32; r++ {
			for c := 0; c < 32; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 32; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT32x32Low8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 32*32)
	for c := 0; c < 8; c++ {
		for r := 0; r < 8; r++ {
			coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 32*32)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT32x32Low8(dst, 32, dst, 32, coeff, 32) {
				b.Fatal("declined")
			}
		}
	})
	raw := make([]int32, 32*32)
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, 8, 8); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// TestInverseAddDCT32x32Low16MatchesStagedPath: every block the fused Low16
// path accepts reproduces the staged transform plus the raw add byte for
// byte over the (rows, cols) shapes up to 16x16, including the eight-row
// shapes that skip the second row strip; declined blocks leave dst untouched
// and no block at realistic magnitudes declines.
func TestInverseAddDCT32x32Low16MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(16), 1+rng.Intn(16)
		if trial%4 == 0 {
			activeRows = 1 + rng.Intn(8)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT32x32Low16(got, stride, got, stride, coeff, 32, activeRows)
		// Sixteen coefficients of +-4096 in one row exceed the per-lane
		// guard, so only the +-64 and +-1024 classes must never decline.
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT32x32Low16(got, stride, pred, stride, coeff, 32, activeRows) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 32; r++ {
			for c := 0; c < 32; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 32; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT32x32Low16(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 32*32)
	for c := 0; c < 16; c++ {
		for r := 0; r < 16; r++ {
			coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 32*32)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT32x32Low16(dst, 32, dst, 32, coeff, 32, 16) {
				b.Fatal("declined")
			}
		}
	})
	raw := make([]int32, 32*32)
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, 16, 16); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// itxFused64Oracle is the staged path the fused 64x64 kernel replaces.
func itxFused64Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 64*64)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 64, Height: 64}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 64; r++ {
		for c := 0; c < 64; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*64+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT64x64MatchesStagedPath: every (rows, cols) shape within
// the 32x32 corner, every network form pairing, byte-exact against the
// staged path; declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT64x64MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 64 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 64*stride)
	got := make([]byte, 64*stride)
	want := make([]byte, 64*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 6000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		case 1:
			activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused64Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT64x64(got, stride, got, stride, coeff, 32, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT64x64(got, stride, pred, stride, coeff, 32, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 64; r++ {
			for c := 0; c < 64; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 64; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
}

func BenchmarkInverseAddDCT64x64(b *testing.B) {
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low8", 8, 8}, {"low16", 16, 16}, {"low32", 32, 32}} {
		rng := rand.New(rand.NewSource(5))
		coeff := make([]int32, 32*32)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 64*64)
		raw := make([]int32, 64*64)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT64x64(dst, 64, dst, 64, coeff, 32, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 64, Height: 64}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestInverseAddDCT64x64HBDMatchesStagedPath: every accepted 10- and 12-bit
// 64x64 block (every network form pairing within the 32x32 corner)
// reproduces the staged transform plus the raw add sample for sample, in
// place and with a separate prediction, without writing past the block; a
// declined block is left untouched and low magnitudes never decline.
func TestInverseAddDCT64x64HBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1064))
	const n = 64
	const stride = 2 * (n + 8)
	for _, bd := range []uint8{10, 12} {
		coeff := make([]int32, 32*32)
		raw := make([]int32, n*n)
		pred := make([]byte, n*stride)
		got := make([]byte, n*stride)
		want := make([]byte, n*stride)
		maxV := 1<<bd - 1
		accepted, declined, lowDeclined, low := 0, 0, 0, 0
		for trial := 0; trial < 4000; trial++ {
			clear(coeff)
			activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
			switch trial % 4 {
			case 0:
				activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
			case 1:
				activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
			}
			mode := trial % 6
			mag := []int{64, 1024, 4096, 32767, 70000, 300000}[mode]
			for c := 0; c < activeCols; c++ {
				for r := 0; r < activeRows; r++ {
					if rng.Intn(3) == 0 {
						continue
					}
					coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
				}
			}
			for i := 0; i < len(pred); i += 2 {
				v := rng.Intn(maxV + 1)
				if trial%7 == 0 {
					v = maxV * rng.Intn(2)
				}
				binary.LittleEndian.PutUint16(pred[i:], uint16(v))
			}
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: n, Height: n}, TypeDCTDCT, bd, activeRows, activeCols); err != nil {
				t.Fatal(err)
			}
			copy(want, pred)
			for r := 0; r < n; r++ {
				for c := 0; c < n; c++ {
					p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
					v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
					binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
				}
			}
			copy(got, pred)
			ok := InverseAddDCT64x64HBD(got, stride, got, stride, coeff, 32, activeRows, activeCols, bd)
			if mode <= 1 {
				low++
			}
			if !ok {
				declined++
				if mode <= 1 {
					lowDeclined++
				}
				for i := range got {
					if got[i] != pred[i] {
						t.Fatalf("bd %d trial %d: declined block modified dst at %d", bd, trial, i)
					}
				}
				continue
			}
			accepted++
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
				}
			}
			for i := range got {
				got[i] = 0xa5
			}
			if !InverseAddDCT64x64HBD(got, stride, pred, stride, coeff, 32, activeRows, activeCols, bd) {
				t.Fatalf("bd %d trial %d: separate-pred call declined after the in-place call accepted", bd, trial)
			}
			for r := 0; r < n; r++ {
				for c := 0; c < stride; c++ {
					w := want[r*stride+c]
					if c >= 2*n {
						w = 0xa5
					}
					if got[r*stride+c] != w {
						t.Fatalf("bd %d trial %d: separate-pred mismatch at row %d byte %d", bd, trial, r, c)
					}
				}
			}
		}
		t.Logf("bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", bd, accepted, declined, lowDeclined, low)
		if lowDeclined != 0 || accepted == 0 {
			t.Fatalf("bd %d: %d low-magnitude declines, %d accepted", bd, lowDeclined, accepted)
		}
	}
}

// itxFused8Oracle is the staged path the fused 8x8 kernel replaces.
func itxFused8Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 8*8)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 8, Height: 8}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*8+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT8x8MatchesStagedPath: every (rows, cols) shape, byte-exact
// against the staged path; declined blocks untouched; no declines at
// +-64 / +-1024 / +-4096 (eight coefficients of +-4096 stay under the guard).
func TestInverseAddDCT8x8MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 8 + 8
	coeff := make([]int32, 8*8)
	pred := make([]byte, 8*stride)
	got := make([]byte, 8*stride)
	want := make([]byte, 8*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 30000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(8), 1+rng.Intn(8)
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*8+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused8Oracle(t, want, stride, pred, stride, coeff, 8, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT8x8(got, stride, got, stride, coeff, 8, activeRows, activeCols)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT8x8(got, stride, pred, stride, coeff, 8, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 8; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT8x8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 8*8)
	for c := 0; c < 8; c++ {
		for r := 0; r < 8; r++ {
			coeff[c*8+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 8*8)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	raw := make([]int32, 8*8)
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT8x8(dst, 8, dst, 8, coeff, 8, 8, 8) {
				b.Fatal("declined")
			}
		}
	})
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 8, Size{Width: 8, Height: 8}, TypeDCTDCT, 8, 8, 8); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// TestInverseAddDCT32x64GuardEdge drives the 32-coded-column row strips (the
// split forms, whose row guard is checked after the even half has run) with
// one coefficient row whose L1 is exactly the guard bound and one past it:
// past the bound the block declines with dst untouched; at the bound it
// declines untouched or matches the staged path byte for byte.
func TestInverseAddDCT32x64GuardEdge(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	for _, n := range []int{32, 64} {
		stride := n + 8
		coeff := make([]int32, 32*32)
		pred := make([]byte, n*stride)
		got := make([]byte, n*stride)
		want := make([]byte, n*stride)
		accepted := 0
		for trial := 0; trial < 400; trial++ {
			clear(coeff)
			l1 := itx16ColumnL1Bound + trial%2
			row := rng.Intn(8)
			activeRows := 24 + rng.Intn(9)
			if row >= activeRows {
				row = 0
			}
			// Split l1 over the 32 columns of one row, random signs.
			left := l1
			for c := 0; c < 32; c++ {
				v := left / (32 - c)
				if c < 31 {
					v += rng.Intn(v/2+1) - v/4
				}
				v = min(v, left)
				left -= v
				if rng.Intn(2) == 0 {
					v = -v
				}
				coeff[c*32+row] = int32(v)
			}
			for c := 0; c < 32; c++ {
				for r := 0; r < activeRows; r++ {
					if r != row && rng.Intn(4) == 0 {
						coeff[c*32+r] = int32(rng.Intn(65) - 32)
					}
				}
			}
			for i := range pred {
				pred[i] = byte(rng.Intn(256))
			}
			copy(got, pred)
			var ok bool
			if n == 32 {
				ok = InverseAddDCT32x32Full(got, stride, got, stride, coeff, 32, activeRows, 32)
			} else {
				ok = InverseAddDCT64x64(got, stride, got, stride, coeff, 32, activeRows, 32)
			}
			if !ok {
				for i := range got {
					if got[i] != pred[i] {
						t.Fatalf("n %d trial %d: declined block modified dst at %d", n, trial, i)
					}
				}
				continue
			}
			if l1 > itx16ColumnL1Bound {
				t.Fatalf("n %d trial %d: row L1 %d past the bound accepted", n, trial, l1)
			}
			accepted++
			copy(want, pred)
			if n == 32 {
				itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, 32)
			} else {
				itxFused64Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, 32)
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("n %d trial %d: mismatch at row %d col %d", n, trial, i/stride, i%stride)
				}
			}
		}
		if accepted == 0 {
			t.Fatalf("n %d: no block at the bound accepted", n)
		}
		t.Logf("n %d: %d of 200 at the bound accepted", n, accepted)
	}
}

// itxFusedHBDOracle is the staged transform at bitDepth plus the raw 16bpc
// add: dst = clip(pred + Round2(raw, 4), 0, 2^bd - 1).
func itxFusedHBDOracle(t *testing.T, n int, bitDepth uint8, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, n*n)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: uint8(n), Height: uint8(n)}, TypeDCTDCT, bitDepth, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	maxV := 1<<bitDepth - 1
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			p := int(binary.LittleEndian.Uint16(pred[r*predStride+2*c:]))
			v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
			binary.LittleEndian.PutUint16(dst[r*dstStride+2*c:], uint16(v))
		}
	}
}

func itxFusedHBD(n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	switch n {
	case 8:
		return InverseAddDCT8x8HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	case 16:
		return InverseAddDCT16x16HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	default:
		return InverseAddDCT32x32HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	}
}

// TestInverseAddDCTHBDMatchesStagedPath: every accepted 10- and 12-bit
// block reproduces the staged transform plus the raw add sample for sample,
// in place and with a separate prediction, without writing past the block;
// a declined block is left untouched.
func TestInverseAddDCTHBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1016))
	for _, n := range []int{8, 16, 32} {
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (n + 8)
			coeff := make([]int32, n*n)
			pred := make([]byte, n*stride)
			got := make([]byte, n*stride)
			want := make([]byte, n*stride)
			maxV := 1<<bd - 1
			trials := map[int]int{8: 6000, 16: 8000, 32: 12000}[n]
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < trials; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(min(n, 8)), 1+rng.Intn(min(n, 8))
				case 1:
					activeRows, activeCols = 1+rng.Intn(min(n, 16)), 1+rng.Intn(min(n, 16))
				}
				mode := trial % 6
				mag := []int{64, 256, 1024, 4096, 32767, 300000}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%7 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				copy(want, pred)
				itxFusedHBDOracle(t, n, bd, want, stride, pred, stride, coeff, n, activeRows, activeCols)
				copy(got, pred)
				ok := itxFusedHBD(n, got, stride, got, stride, coeff, n, activeRows, activeCols, bd)
				if mode <= 1 {
					low++
				}
				if !ok {
					declined++
					if mode <= 1 {
						lowDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("n %d bd %d trial %d: declined block modified dst at %d", n, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n %d bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", n, bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !itxFusedHBD(n, got, stride, pred, stride, coeff, n, activeRows, activeCols, bd) {
					t.Fatalf("n %d bd %d trial %d: separate-pred call declined after the in-place call accepted", n, bd, trial)
				}
				for r := 0; r < n; r++ {
					for c := 0; c < stride; c++ {
						w := want[r*stride+c]
						if c >= 2*n {
							w = 0xa5
						}
						if got[r*stride+c] != w {
							t.Fatalf("n %d bd %d trial %d: separate-pred mismatch at row %d byte %d", n, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("n %d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", n, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("n %d bd %d: %d low-magnitude declines, %d accepted", n, bd, lowDeclined, accepted)
			}
		}
	}
}

// TestInverseAddHybridHBDMatchesStagedPath: every accepted 10- and 12-bit
// 16x16 and 8x8 DCT/ADST/flipped-ADST block reproduces the staged transform
// plus the raw add sample for sample, in place and with a separate
// prediction, without writing past the block; a declined block is left
// untouched, and the other transform types are declined.
func TestInverseAddHybridHBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1017))
	types := []Type{TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
		TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST}
	fused := func(n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, txType Type, activeRows, activeCols int, bd uint8) bool {
		if n == 8 {
			return InverseAdd8x8HybridHBD(dst, dstStride, pred, predStride, coeff, n, txType, activeRows, activeCols, bd)
		}
		return InverseAdd16x16HybridHBD(dst, dstStride, pred, predStride, coeff, n, txType, activeRows, activeCols, bd)
	}
	for _, n := range []int{8, 16} {
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (n + 8)
			coeff := make([]int32, n*n)
			raw := make([]int32, n*n)
			pred := make([]byte, n*stride)
			got := make([]byte, n*stride)
			want := make([]byte, n*stride)
			maxV := 1<<bd - 1
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < 12000; trial++ {
				txType := types[trial%len(types)]
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
				if trial%3 == 0 {
					activeRows, activeCols = 1+rng.Intn(min(n, 4)), 1+rng.Intn(min(n, 8))
				}
				mode := (trial / len(types)) % 6
				mag := []int{64, 256, 1024, 4096, 32767, 300000}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%7 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, txType, bd, activeRows, activeCols); err != nil {
					t.Fatal(err)
				}
				copy(want, pred)
				for r := 0; r < n; r++ {
					for c := 0; c < n; c++ {
						p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
						v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
						binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
					}
				}
				copy(got, pred)
				ok := fused(n, got, stride, got, stride, coeff, txType, activeRows, activeCols, bd)
				if mode <= 1 {
					low++
				}
				if !ok {
					declined++
					if mode <= 1 {
						lowDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("n %d bd %d trial %d: declined block modified dst at %d", n, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n %d bd %d type %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", n, bd, txType, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !fused(n, got, stride, pred, stride, coeff, txType, activeRows, activeCols, bd) {
					t.Fatalf("n %d bd %d trial %d: separate-pred call declined after the in-place call accepted", n, bd, trial)
				}
				for r := 0; r < n; r++ {
					for c := 0; c < stride; c++ {
						w := want[r*stride+c]
						if c >= 2*n {
							w = 0xa5
						}
						if got[r*stride+c] != w {
							t.Fatalf("n %d bd %d trial %d: separate-pred mismatch at row %d byte %d", n, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("n %d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", n, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("n %d bd %d: %d low-magnitude declines, %d accepted", n, bd, lowDeclined, accepted)
			}
			for _, other := range []Type{TypeDCTDCT, TypeIDTX, TypeVDCT, TypeHDCT} {
				if fused(n, got, stride, pred, stride, coeff, other, 1, 1, bd) {
					t.Fatalf("n %d: type %d accepted", n, other)
				}
			}
		}
	}
}

// itxHybridTypesUnderTest are the 16x16 hybrid types; the 8x8 kernels
// also cover the four ADST/identity pairings (itxHybridTypesUnderTest8).
var itxHybridTypesUnderTest = []Type{
	TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
	TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST, TypeVDCT, TypeHDCT,
}

var itxHybridTypesUnderTest8 = append(append([]Type{}, itxHybridTypesUnderTest...),
	TypeVADST, TypeHADST, TypeVFlipADST, TypeHFlipADST)

// itxFusedHybridOracle is the staged path the fused hybrid kernels replace.
func itxFusedHybridOracle(t *testing.T, n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, txType Type, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, n*n)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, txType, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*n+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

func testInverseAddHybrid(t *testing.T, n int, trials int, fused func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool) {
	rng := rand.New(rand.NewSource(int64(53 + n)))
	stride := n + 8
	coeff := make([]int32, n*n)
	pred := make([]byte, n*stride)
	got := make([]byte, n*stride)
	want := make([]byte, n*stride)
	types := itxHybridTypesUnderTest
	if n == 8 {
		types = itxHybridTypesUnderTest8
	}
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < trials; trial++ {
		clear(coeff)
		txType := types[trial%len(types)]
		activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
		switch (trial / len(types)) % 3 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
		case 1:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		}
		mode := (trial / 7) % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFusedHybridOracle(t, n, want, stride, pred, stride, coeff, txType, activeRows, activeCols)
		copy(got, pred)
		ok := fused(got, stride, got, stride, coeff, n, txType, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (type %d, mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !fused(got, stride, pred, stride, coeff, n, txType, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < n; r++ {
			for c := 0; c < n; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := n; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("%dx%d: accepted %d, declined %d (realistic magnitudes: %d of %d declined)", n, n, accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
	if fused(got, stride, pred, stride, coeff, n, TypeDCTDCT, 1, 1) {
		t.Fatal("DCT_DCT accepted by the hybrid kernel")
	}
	if fused(got, stride, pred, stride, coeff, n, TypeIDTX, 1, 1) {
		t.Fatal("IDTX accepted by the hybrid kernel")
	}
}

// TestInverseAdd16x16HybridMatchesStagedPath: every hybrid type, every
// (rows, cols) shape and form pairing, byte-exact against the staged path.
func TestInverseAdd16x16HybridMatchesStagedPath(t *testing.T) {
	testInverseAddHybrid(t, 16, 40000, InverseAdd16x16Hybrid)
}

// TestInverseAdd8x8HybridMatchesStagedPath is the 8x8 counterpart.
func TestInverseAdd8x8HybridMatchesStagedPath(t *testing.T) {
	testInverseAddHybrid(t, 8, 40000, InverseAdd8x8Hybrid)
}

func BenchmarkInverseAddHybrid(b *testing.B) {
	for _, shape := range []struct {
		name       string
		n          int
		rows, cols int
	}{{"16x16/low4", 16, 4, 4}, {"16x16/full", 16, 16, 16}, {"8x8/full", 8, 8, 8}} {
		rng := rand.New(rand.NewSource(5))
		n := shape.n
		coeff := make([]int32, n*n)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*n+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, n*n)
		raw := make([]int32, n*n)
		b.Run(shape.name+"/adst-adst/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				var ok bool
				if n == 16 {
					ok = InverseAdd16x16Hybrid(dst, n, dst, n, coeff, n, TypeADSTADST, shape.rows, shape.cols)
				} else {
					ok = InverseAdd8x8Hybrid(dst, n, dst, n, coeff, n, TypeADSTADST, shape.rows, shape.cols)
				}
				if !ok {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/adst-adst/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, TypeADSTADST, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// itxFusedRectOracle is the staged path the fused rectangular kernels
// replace: the raw column output, Round2(., 4), added to pred and saturated.
func itxFusedRectOracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, size Size, activeRows, activeCols int) {
	t.Helper()
	w, h := int(size.Width), int(size.Height)
	raw := make([]int32, w*h)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, size, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < h; r++ {
		for c := 0; c < w; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*w+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCTRectMatchesStagedPath: every (rows, cols) shape of the
// 16x32 and 32x16 blocks, every network form pairing, byte-exact against the
// staged path; declined blocks untouched; no declines at +-64 / +-1024;
// magnitudes up to 2^29 exercise the guards and the rect2 pre-clamp.
func TestInverseAddDCTRectMatchesStagedPath(t *testing.T) {
	for _, tc := range []struct {
		name string
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{"16x32", 16, 32, InverseAddDCT16x32},
		{"32x16", 32, 16, InverseAddDCT32x16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(61 + tc.w)))
			size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
			stride := tc.w + 8
			coeff := make([]int32, tc.w*tc.h)
			pred := make([]byte, tc.h*stride)
			got := make([]byte, tc.h*stride)
			want := make([]byte, tc.h*stride)
			accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
			for trial := 0; trial < 40000; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(tc.h), 1+rng.Intn(tc.w)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
				case 1:
					activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
				}
				mode := trial % 6
				mag := []int{64, 1024, 4096, 32767, 70000, 1 << 29}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*tc.h+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := range pred {
					pred[i] = byte(rng.Intn(256))
				}
				copy(want, pred)
				itxFusedRectOracle(t, want, stride, pred, stride, coeff, tc.h, size, activeRows, activeCols)
				copy(got, pred)
				ok := tc.fn(got, stride, got, stride, coeff, tc.h, activeRows, activeCols)
				if mode <= 1 {
					realistic++
				}
				if !ok {
					declined++
					if mode <= 1 {
						realisticDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
							trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
					}
				}
				for i := range got {
					got[i] = byte(0xa5)
				}
				if !tc.fn(got, stride, pred, stride, coeff, tc.h, activeRows, activeCols) {
					t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
				}
				for r := 0; r < tc.h; r++ {
					for c := 0; c < tc.w; c++ {
						if got[r*stride+c] != want[r*stride+c] {
							t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
						}
					}
					for c := tc.w; c < stride; c++ {
						if got[r*stride+c] != 0xa5 {
							t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
						}
					}
				}
			}
			t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
			if realisticDeclined != 0 {
				t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
			}
		})
	}
}

func BenchmarkInverseAddDCTRect(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	for _, tc := range []struct {
		name string
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{"16x32", 16, 32, InverseAddDCT16x32},
		{"32x16", 32, 16, InverseAddDCT32x16},
	} {
		size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
		for _, shape := range []struct {
			name       string
			rows, cols int
		}{{"low4", 4, 4}, {"low8", 8, 8}, {"full", tc.h, tc.w}} {
			coeff := make([]int32, tc.w*tc.h)
			for c := 0; c < shape.cols; c++ {
				for r := 0; r < shape.rows; r++ {
					coeff[c*tc.h+r] = int32(rng.Intn(2049) - 1024)
				}
			}
			dst := make([]byte, tc.w*tc.h)
			for i := range dst {
				dst[i] = byte(rng.Intn(256))
			}
			raw := make([]int32, tc.w*tc.h)
			b.Run(tc.name+"/"+shape.name+"/fused", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if !tc.fn(dst, tc.w, dst, tc.w, coeff, tc.h, shape.rows, shape.cols) {
						b.Fatal("declined")
					}
				}
			})
			b.Run(tc.name+"/"+shape.name+"/staged-transform-only", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if err := InverseBlockBitDepthRawBounded(raw, coeff, tc.h, size, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// TestInverseAddDCTRectScaleEdges: a lone coefficient across the rect2 clamp
// boundary (scale(46341) is the first value past the int16 row clamp) and the
// pre-clamp boundary, at several positions, against the staged path.
func TestInverseAddDCTRectScaleEdges(t *testing.T) {
	for _, tc := range []struct {
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{16, 32, InverseAddDCT16x32},
		{32, 16, InverseAddDCT32x16},
	} {
		size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
		pred := make([]byte, tc.w*tc.h)
		for i := range pred {
			pred[i] = byte(i * 7)
		}
		for _, v := range []int32{0, 1, 2, 100, 32767, 46339, 46340, 46341, 46342, 65535, 1 << 20, 1<<22 - 1, 1 << 22, 1<<22 + 1, 1 << 30, 1<<31 - 1} {
			for _, sign := range []int32{1, -1} {
				for _, pos := range [][2]int{{0, 0}, {1, 0}, {0, 1}, {3, 5}} {
					coeff := make([]int32, tc.w*tc.h)
					coeff[pos[1]*tc.h+pos[0]] = sign * v
					want := append([]byte(nil), pred...)
					itxFusedRectOracle(t, want, tc.w, pred, tc.w, coeff, tc.h, size, pos[0]+1, pos[1]+1)
					got := append([]byte(nil), pred...)
					if !tc.fn(got, tc.w, got, tc.w, coeff, tc.h, pos[0]+1, pos[1]+1) {
						continue // declined by the guard: the staged path serves it
					}
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("%dx%d coeff %d at row %d col %d: mismatch at %d: fused %d staged %d", tc.w, tc.h, sign*v, pos[0], pos[1], i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestInverseAddDCTRect2HBDMatchesStagedPath: every accepted 10- and 12-bit
// 16x32 and 32x16 DCT_DCT block reproduces the staged transform plus the raw
// add sample for sample, in place and with a separate prediction, without
// writing past the block; a declined block is left untouched. A block with
// a single coefficient whose 1/sqrt(2)-scaled value leaves int16 but stays
// inside the 10- and 12-bit row clamps (bd + 8 bits) is kept by the staged
// path; the fused load saturates it to +-32767, above the L1 guard, so the
// fused path must decline it.
func TestInverseAddDCTRect2HBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1032))
	for _, shape := range [][2]int{{16, 32}, {32, 16}} {
		w, h := shape[0], shape[1]
		fused := InverseAddDCT16x32HBD
		if w == 32 {
			fused = InverseAddDCT32x16HBD
		}
		cs := min(h, 32)
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (w + 8)
			coeff := make([]int32, w*cs)
			raw := make([]int32, w*h)
			pred := make([]byte, h*stride)
			got := make([]byte, h*stride)
			want := make([]byte, h*stride)
			maxV := 1<<bd - 1
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < 8000; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(h), 1+rng.Intn(w)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
				case 1:
					activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
				}
				mode := trial % 7
				mag := []int{64, 1024, 4096, 32767, 0, 300000, 1 << 23}[mode]
				// Mode 4: one coefficient whose scaled value |c|*181/256 lies
				// between int16 and the row clamp (46341 < |c| < 185000 keeps it
				// under 2^17 - 1, the 10-bit clamp).
				between := mode == 4
				if between {
					v := int32(47000 + rng.Intn(133000))
					if rng.Intn(2) == 0 {
						v = -v
					}
					coeff[rng.Intn(activeCols)*cs+rng.Intn(activeRows)] = v
				}
				for c := 0; c < activeCols && !between; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*cs+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%5 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				if err := InverseBlockBitDepthRawBounded(raw, coeff, cs, Size{Width: uint8(w), Height: uint8(h)}, TypeDCTDCT, bd, activeRows, activeCols); err != nil {
					t.Fatal(err)
				}
				copy(want, pred)
				for r := 0; r < h; r++ {
					for c := 0; c < w; c++ {
						p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
						v := min(max(p+int((raw[r*w+c]+8)>>4), 0), maxV)
						binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
					}
				}
				copy(got, pred)
				ok := fused(got, stride, got, stride, coeff, cs, activeRows, activeCols, bd)
				if mode <= 1 {
					low++
				}
				if between && ok {
					t.Fatalf("%dx%d bd %d trial %d: a coefficient past the int16 range after the rect2 scale was accepted", w, h, bd, trial)
				}
				if !ok {
					declined++
					if mode <= 1 {
						lowDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("%dx%d bd %d trial %d: declined block modified dst at %d", w, h, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("%dx%d bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", w, h, bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !fused(got, stride, pred, stride, coeff, cs, activeRows, activeCols, bd) {
					t.Fatalf("%dx%d bd %d trial %d: separate-pred call declined after the in-place call accepted", w, h, bd, trial)
				}
				for r := 0; r < h; r++ {
					for c := 0; c < stride; c++ {
						wv := want[r*stride+c]
						if c >= 2*w {
							wv = 0xa5
						}
						if got[r*stride+c] != wv {
							t.Fatalf("%dx%d bd %d trial %d: separate-pred mismatch at row %d byte %d", w, h, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("%dx%d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", w, h, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("%dx%d bd %d: %d low-magnitude declines, %d accepted", w, h, bd, lowDeclined, accepted)
			}
		}
	}
}

type itxSmallFused func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool

// testInverseAddSmall checks a generated w x h kernel set against the
// staged transform and residual add for every type but IDTX: coefficients
// inside the coded extent (zero past it, as the decoder's coefficient
// buffers are), magnitudes past the int16 clamps, and separate pred/dst
// buffers. Realistic magnitudes must never decline.
func testInverseAddSmall(t *testing.T, w, h int, fused itxSmallFused) {
	rng := rand.New(rand.NewSource(int64(97 + w*h + w)))
	stride := w + 8
	coeff := make([]int32, w*h)
	raw := make([]int32, w*h)
	pred := make([]byte, h*stride)
	got := make([]byte, h*stride)
	want := make([]byte, h*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 60000; trial++ {
		txType := Type(trial % int(TypeCount))
		if txType == TypeIDTX || !txType.Supported(Size{Width: uint8(w), Height: uint8(h)}) {
			continue
		}
		// A 16-point side puts the block in the 16x16 transform sets,
		// which have no V_ADST, H_ADST, V_FLIPADST or H_FLIPADST: the
		// kernels decline those (the staged path runs them).
		if max(w, h) == 16 && txType >= TypeVADST {
			if fused(got, stride, pred, stride, coeff, h, txType, 1, 1) {
				t.Fatalf("type %d accepted at %dx%d", txType, w, h)
			}
			continue
		}
		activeRows, activeCols := 1+rng.Intn(h), 1+rng.Intn(w)
		mode := (trial / int(TypeCount)) % 6
		mag := []int{16, 256, 1024, 4096, 32767, 70000}[mode]
		clear(coeff)
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) != 0 {
					coeff[c*h+r] = int32(rng.Intn(2*mag+1) - mag)
				}
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		if err := InverseBlockBitDepthRawBounded(raw, coeff, h, Size{Width: uint8(w), Height: uint8(h)}, txType, 8, activeRows, activeCols); err != nil {
			t.Fatal(err)
		}
		copy(want, pred)
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				v := int(pred[r*stride+c]) + int((int64(raw[r*w+c])+8)>>4)
				want[r*stride+c] = byte(min(max(v, 0), 255))
			}
		}
		copy(got, pred)
		ok := fused(got, stride, got, stride, coeff, h, txType, activeRows, activeCols)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst", trial)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (type %d, mag %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mag, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = 0xa5
		}
		if !fused(got, stride, pred, stride, coeff, h, txType, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < h; r++ {
			for c := 0; c < stride; c++ {
				wv := want[r*stride+c]
				if c >= w {
					wv = 0xa5
				}
				if got[r*stride+c] != wv {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("%dx%d: accepted %d, declined %d (realistic magnitudes: %d of %d declined)", w, h, accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
	if fused(got, stride, pred, stride, coeff, h, TypeIDTX, 1, 1) {
		t.Fatal("IDTX accepted")
	}
}

func TestInverseAdd4x4GenMatchesStaged(t *testing.T) {
	testInverseAddSmall(t, 4, 4, func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
		return InverseAdd4x4(dst, dstStride, pred, predStride, coeff, coeffStride, txType, activeRows)
	})
}
func TestInverseAdd4x8GenMatchesStaged(t *testing.T)  { testInverseAddSmall(t, 4, 8, InverseAdd4x8) }
func TestInverseAdd8x4GenMatchesStaged(t *testing.T)  { testInverseAddSmall(t, 8, 4, InverseAdd8x4) }
func TestInverseAdd4x16GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 4, 16, InverseAdd4x16) }
func TestInverseAdd16x4GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 16, 4, InverseAdd16x4) }
func TestInverseAdd8x16GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 8, 16, InverseAdd8x16) }
func TestInverseAdd16x8GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 16, 8, InverseAdd16x8) }

// invGlueSIMDAvailable reports whether init bound the SIMD glue kernels: always
// on arm64 (NEON is part of the baseline).
func invGlueSIMDAvailable() bool { return true }

// row2TestImpls returns the arm64 adapters directly so the differential test
// always exercises every arm64 kernel for bit-exactness, even the ones the
// dispatcher leaves unbound for performance reasons.
func row2TestImpls() []row2TestFunc {
	return []row2TestFunc{
		{"DCT8", dct8Size, inverseDCT8Row2SIMDAdapter, inverseDCT8Row2PureGo},
		{"DCT16", dct16Size, inverseDCT16Row2SIMDAdapter, inverseDCT16Row2PureGo},
		{"DCT32", dct32Size, inverseDCT32Row2SIMDAdapter, inverseDCT32Row2PureGo},
	}
}

// row4TestImpls returns the arm64 four-row adapters directly so the
// differential test always exercises the four-row kernels for bit-exactness.
func row4TestImpls() []row4TestFunc {
	return []row4TestFunc{
		{"DCT32Row4", dct32Size, inverseDCT32Row4SIMDAdapter, inverseDCT32Row4PureGo},
		{"DCT64Row4", dct64Size, inverseDCT64Row4Impl, inverseDCT64Row4PureGo},
		{"ADST16Row4", adst16Size, inverseADST16Row4SIMDAdapter, inverseADST16Row4PureGo},
		{"ADST16Row4Flip", adst16Size, inverseADST16Row4FlipSIMDAdapter, inverseADST16Row4FlipPureGo},
	}
}
