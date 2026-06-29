package transform

import (
	"math/rand"
	"testing"
)

// TestForwardDCT8x8ImplMatchesPureGo proves the dispatched 8x8 forward DCT
// kernel bit-exact with the portable reference across random 8-bit residual
// ranges and strides.
func TestForwardDCT8x8ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	const resStride, coeffStride = 23, 17
	residual := make([]int16, resStride*16)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255 // 8-bit residual range
		}
		want := make([]int32, coeffStride*16)
		got := make([]int32, coeffStride*16)
		forwardDCT8x8PureGo(want, coeffStride, residual, resStride)
		forwardDCT8x8Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT8x8(b *testing.B) {
	var residual [64]int16
	for i := range residual {
		residual[i] = int16(i*7%400) - 200
	}
	var coeff [64]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT8x8Impl(coeff[:], 8, residual[:], 8)
	}
}

// TestForwardDCT4x4ImplMatchesPureGo proves the dispatched 4x4 forward DCT
// kernel bit-exact with the portable reference across random 8-bit residual
// ranges and strides.
func TestForwardDCT4x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	const resStride, coeffStride = 13, 9
	residual := make([]int16, resStride*8)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255 // 8-bit residual range
		}
		want := make([]int32, coeffStride*8)
		got := make([]int32, coeffStride*8)
		forwardDCT4x4PureGo(want, coeffStride, residual, resStride)
		forwardDCT4x4Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT4x4(b *testing.B) {
	var residual [16]int16
	for i := range residual {
		residual[i] = int16(i*29%400) - 200
	}
	var coeff [16]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT4x4Impl(coeff[:], 4, residual[:], 4)
	}
}

// TestForwardDCT16x16ImplMatchesPureGo proves the dispatched 16x16 forward
// DCT kernel bit-exact with the portable reference across random 8-bit
// residual ranges and strides.
func TestForwardDCT16x16ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(83))
	const resStride, coeffStride = 37, 19
	residual := make([]int16, resStride*32)
	for trial := range 2000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*32)
		got := make([]int32, coeffStride*32)
		forwardDCT16x16PureGo(want, coeffStride, residual, resStride)
		forwardDCT16x16Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT16x16(b *testing.B) {
	var residual [256]int16
	for i := range residual {
		residual[i] = int16(i*11%400) - 200
	}
	var coeff [256]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT16x16Impl(coeff[:], 16, residual[:], 16)
	}
}

// TestForwardDCT32x32ImplMatchesPureGo proves the dispatched 32x32 forward
// DCT kernel bit-exact with the portable reference across random 8-bit
// residual ranges and strides.
func TestForwardDCT32x32ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(89))
	const resStride, coeffStride = 41, 35
	residual := make([]int16, resStride*64)
	for trial := range 1000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*64)
		got := make([]int32, coeffStride*64)
		forwardDCT32x32PureGo(want, coeffStride, residual, resStride)
		forwardDCT32x32Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT32x32(b *testing.B) {
	var residual [1024]int16
	for i := range residual {
		residual[i] = int16(i*13%400) - 200
	}
	var coeff [1024]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT32x32Impl(coeff[:], 32, residual[:], 32)
	}
}

func BenchmarkForwardDCT64x64(b *testing.B) {
	var residual [64 * 64]int16
	for i := range residual {
		residual[i] = int16(i*17%400) - 200
	}
	var coeff [32 * 32]int32
	b.ReportAllocs()
	for b.Loop() {
		_ = ForwardDCT64x64(coeff[:], 32, residual[:], 64)
	}
}

func BenchmarkForwardDCT32x64(b *testing.B) {
	var residual [32 * 64]int16
	for i := range residual {
		residual[i] = int16(i*19%400) - 200
	}
	var coeff [32 * 32]int32
	b.ReportAllocs()
	for b.Loop() {
		_ = ForwardDCT32x64(coeff[:], 32, residual[:], 32)
	}
}

func BenchmarkForwardDCT64x32(b *testing.B) {
	var residual [64 * 32]int16
	for i := range residual {
		residual[i] = int16(i*23%400) - 200
	}
	var coeff [32 * 32]int32
	b.ReportAllocs()
	for b.Loop() {
		_ = ForwardDCT64x32(coeff[:], 32, residual[:], 64)
	}
}

func TestFwdDCT64MatchesPinnedLibaomVectors(t *testing.T) {
	cases := []struct {
		name   string
		cosBit uint
		input  func(i int) int32
		want   [64]int32
	}{
		{
			name:   "impulse_q13",
			cosBit: 13,
			input: func(i int) int32 {
				if i == 0 {
					return 1000
				}
				return 0
			},
			want: [64]int32{707, 1000, 999, 997, 995, 992, 989, 985, 981, 976, 970, 964, 957, 950, 942, 933, 924, 914, 904, 893, 882, 870, 858, 845, 831, 818, 803, 788, 773, 757, 741, 724, 707, 690, 672, 653, 634, 615, 596, 576, 556, 535, 514, 493, 471, 450, 428, 405, 383, 360, 337, 314, 290, 267, 243, 219, 195, 171, 147, 122, 98, 74, 49, 25},
		},
		{
			name:   "dense_q10",
			cosBit: 10,
			input: func(i int) int32 {
				return int32((i*37+11)%511) - 255
			},
			want: [64]int32{-573, -260, -1256, -205, -993, -606, -1000, -898, -2660, -1920, 3592, -359, 586, -82, -205, -81, -472, -910, -1272, 1712, 375, -112, 450, -468, 234, -418, -458, -958, 889, 695, -332, 504, -361, 104, -137, -483, -601, 312, 901, -377, 411, -159, -95, 99, -522, -423, 39, 876, -196, 157, 125, -315, 266, -501, -395, -47, 728, 71, -94, 336, -405, 260, -333, -529},
		},
		{
			name:   "mixed_q11",
			cosBit: 11,
			input: func(i int) int32 {
				if i%7 == 0 {
					return int32(i*13 - 300)
				}
				return int32((i%5)-2) * 17
			},
			want: [64]int32{750, -1907, 55, -524, 150, -451, 108, -343, 5, -482, 23, -404, 146, -589, 186, -767, 139, -2010, 1000, -1624, -133, -199, 21, -346, -83, -819, 478, -90, 115, -214, 210, -492, 227, -726, 205, -1762, 909, -1005, -303, 415, -29, 123, -57, 48, -111, -189, -35, -187, 58, -454, -100, -818, 580, -1407, 843, -347, -458, 944, -69, 415, -92, 262, -112, 3},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var in, got [64]int32
			for i := range in {
				in[i] = tc.input(i)
			}
			fwdDCT64(&in, &got, fwdCospiForBit(tc.cosBit), tc.cosBit)
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("coeff[%d]=%d want %d", i, got[i], tc.want[i])
				}
			}
		})
	}
}
