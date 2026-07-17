package transform

import (
	"math"
	"testing"
)

func TestInverseADST1DRoundTripMatchesLibaomShape(t *testing.T) {
	for _, length := range []int{4, 8, 16} {
		input := make([]int32, length)
		for i := range input {
			input[i] = int32(((i*41+23)%37)-18) * 3
		}
		got := referenceADST1DInt(input)
		inverseADST1D(got, 1, length, minInt16, maxInt16)
		normalizationShift := getMaxBit(length) - 1
		for i := range got {
			normalized := clipInt32(roundShift(int64(got[i]), normalizationShift))
			if diff := abs32(normalized - input[i]); diff > 2 {
				t.Fatalf("length=%d coeff=%d got=%d normalized=%d want=%d diff=%d", length, i, got[i], normalized, input[i], diff)
			}
		}
	}
}

func TestInverseADST1DStrided(t *testing.T) {
	compact := []int32{100, -20, 30, 7}
	want := append([]int32(nil), compact...)
	inverseADST1D(want, 1, 4, minInt16, maxInt16)

	values := []int32{100, 99, -20, 99, 30, 99, 7}
	inverseADST1D(values, 2, 4, minInt16, maxInt16)
	for i := range want {
		if values[i*2] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i*2, values[i*2], want[i])
		}
	}
	for _, i := range []int{1, 3, 5} {
		if values[i] != 99 {
			t.Fatalf("sentinel values[%d]=%d want 99", i, values[i])
		}
	}
}

func TestInverseFlipADST1DReversesADST(t *testing.T) {
	for _, length := range []int{4, 8, 16} {
		input := make([]int32, length)
		for i := range input {
			input[i] = int32(((i*17+9)%29)-14) * 6
		}
		adst := append([]int32(nil), input...)
		flip := append([]int32(nil), input...)
		inverseADST1D(adst, 1, length, minInt16, maxInt16)
		inverseFlipADST1D(flip, 1, length, minInt16, maxInt16)
		for i := range length {
			if flip[i] != adst[length-1-i] {
				t.Fatalf("length=%d flip[%d]=%d want ADST[%d]=%d", length, i, flip[i], length-1-i, adst[length-1-i])
			}
		}
	}
}

func TestInverseADST4OverflowRegressionFromLibaom(t *testing.T) {
	values := []int32{300000, 0, 300000, 300000}
	inverseADST1D(values, 1, 4, minInt32, maxInt32)
	if values[0] <= 0 {
		t.Fatalf("ADST4 overflow regression values[0]=%d want > 0", values[0])
	}
}

func referenceADST1DInt(input []int32) []int32 {
	size := len(input)
	if size == 4 {
		return referenceFADST4New(input)
	}
	out := make([]int32, size)
	for k := range size {
		sum := 0.0
		for n := range size {
			angle := math.Pi * float64((2*n+1)*(2*k+1)) / float64(4*size)
			sum += float64(input[n]) * math.Sin(angle)
		}
		out[k] = int32(math.Round(sum))
	}
	return out
}

func referenceFADST4New(input []int32) []int32 {
	x0 := int64(input[0])
	x1 := int64(input[1])
	x2 := int64(input[2])
	x3 := int64(input[3])
	if x0|x1|x2|x3 == 0 {
		return []int32{0, 0, 0, 0}
	}

	s0 := int64(5283) * x0
	s1 := int64(15212) * x0
	s2 := int64(9929) * x1
	s3 := int64(5283) * x1
	s4 := int64(13377) * x2
	s5 := int64(15212) * x3
	s6 := int64(9929) * x3
	s7 := x0 + x1 - x3

	x0 = s0 + s2 + s5
	x1 = int64(13377) * s7
	x2 = s1 - s3 + s6
	x3 = s4

	s0 = x0 + x3
	s1 = x1
	s2 = x2 - x3
	s3 = x2 - x0 + x3

	return []int32{
		clipInt32(roundShift(s0, 14)),
		clipInt32(roundShift(s1, 14)),
		clipInt32(roundShift(s2, 14)),
		clipInt32(roundShift(s3, 14)),
	}
}

func getMaxBit(x int) int {
	maxBit := -1
	for x != 0 {
		x >>= 1
		maxBit++
	}
	return maxBit
}

// FuzzInverseADSTBlock exercises the inverse ADST/FlipADST paths through
// InverseBlock across all square and rectangular sizes that ADST supports,
// including hybrid combinations with DCT and the vertical/horizontal-only
// variants. Coefficient magnitudes are pushed near int16 limits to stress the
// internal clipRange clamps in inverseADST8/16.
func FuzzInverseADSTBlock(f *testing.F) {
	f.Add(uint8(0), uint8(0), int16(0), int16(0), int16(0))
	f.Add(uint8(1), uint8(0), int16(64), int16(-21), int16(13))
	f.Add(uint8(2), uint8(3), int16(-128), int16(32), int16(-17))
	f.Add(uint8(3), uint8(4), int16(255), int16(-127), int16(63))
	f.Add(uint8(5), uint8(6), int16(1024), int16(-512), int16(256))
	f.Add(uint8(7), uint8(1), int16(32767), int16(-32768), int16(16384))
	f.Add(uint8(8), uint8(7), int16(-32768), int16(32767), int16(-16384))

	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 4, Height: 16},
		{Width: 8, Height: 4},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 16, Height: 4},
		{Width: 16, Height: 8},
		{Width: 16, Height: 16},
	}
	types := []Type{
		TypeADSTDCT,
		TypeDCTADST,
		TypeADSTADST,
		TypeFlipADSTDCT,
		TypeDCTFlipADST,
		TypeFlipADSTFlipADST,
		TypeADSTFlipADST,
		TypeFlipADSTADST,
		TypeVADST,
		TypeHADST,
		TypeVFlipADST,
		TypeHFlipADST,
	}

	f.Fuzz(func(t *testing.T, rawSize uint8, rawType uint8, c0 int16, c1 int16, c2 int16) {
		size := sizes[int(rawSize)%len(sizes)]
		typ := types[int(rawType)%len(types)]
		if !typ.Supported(size) {
			return
		}

		width := int(size.Width)
		height := int(size.Height)
		coeffStride := height + 2
		dstStride := width + 3
		coeff := make([]int32, coeffStride*width)
		dst := make([]int16, dstStride*height)
		scratchLen, err := ScratchLenForType(typ, size)
		if err != nil {
			t.Fatalf("ScratchLenForType err=%v", err)
		}
		scratch := make([]int32, scratchLen+3)

		seeds := [3]int16{c0, c1, c2}
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				base := int32(seeds[(row+col)%3])
				if (row^col)&1 != 0 {
					base = -base
				}
				coeff[col*coeffStride+row] = base
			}
		}

		const sentinel = int16(0x6c6c)
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				dst[row*dstStride+col] = sentinel
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			scratch[i] = 0x5b5b5b5b
		}

		if err := InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, typ); err != nil {
			t.Fatalf("InverseBlock size=%dx%d type=%d err=%v", size.Width, size.Height, typ, err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != sentinel {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			if scratch[i] != 0x5b5b5b5b {
				t.Fatalf("scratch padding[%d]=%d want sentinel", i, scratch[i])
			}
		}
	})
}

// TestInverseADSTCoreMatchesTo asserts the codegen sources are bit-for-bit
// identical to their canonical transforms over random and extremal inputs
// across every supported column-stage clamp bound.
func TestInverseADSTCoreMatchesTo(t *testing.T) {
	cases := []struct {
		name   string
		length int
		want   func([]int32, int, int32, int32)
		core   func([]int32, int, int32, int32)
	}{
		{name: "ADST8", length: 8, want: inverseADST8, core: inverseADST8Core},
		{name: "ADST16", length: 16, want: inverseADST16, core: inverseADST16Core},
	}
	edges := []int32{0, 1, -1, 7, -7, -32768, 32767, -131072, 131071, -524288, 524287}
	bounds := []int32{1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19}
	rng := uint64(0x9e3779b97f4a7c15)
	next := func() int32 {
		rng ^= rng << 13
		rng ^= rng >> 7
		rng ^= rng << 17
		return int32(rng)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, b := range bounds {
				min, max := -b, b-1
				clamp := func(v int32) int32 {
					if v < min {
						return min
					}
					if v > max {
						return max
					}
					return v
				}
				for iter := 0; iter < 20000; iter++ {
					var in [16]int32
					for i := 0; i < tc.length; i++ {
						var raw int32
						if iter&1 == 0 {
							raw = edges[int(uint32(next()))%len(edges)]
						} else {
							raw = next() >> (uint(next()) % 13)
						}
						in[i] = clamp(raw)
					}
					want := in
					got := in
					tc.want(want[:tc.length], 1, min, max)
					tc.core(got[:tc.length], 1, min, max)
					if want != got {
						t.Fatalf("bound=%d iter=%d\n in=%v\nwant=%v\n got=%v", b, iter, in, want, got)
					}
				}
			}
		})
	}
}
