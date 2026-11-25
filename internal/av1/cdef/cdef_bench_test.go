package cdef

import (
	"testing"
)

// benchFilterBlockParams is the worst-case 8x8 shape: both primary and
// secondary taps active (so the clip path runs) at 12-bit precision.
func benchFilterBlockParams() ([]uint16, []uint16, int, BlockFilterParams) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 12, 0, 0)
	dst := make([]uint16, 64)
	params := BlockFilterParams{
		PrimaryStrength:   15 << 4,
		SecondaryStrength: 4 << 4,
		Direction:         4,
		PrimaryDamping:    7,
		SecondaryDamping:  7,
		CoeffShift:        4,
		Width:             8,
		Height:            8,
	}
	return dst, input, cdefBlockOrigin(), params
}

// BenchmarkFilterBlockPureGo and BenchmarkFilterBlockDispatch bracket the
// per-kernel speedup: the first always runs the pure-Go reference; the second
// runs whatever the dispatcher resolved (NEON on arm64). Both time only the
// inner kernel.
func BenchmarkFilterBlockPureGo(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockPureGo(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockDispatch(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockImpl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockDispatch4x8(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 8
	b.ReportAllocs()
	for b.Loop() {
		filterBlockImpl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockPureGo4x8(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 8
	b.ReportAllocs()
	for b.Loop() {
		filterBlockPureGo(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockDispatch4x4(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 4
	b.ReportAllocs()
	for b.Loop() {
		filterBlockImpl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockPureGo4x4(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 4
	b.ReportAllocs()
	for b.Loop() {
		filterBlockPureGo(dst, 8, 0, input, origin, params)
	}
}

// benchFilterBlockU8Params is the worst-case 8-bit 8x8 shape: both primary
// and secondary taps active so the clip path runs.
func benchFilterBlockU8Params() ([]byte, []uint16, int, BlockFilterParams) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	params := BlockFilterParams{
		PrimaryStrength:   15,
		SecondaryStrength: 4,
		Direction:         4,
		PrimaryDamping:    5,
		SecondaryDamping:  5,
		CoeffShift:        0,
		Width:             8,
		Height:            8,
	}
	return dst, input, cdefBlockOrigin(), params
}

// BenchmarkFilterBlockU8PureGo / BenchmarkFilterBlockU8Dispatch bracket the
// 8-bit-dst kernel speedup; compare against BenchmarkFilterBlockDispatch to
// see the uint8-store vs uint16-store delta at 8 bits.
func BenchmarkFilterBlockU8PureGo(b *testing.B) {
	dst, input, origin, params := benchFilterBlockU8Params()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8PureGo(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockU8Dispatch(b *testing.B) {
	dst, input, origin, params := benchFilterBlockU8Params()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8Impl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockU8Dispatch4x8(b *testing.B) {
	dst, input, origin, params := benchFilterBlockU8Params()
	params.Width = 4
	params.Height = 8
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8Impl(dst, 8, 0, input, origin, params)
	}
}

// BenchmarkFilterUnitBlocksU8Dispatch runs the 8-bit unit loop over one full
// 64x64 unit of 8x8 blocks through the resolved per-unit dispatch.
func BenchmarkFilterUnitBlocksU8Dispatch(b *testing.B) {
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(i * 29 & 255)
	}
	inputOrigin := VerticalBorder*BStride + HorizontalBorder
	var directions DirectionGrid
	var variances VarianceGrid
	var blocks []BlockPosition
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			directions[by][bx] = uint8((by + bx) & 7)
			variances[by][bx] = int32(bx * by << 10)
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	dst := make([]byte, BStride*BlockSize)
	u := unitFilterParams{
		primaryStrength:   5,
		secondaryStrength: 2,
		damping:           4,
		bwLog2:            3,
		bhLog2:            3,
		blockWidth:        8,
		blockHeight:       8,
		lumaAdjust:        true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := filterUnitBlocksU8(dst, BStride, input, inputOrigin, blocks, &directions, &variances, u); err != nil {
			b.Fatal(err)
		}
	}
}
