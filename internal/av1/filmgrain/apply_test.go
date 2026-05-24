package filmgrain

import (
	"errors"
	"testing"
)

func TestApplyLumaRowNoOverlap(t *testing.T) {
	const (
		width  = 4
		height = 2
		stride = 8
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0xd9, 0, 0, 0, 0, 64)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		BitDepth:     8,
		ScalingShift: 8,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 116 {
		t.Fatalf("first sample=%d want 116", got)
	}
	if got := dst[1]; got != 100 {
		t.Fatalf("neighbor sample=%d want 100", got)
	}
	if got := dst[stride]; got != 100 {
		t.Fatalf("next-row sample=%d want 100", got)
	}
}

func TestApplyLumaRowLeftOverlap(t *testing.T) {
	const (
		width  = 34
		height = 1
		stride = 40
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0xec, 0, 0, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0xd9, 1, 0, 0, 0, 100)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[32]; got != 124 {
		t.Fatalf("left-overlap sample=%d want 124", got)
	}
	if got := dst[33]; got != 100 {
		t.Fatalf("second overlap sample=%d want 100", got)
	}
}

func TestApplyLumaRowTopOverlap(t *testing.T) {
	const (
		width  = 1
		height = 2
		stride = 8
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0x6b, 0, 0, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0xd9, 0, 1, 0, 0, 100)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		Row:          1,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 124 {
		t.Fatalf("top-overlap sample=%d want 124", got)
	}
	if got := dst[stride]; got != 100 {
		t.Fatalf("second row sample=%d want 100", got)
	}
}

func TestApplyLumaRowCornerOverlap(t *testing.T) {
	const (
		width  = 34
		height = 2
		stride = 40
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0xb5, 0, 0, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0x6b, 1, 0, 0, 0, 100)
	setTestLumaRowGrain(t, grain, 0xec, 0, 1, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0xd9, 1, 1, 0, 0, 60)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		Row:          1,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[32]; got != 126 {
		t.Fatalf("corner-overlap sample=%d want 126", got)
	}
}

func TestApplyLumaRowRejectsInvalidInputs(t *testing.T) {
	dst, src := testLumaRowBuffers(4, 2, 8, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	params := LumaRowParams{Width: 4, Height: 2, Stride: 8, BitDepth: 8, ScalingShift: 8}
	tests := []struct {
		name   string
		dst    []uint16
		src    []uint16
		grain  []int16
		lut    []uint8
		params LumaRowParams
	}{
		{name: "short-dst", dst: dst[:8], src: src, grain: grain, lut: lut[:], params: params},
		{name: "short-src", dst: dst, src: src[:8], grain: grain, lut: lut[:], params: params},
		{name: "short-grain", dst: dst, src: src, grain: grain[:LumaGrainSamples-1], lut: lut[:], params: params},
		{name: "short-lut", dst: dst, src: src, grain: grain, lut: lut[:ScalingLUTSize-1], params: params},
		{name: "width", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowWidth(params, 0)},
		{name: "height", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowHeight(params, LumaBlockSize+1)},
		{name: "stride", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowStride(params, 3)},
		{name: "row", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowIndex(params, -1)},
		{name: "bit-depth", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowBitDepth(params, 9)},
		{name: "scaling-shift", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowScalingShift(params, 7)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ApplyLumaRow(tt.dst, tt.src, tt.grain, tt.lut, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("ApplyLumaRow err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func TestApplyLumaRowAllocs(t *testing.T) {
	const (
		width  = 64
		height = 32
		stride = 64
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := testPatternLumaRowGrain()
	lut := testLumaRowScaling(64)
	params := LumaRowParams{
		Seed:                  0x1234,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		Overlap:               true,
		ClipToRestrictedRange: true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyLumaRow allocated: %f", allocs)
	}
}

func BenchmarkApplyLumaRow(b *testing.B) {
	const (
		width  = 64
		height = 32
		stride = 64
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := testPatternLumaRowGrain()
	lut := testLumaRowScaling(64)
	params := LumaRowParams{
		Seed:         0x1234,
		Width:        width,
		Height:       height,
		Stride:       stride,
		Row:          1,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
			b.Fatal(err)
		}
	}
	if dst[0] == 0 {
		b.Fatal("unexpected zero sample")
	}
}

func testLumaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func testLumaRowScaling(value uint8) [ScalingLUTSize]uint8 {
	var lut [ScalingLUTSize]uint8
	for i := range lut {
		lut[i] = value
	}
	return lut
}

func testPatternLumaRowGrain() []int16 {
	grain := make([]int16, LumaGrainSamples)
	for y := 0; y < LumaGrainHeight; y++ {
		for x := 0; x < LumaGrainWidth; x++ {
			grain[y*LumaGrainWidth+x] = int16((x+y)%64 - 32)
		}
	}
	return grain
}

func setTestLumaRowGrain(t *testing.T, grain []int16, offset uint8, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	col := lumaGrainOffset(int(offset>>4)) + x + LumaBlockSize*blockCol
	row := lumaGrainOffset(int(offset&0x0f)) + y + LumaBlockSize*blockRow
	if col < 0 || col >= LumaGrainWidth || row < 0 || row >= LumaGrainHeight {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*LumaGrainWidth+col] = value
}

func withLumaRowWidth(params LumaRowParams, width int) LumaRowParams {
	params.Width = width
	return params
}

func withLumaRowHeight(params LumaRowParams, height int) LumaRowParams {
	params.Height = height
	return params
}

func withLumaRowStride(params LumaRowParams, stride int) LumaRowParams {
	params.Stride = stride
	return params
}

func withLumaRowIndex(params LumaRowParams, row int) LumaRowParams {
	params.Row = row
	return params
}

func withLumaRowBitDepth(params LumaRowParams, bitDepth uint8) LumaRowParams {
	params.BitDepth = bitDepth
	return params
}

func withLumaRowScalingShift(params LumaRowParams, shift uint8) LumaRowParams {
	params.ScalingShift = shift
	return params
}
