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

func TestApplyChromaRowNoOverlap(t *testing.T) {
	const (
		width      = 4
		height     = 2
		stride     = 8
		lumaStride = 8
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0xd9, false, false, 0, 0, 0, 0, 64)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		BitDepth:              8,
		ScalingShift:          8,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 116 {
		t.Fatalf("first sample=%d want 116", got)
	}
	if got := dst[1]; got != 100 {
		t.Fatalf("neighbor sample=%d want 100", got)
	}
}

func TestApplyChromaRowUsesChromaScalingFormula(t *testing.T) {
	const (
		width      = 1
		height     = 1
		stride     = 4
		lumaStride = 4
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	var lut [ScalingLUTSize]uint8
	lut[100] = 128
	setTestChromaRowGrain(t, grain, 0xd9, false, false, 0, 0, 0, 0, 32)

	params := ChromaRowParams{
		Seed:           0,
		Width:          width,
		Height:         height,
		Stride:         stride,
		LumaStride:     lumaStride,
		BitDepth:       8,
		ScalingShift:   8,
		ChromaMult:     32,
		ChromaLumaMult: 32,
		ChromaOffset:   10,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 116 {
		t.Fatalf("chroma-scaled sample=%d want 116", got)
	}
}

func TestApplyChromaRowLeftOverlap420(t *testing.T) {
	const (
		width      = 17
		height     = 1
		stride     = 20
		lumaStride = 40
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0xec, true, true, 0, 0, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0xd9, true, true, 1, 0, 0, 0, 100)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[16]; got != 122 {
		t.Fatalf("left-overlap sample=%d want 122", got)
	}
}

func TestApplyChromaRowTopOverlap420(t *testing.T) {
	const (
		width      = 1
		height     = 1
		stride     = 8
		lumaStride = 8
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0x6b, true, true, 0, 0, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0xd9, true, true, 0, 1, 0, 0, 100)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 122 {
		t.Fatalf("top-overlap sample=%d want 122", got)
	}
}

func TestApplyChromaRowCornerOverlap420(t *testing.T) {
	const (
		width      = 17
		height     = 1
		stride     = 20
		lumaStride = 40
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0xb5, true, true, 0, 0, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0x6b, true, true, 1, 0, 0, 0, 100)
	setTestChromaRowGrain(t, grain, 0xec, true, true, 0, 1, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0xd9, true, true, 1, 1, 0, 0, 60)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[16]; got != 125 {
		t.Fatalf("corner-overlap sample=%d want 125", got)
	}
}

func TestApplyChromaRowClipsRestrictedRange(t *testing.T) {
	const (
		width      = 1
		height     = 1
		stride     = 4
		lumaStride = 4
	)
	dst, src := testChromaRowBuffers(width, height, stride, 250)
	luma := testChromaRowLuma(width, height, lumaStride, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(255)
	setTestChromaRowGrain(t, grain, 0xd9, false, false, 0, 0, 0, 0, 128)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		BitDepth:              8,
		ScalingShift:          8,
		ClipToRestrictedRange: true,
		MatrixIdentity:        true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != ChromaIdentityMax {
		t.Fatalf("restricted sample=%d want %d", got, ChromaIdentityMax)
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

func TestApplyChromaRowRejectsInvalidInputs(t *testing.T) {
	dst, src := testChromaRowBuffers(4, 2, 8, 100)
	luma := testChromaRowLuma(4, 2, 8, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	params := ChromaRowParams{Width: 4, Height: 2, Stride: 8, LumaStride: 8, BitDepth: 8, ScalingShift: 8}
	tests := []struct {
		name   string
		dst    []uint16
		src    []uint16
		luma   []uint16
		grain  []int16
		lut    []uint8
		params ChromaRowParams
	}{
		{name: "short-dst", dst: dst[:8], src: src, luma: luma, grain: grain, lut: lut[:], params: params},
		{name: "short-src", dst: dst, src: src[:8], luma: luma, grain: grain, lut: lut[:], params: params},
		{name: "short-luma", dst: dst, src: src, luma: luma[:8], grain: grain, lut: lut[:], params: params},
		{name: "short-grain", dst: dst, src: src, luma: luma, grain: grain[:ChromaGrainSamples-1], lut: lut[:], params: params},
		{name: "short-lut", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:ScalingLUTSize-1], params: params},
		{name: "width", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowWidth(params, 0)},
		{name: "height", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowHeight(params, LumaBlockSize+1)},
		{name: "stride", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowStride(params, 3)},
		{name: "luma-stride", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowLumaStride(params, 3)},
		{name: "row", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowIndex(params, -1)},
		{name: "bit-depth", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowBitDepth(params, 9)},
		{name: "scaling-shift", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowScalingShift(params, 7)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ApplyChromaRow(tt.dst, tt.src, tt.luma, tt.grain, tt.lut, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("ApplyChromaRow err=%v want %v", err, ErrInvalidParams)
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

func TestApplyChromaRowAllocs(t *testing.T) {
	const (
		width      = 32
		height     = 16
		stride     = 32
		lumaStride = 64
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := testPatternChromaRowGrain()
	lut := testLumaRowScaling(64)
	params := ChromaRowParams{
		Seed:                  0x1234,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ClipToRestrictedRange: true,
		ChromaScalingFromLuma: true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyChromaRow allocated: %f", allocs)
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

func BenchmarkApplyChromaRow(b *testing.B) {
	const (
		width      = 32
		height     = 16
		stride     = 32
		lumaStride = 64
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := testPatternChromaRowGrain()
	lut := testLumaRowScaling(64)
	params := ChromaRowParams{
		Seed:                  0x1234,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
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

func testChromaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func testChromaRowLuma(width int, height int, stride int, subsamplingX bool, subsamplingY bool, value uint16) []uint16 {
	shiftX := chromaSubsamplingShift(subsamplingX)
	shiftY := chromaSubsamplingShift(subsamplingY)
	rows := ((height - 1) << shiftY) + 1
	samples := make([]uint16, stride*rows)
	for y := 0; y < rows; y++ {
		for x := 0; x < ((width-1)<<shiftX)+1+shiftX; x++ {
			samples[y*stride+x] = value
		}
	}
	return samples
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

func testPatternChromaRowGrain() []int16 {
	grain := make([]int16, ChromaGrainSamples)
	for y := 0; y < ChromaGrainHeight; y++ {
		for x := 0; x < ChromaGrainWidth; x++ {
			grain[y*ChromaGrainWidth+x] = int16((x+y)%64 - 32)
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

func setTestChromaRowGrain(t *testing.T, grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	shiftX := chromaSubsamplingShift(subsamplingX)
	shiftY := chromaSubsamplingShift(subsamplingY)
	col := chromaGrainOffset(int(offset>>4), shiftX) + x + (LumaBlockSize>>shiftX)*blockCol
	row := chromaGrainOffset(int(offset&0x0f), shiftY) + y + (LumaBlockSize>>shiftY)*blockRow
	width, height := chromaGrainDimensions(subsamplingX, subsamplingY)
	if col < 0 || col >= width || row < 0 || row >= height {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*ChromaGrainWidth+col] = value
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

func withChromaRowWidth(params ChromaRowParams, width int) ChromaRowParams {
	params.Width = width
	return params
}

func withChromaRowHeight(params ChromaRowParams, height int) ChromaRowParams {
	params.Height = height
	return params
}

func withChromaRowStride(params ChromaRowParams, stride int) ChromaRowParams {
	params.Stride = stride
	return params
}

func withChromaRowLumaStride(params ChromaRowParams, stride int) ChromaRowParams {
	params.LumaStride = stride
	return params
}

func withChromaRowIndex(params ChromaRowParams, row int) ChromaRowParams {
	params.Row = row
	return params
}

func withChromaRowBitDepth(params ChromaRowParams, bitDepth uint8) ChromaRowParams {
	params.BitDepth = bitDepth
	return params
}

func withChromaRowScalingShift(params ChromaRowParams, shift uint8) ChromaRowParams {
	params.ScalingShift = shift
	return params
}
