package cdef

import (
	"errors"
	"testing"
)

func TestCopyRectMatchesLibaomCorpus(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	for _, width := range []int{4, 8, 16, 64} {
		for _, height := range []int{2, 4, 17, 64} {
			src8Stride := width + 5
			src16Stride := width + 7
			dstStride := width + 9
			src8 := make([]uint8, src8Stride*height)
			src16 := make([]uint16, src16Stride*height)
			got8 := make([]uint16, dstStride*height)
			got16 := make([]uint16, dstStride*height)
			want8 := make([]uint16, dstStride*height)
			want16 := make([]uint16, dstStride*height)
			for row := range height {
				for col := range width {
					v := uint16(rnd.generate(1 << 16))
					src8[row*src8Stride+col] = uint8(v)
					src16[row*src16Stride+col] = v
				}
			}
			copyRect8To16Reference(want8, dstStride, src8, src8Stride, width, height)
			copyRect16To16Reference(want16, dstStride, src16, src16Stride, width, height)
			if err := CopyRect8To16(got8, dstStride, src8, src8Stride, width, height); err != nil {
				t.Fatalf("CopyRect8To16 width=%d height=%d: %v", width, height, err)
			}
			if err := CopyRect16To16(got16, dstStride, src16, src16Stride, width, height); err != nil {
				t.Fatalf("CopyRect16To16 width=%d height=%d: %v", width, height, err)
			}
			assertUint16SlicesEqual(t, got8, want8)
			assertUint16SlicesEqual(t, got16, want16)
		}
	}
}

func TestCopyRectRejectsInvalidInputs(t *testing.T) {
	if err := CopyRect8To16(make([]uint16, 4), 2, make([]uint8, 4), 2, 3, 2); !errors.Is(err, ErrInvalidCDEF) {
		t.Fatalf("CopyRect8To16 err=%v want %v", err, ErrInvalidCDEF)
	}
	if err := CopyRect16To16(make([]uint16, 4), 2, make([]uint16, 4), 2, 2, 3); !errors.Is(err, ErrInvalidCDEF) {
		t.Fatalf("CopyRect16To16 err=%v want %v", err, ErrInvalidCDEF)
	}
}

func TestFilterBlockMatchesLibaomCorpus(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	for _, size := range []struct {
		width  int
		height int
	}{
		{4, 4},
		{4, 8},
		{8, 4},
		{8, 8},
	} {
		for boundary := range 16 {
			for _, depth := range []int{8, 10, 12} {
				shift := depth - 8
				for iter := range 2 {
					input := makeCDEFBlockInput(rnd, depth, boundary, iter)
					for dir := range 8 {
						for _, pri := range cdefPrimaryStrengthCorpus(shift) {
							for _, sec := range cdefSecondaryStrengthCorpus(shift) {
								got := make([]uint16, 8*8)
								want := make([]uint16, 8*8)
								params := BlockFilterParams{
									PrimaryStrength:   pri,
									SecondaryStrength: sec,
									Direction:         dir,
									PrimaryDamping:    3 + shift + boundary%3,
									SecondaryDamping:  4 + shift + (boundary>>2)%3,
									CoeffShift:        shift,
									Width:             size.width,
									Height:            size.height,
								}
								filterBlockLibaomReference(want, 8, 0, input, cdefBlockOrigin(), params)
								if err := FilterBlock(got, 8, 0, input, cdefBlockOrigin(), params); err != nil {
									t.Fatalf("size=%dx%d boundary=%d depth=%d dir=%d pri=%d sec=%d: %v", size.width, size.height, boundary, depth, dir, pri, sec, err)
								}
								assertBlockEqual(t, got, want, 8, size.width, size.height)
							}
						}
					}
				}
			}
		}
	}
}

func TestFilterFrameBlocksMatchesLibaomReference(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	const coeffShift = 2
	blocks := []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 1}, {BY: 1, BX: 0}, {BY: 1, BX: 1}}
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(1 << 10))
	}
	var dirs DirectionGrid
	var vars VarianceGrid
	got := make([]uint16, 16*16)
	want := make([]uint16, 16*16)
	params := FrameFilterParams{
		Plane:             PlaneY,
		Level:             9,
		SecondaryStrength: 2,
		Damping:           5,
		CoeffShift:        coeffShift,
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		srcOrigin := cdefBlockOrigin() + (by*BStride)<<3 + (bx << 3)
		dir, variance := findDirectionLibaomReference(input[srcOrigin:], BStride, coeffShift)
		dirs[by][bx] = dir
		vars[by][bx] = variance
		strength := adjustStrengthReference(params.Level<<coeffShift, variance)
		filterBlockLibaomReference(want, 16, (by<<3)*16+(bx<<3), input, srcOrigin, BlockFilterParams{
			PrimaryStrength:   strength,
			SecondaryStrength: params.SecondaryStrength << coeffShift,
			Direction:         dir,
			PrimaryDamping:    params.Damping + coeffShift,
			SecondaryDamping:  params.Damping + coeffShift,
			CoeffShift:        coeffShift,
			Width:             8,
			Height:            8,
		})
	}
	var gotDirs DirectionGrid
	var gotVars VarianceGrid
	if err := FilterFrameBlocks(got, 16, input, cdefBlockOrigin(), blocks, &gotDirs, &gotVars, params); err != nil {
		t.Fatal(err)
	}
	assertBlockEqual(t, got, want, 16, 16, 16)
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if gotDirs[by][bx] != dirs[by][bx] || gotVars[by][bx] != vars[by][bx] {
			t.Fatalf("block=(%d,%d) dir,var=%d,%d want %d,%d", by, bx, gotDirs[by][bx], gotVars[by][bx], dirs[by][bx], vars[by][bx])
		}
	}
}

func TestFilterFrameBlocksConvertsChromaDirections(t *testing.T) {
	blocks := []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 1}, {BY: 0, BX: 2}, {BY: 0, BX: 3}}
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = 128
	}
	var dirs DirectionGrid
	var vars VarianceGrid
	for i, block := range blocks {
		dirs[block.BY][block.BX] = i
	}
	dst := make([]uint16, 16*8)
	err := FilterFrameBlocks(dst, 16, input, cdefBlockOrigin(), blocks, &dirs, &vars, FrameFilterParams{
		XDec:              1,
		Plane:             PlaneU,
		Level:             4,
		SecondaryStrength: 1,
		Damping:           5,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{7, 0, 2, 4}
	for i, block := range blocks {
		if got := dirs[block.BY][block.BX]; got != want[i] {
			t.Fatalf("dir[%d]=%d want %d", i, got, want[i])
		}
	}
}

func TestFilterBlockRejectsInvalidInputs(t *testing.T) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	valid := BlockFilterParams{
		PrimaryStrength:   4,
		SecondaryStrength: 2,
		Direction:         0,
		PrimaryDamping:    3,
		SecondaryDamping:  3,
		Width:             8,
		Height:            8,
	}
	tests := []struct {
		name   string
		dst    []uint16
		stride int
		origin int
		input  []uint16
		inOff  int
		params BlockFilterParams
	}{
		{name: "short-dst", dst: make([]uint16, 63), stride: 8, input: input, inOff: cdefBlockOrigin(), params: valid},
		{name: "short-input", dst: make([]uint16, 64), stride: 8, input: input[:cdefBlockOrigin()], inOff: cdefBlockOrigin(), params: valid},
		{name: "bad-dir", dst: make([]uint16, 64), stride: 8, input: input, inOff: cdefBlockOrigin(), params: withCDEFDirection(valid, 8)},
		{name: "bad-size", dst: make([]uint16, 64), stride: 8, input: input, inOff: cdefBlockOrigin(), params: withCDEFSize(valid, 6, 8)},
		{name: "negative-strength", dst: make([]uint16, 64), stride: 8, input: input, inOff: cdefBlockOrigin(), params: withCDEFPrimary(valid, -1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := FilterBlock(tt.dst, tt.stride, tt.origin, tt.input, tt.inOff, tt.params); !errors.Is(err, ErrInvalidCDEF) {
				t.Fatalf("err=%v want %v", err, ErrInvalidCDEF)
			}
		})
	}
}

func TestCDEFAllocs(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(1 << 12))
	}
	src8 := make([]uint8, 16*8)
	for i := range src8 {
		src8[i] = uint8(rnd.generate(256))
	}
	dst := make([]uint16, 16*16)
	blocks := []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 1}, {BY: 1, BX: 0}, {BY: 1, BX: 1}}
	var dirs DirectionGrid
	var vars VarianceGrid
	params := BlockFilterParams{
		PrimaryStrength:   19 << 4,
		SecondaryStrength: 4 << 4,
		Direction:         3,
		PrimaryDamping:    7,
		SecondaryDamping:  7,
		CoeffShift:        4,
		Width:             8,
		Height:            8,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := CopyRect8To16(dst, 16, src8, 16, 8, 8); err != nil {
			t.Fatal(err)
		}
		if err := CopyRect16To16(dst, 16, dst, 16, 8, 8); err != nil {
			t.Fatal(err)
		}
		if err := FilterBlock(dst, 16, 0, input, cdefBlockOrigin(), params); err != nil {
			t.Fatal(err)
		}
		if err := FilterFrameBlocks(dst, 16, input, cdefBlockOrigin(), blocks, &dirs, &vars, FrameFilterParams{
			Plane:             PlaneY,
			Level:             8,
			SecondaryStrength: 2,
			Damping:           5,
			CoeffShift:        4,
		}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("CDEF helpers allocated: %f", allocs)
	}
}

func FuzzFilterBlock(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7}, uint8(0), uint8(0), uint8(0), uint8(8), uint8(8))
	f.Add([]byte{255, 128, 64, 32, 16, 8}, uint8(15), uint8(7), uint8(4), uint8(4), uint8(8))

	f.Fuzz(func(t *testing.T, data []byte, rawBoundary uint8, rawDir uint8, rawShift uint8, rawWidth uint8, rawHeight uint8) {
		shift := int(rawShift % 5)
		width := 4
		if rawWidth&1 == 0 {
			width = 8
		}
		height := 4
		if rawHeight&1 == 0 {
			height = 8
		}
		input := make([]uint16, (8+2*VerticalBorder)*BStride)
		max := uint16((1 << (8 + shift)) - 1)
		for i := range input {
			if len(data) == 0 {
				input[i] = 0
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			input[i] = (lo | hi<<8) & max
		}
		applyCDEFBoundary(input, rawBoundary&15)
		params := BlockFilterParams{
			PrimaryStrength:   int(rawDir%20) << shift,
			SecondaryStrength: cdefSecondaryStrengthCorpus(shift)[int(rawBoundary)%4],
			Direction:         int(rawDir % 8),
			PrimaryDamping:    3 + shift + int(rawBoundary%3),
			SecondaryDamping:  3 + shift + int((rawBoundary>>2)%3),
			CoeffShift:        shift,
			Width:             width,
			Height:            height,
		}
		got := make([]uint16, 64)
		want := make([]uint16, 64)
		filterBlockLibaomReference(want, 8, 0, input, cdefBlockOrigin(), params)
		if err := FilterBlock(got, 8, 0, input, cdefBlockOrigin(), params); err != nil {
			t.Fatalf("FilterBlock err=%v", err)
		}
		assertBlockEqual(t, got, want, 8, width, height)
	})
}

func FuzzFilterFrameBlocks(f *testing.F) {
	f.Add([]byte{0, 16, 32, 64, 128, 255}, uint8(0), uint8(0), uint8(2))
	f.Fuzz(func(t *testing.T, data []byte, rawLevel uint8, rawSecondary uint8, rawShift uint8) {
		shift := int(rawShift % 5)
		input := make([]uint16, InputBufferSize)
		max := uint16((1 << (8 + shift)) - 1)
		for i := range input {
			if len(data) == 0 {
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			input[i] = (lo | hi<<8) & max
		}
		blocks := []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 1}, {BY: 1, BX: 0}, {BY: 1, BX: 1}}
		dst := make([]uint16, 16*16)
		var dirs DirectionGrid
		var vars VarianceGrid
		err := FilterFrameBlocks(dst, 16, input, cdefBlockOrigin(), blocks, &dirs, &vars, FrameFilterParams{
			Plane:             PlaneY,
			Level:             int(rawLevel % 16),
			SecondaryStrength: int(rawSecondary % 4),
			Damping:           3 + int(rawLevel%4),
			CoeffShift:        shift,
		})
		if err != nil {
			t.Fatalf("FilterFrameBlocks err=%v", err)
		}
		for _, sample := range dst {
			if sample > max {
				t.Fatalf("sample=%d max=%d", sample, max)
			}
		}
	})
}

func BenchmarkFilterBlock(b *testing.B) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 12, 0, 0)
	dst := make([]uint16, 64)
	params := BlockFilterParams{
		PrimaryStrength:   19 << 4,
		SecondaryStrength: 4 << 4,
		Direction:         4,
		PrimaryDamping:    7,
		SecondaryDamping:  7,
		CoeffShift:        4,
		Width:             8,
		Height:            8,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = FilterBlock(dst, 8, 0, input, cdefBlockOrigin(), params)
	}
}

func BenchmarkFilterFrameBlocks(b *testing.B) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(1 << 10))
	}
	blocks := []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 1}, {BY: 1, BX: 0}, {BY: 1, BX: 1}}
	dst := make([]uint16, 16*16)
	var dirs DirectionGrid
	var vars VarianceGrid
	params := FrameFilterParams{
		Plane:             PlaneY,
		Level:             9,
		SecondaryStrength: 2,
		Damping:           5,
		CoeffShift:        2,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = FilterFrameBlocks(dst, 16, input, cdefBlockOrigin(), blocks, &dirs, &vars, params)
	}
}

func BenchmarkCopyRect8To16(b *testing.B) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	const width = 64
	const height = 64
	srcStride := width + 5
	dstStride := width + 9
	src := make([]uint8, srcStride*height)
	dst := make([]uint16, dstStride*height)
	for i := range src {
		src[i] = uint8(rnd.generate(256))
	}
	b.SetBytes(int64(width * height))
	b.ReportAllocs()
	for b.Loop() {
		_ = CopyRect8To16(dst, dstStride, src, srcStride, width, height)
	}
}

func BenchmarkCopyRect16To16(b *testing.B) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	const width = 64
	const height = 64
	srcStride := width + 7
	dstStride := width + 9
	src := make([]uint16, srcStride*height)
	dst := make([]uint16, dstStride*height)
	for i := range src {
		src[i] = uint16(rnd.generate(1 << 16))
	}
	b.SetBytes(int64(width * height * 2))
	b.ReportAllocs()
	for b.Loop() {
		_ = CopyRect16To16(dst, dstStride, src, srcStride, width, height)
	}
}

func copyRect8To16Reference(dst []uint16, dstStride int, src []uint8, srcStride int, width int, height int) {
	for row := range height {
		for col := range width {
			dst[row*dstStride+col] = uint16(src[row*srcStride+col])
		}
	}
}

func copyRect16To16Reference(dst []uint16, dstStride int, src []uint16, srcStride int, width int, height int) {
	for row := range height {
		for col := range width {
			dst[row*dstStride+col] = src[row*srcStride+col]
		}
	}
}

func filterBlockLibaomReference(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	clippingRequired := params.PrimaryStrength != 0 && params.SecondaryStrength != 0
	priTaps := referenceCDEFPrimaryTaps[(params.PrimaryStrength>>params.CoeffShift)&1]
	for row := 0; row < params.Height; row++ {
		for col := 0; col < params.Width; col++ {
			base := inputOrigin + row*BStride + col
			sum := 0
			x := int(input[base])
			maxSample := x
			minSample := x
			for k := range 2 {
				if params.PrimaryStrength != 0 {
					p0 := int(input[base+referenceCDEFDirections[params.Direction][k]])
					p1 := int(input[base-referenceCDEFDirections[params.Direction][k]])
					sum += priTaps[k] * constrainReference(p0-x, params.PrimaryStrength, params.PrimaryDamping)
					sum += priTaps[k] * constrainReference(p1-x, params.PrimaryStrength, params.PrimaryDamping)
					if clippingRequired {
						if p0 != VeryLarge && p0 > maxSample {
							maxSample = p0
						}
						if p1 != VeryLarge && p1 > maxSample {
							maxSample = p1
						}
						if p0 < minSample {
							minSample = p0
						}
						if p1 < minSample {
							minSample = p1
						}
					}
				}
				if params.SecondaryStrength != 0 {
					s0 := int(input[base+referenceCDEFDirections[(params.Direction+2)&7][k]])
					s1 := int(input[base-referenceCDEFDirections[(params.Direction+2)&7][k]])
					s2 := int(input[base+referenceCDEFDirections[(params.Direction+6)&7][k]])
					s3 := int(input[base-referenceCDEFDirections[(params.Direction+6)&7][k]])
					if clippingRequired {
						for _, v := range []int{s0, s1, s2, s3} {
							if v != VeryLarge && v > maxSample {
								maxSample = v
							}
							if v < minSample {
								minSample = v
							}
						}
					}
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s0-x, params.SecondaryStrength, params.SecondaryDamping)
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s1-x, params.SecondaryStrength, params.SecondaryDamping)
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s2-x, params.SecondaryStrength, params.SecondaryDamping)
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s3-x, params.SecondaryStrength, params.SecondaryDamping)
				}
			}
			y := x + ((8 + sum - boolToIntReference(sum < 0)) >> 4)
			if clippingRequired {
				y = clampReference(y, minSample, maxSample)
			}
			dst[dstOrigin+row*dstStride+col] = uint16(y)
		}
	}
}

var referenceCDEFDirections = [8][2]int{
	{-1*BStride + 1, -2*BStride + 2},
	{1, -1*BStride + 2},
	{1, 2},
	{1, 1*BStride + 2},
	{1*BStride + 1, 2*BStride + 2},
	{1 * BStride, 2*BStride + 1},
	{1 * BStride, 2 * BStride},
	{1 * BStride, 2*BStride - 1},
}

var referenceCDEFPrimaryTaps = [2][2]int{{4, 2}, {3, 3}}
var referenceCDEFSecondaryTaps = [2]int{2, 1}

func makeCDEFBlockInput(rnd *cdefRandom, depth int, boundary int, iteration int) []uint16 {
	input := make([]uint16, (8+2*VerticalBorder)*BStride)
	mask := uint16((1 << depth) - 1)
	level := uint16(iteration << (depth - 8))
	for i := range input {
		input[i] = minUint16((uint16(rnd.generate(1<<16))&mask)+level, mask)
	}
	applyCDEFBoundary(input, uint8(boundary))
	return input
}

func applyCDEFBoundary(input []uint16, boundary uint8) {
	ysize := 8 + 2*VerticalBorder
	if boundary&1 != 0 {
		for row := range ysize {
			for col := range HorizontalBorder {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if boundary&2 != 0 {
		for row := range ysize {
			for col := HorizontalBorder + 8; col < BStride; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if boundary&4 != 0 {
		for row := range VerticalBorder {
			for col := range BStride {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if boundary&8 != 0 {
		for row := VerticalBorder + 8; row < ysize; row++ {
			for col := range BStride {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
}

func cdefBlockOrigin() int {
	return VerticalBorder*BStride + HorizontalBorder
}

func cdefPrimaryStrengthCorpus(shift int) []int {
	return []int{0, 1 << shift, 4 << shift, 8 << shift, 19 << shift}
}

func cdefSecondaryStrengthCorpus(shift int) []int {
	return []int{0, 1 << shift, 2 << shift, 4 << shift}
}

func constrainReference(diff int, threshold int, damping int) int {
	if threshold == 0 {
		return 0
	}
	shift := max(damping-msbReference(threshold), 0)
	return signReference(diff) * clampReference(threshold-(absReference(diff)>>shift), 0, absReference(diff))
}

func adjustStrengthReference(strength int, variance int32) int {
	if variance == 0 {
		return 0
	}
	i := 0
	if v := variance >> 6; v != 0 {
		i = min(msbReference(int(v)), 12)
	}
	return (strength*(4+i) + 8) >> 4
}

func msbReference(v int) int {
	n := 0
	for v > 1 {
		v >>= 1
		n++
	}
	return n
}

func clampReference(v int, lo int, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func signReference(v int) int {
	if v < 0 {
		return -1
	}
	return 1
}

func absReference(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func boolToIntReference(v bool) int {
	if v {
		return 1
	}
	return 0
}

func minUint16(a uint16, b uint16) uint16 {
	if a < b {
		return a
	}
	return b
}

func assertBlockEqual(t *testing.T, got []uint16, want []uint16, stride int, width int, height int) {
	t.Helper()
	for row := range height {
		for col := range width {
			if got[row*stride+col] != want[row*stride+col] {
				t.Fatalf("sample[%d,%d]=%d want %d", row, col, got[row*stride+col], want[row*stride+col])
			}
		}
	}
}

func assertUint16SlicesEqual(t *testing.T, got []uint16, want []uint16) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("sample[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func withCDEFDirection(params BlockFilterParams, dir int) BlockFilterParams {
	params.Direction = dir
	return params
}

func withCDEFSize(params BlockFilterParams, width int, height int) BlockFilterParams {
	params.Width = width
	params.Height = height
	return params
}

func withCDEFPrimary(params BlockFilterParams, primary int) BlockFilterParams {
	params.PrimaryStrength = primary
	return params
}
