package cdef

import (
	"errors"
	"testing"
)

func TestCopyRectMatchesLibaomCorpus(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	for _, width := range []int{4, 8, 16, 64} {
		for _, height := range []int{2, 4, 17, 64} {
			src16Stride := width + 7
			dstStride := width + 9
			src16 := make([]uint16, src16Stride*height)
			got16 := make([]uint16, dstStride*height)
			want16 := make([]uint16, dstStride*height)
			for row := range height {
				for col := range width {
					src16[row*src16Stride+col] = uint16(rnd.generate(1 << 16))
				}
			}
			copyRect16To16Reference(want16, dstStride, src16, src16Stride, width, height)
			if err := CopyRect16To16(got16, dstStride, src16, src16Stride, width, height); err != nil {
				t.Fatalf("CopyRect16To16 width=%d height=%d: %v", width, height, err)
			}
			assertUint16SlicesEqual(t, got16, want16)
		}
	}
}

func TestCopyRectRejectsInvalidInputs(t *testing.T) {
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
									PrimaryStrength:   uint8(pri),
									SecondaryStrength: uint8(sec),
									Direction:         uint8(dir),
									PrimaryDamping:    uint8(3 + shift + boundary%3),
									SecondaryDamping:  uint8(4 + shift + (boundary>>2)%3),
									CoeffShift:        uint8(shift),
									Width:             uint8(size.width),
									Height:            uint8(size.height),
								}
								filterBlockLibaomReference(want, 8, 0, input, cdefBlockOrigin(), params)
								filterBlockUnchecked(got, 8, 0, input, cdefBlockOrigin(), params)
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
		dirs[by][bx] = uint8(dir)
		vars[by][bx] = variance
		strength := adjustStrengthReference(int(params.Level)<<coeffShift, variance)
		filterBlockLibaomReference(want, 16, (by<<3)*16+(bx<<3), input, srcOrigin, BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(params.SecondaryStrength << coeffShift),
			Direction:         uint8(dir),
			PrimaryDamping:    uint8(params.Damping + coeffShift),
			SecondaryDamping:  uint8(params.Damping + coeffShift),
			CoeffShift:        uint8(coeffShift),
			Width:             8,
			Height:            8,
		})
	}
	var gotDirs DirectionGrid
	var gotVars VarianceGrid
	if err := FilterFrameBlocksTrusted(got, 16, input, cdefBlockOrigin(), blocks, &gotDirs, &gotVars, params); err != nil {
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
		dirs[block.BY][block.BX] = uint8(i)
	}
	dst := make([]uint16, 16*8)
	err := FilterFrameBlocksTrusted(dst, 16, input, cdefBlockOrigin(), blocks, &dirs, &vars, FrameFilterParams{
		XDec:              1,
		Plane:             PlaneU,
		Level:             4,
		SecondaryStrength: 1,
		Damping:           5,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{7, 0, 2, 4}
	for i, block := range blocks {
		if got := dirs[block.BY][block.BX]; got != want[i] {
			t.Fatalf("dir[%d]=%d want %d", i, got, want[i])
		}
	}
}

func TestCDEFAllocs(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(1 << 12))
	}
	dst := make([]uint16, 16*16)
	blocks := []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 1}, {BY: 1, BX: 0}, {BY: 1, BX: 1}}
	var dirs DirectionGrid
	var vars VarianceGrid
	params := BlockFilterParams{
		PrimaryStrength:   15 << 4,
		SecondaryStrength: 4 << 4,
		Direction:         3,
		PrimaryDamping:    7,
		SecondaryDamping:  7,
		CoeffShift:        4,
		Width:             8,
		Height:            8,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := CopyRect16To16(dst, 16, dst, 16, 8, 8); err != nil {
			t.Fatal(err)
		}
		filterBlockUnchecked(dst, 16, 0, input, cdefBlockOrigin(), params)
		if err := FilterFrameBlocksTrusted(dst, 16, input, cdefBlockOrigin(), blocks, &dirs, &vars, FrameFilterParams{
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

func copyRect16To16Reference(dst []uint16, dstStride int, src []uint16, srcStride int, width int, height int) {
	for row := range height {
		for col := range width {
			dst[row*dstStride+col] = src[row*srcStride+col]
		}
	}
}

func filterBlockLibaomReference(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	primaryDamping := int(params.PrimaryDamping)
	secondaryDamping := int(params.SecondaryDamping)
	coeffShift := int(params.CoeffShift)
	width := int(params.Width)
	height := int(params.Height)

	clippingRequired := primaryStrength != 0 && secondaryStrength != 0
	priTaps := referenceCDEFPrimaryTaps[(primaryStrength>>coeffShift)&1]
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			base := inputOrigin + row*BStride + col
			sum := 0
			x := int(input[base])
			maxSample := x
			minSample := x
			for k := range 2 {
				if primaryStrength != 0 {
					p0 := int(input[base+referenceCDEFDirections[direction][k]])
					p1 := int(input[base-referenceCDEFDirections[direction][k]])
					sum += priTaps[k] * constrainReference(p0-x, primaryStrength, primaryDamping)
					sum += priTaps[k] * constrainReference(p1-x, primaryStrength, primaryDamping)
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
				if secondaryStrength != 0 {
					s0 := int(input[base+referenceCDEFDirections[(direction+2)&7][k]])
					s1 := int(input[base-referenceCDEFDirections[(direction+2)&7][k]])
					s2 := int(input[base+referenceCDEFDirections[(direction+6)&7][k]])
					s3 := int(input[base-referenceCDEFDirections[(direction+6)&7][k]])
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
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s0-x, secondaryStrength, secondaryDamping)
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s1-x, secondaryStrength, secondaryDamping)
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s2-x, secondaryStrength, secondaryDamping)
					sum += referenceCDEFSecondaryTaps[k] * constrainReference(s3-x, secondaryStrength, secondaryDamping)
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
	return []int{0, 1 << shift, 4 << shift, 8 << shift, 15 << shift}
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
