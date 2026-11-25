package cdef

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"
	"unsafe"
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

const cdefDeterministicSeed = 0xbaba

func TestFindDirectionKnownPatterns(t *testing.T) {
	const stride = 8
	const size = stride * 8
	tests := []struct {
		name    string
		fill    func([]uint16)
		wantDir int
		wantVar int32
	}{
		{
			name: "constant",
			fill: func(img []uint16) {
				for i := range img {
					img[i] = 128
				}
			},
			wantDir: 0,
			wantVar: 0,
		},
		{
			name: "horizontal",
			fill: func(img []uint16) {
				for row := range 8 {
					for col := range 8 {
						img[row*stride+col] = uint16(row * 32)
					}
				}
			},
			wantDir: 2,
			wantVar: 282240,
		},
		{
			name: "vertical",
			fill: func(img []uint16) {
				for row := range 8 {
					for col := range 8 {
						img[row*stride+col] = uint16(col * 32)
					}
				}
			},
			wantDir: 6,
			wantVar: 282240,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := make([]uint16, size)
			tt.fill(img)
			gotDir, gotVar, err := FindDirection(img, stride, 0)
			if err != nil {
				t.Fatal(err)
			}
			if gotDir != tt.wantDir || gotVar != tt.wantVar {
				t.Fatalf("dir,var=%d,%d want %d,%d", gotDir, gotVar, tt.wantDir, tt.wantVar)
			}
		})
	}
}

func TestFindDirectionMatchesLibaomCorpus(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	for depth := 8; depth <= 12; depth += 2 {
		max := uint16((1 << depth) - 1)
		coeffShift := depth - 8
		for iter := range 128 {
			stride := 8 + rnd.pseudoUniform(9)
			img := make([]uint16, stride*8)
			for i := range img {
				img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
			}
			wantDir, wantVar := findDirectionLibaomReference(img, stride, coeffShift)
			gotDir, gotVar, err := FindDirection(img, stride, coeffShift)
			if err != nil {
				t.Fatalf("depth=%d iter=%d: %v", depth, iter, err)
			}
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("depth=%d iter=%d dir,var=%d,%d want %d,%d", depth, iter, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionDualMatchesSingle(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed)
	stride := 16
	img := make([]uint16, stride*8)
	for i := range img {
		img[i] = uint16(rnd.pseudoUniform(1 << 10))
	}
	wantDir1, wantVar1, err := FindDirection(img, stride, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantDir2, wantVar2, err := FindDirection(img[8:], stride, 2)
	if err != nil {
		t.Fatal(err)
	}
	gotDir1, gotVar1, gotDir2, gotVar2, err := FindDirectionDual(img, img[8:], stride, 2)
	if err != nil {
		t.Fatal(err)
	}
	if gotDir1 != wantDir1 || gotVar1 != wantVar1 || gotDir2 != wantDir2 || gotVar2 != wantVar2 {
		t.Fatalf("dual=(%d,%d),(%d,%d) want (%d,%d),(%d,%d)", gotDir1, gotVar1, gotDir2, gotVar2, wantDir1, wantVar1, wantDir2, wantVar2)
	}
}

func TestFindDirectionRejectsInvalidInputs(t *testing.T) {
	img := make([]uint16, 64)
	tests := []struct {
		name       string
		img        []uint16
		stride     int
		coeffShift int
	}{
		{name: "short", img: img[:63], stride: 8},
		{name: "small-stride", img: img, stride: 7},
		{name: "negative-shift", img: img, stride: 8, coeffShift: -1},
		{name: "large-shift", img: img, stride: 8, coeffShift: 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, err := FindDirection(tt.img, tt.stride, tt.coeffShift); err != ErrInvalidCDEF {
				t.Fatalf("err=%v want %v", err, ErrInvalidCDEF)
			}
		})
	}
}

func TestFindDirectionAllocs(t *testing.T) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 37) & 0xfff)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := FindDirection(img, 8, 4); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FindDirection allocated: %f", allocs)
	}
}

func FuzzFindDirection(f *testing.F) {
	f.Add([]byte{0, 16, 32, 64, 128, 255}, uint8(0), uint8(0))
	f.Add([]byte{255, 0, 127, 32, 64}, uint8(4), uint8(2))

	f.Fuzz(func(t *testing.T, data []byte, rawStride uint8, rawCoeffShift uint8) {
		stride := 8 + int(rawStride%8)
		coeffShift := int(rawCoeffShift % 5)
		img := make([]uint16, stride*8)
		max := uint16((1 << (coeffShift + 8)) - 1)
		for i := range img {
			if len(data) == 0 {
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			img[i] = (lo | hi<<8) & max
		}
		dir, variance, err := FindDirection(img, stride, coeffShift)
		if err != nil {
			t.Fatalf("FindDirection err=%v", err)
		}
		if dir < 0 || dir > 7 {
			t.Fatalf("dir=%d out of range", dir)
		}
		if variance < 0 {
			t.Fatalf("variance=%d want non-negative", variance)
		}
	})
}

func BenchmarkFindDirection(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xfff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = FindDirection(img, 8, 4)
	}
}

func BenchmarkFindDirection8(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = FindDirection(img, 8, 0)
	}
}

func BenchmarkFindDirectionDual(b *testing.B) {
	const stride = 16
	img := make([]uint16, stride*8)
	for i := range img {
		img[i] = uint16((i * 41) & 0xfff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _, _ = FindDirectionDual(img, img[8:], stride, 4)
	}
}

func BenchmarkFindDirectionDual8(b *testing.B) {
	const stride = 16
	img := make([]uint16, stride*8)
	for i := range img {
		img[i] = uint16((i * 41) & 0xff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _, _ = FindDirectionDual(img, img[8:], stride, 0)
	}
}

func findDirectionLibaomReference(img []uint16, stride int, coeffShift int) (int, int32) {
	var cost [8]int32
	var partial [8][15]int
	divTable := [...]int{0, 840, 420, 280, 210, 168, 140, 120, 105}
	for i := range 8 {
		for j := range 8 {
			x := int(img[i*stride+j]>>coeffShift) - 128
			partial[0][i+j] += x
			partial[1][i+j/2] += x
			partial[2][i] += x
			partial[3][3+i-j/2] += x
			partial[4][7+i-j] += x
			partial[5][3-i/2+j] += x
			partial[6][j] += x
			partial[7][i/2+j] += x
		}
	}
	for i := range 8 {
		cost[2] += int32(partial[2][i] * partial[2][i])
		cost[6] += int32(partial[6][i] * partial[6][i])
	}
	cost[2] *= int32(divTable[8])
	cost[6] *= int32(divTable[8])
	for i := range 7 {
		cost[0] += int32((partial[0][i]*partial[0][i] + partial[0][14-i]*partial[0][14-i]) * divTable[i+1])
		cost[4] += int32((partial[4][i]*partial[4][i] + partial[4][14-i]*partial[4][14-i]) * divTable[i+1])
	}
	cost[0] += int32(partial[0][7] * partial[0][7] * divTable[8])
	cost[4] += int32(partial[4][7] * partial[4][7] * divTable[8])
	for i := 1; i < 8; i += 2 {
		for j := range 5 {
			cost[i] += int32(partial[i][3+j] * partial[i][3+j])
		}
		cost[i] *= int32(divTable[8])
		for j := range 3 {
			cost[i] += int32((partial[i][j]*partial[i][j] + partial[i][10-j]*partial[i][10-j]) * divTable[2*j+2])
		}
	}
	bestCost := int32(0)
	bestDir := 0
	for i := range 8 {
		if cost[i] > bestCost {
			bestCost = cost[i]
			bestDir = i
		}
	}
	return bestDir, (bestCost - cost[(bestDir+4)&7]) >> 10
}

type cdefRandom struct {
	state uint32
}

func newCDEFRandom(seed uint32) *cdefRandom {
	return &cdefRandom{state: seed}
}

func (r *cdefRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *cdefRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}

// TestFilterBlockDispatchMatchesPureGo is the bit-exactness guard for the
// dispatched CDEF block filter. It exercises the resolved dispatch slot
// (filterBlockImpl, which routes through the Go SIMD kernel when the SIMD
// experiment is on) against the
// canonical pure-Go reference across the full matrix of directions,
// primary/secondary strengths, damping, bit depths, and the four CDEF border
// configurations. Every output sample must match the reference exactly.
func TestFilterBlockDispatchMatchesPureGo(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()

	for _, depth := range []int{8, 10, 12} {
		coeffShift := depth - 8
		shapes := [...]struct {
			width  int
			height int
		}{
			{width: 8, height: 8},
			{width: 8, height: 4},
			{width: 4, height: 8},
			{width: 4, height: 4},
		}
		for boundary := 0; boundary < 16; boundary++ {
			input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), depth, boundary, boundary+1)
			for _, shape := range shapes {
				for dir := 0; dir <= 7; dir++ {
					for _, pri := range cdefPrimaryStrengthCorpus(coeffShift) {
						for _, sec := range cdefSecondaryStrengthCorpus(coeffShift) {
							for _, damping := range []int{3 + coeffShift, 5 + coeffShift, 6 + coeffShift} {
								params := BlockFilterParams{
									PrimaryStrength:   uint8(pri),
									SecondaryStrength: uint8(sec),
									Direction:         uint8(dir),
									PrimaryDamping:    uint8(damping),
									SecondaryDamping:  uint8(damping),
									CoeffShift:        uint8(coeffShift),
									Width:             uint8(shape.width),
									Height:            uint8(shape.height),
								}
								want := make([]uint16, dstStride*8)
								got := make([]uint16, dstStride*8)
								filterBlockPureGo(want, dstStride, 0, input, origin, params)
								filterBlockImpl(got, dstStride, 0, input, origin, params)
								for i := range want {
									if got[i] != want[i] {
										t.Fatalf("depth=%d boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
											depth, boundary, shape.width, shape.height, dir, pri, sec, damping, i, got[i], want[i])
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestFilterBlockDispatchIsZeroAlloc protects the hot-path contract that the
// dispatched filter does not allocate per call.
func TestFilterBlockDispatchIsZeroAlloc(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 5, 3)
	dst := make([]uint16, dstStride*8)
	params := BlockFilterParams{
		PrimaryStrength:   12,
		SecondaryStrength: 4,
		Direction:         3,
		PrimaryDamping:    5,
		SecondaryDamping:  5,
		CoeffShift:        0,
		Width:             8,
		Height:            8,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		filterBlockImpl(dst, dstStride, 0, input, origin, params)
	})
	if allocs != 0 {
		t.Fatalf("dispatch allocated %f times per call", allocs)
	}
}

func TestTrustedNoPrimaryDirectionMatchesFilterAndLeavesGridsUntouched(t *testing.T) {
	for _, tc := range []struct {
		name   string
		plane  Plane
		xDec   uint8
		yDec   uint8
		shift  uint8
		width  int
		height int
	}{
		{name: "luma_8bit", plane: PlaneY, width: 8, height: 8},
		{name: "luma_10bit", plane: PlaneY, shift: 2, width: 8, height: 8},
		{name: "chroma_422_12bit", plane: PlaneU, xDec: 1, shift: 4, width: 4, height: 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := FrameFilterParams{
				XDec: tc.xDec, YDec: tc.yDec, Plane: tc.plane,
				Level: 0, SecondaryStrength: 2, Damping: 4, CoeffShift: tc.shift,
			}
			input := make([]uint16, InputBufferSize)
			for i := range input {
				input[i] = uint16((i*37 + i/11*13) & ((1 << (8 + tc.shift)) - 1))
			}
			blocks := []BlockPosition{{BY: 0, BX: 0}}
			stride := tc.width + 3
			want := make([]uint16, stride*tc.height)
			got := make([]uint16, stride*tc.height)
			for i := range want {
				want[i] = uint16(i*19 + 7)
				got[i] = want[i]
			}
			var wantDirections, gotDirections DirectionGrid
			var wantVariances, gotVariances VarianceGrid
			for by := range NBlocks {
				for bx := range NBlocks {
					wantDirections[by][bx] = 5
					gotDirections[by][bx] = 5
					wantVariances[by][bx] = 0x12345
					gotVariances[by][bx] = 0x12345
				}
			}
			if err := FilterFrameBlocksTrusted(want, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &wantDirections, &wantVariances, params); err != nil {
				t.Fatalf("trusted filter: %v", err)
			}
			if err := FilterFrameBlocksTrustedNoPrimaryDirection(got, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &gotDirections, &gotVariances, params); err != nil {
				t.Fatalf("no-direction filter: %v", err)
			}
			for row := range tc.height {
				for col := range tc.width {
					i := row*stride + col
					if got[i] != want[i] {
						t.Fatalf("pixel (%d,%d): got=%d want=%d", col, row, got[i], want[i])
					}
				}
			}
			for by := range NBlocks {
				for bx := range NBlocks {
					if gotDirections[by][bx] != 5 || gotVariances[by][bx] != 0x12345 {
						t.Fatalf("grid changed at (%d,%d): dir=%d var=%d", bx, by, gotDirections[by][bx], gotVariances[by][bx])
					}
				}
			}
		})
	}
}

func TestTrustedU8NoPrimaryDirectionMatchesFilterAndLeavesGridsUntouched(t *testing.T) {
	for _, tc := range []struct {
		name   string
		plane  Plane
		xDec   uint8
		yDec   uint8
		width  int
		height int
	}{
		{name: "luma", plane: PlaneY, width: 8, height: 8},
		{name: "chroma_422", plane: PlaneU, xDec: 1, width: 4, height: 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := FrameFilterParams{
				XDec: tc.xDec, YDec: tc.yDec, Plane: tc.plane,
				Level: 0, SecondaryStrength: 2, Damping: 4,
			}
			input := make([]uint16, InputBufferSize)
			for i := range input {
				input[i] = uint16((i*29 + i/7*17) & 255)
			}
			blocks := []BlockPosition{{BY: 0, BX: 0}}
			stride := tc.width + 3
			want := make([]byte, stride*tc.height)
			got := make([]byte, stride*tc.height)
			for i := range want {
				want[i] = byte(i*23 + 9)
				got[i] = want[i]
			}
			var wantDirections, gotDirections DirectionGrid
			var wantVariances, gotVariances VarianceGrid
			for by := range NBlocks {
				for bx := range NBlocks {
					wantDirections[by][bx] = 5
					gotDirections[by][bx] = 5
					wantVariances[by][bx] = 0x12345
					gotVariances[by][bx] = 0x12345
				}
			}
			if err := FilterFrameBlocksU8Trusted(want, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &wantDirections, &wantVariances, params); err != nil {
				t.Fatalf("trusted filter: %v", err)
			}
			if err := FilterFrameBlocksU8TrustedNoPrimaryDirection(got, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &gotDirections, &gotVariances, params); err != nil {
				t.Fatalf("no-direction filter: %v", err)
			}
			for row := range tc.height {
				for col := range tc.width {
					i := row*stride + col
					if got[i] != want[i] {
						t.Fatalf("pixel (%d,%d): got=%d want=%d", col, row, got[i], want[i])
					}
				}
			}
			for by := range NBlocks {
				for bx := range NBlocks {
					if gotDirections[by][bx] != 5 || gotVariances[by][bx] != 0x12345 {
						t.Fatalf("grid changed at (%d,%d): dir=%d var=%d", bx, by, gotDirections[by][bx], gotVariances[by][bx])
					}
				}
			}
		})
	}
}

// searchOracle is the per-trial search as the encoder ran it: copy the
// reconstruction into a scratch, filter every rowStep-th row of each block
// with FilterBlocksU8SubsampledTrusted, then the row-stepped SSE against the
// source (compute_cdef_dist_8bit).
func searchOracle(src []byte, srcStride int, in []uint16, origin int, blocks []BlockPosition, dir *DirectionGrid, vari *VarianceGrid, damping int, trials []SearchTrial, rowStep int) []uint64 {
	out := make([]uint64, len(trials))
	tmp := make([]byte, 64*64)
	for t, tr := range trials {
		for r := 0; r < 64; r++ {
			for c := 0; c < 64; c++ {
				tmp[r*64+c] = byte(in[origin+r*BStride+c])
			}
		}
		if tr.Primary != 0 || tr.Secondary != 0 {
			params := FrameFilterParams{Plane: PlaneY, Level: tr.Primary, SecondaryStrength: tr.Secondary, Damping: uint8(damping)}
			if err := FilterBlocksU8SubsampledTrusted(tmp, 64, in, origin, blocks, dir, vari, params, rowStep); err != nil {
				panic(err)
			}
		}
		var sum uint64
		for _, b := range blocks {
			for i := 0; i < 8; i += rowStep {
				so := (int(b.BY)*8+i)*srcStride + int(b.BX)*8
				fo := (int(b.BY)*8+i)*64 + int(b.BX)*8
				for j := 0; j < 8; j++ {
					e := int32(src[so+j]) - int32(tmp[fo+j])
					sum += uint64(e * e)
				}
			}
		}
		out[t] = sum
	}
	return out
}

// TestSearchBlocksU8MatchesPerTrial: the batched search returns, per trial,
// the per-trial filter-then-measure sums, across random strengths, row
// steps, directions, variances and block lists, including lists longer than
// one kernel chunk.
func TestSearchBlocksU8MatchesPerTrial(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	const srcStride = 72
	src := make([]byte, 64*srcStride)
	secondaries := []uint8{0, 1, 2, 4}
	for iter := 0; iter < 300; iter++ {
		for i := range in {
			in[i] = uint16(rng.Intn(256))
			if rng.Intn(40) == 0 {
				in[i] = VeryLarge
			}
		}
		// Block interiors are reconstruction samples; sentinels live in the halo.
		for r := 0; r < 64; r++ {
			for c := 0; c < 64; c++ {
				if in[origin+r*BStride+c] == VeryLarge {
					in[origin+r*BStride+c] = uint16(rng.Intn(256))
				}
			}
		}
		switch rng.Intn(3) {
		case 0: // source near the reconstruction, as real content is
			for r := 0; r < 64; r++ {
				for c := 0; c < 64; c++ {
					v := int(in[origin+r*BStride+c]) + rng.Intn(21) - 10
					src[r*srcStride+c] = byte(min(max(v, 0), 255))
				}
			}
		default:
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
		}
		var blocks []BlockPosition
		for by := 0; by < 8; by++ {
			for bx := 0; bx < 8; bx++ {
				if rng.Intn(3) != 0 {
					blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		if len(blocks) == 0 {
			blocks = append(blocks, BlockPosition{BY: uint8(rng.Intn(8)), BX: uint8(rng.Intn(8))})
		}
		var dir DirectionGrid
		var vari VarianceGrid
		for by := range dir {
			for bx := range dir[by] {
				dir[by][bx] = uint8(rng.Intn(8))
				switch rng.Intn(4) {
				case 0:
					vari[by][bx] = 0
				case 1:
					vari[by][bx] = int32(rng.Intn(64))
				default:
					vari[by][bx] = int32(rng.Intn(1 << 20))
				}
			}
		}
		damping := 3 + rng.Intn(4)
		rowStep := []int{1, 2, 4}[rng.Intn(3)]
		nTrials := 1 + rng.Intn(20)
		trials := make([]SearchTrial, nTrials)
		for i := range trials {
			trials[i] = SearchTrial{Primary: uint8(rng.Intn(16)), Secondary: secondaries[rng.Intn(4)]}
			if rng.Intn(6) == 0 {
				trials[i] = SearchTrial{}
			}
		}
		want := searchOracle(src, srcStride, in, origin, blocks, &dir, &vari, damping, trials, rowStep)
		got := make([]uint64, nTrials)
		if !SearchBlocksU8Trusted(src, srcStride, in, origin, blocks, &dir, &vari, damping, trials, rowStep, got) {
			t.Skip("no batched search kernel in this build")
		}
		for i := range trials {
			if got[i] != want[i] {
				t.Fatalf("iter %d trial %d %+v (damping %d, step %d, %d blocks): got %d want %d", iter, i, trials[i], damping, rowStep, len(blocks), got[i], want[i])
			}
		}
	}
}

// TestSearchBlocksU8RejectsOutOfContract: geometry outside the trusted
// contract is refused rather than read.
func TestSearchBlocksU8RejectsOutOfContract(t *testing.T) {
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	src := make([]byte, 64*64)
	var dir DirectionGrid
	var vari VarianceGrid
	trials := []SearchTrial{{15, 2}}
	sse := make([]uint64, 1)
	ok := SearchBlocksU8Trusted(src, 64, in, origin, []BlockPosition{{BY: 7, BX: 7}}, &dir, &vari, 3, trials, 1, sse)
	if !ok {
		t.Skip("no batched search kernel in this build")
	}
	if SearchBlocksU8Trusted(src[:64*63], 64, in, origin, []BlockPosition{{BY: 7, BX: 7}}, &dir, &vari, 3, trials, 1, sse) {
		t.Fatal("accepted a source plane missing the last block row")
	}
	if SearchBlocksU8Trusted(src, 64, in, origin, []BlockPosition{{BY: 8, BX: 0}}, &dir, &vari, 3, trials, 1, sse) {
		t.Fatal("accepted a block outside the 64x64 filter block")
	}
	if SearchBlocksU8Trusted(src, 64, in, origin, nil, &dir, &vari, 3, trials, 0, sse) {
		t.Fatal("accepted row step 0")
	}
}

func BenchmarkSearchBlocksU8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	for i := range in {
		in[i] = uint16(rng.Intn(256))
	}
	src := make([]byte, 64*64)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	var blocks []BlockPosition
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	var dir DirectionGrid
	var vari VarianceGrid
	for by := range dir {
		for bx := range dir[by] {
			dir[by][bx] = uint8(rng.Intn(8))
			vari[by][bx] = int32(rng.Intn(1 << 16))
		}
	}
	// SVT rtc level 8 on a base-layer inter frame: (0,0), (15,0), (0,2),
	// (15,2) at subsampling 4.
	trials := []SearchTrial{{0, 0}, {15, 0}, {0, 2}, {15, 2}}
	sse := make([]uint64, len(trials))
	b.Run("level8-step4", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SearchBlocksU8Trusted(src, 64, in, origin, blocks, &dir, &vari, 4, trials, 4, sse)
		}
	})
	b.Run("level8-step4-oracle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			searchOracle(src, 64, in, origin, blocks, &dir, &vari, 4, trials, 4)
		}
	})
	// Level 5 on an I slice: six trials, every row.
	trials5 := []SearchTrial{{0, 0}, {7, 0}, {15, 0}, {0, 2}, {7, 2}, {15, 2}}
	sse5 := make([]uint64, len(trials5))
	b.Run("level5-step1", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SearchBlocksU8Trusted(src, 64, in, origin, blocks, &dir, &vari, 4, trials5, 1, sse5)
		}
	})
}

// TestFilterBlocksU8SubsampledMatchesFullRows: every row the subsampled
// search filter writes equals the same row of the full-block filter, and
// the rows it skips stay untouched.
func TestFilterBlocksU8SubsampledMatchesFullRows(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	for trial := 0; trial < 200; trial++ {
		dec := rng.Intn(2)
		size := 64 >> dec
		for i := range in {
			in[i] = uint16(rng.Intn(256))
			if rng.Intn(50) == 0 {
				in[i] = VeryLarge
			}
		}
		for r := 0; r < size; r++ {
			for c := 0; c < size; c++ {
				if in[origin+r*BStride+c] == VeryLarge {
					in[origin+r*BStride+c] = 7
				}
			}
		}
		base := make([]byte, size*size)
		for r := 0; r < size; r++ {
			for c := 0; c < size; c++ {
				base[r*size+c] = byte(in[origin+r*BStride+c])
			}
		}
		var blocks []BlockPosition
		for by := 0; by < 8; by++ {
			for bx := 0; bx < 8; bx++ {
				if rng.Intn(4) != 0 {
					blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		var dir DirectionGrid
		var vari VarianceGrid
		plane := PlaneY
		if dec == 1 {
			plane = PlaneU
		}
		dirSrc := make([]byte, 64*64)
		for i := range dirSrc {
			dirSrc[i] = byte(rng.Intn(256))
		}
		if err := FilterFrameBlocksU8Trusted(dirSrc, 64, nil, 0, blocks, &dir, &vari, FrameFilterParams{Plane: PlaneY, Damping: 3}); err != nil {
			t.Fatal(err)
		}
		params := FrameFilterParams{XDec: uint8(dec), YDec: uint8(dec), Plane: plane,
			Level: uint8(rng.Intn(16)), SecondaryStrength: uint8([]int{0, 1, 2, 4}[rng.Intn(4)]), Damping: uint8(3 + rng.Intn(4))}
		full := append([]byte(nil), base...)
		if err := FilterFrameBlocksU8TrustedNoPrimaryDirection(full, size, in, origin, blocks, &dir, &vari, params); err != nil {
			t.Fatal(err)
		}
		for _, step := range []int{2, 4} {
			sub := append([]byte(nil), base...)
			if err := FilterBlocksU8SubsampledTrusted(sub, size, in, origin, blocks, &dir, &vari, params, step); err != nil {
				t.Fatal(err)
			}
			for r := 0; r < size; r++ {
				want := base[r*size : r*size+size]
				if (r%(8>>dec))%step == 0 {
					want = full[r*size : r*size+size]
				}
				if !bytes.Equal(sub[r*size:r*size+size], want) {
					t.Fatalf("trial %d dec %d step %d row %d differs", trial, dec, step, r)
				}
			}
		}
	}
}

// filterUnitBlocksReference is the trusted-path oracle for the unit-level
// loop: the same strength/direction derivation as filterUnitBlocksPureGo but
// routed straight to filterBlockPureGo, bypassing every dispatch slot.
func filterUnitBlocksReference(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := u.primaryStrength
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		filterBlockPureGo(dst, dstStride, dstOrigin, input, srcOrigin, BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(u.secondaryStrength),
			Direction:         uint8(dir),
			PrimaryDamping:    uint8(u.damping),
			SecondaryDamping:  uint8(u.damping),
			CoeffShift:        uint8(u.coeffShift),
			Width:             uint8(u.blockWidth),
			Height:            uint8(u.blockHeight),
		})
	}
}

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "BlockFilterParams", size: unsafe.Sizeof(BlockFilterParams{}), max: 8},
		{name: "FrameFilterParams", size: unsafe.Sizeof(FrameFilterParams{}), max: 7},
		{name: "DirectionGrid", size: unsafe.Sizeof(DirectionGrid{}), max: 256},
		{name: "VarianceGrid", size: unsafe.Sizeof(VarianceGrid{}), max: 1024},
		{name: "cdefDirections", size: unsafe.Sizeof(cdefDirections), max: 48},
		{name: "cdefPrimaryTaps", size: unsafe.Sizeof(cdefPrimaryTaps), max: 4},
		{name: "cdefSecondaryTaps", size: unsafe.Sizeof(cdefSecondaryTaps), max: 2},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}

func TestDecodeStrength(t *testing.T) {
	tests := []struct {
		name      string
		packed    uint8
		level     int
		secondary int
		wantErr   error
	}{
		{name: "zero", packed: 0, level: 0, secondary: 0},
		{name: "secondary-one", packed: 1, level: 0, secondary: 1},
		{name: "secondary-two", packed: 2, level: 0, secondary: 2},
		{name: "secondary-three-maps-to-four", packed: 3, level: 0, secondary: 4},
		{name: "primary-and-secondary", packed: 43, level: 10, secondary: 4},
		{name: "max", packed: 63, level: 15, secondary: 4},
		{name: "overflow", packed: 64, wantErr: ErrInvalidCDEF},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			level, secondary, err := DecodeStrength(tt.packed)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("err=%v want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if level != tt.level || secondary != tt.secondary {
				t.Fatalf("level=%d secondary=%d want %d,%d", level, secondary, tt.level, tt.secondary)
			}
		})
	}
}

func TestFrameFilterParamsFromStrength(t *testing.T) {
	params, err := FrameFilterParamsFromStrength(PlaneU, 1, 0, 43, 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := FrameFilterParams{
		XDec:              1,
		YDec:              0,
		Plane:             PlaneU,
		Level:             10,
		SecondaryStrength: 4,
		Damping:           5,
		CoeffShift:        2,
	}
	if params != want {
		t.Fatalf("params=%+v want %+v", params, want)
	}
}

func TestFrameFilterParamsFromStrengthRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name       string
		plane      Plane
		xDec       int
		yDec       int
		packed     uint8
		damping    int
		coeffShift int
	}{
		{name: "bad-plane", plane: Plane(3), xDec: 0, yDec: 0, damping: 3},
		{name: "bad-xdec", plane: PlaneY, xDec: 2, yDec: 0, damping: 3},
		{name: "bad-ydec", plane: PlaneY, xDec: 0, yDec: -1, damping: 3},
		{name: "bad-packed", plane: PlaneY, xDec: 0, yDec: 0, packed: 64, damping: 3},
		{name: "bad-damping", plane: PlaneY, xDec: 0, yDec: 0, damping: -1},
		{name: "bad-shift", plane: PlaneY, xDec: 0, yDec: 0, damping: 3, coeffShift: 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := FrameFilterParamsFromStrength(tt.plane, tt.xDec, tt.yDec, tt.packed, tt.damping, tt.coeffShift)
			if !errors.Is(err, ErrInvalidCDEF) {
				t.Fatalf("err=%v want %v", err, ErrInvalidCDEF)
			}
		})
	}
}

func TestCDEFStrengthHelpersAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := DecodeStrength(63); err != nil {
			t.Fatal(err)
		}
		if _, err := FrameFilterParamsFromStrength(PlaneV, 1, 1, 63, 6, 1); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("CDEF strength helpers allocated: %f", allocs)
	}
}

// Differential proof for the 8-bit-pixel CDEF kernel family (filter_u8.go,
// direction_u8.go): every entry must be bit-identical to the uint16 kernels
// fed the same widened input + 0x4000 sentinels, followed by a uint8
// narrowing store. The tests run the resolved dispatch slots, so on arm64
// they cover the NEON kernels and on native amd64 the AVX2-backed wrappers;
// the pure-Go references are asserted explicitly as well.

// TestFilterBlockU8MatchesU16Narrow proves the per-block u8 kernel identity
// across directions, strengths, dampings, block shapes, and all sixteen
// halo-sentinel patterns, and that the uint16 result always fits a byte under
// the 8-bit contract (real interior pixels).
func TestFilterBlockU8MatchesU16Narrow(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {4, 8}, {4, 4},
	}
	for boundary := 0; boundary < 16; boundary++ {
		for iter := range 3 {
			input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^uint32(iter*977)), 8, boundary, iter)
			for _, shape := range shapes {
				for dir := 0; dir <= 7; dir++ {
					for _, pri := range cdefPrimaryStrengthCorpus(0) {
						for _, sec := range cdefSecondaryStrengthCorpus(0) {
							if pri == 0 && sec == 0 {
								continue
							}
							for _, damping := range []int{3, 5, 6} {
								params := BlockFilterParams{
									PrimaryStrength:   uint8(pri),
									SecondaryStrength: uint8(sec),
									Direction:         uint8(dir),
									PrimaryDamping:    uint8(damping),
									SecondaryDamping:  uint8(damping),
									CoeffShift:        0,
									Width:             uint8(shape.width),
									Height:            uint8(shape.height),
								}
								want16 := make([]uint16, dstStride*8)
								filterBlockPureGo(want16, dstStride, 0, input, origin, params)

								gotRef := make([]byte, dstStride*8)
								filterBlockU8PureGo(gotRef, dstStride, 0, input, origin, params)
								gotImpl := make([]byte, dstStride*8)
								filterBlockU8Impl(gotImpl, dstStride, 0, input, origin, params)

								for row := 0; row < shape.height; row++ {
									for col := 0; col < shape.width; col++ {
										i := row*dstStride + col
										if want16[i] > 255 {
											t.Fatalf("u16 result exceeds byte range: boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d val=%d",
												boundary, shape.width, shape.height, dir, pri, sec, damping, i, want16[i])
										}
										if gotRef[i] != byte(want16[i]) {
											t.Fatalf("pure-Go u8 mismatch: boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
												boundary, shape.width, shape.height, dir, pri, sec, damping, i, gotRef[i], want16[i])
										}
										if gotImpl[i] != byte(want16[i]) {
											t.Fatalf("dispatched u8 mismatch: boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
												boundary, shape.width, shape.height, dir, pri, sec, damping, i, gotImpl[i], want16[i])
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestFindDirectionU8MatchesU16 proves the 8-bit direction search (scalar
// reference and the resolved dispatch slot, single and dual) returns exactly
// the uint16 search's direction and variance on widened pixels.
func TestFindDirectionU8MatchesU16(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x5538_4449)
	for _, stride := range []int{8, 11, 64, 640} {
		for iter := range 64 {
			img8 := make([]byte, stride*8+8)
			img16 := make([]uint16, len(img8))
			for i := range img8 {
				var v uint32
				switch iter % 4 {
				case 0:
					v = rnd.generate(256)
				case 1:
					v = rnd.generate(8) // flat block, exercises variance 0
				case 2:
					v = 255 - rnd.generate(4)
				default:
					v = rnd.generate(2) * 255
				}
				img8[i] = byte(v)
				img16[i] = uint16(v)
			}
			wantDir, wantVar := findDirectionScalar(img16, stride, 0)
			gotDir, gotVar := findDirectionU8Scalar(img8, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("scalar u8 direction mismatch: stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			gotDir, gotVar = findDirectionU8Unchecked(img8, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("dispatched u8 direction mismatch: stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			if stride >= 16 {
				wd1, wv1, wd2, wv2 := findDirectionDualUnchecked(img16, img16[8:], stride, 0)
				gd1, gv1, gd2, gv2 := findDirectionDualU8Unchecked(img8, img8[8:], stride)
				if gd1 != wd1 || gv1 != wv1 || gd2 != wd2 || gv2 != wv2 {
					t.Fatalf("dual u8 direction mismatch: stride=%d iter=%d got=(%d,%d,%d,%d) want=(%d,%d,%d,%d)",
						stride, iter, gd1, gv1, gd2, gv2, wd1, wv1, wd2, wv2)
				}
			}
		}
	}
}

// cdefU8UnitCase describes one synthetic filter unit for the frame-level
// differential.
type cdefU8UnitCase struct {
	plane    Plane
	xDec     int
	yDec     int
	unitW    int
	unitH    int
	level    int
	sec      int
	damping  int
	skipMask uint64 // bit per block position: 1 = drop from the block list
	halo     uint8  // applyCDEFBoundary-style sentinel pattern
}

// TestFilterFrameBlocksU8MatchesU16 is the unit-level differential: for a
// matrix of planes/subsampling/unit sizes/strengths/skip patterns/sentinel
// halos it runs the trusted uint16 frame filter into a prefilled uint16 unit
// and the trusted uint8 frame filter in place on a byte plane, then requires
// identical bytes everywhere (filtered blocks, skipped blocks, and pixels
// outside the unit) plus identical direction/variance grids.
func TestFilterFrameBlocksU8MatchesU16(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x5538_4642)
	cases := make([]cdefU8UnitCase, 0, 256)
	for _, geom := range []struct {
		plane      Plane
		xDec, yDec int
	}{
		{PlaneY, 0, 0},
		{PlaneU, 1, 1},
		{PlaneV, 1, 1},
		{PlaneU, 1, 0}, // 4:2:2 (chroma direction conversion path)
		{PlaneU, 0, 0}, // 4:4:4
	} {
		unitWFull := BlockSize >> geom.xDec
		unitHFull := BlockSize >> geom.yDec
		blockW := 8 >> geom.xDec
		blockH := 8 >> geom.yDec
		for _, size := range []struct{ w, h int }{
			{unitWFull, unitHFull},                   // full unit
			{unitWFull - blockW, unitHFull - blockH}, // partial right/bottom
			{blockW, blockH},                         // single block
			{unitWFull, blockH * 2},                  // short unit
		} {
			for _, str := range []struct{ level, sec int }{
				{0, 0}, // direction-only luma pass shape
				{9, 0}, {0, 2}, {15, 4}, {1, 1}, {4, 2},
			} {
				for _, damping := range []int{3, 6} {
					cases = append(cases, cdefU8UnitCase{
						plane: geom.plane, xDec: geom.xDec, yDec: geom.yDec,
						unitW: size.w, unitH: size.h,
						level: str.level, sec: str.sec, damping: damping,
						skipMask: uint64(rnd.generate(1<<16)) | uint64(rnd.generate(1<<16))<<16,
						halo:     uint8(rnd.generate(16)),
					})
				}
			}
		}
	}
	for i, tc := range cases {
		runCDEFU8UnitDifferential(t, rnd, tc, i)
	}
}

func runCDEFU8UnitDifferential(t *testing.T, rnd *cdefRandom, tc cdefU8UnitCase, caseIdx int) {
	t.Helper()
	blockW := 8 >> tc.xDec
	blockH := 8 >> tc.yDec
	blockCols := (tc.unitW + blockW - 1) / blockW
	blockRows := (tc.unitH + blockH - 1) / blockH

	// Frame plane: the unit sits at an offset inside a larger byte plane so
	// out-of-unit writes would be caught.
	const margin = 24
	stride := tc.unitW + 2*margin + 3
	height := tc.unitH + 2*margin
	frame := make([]byte, stride*height)
	for i := range frame {
		frame[i] = byte(rnd.generate(256))
	}
	unitOrigin := margin*stride + margin

	// CDEF input buffer: interior = frame unit pixels, halo = random real
	// neighbours, then the sentinel pattern (frame edges) on top.
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(256))
	}
	origin := VerticalBorder*BStride + HorizontalBorder
	for row := 0; row < tc.unitH; row++ {
		for col := 0; col < tc.unitW; col++ {
			input[origin+row*BStride+col] = uint16(frame[unitOrigin+row*stride+col])
		}
	}
	applyCDEFU8UnitHalo(input, tc.unitW, tc.unitH, tc.halo)

	blocks := make([]BlockPosition, 0, blockCols*blockRows)
	bit := 0
	for by := 0; by < blockRows; by++ {
		for bx := 0; bx < blockCols; bx++ {
			if tc.skipMask&(1<<uint(bit%64)) == 0 {
				blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
			}
			bit++
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, BlockPosition{BY: 0, BX: 0})
	}

	params := FrameFilterParams{
		XDec:              uint8(tc.xDec),
		YDec:              uint8(tc.yDec),
		Plane:             tc.plane,
		Level:             uint8(tc.level),
		SecondaryStrength: uint8(tc.sec),
		Damping:           uint8(tc.damping),
		CoeffShift:        0,
	}

	// uint16 path: prefill the unit dst with the source pixels (the decoder's
	// skip prefill), filter, then narrow.
	var wantDirs, gotDirs DirectionGrid
	var wantVars, gotVars VarianceGrid
	if tc.plane != PlaneY {
		// Chroma reuses luma directions; seed both grids identically.
		for by := range NBlocks {
			for bx := range NBlocks {
				d := uint8(rnd.generate(8))
				wantDirs[by][bx] = d
				gotDirs[by][bx] = d
				v := int32(rnd.generate(1 << 20))
				wantVars[by][bx] = v
				gotVars[by][bx] = v
			}
		}
	}
	want16 := make([]uint16, stride*height)
	for i, v := range frame {
		want16[i] = uint16(v)
	}
	if err := FilterFrameBlocksTrusted(want16[unitOrigin:], stride, input, origin, blocks, &wantDirs, &wantVars, params); err != nil {
		t.Fatalf("case %d: u16 filter: %v", caseIdx, err)
	}

	// uint8 path: in place on a copy of the frame.
	got := make([]byte, len(frame))
	copy(got, frame)
	if err := FilterFrameBlocksU8Trusted(got[unitOrigin:], stride, input, origin, blocks, &gotDirs, &gotVars, params); err != nil {
		t.Fatalf("case %d: u8 filter: %v", caseIdx, err)
	}

	if gotDirs != wantDirs {
		t.Fatalf("case %d (%+v): direction grids differ", caseIdx, tc)
	}
	if gotVars != wantVars {
		t.Fatalf("case %d (%+v): variance grids differ", caseIdx, tc)
	}
	for i := range got {
		if want16[i] > 255 {
			t.Fatalf("case %d (%+v): u16 result exceeds byte range at %d: %d", caseIdx, tc, i, want16[i])
		}
		if got[i] != byte(want16[i]) {
			row := i / stride
			col := i % stride
			t.Fatalf("case %d (%+v): byte mismatch at row=%d col=%d got=%d want=%d", caseIdx, tc, row, col, got[i], want16[i])
		}
	}
}

// applyCDEFU8UnitHalo overlays the VeryLarge sentinel on the unit halo in the
// same four-edge pattern as applyCDEFBoundary, but for an arbitrary unit
// geometry (the decoder writes sentinels for missing frame edges).
func applyCDEFU8UnitHalo(input []uint16, unitW int, unitH int, halo uint8) {
	fillH := unitH + 2*VerticalBorder
	if halo&1 != 0 { // left
		for row := 0; row < fillH; row++ {
			for col := 0; col < HorizontalBorder; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if halo&2 != 0 { // right
		for row := 0; row < fillH; row++ {
			for col := HorizontalBorder + unitW; col < BStride; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if halo&4 != 0 { // top
		for row := 0; row < VerticalBorder; row++ {
			for col := 0; col < BStride; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if halo&8 != 0 { // bottom
		for row := VerticalBorder + unitH; row < fillH; row++ {
			for col := 0; col < BStride; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
}

// TestFilterFrameBlocksU8RejectsHighBitDepth pins the 8-bit-only contract.
func TestFilterFrameBlocksU8RejectsHighBitDepth(t *testing.T) {
	var dirs DirectionGrid
	var vars VarianceGrid
	params := FrameFilterParams{Plane: PlaneY, Level: 4, Damping: 4, CoeffShift: 2}
	err := FilterFrameBlocksU8Trusted(make([]byte, 64*64), 64, make([]uint16, InputBufferSize), cdefBlockOrigin(), []BlockPosition{{}}, &dirs, &vars, params)
	if err == nil {
		t.Fatal("expected CoeffShift != 0 to be rejected")
	}
}

// TestFilterFrameBlocksU8IsZeroAlloc protects the hot-path contract.
func TestFilterFrameBlocksU8IsZeroAlloc(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x414c_4c38)
	stride := 96
	frame := make([]byte, stride*96)
	for i := range frame {
		frame[i] = byte(rnd.generate(256))
	}
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(256))
	}
	blocks := make([]BlockPosition, 0, 64)
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	var dirs DirectionGrid
	var vars VarianceGrid
	params := FrameFilterParams{Plane: PlaneY, Level: 9, SecondaryStrength: 2, Damping: 5}
	allocs := testing.AllocsPerRun(32, func() {
		if err := FilterFrameBlocksU8Trusted(frame, stride, input, cdefBlockOrigin(), blocks, &dirs, &vars, params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FilterFrameBlocksU8Trusted allocates: %v allocs/op", allocs)
	}
}
