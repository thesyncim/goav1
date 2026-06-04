package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicCDEFStrengthDirectionAndFrameFilter(t *testing.T) {
	level, secondary, err := av1.DecodeCDEFStrength(43)
	if err != nil {
		t.Fatal(err)
	}
	if level != 10 || secondary != 4 {
		t.Fatalf("level=%d secondary=%d want 10,4", level, secondary)
	}
	params, err := av1.CDEFFrameFilterParamsFromStrength(av1.CDEFPlaneU, 1, 0, 43, 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	if params != (av1.CDEFFrameFilterParams{
		XDec:              1,
		YDec:              0,
		Plane:             av1.CDEFPlaneU,
		Level:             10,
		SecondaryStrength: 4,
		Damping:           5,
		CoeffShift:        2,
	}) {
		t.Fatalf("params=%+v", params)
	}

	img := make([]uint16, 8*8)
	for row := range 8 {
		for col := range 8 {
			img[row*8+col] = uint16(col * 32)
		}
	}
	dir, variance, err := av1.FindCDEFDirection(img, 8, 0)
	if err != nil {
		t.Fatal(err)
	}
	if dir != 6 || variance != 282240 {
		t.Fatalf("dir,var=%d,%d want 6,282240", dir, variance)
	}

	input, origin := publicCDEFInput(8, 8)
	dst := make([]uint16, 8*8)
	var directions av1.CDEFDirectionGrid
	var variances av1.CDEFVarianceGrid
	blocks := []av1.CDEFBlockPosition{{BY: 0, BX: 0}}
	if err := av1.FilterCDEFFrameBlocks(dst, 8, input, origin, blocks, &directions, &variances, av1.CDEFFrameFilterParams{
		Plane:      av1.CDEFPlaneY,
		Damping:    3,
		CoeffShift: 0,
	}); err != nil {
		t.Fatal(err)
	}
	for row := range 8 {
		for col := range 8 {
			want := uint16(10*row + col)
			if got := dst[row*8+col]; got != want {
				t.Fatalf("dst(%d,%d)=%d want %d", col, row, got, want)
			}
		}
	}
}

func TestPublicCDEFBlockFilter(t *testing.T) {
	input, origin := publicCDEFInput(8, 8)
	dst := make([]uint16, 8*8)
	if err := av1.FilterCDEFBlock(dst, 8, 0, input, origin, av1.CDEFBlockFilterParams{
		Direction:      0,
		PrimaryDamping: 3,
		Width:          8,
		Height:         8,
	}); err != nil {
		t.Fatal(err)
	}
	for row := range 8 {
		for col := range 8 {
			want := uint16(10*row + col)
			if got := dst[row*8+col]; got != want {
				t.Fatalf("dst(%d,%d)=%d want %d", col, row, got, want)
			}
		}
	}
}

func TestPublicRestorationWienerPreservesConstantBlock(t *testing.T) {
	width, height := 17, 13
	src, stride, origin := publicRestorationSource(width, height, av1.RestorationWienerHalfwin, av1.RestorationWienerHalfwin, 85)
	dst := make([]uint16, width*height)
	scratchLen, err := av1.RestorationWienerScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	if err := av1.ApplyRestorationWiener(src, stride, origin, dst, width, width, height, av1.DefaultRestorationWienerInfo(), 8, make([]uint16, scratchLen)); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 85 {
			t.Fatalf("dst[%d]=%d want 85", i, got)
		}
	}

	filter := av1.NewRestorationWienerFilter(av1.RestorationWienerTap0Mid, av1.RestorationWienerTap1Mid, av1.RestorationWienerTap2Mid)
	if filter != av1.DefaultRestorationWienerInfo().VFilter {
		t.Fatalf("filter=%+v want default", filter)
	}
}

func TestPublicRestorationSelfguidedConstantBlock(t *testing.T) {
	width, height := 16, 16
	src, stride, origin := publicRestorationSource(width, height, av1.RestorationSGRProjBorderHorz, av1.RestorationSGRProjBorderVert, 341)
	dst := make([]uint16, width*height)
	scratchLen, err := av1.RestorationSelfguidedScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	params, err := av1.RestorationSGRParamsByIndex(0)
	if err != nil {
		t.Fatal(err)
	}
	xq := av1.DecodeRestorationSGRXQ([2]int8{13, -9}, params)
	if xq[0] != 13 || xq[1] == 0 {
		t.Fatalf("xq=%+v", xq)
	}
	if err := av1.ApplyRestorationSelfguided(src, stride, origin, dst, width, width, height, 0, [2]int8{13, -9}, 10, make([]int32, scratchLen)); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != dst[0] || got > 0x3ff {
			t.Fatalf("dst[%d]=%d first=%d max=1023", i, got, dst[0])
		}
	}
}

func TestPublicPostFilterRejectsInvalid(t *testing.T) {
	if _, _, err := av1.DecodeCDEFStrength(64); !errors.Is(err, av1.ErrCDEFInvalidCDEF) {
		t.Fatalf("DecodeCDEFStrength err=%v want %v", err, av1.ErrCDEFInvalidCDEF)
	}
	if _, _, err := av1.FindCDEFDirection(make([]uint16, 8), 8, 0); !errors.Is(err, av1.ErrCDEFInvalidCDEF) {
		t.Fatalf("FindCDEFDirection err=%v want %v", err, av1.ErrCDEFInvalidCDEF)
	}
	if _, err := av1.RestorationWienerScratchLen(0, 8); !errors.Is(err, av1.ErrRestorationInvalidInput) {
		t.Fatalf("RestorationWienerScratchLen err=%v want %v", err, av1.ErrRestorationInvalidInput)
	}
	if _, err := av1.RestorationSGRParamsByIndex(av1.RestorationSGRProjParams); !errors.Is(err, av1.ErrRestorationInvalidInput) {
		t.Fatalf("RestorationSGRParamsByIndex err=%v want %v", err, av1.ErrRestorationInvalidInput)
	}
	if _, err := av1.RestorationSelfguidedScratchLen(0, 8); !errors.Is(err, av1.ErrRestorationInvalidInput) {
		t.Fatalf("RestorationSelfguidedScratchLen err=%v want %v", err, av1.ErrRestorationInvalidInput)
	}
}

func TestPublicPostFilterAllocs(t *testing.T) {
	cdefInput, cdefOrigin := publicCDEFInput(8, 8)
	cdefDst := make([]uint16, 8*8)
	var directions av1.CDEFDirectionGrid
	var variances av1.CDEFVarianceGrid
	blocks := []av1.CDEFBlockPosition{{BY: 0, BX: 0}}

	width, height := 16, 16
	wienerSrc, wienerStride, wienerOrigin := publicRestorationSource(width, height, av1.RestorationWienerHalfwin, av1.RestorationWienerHalfwin, 85)
	wienerDst := make([]uint16, width*height)
	wienerScratchLen, err := av1.RestorationWienerScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	wienerScratch := make([]uint16, wienerScratchLen)

	sgrSrc, sgrStride, sgrOrigin := publicRestorationSource(width, height, av1.RestorationSGRProjBorderHorz, av1.RestorationSGRProjBorderVert, 341)
	sgrDst := make([]uint16, width*height)
	sgrScratchLen, err := av1.RestorationSelfguidedScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	sgrScratch := make([]int32, sgrScratchLen)

	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := av1.DecodeCDEFStrength(63); err != nil {
			t.Fatalf("DecodeCDEFStrength err=%v", err)
		}
		if err := av1.FilterCDEFFrameBlocks(cdefDst, 8, cdefInput, cdefOrigin, blocks, &directions, &variances, av1.CDEFFrameFilterParams{Plane: av1.CDEFPlaneY, Damping: 3}); err != nil {
			t.Fatalf("FilterCDEFFrameBlocks err=%v", err)
		}
		if err := av1.ApplyRestorationWiener(wienerSrc, wienerStride, wienerOrigin, wienerDst, width, width, height, av1.DefaultRestorationWienerInfo(), 8, wienerScratch); err != nil {
			t.Fatalf("ApplyRestorationWiener err=%v", err)
		}
		if err := av1.ApplyRestorationSelfguided(sgrSrc, sgrStride, sgrOrigin, sgrDst, width, width, height, 0, [2]int8{13, -9}, 10, sgrScratch); err != nil {
			t.Fatalf("ApplyRestorationSelfguided err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicCDEFInput(width int, height int) ([]uint16, int) {
	input := make([]uint16, av1.CDEFInputBufferSize)
	for i := range input {
		input[i] = av1.CDEFVeryLarge
	}
	origin := av1.CDEFVerticalBorder*av1.CDEFBStride + av1.CDEFHorizontalBorder
	for row := range height {
		for col := range width {
			input[origin+row*av1.CDEFBStride+col] = uint16(10*row + col)
		}
	}
	return input, origin
}

func publicRestorationSource(width int, height int, borderHorz int, borderVert int, value uint16) ([]uint16, int, int) {
	stride := width + 2*borderHorz
	origin := borderVert*stride + borderHorz
	src := make([]uint16, stride*(height+2*borderVert))
	for i := range src {
		src[i] = value
	}
	return src, stride, origin
}
