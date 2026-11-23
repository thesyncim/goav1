package goav1

import (
	"errors"
	"testing"

	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

func TestPublicCDEFStrengthDirectionAndFrameFilter(t *testing.T) {
	level, secondary, err := internalcdef.DecodeStrength(43)
	if err != nil {
		t.Fatal(err)
	}
	if level != 10 || secondary != 4 {
		t.Fatalf("level=%d secondary=%d want 10,4", level, secondary)
	}
	params, err := internalcdef.FrameFilterParamsFromStrength(internalcdef.PlaneU, 1, 0, 43, 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	if params != (internalcdef.FrameFilterParams{
		XDec:              1,
		YDec:              0,
		Plane:             internalcdef.PlaneU,
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
	dir, variance, err := internalcdef.FindDirection(img, 8, 0)
	if err != nil {
		t.Fatal(err)
	}
	if dir != 6 || variance != 282240 {
		t.Fatalf("dir,var=%d,%d want 6,282240", dir, variance)
	}

	input, origin := publicCDEFInput(8, 8)
	dst := make([]uint16, 8*8)
	var directions internalcdef.DirectionGrid
	var variances internalcdef.VarianceGrid
	blocks := []internalcdef.BlockPosition{{BY: 0, BX: 0}}
	if err := internalcdef.FilterFrameBlocks(dst, 8, input, origin, blocks, &directions, &variances, internalcdef.FrameFilterParams{
		Plane:      internalcdef.PlaneY,
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
	if err := internalcdef.FilterBlock(dst, 8, 0, input, origin, internalcdef.BlockFilterParams{
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
	src, stride, origin := publicRestorationSource(width, height, internalrestoration.WienerHalfwin, internalrestoration.WienerHalfwin, 85)
	dst := make([]uint16, width*height)
	scratchLen, err := internalrestoration.WienerScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	if err := internalrestoration.ApplyWienerRestoration(src, stride, origin, dst, width, width, height, internalrestoration.DefaultWienerInfo(), 8, make([]uint16, scratchLen)); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 85 {
			t.Fatalf("dst[%d]=%d want 85", i, got)
		}
	}

	filter := internalrestoration.NewWienerFilter(internalrestoration.WienerTap0Mid, internalrestoration.WienerTap1Mid, internalrestoration.WienerTap2Mid)
	if filter != internalrestoration.DefaultWienerInfo().VFilter {
		t.Fatalf("filter=%+v want default", filter)
	}
}

func TestPublicRestorationSelfguidedConstantBlock(t *testing.T) {
	width, height := 16, 16
	src, stride, origin := publicRestorationSource(width, height, internalrestoration.SGRProjBorderHorz, internalrestoration.SGRProjBorderVert, 341)
	dst := make([]uint16, width*height)
	scratchLen, err := internalrestoration.SelfguidedScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	params, err := restorationSGRParamsByIndex(0)
	if err != nil {
		t.Fatal(err)
	}
	xq := internalrestoration.DecodeSGRXQ([2]int8{13, -9}, params)
	if xq[0] != 13 || xq[1] == 0 {
		t.Fatalf("xq=%+v", xq)
	}
	if err := internalrestoration.ApplySelfguidedRestoration(src, stride, origin, dst, width, width, height, 0, [2]int8{13, -9}, 10, make([]int32, scratchLen)); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != dst[0] || got > 0x3ff {
			t.Fatalf("dst[%d]=%d first=%d max=1023", i, got, dst[0])
		}
	}
}

func TestPublicPostFilterRejectsInvalid(t *testing.T) {
	if _, _, err := internalcdef.DecodeStrength(64); !errors.Is(err, internalcdef.ErrInvalidCDEF) {
		t.Fatalf("DecodeCDEFStrength err=%v want %v", err, internalcdef.ErrInvalidCDEF)
	}
	if _, _, err := internalcdef.FindDirection(make([]uint16, 8), 8, 0); !errors.Is(err, internalcdef.ErrInvalidCDEF) {
		t.Fatalf("FindCDEFDirection err=%v want %v", err, internalcdef.ErrInvalidCDEF)
	}
	if _, err := internalrestoration.WienerScratchLen(0, 8); !errors.Is(err, internalrestoration.ErrInvalidRestoration) {
		t.Fatalf("RestorationWienerScratchLen err=%v want %v", err, internalrestoration.ErrInvalidRestoration)
	}
	if _, err := restorationSGRParamsByIndex(internalrestoration.SGRProjParams); !errors.Is(err, internalrestoration.ErrInvalidRestoration) {
		t.Fatalf("RestorationSGRParamsByIndex err=%v want %v", err, internalrestoration.ErrInvalidRestoration)
	}
	if _, err := internalrestoration.SelfguidedScratchLen(0, 8); !errors.Is(err, internalrestoration.ErrInvalidRestoration) {
		t.Fatalf("RestorationSelfguidedScratchLen err=%v want %v", err, internalrestoration.ErrInvalidRestoration)
	}
}

func TestPublicPostFilterAllocs(t *testing.T) {
	cdefInput, cdefOrigin := publicCDEFInput(8, 8)
	cdefDst := make([]uint16, 8*8)
	var directions internalcdef.DirectionGrid
	var variances internalcdef.VarianceGrid
	blocks := []internalcdef.BlockPosition{{BY: 0, BX: 0}}

	width, height := 16, 16
	wienerSrc, wienerStride, wienerOrigin := publicRestorationSource(width, height, internalrestoration.WienerHalfwin, internalrestoration.WienerHalfwin, 85)
	wienerDst := make([]uint16, width*height)
	wienerScratchLen, err := internalrestoration.WienerScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	wienerScratch := make([]uint16, wienerScratchLen)

	sgrSrc, sgrStride, sgrOrigin := publicRestorationSource(width, height, internalrestoration.SGRProjBorderHorz, internalrestoration.SGRProjBorderVert, 341)
	sgrDst := make([]uint16, width*height)
	sgrScratchLen, err := internalrestoration.SelfguidedScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	sgrScratch := make([]int32, sgrScratchLen)

	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := internalcdef.DecodeStrength(63); err != nil {
			t.Fatalf("DecodeCDEFStrength err=%v", err)
		}
		if err := internalcdef.FilterFrameBlocks(cdefDst, 8, cdefInput, cdefOrigin, blocks, &directions, &variances, internalcdef.FrameFilterParams{Plane: internalcdef.PlaneY, Damping: 3}); err != nil {
			t.Fatalf("FilterCDEFFrameBlocks err=%v", err)
		}
		if err := internalrestoration.ApplyWienerRestoration(wienerSrc, wienerStride, wienerOrigin, wienerDst, width, width, height, internalrestoration.DefaultWienerInfo(), 8, wienerScratch); err != nil {
			t.Fatalf("ApplyRestorationWiener err=%v", err)
		}
		if err := internalrestoration.ApplySelfguidedRestoration(sgrSrc, sgrStride, sgrOrigin, sgrDst, width, width, height, 0, [2]int8{13, -9}, 10, sgrScratch); err != nil {
			t.Fatalf("ApplyRestorationSelfguided err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicCDEFInput(width int, height int) ([]uint16, int) {
	input := make([]uint16, internalcdef.InputBufferSize)
	for i := range input {
		input[i] = internalcdef.VeryLarge
	}
	origin := internalcdef.VerticalBorder*internalcdef.BStride + internalcdef.HorizontalBorder
	for row := range height {
		for col := range width {
			input[origin+row*internalcdef.BStride+col] = uint16(10*row + col)
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
