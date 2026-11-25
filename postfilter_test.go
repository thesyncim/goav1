package goav1

import (
	"crypto/md5"
	"errors"
	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalfilmgrain "github.com/thesyncim/goav1/internal/av1/filmgrain"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
	internalsuperres "github.com/thesyncim/goav1/internal/av1/superres"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"os"
	"path/filepath"
	"testing"
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
	if err := internalcdef.FilterFrameBlocksTrusted(dst, 8, input, origin, blocks, &directions, &variances, internalcdef.FrameFilterParams{
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

func TestPublicRestorationWienerPreservesConstantBlock(t *testing.T) {
	width, height := 17, 13
	src, stride, origin := publicRestorationSource(width, height, internalrestoration.WienerHalfwin, internalrestoration.WienerHalfwin, 85)
	dst := make([]uint16, width*height)
	scratchLen, err := internalrestoration.WienerScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	if err := internalrestoration.ApplyWienerRestorationTrusted(src, stride, origin, dst, width, width, height, internalrestoration.DefaultWienerInfo(), 8, make([]uint16, scratchLen)); err != nil {
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
	params := internalrestoration.SGRParameterTable[0]
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
		if err := internalcdef.FilterFrameBlocksTrusted(cdefDst, 8, cdefInput, cdefOrigin, blocks, &directions, &variances, internalcdef.FrameFilterParams{Plane: internalcdef.PlaneY, Damping: 3}); err != nil {
			t.Fatalf("FilterCDEFFrameBlocks err=%v", err)
		}
		if err := internalrestoration.ApplyWienerRestorationTrusted(wienerSrc, wienerStride, wienerOrigin, wienerDst, width, width, height, internalrestoration.DefaultWienerInfo(), 8, wienerScratch); err != nil {
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

func TestPublicSuperResUpscalePlane(t *testing.T) {
	src := []uint16{0, 32, 64, 96, 128, 160, 192, 224}
	dst := make([]uint16, 13)
	srcPlane := internalframe.SamplePlane{Pix: src, Stride: len(src), Width: len(src), Height: 1}
	dstPlane := internalframe.SamplePlane{Pix: dst, Stride: len(dst), Width: len(dst), Height: 1}
	if err := internalsuperres.UpscalePlaneCoded(srcPlane, dstPlane, srcPlane.Width, 8); err != nil {
		t.Fatal(err)
	}
	want := []uint16{0, 11, 33, 54, 73, 92, 112, 132, 151, 171, 191, 214, 226}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d full=%v", i, dst[i], want[i], dst)
		}
	}

	src = []uint16{0, 512, 1023, 400}
	dst = make([]uint16, 7)
	srcPlane = internalframe.SamplePlane{Pix: src, Stride: len(src), Width: len(src), Height: 1}
	dstPlane = internalframe.SamplePlane{Pix: dst, Stride: len(dst), Width: len(dst), Height: 1}
	if err := internalsuperres.UpscalePlaneCoded(srcPlane, dstPlane, srcPlane.Width, 10); err != nil {
		t.Fatal(err)
	}
	want = []uint16{0, 104, 450, 901, 996, 642, 330}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("10-bit dst[%d]=%d want %d full=%v", i, dst[i], want[i], dst)
		}
	}
}

func TestPublicFilmGrainRandomScalingAndSamples(t *testing.T) {
	if internalfilmgrain.Gaussian(0) != 56 || internalfilmgrain.Gaussian(internalfilmgrain.GaussianSequenceLen) != 56 {
		t.Fatalf("unexpected gaussian table lookup")
	}
	rng := internalfilmgrain.NewRandom(0x1234)
	for i, want := range []uint16{0x89, 0x44, 0x22, 0x91, 0xc8, 0xe4, 0xf2, 0xf9} {
		got, err := rng.Number(8)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("random step %d=%#x want %#x", i, got, want)
		}
	}
	if _, err := internalfilmgrain.NewPlaneRandom(0x1234, internalfilmgrain.ChromaPlaneCb); err != nil {
		t.Fatal(err)
	}
	if _, err := internalfilmgrain.NewPlaneRandom(0x1234, internalfilmgrain.ChromaPlaneCr); err != nil {
		t.Fatal(err)
	}
	if _, err := internalfilmgrain.NewStripeRandom(0x1234, 32); err != nil {
		t.Fatal(err)
	}

	var lut [internalfilmgrain.ScalingLUTSize]uint8
	points := []internalfilmgrain.ScalingPoint{{Value: 10, Scaling: 20}, {Value: 20, Scaling: 40}}
	if err := internalfilmgrain.BuildScalingLUT(lut[:], points); err != nil {
		t.Fatal(err)
	}
	for index, want := range map[int]uint8{0: 20, 9: 20, 10: 20, 11: 22, 14: 28, 15: 30, 19: 38, 20: 40, 255: 40} {
		if got := lut[index]; got != want {
			t.Fatalf("lut[%d]=%d want %d", index, got, want)
		}
	}
}

func TestPublicGenerateFilmGrainTemplates(t *testing.T) {
	luma := make([]int16, internalfilmgrain.LumaGrainSamples)
	lumaParams := internalfilmgrain.LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	if err := internalfilmgrain.GenerateLumaGrain(luma, lumaParams); err != nil {
		t.Fatal(err)
	}
	for i, want := range [...]int16{-29, -1, -31, -20, 49, 8, -56, 54, 18, -7, -35, -47, -51, -3, 60, 26} {
		if luma[i] != want {
			t.Fatalf("luma[%d]=%d want %d", i, luma[i], want)
		}
	}
	if got := luma[3*internalfilmgrain.LumaGrainWidth+3]; got != -14 {
		t.Fatalf("luma[3][3]=%d want -14", got)
	}

	chroma := make([]int16, internalfilmgrain.ChromaGrainSamples)
	chromaParams := internalfilmgrain.ChromaGrainParams{
		Seed:            0x1234,
		Plane:           internalfilmgrain.ChromaPlaneCb,
		BitDepth:        8,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
	}
	if err := internalfilmgrain.GenerateChromaGrain(chroma, nil, chromaParams); err != nil {
		t.Fatal(err)
	}
	for i, want := range [...]int16{12, -37, 4, -18, -13, 6, -5, -43, -8, 11, -29, 48, 67, 57, 20, 13} {
		if chroma[i] != want {
			t.Fatalf("chroma[%d]=%d want %d", i, chroma[i], want)
		}
	}
	if chroma[internalfilmgrain.ChromaSubsampledGrainWidth] != 0 ||
		chroma[internalfilmgrain.ChromaSubsampledGrainHeight*internalfilmgrain.ChromaGrainWidth] != 0 {
		t.Fatalf("inactive subsampled chroma samples are not zero")
	}
}

func TestPublicApplyFilmGrainRowsAndSamples(t *testing.T) {
	lut := publicFilmGrainScaling(64)
	dst, src := publicFilmGrainRowBuffers(4, 2, 8, 100)
	lumaGrain := make([]int16, internalfilmgrain.LumaGrainSamples)
	publicSetFilmGrainLuma(lumaGrain, 0xd9, 0, 0, 0, 0, 64)
	lumaParams := internalfilmgrain.LumaRowParams{
		Seed:         0,
		Width:        4,
		Height:       2,
		Stride:       8,
		BitDepth:     8,
		ScalingShift: 8,
	}
	if err := internalfilmgrain.ApplyLumaRow(dst, src, lumaGrain, lut[:], lumaParams); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 116 || dst[1] != 100 || dst[8] != 100 {
		t.Fatalf("luma row dst[0,1,8]=%d,%d,%d want 116,100,100", dst[0], dst[1], dst[8])
	}

	chromaDst, chromaSrc := publicFilmGrainRowBuffers(4, 2, 8, 100)
	lumaSamples := publicFilmGrainLumaSamples(4, 2, 8, false, false, 80)
	chromaGrain := make([]int16, internalfilmgrain.ChromaGrainSamples)
	publicSetFilmGrainChroma(chromaGrain, 0xd9, false, false, 0, 0, 0, 0, 64)
	chromaParams := internalfilmgrain.ChromaRowParams{
		Seed:                  0,
		Width:                 4,
		Height:                2,
		Stride:                8,
		LumaStride:            8,
		BitDepth:              8,
		ScalingShift:          8,
		ChromaScalingFromLuma: true,
	}
	if err := internalfilmgrain.ApplyChromaRow(chromaDst, chromaSrc, lumaSamples, chromaGrain, lut[:], chromaParams); err != nil {
		t.Fatal(err)
	}
	if chromaDst[0] != 116 || chromaDst[1] != 100 {
		t.Fatalf("chroma row dst[0,1]=%d,%d want 116,100", chromaDst[0], chromaDst[1])
	}
}

func TestPublicOutputFilterRejectsInvalid(t *testing.T) {
	src := internalframe.SamplePlane{Pix: make([]uint16, 4), Stride: 4, Width: 4, Height: 1}
	dst := internalframe.SamplePlane{Pix: make([]uint16, 8), Stride: 8, Width: 8, Height: 1}
	if err := internalsuperres.UpscalePlaneCoded(src, dst, src.Width, 7); !errors.Is(err, internalframe.ErrInvalidPlane) {
		t.Fatalf("UpscaleSuperResPlane err=%v want %v", err, internalframe.ErrInvalidPlane)
	}
	if _, err := internalfilmgrain.NewPlaneRandom(0, 0); !errors.Is(err, internalfilmgrain.ErrInvalidParams) {
		t.Fatalf("NewFilmGrainPlaneRandom err=%v want %v", err, internalfilmgrain.ErrInvalidParams)
	}
	rng := internalfilmgrain.NewRandom(0)
	if _, err := rng.Number(0); !errors.Is(err, internalfilmgrain.ErrInvalidParams) {
		t.Fatalf("Number err=%v want %v", err, internalfilmgrain.ErrInvalidParams)
	}
	var lut [internalfilmgrain.ScalingLUTSize]uint8
	if err := internalfilmgrain.BuildScalingLUT(lut[:], []internalfilmgrain.ScalingPoint{{Value: 10}, {Value: 10}}); !errors.Is(err, internalfilmgrain.ErrInvalidParams) {
		t.Fatalf("BuildFilmGrainScalingLUT err=%v want %v", err, internalfilmgrain.ErrInvalidParams)
	}
	if err := internalfilmgrain.GenerateLumaGrain(make([]int16, internalfilmgrain.LumaGrainSamples), internalfilmgrain.LumaGrainParams{BitDepth: 9}); !errors.Is(err, internalfilmgrain.ErrInvalidParams) {
		t.Fatalf("GenerateFilmGrainLuma err=%v want %v", err, internalfilmgrain.ErrInvalidParams)
	}
}

func TestPublicOutputFilterAllocs(t *testing.T) {
	superSrc := internalframe.SamplePlane{Pix: []uint16{0, 32, 64, 96, 128, 160, 192, 224}, Stride: 8, Width: 8, Height: 1}
	superDst := internalframe.SamplePlane{Pix: make([]uint16, 13), Stride: 13, Width: 13, Height: 1}
	points := [...]internalfilmgrain.ScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}}
	var lut [internalfilmgrain.ScalingLUTSize]uint8
	lumaGrain := make([]int16, internalfilmgrain.LumaGrainSamples)
	lumaParams := internalfilmgrain.LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	lumaParams.ARCoeffs[3] = 64
	rowDst, rowSrc := publicFilmGrainRowBuffers(4, 2, 8, 100)
	rowParams := internalfilmgrain.LumaRowParams{
		Seed:         0,
		Width:        4,
		Height:       2,
		Stride:       8,
		BitDepth:     8,
		ScalingShift: 8,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := internalsuperres.UpscalePlaneCoded(superSrc, superDst, superSrc.Width, 8); err != nil {
			t.Fatalf("UpscaleSuperResPlane err=%v", err)
		}
		rng := internalfilmgrain.NewRandom(0x1234)
		if _, err := rng.Number(11); err != nil {
			t.Fatalf("Number err=%v", err)
		}
		if err := internalfilmgrain.BuildScalingLUT(lut[:], points[:]); err != nil {
			t.Fatalf("BuildFilmGrainScalingLUT err=%v", err)
		}
		if err := internalfilmgrain.GenerateLumaGrain(lumaGrain, lumaParams); err != nil {
			t.Fatalf("GenerateFilmGrainLuma err=%v", err)
		}
		if err := internalfilmgrain.ApplyLumaRow(rowDst, rowSrc, lumaGrain, lut[:], rowParams); err != nil {
			t.Fatalf("ApplyFilmGrainLumaRow err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicFilmGrainRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func publicFilmGrainScaling(value uint8) [internalfilmgrain.ScalingLUTSize]uint8 {
	var lut [internalfilmgrain.ScalingLUTSize]uint8
	for i := range lut {
		lut[i] = value
	}
	return lut
}

func publicFilmGrainLumaSamples(width int, height int, stride int, subsamplingX bool, subsamplingY bool, value uint16) []uint16 {
	shiftX := publicFilmGrainShift(subsamplingX)
	shiftY := publicFilmGrainShift(subsamplingY)
	rows := ((height - 1) << shiftY) + 1
	samples := make([]uint16, stride*rows)
	for y := range rows {
		for x := 0; x < ((width-1)<<shiftX)+1+shiftX; x++ {
			samples[y*stride+x] = value
		}
	}
	return samples
}

func publicSetFilmGrainLuma(grain []int16, offset uint8, blockCol int, blockRow int, x int, y int, value int16) {
	col := publicFilmGrainLumaOffset(int(offset>>4)) + x + internalfilmgrain.LumaBlockSize*blockCol
	row := publicFilmGrainLumaOffset(int(offset&0x0f)) + y + internalfilmgrain.LumaBlockSize*blockRow
	grain[row*internalfilmgrain.LumaGrainWidth+col] = value
}

func publicSetFilmGrainChroma(grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int, value int16) {
	shiftX := publicFilmGrainShift(subsamplingX)
	shiftY := publicFilmGrainShift(subsamplingY)
	col := publicFilmGrainChromaOffset(int(offset>>4), shiftX) + x + (internalfilmgrain.LumaBlockSize>>shiftX)*blockCol
	row := publicFilmGrainChromaOffset(int(offset&0x0f), shiftY) + y + (internalfilmgrain.LumaBlockSize>>shiftY)*blockRow
	grain[row*internalfilmgrain.ChromaGrainWidth+col] = value
}

func publicFilmGrainLumaOffset(n int) int {
	return 3 + internalfilmgrain.LumaOverlapSamples*(3+n)
}

func publicFilmGrainChromaOffset(n int, shift int) int {
	return 3 + (internalfilmgrain.LumaOverlapSamples>>shift)*(3+n)
}

func publicFilmGrainShift(subsampled bool) int {
	if subsampled {
		return 1
	}
	return 0
}

func TestPublicDecoderCDEFSparseMD5(t *testing.T) {
	if os.Getenv("GOAV1_PUBLIC_FULLCLIP") != "1" {
		t.Skip("set GOAV1_PUBLIC_FULLCLIP=1 for the focused public corpus gate")
	}
	selected := 0
	for _, clip := range loadPublicFullClipCorpus(t) {
		if clip.name != "p360_intra_q32" && clip.name != "p720_inter_q32" {
			continue
		}
		selected++
		t.Run(clip.name, func(t *testing.T) {
			dec, err := NewDecoderFromIVF(clip.data, WithWorkers(1))
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			for round := range 2 {
				if round != 0 {
					if err := dec.Reset(); err != nil {
						t.Fatal(err)
					}
				}
				result, err := decodePublicFullClip(dec, true)
				if err != nil {
					t.Fatal(err)
				}
				if err := comparePublicFullClipOracle(result, clip.oracle); err != nil {
					t.Fatal(err)
				}
				if result.frames != 48 {
					t.Fatalf("frames=%d want 48", result.frames)
				}
				t.Logf("round=%d frames=%d streamMD5=%x", round, result.frames, result.streamMD5)
			}
		})
	}
	if selected != 2 {
		t.Fatalf("selected=%d want 2", selected)
	}
}

func TestDecoderCompactLoopFilterSelectionAndWorkerFallback(t *testing.T) {
	path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", "profile0-420-8bit-sframe-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	decodeFirst := func(workers int) (*Decoder, [md5.Size]byte) {
		t.Helper()
		dec, err := NewDecoderFromIVF(ivf, WithWorkers(workers), WithMaxFrameDelay(1))
		if err != nil {
			t.Fatalf("NewDecoderFromIVF workers=%d: %v", workers, err)
		}
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			dec.Close()
			t.Fatalf("DecodeNext workers=%d: %v", workers, err)
		}
		if !ok || len(frames) != 1 {
			dec.Close()
			t.Fatalf("DecodeNext workers=%d returned ok=%v frames=%d", workers, ok, len(frames))
		}
		return dec, visibleFrameMD5(frames[0])
	}

	serial, serialMD5 := decodeFirst(1)
	defer serial.Close()
	if serial.useExternal {
		t.Fatal("fixture unexpectedly selected an external output route")
	}
	if serial.postFilter.runner.Result.LoopFilter.Plan.Active == false || serial.postFilter.runner.Result.LoopFilter.Plan.Cells == 0 {
		t.Fatal("fixture did not exercise active loop filtering")
	}
	if len(serial.batch.LoopFilterMap.Records) != 0 {
		t.Fatalf("eligible private one-worker route populated %d legacy loop-filter records", len(serial.batch.LoopFilterMap.Records))
	}

	parallel, parallelMD5 := decodeFirst(2)
	defer parallel.Close()
	if parallel.postFilter.runner.Result.LoopFilter.Plan.Active == false || parallel.postFilter.runner.Result.LoopFilter.Plan.Cells == 0 {
		t.Fatal("two-worker fallback did not exercise active loop filtering")
	}
	if len(parallel.batch.LoopFilterMap.Records) == 0 && !parallel.batch.LoopFilterMasks.MapFree() {
		t.Fatal("two-worker fallback has neither map-free masks nor the loop-filter record map")
	}
	if parallelMD5 != serialMD5 {
		t.Fatalf("worker fallback changed visible output: one-worker=%x two-worker=%x", serialMD5, parallelMD5)
	}
}

func TestDecoderCompactLoopFilterFallsBackForMultiTileAndSuperRes(t *testing.T) {
	for _, tc := range []struct {
		name         string
		file         string
		wantExternal bool
	}{
		{name: "multi-tile", file: "multitile-2x1-rows-256x256.ivf"},
		{name: "super-res", file: "superres-420-8bit-160x128.ivf", wantExternal: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", tc.file)
			ivf, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			dec, err := NewDecoderFromIVF(ivf)
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			frames, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("DecodeNext: %v", err)
			}
			if !ok || len(frames) != 1 {
				t.Fatalf("DecodeNext returned ok=%v frames=%d", ok, len(frames))
			}
			if dec.useExternal != tc.wantExternal {
				t.Fatalf("external output route=%v want %v", dec.useExternal, tc.wantExternal)
			}
			var loopFilterActive bool
			if dec.useExternal {
				loopFilterActive = dec.external.supported.runner.Result.LoopFilter.Plan.Active
			} else {
				loopFilterActive = dec.postFilter.runner.Result.LoopFilter.Plan.Active
			}
			if !loopFilterActive && !tc.wantExternal {
				t.Fatal("fixture did not exercise active loop filtering")
			}
			if loopFilterActive {
				// The non-compact route deblocks from the decode-time masks: either
				// the masks carry levels, tile-edge backups and noskip (no record
				// map, as dav1d), or the record map is bound for the edge-list path.
				if len(dec.batch.LoopFilterMap.Records) == 0 && !dec.batch.LoopFilterMasks.MapFree() {
					t.Fatal("ineligible route has neither map-free masks nor the loop-filter record map")
				}
				t.Logf("fallback verified: external=%v active loop filter, legacy records=%d", dec.useExternal, len(dec.batch.LoopFilterMap.Records))
			} else {
				t.Logf("super-res fixture has no active loop filter; owner predicate test covers fallback gating")
			}
			if internalthreading.FrameWorkLoopFilterCompactStateActive(&dec.compactLF) {
				t.Fatal("ineligible route left compact state active")
			}
		})
	}
}

func TestDecoderCompactLoopFilterOwnerRejectsCopiesAndHooks(t *testing.T) {
	side, compact := compactRootSideData(t)
	batch := &decoderFrameWorkBatchResidualRunner{}
	batch.compactState = compact
	batch.compactOwner = batch
	batch.compactSideOwner = side
	if err := batch.setOwnedCompactSideData(side); err != nil {
		t.Fatal(err)
	}
	post := &decoderFrameWorkReusableSupportedPostFilterRunner{}
	runner := decoderFrameWorkResidualEventRunner{
		Workers:           1,
		BatchRunner:       batch,
		SideData:          side,
		sideRunner:        decoderFrameWorkResidualSideDataRunner{SideData: side, BatchRunner: batch},
		compactState:      compact,
		compactNoSkip:     make([]byte, 8),
		compactPostOwner:  post,
		compactBatchOwner: batch,
		compactSideOwner:  side,
	}
	event := &internaldecoder.Event{
		Kind:       internaldecoder.EventFrameHeader,
		FrameSize:  FrameSize{CodedWidth: 16, Height: 16},
		TileInfo:   internalparser.TileInfo{Cols: 1, Rows: 1},
		LoopFilter: internalparser.LoopFilterParams{LevelY: [2]uint8{1, 1}},
	}
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true}}
	if !runner.ownsCompactLoopFilter(sequence, post, event) {
		t.Fatal("exact owned one-tile route was not accepted")
	}

	copiedSide := *side
	runner.SideData = &copiedSide
	if runner.ownsCompactLoopFilter(sequence, post, event) {
		t.Fatal("copied side-data handle inherited compact ownership")
	}
	runner.SideData = side
	customPost := compactLoopFilterTestPostFilter{}
	if runner.ownsCompactLoopFilter(sequence, customPost, event) {
		t.Fatal("custom postfilter hook inherited compact ownership")
	}
	runner.Workers = 2
	if runner.ownsCompactLoopFilter(sequence, post, event) {
		t.Fatal("multi-worker route inherited compact ownership")
	}
	runner.Workers = 1
	multiTile := *event
	multiTile.Kind = internaldecoder.EventTileGroup
	multiTile.TileGroup = internalparser.TileGroup{TileCount: 2, Final: true}
	if runner.ownsCompactLoopFilter(sequence, post, &multiTile) {
		t.Fatal("multi-tile route inherited compact ownership")
	}
	superRes := *event
	superRes.FrameSize.SuperResEnabled = true
	if runner.ownsCompactLoopFilter(sequence, post, &superRes) {
		t.Fatal("super-res route inherited compact ownership")
	}
	hbdSequence := sequence
	hbdSequence.ColorConfig.BitDepth = 10
	if runner.ownsCompactLoopFilter(hbdSequence, post, event) {
		t.Fatal("high-bit-depth route inherited compact ownership")
	}
}

func TestDecoderCompactLoopFilterRunErrorInvalidatesAndPublicRebindDeactivates(t *testing.T) {
	side, compact := compactRootSideData(t)
	batch := &decoderFrameWorkBatchResidualRunner{}
	batch.compactState = compact
	batch.compactOwner = batch
	batch.compactSideOwner = side
	if err := batch.setOwnedCompactSideData(side); err != nil {
		t.Fatal(err)
	}
	if err := batch.RunPtr(nil); err == nil {
		t.Fatal("RunPtr accepted a missing batch")
	}
	if !internalthreading.FrameWorkLoopFilterCompactStateActive(compact) ||
		internalthreading.FrameWorkLoopFilterCompactReady(&side.LoopFilterMasks, compact, 4, 4) {
		t.Fatal("failed owned RunPtr did not leave active state invalidated and fail-closed")
	}
	if err := batch.SetSideData(*side); err != nil {
		t.Fatal(err)
	}
	if internalthreading.FrameWorkLoopFilterCompactStateActive(compact) || batch.compactBoundSide != nil {
		t.Fatal("successful public SetSideData retained private compact authority")
	}
}

func TestDecoderCompactLoopFilterWarmResetAllocations(t *testing.T) {
	path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", "profile0-420-8bit-sframe-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := NewDecoderFromIVF(ivf, WithWorkers(1))
	if err != nil {
		t.Fatalf("NewDecoderFromIVF: %v", err)
	}
	defer dec.Close()
	decodeAll := func() {
		if err := dec.Reset(); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		visible := 0
		for {
			frames, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("DecodeNext: %v", err)
			}
			if !ok {
				break
			}
			visible += len(frames)
		}
		if visible != 8 {
			t.Fatalf("decoded %d visible frames, want 8", visible)
		}
	}
	decodeAll()
	allocs := testing.AllocsPerRun(20, decodeAll)
	t.Logf("one-worker compact-route warm DecodeNext+Reset allocs/run=%f", allocs)
	if allocs != 0 {
		t.Fatalf("warm compact DecodeNext+Reset allocated %f objects/run, want 0", allocs)
	}
}

func compactRootSideData(t *testing.T) (*decoderFrameWorkSideData, *internalthreading.FrameWorkLoopFilterCompactState) {
	t.Helper()
	side := &decoderFrameWorkSideData{
		CDEFIndexMap: internaldecoder.FrameWorkCDEFIndexMap{Index: make([]uint8, 1), Read: make([]bool, 1), Stride: 1, Rows: 1},
		LoopFilterMap: internaldecoder.FrameWorkLoopFilterMap{
			Records: make([]internalthreading.FrameWorkLoopFilterBlockRecord, 16),
			Stride:  4,
			Rows:    4,
		},
		LoopFilterMasks: internalthreading.FrameWorkLoopFilterMasks{
			Masks:      make([]lfmask.FilterMask, 1),
			LevelCache: make([]uint8, 4*(16)),
			Cols:       4,
			Rows:       4,
			SB128W:     1,
			SB128H:     1,
			Layout:     lfmask.Layout{Mono: true},
		},
		RestorationFrameBuffers: internalthreading.FrameWorkRestorationFrameBuffers{Plan: tile.RestorationFramePlan{Planes: 1}},
	}
	compact := &internalthreading.FrameWorkLoopFilterCompactState{}
	if err := internalthreading.BindFrameWorkLoopFilterCompactState(&side.LoopFilterMasks, compact, make([]byte, 8), internalparser.LoopFilterParams{LevelY: [2]uint8{1, 1}}, internalparser.SegmentationParams{}, internalparser.DeltaParams{}, true); err != nil {
		t.Fatal(err)
	}
	return side, compact
}

type compactLoopFilterTestPostFilter struct{}

func (compactLoopFilterTestPostFilter) Apply(internaldecoder.FrameWorkPostFilterContext) error {
	return nil
}

func visibleFrameMD5(f *Frame) [md5.Size]byte {
	h := md5.New()
	bps := f.Layout.BytesPerSample
	for _, plane := range []FramePlane{f.Y, f.U, f.V} {
		if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * bps
		for row := 0; row < plane.Height; row++ {
			off := row * plane.Stride
			_, _ = h.Write(plane.Pix[off : off+rowBytes])
		}
	}
	var sum [md5.Size]byte
	copy(sum[:], h.Sum(nil))
	return sum
}
