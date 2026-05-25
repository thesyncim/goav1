package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicSuperResUpscalePlane(t *testing.T) {
	src := []uint16{0, 32, 64, 96, 128, 160, 192, 224}
	dst := make([]uint16, 13)
	srcPlane := av1.FrameSamplePlane{Pix: src, Stride: len(src), Width: len(src), Height: 1}
	dstPlane := av1.FrameSamplePlane{Pix: dst, Stride: len(dst), Width: len(dst), Height: 1}
	if err := av1.UpscaleSuperResPlane(srcPlane, dstPlane, 8); err != nil {
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
	srcPlane = av1.FrameSamplePlane{Pix: src, Stride: len(src), Width: len(src), Height: 1}
	dstPlane = av1.FrameSamplePlane{Pix: dst, Stride: len(dst), Width: len(dst), Height: 1}
	if err := av1.UpscaleSuperResPlane(srcPlane, dstPlane, 10); err != nil {
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
	if av1.FilmGrainGaussian(0) != 56 || av1.FilmGrainGaussian(av1.FilmGrainGaussianSequenceLen) != 56 {
		t.Fatalf("unexpected gaussian table lookup")
	}
	rng := av1.NewFilmGrainRandom(0x1234)
	for i, want := range []uint16{0x89, 0x44, 0x22, 0x91, 0xc8, 0xe4, 0xf2, 0xf9} {
		got, err := rng.Number(8)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("random step %d=%#x want %#x", i, got, want)
		}
	}
	cb, err := av1.NewFilmGrainPlaneRandom(0x1234, av1.FilmGrainChromaPlaneCb)
	if err != nil {
		t.Fatal(err)
	}
	cr, err := av1.NewFilmGrainPlaneRandom(0x1234, av1.FilmGrainChromaPlaneCr)
	if err != nil {
		t.Fatal(err)
	}
	stripe, err := av1.NewFilmGrainStripeRandom(0x1234, 32)
	if err != nil {
		t.Fatal(err)
	}
	if cb.Register() != 0xa710 || cr.Register() != 0x5bec || stripe.Register() != 0xc522 {
		t.Fatalf("registers cb=%#x cr=%#x stripe=%#x", cb.Register(), cr.Register(), stripe.Register())
	}

	var lut [av1.FilmGrainScalingLUTSize]uint8
	points := []av1.FilmGrainScalingPoint{{Value: 10, Scaling: 20}, {Value: 20, Scaling: 40}}
	if err := av1.BuildFilmGrainScalingLUT(lut[:], points); err != nil {
		t.Fatal(err)
	}
	for index, want := range map[int]uint8{0: 20, 9: 20, 10: 20, 11: 22, 14: 28, 15: 30, 19: 38, 20: 40, 255: 40} {
		if got := lut[index]; got != want {
			t.Fatalf("lut[%d]=%d want %d", index, got, want)
		}
	}
	if got, err := av1.FilmGrainScaleLUT(lut[:], 40, 10); err != nil || got != 20 {
		t.Fatalf("scale lut=%d err=%v want 20,nil", got, err)
	}

	luma := make([]int16, av1.FilmGrainLumaGrainSamples)
	publicSetFilmGrainLuma(luma, 0xd9, 1, 1, 0, 0, 77)
	if got, err := av1.FilmGrainLumaSample(luma, 0xd9, 1, 1, 0, 0); err != nil || got != 77 {
		t.Fatalf("luma sample=%d err=%v want 77,nil", got, err)
	}
	chroma := make([]int16, av1.FilmGrainChromaGrainSamples)
	publicSetFilmGrainChroma(chroma, 0xd9, true, true, 1, 1, 0, 0, -13)
	if got, err := av1.FilmGrainChromaSample(chroma, 0xd9, true, true, 1, 1, 0, 0); err != nil || got != -13 {
		t.Fatalf("chroma sample=%d err=%v want -13,nil", got, err)
	}
}

func TestPublicGenerateFilmGrainTemplates(t *testing.T) {
	luma := make([]int16, av1.FilmGrainLumaGrainSamples)
	lumaParams := av1.FilmGrainLumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	if err := av1.GenerateFilmGrainLuma(luma, lumaParams); err != nil {
		t.Fatal(err)
	}
	for i, want := range [...]int16{-29, -1, -31, -20, 49, 8, -56, 54, 18, -7, -35, -47, -51, -3, 60, 26} {
		if luma[i] != want {
			t.Fatalf("luma[%d]=%d want %d", i, luma[i], want)
		}
	}
	if got := luma[3*av1.FilmGrainLumaGrainWidth+3]; got != -14 {
		t.Fatalf("luma[3][3]=%d want -14", got)
	}

	chroma := make([]int16, av1.FilmGrainChromaGrainSamples)
	chromaParams := av1.FilmGrainChromaGrainParams{
		Seed:            0x1234,
		Plane:           av1.FilmGrainChromaPlaneCb,
		BitDepth:        8,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
	}
	if err := av1.GenerateFilmGrainChroma(chroma, nil, chromaParams); err != nil {
		t.Fatal(err)
	}
	for i, want := range [...]int16{12, -37, 4, -18, -13, 6, -5, -43, -8, 11, -29, 48, 67, 57, 20, 13} {
		if chroma[i] != want {
			t.Fatalf("chroma[%d]=%d want %d", i, chroma[i], want)
		}
	}
	if chroma[av1.FilmGrainChromaSubsampledGrainWidth] != 0 ||
		chroma[av1.FilmGrainChromaSubsampledGrainHeight*av1.FilmGrainChromaGrainWidth] != 0 {
		t.Fatalf("inactive subsampled chroma samples are not zero")
	}
}

func TestPublicApplyFilmGrainRowsAndSamples(t *testing.T) {
	lut := publicFilmGrainScaling(64)
	if got, err := av1.ApplyFilmGrainLumaSample(100, 64, lut[:], 8, 8, false); err != nil || got != 116 {
		t.Fatalf("ApplyFilmGrainLumaSample=%d err=%v want 116,nil", got, err)
	}
	if got, err := av1.BlendFilmGrainLumaOverlap(20, 100, 0, 8); err != nil || got != 70 {
		t.Fatalf("BlendFilmGrainLumaOverlap=%d err=%v want 70,nil", got, err)
	}

	dst, src := publicFilmGrainRowBuffers(4, 2, 8, 100)
	lumaGrain := make([]int16, av1.FilmGrainLumaGrainSamples)
	publicSetFilmGrainLuma(lumaGrain, 0xd9, 0, 0, 0, 0, 64)
	lumaParams := av1.FilmGrainLumaRowParams{
		Seed:         0,
		Width:        4,
		Height:       2,
		Stride:       8,
		BitDepth:     8,
		ScalingShift: 8,
	}
	if err := av1.ApplyFilmGrainLumaRow(dst, src, lumaGrain, lut[:], lumaParams); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 116 || dst[1] != 100 || dst[8] != 100 {
		t.Fatalf("luma row dst[0,1,8]=%d,%d,%d want 116,100,100", dst[0], dst[1], dst[8])
	}

	chromaDst, chromaSrc := publicFilmGrainRowBuffers(4, 2, 8, 100)
	lumaSamples := publicFilmGrainLumaSamples(4, 2, 8, false, false, 80)
	chromaGrain := make([]int16, av1.FilmGrainChromaGrainSamples)
	publicSetFilmGrainChroma(chromaGrain, 0xd9, false, false, 0, 0, 0, 0, 64)
	chromaParams := av1.FilmGrainChromaRowParams{
		Seed:                  0,
		Width:                 4,
		Height:                2,
		Stride:                8,
		LumaStride:            8,
		BitDepth:              8,
		ScalingShift:          8,
		ChromaScalingFromLuma: true,
	}
	if err := av1.ApplyFilmGrainChromaRow(chromaDst, chromaSrc, lumaSamples, chromaGrain, lut[:], chromaParams); err != nil {
		t.Fatal(err)
	}
	if chromaDst[0] != 116 || chromaDst[1] != 100 {
		t.Fatalf("chroma row dst[0,1]=%d,%d want 116,100", chromaDst[0], chromaDst[1])
	}
}

func TestPublicOutputFilterRejectsInvalid(t *testing.T) {
	src := av1.FrameSamplePlane{Pix: make([]uint16, 4), Stride: 4, Width: 4, Height: 1}
	dst := av1.FrameSamplePlane{Pix: make([]uint16, 8), Stride: 8, Width: 8, Height: 1}
	if err := av1.UpscaleSuperResPlane(src, dst, 7); !errors.Is(err, av1.ErrFrameInvalidPlane) {
		t.Fatalf("UpscaleSuperResPlane err=%v want %v", err, av1.ErrFrameInvalidPlane)
	}
	if _, err := av1.NewFilmGrainPlaneRandom(0, 0); !errors.Is(err, av1.ErrFilmGrainInvalidParams) {
		t.Fatalf("NewFilmGrainPlaneRandom err=%v want %v", err, av1.ErrFilmGrainInvalidParams)
	}
	rng := av1.NewFilmGrainRandom(0)
	if _, err := rng.Number(0); !errors.Is(err, av1.ErrFilmGrainInvalidParams) {
		t.Fatalf("Number err=%v want %v", err, av1.ErrFilmGrainInvalidParams)
	}
	var lut [av1.FilmGrainScalingLUTSize]uint8
	if err := av1.BuildFilmGrainScalingLUT(lut[:], []av1.FilmGrainScalingPoint{{Value: 10}, {Value: 10}}); !errors.Is(err, av1.ErrFilmGrainInvalidParams) {
		t.Fatalf("BuildFilmGrainScalingLUT err=%v want %v", err, av1.ErrFilmGrainInvalidParams)
	}
	if err := av1.GenerateFilmGrainLuma(make([]int16, av1.FilmGrainLumaGrainSamples), av1.FilmGrainLumaGrainParams{BitDepth: 9}); !errors.Is(err, av1.ErrFilmGrainInvalidParams) {
		t.Fatalf("GenerateFilmGrainLuma err=%v want %v", err, av1.ErrFilmGrainInvalidParams)
	}
	if _, err := av1.FilmGrainLumaSample(nil, 0, 0, 0, 0, 0); !errors.Is(err, av1.ErrFilmGrainInvalidParams) {
		t.Fatalf("FilmGrainLumaSample err=%v want %v", err, av1.ErrFilmGrainInvalidParams)
	}
}

func TestPublicOutputFilterAllocs(t *testing.T) {
	superSrc := av1.FrameSamplePlane{Pix: []uint16{0, 32, 64, 96, 128, 160, 192, 224}, Stride: 8, Width: 8, Height: 1}
	superDst := av1.FrameSamplePlane{Pix: make([]uint16, 13), Stride: 13, Width: 13, Height: 1}
	points := [...]av1.FilmGrainScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}}
	var lut [av1.FilmGrainScalingLUTSize]uint8
	lumaGrain := make([]int16, av1.FilmGrainLumaGrainSamples)
	lumaParams := av1.FilmGrainLumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	lumaParams.ARCoeffs[3] = 64
	rowDst, rowSrc := publicFilmGrainRowBuffers(4, 2, 8, 100)
	rowParams := av1.FilmGrainLumaRowParams{
		Seed:         0,
		Width:        4,
		Height:       2,
		Stride:       8,
		BitDepth:     8,
		ScalingShift: 8,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := av1.UpscaleSuperResPlane(superSrc, superDst, 8); err != nil {
			t.Fatalf("UpscaleSuperResPlane err=%v", err)
		}
		rng := av1.NewFilmGrainRandom(0x1234)
		if _, err := rng.Number(11); err != nil {
			t.Fatalf("Number err=%v", err)
		}
		if err := av1.BuildFilmGrainScalingLUT(lut[:], points[:]); err != nil {
			t.Fatalf("BuildFilmGrainScalingLUT err=%v", err)
		}
		if _, err := av1.FilmGrainScaleLUT(lut[:], 1023, 10); err != nil {
			t.Fatalf("FilmGrainScaleLUT err=%v", err)
		}
		if err := av1.GenerateFilmGrainLuma(lumaGrain, lumaParams); err != nil {
			t.Fatalf("GenerateFilmGrainLuma err=%v", err)
		}
		if err := av1.ApplyFilmGrainLumaRow(rowDst, rowSrc, lumaGrain, lut[:], rowParams); err != nil {
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
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func publicFilmGrainScaling(value uint8) [av1.FilmGrainScalingLUTSize]uint8 {
	var lut [av1.FilmGrainScalingLUTSize]uint8
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
	for y := 0; y < rows; y++ {
		for x := 0; x < ((width-1)<<shiftX)+1+shiftX; x++ {
			samples[y*stride+x] = value
		}
	}
	return samples
}

func publicSetFilmGrainLuma(grain []int16, offset uint8, blockCol int, blockRow int, x int, y int, value int16) {
	col := publicFilmGrainLumaOffset(int(offset>>4)) + x + av1.FilmGrainLumaBlockSize*blockCol
	row := publicFilmGrainLumaOffset(int(offset&0x0f)) + y + av1.FilmGrainLumaBlockSize*blockRow
	grain[row*av1.FilmGrainLumaGrainWidth+col] = value
}

func publicSetFilmGrainChroma(grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int, value int16) {
	shiftX := publicFilmGrainShift(subsamplingX)
	shiftY := publicFilmGrainShift(subsamplingY)
	col := publicFilmGrainChromaOffset(int(offset>>4), shiftX) + x + (av1.FilmGrainLumaBlockSize>>shiftX)*blockCol
	row := publicFilmGrainChromaOffset(int(offset&0x0f), shiftY) + y + (av1.FilmGrainLumaBlockSize>>shiftY)*blockRow
	grain[row*av1.FilmGrainChromaGrainWidth+col] = value
}

func publicFilmGrainLumaOffset(n int) int {
	return 3 + av1.FilmGrainLumaOverlapSamples*(3+n)
}

func publicFilmGrainChromaOffset(n int, shift int) int {
	return 3 + (av1.FilmGrainLumaOverlapSamples>>shift)*(3+n)
}

func publicFilmGrainShift(subsampled bool) int {
	if subsampled {
		return 1
	}
	return 0
}
