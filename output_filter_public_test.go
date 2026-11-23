package goav1

import (
	"errors"
	"testing"

	internalfilmgrain "github.com/thesyncim/goav1/internal/av1/filmgrain"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalsuperres "github.com/thesyncim/goav1/internal/av1/superres"
)

func TestPublicSuperResUpscalePlane(t *testing.T) {
	src := []uint16{0, 32, 64, 96, 128, 160, 192, 224}
	dst := make([]uint16, 13)
	srcPlane := internalframe.SamplePlane{Pix: src, Stride: len(src), Width: len(src), Height: 1}
	dstPlane := internalframe.SamplePlane{Pix: dst, Stride: len(dst), Width: len(dst), Height: 1}
	if err := internalsuperres.UpscalePlane(srcPlane, dstPlane, 8); err != nil {
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
	if err := internalsuperres.UpscalePlane(srcPlane, dstPlane, 10); err != nil {
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
	cb, err := internalfilmgrain.NewPlaneRandom(0x1234, internalfilmgrain.ChromaPlaneCb)
	if err != nil {
		t.Fatal(err)
	}
	cr, err := internalfilmgrain.NewPlaneRandom(0x1234, internalfilmgrain.ChromaPlaneCr)
	if err != nil {
		t.Fatal(err)
	}
	stripe, err := internalfilmgrain.NewStripeRandom(0x1234, 32)
	if err != nil {
		t.Fatal(err)
	}
	if cb.Register() != 0xa710 || cr.Register() != 0x5bec || stripe.Register() != 0xc522 {
		t.Fatalf("registers cb=%#x cr=%#x stripe=%#x", cb.Register(), cr.Register(), stripe.Register())
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
	if got, err := internalfilmgrain.ScaleLUT(lut[:], 40, 10); err != nil || got != 20 {
		t.Fatalf("scale lut=%d err=%v want 20,nil", got, err)
	}

	luma := make([]int16, internalfilmgrain.LumaGrainSamples)
	publicSetFilmGrainLuma(luma, 0xd9, 1, 1, 0, 0, 77)
	if got, err := internalfilmgrain.LumaGrainSample(luma, 0xd9, 1, 1, 0, 0); err != nil || got != 77 {
		t.Fatalf("luma sample=%d err=%v want 77,nil", got, err)
	}
	chroma := make([]int16, internalfilmgrain.ChromaGrainSamples)
	publicSetFilmGrainChroma(chroma, 0xd9, true, true, 1, 1, 0, 0, -13)
	if got, err := internalfilmgrain.ChromaGrainSample(chroma, 0xd9, true, true, 1, 1, 0, 0); err != nil || got != -13 {
		t.Fatalf("chroma sample=%d err=%v want -13,nil", got, err)
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
	if got, err := internalfilmgrain.ApplyLumaSample(100, 64, lut[:], 8, 8, false); err != nil || got != 116 {
		t.Fatalf("ApplyFilmGrainLumaSample=%d err=%v want 116,nil", got, err)
	}
	if got, err := internalfilmgrain.BlendLumaOverlap(20, 100, 0, 8); err != nil || got != 70 {
		t.Fatalf("BlendFilmGrainLumaOverlap=%d err=%v want 70,nil", got, err)
	}

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
	if err := internalsuperres.UpscalePlane(src, dst, 7); !errors.Is(err, internalframe.ErrInvalidPlane) {
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
	if _, err := internalfilmgrain.LumaGrainSample(nil, 0, 0, 0, 0, 0); !errors.Is(err, internalfilmgrain.ErrInvalidParams) {
		t.Fatalf("FilmGrainLumaSample err=%v want %v", err, internalfilmgrain.ErrInvalidParams)
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
		if err := internalsuperres.UpscalePlane(superSrc, superDst, 8); err != nil {
			t.Fatalf("UpscaleSuperResPlane err=%v", err)
		}
		rng := internalfilmgrain.NewRandom(0x1234)
		if _, err := rng.Number(11); err != nil {
			t.Fatalf("Number err=%v", err)
		}
		if err := internalfilmgrain.BuildScalingLUT(lut[:], points[:]); err != nil {
			t.Fatalf("BuildFilmGrainScalingLUT err=%v", err)
		}
		if _, err := internalfilmgrain.ScaleLUT(lut[:], 1023, 10); err != nil {
			t.Fatalf("FilmGrainScaleLUT err=%v", err)
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
