package prediction

import (
	"errors"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestCFLAlphaQ3MatchesLibaom(t *testing.T) {
	for jointSign := range int8(cflJointSigns) {
		for _, alphaIndex := range []uint8{0, 1, 15, 16, 17, 31, 127, 255} {
			for _, predType := range []CFLPredType{CFLPredU, CFLPredV} {
				got, err := CFLAlphaQ3(alphaIndex, jointSign, predType)
				if err != nil {
					t.Fatalf("alphaIndex=%d joint=%d pred=%d: %v", alphaIndex, jointSign, predType, err)
				}
				want := cflAlphaQ3Reference(alphaIndex, jointSign, predType)
				if got != want {
					t.Fatalf("alphaIndex=%d joint=%d pred=%d alpha=%d want %d", alphaIndex, jointSign, predType, got, want)
				}
			}
		}
	}
}

func TestCFLSubsampleMatchesLibaomCorpus(t *testing.T) {
	for _, size := range libaomCFLTxSizes {
		for _, sub := range []struct {
			name string
			x    bool
			y    bool
		}{
			{name: "444"},
			{name: "422", x: true},
			{name: "420", x: true, y: true},
		} {
			t.Run(sub.name, func(t *testing.T) {
				rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
				for iter := range 100 {
					stride := size.width + 7
					src8 := make([]uint8, stride*size.height)
					for i := range src8 {
						src8[i] = uint8(rnd.pseudoUniform(1 << 8))
					}
					got8 := make([]uint16, CFLBufSquare)
					want8 := make([]uint16, CFLBufSquare)
					cflSubsample8Reference(want8, src8, stride, size.width, size.height, sub.x, sub.y)
					if err := SubsampleLuma8ToQ3(got8, src8, stride, size.width, size.height, sub.x, sub.y); err != nil {
						t.Fatalf("lbd size=%dx%d iter=%d: %v", size.width, size.height, iter, err)
					}
					if !slices.Equal(got8, want8) {
						t.Fatalf("lbd size=%dx%d iter=%d mismatch", size.width, size.height, iter)
					}

					for _, bitDepth := range []uint8{10, 12} {
						max := int(1 << bitDepth)
						src16 := make([]uint16, stride*size.height)
						for i := range src16 {
							src16[i] = uint16(rnd.pseudoUniform(max))
						}
						got16 := make([]uint16, CFLBufSquare)
						want16 := make([]uint16, CFLBufSquare)
						cflSubsample16Reference(want16, src16, stride, size.width, size.height, sub.x, sub.y)
						if err := SubsampleLuma16ToQ3(got16, src16, stride, size.width, size.height, sub.x, sub.y, bitDepth); err != nil {
							t.Fatalf("hbd size=%dx%d bitDepth=%d iter=%d: %v", size.width, size.height, bitDepth, iter, err)
						}
						if !slices.Equal(got16, want16) {
							t.Fatalf("hbd size=%dx%d bitDepth=%d iter=%d mismatch", size.width, size.height, bitDepth, iter)
						}
					}
				}
			})
		}
	}
}

func TestSubtractCFLAverageMatchesLibaomCorpus(t *testing.T) {
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for _, size := range libaomCFLTxSizes {
		for iter := range 100 {
			src := make([]uint16, CFLBufSquare)
			for row := 0; row < size.height; row++ {
				for col := 0; col < size.width; col++ {
					src[row*CFLBufLine+col] = uint16(rnd.pseudoUniform(1 << 15))
				}
			}
			got := make([]int16, CFLBufSquare)
			want := make([]int16, CFLBufSquare)
			cflSubtractAverageReference(src, want, size.width, size.height)
			if err := SubtractCFLAverage(src, got, size.width, size.height); err != nil {
				t.Fatalf("size=%dx%d iter=%d: %v", size.width, size.height, iter, err)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("size=%dx%d iter=%d mismatch", size.width, size.height, iter)
			}
		}
	}
}

func TestPredictCFLPlaneBlockMatchesLibaomCorpus(t *testing.T) {
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for _, bitDepth := range []uint8{8, 10, 12} {
		bytesPerSample := 1
		if bitDepth > 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		for _, size := range libaomCFLTxSizes {
			for iter := range 100 {
				alphaQ3 := rnd.pseudoUniform(33) - 16
				dc := uint16(rnd.pseudoUniform(int(max) + 1))
				ac := make([]int16, CFLBufSquare)
				for row := 0; row < size.height; row++ {
					for col := 0; col < size.width; col++ {
						ac[row*CFLBufLine+col] = int16(rnd.pseudoUniform(1 << (bitDepth + 3)))
					}
				}
				gotPlane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
				wantPlane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
				fillCFLTestPlane(gotPlane, bytesPerSample, dc)
				fillCFLTestPlane(wantPlane, bytesPerSample, dc)
				cflPredictReference(wantPlane, bytesPerSample, bitDepth, ac, alphaQ3, size.width, size.height)
				if err := PredictCFLPlaneBlock(gotPlane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, ac, alphaQ3); err != nil {
					t.Fatalf("bitDepth=%d size=%dx%d iter=%d: %v", bitDepth, size.width, size.height, iter, err)
				}
				if !slices.Equal(gotPlane.Pix, wantPlane.Pix) {
					t.Fatalf("bitDepth=%d size=%dx%d iter=%d mismatch", bitDepth, size.width, size.height, iter)
				}
			}
		}
	}
}

func TestPredictCFLPlaneBlockVisibleUsesPaddedAC(t *testing.T) {
	plane, _ := testPlane(16, 12, 1, 16)
	fillCFLTestPlane(plane, 1, 64)
	ac := make([]int16, CFLBufSquare)
	for row := range 16 {
		for col := range 16 {
			ac[row*CFLBufLine+col] = int16(row + col)
		}
	}
	if err := PredictCFLPlaneBlockVisible(plane, 1, 8, 0, 0, 16, 12, 16, 16, ac, 8); err != nil {
		t.Fatal(err)
	}
	if got := plane.Pix[11*plane.Stride+15]; got == 64 {
		t.Fatal("visible CfL prediction did not update clipped edge sample")
	}
}

func TestPadCFLReconQ3MatchesLibaom(t *testing.T) {
	recon := make([]uint16, CFLBufSquare)
	for row := range 4 {
		for col := range 6 {
			recon[row*CFLBufLine+col] = uint16(10*row + col)
		}
	}
	gotW, gotH, err := PadCFLReconQ3(recon, 6, 4, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	if gotW != 8 || gotH != 8 {
		t.Fatalf("size=%dx%d want 8x8", gotW, gotH)
	}
	for row := range 4 {
		if recon[row*CFLBufLine+6] != recon[row*CFLBufLine+5] || recon[row*CFLBufLine+7] != recon[row*CFLBufLine+5] {
			t.Fatalf("row %d horizontal pad failed", row)
		}
	}
	for row := 4; row < 8; row++ {
		for col := range 8 {
			if recon[row*CFLBufLine+col] != recon[3*CFLBufLine+col] {
				t.Fatalf("row %d col %d vertical pad=%d want %d", row, col, recon[row*CFLBufLine+col], recon[3*CFLBufLine+col])
			}
		}
	}
}

func TestCFLRejectsInvalidInputs(t *testing.T) {
	recon := make([]uint16, CFLBufSquare)
	ac := make([]int16, CFLBufSquare)
	plane, _ := testPlane(4, 4, 1, 4)
	tests := []struct {
		name string
		fn   func() error
	}{
		{name: "alpha-sign", fn: func() error { _, err := CFLAlphaQ3(0, 8, CFLPredU); return err }},
		{name: "alpha-plane", fn: func() error { _, err := CFLAlphaQ3(0, 0, CFLPredType(9)); return err }},
		{name: "subsample-short-input", fn: func() error { return SubsampleLuma8ToQ3(recon, []uint8{1, 2, 3}, 4, 4, 4, false, false) }},
		{name: "subsample-bad-depth", fn: func() error { return SubsampleLuma16ToQ3(recon, make([]uint16, 16), 4, 4, 4, false, false, 8) }},
		{name: "subtract-invalid-size", fn: func() error { return SubtractCFLAverage(recon, ac, 4, 12) }},
		{name: "predict-short-ac", fn: func() error { return PredictCFLPlaneBlock(plane, 1, 8, 0, 0, 4, 4, ac[:8], 1) }},
		{name: "predict-alpha", fn: func() error { return PredictCFLPlaneBlock(plane, 1, 8, 0, 0, 4, 4, ac, 17) }},
		{name: "pad-invalid-buffer", fn: func() error { _, _, err := PadCFLReconQ3(recon[:8], 4, 4, 4, 4); return err }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.fn(); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestCFLAllocs(t *testing.T) {
	src8 := make([]uint8, 32*32)
	src16 := make([]uint16, 32*32)
	recon := make([]uint16, CFLBufSquare)
	ac := make([]int16, CFLBufSquare)
	plane, _ := testPlane(32, 32, 2, 64)
	fillCFLTestPlane(plane, 2, 512)
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for i := range src8 {
		src8[i] = uint8(rnd.pseudoUniform(1 << 8))
		src16[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := SubsampleLuma8ToQ3(recon, src8, 32, 32, 32, true, true); err != nil {
			t.Fatal(err)
		}
		if err := SubsampleLuma16ToQ3(recon, src16, 32, 32, 32, false, false, 12); err != nil {
			t.Fatal(err)
		}
		if _, _, err := PadCFLReconQ3(recon, 16, 16, 32, 32); err != nil {
			t.Fatal(err)
		}
		if err := SubtractCFLAverage(recon, ac, 32, 32); err != nil {
			t.Fatal(err)
		}
		if err := PredictCFLPlaneBlock(plane, 2, 10, 0, 0, 32, 32, ac, 7); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("CFL primitives allocated: %f", allocs)
	}
}

func FuzzCFLSubsampleAndPredict(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), int8(1), []byte{0, 16, 32, 64, 128, 255})
	f.Add(uint8(6), uint8(2), uint8(3), int8(-4), []byte{255, 127, 1, 2, 3})

	f.Fuzz(func(t *testing.T, rawSize uint8, rawBitDepth uint8, rawSub uint8, rawAlpha int8, data []byte) {
		size := libaomCFLTxSizes[rawSize%uint8(len(libaomCFLTxSizes))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		subX := rawSub&1 != 0
		subY := rawSub&2 != 0
		recon := make([]uint16, CFLBufSquare)
		if bitDepth == 8 {
			input := make([]uint8, size.width*size.height)
			for i := range input {
				input[i] = uint8(fuzzCFLValue(data, i, 0xff))
			}
			if err := SubsampleLuma8ToQ3(recon, input, size.width, size.width, size.height, subX, subY); err != nil {
				t.Fatalf("SubsampleLuma8ToQ3 err=%v", err)
			}
		} else {
			input := make([]uint16, size.width*size.height)
			for i := range input {
				input[i] = fuzzCFLValue(data, i, max)
			}
			if err := SubsampleLuma16ToQ3(recon, input, size.width, size.width, size.height, subX, subY, bitDepth); err != nil {
				t.Fatalf("SubsampleLuma16ToQ3 err=%v", err)
			}
		}
		width := size.width
		height := size.height
		if subX {
			width >>= 1
		}
		if subY {
			height >>= 1
		}
		if !validCFLSize(width, height) {
			return
		}
		ac := make([]int16, CFLBufSquare)
		if err := SubtractCFLAverage(recon, ac, width, height); err != nil {
			t.Fatalf("SubtractCFLAverage err=%v", err)
		}
		alphaQ3 := int(rawAlpha)
		if alphaQ3 < -16 {
			alphaQ3 = -16
		}
		if alphaQ3 > 16 {
			alphaQ3 = 16
		}
		bytesPerSample := 1
		if bitDepth > 8 {
			bytesPerSample = 2
		}
		plane, _ := testPlane(width, height, bytesPerSample, width*bytesPerSample)
		fillCFLTestPlane(plane, bytesPerSample, max>>1)
		if err := PredictCFLPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, width, height, ac, alphaQ3); err != nil {
			t.Fatalf("PredictCFLPlaneBlock err=%v", err)
		}
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				if got := getSample(plane, bytesPerSample, col, row); got > max {
					t.Fatalf("sample(%d,%d)=%d max=%d", col, row, got, max)
				}
			}
		}
	})
}

func BenchmarkPredictCFLPlaneBlock(b *testing.B) {
	ac := make([]int16, CFLBufSquare)
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for i := range ac {
		ac[i] = int16(rnd.pseudoUniform(1 << 15))
	}
	plane, _ := testPlane(32, 32, 2, 64)
	fillCFLTestPlane(plane, 2, 512)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictCFLPlaneBlock(plane, 2, 10, 0, 0, 32, 32, ac, 7)
	}
}

type cflTestSize struct {
	width  int
	height int
}

var libaomCFLTxSizes = [...]cflTestSize{
	{4, 4}, {4, 8}, {4, 16},
	{8, 4}, {8, 8}, {8, 16}, {8, 32},
	{16, 4}, {16, 8}, {16, 16}, {16, 32},
	{32, 8}, {32, 16}, {32, 32},
}

func cflAlphaQ3Reference(alphaIndex uint8, jointSign int8, predType CFLPredType) int {
	sign := cflSignUReference(int(jointSign))
	magnitude := int(alphaIndex) >> cflAlphabetSizeLog2
	if predType == CFLPredV {
		sign = cflSignVReference(int(jointSign))
		magnitude = int(alphaIndex) & (cflAlphabetSize - 1)
	}
	if sign == cflSignZero {
		return 0
	}
	if sign == cflSignPositive {
		return magnitude + 1
	}
	return -magnitude - 1
}

func cflSignUReference(jointSign int) int {
	return ((jointSign + 1) * 11) >> 5
}

func cflSignVReference(jointSign int) int {
	return jointSign + 1 - cflSigns*cflSignUReference(jointSign)
}

func cflSubsample8Reference(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, subX bool, subY bool) {
	switch {
	case subX && subY:
		for row := 0; row < height; row += 2 {
			for col := 0; col < width; col += 2 {
				bot := (row+1)*inputStride + col
				outputQ3[(row>>1)*CFLBufLine+(col>>1)] = uint16(int(input[row*inputStride+col])+int(input[row*inputStride+col+1])+int(input[bot])+int(input[bot+1])) << 1
			}
		}
	case subX:
		for row := range height {
			for col := 0; col < width; col += 2 {
				outputQ3[row*CFLBufLine+(col>>1)] = uint16(int(input[row*inputStride+col])+int(input[row*inputStride+col+1])) << 2
			}
		}
	default:
		for row := range height {
			for col := range width {
				outputQ3[row*CFLBufLine+col] = uint16(input[row*inputStride+col]) << 3
			}
		}
	}
}

func cflSubsample16Reference(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, subX bool, subY bool) {
	switch {
	case subX && subY:
		for row := 0; row < height; row += 2 {
			for col := 0; col < width; col += 2 {
				bot := (row+1)*inputStride + col
				outputQ3[(row>>1)*CFLBufLine+(col>>1)] = (input[row*inputStride+col] + input[row*inputStride+col+1] + input[bot] + input[bot+1]) << 1
			}
		}
	case subX:
		for row := range height {
			for col := 0; col < width; col += 2 {
				outputQ3[row*CFLBufLine+(col>>1)] = (input[row*inputStride+col] + input[row*inputStride+col+1]) << 2
			}
		}
	default:
		for row := range height {
			for col := range width {
				outputQ3[row*CFLBufLine+col] = input[row*inputStride+col] << 3
			}
		}
	}
}

func cflSubtractAverageReference(srcQ3 []uint16, dstQ3 []int16, width int, height int) {
	log2, _ := log2PowerOfTwoInt(width * height)
	sum := (width * height) >> 1
	for row := range height {
		for col := range width {
			sum += int(srcQ3[row*CFLBufLine+col])
		}
	}
	avg := sum >> log2
	for row := range height {
		for col := range width {
			dstQ3[row*CFLBufLine+col] = int16(int(srcQ3[row*CFLBufLine+col]) - avg)
		}
	}
}

func cflPredictReference(plane frame.Plane, bytesPerSample int, bitDepth uint8, acQ3 []int16, alphaQ3 int, width int, height int) {
	max := int((1 << bitDepth) - 1)
	for row := range height {
		for col := range width {
			current := int(getSample(plane, bytesPerSample, col, row))
			scaled := cflRoundPowerOfTwoSignedReference(alphaQ3*int(acQ3[row*CFLBufLine+col]), 6)
			setCFLTestSample(plane, bytesPerSample, col, row, uint16(cflClampReference(current+scaled, 0, max)))
		}
	}
}

func cflRoundPowerOfTwoSignedReference(value int, bits int) int {
	if value < 0 {
		return -((-value + (1 << (bits - 1))) >> bits)
	}
	return (value + (1 << (bits - 1))) >> bits
}

func cflClampReference(v int, lo int, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func fillCFLTestPlane(plane frame.Plane, bytesPerSample int, value uint16) {
	for row := 0; row < plane.Height; row++ {
		for col := 0; col < plane.Width; col++ {
			setCFLTestSample(plane, bytesPerSample, col, row, value)
		}
	}
}

func setCFLTestSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func fuzzCFLValue(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}
