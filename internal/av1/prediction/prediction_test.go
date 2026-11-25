package prediction

import (
	"bytes"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"slices"
	"testing"
)

// Frozen pre-change pixel walk; do not delegate to the optimized range helper.
func originalDirectionalZ2Ranges(width int, height int, upsampleAbove int, upsampleLeft int, dx int, dy int) (sampleIndexRange, sampleIndexRange, error) {
	minBaseX := -(1 << upsampleAbove)
	minBaseY := -(1 << upsampleLeft)
	fracBitsX := 6 - upsampleAbove
	fracBitsY := 6 - upsampleLeft
	var aboveRange sampleIndexRange
	var leftRange sampleIndexRange
	for row := range height {
		for col := range width {
			y := row + 1
			x := (col << 6) - y*dx
			baseX := x >> fracBitsX
			if baseX >= minBaseX {
				originalRangeAdd(&aboveRange, baseX)
				originalRangeAdd(&aboveRange, baseX+1)
				continue
			}
			x = col + 1
			y = (row << 6) - x*dy
			baseY := y >> fracBitsY
			if baseY < minBaseY {
				return sampleIndexRange{}, sampleIndexRange{}, ErrInvalidPrediction
			}
			originalRangeAdd(&leftRange, baseY)
			originalRangeAdd(&leftRange, baseY+1)
		}
	}
	return aboveRange, leftRange, nil
}

func originalRangeAdd(r *sampleIndexRange, index int) {
	if !r.used {
		r.used, r.min, r.max = true, index, index
		return
	}
	if index < r.min {
		r.min = index
	}
	if index > r.max {
		r.max = index
	}
}

func TestDirectionalZ2EndpointRangesMatchOriginal(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128}
	for angle := 91; angle < 180; angle++ {
		dx, dy := DirectionalDX(angle), DirectionalDY(angle)
		if dx <= 0 || dy <= 0 {
			continue
		}
		for _, w := range sizes {
			for _, h := range sizes {
				for ua := range 2 {
					for ul := range 2 {
						a, l, err := directionalZ2Ranges(w, h, ua, ul, dx, dy)
						wa, wl, we := originalDirectionalZ2Ranges(w, h, ua, ul, dx, dy)
						if a != wa || l != wl || err != we {
							t.Fatalf("%dx%d angle%d up%d,%d: got%+v/%+v/%v want%+v/%+v/%v", w, h, angle, ua, ul, a, l, err, wa, wl, we)
						}
					}
				}
			}
		}
	}
	// Invalid left projections and inputs beyond the algebraic fast domain.
	for _, c := range [][6]int{{16, 16, 0, 0, 1023, 1023}, {129, 3, 0, 1, 64, 64}, {4, 4, 2, 0, 64, 64}, {3, 4, 0, 0, 0, 0}, {2, 1, 0, 0, 2048, 1}, {0, 4, 0, 0, 64, 64}, {4, -1, 0, 0, 64, 64}} {
		a, l, e := directionalZ2Ranges(c[0], c[1], c[2], c[3], c[4], c[5])
		wa, wl, we := originalDirectionalZ2Ranges(c[0], c[1], c[2], c[3], c[4], c[5])
		if a != wa || l != wl || e != we {
			t.Fatalf("fallback%v: got%+v/%+v/%v want%+v/%+v/%v", c, a, l, e, wa, wl, we)
		}
	}
}

func TestByteEdgePredictionMatchesValidated(t *testing.T) {
	for _, shape := range [][4]int{{4, 4, 4, 4}, {15, 7, 16, 8}, {32, 32, 32, 32}, {63, 31, 64, 32}} {
		w, h, pw, ph := shape[0], shape[1], shape[2], shape[3]
		edges := IntraEdges{Above: make([]uint16, pw), Left: make([]uint16, ph), AboveLeft: 255, AboveAvailable: true, LeftAvailable: true, AboveLeftAvailable: true}
		for i := range edges.Above {
			edges.Above[i] = uint16((i*79 + 3) & 255)
		}
		for i := range edges.Left {
			edges.Left[i] = uint16((i*71 + 131) & 255)
		}
		for mode := IntraModeDC; mode <= IntraModeSmoothHorizontal; mode++ {
			for missing := 0; missing < 4; missing++ {
				e := edges
				e.AboveAvailable = missing&1 == 0
				e.LeftAvailable = missing&2 == 0
				want := frame.Plane{Pix: bytes.Repeat([]byte{0x5a}, (w+4)*(h+2)), Stride: w + 4, Width: w + 4, Height: h + 2}
				got := want
				got.Pix = slices.Clone(want.Pix)
				we := PredictIntraPlaneBlockWithExtent(want, 1, 8, 1, 1, w, h, pw, ph, mode, e)
				ge := PredictIntraPlaneBlockWithExtentBounded(got, 1, 8, 1, 1, w, h, pw, ph, mode, e, true)
				if we != ge || !bytes.Equal(want.Pix, got.Pix) {
					t.Fatalf("shape%v mode%d missing%d: error%v/%v or pixels differ", shape, mode, missing, we, ge)
				}
			}
		}
		if pw <= 32 && ph <= 32 {
			for mode := FilterIntraModeDC; mode < FilterIntraModes; mode++ {
				want := frame.Plane{Pix: bytes.Repeat([]byte{0x5a}, (w+4)*(h+2)), Stride: w + 4, Width: w + 4, Height: h + 2}
				got := want
				got.Pix = slices.Clone(want.Pix)
				we := PredictFilterIntraPlaneBlockWithExtentBounded(want, 1, 8, 1, 1, w, h, pw, ph, mode, edges, false)
				ge := PredictFilterIntraPlaneBlockWithExtentBounded(got, 1, 8, 1, 1, w, h, pw, ph, mode, edges, true)
				if we != ge || !bytes.Equal(want.Pix, got.Pix) {
					t.Fatalf("filter shape%v mode%d: %v/%v or pixels differ", shape, mode, we, ge)
				}
			}
		}
	}
}

func TestByteEdgeFiltersMatchValidated(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 8, 17, 33, 65, 129, 257} {
		for strength := uint8(0); strength < 4; strength++ {
			want := make([]uint16, n)
			for i := range want {
				want[i] = uint16((i*79 + 3) & 255)
			}
			got := slices.Clone(want)
			ws, gs := make([]uint16, n+2), make([]uint16, n+2)
			we := FilterIntraEdgeBounded(want, ws, strength, 8, false)
			ge := FilterIntraEdgeBounded(got, gs, strength, 8, true)
			if we != ge || !slices.Equal(want, got) || !slices.Equal(ws, gs) {
				t.Fatalf("filter n%d strength%d mismatch", n, strength)
			}
		}
	}
	for n := 1; n <= 16; n++ {
		want := make([]uint16, 2*n+4)
		for i := range want {
			want[i] = uint16((i*79 + 3) & 255)
		}
		got := slices.Clone(want)
		ws, gs := make([]uint16, n+5), make([]uint16, n+5)
		we := UpsampleIntraEdgeBounded(want, 2, n, ws, 8, false)
		ge := UpsampleIntraEdgeBounded(got, 2, n, gs, 8, true)
		if we != ge || !slices.Equal(want, got) || !slices.Equal(ws, gs) {
			t.Fatalf("upsample n%d mismatch", n)
		}
	}
}

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

var libaomDRDerivative = [90]int16{
	0, 0, 0,
	1023, 0, 0,
	547, 0, 0,
	372, 0, 0, 0, 0,
	273, 0, 0,
	215, 0, 0,
	178, 0, 0,
	151, 0, 0,
	132, 0, 0,
	116, 0, 0,
	102, 0, 0, 0,
	90, 0, 0,
	80, 0, 0,
	71, 0, 0,
	64, 0, 0,
	57, 0, 0,
	51, 0, 0,
	45, 0, 0, 0,
	40, 0, 0,
	35, 0, 0,
	31, 0, 0,
	27, 0, 0,
	23, 0, 0,
	19, 0, 0,
	15, 0, 0, 0, 0,
	11, 0, 0,
	7, 0, 0,
	3, 0, 0,
}

func TestDirectionalDerivativesMatchLibaom(t *testing.T) {
	if !slices.Equal(drIntraDerivative[:], libaomDRDerivative[:]) {
		t.Fatalf("drIntraDerivative=%v want %v", drIntraDerivative, libaomDRDerivative)
	}
	for angle := -16; angle <= 286; angle++ {
		if got, want := DirectionalDX(angle), directionalDXLibaomReference(angle); got != want {
			t.Fatalf("dx angle=%d got=%d want=%d", angle, got, want)
		}
		if got, want := DirectionalDY(angle), directionalDYLibaomReference(angle); got != want {
			t.Fatalf("dy angle=%d got=%d want=%d", angle, got, want)
		}
	}
}

func TestPredictDirectionalIntraPlaneBlockKnownVectors(t *testing.T) {
	tests := []struct {
		name   string
		angle  int
		width  int
		height int
		edges  DirectionalEdges
		want   []uint16
	}{
		{
			name:   "zone1-45",
			angle:  45,
			width:  4,
			height: 4,
			edges: DirectionalEdges{
				Above:       []uint16{10, 20, 30, 40, 50, 60, 70, 80},
				AboveOrigin: 0,
			},
			want: []uint16{
				20, 30, 40, 50,
				30, 40, 50, 60,
				40, 50, 60, 70,
				50, 60, 70, 80,
			},
		},
		{
			name:   "zone2-135",
			angle:  135,
			width:  4,
			height: 4,
			edges: DirectionalEdges{
				Above:       []uint16{0, 0, 0, 11, 21, 31, 41, 51},
				Left:        []uint16{0, 0, 0, 0, 101, 111, 121, 131},
				AboveOrigin: 4,
				LeftOrigin:  4,
			},
			want: []uint16{
				11, 21, 31, 41,
				101, 11, 21, 31,
				111, 101, 11, 21,
				121, 111, 101, 11,
			},
		},
		{
			name:   "zone3-225",
			angle:  225,
			width:  4,
			height: 4,
			edges: DirectionalEdges{
				Left:       []uint16{100, 110, 120, 130, 140, 150, 160, 170},
				LeftOrigin: 0,
			},
			want: []uint16{
				110, 120, 130, 140,
				120, 130, 140, 150,
				130, 140, 150, 160,
				140, 150, 160, 170,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plane, _ := testPlane(tt.width, tt.height, 1, tt.width)
			if err := PredictDirectionalIntraPlaneBlock(plane, 1, 8, 0, 0, tt.width, tt.height, tt.angle, tt.edges); err != nil {
				t.Fatal(err)
			}
			got := collectPlaneSamples(plane, 1, tt.width, tt.height)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("prediction=%v want %v", got, tt.want)
			}
		})
	}
}

// TestPredictDirectionalIntraPlaneBlockCornerFrame32x32D157 captures the exact
// libaom prediction output for the 32x32 D157 (p_angle=151) block at frame
// origin (0,0) where there are no top/left neighbours and the edge filter is
// effectively disabled (n_top_px = n_left_px = 0). The Quantizer-32 / 34x34
// fast vector exercises this exact predictor as the first decoded block.
//
// Edges follow libaom's build_directional_and_filter_intra_predictors fallback
// when no neighbouring samples are available:
//
//	above[-1] = 128 (corner-filter output equals pre-filter value 128)
//	above[ i] = 127 for i = 0..63 (top-right extension is irrelevant for z2)
//	left[-1]  = 128
//	left[ i]  = 129 for i = 0..63 (bottom-left extension is irrelevant for z2)
//
// The golden values are emitted by goav1's patched libaom
// (GOAV1_TRACE_INTRA=1) for the av1-1-b8-01-size-34x34 vector and locked in
// here so a future intra-prediction edit cannot silently regress directional
// zone-2 reconstruction at frame corners.
func TestPredictDirectionalIntraPlaneBlockCornerFrame32x32D157(t *testing.T) {
	const (
		width  = 32
		height = 32
		angle  = 151
		origin = 64
	)
	edges := DirectionalEdges{
		Above:       make([]uint16, 256),
		Left:        make([]uint16, 256),
		AboveOrigin: origin,
		LeftOrigin:  origin,
	}
	for i := range edges.Above {
		edges.Above[i] = 127
	}
	for i := range edges.Left {
		edges.Left[i] = 129
	}
	edges.Above[origin-1] = 128
	edges.Left[origin-1] = 128

	want := [32 * 32]uint16{
		128, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 128, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 128, 127, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127, 127, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 128, 127,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
		129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
	}

	plane, _ := testPlane(width, height, 1, width)
	if err := PredictDirectionalIntraPlaneBlock(plane, 1, 8, 0, 0, width, height, angle, edges); err != nil {
		t.Fatal(err)
	}
	got := collectPlaneSamples(plane, 1, width, height)
	if !slices.Equal(got, want[:]) {
		for row := range height {
			gotRow := got[row*width : (row+1)*width]
			wantRow := want[row*width : (row+1)*width]
			if !slices.Equal(gotRow, wantRow) {
				t.Logf("row=%d got =%v", row, gotRow)
				t.Logf("row=%d want=%v", row, wantRow)
			}
		}
		t.Fatalf("32x32 D157 corner prediction does not match libaom reference")
	}
}

// TestPredictDirectionalIntraPlaneBlockCornerFrame32x32D157HBD mirrors the
// 8-bit corner test for 10-bit content. libaom's default-edge constants are
// base±1 where base = 128 << (bit_depth - 8), so for 10-bit the predictor
// fills above with 511, left with 513, and corner with 512. Each output
// sample is the 8-bit reference scaled by 4 (since the predictor blends
// constant neighbours and the dr_predictor rounding is exact for constant
// inputs).
func TestPredictDirectionalIntraPlaneBlockCornerFrame32x32D157HBD(t *testing.T) {
	const (
		width    = 32
		height   = 32
		angle    = 151
		origin   = 64
		bitDepth = uint8(10)
	)
	edges := DirectionalEdges{
		Above:       make([]uint16, 256),
		Left:        make([]uint16, 256),
		AboveOrigin: origin,
		LeftOrigin:  origin,
	}
	for i := range edges.Above {
		edges.Above[i] = 511
	}
	for i := range edges.Left {
		edges.Left[i] = 513
	}
	edges.Above[origin-1] = 512
	edges.Left[origin-1] = 512

	plane, _ := testPlane(width, height, 2, width*2)
	if err := PredictDirectionalIntraPlaneBlock(plane, 2, bitDepth, 0, 0, width, height, angle, edges); err != nil {
		t.Fatal(err)
	}
	got := collectPlaneSamples(plane, 2, width, height)
	want := [32 * 32]uint16{
		512, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 512, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 512, 511, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511, 511, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 512, 511,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
		513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513,
	}
	if !slices.Equal(got, want[:]) {
		for row := range height {
			gotRow := got[row*width : (row+1)*width]
			wantRow := want[row*width : (row+1)*width]
			if !slices.Equal(gotRow, wantRow) {
				t.Logf("row=%d got =%v", row, gotRow)
				t.Logf("row=%d want=%v", row, wantRow)
			}
		}
		t.Fatalf("10-bit 32x32 D157 corner prediction does not match expected")
	}
}

func TestPredictDirectionalIntraPlaneBlockMatchesLibaomSaturatedCorpus(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		for _, size := range libaomDRTxSizes {
			for _, angleRange := range [][2]int{{1, 89}, {91, 179}, {181, 269}} {
				for enableUpsample := range 2 {
					for angle := angleRange[0]; angle <= angleRange[1]; angle++ {
						if DirectionalDX(angle) == 0 || DirectionalDY(angle) == 0 {
							continue
						}
						edges := saturatedDirectionalEdges(size.width, size.height, angle, enableUpsample == 1, max)
						plane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
						if err := PredictDirectionalIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, angle, edges); err != nil {
							t.Fatalf("bitDepth=%d size=%dx%d angle=%d upsample=%d: %v", bitDepth, size.width, size.height, angle, enableUpsample, err)
						}
						for row := 0; row < size.height; row++ {
							for col := 0; col < size.width; col++ {
								if got := getSample(plane, bytesPerSample, col, row); got != max {
									t.Fatalf("bitDepth=%d size=%dx%d angle=%d upsample=%d sample(%d,%d)=%d want %d", bitDepth, size.width, size.height, angle, enableUpsample, col, row, got, max)
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestPredictDirectionalIntraPlaneBlockMatchesLibaomRandomCorpus(t *testing.T) {
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for _, bitDepth := range []uint8{8, 10, 12} {
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		for _, tc := range []struct {
			width          int
			height         int
			angle          int
			enableUpsample bool
		}{
			{4, 4, 3, false},
			{4, 8, 45, true},
			{8, 4, 87, true},
			{8, 8, 93, false},
			{16, 16, 113, true},
			{8, 16, 157, true},
			{16, 8, 183, false},
			{8, 8, 203, true},
			{4, 16, 267, true},
		} {
			if DirectionalDX(tc.angle) == 0 || DirectionalDY(tc.angle) == 0 {
				continue
			}
			edges := randomDirectionalEdges(tc.width, tc.height, tc.angle, tc.enableUpsample, max, rnd)
			want, err := predictDirectionalLibaomReference(tc.width, tc.height, tc.angle, edges)
			if err != nil {
				t.Fatalf("reference width=%d height=%d angle=%d: %v", tc.width, tc.height, tc.angle, err)
			}
			plane, _ := testPlane(tc.width, tc.height, bytesPerSample, tc.width*bytesPerSample)
			if err := PredictDirectionalIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, tc.width, tc.height, tc.angle, edges); err != nil {
				t.Fatalf("bitDepth=%d width=%d height=%d angle=%d: %v", bitDepth, tc.width, tc.height, tc.angle, err)
			}
			got := collectPlaneSamples(plane, bytesPerSample, tc.width, tc.height)
			if !slices.Equal(got, want) {
				t.Fatalf("bitDepth=%d width=%d height=%d angle=%d got=%v want=%v", bitDepth, tc.width, tc.height, tc.angle, got, want)
			}
		}
	}
}

func TestPredictDirectionalIntraPlaneBlockRejectsInvalidInputs(t *testing.T) {
	plane, _ := testPlane(4, 4, 1, 4)
	valid := DirectionalEdges{
		Above:       make([]uint16, 64),
		Left:        make([]uint16, 64),
		AboveOrigin: 16,
		LeftOrigin:  16,
	}
	tests := []struct {
		name  string
		angle int
		edges DirectionalEdges
	}{
		{name: "angle-zero", angle: 0, edges: valid},
		{name: "angle-270", angle: 270, edges: valid},
		{name: "unused-angle-derivative", angle: 1, edges: valid},
		{name: "short-above", angle: 45, edges: DirectionalEdges{Above: []uint16{1, 2}, AboveOrigin: 0}},
		{name: "short-left", angle: 225, edges: DirectionalEdges{Left: []uint16{1, 2}, LeftOrigin: 0}},
		{name: "negative-zone2-window", angle: 135, edges: DirectionalEdges{Above: make([]uint16, 8), Left: make([]uint16, 8)}},
		{name: "sample-out-of-range", angle: 45, edges: DirectionalEdges{Above: []uint16{0, 1, 2, 3, 4, 5, 6, 256}, AboveOrigin: 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := PredictDirectionalIntraPlaneBlock(plane, 1, 8, 0, 0, 4, 4, tt.angle, tt.edges); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestPredictDirectionalIntraPlaneBlockAllocs(t *testing.T) {
	plane, _ := testPlane(16, 16, 2, 16*2)
	edges := randomDirectionalEdges(16, 16, 113, true, 0x3ff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictDirectionalIntraPlaneBlock(plane, 2, 10, 0, 0, 16, 16, 113, edges); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("directional prediction allocated: %f", allocs)
	}
}

func FuzzPredictDirectionalIntraPlaneBlock(f *testing.F) {
	f.Add(uint8(4), uint8(4), uint16(45), uint8(0), uint8(0), []byte{10, 20, 30, 40, 50, 60, 70, 80})
	f.Add(uint8(8), uint8(8), uint16(135), uint8(1), uint8(1), []byte{1, 2, 3, 4, 5, 6})
	f.Add(uint8(16), uint8(4), uint16(225), uint8(2), uint8(1), []byte{200, 150, 100, 50})

	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawAngle uint16, rawBitDepth uint8, rawUpsample uint8, data []byte) {
		width := 4 * (int(rawW)%4 + 1)
		height := 4 * (int(rawH)%4 + 1)
		angle := int(rawAngle%269) + 1
		if DirectionalDX(angle) == 0 || DirectionalDY(angle) == 0 {
			return
		}
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		edges := fuzzDirectionalEdges(width, height, angle, rawUpsample&1 != 0, max, data)
		plane, _ := testPlane(width, height, bytesPerSample, width*bytesPerSample)
		if err := PredictDirectionalIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, width, height, angle, edges); err != nil {
			t.Fatalf("PredictDirectionalIntraPlaneBlock err=%v", err)
		}
		for row := range height {
			for col := range width {
				if got := getSample(plane, bytesPerSample, col, row); got > max {
					t.Fatalf("sample(%d,%d)=%d exceeds max %d", col, row, got, max)
				}
			}
		}
	})
}

func BenchmarkPredictDirectionalIntraPlaneBlock(b *testing.B) {
	plane, _ := testPlane(32, 32, 2, 32*2)
	edges := randomDirectionalEdges(32, 32, 203, false, 0xfff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictDirectionalIntraPlaneBlock(plane, 2, 12, 0, 0, 32, 32, 203, edges)
	}
}

type directionalTestSize struct {
	width  int
	height int
}

var libaomDRTxSizes = [...]directionalTestSize{
	{4, 4}, {8, 8}, {16, 16}, {32, 32}, {64, 64},
	{4, 8}, {8, 4}, {8, 16}, {16, 8}, {16, 32},
	{32, 16}, {32, 64}, {64, 32}, {4, 16}, {16, 4},
	{8, 32}, {32, 8}, {16, 64}, {64, 16},
}

func collectPlaneSamples(plane frame.Plane, bytesPerSample int, width int, height int) []uint16 {
	out := make([]uint16, 0, width*height)
	for row := range height {
		for col := range width {
			out = append(out, getSample(plane, bytesPerSample, col, row))
		}
	}
	return out
}

func saturatedDirectionalEdges(width int, height int, angle int, enableUpsample bool, value uint16) DirectionalEdges {
	edges := directionalEdgesForAngle(width, height, angle, enableUpsample)
	for i := range edges.Above {
		edges.Above[i] = value
	}
	for i := range edges.Left {
		edges.Left[i] = value
	}
	return edges
}

func randomDirectionalEdges(width int, height int, angle int, enableUpsample bool, max uint16, rnd *libaomIntraEdgeRandom) DirectionalEdges {
	edges := directionalEdgesForAngle(width, height, angle, enableUpsample)
	hi := int(max) + 1
	for i := range edges.Above {
		edges.Above[i] = uint16(rnd.pseudoUniform(hi))
	}
	for i := range edges.Left {
		edges.Left[i] = uint16(rnd.pseudoUniform(hi))
	}
	return edges
}

func fuzzDirectionalEdges(width int, height int, angle int, enableUpsample bool, max uint16, data []byte) DirectionalEdges {
	edges := directionalEdgesForAngle(width, height, angle, enableUpsample)
	for i := range edges.Above {
		edges.Above[i] = fuzzSample(data, i, max)
	}
	for i := range edges.Left {
		edges.Left[i] = fuzzSample(data, len(edges.Above)+i, max)
	}
	return edges
}

func fuzzSample(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}

func directionalEdgesForAngle(width int, height int, angle int, enableUpsample bool) DirectionalEdges {
	origin := 128
	length := 512
	edges := DirectionalEdges{
		Above:       make([]uint16, length),
		Left:        make([]uint16, length),
		AboveOrigin: origin,
		LeftOrigin:  origin,
	}
	if enableUpsample {
		edges.UpsampleAbove = UseIntraEdgeUpsample(width, height, angle-90, false)
		edges.UpsampleLeft = UseIntraEdgeUpsample(height, width, angle-180, false)
	}
	return edges
}

func predictDirectionalLibaomReference(width int, height int, angle int, edges DirectionalEdges) ([]uint16, error) {
	if angle <= 0 || angle >= 270 {
		return nil, ErrInvalidPrediction
	}
	out := make([]uint16, width*height)
	upsampleAbove := boolToInt(edges.UpsampleAbove)
	upsampleLeft := boolToInt(edges.UpsampleLeft)
	switch {
	case angle == 90:
		for row := range height {
			for col := range width {
				out[row*width+col] = edges.Above[edges.AboveOrigin+col]
			}
		}
	case angle == 180:
		for row := range height {
			for col := range width {
				out[row*width+col] = edges.Left[edges.LeftOrigin+row]
			}
		}
	case angle < 90:
		dx := DirectionalDX(angle)
		if dx <= 0 {
			return nil, ErrInvalidPrediction
		}
		maxBaseX := (width + height - 1) << upsampleAbove
		fracBits := 6 - upsampleAbove
		baseInc := 1 << upsampleAbove
		x := dx
		for row := 0; row < height; row++ {
			base := x >> fracBits
			shift := ((x << upsampleAbove) & 0x3f) >> 1
			if base >= maxBaseX {
				for ; row < height; row++ {
					for col := range width {
						out[row*width+col] = edges.Above[edges.AboveOrigin+maxBaseX]
					}
				}
				return out, nil
			}
			for col := range width {
				if base < maxBaseX {
					p0 := int(edges.Above[edges.AboveOrigin+base])
					p1 := int(edges.Above[edges.AboveOrigin+base+1])
					out[row*width+col] = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
				} else {
					out[row*width+col] = edges.Above[edges.AboveOrigin+maxBaseX]
				}
				base += baseInc
			}
			x += dx
		}
	case angle < 180:
		dx := DirectionalDX(angle)
		dy := DirectionalDY(angle)
		if dx <= 0 || dy <= 0 {
			return nil, ErrInvalidPrediction
		}
		minBaseX := -(1 << upsampleAbove)
		minBaseY := -(1 << upsampleLeft)
		fracBitsX := 6 - upsampleAbove
		fracBitsY := 6 - upsampleLeft
		for row := range height {
			for col := range width {
				y := row + 1
				x := (col << 6) - y*dx
				baseX := x >> fracBitsX
				var value uint16
				if baseX >= minBaseX {
					shift := ((x * (1 << upsampleAbove)) & 0x3f) >> 1
					p0 := int(edges.Above[edges.AboveOrigin+baseX])
					p1 := int(edges.Above[edges.AboveOrigin+baseX+1])
					value = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
				} else {
					x = col + 1
					y = (row << 6) - x*dy
					baseY := y >> fracBitsY
					if baseY < minBaseY {
						return nil, ErrInvalidPrediction
					}
					shift := ((y * (1 << upsampleLeft)) & 0x3f) >> 1
					p0 := int(edges.Left[edges.LeftOrigin+baseY])
					p1 := int(edges.Left[edges.LeftOrigin+baseY+1])
					value = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
				}
				out[row*width+col] = value
			}
		}
	default:
		dy := DirectionalDY(angle)
		if dy <= 0 {
			return nil, ErrInvalidPrediction
		}
		maxBaseY := (width + height - 1) << upsampleLeft
		fracBits := 6 - upsampleLeft
		baseInc := 1 << upsampleLeft
		y := dy
		for col := range width {
			base := y >> fracBits
			shift := ((y << upsampleLeft) & 0x3f) >> 1
			for row := 0; row < height; row++ {
				if base < maxBaseY {
					p0 := int(edges.Left[edges.LeftOrigin+base])
					p1 := int(edges.Left[edges.LeftOrigin+base+1])
					out[row*width+col] = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
				} else {
					for ; row < height; row++ {
						out[row*width+col] = edges.Left[edges.LeftOrigin+maxBaseY]
					}
					break
				}
				base += baseInc
			}
			y += dy
		}
	}
	return out, nil
}

func directionalDXLibaomReference(angle int) int {
	if angle > 0 && angle < 90 {
		return int(libaomDRDerivative[angle])
	}
	if angle > 90 && angle < 180 {
		return int(libaomDRDerivative[180-angle])
	}
	return 1
}

func directionalDYLibaomReference(angle int) int {
	if angle > 90 && angle < 180 {
		return int(libaomDRDerivative[angle-90])
	}
	if angle > 180 && angle < 270 {
		return int(libaomDRDerivative[270-angle])
	}
	return 1
}

// TestPredictDirectionalIntraPlaneBlockExtentBottomEdge guards the
// partially-visible directional fix: a zone-1 (D45-style) transform block that
// straddles the bottom frame edge predicts only its visible top rows but must
// drive its base-index accumulation off the FULL transform extent, exactly as
// libaom av1_dr_prediction_z1_c. Predicting with the clipped (visible) height
// shrinks max_base_x and makes the bottom visible rows clamp to the last
// reference sample early, producing a flat run where libaom keeps
// interpolating from the extended above-right reference. This test builds a
// monotonically increasing above edge so a clamp is observable as a flat tail
// and asserts the WithExtent result equals the top half of a full-height
// prediction (and differs from the clipped-dimension prediction).
func TestPredictDirectionalIntraPlaneBlockExtentBottomEdge(t *testing.T) {
	const (
		width      = 16
		visibleH   = 8
		predHeight = 16
		angle      = 36 // zone 1, dx > 64 with fractional shift
		origin     = 8
	)
	mkEdges := func() DirectionalEdges {
		e := DirectionalEdges{
			Above:       make([]uint16, origin+width+predHeight+8),
			Left:        make([]uint16, origin+predHeight+8),
			AboveOrigin: origin,
			LeftOrigin:  origin,
		}
		// Strictly increasing above edge so any early clamp shows up as a flat
		// tail on the bottom-right of the block.
		for i := range e.Above {
			e.Above[i] = uint16(20 + i)
		}
		for i := range e.Left {
			e.Left[i] = uint16(20 + i)
		}
		return e
	}

	// Reference: predict the full 16x16 block, then keep the visible top 8 rows.
	full, _ := testPlane(width, predHeight, 1, width)
	if err := PredictDirectionalIntraPlaneBlock(full, 1, 8, 0, 0, width, predHeight, angle, mkEdges()); err != nil {
		t.Fatal(err)
	}
	fullSamples := collectPlaneSamples(full, 1, width, predHeight)
	wantVisible := fullSamples[:width*visibleH]

	// WithExtent: predict only the visible 8 rows but drive the base math off
	// the full 16-row extent. Must equal the top 8 rows of the full prediction.
	ext, _ := testPlane(width, visibleH, 1, width)
	if err := PredictDirectionalIntraPlaneBlockWithExtent(ext, 1, 8, 0, 0, width, visibleH, width, predHeight, angle, mkEdges()); err != nil {
		t.Fatal(err)
	}
	gotExt := collectPlaneSamples(ext, 1, width, visibleH)
	if !slices.Equal(gotExt, wantVisible) {
		t.Fatalf("WithExtent visible rows do not match full-height prediction\n got=%v\nwant=%v", gotExt, wantVisible)
	}

	// Sanity: the legacy clipped-dimension prediction (extent == visible)
	// diverges from libaom on the bottom-right because max_base_x shrinks.
	clip, _ := testPlane(width, visibleH, 1, width)
	if err := PredictDirectionalIntraPlaneBlockWithExtent(clip, 1, 8, 0, 0, width, visibleH, width, visibleH, angle, mkEdges()); err != nil {
		t.Fatal(err)
	}
	gotClip := collectPlaneSamples(clip, 1, width, visibleH)
	if slices.Equal(gotClip, wantVisible) {
		t.Fatalf("clipped-dimension prediction unexpectedly matched the extended reference; test no longer guards the regression")
	}
}

// withPureGoDirKernels temporarily forces the dispatched directional-zone
// kernels (the zone-2 "above" run and the zone-3 column fill) back to their
// pure-Go references, runs fn, then restores the resolved slots (NEON on arm64,
// AVX2 on amd64 when active). It lets a differential test capture the SIMD
// output and the reference output through the identical predictor entry point.
func withPureGoDirKernels(fn func()) {
	savedAbove := dirAboveRun8Impl
	savedLeft := dirLeftCol8Impl
	dirAboveRun8Impl = dirAboveRun8PureGo
	dirLeftCol8Impl = dirLeftCol8PureGo
	defer func() {
		dirAboveRun8Impl = savedAbove
		dirLeftCol8Impl = savedLeft
	}()
	fn()
}

// dirBlockDims are the square AV1 block dimensions; the test pairs every width
// with every height to also cover the non-square shapes.
var dirBlockDims = []int{4, 8, 16, 32, 64}

// TestDirectionalKernelsDispatchMatchesPureGo guards the dispatched zone-2 and
// zone-3 directional-interpolation kernels in isolation: the resolved slots
// (NEON on arm64) must reproduce the pure-Go references sample for sample. It
// runs on every build; where no tuned variant exists the comparison is trivial.
func TestDirectionalKernelsDispatchMatchesPureGo(t *testing.T) {
	rnd := newLibaomIntraEdgeRandom(0x5151)

	// zone-2 above run: contiguous forward source, no clamp.
	for _, count := range []int{8, 16, 24, 32, 56, 64} {
		ref := make([]uint16, count+1)
		for i := range ref {
			ref[i] = uint16(rnd.pseudoUniform(256))
		}
		for _, shift := range []int{0, 1, 5, 10, 16, 23, 31} {
			got := make([]byte, count)
			want := make([]byte, count)
			dirAboveRun8Impl(got, ref, shift, count)
			dirAboveRun8PureGo(want, ref, shift, count)
			if !slices.Equal(got, want) {
				t.Fatalf("dirAboveRun8 count=%d shift=%d got=%v want=%v", count, shift, got, want)
			}
		}
	}

	// zone-3 column fill: contiguous forward source, strided byte stores.
	for _, count := range []int{8, 16, 24, 32, 56, 64} {
		ref := make([]uint16, count+1)
		for i := range ref {
			ref[i] = uint16(rnd.pseudoUniform(256))
		}
		for _, stride := range []int{1, 8, 17, 32, 71} {
			for _, shift := range []int{0, 1, 5, 10, 16, 23, 31} {
				gotBuf := make([]byte, count*stride+1)
				wantBuf := make([]byte, count*stride+1)
				dirLeftCol8Impl(gotBuf, stride, ref, shift, count)
				dirLeftCol8PureGo(wantBuf, stride, ref, shift, count)
				if !slices.Equal(gotBuf, wantBuf) {
					t.Fatalf("dirLeftCol8 count=%d stride=%d shift=%d differs", count, stride, shift)
				}
			}
		}
	}
}

// TestDirectionalZ2Z3DispatchMatchesPureGo is the end-to-end byte-exactness
// guard for the zone-2 and zone-3 NEON kernels. For every block shape
// (4x4..64x64, square and non-square), every zone-2 angle (91..179) and zone-3
// angle (181..269), upsample on/off, and a couple of partially-visible extents,
// it runs PredictDirectionalIntraPlaneBlock with the resolved dispatch slots
// (NEON on arm64) and with the slots forced to pure-Go, and asserts the two
// planes are byte-identical. It also cross-checks both against the independent
// predictDirectionalLibaomReference re-derivation.
func TestDirectionalZ2Z3DispatchMatchesPureGo(t *testing.T) {
	rnd := newLibaomIntraEdgeRandom(0xC0FFEE)

	type extent struct{ pw, ph int } // 0,0 means full

	for _, w := range dirBlockDims {
		for _, h := range dirBlockDims {
			// Zone 2 = (90,180), Zone 3 = (180,270). Sweep every integer angle;
			// angles 90/180/270 are vertical/horizontal and out of scope here.
			var angles []int
			for a := 91; a <= 179; a++ {
				angles = append(angles, a)
			}
			for a := 181; a <= 269; a++ {
				angles = append(angles, a)
			}

			for _, angle := range angles {
				for _, up := range []bool{false, true} {
					// Partial extents only make sense up to the block size and
					// only exercise extra reference rows/cols, not the kernels'
					// shape, so keep them small and cheap.
					extents := []extent{{0, 0}}
					if w >= 8 && h >= 8 {
						extents = append(extents, extent{w + 4, h + 4})
					}
					for _, ext := range extents {
						edges := randomDirectionalEdges(w, h, angle, up, 0xff, rnd)

						pw, ph := w, h
						if ext.pw != 0 {
							pw, ph = ext.pw, ext.ph
						}

						// Reference re-derivation (full block only).
						var refOut []uint16
						var refErr error
						if ext.pw == 0 {
							refOut, refErr = predictDirectionalLibaomReference(w, h, angle, edges)
						}

						// Resolved dispatch (NEON on arm64).
						neonPlane, _ := testPlane(w, h, 1, w+3)
						neonErr := PredictDirectionalIntraPlaneBlockWithExtent(neonPlane, 1, 8, 0, 0, w, h, pw, ph, angle, edges)

						// Forced pure-Go.
						var pureErr error
						purePlane, _ := testPlane(w, h, 1, w+3)
						withPureGoDirKernels(func() {
							pureErr = PredictDirectionalIntraPlaneBlockWithExtent(purePlane, 1, 8, 0, 0, w, h, pw, ph, angle, edges)
						})

						if (neonErr == nil) != (pureErr == nil) {
							t.Fatalf("w=%d h=%d angle=%d up=%v ext=%v err mismatch: neon=%v pure=%v",
								w, h, angle, up, ext, neonErr, pureErr)
						}
						if neonErr != nil {
							continue
						}

						neon := collectPlaneSamples(neonPlane, 1, w, h)
						pure := collectPlaneSamples(purePlane, 1, w, h)
						if !slices.Equal(neon, pure) {
							diffDirectional(t, "neon-vs-pure", w, h, angle, up, ext.pw, ext.ph, neon, pure)
						}
						if refErr == nil && refOut != nil {
							if !slices.Equal(neon, refOut) {
								diffDirectional(t, "neon-vs-ref", w, h, angle, up, ext.pw, ext.ph, neon, refOut)
							}
						}
					}
				}
			}
		}
	}
}

func diffDirectional(t *testing.T, tag string, w, h, angle int, up bool, pw, ph int, got, want []uint16) {
	t.Helper()
	for row := 0; row < h; row++ {
		for col := 0; col < w; col++ {
			i := row*w + col
			if got[i] != want[i] {
				t.Fatalf("%s w=%d h=%d angle=%d up=%v ext=%dx%d row=%d col=%d got=%d want=%d",
					tag, w, h, angle, up, pw, ph, row, col, got[i], want[i])
			}
		}
	}
}

// Pinned outputs from libaom v3.13.1's av1_filter_intra_predictor_c, captured
// via a standalone harness linked against /tmp/libaom-v3.13.1/out/libaom.a.
// Edge samples below were fed verbatim into both libaom and goav1; the
// expected[] vectors are the bytes libaom wrote into its destination buffer.
// This pins all five filter-intra modes (DC, V, H, D157, PAETH) for two block
// sizes (4x4 and 8x8) at 8-bit depth.

type filterIntraGoldenCase struct {
	name      string
	mode      FilterIntraMode
	width     int
	height    int
	aboveLeft uint16
	above     []uint16
	left      []uint16
	expected  []uint16
}

var filterIntraLibaomGolden = []filterIntraGoldenCase{
	{
		name:  "DC_4x4",
		mode:  FilterIntraModeDC,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			120, 144, 58, 67,
			139, 145, 97, 95,
			55, 80, 64, 72,
			40, 63, 60, 61,
		},
	},
	{
		name:  "DC_8x8",
		mode:  FilterIntraModeDC,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			120, 144, 58, 67, 99, 112, 119, 111,
			139, 145, 97, 95, 108, 111, 120, 113,
			55, 80, 64, 72, 86, 94, 102, 104,
			40, 63, 60, 61, 72, 81, 90, 93,
			82, 86, 78, 75, 78, 83, 87, 91,
			49, 57, 59, 65, 70, 74, 79, 84,
			177, 151, 130, 115, 106, 99, 97, 95,
			103, 98, 94, 103, 102, 97, 95, 97,
		},
	},
	{
		name:  "V_4x4",
		mode:  FilterIntraModeVertical,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			132, 163, 26, 50,
			153, 175, 34, 54,
			84, 134, 7, 40,
			84, 133, 6, 40,
		},
	},
	{
		name:  "V_8x8",
		mode:  FilterIntraModeVertical,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			132, 163, 26, 50, 109, 121, 136, 108,
			153, 175, 34, 54, 111, 123, 137, 109,
			84, 134, 7, 40, 102, 118, 134, 107,
			84, 133, 6, 40, 102, 118, 134, 107,
			124, 157, 22, 48, 107, 121, 136, 108,
			93, 139, 10, 42, 103, 119, 135, 107,
			203, 205, 54, 64, 117, 127, 141, 110,
			125, 158, 23, 48, 107, 121, 137, 108,
		},
	},
	{
		name:  "H_4x4",
		mode:  FilterIntraModeHorizontal,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			131, 155, 91, 108,
			149, 162, 130, 138,
			32, 39, 23, 27,
			28, 31, 23, 25,
		},
	},
	{
		name:  "H_8x8",
		mode:  FilterIntraModeHorizontal,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			131, 155, 91, 108, 139, 147, 155, 141,
			149, 162, 130, 138, 154, 157, 161, 155,
			32, 39, 23, 27, 35, 37, 39, 36,
			28, 31, 23, 25, 29, 30, 31, 29,
			90, 92, 88, 89, 91, 92, 92, 91,
			40, 41, 39, 39, 40, 40, 41, 40,
			216, 216, 215, 215, 216, 216, 216, 216,
			90, 91, 90, 90, 90, 90, 91, 90,
		},
	},
	{
		name:  "D157_4x4",
		mode:  FilterIntraModeD157,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			104, 121, 64, 57,
			123, 125, 96, 82,
			60, 87, 85, 85,
			42, 61, 67, 70,
		},
	},
	{
		name:  "D157_8x8",
		mode:  FilterIntraModeD157,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			104, 121, 64, 57, 82, 97, 114, 107,
			123, 125, 96, 82, 85, 91, 101, 103,
			60, 87, 85, 85, 85, 88, 94, 98,
			42, 61, 67, 70, 76, 81, 87, 91,
			73, 70, 71, 69, 72, 76, 81, 85,
			50, 58, 64, 65, 68, 72, 76, 79,
			155, 117, 100, 82, 77, 75, 76, 77,
			107, 109, 107, 97, 89, 84, 82, 81,
		},
	},
	{
		name:  "PAETH_4x4",
		mode:  FilterIntraModePaeth,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			143, 177, 60, 85,
			165, 192, 90, 100,
			65, 102, 20, 36,
			60, 96, 18, 37,
		},
	},
	{
		name:  "PAETH_8x8",
		mode:  FilterIntraModePaeth,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			143, 177, 60, 85, 134, 141, 153, 126,
			165, 192, 90, 100, 139, 147, 154, 132,
			65, 102, 20, 36, 78, 93, 103, 88,
			60, 96, 18, 37, 74, 90, 93, 88,
			112, 135, 63, 75, 103, 112, 112, 105,
			64, 92, 34, 46, 73, 86, 88, 86,
			215, 217, 156, 155, 165, 163, 158, 149,
			102, 119, 81, 84, 100, 106, 110, 107,
		},
	},
}

// TestFilterIntraPredictorMatchesLibaomGolden pins every filter-intra mode
// against vectors captured directly from libaom v3.13.1's
// av1_filter_intra_predictor_c. These vectors are independent of the goav1
// implementation, so they catch regressions in the rounding, clamping, tap
// table, and edge-sample ordering paths simultaneously.
func TestFilterIntraPredictorMatchesLibaomGolden(t *testing.T) {
	for _, tc := range filterIntraLibaomGolden {
		t.Run(tc.name, func(t *testing.T) {
			edges := IntraEdges{
				Above:              tc.above,
				Left:               tc.left,
				AboveLeft:          tc.aboveLeft,
				AboveAvailable:     true,
				LeftAvailable:      true,
				AboveLeftAvailable: true,
			}
			plane, _ := testPlane(tc.width, tc.height, 1, tc.width)
			if err := PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, tc.width, tc.height, tc.width, tc.height, tc.mode, edges, false); err != nil {
				t.Fatalf("PredictFilterIntraPlaneBlock err=%v", err)
			}
			got := collectPlaneSamples(plane, 1, tc.width, tc.height)
			if !slices.Equal(got, tc.expected) {
				t.Fatalf("mode=%d %dx%d\n got=%v\nwant=%v", tc.mode, tc.width, tc.height, got, tc.expected)
			}
		})
	}
}

var libaomFilterIntraTaps = [FilterIntraModes][8][8]int8{
	{
		{-6, 10, 0, 0, 0, 12, 0, 0},
		{-5, 2, 10, 0, 0, 9, 0, 0},
		{-3, 1, 1, 10, 0, 7, 0, 0},
		{-3, 1, 1, 2, 10, 5, 0, 0},
		{-4, 6, 0, 0, 0, 2, 12, 0},
		{-3, 2, 6, 0, 0, 2, 9, 0},
		{-3, 2, 2, 6, 0, 2, 7, 0},
		{-3, 1, 2, 2, 6, 3, 5, 0},
	},
	{
		{-10, 16, 0, 0, 0, 10, 0, 0},
		{-6, 0, 16, 0, 0, 6, 0, 0},
		{-4, 0, 0, 16, 0, 4, 0, 0},
		{-2, 0, 0, 0, 16, 2, 0, 0},
		{-10, 16, 0, 0, 0, 0, 10, 0},
		{-6, 0, 16, 0, 0, 0, 6, 0},
		{-4, 0, 0, 16, 0, 0, 4, 0},
		{-2, 0, 0, 0, 16, 0, 2, 0},
	},
	{
		{-8, 8, 0, 0, 0, 16, 0, 0},
		{-8, 0, 8, 0, 0, 16, 0, 0},
		{-8, 0, 0, 8, 0, 16, 0, 0},
		{-8, 0, 0, 0, 8, 16, 0, 0},
		{-4, 4, 0, 0, 0, 0, 16, 0},
		{-4, 0, 4, 0, 0, 0, 16, 0},
		{-4, 0, 0, 4, 0, 0, 16, 0},
		{-4, 0, 0, 0, 4, 0, 16, 0},
	},
	{
		{-2, 8, 0, 0, 0, 10, 0, 0},
		{-1, 3, 8, 0, 0, 6, 0, 0},
		{-1, 2, 3, 8, 0, 4, 0, 0},
		{0, 1, 2, 3, 8, 2, 0, 0},
		{-1, 4, 0, 0, 0, 3, 10, 0},
		{-1, 3, 4, 0, 0, 4, 6, 0},
		{-1, 2, 3, 4, 0, 4, 4, 0},
		{-1, 2, 2, 3, 4, 3, 3, 0},
	},
	{
		{-12, 14, 0, 0, 0, 14, 0, 0},
		{-10, 0, 14, 0, 0, 12, 0, 0},
		{-9, 0, 0, 14, 0, 11, 0, 0},
		{-8, 0, 0, 0, 14, 10, 0, 0},
		{-10, 12, 0, 0, 0, 0, 14, 0},
		{-9, 1, 12, 0, 0, 0, 12, 0},
		{-8, 0, 0, 12, 0, 1, 11, 0},
		{-7, 0, 0, 1, 12, 1, 9, 0},
	},
}

var libaomFilterIntraTxSizes = [...]directionalTestSize{
	{4, 4}, {8, 8}, {16, 16}, {32, 32}, {4, 8}, {8, 4}, {8, 16},
	{16, 8}, {16, 32}, {32, 16}, {4, 16}, {16, 4}, {8, 32}, {32, 8},
}

func TestFilterIntraTapsMatchLibaom(t *testing.T) {
	if !slices.Equal(filterIntraTaps[:], libaomFilterIntraTaps[:]) {
		t.Fatalf("filterIntraTaps=%v want %v", filterIntraTaps, libaomFilterIntraTaps)
	}
}

func TestFilterIntraPredictorMatchesLibaomKnownVector(t *testing.T) {
	edges := IntraEdges{
		Above:              []uint16{10, 40, 100, 180},
		Left:               []uint16{20, 60, 120, 200},
		AboveLeft:          30,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	plane, _ := testPlane(4, 4, 1, 4)
	if err := PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, 4, 4, 4, 4, FilterIntraModePaeth, edges, false); err != nil {
		t.Fatal(err)
	}
	want := []uint16{
		4, 31, 84, 155,
		41, 59, 103, 163,
		96, 104, 139, 188,
		168, 163, 192, 222,
	}
	got := collectPlaneSamples(plane, 1, 4, 4)
	if !slices.Equal(got, want) {
		t.Fatalf("prediction=%v want %v", got, want)
	}
}

func TestFilterIntraPredictorWithExtentClipsVisiblePixels(t *testing.T) {
	edges := IntraEdges{
		Above:              []uint16{10, 40, 100, 180, 220, 230, 240, 250},
		Left:               []uint16{20, 60, 120, 200, 210, 220, 230, 240},
		AboveLeft:          30,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	plane, _ := testPlane(6, 6, 1, 6)
	if err := PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, 6, 6, 8, 8, FilterIntraModePaeth, edges, false); err != nil {
		t.Fatal(err)
	}
	full := filterIntraLibaomReference(8, 8, FilterIntraModePaeth, edges, 0xff)
	got := collectPlaneSamples(plane, 1, 6, 6)
	for row := range 6 {
		for col := range 6 {
			if want := full[row*8+col]; got[row*6+col] != want {
				t.Fatalf("sample(%d,%d)=%d want %d", col, row, got[row*6+col], want)
			}
		}
	}
}

func TestFilterIntraPredictorMatchesLibaomCorpus(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		mask := uint16((1 << bitDepth) - 1)
		for mode := range FilterIntraModes {
			for _, size := range libaomFilterIntraTxSizes {
				rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
				for iter := range 32 {
					edges := randomStaticIntraEdges(size.width, size.height, mask, rnd)
					want := filterIntraLibaomReference(size.width, size.height, mode, edges, mask)
					plane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
					if err := PredictFilterIntraPlaneBlockWithExtentBounded(plane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, size.width, size.height, mode, edges, false); err != nil {
						t.Fatalf("bitDepth=%d mode=%d size=%dx%d iter=%d: %v", bitDepth, mode, size.width, size.height, iter, err)
					}
					got := collectPlaneSamples(plane, bytesPerSample, size.width, size.height)
					if !slices.Equal(got, want) {
						t.Fatalf("bitDepth=%d mode=%d size=%dx%d iter=%d got=%v want=%v", bitDepth, mode, size.width, size.height, iter, got, want)
					}
				}
			}
		}
	}
}

func TestFilterIntraPredictorRejectsInvalidInputs(t *testing.T) {
	plane, _ := testPlane(32, 32, 1, 32)
	valid := randomStaticIntraEdges(32, 32, 0xff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	tests := []struct {
		name   string
		width  int
		height int
		mode   FilterIntraMode
		edges  IntraEdges
	}{
		{name: "too-wide", width: 64, height: 16, mode: FilterIntraModeDC, edges: valid},
		{name: "too-tall", width: 16, height: 64, mode: FilterIntraModeDC, edges: valid},
		{name: "bad-width-step", width: 6, height: 8, mode: FilterIntraModeDC, edges: valid},
		{name: "bad-height-step", width: 8, height: 5, mode: FilterIntraModeDC, edges: valid},
		{name: "invalid-mode", width: 8, height: 8, mode: FilterIntraModes, edges: valid},
		{name: "missing-above-left", width: 8, height: 8, mode: FilterIntraModeDC, edges: IntraEdges{Above: make([]uint16, 8), Left: make([]uint16, 8), AboveAvailable: true, LeftAvailable: true}},
		{name: "sample-out-of-range", width: 8, height: 8, mode: FilterIntraModeDC, edges: IntraEdges{Above: []uint16{0, 0, 0, 0, 0, 0, 0, 256}, Left: make([]uint16, 8), AboveAvailable: true, LeftAvailable: true, AboveLeftAvailable: true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, tt.width, tt.height, tt.width, tt.height, tt.mode, tt.edges, false); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestFilterIntraPredictorAllocs(t *testing.T) {
	plane, _ := testPlane(32, 32, 2, 32*2)
	edges := randomStaticIntraEdges(32, 32, 0xfff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictFilterIntraPlaneBlockWithExtentBounded(plane, 2, 12, 0, 0, 32, 32, 32, 32, FilterIntraModePaeth, edges, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("filter intra predictor allocated: %f", allocs)
	}
}

func FuzzFilterIntraPredictor(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), []byte{0, 64, 128, 255})
	f.Add(uint8(4), uint8(3), uint8(1), uint8(2), []byte{255, 0, 127, 32})

	f.Fuzz(func(t *testing.T, rawMode uint8, rawSize uint8, rawBitDepth uint8, rawSeed uint8, data []byte) {
		size := libaomFilterIntraTxSizes[rawSize%uint8(len(libaomFilterIntraTxSizes))]
		mode := FilterIntraMode(rawMode % uint8(FilterIntraModes))
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		edges := fuzzStaticIntraEdges(size.width, size.height, max, rawSeed, data)
		plane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
		if err := PredictFilterIntraPlaneBlockWithExtentBounded(plane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, size.width, size.height, mode, edges, false); err != nil {
			t.Fatalf("PredictFilterIntraPlaneBlock err=%v", err)
		}
		for row := 0; row < size.height; row++ {
			for col := 0; col < size.width; col++ {
				if got := getSample(plane, bytesPerSample, col, row); got > max {
					t.Fatalf("sample(%d,%d)=%d exceeds max %d", col, row, got, max)
				}
			}
		}
	})
}

func BenchmarkFilterIntraPredictor(b *testing.B) {
	plane, _ := testPlane(32, 32, 2, 32*2)
	edges := randomStaticIntraEdges(32, 32, 0xfff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictFilterIntraPlaneBlockWithExtentBounded(plane, 2, 12, 0, 0, 32, 32, 32, 32, FilterIntraModeD157, edges, false)
	}
}

func filterIntraLibaomReference(width int, height int, mode FilterIntraMode, edges IntraEdges, max uint16) []uint16 {
	var buffer [33][33]uint16
	for row := range height {
		buffer[row+1][0] = edges.Left[row]
	}
	buffer[0][0] = edges.AboveLeft
	copy(buffer[0][1:width+1], edges.Above[:width])

	for row := 1; row < height+1; row += 2 {
		for col := 1; col < width+1; col += 4 {
			p := [7]uint16{
				buffer[row-1][col-1],
				buffer[row-1][col],
				buffer[row-1][col+1],
				buffer[row-1][col+2],
				buffer[row-1][col+3],
				buffer[row][col-1],
				buffer[row+1][col-1],
			}
			for k := range 8 {
				sum := 0
				for i := range len(p) {
					sum += int(libaomFilterIntraTaps[mode][k][i]) * int(p[i])
				}
				sample := roundPowerOfTwo(sum, filterIntraScaleBits)
				if sample < 0 {
					sample = 0
				} else if sample > int(max) {
					sample = int(max)
				}
				buffer[row+(k>>2)][col+(k&3)] = uint16(sample)
			}
		}
	}

	out := make([]uint16, 0, width*height)
	for row := range height {
		out = append(out, buffer[row+1][1:width+1]...)
	}
	return out
}

// TestPredictIntra8MatchesPlaneEntry checks the decoder's 8-bit entry against
// PredictIntraPlaneBlock for every mode, block shape and DC edge availability,
// with the edges read from the same plane.
func TestPredictIntra8MatchesPlaneEntry(t *testing.T) {
	sizes := []int{4, 8, 16, 32, 64}
	modes := []IntraMode{IntraModeDC, IntraModeVertical, IntraModeHorizontal, IntraModePaeth, IntraModeSmooth, IntraModeSmoothVertical, IntraModeSmoothHorizontal}
	const x, y, stride, rows = 3, 2, 160, 72
	seed := uint32(1)
	for _, w := range sizes {
		for _, h := range sizes {
			if w > 4*h || h > 4*w {
				continue
			}
			for _, mode := range modes {
				for avail := 0; avail < 4; avail++ {
					haveTop, haveLeft := avail&1 != 0, avail&2 != 0
					if mode != IntraModeDC && (!haveTop || !haveLeft) {
						continue
					}
					for pattern := 0; pattern < 4; pattern++ {
						pix := make([]byte, stride*rows)
						for i := range pix {
							switch pattern {
							case 0:
								seed = seed*1664525 + 1013904223
								pix[i] = byte(seed >> 24)
							case 1:
								pix[i] = 0xff
							case 2:
								pix[i] = byte(i & 1 * 0xff)
							default:
								pix[i] = byte(i * 37)
							}
						}
						want := append([]byte(nil), pix...)
						plane := frame.Plane{Pix: want, Stride: stride, Width: stride, Height: rows}
						var edges IntraEdges
						above := make([]uint16, w)
						left := make([]uint16, h)
						off := y*stride + x
						for i := range above {
							above[i] = uint16(want[off-stride+i])
						}
						for i := range left {
							left[i] = uint16(want[off-1+i*stride])
						}
						if haveTop {
							edges.Above, edges.AboveAvailable = above, true
						}
						if haveLeft {
							edges.Left, edges.LeftAvailable = left, true
						}
						edges.AboveLeft, edges.AboveLeftAvailable = uint16(want[off-stride-1]), true
						if err := PredictIntraPlaneBlock(plane, 1, 8, x, y, w, h, mode, edges); err != nil {
							t.Fatal(err)
						}
						var top, leftIn []byte
						if haveTop {
							top = pix[off-stride:]
						}
						if haveLeft {
							leftIn = make([]byte, h)
							for i := range leftIn {
								leftIn[i] = pix[off-1+i*stride]
							}
						}
						scratch := make([]uint16, w+h)
						end := off + (h-1)*stride + w
						if err := PredictIntra8(pix[off:end], stride, w, h, mode, top, leftIn, pix[off-stride-1], scratch); err != nil {
							t.Fatal(err)
						}
						if !bytes.Equal(pix, want) {
							t.Fatalf("%s", fmt.Sprintf("mode %d %dx%d avail %d pattern %d: output differs", mode, w, h, avail, pattern))
						}
					}
				}
			}
		}
	}
}

func TestPredictIntra8RejectsInvalidShapes(t *testing.T) {
	pix := make([]byte, 64*64)
	edge := make([]byte, 64)
	wide := make([]uint16, 128)
	for _, c := range []struct{ w, h, stride int }{{0, 8, 64}, {8, 0, 64}, {12, 8, 64}, {128, 8, 128}, {8, 8, 4}, {64, 64, 63}} {
		if err := PredictIntra8(pix, c.stride, c.w, c.h, IntraModeSmooth, pix, edge, 0, wide); err == nil {
			t.Fatalf("%dx%d stride %d accepted", c.w, c.h, c.stride)
		}
	}
	if err := PredictIntra8(pix, 64, 8, 8, IntraModeVertical, nil, edge, 0, wide); err == nil {
		t.Fatal("vertical without top accepted")
	}
	if err := PredictIntra8(pix, 64, 8, 8, IntraModeDC, pix[:4], nil, 0, wide); err == nil {
		t.Fatal("DC with short top accepted")
	}
	if err := PredictIntra8(pix, 64, 8, 8, IntraModeSmooth, pix, edge, 0, wide[:15]); err == nil {
		t.Fatal("smooth with short scratch accepted")
	}
}

// BenchmarkPredictIntra8 times the decoder's 8-bit entry per mode and square
// size, the per-call figure compared against dav1d's ipred_*_8bpc_neon.
func BenchmarkPredictIntra8(b *testing.B) {
	pix := make([]byte, 160*72)
	for i := range pix {
		pix[i] = byte(i*37 + 11)
	}
	left := make([]byte, 64)
	for i := range left {
		left[i] = byte(i*53 + 5)
	}
	scratch := make([]uint16, 128)
	off := 2*160 + 3
	names := []string{"dc", "v", "h", "paeth", "smooth", "smooth_v", "smooth_h"}
	for mode, name := range names {
		for _, n := range []int{4, 8, 16, 32, 64} {
			b.Run(fmt.Sprintf("%s/%dx%d", name, n, n), func(b *testing.B) {
				for b.Loop() {
					_ = PredictIntra8(pix[off:off+(n-1)*160+n], 160, n, n, IntraMode(mode), pix[off-160:], left[:n], 9, scratch)
				}
			})
		}
	}
}

func TestPredictIntra8Allocs(t *testing.T) {
	pix := make([]byte, 160*72)
	left := make([]byte, 64)
	scratch := make([]uint16, 128)
	off := 2*160 + 3
	allocs := testing.AllocsPerRun(50, func() {
		for _, mode := range []IntraMode{IntraModeDC, IntraModeVertical, IntraModeHorizontal, IntraModePaeth, IntraModeSmooth, IntraModeSmoothVertical, IntraModeSmoothHorizontal} {
			_ = PredictIntra8(pix[off:off+15*160+16], 160, 16, 16, mode, pix[off-160:], left[:16], 0, scratch)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs = %v", allocs)
	}
}

// blockSizes are the AV1 intra block dimensions the predictors run on.
var dispatchBlockDims = []int{4, 8, 16, 32, 64}

func makeDispatchBlock(width, height, bytesPerSample int) planeBlock {
	rowBytes := width * bytesPerSample
	stride := rowBytes + 7*bytesPerSample // non-tight stride to catch stride bugs
	return planeBlock{
		pix:      make([]byte, stride*height),
		stride:   stride,
		width:    width,
		height:   height,
		rowBytes: rowBytes,
	}
}

func cloneBlock(b planeBlock) planeBlock {
	c := b
	c.pix = make([]byte, len(b.pix))
	copy(c.pix, b.pix)
	return c
}

func samplesFor(n int, max uint16, seed uint32) ([]uint16, uint32) {
	s := make([]uint16, n)
	for i := range s {
		seed = seed*1664525 + 1013904223
		s[i] = uint16((seed >> 8) % (uint32(max) + 1))
	}
	return s, seed
}

func diffBlocks(t *testing.T, name string, depth, w, h int, got, want planeBlock) {
	t.Helper()
	for row := 0; row < h; row++ {
		for col := 0; col < w*1; col++ {
			gi := row*got.stride + col
			wi := row*want.stride + col
			if got.pix[gi] != want.pix[wi] {
				t.Fatalf("%s depth=%d %dx%d row=%d col=%d got=%d want=%d",
					name, depth, w, h, row, col, got.pix[gi], want.pix[wi])
			}
		}
	}
}

// TestIntraStaticDispatchMatchesPureGo is the bit-exactness guard for the
// dispatched intra static predictors. It exercises the resolved dispatch slots
// (which route through NEON asm on arm64) against the canonical pure-Go
// reference across PAETH and the three SMOOTH variants, every AV1 block size,
// and 8/10/12-bit depths. Every output sample must match the reference exactly.
func TestIntraStaticDispatchMatchesPureGo(t *testing.T) {
	for _, depth := range []struct {
		bps int
		max uint16
	}{{1, 0xff}, {2, 0x3ff}, {2, 0xfff}} {
		var seed uint32 = 0x12345 + uint32(depth.max)
		for _, w := range dispatchBlockDims {
			for _, h := range dispatchBlockDims {
				above, s1 := samplesFor(w, depth.max, seed)
				left, s2 := samplesFor(h, depth.max, s1)
				seed = s2
				aboveLeft := above[0] ^ left[0]
				if aboveLeft > depth.max {
					aboveLeft = depth.max
				}
				weightsW, _ := smoothWeightsForSize(w)
				weightsH, _ := smoothWeightsForSize(h)
				belowPred := left[h-1]
				rightPred := above[w-1]

				base := makeDispatchBlock(w, h, depth.bps)

				// PAETH
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictPaethImpl(got, depth.bps, above, left, aboveLeft)
					predictPaethPureGo(want, depth.bps, above, left, aboveLeft)
					diffBlocks(t, "paeth", int(depth.max), w*depth.bps, h, got, want)
				}
				// SMOOTH
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictSmoothImpl(got, depth.bps, weightsW, weightsH, above, left, belowPred, rightPred)
					predictSmoothPureGo(want, depth.bps, weightsW, weightsH, above, left, belowPred, rightPred)
					diffBlocks(t, "smooth", int(depth.max), w*depth.bps, h, got, want)
				}
				// SMOOTH_V
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictSmoothVerticalImpl(got, depth.bps, weightsH, above, belowPred)
					predictSmoothVerticalPureGo(want, depth.bps, weightsH, above, belowPred)
					diffBlocks(t, "smooth_v", int(depth.max), w*depth.bps, h, got, want)
				}
				// SMOOTH_H
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictSmoothHorizontalImpl(got, depth.bps, weightsW, left, rightPred)
					predictSmoothHorizontalPureGo(want, depth.bps, weightsW, left, rightPred)
					diffBlocks(t, "smooth_h", int(depth.max), w*depth.bps, h, got, want)
				}
			}
		}
	}
}

// TestIntraStaticDispatchEdgeValues stresses the PAETH tie-break paths and the
// SMOOTH rounding boundary with degenerate edge samples (all zero, all max,
// alternating) which exercise the saturating / select corners of the asm.
func TestIntraStaticDispatchEdgeValues(t *testing.T) {
	const w, h = 16, 16
	patterns := []func(i int) uint16{
		func(i int) uint16 { return 0 },
		func(i int) uint16 { return 255 },
		func(i int) uint16 {
			if i%2 == 0 {
				return 0
			}
			return 255
		},
		func(i int) uint16 { return uint16(i * 17 % 256) },
	}
	for ai, ap := range patterns {
		for li, lp := range patterns {
			above := make([]uint16, w)
			left := make([]uint16, h)
			for i := range above {
				above[i] = ap(i)
			}
			for i := range left {
				left[i] = lp(i)
			}
			for _, al := range []uint16{0, 128, 255} {
				weightsW, _ := smoothWeightsForSize(w)
				weightsH, _ := smoothWeightsForSize(h)
				base := makeDispatchBlock(w, h, 1)

				got := cloneBlock(base)
				want := cloneBlock(base)
				predictPaethImpl(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-edge", 0xff, w, h, got, want)

				got = cloneBlock(base)
				want = cloneBlock(base)
				predictSmoothImpl(got, 1, weightsW, weightsH, above, left, left[h-1], above[w-1])
				predictSmoothPureGo(want, 1, weightsW, weightsH, above, left, left[h-1], above[w-1])
				diffBlocks(t, "smooth-edge", 0xff, w, h, got, want)
				_ = ai
				_ = li
			}
		}
	}
}

// TestIntraStaticDispatchZeroAlloc protects the hot-path contract that the
// dispatched predictors do not allocate per call.
func TestIntraStaticDispatchZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	above, _ := samplesFor(w, 0xff, 7)
	left, _ := samplesFor(h, 0xff, 99)
	weightsW, _ := smoothWeightsForSize(w)
	weightsH, _ := smoothWeightsForSize(h)
	block := makeDispatchBlock(w, h, 1)

	checks := []struct {
		name string
		fn   func()
	}{
		{"paeth", func() { predictPaethImpl(block, 1, above, left, 123) }},
		{"smooth", func() { predictSmoothImpl(block, 1, weightsW, weightsH, above, left, left[h-1], above[w-1]) }},
		{"smooth_v", func() { predictSmoothVerticalImpl(block, 1, weightsH, above, left[h-1]) }},
		{"smooth_h", func() { predictSmoothHorizontalImpl(block, 1, weightsW, left, above[w-1]) }},
	}
	for _, c := range checks {
		if allocs := testing.AllocsPerRun(1000, c.fn); allocs != 0 {
			t.Fatalf("%s allocated %f times per call", c.name, allocs)
		}
	}
}

const (
	libaomIntraEdgeDeterministicSeed = 0xbaba
	libaomIntraEdgeIterations        = 128
)

func TestFilterIntraEdgeMatchesLibaomKnownVectors(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		strength uint8
		edge     []uint16
		want     []uint16
	}{
		{
			name:     "strength-zero",
			bitDepth: 8,
			strength: 0,
			edge:     []uint16{2, 5, 9, 17, 33},
			want:     []uint16{2, 5, 9, 17, 33},
		},
		{
			name:     "eight-bit-strength-one",
			bitDepth: 8,
			strength: 1,
			edge:     []uint16{10, 20, 40, 80, 160},
			want:     []uint16{10, 23, 45, 90, 140},
		},
		{
			name:     "ten-bit-strength-two",
			bitDepth: 10,
			strength: 2,
			edge:     []uint16{0, 1023, 512, 17, 999, 33},
			want:     []uint16{0, 544, 517, 479, 390, 335},
		},
		{
			name:     "twelve-bit-strength-three",
			bitDepth: 12,
			strength: 3,
			edge:     []uint16{0, 4095, 3000, 7, 2048},
			want:     []uint16{0, 1775, 2032, 2032, 1657},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.Clone(tt.edge)
			scratch := make([]uint16, len(got))
			if err := FilterIntraEdgeBounded(got, scratch, tt.strength, tt.bitDepth, false); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("edge=%v want %v", got, tt.want)
			}
		})
	}
}

func TestFilterIntraEdgeMatchesLibaomOperationCorpus(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		hi := 1 << bitDepth
		rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
		for iter := range libaomIntraEdgeIterations {
			strength := uint8(rnd.pseudoUniform(4))
			size := 4*(rnd.pseudoUniform(128/4)+1) + 1
			got := make([]uint16, size)
			for i := range got {
				if bitDepth == 8 {
					got[i] = uint16(rnd.rand8())
				} else {
					got[i] = uint16(rnd.pseudoUniform(hi))
				}
			}
			want := slices.Clone(got)
			filterIntraEdgeLibaomReference(want, strength)

			scratch := make([]uint16, size)
			if err := FilterIntraEdgeBounded(got, scratch, strength, bitDepth, false); err != nil {
				t.Fatalf("bitDepth=%d iter=%d strength=%d size=%d: %v", bitDepth, iter, strength, size, err)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("bitDepth=%d iter=%d strength=%d size=%d got=%v want=%v", bitDepth, iter, strength, size, got, want)
			}
		}
	}
}

func TestFilterIntraEdgeRejectsInvalidInputs(t *testing.T) {
	scratch := make([]uint16, intraEdgeMaxSize)
	tests := []struct {
		name     string
		edge     []uint16
		scratch  []uint16
		strength uint8
		bitDepth uint8
	}{
		{name: "empty", edge: nil, scratch: scratch, strength: 1, bitDepth: 8},
		{name: "too-large", edge: make([]uint16, intraEdgeMaxSize+1), scratch: make([]uint16, intraEdgeMaxSize+1), strength: 1, bitDepth: 8},
		{name: "invalid-strength", edge: []uint16{1, 2, 3}, scratch: scratch, strength: 4, bitDepth: 8},
		{name: "invalid-bitdepth", edge: []uint16{1, 2, 3}, scratch: scratch, strength: 1, bitDepth: 9},
		{name: "sample-out-of-range", edge: []uint16{1, 256, 3}, scratch: scratch, strength: 1, bitDepth: 8},
		{name: "short-scratch", edge: []uint16{1, 2, 3}, scratch: []uint16{0, 0}, strength: 1, bitDepth: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := FilterIntraEdgeBounded(tt.edge, tt.scratch, tt.strength, tt.bitDepth, false); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestFilterIntraEdgeStrengthZeroDoesNotRequireScratch(t *testing.T) {
	edge := []uint16{1, 2, 3, 4}
	if err := FilterIntraEdgeBounded(edge, nil, 0, 8, false); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(edge, []uint16{1, 2, 3, 4}) {
		t.Fatalf("edge changed: %v", edge)
	}
}

func TestFilterIntraEdgeCornerMatchesLibaomKnownVector(t *testing.T) {
	above := []uint16{0, 25, 40, 55}
	left := []uint16{0, 25, 100, 120}
	if err := FilterIntraEdgeCorner(above, 2, left, 2, 8); err != nil {
		t.Fatal(err)
	}
	if above[1] != 53 || left[1] != 53 {
		t.Fatalf("corner above=%v left=%v want shared 53", above, left)
	}
	if above[2] != 40 || left[2] != 100 {
		t.Fatalf("edge endpoints changed above=%v left=%v", above, left)
	}
}

func TestFilterIntraEdgeAllocs(t *testing.T) {
	edge := make([]uint16, intraEdgeMaxSize)
	scratch := make([]uint16, intraEdgeMaxSize)
	for i := range edge {
		edge[i] = uint16((i * 17) & 0xfff)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := FilterIntraEdgeBounded(edge, scratch, 3, 12, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FilterIntraEdge allocated: %f", allocs)
	}
}

func FuzzFilterIntraEdge(f *testing.F) {
	f.Add([]byte{0, 16, 32, 64, 128, 255}, uint8(1), uint8(0))
	f.Add([]byte{255, 0, 127, 8, 240}, uint8(2), uint8(1))
	f.Add([]byte{1, 2, 3, 4}, uint8(3), uint8(2))

	f.Fuzz(func(t *testing.T, data []byte, rawStrength uint8, rawBitDepth uint8) {
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		size := len(data)%intraEdgeMaxSize + 1
		edge := make([]uint16, size)
		for i := range edge {
			if len(data) == 0 {
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			edge[i] = (lo | hi<<8) & max
		}
		before := slices.Clone(edge)
		strength := rawStrength % 4
		scratch := make([]uint16, size)
		if err := FilterIntraEdgeBounded(edge, scratch, strength, bitDepth, false); err != nil {
			t.Fatalf("FilterIntraEdge err=%v", err)
		}
		for i, sample := range edge {
			if sample > max {
				t.Fatalf("edge[%d]=%d exceeds max %d", i, sample, max)
			}
		}
		if strength == 0 && !slices.Equal(edge, before) {
			t.Fatalf("strength-zero edge=%v want %v", edge, before)
		}
	})
}

func BenchmarkFilterIntraEdge(b *testing.B) {
	edge := make([]uint16, intraEdgeMaxSize)
	scratch := make([]uint16, intraEdgeMaxSize)
	for i := range edge {
		edge[i] = uint16((i * 31) & 0xfff)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = FilterIntraEdgeBounded(edge, scratch, 3, 12, false)
	}
}

func TestUpsampleIntraEdgeMatchesLibaomKnownVectors(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		edge     []uint16
		origin   int
		size     int
		want     []uint16
	}{
		{
			name:     "eight-bit",
			bitDepth: 8,
			edge:     []uint16{99, 99, 0, 10, 20, 40, 80, 160, 0, 0, 0},
			origin:   4,
			size:     4,
			want:     []uint16{10, 14, 20, 28, 40, 56, 80, 123, 160},
		},
		{
			name:     "twelve-bit-clipped",
			bitDepth: 12,
			edge:     []uint16{77, 77, 0, 0, 4095, 4095, 0, 0, 0, 0, 0},
			origin:   4,
			size:     4,
			want:     []uint16{0, 2048, 4095, 4095, 4095, 2048, 0, 0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.Clone(tt.edge)
			scratch := make([]uint16, tt.size+3)
			if err := UpsampleIntraEdgeBounded(got, tt.origin, tt.size, scratch, tt.bitDepth, false); err != nil {
				t.Fatal(err)
			}
			window := got[tt.origin-2 : tt.origin+2*tt.size-1]
			if !slices.Equal(window, tt.want) {
				t.Fatalf("edge=%v want %v", window, tt.want)
			}
		})
	}
}

func TestUpsampleIntraEdgeMatchesLibaomOperationCorpus(t *testing.T) {
	const (
		offset  = 16
		bufSize = 2*64 + 32
	)
	for _, bitDepth := range []uint8{8, 10, 12} {
		hi := 1 << bitDepth
		rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
		for iter := range libaomIntraEdgeIterations {
			size := 4 * (rnd.pseudoUniform(4) + 1)
			got := make([]uint16, bufSize)
			pix := uint16(0)
			for i := 0; i < offset+size; i++ {
				if bitDepth == 8 {
					pix = uint16(rnd.rand8())
				} else {
					pix = uint16(rnd.pseudoUniform(hi))
				}
				got[i] = pix
			}
			for i := offset + size; i < len(got); i++ {
				got[i] = pix
			}
			want := slices.Clone(got)
			upsampleIntraEdgeLibaomReference(want, offset, size, bitDepth)

			scratch := make([]uint16, size+3)
			if err := UpsampleIntraEdgeBounded(got, offset, size, scratch, bitDepth, false); err != nil {
				t.Fatalf("bitDepth=%d iter=%d size=%d: %v", bitDepth, iter, size, err)
			}
			end := offset + (size-1)*2 + 1
			if !slices.Equal(got[offset-2:end], want[offset-2:end]) {
				t.Fatalf("bitDepth=%d iter=%d size=%d got=%v want=%v", bitDepth, iter, size, got[offset-2:end], want[offset-2:end])
			}
		}
	}
}

func TestUpsampleIntraEdgeRejectsInvalidInputs(t *testing.T) {
	edge := []uint16{0, 0, 1, 2, 3, 4, 0, 0, 0}
	scratch := make([]uint16, intraEdgeMaxUpsampleSize+3)
	tests := []struct {
		name     string
		edge     []uint16
		origin   int
		size     int
		scratch  []uint16
		bitDepth uint8
	}{
		{name: "zero-size", edge: edge, origin: 2, size: 0, scratch: scratch, bitDepth: 8},
		{name: "too-large", edge: make([]uint16, 64), origin: 2, size: intraEdgeMaxUpsampleSize + 1, scratch: make([]uint16, intraEdgeMaxUpsampleSize+4), bitDepth: 8},
		{name: "origin-before-negative-window", edge: edge, origin: 1, size: 4, scratch: scratch, bitDepth: 8},
		{name: "source-out-of-range", edge: []uint16{0, 0, 1}, origin: 2, size: 2, scratch: scratch, bitDepth: 8},
		{name: "write-out-of-range", edge: edge[:7], origin: 2, size: 4, scratch: scratch, bitDepth: 8},
		{name: "short-scratch", edge: edge, origin: 2, size: 4, scratch: []uint16{0, 0, 0, 0, 0, 0}, bitDepth: 8},
		{name: "invalid-bitdepth", edge: edge, origin: 2, size: 4, scratch: scratch, bitDepth: 9},
		{name: "sample-out-of-range", edge: []uint16{0, 0, 1, 256, 3, 4, 0, 0, 0}, origin: 2, size: 4, scratch: scratch, bitDepth: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := UpsampleIntraEdgeBounded(tt.edge, tt.origin, tt.size, tt.scratch, tt.bitDepth, false); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestUpsampleIntraEdgeAllocs(t *testing.T) {
	edge := make([]uint16, 64)
	const origin = 16
	const size = 16
	for i := origin - 1; i < origin+size; i++ {
		edge[i] = uint16((i * 29) & 0x3ff)
	}
	for i := origin + size; i < len(edge); i++ {
		edge[i] = edge[origin+size-1]
	}
	scratch := make([]uint16, size+3)

	allocs := testing.AllocsPerRun(1000, func() {
		if err := UpsampleIntraEdgeBounded(edge, origin, size, scratch, 10, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("UpsampleIntraEdge allocated: %f", allocs)
	}
}

func FuzzUpsampleIntraEdge(f *testing.F) {
	f.Add([]byte{0, 10, 20, 40, 80, 160, 160}, uint8(0), uint8(0))
	f.Add([]byte{0xff, 0, 0xff, 0, 0x80, 0x40}, uint8(1), uint8(2))

	f.Fuzz(func(t *testing.T, data []byte, rawSize uint8, rawBitDepth uint8) {
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		size := 4 * (int(rawSize)%4 + 1)
		origin := 16
		edge := make([]uint16, origin+2*size-1)
		for i := origin - 1; i < origin+size; i++ {
			if len(data) == 0 {
				continue
			}
			lo := uint16(data[i%len(data)])
			hi := uint16(data[(i+1)%len(data)])
			edge[i] = (lo | hi<<8) & max
		}
		for i := origin + size; i < len(edge); i++ {
			edge[i] = edge[origin+size-1]
		}
		scratch := make([]uint16, size+3)
		if err := UpsampleIntraEdgeBounded(edge, origin, size, scratch, bitDepth, false); err != nil {
			t.Fatalf("UpsampleIntraEdge err=%v", err)
		}
		for i, sample := range edge[origin-2:] {
			if sample > max {
				t.Fatalf("edge[%d]=%d exceeds max %d", origin-2+i, sample, max)
			}
		}
	})
}

func BenchmarkUpsampleIntraEdge(b *testing.B) {
	edge := make([]uint16, 64)
	const origin = 16
	const size = 16
	for i := origin - 1; i < origin+size; i++ {
		edge[i] = uint16((i * 31) & 0xfff)
	}
	for i := origin + size; i < len(edge); i++ {
		edge[i] = edge[origin+size-1]
	}
	scratch := make([]uint16, size+3)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = UpsampleIntraEdgeBounded(edge, origin, size, scratch, 12, false)
	}
}

func TestIntraEdgeFilterStrengthMatchesLibaomThresholds(t *testing.T) {
	tests := []struct {
		name           string
		block0         int
		block1         int
		delta          int
		smoothNeighbor bool
		want           uint8
	}{
		{name: "type0-small-below", block0: 4, block1: 4, delta: 55, want: 0},
		{name: "type0-small-at", block0: 4, block1: 4, delta: 56, want: 1},
		{name: "type0-twelve-at", block0: 4, block1: 8, delta: 40, want: 1},
		{name: "type0-sixteen-below", block0: 8, block1: 8, delta: 39, want: 0},
		{name: "type0-twenty-four-one", block0: 8, block1: 16, delta: 8, want: 1},
		{name: "type0-twenty-four-two", block0: 8, block1: 16, delta: 16, want: 2},
		{name: "type0-twenty-four-three", block0: 8, block1: 16, delta: 32, want: 3},
		{name: "type0-thirty-two-one", block0: 16, block1: 16, delta: 1, want: 1},
		{name: "type0-thirty-two-two", block0: 16, block1: 16, delta: 4, want: 2},
		{name: "type0-thirty-two-three", block0: 16, block1: 16, delta: 32, want: 3},
		{name: "type0-large", block0: 32, block1: 16, delta: 1, want: 3},
		{name: "type1-small-below", block0: 4, block1: 4, delta: 39, smoothNeighbor: true, want: 0},
		{name: "type1-small-one", block0: 4, block1: 4, delta: 40, smoothNeighbor: true, want: 1},
		{name: "type1-small-two", block0: 4, block1: 4, delta: 64, smoothNeighbor: true, want: 2},
		{name: "type1-sixteen-one", block0: 8, block1: 8, delta: 20, smoothNeighbor: true, want: 1},
		{name: "type1-sixteen-two", block0: 8, block1: 8, delta: 48, smoothNeighbor: true, want: 2},
		{name: "type1-twenty-four-below", block0: 8, block1: 16, delta: 3, smoothNeighbor: true, want: 0},
		{name: "type1-twenty-four-three", block0: 8, block1: 16, delta: 4, smoothNeighbor: true, want: 3},
		{name: "type1-large", block0: 32, block1: 16, delta: -1, smoothNeighbor: true, want: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IntraEdgeFilterStrength(tt.block0, tt.block1, tt.delta, tt.smoothNeighbor)
			if got != tt.want {
				t.Fatalf("strength=%d want %d", got, tt.want)
			}
		})
	}
}

func TestUseIntraEdgeUpsampleMatchesLibaomThresholds(t *testing.T) {
	tests := []struct {
		name           string
		block0         int
		block1         int
		delta          int
		smoothNeighbor bool
		want           bool
	}{
		{name: "zero-delta", block0: 8, block1: 8, delta: 0, want: false},
		{name: "large-delta", block0: 8, block1: 8, delta: 40, want: false},
		{name: "type0-sixteen", block0: 8, block1: 8, delta: 39, want: true},
		{name: "type0-too-wide", block0: 16, block1: 8, delta: 39, want: false},
		{name: "type1-eight", block0: 4, block1: 4, delta: -39, smoothNeighbor: true, want: true},
		{name: "type1-too-wide", block0: 8, block1: 8, delta: 39, smoothNeighbor: true, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := UseIntraEdgeUpsample(tt.block0, tt.block1, tt.delta, tt.smoothNeighbor)
			if got != tt.want {
				t.Fatalf("upsample=%t want %t", got, tt.want)
			}
		})
	}
}

func TestIntraEdgeDecisionsMatchLibaomReferenceCorpus(t *testing.T) {
	for _, block0 := range []int{4, 8, 12, 16, 24, 32, 64} {
		for _, block1 := range []int{4, 8, 12, 16, 24, 32, 64} {
			for delta := -90; delta <= 90; delta++ {
				for _, smoothNeighbor := range []bool{false, true} {
					gotStrength := IntraEdgeFilterStrength(block0, block1, delta, smoothNeighbor)
					wantStrength := intraEdgeFilterStrengthLibaomReference(block0, block1, delta, smoothNeighbor)
					if gotStrength != wantStrength {
						t.Fatalf("strength block0=%d block1=%d delta=%d smooth=%t got=%d want=%d", block0, block1, delta, smoothNeighbor, gotStrength, wantStrength)
					}
					gotUpsample := UseIntraEdgeUpsample(block0, block1, delta, smoothNeighbor)
					wantUpsample := useIntraEdgeUpsampleLibaomReference(block0, block1, delta, smoothNeighbor)
					if gotUpsample != wantUpsample {
						t.Fatalf("upsample block0=%d block1=%d delta=%d smooth=%t got=%t want=%t", block0, block1, delta, smoothNeighbor, gotUpsample, wantUpsample)
					}
				}
			}
		}
	}
}

func TestIntraEdgeDecisionsAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		_ = IntraEdgeFilterStrength(16, 16, -32, false)
		_ = IntraEdgeFilterStrength(8, 8, 48, true)
		_ = UseIntraEdgeUpsample(8, 8, 39, false)
		_ = UseIntraEdgeUpsample(4, 4, -39, true)
	})
	if allocs != 0 {
		t.Fatalf("intra edge decisions allocated: %f", allocs)
	}
}

func FuzzIntraEdgeDecisions(f *testing.F) {
	f.Add(uint8(4), uint8(4), int16(56), false)
	f.Add(uint8(8), uint8(16), int16(-32), false)
	f.Add(uint8(4), uint8(4), int16(39), true)

	f.Fuzz(func(t *testing.T, rawBlock0 uint8, rawBlock1 uint8, rawDelta int16, smoothNeighbor bool) {
		block0 := int(rawBlock0%64) + 1
		block1 := int(rawBlock1%64) + 1
		delta := int(rawDelta)
		strength := IntraEdgeFilterStrength(block0, block1, delta, smoothNeighbor)
		wantStrength := intraEdgeFilterStrengthLibaomReference(block0, block1, delta, smoothNeighbor)
		if strength != wantStrength {
			t.Fatalf("strength=%d want %d", strength, wantStrength)
		}
		upsample := UseIntraEdgeUpsample(block0, block1, delta, smoothNeighbor)
		wantUpsample := useIntraEdgeUpsampleLibaomReference(block0, block1, delta, smoothNeighbor)
		if upsample != wantUpsample {
			t.Fatalf("upsample=%t want %t", upsample, wantUpsample)
		}
	})
}

func filterIntraEdgeLibaomReference(edge []uint16, strength uint8) {
	if strength == 0 {
		return
	}
	scratch := slices.Clone(edge)
	kernel := intraEdgeFilterKernels[strength-1]
	for i := 1; i < len(edge); i++ {
		sum := 0
		for j := range intraEdgeTaps {
			k := i - 2 + j
			if k < 0 {
				k = 0
			} else if k > len(edge)-1 {
				k = len(edge) - 1
			}
			sum += int(scratch[k]) * kernel[j]
		}
		edge[i] = uint16((sum + 8) >> 4)
	}
}

func upsampleIntraEdgeLibaomReference(edge []uint16, origin int, size int, bitDepth uint8) {
	max := int((1 << bitDepth) - 1)
	scratch := make([]uint16, size+3)
	scratch[0] = edge[origin-1]
	scratch[1] = edge[origin-1]
	copy(scratch[2:2+size], edge[origin:origin+size])
	scratch[size+2] = edge[origin+size-1]

	edge[origin-2] = scratch[0]
	for i := range size {
		sum := -int(scratch[i]) + 9*int(scratch[i+1]) + 9*int(scratch[i+2]) - int(scratch[i+3])
		sample := (sum + 8) >> 4
		if sample < 0 {
			sample = 0
		} else if sample > max {
			sample = max
		}
		edge[origin+2*i-1] = uint16(sample)
		edge[origin+2*i] = scratch[i+2]
	}
}

func intraEdgeFilterStrengthLibaomReference(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) uint8 {
	d := absInt(delta)
	blockWH := blockSize0 + blockSize1
	strength := uint8(0)
	if !smoothNeighbor {
		if blockWH <= 8 {
			if d >= 56 {
				strength = 1
			}
		} else if blockWH <= 12 {
			if d >= 40 {
				strength = 1
			}
		} else if blockWH <= 16 {
			if d >= 40 {
				strength = 1
			}
		} else if blockWH <= 24 {
			if d >= 8 {
				strength = 1
			}
			if d >= 16 {
				strength = 2
			}
			if d >= 32 {
				strength = 3
			}
		} else if blockWH <= 32 {
			if d >= 1 {
				strength = 1
			}
			if d >= 4 {
				strength = 2
			}
			if d >= 32 {
				strength = 3
			}
		} else if d >= 1 {
			strength = 3
		}
		return strength
	}

	if blockWH <= 8 {
		if d >= 40 {
			strength = 1
		}
		if d >= 64 {
			strength = 2
		}
	} else if blockWH <= 16 {
		if d >= 20 {
			strength = 1
		}
		if d >= 48 {
			strength = 2
		}
	} else if blockWH <= 24 {
		if d >= 4 {
			strength = 3
		}
	} else if d >= 1 {
		strength = 3
	}
	return strength
}

func useIntraEdgeUpsampleLibaomReference(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) bool {
	d := absInt(delta)
	blockWH := blockSize0 + blockSize1
	if d == 0 || d >= 40 {
		return false
	}
	if smoothNeighbor {
		return blockWH <= 8
	}
	return blockWH <= 16
}

type libaomIntraEdgeRandom struct {
	state uint32
}

func newLibaomIntraEdgeRandom(seed uint32) *libaomIntraEdgeRandom {
	return &libaomIntraEdgeRandom{state: seed}
}

func (r *libaomIntraEdgeRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *libaomIntraEdgeRandom) rand8() uint8 {
	return uint8((r.generate(1<<31) >> 23) & 0xff)
}

func (r *libaomIntraEdgeRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}

// These tests guard the dispatched DC-sum, CfL and directional-interpolation
// kernels: the resolved dispatch slots (NEON on arm64, AVX2 on amd64 when
// active) must match the canonical pure-Go references sample for sample. They
// run on every build; on architectures with no tuned variant the slot is the
// reference itself and the comparison is trivially satisfied.

func TestSumSamplesDispatchMatchesPureGo(t *testing.T) {
	for _, n := range []int{4, 8, 12, 16, 20, 32, 64, 100} {
		for _, max := range []uint16{0xff, 0x3ff, 0xfff} {
			s, _ := samplesFor(n, max, 0x9e3d+uint32(n)+uint32(max))
			got := sumSamplesImpl(s)
			want := sumSamplesPureGo(s)
			if got != want {
				t.Fatalf("sumSamples n=%d max=%d got=%d want=%d", n, max, got, want)
			}
		}
	}
}

func TestSubsampleLuma8DispatchMatchesPureGo(t *testing.T) {
	type cfg struct {
		w, h       int
		subX, subY bool
	}
	cfgs := []cfg{
		{8, 8, false, false}, {16, 16, false, false}, {32, 32, false, false},
		{16, 16, true, false}, {32, 16, true, false}, {16, 8, true, false},
		{16, 16, true, true}, {32, 32, true, true}, {16, 32, true, true},
		{8, 8, true, true}, {4, 4, true, true}, // width 4 -> pure-Go fallback
	}
	var seed uint32 = 0x1234
	for _, c := range cfgs {
		stride := c.w + 5
		input := make([]uint8, stride*c.h)
		for i := range input {
			seed = seed*1664525 + 1013904223
			input[i] = uint8(seed >> 13)
		}
		outW, outH := c.w, c.h
		if c.subX {
			outW >>= 1
		}
		if c.subY {
			outH >>= 1
		}
		got := make([]uint16, CFLBufSquare)
		want := make([]uint16, CFLBufSquare)
		subsampleLuma8Impl(got, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		subsampleLuma8PureGo(want, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("subsample %+v idx=%d got=%d want=%d", c, i, got[i], want[i])
			}
		}
	}
}

func TestSubtractCFLAverageDispatchMatchesPureGo(t *testing.T) {
	var seed uint32 = 0x871
	for _, dim := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {8, 32}, {32, 8}} {
		w, h := dim[0], dim[1]
		src := make([]uint16, CFLBufSquare)
		for row := 0; row < h; row++ {
			for col := 0; col < w; col++ {
				seed = seed*1664525 + 1013904223
				src[row*CFLBufLine+col] = uint16(seed >> 17)
			}
		}
		log2, _ := log2PowerOfTwoInt(w * h)
		got := make([]int16, CFLBufSquare)
		want := make([]int16, CFLBufSquare)
		subtractCFLAverageImpl(src, got, w, h, log2)
		subtractCFLAveragePureGo(src, want, w, h, log2)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("subtract-average %dx%d idx=%d got=%d want=%d", w, h, i, got[i], want[i])
			}
		}
	}
}

func TestApplyCFLDispatchMatchesPureGo(t *testing.T) {
	dims := [][2]int{{8, 8}, {16, 16}, {32, 32}, {8, 16}, {16, 8}, {4, 4}, {32, 8}}
	depths := []struct {
		bps int
		max uint16
	}{{1, 0xff}, {2, 0x3ff}, {2, 0xfff}}
	var seed uint32 = 0xabc
	for _, d := range depths {
		for _, dim := range dims {
			w, h := dim[0], dim[1]
			ac := make([]int16, CFLBufSquare)
			for i := range ac {
				seed = seed*1664525 + 1013904223
				ac[i] = int16(int32(seed>>8)%4096 - 2048)
			}
			for _, alpha := range []int{-16, -7, -1, 0, 1, 5, 16} {
				base := makeDispatchBlock(w, h, d.bps)
				// seed the chroma DC with mid-range values
				for i := range base.pix {
					base.pix[i] = byte((i*7 + 31) & 0xff)
				}
				got := cloneBlock(base)
				want := cloneBlock(base)
				applyCFLImpl(got, d.bps, w, h, ac, alpha, d.max)
				applyCFLPureGo(want, d.bps, w, h, ac, alpha, d.max)
				diffBlocks(t, "applycfl", int(d.max), w*d.bps, h, got, want)
			}
		}
	}
}

func TestDirRowInterp8DispatchMatchesPureGo(t *testing.T) {
	var seed uint32 = 0x55aa
	for _, width := range []int{4, 8, 16, 32, 64} {
		for _, maxBase := range []int{width + 1, width + 8, 200, 8} {
			above := make([]uint16, 300)
			for i := range above {
				seed = seed*1664525 + 1013904223
				above[i] = uint16(seed >> 24)
			}
			for _, base := range []int{0, 1, 3, 7, maxBase - 1, maxBase, maxBase + 4} {
				if base < 0 {
					continue
				}
				for _, shift := range []int{0, 1, 5, 16, 31} {
					got := make([]byte, width)
					want := make([]byte, width)
					dirRowInterp8Impl(got, above, base, shift, maxBase, width)
					dirRowInterp8PureGo(want, above, base, shift, maxBase, width)
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("dirrow w=%d base=%d shift=%d maxBase=%d col=%d got=%d want=%d",
								width, base, shift, maxBase, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

var libaomSmoothWeights = [...]uint16{
	255, 149, 85, 64,
	255, 197, 146, 105, 73, 50, 37, 32,
	255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
	255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
	66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
	255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156,
	150, 144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
	65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15,
	13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
}

func TestSmoothWeightsMatchLibaom(t *testing.T) {
	if !slices.Equal(smoothWeights[:], libaomSmoothWeights[:]) {
		t.Fatalf("smoothWeights=%v want %v", smoothWeights, libaomSmoothWeights)
	}
}

func TestStaticIntraPredictorsMatchLibaomKnownVectors(t *testing.T) {
	tests := []struct {
		name   string
		mode   IntraMode
		width  int
		height int
		edges  IntraEdges
		want   []uint16
	}{
		{
			name:   "paeth",
			mode:   IntraModePaeth,
			width:  4,
			height: 4,
			edges: IntraEdges{
				Above:              []uint16{10, 40, 100, 180},
				Left:               []uint16{20, 60, 120, 200},
				AboveLeft:          30,
				AboveAvailable:     true,
				LeftAvailable:      true,
				AboveLeftAvailable: true,
			},
			want: []uint16{
				10, 30, 100, 180,
				30, 60, 100, 180,
				120, 120, 120, 180,
				200, 200, 200, 200,
			},
		},
		{
			name:   "smooth-vertical",
			mode:   IntraModeSmoothVertical,
			width:  4,
			height: 4,
			edges: IntraEdges{
				Above:          []uint16{0, 64, 128, 255},
				Left:           []uint16{10, 20, 30, 40},
				AboveAvailable: true,
				LeftAvailable:  true,
			},
			want: []uint16{
				0, 64, 128, 254,
				17, 54, 91, 165,
				27, 48, 69, 111,
				30, 46, 62, 94,
			},
		},
		{
			name:   "smooth-horizontal",
			mode:   IntraModeSmoothHorizontal,
			width:  4,
			height: 4,
			edges: IntraEdges{
				Above:          []uint16{15, 30, 45, 80},
				Left:           []uint16{0, 64, 128, 255},
				AboveAvailable: true,
				LeftAvailable:  true,
			},
			want: []uint16{
				0, 33, 53, 60,
				64, 71, 75, 76,
				128, 108, 96, 92,
				254, 182, 138, 124,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plane, _ := testPlane(tt.width, tt.height, 1, tt.width)
			if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, tt.width, tt.height, tt.mode, tt.edges); err != nil {
				t.Fatal(err)
			}
			got := collectPlaneSamples(plane, 1, tt.width, tt.height)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("prediction=%v want %v", got, tt.want)
			}
		})
	}
}

func TestStaticIntraPredictorsMatchLibaomCorpus(t *testing.T) {
	for _, mode := range []IntraMode{IntraModePaeth, IntraModeSmooth, IntraModeSmoothVertical, IntraModeSmoothHorizontal} {
		for _, bitDepth := range []uint8{8, 10, 12} {
			bytesPerSample := 1
			if bitDepth != 8 {
				bytesPerSample = 2
			}
			mask := uint16((1 << bitDepth) - 1)
			rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
			for _, size := range libaomDRTxSizes {
				for iter := range 16 {
					edges := randomStaticIntraEdges(size.width, size.height, mask, rnd)
					if iter == 0 {
						for i := range edges.Above {
							edges.Above[i] = mask
						}
						for i := range edges.Left {
							edges.Left[i] = mask
						}
						edges.AboveLeft = mask
					}
					want := predictStaticIntraLibaomReference(size.width, size.height, mode, edges)
					plane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
					if err := PredictIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, mode, edges); err != nil {
						t.Fatalf("mode=%d bitDepth=%d size=%dx%d iter=%d: %v", mode, bitDepth, size.width, size.height, iter, err)
					}
					got := collectPlaneSamples(plane, bytesPerSample, size.width, size.height)
					if !slices.Equal(got, want) {
						t.Fatalf("mode=%d bitDepth=%d size=%dx%d iter=%d got=%v want=%v", mode, bitDepth, size.width, size.height, iter, got, want)
					}
				}
			}
		}
	}
}

func TestStaticIntraPredictorsRejectInvalidInputs(t *testing.T) {
	plane, _ := testPlane(8, 8, 1, 8)
	valid := IntraEdges{
		Above:              make([]uint16, 8),
		Left:               make([]uint16, 8),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	tests := []struct {
		name   string
		mode   IntraMode
		width  int
		height int
		edges  IntraEdges
	}{
		{name: "paeth-missing-above-left", mode: IntraModePaeth, width: 8, height: 8, edges: IntraEdges{Above: make([]uint16, 8), Left: make([]uint16, 8), AboveAvailable: true, LeftAvailable: true}},
		{name: "paeth-above-left-out-of-range", mode: IntraModePaeth, width: 8, height: 8, edges: IntraEdges{Above: make([]uint16, 8), Left: make([]uint16, 8), AboveLeft: 256, AboveAvailable: true, LeftAvailable: true, AboveLeftAvailable: true}},
		{name: "smooth-missing-above", mode: IntraModeSmooth, width: 8, height: 8, edges: IntraEdges{Left: make([]uint16, 8), LeftAvailable: true}},
		{name: "smooth-invalid-width", mode: IntraModeSmooth, width: 5, height: 8, edges: valid},
		{name: "smooth-sample-out-of-range", mode: IntraModeSmoothHorizontal, width: 8, height: 8, edges: IntraEdges{Above: []uint16{0, 0, 0, 0, 0, 0, 0, 256}, Left: make([]uint16, 8), AboveAvailable: true, LeftAvailable: true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, tt.width, tt.height, tt.mode, tt.edges); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestStaticIntraPredictorsAllocs(t *testing.T) {
	plane, _ := testPlane(32, 32, 2, 32*2)
	edges := randomStaticIntraEdges(32, 32, 0xfff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 32, 32, IntraModePaeth, edges); err != nil {
			t.Fatal(err)
		}
		if err := PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 32, 32, IntraModeSmooth, edges); err != nil {
			t.Fatal(err)
		}
		if err := PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 32, 32, IntraModeSmoothVertical, edges); err != nil {
			t.Fatal(err)
		}
		if err := PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 32, 32, IntraModeSmoothHorizontal, edges); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("static intra predictors allocated: %f", allocs)
	}
}

func FuzzStaticIntraPredictors(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), []byte{0, 64, 128, 255})
	f.Add(uint8(1), uint8(1), uint8(1), uint8(1), []byte{255, 0, 127, 32})
	f.Add(uint8(2), uint8(2), uint8(2), uint8(2), []byte{1, 2, 3, 4, 5, 6})

	f.Fuzz(func(t *testing.T, rawMode uint8, rawSize uint8, rawBitDepth uint8, rawSeed uint8, data []byte) {
		modes := [...]IntraMode{IntraModePaeth, IntraModeSmooth, IntraModeSmoothVertical, IntraModeSmoothHorizontal}
		sizes := [...]int{4, 8, 16, 32, 64}
		mode := modes[rawMode%uint8(len(modes))]
		width := sizes[rawSize%uint8(len(sizes))]
		height := sizes[(rawSize/5)%uint8(len(sizes))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		edges := fuzzStaticIntraEdges(width, height, max, rawSeed, data)
		plane, _ := testPlane(width, height, bytesPerSample, width*bytesPerSample)
		if err := PredictIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, width, height, mode, edges); err != nil {
			t.Fatalf("PredictIntraPlaneBlock err=%v", err)
		}
		for row := range height {
			for col := range width {
				if got := getSample(plane, bytesPerSample, col, row); got > max {
					t.Fatalf("sample(%d,%d)=%d exceeds max %d", col, row, got, max)
				}
			}
		}
	})
}

func BenchmarkStaticIntraPredictors(b *testing.B) {
	plane, _ := testPlane(64, 64, 2, 64*2)
	edges := randomStaticIntraEdges(64, 64, 0xfff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 64, 64, IntraModeSmooth, edges)
	}
}

func randomStaticIntraEdges(width int, height int, mask uint16, rnd *libaomIntraEdgeRandom) IntraEdges {
	edges := IntraEdges{
		Above:              make([]uint16, width),
		Left:               make([]uint16, height),
		AboveLeft:          uint16(rnd.pseudoUniform(int(mask) + 1)),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	for i := range edges.Above {
		edges.Above[i] = uint16(rnd.pseudoUniform(int(mask) + 1))
	}
	for i := range edges.Left {
		edges.Left[i] = uint16(rnd.pseudoUniform(int(mask) + 1))
	}
	return edges
}

func fuzzStaticIntraEdges(width int, height int, max uint16, seed uint8, data []byte) IntraEdges {
	edges := IntraEdges{
		Above:              make([]uint16, width),
		Left:               make([]uint16, height),
		AboveLeft:          fuzzSample(data, int(seed), max),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	for i := range edges.Above {
		edges.Above[i] = fuzzSample(data, int(seed)+i+1, max)
	}
	for i := range edges.Left {
		edges.Left[i] = fuzzSample(data, int(seed)+len(edges.Above)+i+1, max)
	}
	return edges
}

func predictStaticIntraLibaomReference(width int, height int, mode IntraMode, edges IntraEdges) []uint16 {
	out := make([]uint16, width*height)
	weightsW := libaomSmoothWeights[width-4 : width-4+width]
	weightsH := libaomSmoothWeights[height-4 : height-4+height]
	scale := uint16(1 << smoothWeightLog2Scale)
	belowPred := edges.Left[height-1]
	rightPred := edges.Above[width-1]
	for row := range height {
		for col := range width {
			var pred uint16
			switch mode {
			case IntraModePaeth:
				pred = paethPredictorSingleReference(edges.Left[row], edges.Above[col], edges.AboveLeft)
			case IntraModeSmooth:
				sum := uint32(weightsH[row])*uint32(edges.Above[col]) +
					uint32(scale-weightsH[row])*uint32(belowPred) +
					uint32(weightsW[col])*uint32(edges.Left[row]) +
					uint32(scale-weightsW[col])*uint32(rightPred)
				pred = uint16(divideRoundReference(sum, 1+smoothWeightLog2Scale))
			case IntraModeSmoothVertical:
				sum := uint32(weightsH[row])*uint32(edges.Above[col]) +
					uint32(scale-weightsH[row])*uint32(belowPred)
				pred = uint16(divideRoundReference(sum, smoothWeightLog2Scale))
			case IntraModeSmoothHorizontal:
				sum := uint32(weightsW[col])*uint32(edges.Left[row]) +
					uint32(scale-weightsW[col])*uint32(rightPred)
				pred = uint16(divideRoundReference(sum, smoothWeightLog2Scale))
			}
			out[row*width+col] = pred
		}
	}
	return out
}

func paethPredictorSingleReference(left uint16, top uint16, topLeft uint16) uint16 {
	base := int(top) + int(left) - int(topLeft)
	pLeft := absInt(base - int(left))
	pTop := absInt(base - int(top))
	pTopLeft := absInt(base - int(topLeft))
	if pLeft <= pTop && pLeft <= pTopLeft {
		return left
	}
	if pTop <= pTopLeft {
		return top
	}
	return topLeft
}

func divideRoundReference(value uint32, bits int) uint32 {
	return (value + (1 << (bits - 1))) >> bits
}

func TestPredictIntraPlaneBlockDC(t *testing.T) {
	plane, _ := testPlane(6, 5, 1, 8)
	for i := range plane.Pix {
		plane.Pix[i] = 0xee
	}
	edges := IntraEdges{
		Above:          []uint16{10, 20, 30},
		Left:           []uint16{50, 70},
		AboveAvailable: true,
		LeftAvailable:  true,
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 2, 1, 3, 2, IntraModeDC, edges); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			got := getSample(plane, 1, x, y)
			want := uint16(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = 36
			}
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictIntraPlaneBlockDCMissingEdgesUsesMidpoint(t *testing.T) {
	plane, _ := testPlane(3, 2, 2, 8)
	if err := PredictIntraPlaneBlock(plane, 2, 10, 0, 0, 3, 2, IntraModeDC, IntraEdges{}); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			if got := getSample(plane, 2, x, y); got != 512 {
				t.Fatalf("sample(%d,%d)=%d want 512", x, y, got)
			}
		}
	}
}

func TestPredictIntraPlaneBlockDCMatchesLibaomCorpus(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		for _, size := range libaomDRTxSizes {
			rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
			for iter := range 16 {
				baseEdges := randomStaticIntraEdges(size.width, size.height, max, rnd)
				for _, availability := range []struct {
					name  string
					above bool
					left  bool
				}{
					{name: "none"},
					{name: "above", above: true},
					{name: "left", left: true},
					{name: "both", above: true, left: true},
				} {
					edges := baseEdges
					edges.AboveAvailable = availability.above
					edges.LeftAvailable = availability.left
					plane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
					if err := PredictIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, IntraModeDC, edges); err != nil {
						t.Fatalf("bitDepth=%d size=%dx%d iter=%d availability=%s: %v", bitDepth, size.width, size.height, iter, availability.name, err)
					}
					want := dcPredictionLibaomReference(size.width, size.height, max, edges)
					for row := 0; row < size.height; row++ {
						for col := 0; col < size.width; col++ {
							if got := getSample(plane, bytesPerSample, col, row); got != want {
								t.Fatalf("bitDepth=%d size=%dx%d iter=%d availability=%s sample(%d,%d)=%d want %d", bitDepth, size.width, size.height, iter, availability.name, col, row, got, want)
							}
						}
					}
				}
			}
		}
	}
}

func TestPredictIntraPlaneBlockVerticalHighBitDepth(t *testing.T) {
	plane, _ := testPlane(4, 3, 2, 10)
	edges := IntraEdges{
		Above:          []uint16{100, 200, 300, 400},
		AboveAvailable: true,
	}
	if err := PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 4, 3, IntraModeVertical, edges); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 4 {
			want := uint16((x + 1) * 100)
			if got := getSample(plane, 2, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictIntraPlaneBlockHorizontal(t *testing.T) {
	plane, _ := testPlane(5, 3, 1, 8)
	edges := IntraEdges{
		Left:          []uint16{7, 8, 9},
		LeftAvailable: true,
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 1, 0, 4, 3, IntraModeHorizontal, edges); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := 1; x < 5; x++ {
			want := uint16(7 + y)
			if got := getSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictIntraPlaneBlockRejectsInvalidInputs(t *testing.T) {
	plane, _ := testPlane(4, 4, 1, 4)
	if err := PredictIntraPlaneBlock(plane, 1, 10, 0, 0, 1, 1, IntraModeDC, IntraEdges{}); !errors.Is(err, ErrInvalidPrediction) {
		t.Fatalf("bitdepth mismatch err=%v want %v", err, ErrInvalidPrediction)
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 5, 1, IntraModeDC, IntraEdges{}); !errors.Is(err, ErrInvalidPrediction) {
		t.Fatalf("outside block err=%v want %v", err, ErrInvalidPrediction)
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 2, 1, IntraModeVertical, IntraEdges{}); !errors.Is(err, ErrInvalidPrediction) {
		t.Fatalf("missing above err=%v want %v", err, ErrInvalidPrediction)
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 2, 1, IntraModeVertical, IntraEdges{Above: []uint16{1}, AboveAvailable: true}); !errors.Is(err, ErrInvalidPrediction) {
		t.Fatalf("short above err=%v want %v", err, ErrInvalidPrediction)
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 1, 2, IntraModeHorizontal, IntraEdges{Left: []uint16{1}, LeftAvailable: true}); !errors.Is(err, ErrInvalidPrediction) {
		t.Fatalf("short left err=%v want %v", err, ErrInvalidPrediction)
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 1, 1, IntraModeDC, IntraEdges{Above: []uint16{256}, AboveAvailable: true}); !errors.Is(err, ErrInvalidPrediction) {
		t.Fatalf("invalid sample err=%v want %v", err, ErrInvalidPrediction)
	}
	if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 1, 1, IntraMode(99), IntraEdges{}); !errors.Is(err, ErrInvalidPrediction) {
		t.Fatalf("invalid mode err=%v want %v", err, ErrInvalidPrediction)
	}
}

func TestPredictIntraPlaneBlockAllocs(t *testing.T) {
	plane, _ := testPlane(16, 16, 1, 16)
	edges := IntraEdges{
		Above:          make([]uint16, 16),
		Left:           make([]uint16, 16),
		AboveAvailable: true,
		LeftAvailable:  true,
	}
	for i := range 16 {
		edges.Above[i] = 90
		edges.Left[i] = 92
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 16, 16, IntraModeDC, edges); err != nil {
			t.Fatal(err)
		}
		if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 16, 16, IntraModeVertical, edges); err != nil {
			t.Fatal(err)
		}
		if err := PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 16, 16, IntraModeHorizontal, edges); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("intra prediction allocated: %f", allocs)
	}
}

func FuzzPredictIntraPlaneBlock(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(1), uint8(0), uint8(0), uint8(4), uint8(4), uint8(IntraModeDC), uint16(90), uint16(100), true, true)
	f.Add(uint8(17), uint8(9), uint8(2), uint8(3), uint8(2), uint8(8), uint8(4), uint8(IntraModeVertical), uint16(512), uint16(700), true, false)
	f.Add(uint8(5), uint8(5), uint8(1), uint8(4), uint8(4), uint8(1), uint8(1), uint8(IntraModeHorizontal), uint16(12), uint16(27), false, true)

	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawBPS uint8, rawX uint8, rawY uint8, rawBW uint8, rawBH uint8, rawMode uint8, aboveValue uint16, leftValue uint16, aboveAvailable bool, leftAvailable bool) {
		width := int(rawW%32) + 1
		height := int(rawH%32) + 1
		bytesPerSample := int(rawBPS%2) + 1
		bitDepth := uint8(8)
		max := uint16(0xff)
		if bytesPerSample == 2 {
			bitDepth = 10
			max = 0x3ff
		}
		stride := (width + 5) * bytesPerSample
		plane, _ := testPlane(width, height, bytesPerSample, stride)

		x := int(rawX) % width
		y := int(rawY) % height
		blockW := int(rawBW)%width + 1
		blockH := int(rawBH)%height + 1
		if blockW > width-x {
			blockW = width - x
		}
		if blockH > height-y {
			blockH = height - y
		}

		above := make([]uint16, blockW)
		left := make([]uint16, blockH)
		aboveValue &= max
		leftValue &= max
		for i := range above {
			above[i] = aboveValue
		}
		for i := range left {
			left[i] = leftValue
		}
		mode := IntraMode(rawMode % 3)
		if mode == IntraModeVertical {
			aboveAvailable = true
		}
		if mode == IntraModeHorizontal {
			leftAvailable = true
		}
		edges := IntraEdges{
			Above:          above,
			Left:           left,
			AboveAvailable: aboveAvailable,
			LeftAvailable:  leftAvailable,
		}
		if err := PredictIntraPlaneBlock(plane, bytesPerSample, bitDepth, x, y, blockW, blockH, mode, edges); err != nil {
			t.Fatalf("PredictIntraPlaneBlock err=%v", err)
		}

		wantDC := uint16((int(max) + 1) >> 1)
		if aboveAvailable || leftAvailable {
			sum := 0
			count := 0
			if aboveAvailable {
				sum += int(aboveValue) * blockW
				count += blockW
			}
			if leftAvailable {
				sum += int(leftValue) * blockH
				count += blockH
			}
			wantDC = uint16((sum + (count >> 1)) / count)
		}
		for row := 0; row < blockH; row++ {
			for col := 0; col < blockW; col++ {
				want := wantDC
				if mode == IntraModeVertical {
					want = aboveValue
				}
				if mode == IntraModeHorizontal {
					want = leftValue
				}
				got := getSample(plane, bytesPerSample, x+col, y+row)
				if got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", col, row, got, want)
				}
			}
		}
	})
}

func FuzzDCPredictor(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint16(90), uint16(100), true, true)
	f.Add(uint8(1), uint8(2), uint8(1), uint16(512), uint16(700), true, false)
	f.Add(uint8(4), uint8(4), uint8(2), uint16(4095), uint16(1), false, true)

	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawBitDepth uint8, aboveValue uint16, leftValue uint16, aboveAvailable bool, leftAvailable bool) {
		sizes := [...]int{4, 8, 16, 32, 64}
		width := sizes[rawW%uint8(len(sizes))]
		height := sizes[rawH%uint8(len(sizes))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		aboveValue &= max
		leftValue &= max
		edges := IntraEdges{
			Above:          make([]uint16, width),
			Left:           make([]uint16, height),
			AboveAvailable: aboveAvailable,
			LeftAvailable:  leftAvailable,
		}
		for i := range edges.Above {
			edges.Above[i] = aboveValue
		}
		for i := range edges.Left {
			edges.Left[i] = leftValue
		}
		plane, _ := testPlane(width, height, bytesPerSample, width*bytesPerSample)
		if err := PredictIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, width, height, IntraModeDC, edges); err != nil {
			t.Fatalf("PredictIntraPlaneBlock err=%v", err)
		}
		want := dcPredictionLibaomReference(width, height, max, edges)
		for row := range height {
			for col := range width {
				if got := getSample(plane, bytesPerSample, col, row); got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", col, row, got, want)
				}
			}
		}
	})
}

func BenchmarkPredictIntraPlaneBlockDC(b *testing.B) {
	plane, _ := testPlane(64, 64, 1, 64)
	edges := benchmarkEdges(64, 64, 91, 93)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 64, 64, IntraModeDC, edges)
	}
}

func dcPredictionLibaomReference(width int, height int, max uint16, edges IntraEdges) uint16 {
	sum := 0
	count := 0
	if edges.AboveAvailable {
		for _, sample := range edges.Above[:width] {
			sum += int(sample)
		}
		count += width
	}
	if edges.LeftAvailable {
		for _, sample := range edges.Left[:height] {
			sum += int(sample)
		}
		count += height
	}
	if count == 0 {
		return uint16((int(max) + 1) >> 1)
	}
	if edges.AboveAvailable && edges.LeftAvailable {
		if value, ok := dcRectPredictionValue(width, height, max, sum); ok {
			return value
		}
	}
	return uint16((sum + (count >> 1)) / count)
}

func BenchmarkPredictIntraPlaneBlockVertical(b *testing.B) {
	plane, _ := testPlane(64, 64, 1, 64)
	edges := benchmarkEdges(64, 64, 91, 93)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 64, 64, IntraModeVertical, edges)
	}
}

func BenchmarkPredictIntraPlaneBlockHorizontal(b *testing.B) {
	plane, _ := testPlane(64, 64, 1, 64)
	edges := benchmarkEdges(64, 64, 91, 93)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 64, 64, IntraModeHorizontal, edges)
	}
}

func benchmarkEdges(width int, height int, aboveValue uint16, leftValue uint16) IntraEdges {
	above := make([]uint16, width)
	left := make([]uint16, height)
	for i := range above {
		above[i] = aboveValue
	}
	for i := range left {
		left[i] = leftValue
	}
	return IntraEdges{
		Above:          above,
		Left:           left,
		AboveAvailable: true,
		LeftAvailable:  true,
	}
}

func testPlane(width int, height int, bytesPerSample int, stride int) (frame.Plane, []byte) {
	buf := make([]byte, stride*height)
	return frame.Plane{
		Pix:    buf,
		Stride: stride,
		Width:  width,
		Height: height,
	}, buf
}

func getSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}
