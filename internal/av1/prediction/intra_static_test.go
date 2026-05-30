package prediction

import (
	"errors"
	"slices"
	"testing"
)

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
