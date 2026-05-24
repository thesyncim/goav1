package prediction

import (
	"errors"
	"slices"
	"testing"
)

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
	if err := PredictFilterIntraPlaneBlock(plane, 1, 8, 0, 0, 4, 4, FilterIntraModePaeth, edges); err != nil {
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
	if err := PredictFilterIntraPlaneBlockWithExtent(plane, 1, 8, 0, 0, 6, 6, 8, 8, FilterIntraModePaeth, edges); err != nil {
		t.Fatal(err)
	}
	full := filterIntraLibaomReference(8, 8, FilterIntraModePaeth, edges, 0xff)
	got := collectPlaneSamples(plane, 1, 6, 6)
	for row := 0; row < 6; row++ {
		for col := 0; col < 6; col++ {
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
		for mode := FilterIntraModeDC; mode < FilterIntraModes; mode++ {
			for _, size := range libaomFilterIntraTxSizes {
				rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
				for iter := 0; iter < 32; iter++ {
					edges := randomStaticIntraEdges(size.width, size.height, mask, rnd)
					want := filterIntraLibaomReference(size.width, size.height, mode, edges, mask)
					plane, _ := testPlane(size.width, size.height, bytesPerSample, size.width*bytesPerSample)
					if err := PredictFilterIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, mode, edges); err != nil {
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
			if err := PredictFilterIntraPlaneBlock(plane, 1, 8, 0, 0, tt.width, tt.height, tt.mode, tt.edges); !errors.Is(err, ErrInvalidPrediction) {
				t.Fatalf("err=%v want %v", err, ErrInvalidPrediction)
			}
		})
	}
}

func TestFilterIntraPredictorAllocs(t *testing.T) {
	plane, _ := testPlane(32, 32, 2, 32*2)
	edges := randomStaticIntraEdges(32, 32, 0xfff, newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed))
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictFilterIntraPlaneBlock(plane, 2, 12, 0, 0, 32, 32, FilterIntraModePaeth, edges); err != nil {
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
		if err := PredictFilterIntraPlaneBlock(plane, bytesPerSample, bitDepth, 0, 0, size.width, size.height, mode, edges); err != nil {
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
		_ = PredictFilterIntraPlaneBlock(plane, 2, 12, 0, 0, 32, 32, FilterIntraModeD157, edges)
	}
}

func filterIntraLibaomReference(width int, height int, mode FilterIntraMode, edges IntraEdges, max uint16) []uint16 {
	var buffer [33][33]uint16
	for row := 0; row < height; row++ {
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
			for k := 0; k < 8; k++ {
				sum := 0
				for i := 0; i < len(p); i++ {
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
	for row := 0; row < height; row++ {
		out = append(out, buffer[row+1][1:width+1]...)
	}
	return out
}
