package prediction

import (
	"errors"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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

func TestPredictDirectionalIntraPlaneBlockMatchesLibaomSaturatedCorpus(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		bytesPerSample := 1
		if bitDepth != 8 {
			bytesPerSample = 2
		}
		max := uint16((1 << bitDepth) - 1)
		for _, size := range libaomDRTxSizes {
			for _, angleRange := range [][2]int{{1, 89}, {91, 179}, {181, 269}} {
				for enableUpsample := 0; enableUpsample < 2; enableUpsample++ {
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
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
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
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
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
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				out[row*width+col] = edges.Above[edges.AboveOrigin+col]
			}
		}
	case angle == 180:
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
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
					for col := 0; col < width; col++ {
						out[row*width+col] = edges.Above[edges.AboveOrigin+maxBaseX]
					}
				}
				return out, nil
			}
			for col := 0; col < width; col++ {
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
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
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
		for col := 0; col < width; col++ {
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
