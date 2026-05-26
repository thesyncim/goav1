package loopfilter

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestFilter14EdgeHorizontalFlat8Bit(t *testing.T) {
	plane := testPlane(4, 14, 1, 4)
	for x := range 4 {
		for y := range 7 {
			setSample(plane, 1, x, y, 90)
		}
		for y := 7; y < 14; y++ {
			setSample(plane, 1, x, y, 100)
		}
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter14Edge(plane, 1, 8, EdgeHorizontal, 0, 7, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	want := []uint16{90, 91, 91, 92, 93, 93, 94, 96, 97, 98, 98, 99, 99, 100}
	for x := range 4 {
		for y, value := range want {
			if got := getSample(plane, 1, x, y); got != value {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, value)
			}
		}
	}
}

func TestFilter14EdgeHighBitDepth(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		plane := testPlane(4, 14, 2, 8)
		for x := range 4 {
			for y := range 7 {
				setSample(plane, 2, x, y, 900)
			}
			for y := 7; y < 14; y++ {
				setSample(plane, 2, x, y, 940)
			}
		}

		thresholds := Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3}
		if err := Filter14Edge(plane, 2, bitDepth, EdgeHorizontal, 0, 7, 4, thresholds); err != nil {
			t.Fatal(err)
		}

		want := []uint16{900, 903, 905, 908, 910, 913, 918, 923, 928, 930, 933, 935, 938, 940}
		for x := range 4 {
			for y, value := range want {
				if got := getSample(plane, 2, x, y); got != value {
					t.Fatalf("bitdepth=%d sample(%d,%d)=%d want %d", bitDepth, x, y, got, value)
				}
			}
		}
	}
}

func TestFilter14EdgeMatchesCReference(t *testing.T) {
	tests := []struct {
		name       string
		edge       Edge
		bitDepth   uint8
		thresholds Thresholds
		samples    [][14]int
	}{
		{
			name:       "horizontal 8 bit",
			edge:       EdgeHorizontal,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][14]int{
				{90, 90, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 80, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{10, 10, 10, 10, 10, 10, 10, 200, 200, 200, 200, 200, 200, 200},
			},
		},
		{
			name:       "vertical 8 bit",
			edge:       EdgeVertical,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][14]int{
				{90, 90, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 80, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{10, 10, 10, 10, 10, 10, 10, 200, 200, 200, 200, 200, 200, 200},
			},
		},
		{
			name:       "horizontal high bit depth",
			edge:       EdgeHorizontal,
			bitDepth:   10,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][14]int{
				{900, 900, 900, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{850, 850, 850, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{200, 200, 200, 200, 200, 200, 200, 900, 900, 900, 900, 900, 900, 900},
			},
		},
		{
			name:       "vertical high bit depth",
			edge:       EdgeVertical,
			bitDepth:   12,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][14]int{
				{900, 900, 900, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{850, 850, 850, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{200, 200, 200, 200, 200, 200, 200, 900, 900, 900, 900, 900, 900, 900},
			},
		},
	}

	for _, tt := range tests {
		bytesPerSample := 1
		if tt.bitDepth != 8 {
			bytesPerSample = 2
		}

		got := filter14ReferencePlane(tt.edge, bytesPerSample, tt.samples)
		want := cloneTestPlane(got)
		for i, sample := range tt.samples {
			ref := cReferenceFilter14(sample, tt.bitDepth, tt.thresholds)
			writeFilter14ReferenceSample(want, bytesPerSample, tt.edge, i, ref)
		}

		x, y, length := 0, 7, len(tt.samples)
		if tt.edge == EdgeVertical {
			x, y = 7, 0
		}
		if err := Filter14Edge(got, bytesPerSample, tt.bitDepth, tt.edge, x, y, length, tt.thresholds); err != nil {
			t.Fatalf("%s Filter14Edge: %v", tt.name, err)
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("%s got=%v want=%v", tt.name, got.Pix, want.Pix)
		}
	}
}

func TestFilter14EdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(14, 14, 1, 14)
	tests := []struct {
		name           string
		plane          frame.Plane
		bytesPerSample int
		bitDepth       uint8
		edge           Edge
		x              int
		y              int
		length         int
	}{
		{name: "sample width", plane: plane, bytesPerSample: 2, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 7, length: 4},
		{name: "edge", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: Edge(9), x: 0, y: 7, length: 4},
		{name: "length", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 7},
		{name: "horizontal top", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 6, length: 4},
		{name: "horizontal bottom", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 8, length: 4},
		{name: "vertical left", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 6, y: 0, length: 4},
		{name: "vertical right", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 8, y: 0, length: 4},
	}
	for _, tt := range tests {
		err := Filter14Edge(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.edge, tt.x, tt.y, tt.length, Thresholds{})
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestFilter14EdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := Filter14Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter14Edge allocated: %f", allocs)
	}
}

func FuzzFilter14Edge(f *testing.F) {
	f.Add(uint8(14), uint8(14), uint8(0), uint8(20), uint8(25), uint8(10))
	f.Add(uint8(16), uint8(16), uint8(1), uint8(63), uint8(132), uint8(3))
	f.Fuzz(func(t *testing.T, rawWidth uint8, rawHeight uint8, rawEdge uint8, rawLimit uint8, rawBlockLimit uint8, rawHEV uint8) {
		width := int(rawWidth%32) + 14
		height := int(rawHeight%32) + 14
		edge := Edge(rawEdge & 1)
		bytesPerSample := 1
		bitDepth := uint8(8)
		stride := width
		max := uint16(0xff)
		if rawEdge&2 != 0 {
			bytesPerSample = 2
			bitDepth = 10
			if rawEdge&4 != 0 {
				bitDepth = 12
			}
			stride = width * bytesPerSample
			max = uint16((1 << bitDepth) - 1)
		}
		plane := testPlane(width, height, bytesPerSample, stride)
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setSample(plane, bytesPerSample, x, y, uint16((x*19+y*23)&int(max)))
			}
		}
		x := 7
		y := 7
		length := width - x
		if edge == EdgeVertical {
			length = height - y
		}
		err := Filter14Edge(plane, bytesPerSample, bitDepth, edge, x, y, length, Thresholds{
			Limit:            rawLimit,
			BlockLimit:       rawBlockLimit,
			HighEdgeVariance: rawHEV,
		})
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				if got := getSample(plane, bytesPerSample, x, y); got > max {
					t.Fatalf("sample(%d,%d)=%d max=%d", x, y, got, max)
				}
			}
		}
	})
}

func filter14ReferencePlane(edge Edge, bytesPerSample int, samples [][14]int) frame.Plane {
	if edge == EdgeHorizontal {
		plane := testPlane(len(samples), 14, bytesPerSample, len(samples)*bytesPerSample)
		for i, sample := range samples {
			writeFilter14ReferenceSample(plane, bytesPerSample, edge, i, sample)
		}
		return plane
	}
	plane := testPlane(14, len(samples), bytesPerSample, 14*bytesPerSample)
	for i, sample := range samples {
		writeFilter14ReferenceSample(plane, bytesPerSample, edge, i, sample)
	}
	return plane
}

func writeFilter14ReferenceSample(plane frame.Plane, bytesPerSample int, edge Edge, index int, sample [14]int) {
	for i, value := range sample {
		if edge == EdgeHorizontal {
			setSample(plane, bytesPerSample, index, i, uint16(value))
			continue
		}
		setSample(plane, bytesPerSample, i, index, uint16(value))
	}
}

func cReferenceFilter14(in [14]int, bitDepth uint8, thresholds Thresholds) [14]int {
	scale := 1 << int(bitDepth-8)
	limit := int(thresholds.Limit) * scale
	blimit := int(thresholds.BlockLimit) * scale

	p6, p5, p4, p3 := in[0], in[1], in[2], in[3]
	p2, p1, p0 := in[4], in[5], in[6]
	q0, q1, q2, q3 := in[7], in[8], in[9], in[10]
	q4, q5, q6 := in[11], in[12], in[13]
	if !cReferenceFilterMask4(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3) {
		return in
	}
	flat := cReferenceFlatMask4(scale, p3, p2, p1, p0, q0, q1, q2, q3)
	flat2 := cReferenceFlatMask4(scale, p6, p5, p4, p0, q0, q4, q5, q6)
	if !flat || !flat2 {
		narrow := cReferenceFilter8([8]int{p3, p2, p1, p0, q0, q1, q2, q3}, bitDepth, thresholds)
		in[3], in[4], in[5], in[6] = narrow[0], narrow[1], narrow[2], narrow[3]
		in[7], in[8], in[9], in[10] = narrow[4], narrow[5], narrow[6], narrow[7]
		return in
	}

	in[1] = cRoundPowerOfTwo(p6*7+p5*2+p4*2+p3+p2+p1+p0+q0, 4)
	in[2] = cRoundPowerOfTwo(p6*5+p5*2+p4*2+p3*2+p2+p1+p0+q0+q1, 4)
	in[3] = cRoundPowerOfTwo(p6*4+p5+p4*2+p3*2+p2*2+p1+p0+q0+q1+q2, 4)
	in[4] = cRoundPowerOfTwo(p6*3+p5+p4+p3*2+p2*2+p1*2+p0+q0+q1+q2+q3, 4)
	in[5] = cRoundPowerOfTwo(p6*2+p5+p4+p3+p2*2+p1*2+p0*2+q0+q1+q2+q3+q4, 4)
	in[6] = cRoundPowerOfTwo(p6+p5+p4+p3+p2+p1*2+p0*2+q0*2+q1+q2+q3+q4+q5, 4)
	in[7] = cRoundPowerOfTwo(p5+p4+p3+p2+p1+p0*2+q0*2+q1*2+q2+q3+q4+q5+q6, 4)
	in[8] = cRoundPowerOfTwo(p4+p3+p2+p1+p0+q0*2+q1*2+q2*2+q3+q4+q5+q6*2, 4)
	in[9] = cRoundPowerOfTwo(p3+p2+p1+p0+q0+q1*2+q2*2+q3*2+q4+q5+q6*3, 4)
	in[10] = cRoundPowerOfTwo(p2+p1+p0+q0+q1+q2*2+q3*2+q4*2+q5+q6*4, 4)
	in[11] = cRoundPowerOfTwo(p1+p0+q0+q1+q2+q3*2+q4*2+q5*2+q6*5, 4)
	in[12] = cRoundPowerOfTwo(p0+q0+q1+q2+q3+q4*2+q5*2+q6*7, 4)
	return in
}

func BenchmarkFilter14EdgeHorizontal(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds)
	}
}

func BenchmarkFilter14EdgeVertical(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 1, 8, EdgeVertical, 32, 0, 64, thresholds)
	}
}
