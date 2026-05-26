package loopfilter

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestFilter8EdgeHorizontalFlat8Bit(t *testing.T) {
	plane := testPlane(4, 8, 1, 4)
	for x := range 4 {
		for y := range 4 {
			setSample(plane, 1, x, y, 90)
		}
		for y := 4; y < 8; y++ {
			setSample(plane, 1, x, y, 100)
		}
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter8Edge(plane, 1, 8, EdgeHorizontal, 0, 4, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for x := range 4 {
		if got := getSample(plane, 1, x, 0); got != 90 {
			t.Fatalf("p3 sample=%d want 90", got)
		}
		if got := getSample(plane, 1, x, 1); got != 91 {
			t.Fatalf("p2 sample=%d want 91", got)
		}
		if got := getSample(plane, 1, x, 2); got != 93 {
			t.Fatalf("p1 sample=%d want 93", got)
		}
		if got := getSample(plane, 1, x, 3); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, x, 4); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, x, 5); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
		if got := getSample(plane, 1, x, 6); got != 99 {
			t.Fatalf("q2 sample=%d want 99", got)
		}
		if got := getSample(plane, 1, x, 7); got != 100 {
			t.Fatalf("q3 sample=%d want 100", got)
		}
	}
}

func TestFilter8EdgeVerticalFallback8Bit(t *testing.T) {
	plane := testPlane(8, 4, 1, 8)
	for y := range 4 {
		setSample(plane, 1, 0, y, 80)
		setSample(plane, 1, 1, y, 90)
		setSample(plane, 1, 2, y, 90)
		setSample(plane, 1, 3, y, 90)
		setSample(plane, 1, 4, y, 100)
		setSample(plane, 1, 5, y, 100)
		setSample(plane, 1, 6, y, 100)
		setSample(plane, 1, 7, y, 100)
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter8Edge(plane, 1, 8, EdgeVertical, 4, 0, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for y := range 4 {
		if got := getSample(plane, 1, 0, y); got != 80 {
			t.Fatalf("p3 sample=%d want 80", got)
		}
		if got := getSample(plane, 1, 1, y); got != 90 {
			t.Fatalf("p2 sample=%d want 90", got)
		}
		if got := getSample(plane, 1, 2, y); got != 92 {
			t.Fatalf("p1 sample=%d want 92", got)
		}
		if got := getSample(plane, 1, 3, y); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, 4, y); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, 5, y); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
		if got := getSample(plane, 1, 6, y); got != 100 {
			t.Fatalf("q2 sample=%d want 100", got)
		}
		if got := getSample(plane, 1, 7, y); got != 100 {
			t.Fatalf("q3 sample=%d want 100", got)
		}
	}
}

func TestFilter8EdgeHighBitDepth(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		plane := testPlane(4, 8, 2, 8)
		for x := range 4 {
			for y := range 4 {
				setSample(plane, 2, x, y, 900)
			}
			for y := 4; y < 8; y++ {
				setSample(plane, 2, x, y, 940)
			}
		}

		thresholds := Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3}
		if err := Filter8Edge(plane, 2, bitDepth, EdgeHorizontal, 0, 4, 4, thresholds); err != nil {
			t.Fatal(err)
		}

		for x := range 4 {
			if got := getSample(plane, 2, x, 1); got != 905 {
				t.Fatalf("bitdepth=%d p2 sample=%d want 905", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 2); got != 910 {
				t.Fatalf("bitdepth=%d p1 sample=%d want 910", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 3); got != 915 {
				t.Fatalf("bitdepth=%d p0 sample=%d want 915", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 4); got != 925 {
				t.Fatalf("bitdepth=%d q0 sample=%d want 925", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 5); got != 930 {
				t.Fatalf("bitdepth=%d q1 sample=%d want 930", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 6); got != 935 {
				t.Fatalf("bitdepth=%d q2 sample=%d want 935", bitDepth, got)
			}
		}
	}
}

func TestFilter8EdgeMatchesCReference(t *testing.T) {
	tests := []struct {
		name       string
		edge       Edge
		bitDepth   uint8
		thresholds Thresholds
		samples    [][8]int
	}{
		{
			name:       "horizontal 8 bit",
			edge:       EdgeHorizontal,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][8]int{
				{90, 90, 90, 90, 100, 100, 100, 100},
				{80, 90, 90, 90, 100, 100, 100, 100},
				{70, 70, 70, 90, 100, 130, 130, 130},
				{10, 10, 10, 10, 200, 200, 200, 200},
			},
		},
		{
			name:       "vertical 8 bit",
			edge:       EdgeVertical,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][8]int{
				{90, 90, 90, 90, 100, 100, 100, 100},
				{80, 90, 90, 90, 100, 100, 100, 100},
				{70, 70, 70, 90, 100, 130, 130, 130},
				{10, 10, 10, 10, 200, 200, 200, 200},
			},
		},
		{
			name:       "horizontal high bit depth",
			edge:       EdgeHorizontal,
			bitDepth:   10,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][8]int{
				{900, 900, 900, 900, 940, 940, 940, 940},
				{850, 900, 900, 900, 940, 940, 940, 940},
				{200, 200, 200, 200, 900, 900, 900, 900},
			},
		},
		{
			name:       "vertical high bit depth",
			edge:       EdgeVertical,
			bitDepth:   12,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][8]int{
				{900, 900, 900, 900, 940, 940, 940, 940},
				{850, 900, 900, 900, 940, 940, 940, 940},
				{200, 200, 200, 200, 900, 900, 900, 900},
			},
		},
	}

	for _, tt := range tests {
		bytesPerSample := 1
		if tt.bitDepth != 8 {
			bytesPerSample = 2
		}

		got := filter8ReferencePlane(tt.edge, bytesPerSample, tt.samples)
		want := cloneTestPlane(got)
		for i, sample := range tt.samples {
			ref := cReferenceFilter8(sample, tt.bitDepth, tt.thresholds)
			writeFilter8ReferenceSample(want, bytesPerSample, tt.edge, i, ref)
		}

		x, y, length := 0, 4, len(tt.samples)
		if tt.edge == EdgeVertical {
			x, y = 4, 0
		}
		if err := Filter8Edge(got, bytesPerSample, tt.bitDepth, tt.edge, x, y, length, tt.thresholds); err != nil {
			t.Fatalf("%s Filter8Edge: %v", tt.name, err)
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("%s got=%v want=%v", tt.name, got.Pix, want.Pix)
		}
	}
}

func TestFilter8EdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(8, 8, 1, 8)
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
		{name: "sample width", plane: plane, bytesPerSample: 2, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 4, length: 4},
		{name: "edge", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: Edge(9), x: 0, y: 4, length: 4},
		{name: "length", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 4},
		{name: "horizontal top", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 3, length: 4},
		{name: "horizontal bottom", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 5, length: 4},
		{name: "vertical left", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 3, y: 0, length: 4},
		{name: "vertical right", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 5, y: 0, length: 4},
	}
	for _, tt := range tests {
		err := Filter8Edge(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.edge, tt.x, tt.y, tt.length, Thresholds{})
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestFilter8EdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := Filter8Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter8Edge allocated: %f", allocs)
	}
}

func FuzzFilter8Edge(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), uint8(20), uint8(25), uint8(10))
	f.Add(uint8(10), uint8(10), uint8(1), uint8(63), uint8(132), uint8(3))
	f.Fuzz(func(t *testing.T, rawWidth uint8, rawHeight uint8, rawEdge uint8, rawLimit uint8, rawBlockLimit uint8, rawHEV uint8) {
		width := int(rawWidth%32) + 8
		height := int(rawHeight%32) + 8
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
		x := 4
		y := 4
		length := width - x
		if edge == EdgeVertical {
			length = height - y
		}
		err := Filter8Edge(plane, bytesPerSample, bitDepth, edge, x, y, length, Thresholds{
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

func filter8ReferencePlane(edge Edge, bytesPerSample int, samples [][8]int) frame.Plane {
	if edge == EdgeHorizontal {
		plane := testPlane(len(samples), 8, bytesPerSample, len(samples)*bytesPerSample)
		for i, sample := range samples {
			writeFilter8ReferenceSample(plane, bytesPerSample, edge, i, sample)
		}
		return plane
	}
	plane := testPlane(8, len(samples), bytesPerSample, 8*bytesPerSample)
	for i, sample := range samples {
		writeFilter8ReferenceSample(plane, bytesPerSample, edge, i, sample)
	}
	return plane
}

func writeFilter8ReferenceSample(plane frame.Plane, bytesPerSample int, edge Edge, index int, sample [8]int) {
	for i, value := range sample {
		if edge == EdgeHorizontal {
			setSample(plane, bytesPerSample, index, i, uint16(value))
			continue
		}
		setSample(plane, bytesPerSample, i, index, uint16(value))
	}
}

func cReferenceFilter8(in [8]int, bitDepth uint8, thresholds Thresholds) [8]int {
	scale := 1 << int(bitDepth-8)
	limit := int(thresholds.Limit) * scale
	blimit := int(thresholds.BlockLimit) * scale
	hev := int(thresholds.HighEdgeVariance) * scale
	min := -128 * scale
	max := 128*scale - 1
	center := 128 * scale

	p3, p2, p1, p0 := in[0], in[1], in[2], in[3]
	q0, q1, q2, q3 := in[4], in[5], in[6], in[7]
	if !cReferenceFilterMask4(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3) {
		return in
	}
	if cReferenceFlatMask4(scale, p3, p2, p1, p0, q0, q1, q2, q3) {
		in[1] = cRoundPowerOfTwo(p3+p3+p3+2*p2+p1+p0+q0, 3)
		in[2] = cRoundPowerOfTwo(p3+p3+p2+2*p1+p0+q0+q1, 3)
		in[3] = cRoundPowerOfTwo(p3+p2+p1+2*p0+q0+q1+q2, 3)
		in[4] = cRoundPowerOfTwo(p2+p1+p0+2*q0+q1+q2+q3, 3)
		in[5] = cRoundPowerOfTwo(p1+p0+q0+2*q1+q2+q3+q3, 3)
		in[6] = cRoundPowerOfTwo(p0+q0+q1+2*q2+q3+q3+q3, 3)
		return in
	}
	in[2], in[3], in[4], in[5] = cReferenceFilter4(p1, p0, q0, q1, hev, min, max, center)
	return in
}

func cReferenceFilterMask4(limit int, blimit int, p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int) bool {
	return cAbs(p3-p2) <= limit &&
		cAbs(p2-p1) <= limit &&
		cAbs(p1-p0) <= limit &&
		cAbs(q1-q0) <= limit &&
		cAbs(q2-q1) <= limit &&
		cAbs(q3-q2) <= limit &&
		cAbs(p0-q0)*2+cAbs(p1-q1)/2 <= blimit
}

func cReferenceFlatMask4(thresh int, p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int) bool {
	return cAbs(p1-p0) <= thresh &&
		cAbs(q1-q0) <= thresh &&
		cAbs(p2-p0) <= thresh &&
		cAbs(q2-q0) <= thresh &&
		cAbs(p3-p0) <= thresh &&
		cAbs(q3-q0) <= thresh
}

func BenchmarkFilter8EdgeHorizontal(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds)
	}
}

func BenchmarkFilter8EdgeVertical(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 1, 8, EdgeVertical, 32, 0, 64, thresholds)
	}
}
