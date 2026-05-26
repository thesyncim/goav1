package loopfilter

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestFilter4EdgeHorizontal8Bit(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for x := range 4 {
		setSample(plane, 1, x, 0, 90)
		setSample(plane, 1, x, 1, 90)
		setSample(plane, 1, x, 2, 100)
		setSample(plane, 1, x, 3, 100)
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 2, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for x := range 4 {
		if got := getSample(plane, 1, x, 0); got != 92 {
			t.Fatalf("p1 sample=%d want 92", got)
		}
		if got := getSample(plane, 1, x, 1); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, x, 2); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, x, 3); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
	}
}

func TestFilter4EdgeVerticalHighVariance8Bit(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for y := range 4 {
		setSample(plane, 1, 0, y, 70)
		setSample(plane, 1, 1, y, 90)
		setSample(plane, 1, 2, y, 100)
		setSample(plane, 1, 3, y, 130)
	}

	thresholds := Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3}
	if err := Filter4Edge(plane, 1, 8, EdgeVertical, 2, 0, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for y := range 4 {
		if got := getSample(plane, 1, 0, y); got != 70 {
			t.Fatalf("p1 sample=%d want 70", got)
		}
		if got := getSample(plane, 1, 1, y); got != 86 {
			t.Fatalf("p0 sample=%d want 86", got)
		}
		if got := getSample(plane, 1, 2, y); got != 104 {
			t.Fatalf("q0 sample=%d want 104", got)
		}
		if got := getSample(plane, 1, 3, y); got != 130 {
			t.Fatalf("q1 sample=%d want 130", got)
		}
	}
}

func TestFilter4EdgeHighBitDepth(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		plane := testPlane(4, 4, 2, 8)
		for x := range 4 {
			setSample(plane, 2, x, 0, 900)
			setSample(plane, 2, x, 1, 900)
			setSample(plane, 2, x, 2, 940)
			setSample(plane, 2, x, 3, 940)
		}

		thresholds := Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3}
		if err := Filter4Edge(plane, 2, bitDepth, EdgeHorizontal, 0, 2, 4, thresholds); err != nil {
			t.Fatal(err)
		}

		for x := range 4 {
			if got := getSample(plane, 2, x, 0); got != 908 {
				t.Fatalf("bitdepth=%d p1 sample=%d want 908", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 1); got != 915 {
				t.Fatalf("bitdepth=%d p0 sample=%d want 915", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 2); got != 925 {
				t.Fatalf("bitdepth=%d q0 sample=%d want 925", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 3); got != 932 {
				t.Fatalf("bitdepth=%d q1 sample=%d want 932", bitDepth, got)
			}
		}
	}
}

func TestFilter4EdgeLeavesMaskedSamples(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for x := range 4 {
		setSample(plane, 1, x, 0, 10)
		setSample(plane, 1, x, 1, 10)
		setSample(plane, 1, x, 2, 200)
		setSample(plane, 1, x, 3, 200)
	}
	if err := Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 2, 4, Thresholds{Limit: 63, BlockLimit: 20}); err != nil {
		t.Fatal(err)
	}
	for x := range 4 {
		if got := getSample(plane, 1, x, 1); got != 10 {
			t.Fatalf("p0 sample=%d want 10", got)
		}
		if got := getSample(plane, 1, x, 2); got != 200 {
			t.Fatalf("q0 sample=%d want 200", got)
		}
	}
}

func TestFilter4EdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
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
		{name: "sample width", plane: plane, bytesPerSample: 2, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 2, length: 4},
		{name: "edge", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: Edge(9), x: 0, y: 2, length: 4},
		{name: "length", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 2},
		{name: "horizontal top", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 1, length: 4},
		{name: "horizontal width", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 1, y: 2, length: 4},
		{name: "vertical left", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 1, y: 0, length: 4},
		{name: "vertical height", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 2, y: 1, length: 4},
	}
	for _, tt := range tests {
		err := Filter4Edge(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.edge, tt.x, tt.y, tt.length, Thresholds{})
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestFilter4EdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter4Edge allocated: %f", allocs)
	}
}

func FuzzFilter4Edge(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), uint8(20), uint8(25), uint8(10))
	f.Add(uint8(10), uint8(10), uint8(1), uint8(63), uint8(132), uint8(3))
	f.Fuzz(func(t *testing.T, rawWidth uint8, rawHeight uint8, rawEdge uint8, rawLimit uint8, rawBlockLimit uint8, rawHEV uint8) {
		width := int(rawWidth%32) + 4
		height := int(rawHeight%32) + 4
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
		x := 2
		y := 2
		length := width - x
		if edge == EdgeVertical {
			length = height - y
		}
		err := Filter4Edge(plane, bytesPerSample, bitDepth, edge, x, y, length, Thresholds{
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

func BenchmarkFilter4EdgeHorizontal(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds)
	}
}

func BenchmarkFilter4EdgeVertical(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter4Edge(plane, 1, 8, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func testPlane(width int, height int, bytesPerSample int, stride int) frame.Plane {
	return frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
}

func setSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func getSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}
