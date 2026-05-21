package prediction

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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

func TestPredictIntraPlaneBlockVerticalHighBitDepth(t *testing.T) {
	plane, _ := testPlane(4, 3, 2, 10)
	edges := IntraEdges{
		Above:          []uint16{100, 200, 300, 400},
		AboveAvailable: true,
	}
	if err := PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 4, 3, IntraModeVertical, edges); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 3; y++ {
		for x := 0; x < 4; x++ {
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
	for y := 0; y < 3; y++ {
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
	for i := 0; i < 16; i++ {
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

func BenchmarkPredictIntraPlaneBlockDC(b *testing.B) {
	plane, _ := testPlane(64, 64, 1, 64)
	edges := benchmarkEdges(64, 64, 91, 93)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 64, 64, IntraModeDC, edges)
	}
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
