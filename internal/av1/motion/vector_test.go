package motion

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestFullpelVectorAndReferenceOrigin(t *testing.T) {
	mv, err := FullpelVector(2, -1)
	if err != nil {
		t.Fatal(err)
	}
	if mv != (Vector{Row: -8, Col: 16}) {
		t.Fatalf("mv=%+v", mv)
	}
	if !mv.IsFullpel() {
		t.Fatal("fullpel vector reported fractional")
	}
	dx, dy, err := mv.FullpelOffset()
	if err != nil {
		t.Fatal(err)
	}
	if dx != 2 || dy != -1 {
		t.Fatalf("offset=%d,%d want 2,-1", dx, dy)
	}
	refX, refY, err := FullpelReferenceOrigin(5, 6, mv)
	if err != nil {
		t.Fatal(err)
	}
	if refX != 7 || refY != 5 {
		t.Fatalf("origin=%d,%d want 7,5", refX, refY)
	}
	if _, _, err := (Vector{Col: 4}).FullpelOffset(); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("fractional offset err=%v want %v", err, ErrInvalidMotion)
	}
}

func TestPredictInterPlaneBlock8Bit(t *testing.T) {
	src, _ := testPlane(7, 6, 1, 9)
	dst, _ := testPlane(7, 6, 1, 9)
	for i := range dst.Pix {
		dst.Pix[i] = 0xee
	}
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setSample(src, 1, x, y, uint16(10*y+x))
		}
	}
	mv, err := FullpelVector(-1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := PredictInterPlaneBlock(dst, src, 1, 2, 1, 3, 2, mv); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < dst.Height; y++ {
		for x := 0; x < dst.Width; x++ {
			got := getSample(dst, 1, x, y)
			want := uint16(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = uint16(10*(y+1) + (x - 1))
			}
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictInterPlaneBlockHighBitDepth(t *testing.T) {
	src, _ := testPlane(6, 5, 2, 14)
	dst, _ := testPlane(6, 5, 2, 14)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setSample(src, 2, x, y, uint16(1000+y*10+x))
		}
	}
	mv, err := FullpelVector(1, -1)
	if err != nil {
		t.Fatal(err)
	}
	if err := PredictInterPlaneBlock(dst, src, 2, 1, 2, 3, 2, mv); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			got := getSample(dst, 2, 1+x, 2+y)
			want := uint16(1000 + (1+y)*10 + 2 + x)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictInterPlaneBlockRejectsInvalidInputs(t *testing.T) {
	src, _ := testPlane(4, 4, 1, 4)
	dst, _ := testPlane(4, 4, 1, 4)
	if err := PredictInterPlaneBlock(dst, src, 1, 0, 0, 1, 1, Vector{Col: 4}); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("fractional err=%v want %v", err, ErrInvalidMotion)
	}
	mv, err := FullpelVector(-1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := PredictInterPlaneBlock(dst, src, 1, 0, 0, 1, 1, mv); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("outside ref err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlock(dst, src, 3, 0, 0, 1, 1, Vector{}); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("invalid sample width err=%v want %v", err, ErrInvalidMotion)
	}
	if _, err := FullpelVector(int(maxInt32)/SubpelScale+1, 0); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("overflow vector err=%v want %v", err, ErrInvalidMotion)
	}
}

func TestPredictInterPlaneBlockAllocs(t *testing.T) {
	src, _ := testPlane(16, 16, 1, 16)
	dst, _ := testPlane(16, 16, 1, 16)
	mv, err := FullpelVector(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictInterPlaneBlock(dst, src, 1, 0, 0, 16, 16, mv); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("inter prediction allocated: %f", allocs)
	}
}

func FuzzPredictInterPlaneBlock(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(1), uint8(0), uint8(0), uint8(0), uint8(0), uint8(4), uint8(4))
	f.Add(uint8(17), uint8(9), uint8(2), uint8(3), uint8(2), uint8(1), uint8(1), uint8(8), uint8(4))
	f.Add(uint8(5), uint8(5), uint8(1), uint8(4), uint8(4), uint8(2), uint8(2), uint8(1), uint8(1))

	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawBPS uint8, rawDstX uint8, rawDstY uint8, rawRefX uint8, rawRefY uint8, rawBW uint8, rawBH uint8) {
		width := int(rawW%32) + 1
		height := int(rawH%32) + 1
		bytesPerSample := int(rawBPS%2) + 1
		stride := (width + 7) * bytesPerSample
		src, _ := testPlane(width, height, bytesPerSample, stride)
		dst, _ := testPlane(width, height, bytesPerSample, stride)

		blockW := int(rawBW)%width + 1
		blockH := int(rawBH)%height + 1
		dstX := int(rawDstX) % (width - blockW + 1)
		dstY := int(rawDstY) % (height - blockH + 1)
		refX := int(rawRefX) % (width - blockW + 1)
		refY := int(rawRefY) % (height - blockH + 1)

		for y := 0; y < src.Height; y++ {
			for x := 0; x < src.Width; x++ {
				value := uint16((y*width + x) & 0xff)
				if bytesPerSample == 2 {
					value = uint16((y*width + x) & 0x3ff)
				}
				setSample(src, bytesPerSample, x, y, value)
			}
		}

		mv, err := FullpelVector(refX-dstX, refY-dstY)
		if err != nil {
			t.Fatal(err)
		}
		if err := PredictInterPlaneBlock(dst, src, bytesPerSample, dstX, dstY, blockW, blockH, mv); err != nil {
			t.Fatalf("PredictInterPlaneBlock err=%v", err)
		}
		for y := 0; y < blockH; y++ {
			for x := 0; x < blockW; x++ {
				got := getSample(dst, bytesPerSample, dstX+x, dstY+y)
				want := getSample(src, bytesPerSample, refX+x, refY+y)
				if got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
				}
			}
		}
	})
}

func BenchmarkFullpelReferenceOrigin(b *testing.B) {
	mv, err := FullpelVector(2, -1)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = FullpelReferenceOrigin(64, 64, mv)
	}
}

func BenchmarkPredictInterPlaneBlock(b *testing.B) {
	src, _ := testPlane(64, 64, 1, 64)
	dst, _ := testPlane(64, 64, 1, 64)
	mv, err := FullpelVector(0, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictInterPlaneBlock(dst, src, 1, 0, 0, 64, 64, mv)
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

func setSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}
