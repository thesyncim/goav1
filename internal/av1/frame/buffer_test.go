package frame

import (
	"errors"
	"testing"
)

func TestRequiredSize420Aligned(t *testing.T) {
	layout, err := RequiredSize(Format{
		Width:        16,
		Height:       9,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	if err != nil {
		t.Fatal(err)
	}

	if layout.YStride != 64 || layout.UStride != 64 || layout.VStride != 64 {
		t.Fatalf("strides: %+v", layout)
	}
	if layout.ChromaWidth != 8 || layout.ChromaHeight != 5 {
		t.Fatalf("chroma: %dx%d", layout.ChromaWidth, layout.ChromaHeight)
	}
	if layout.Size != 64*9+64*5*2 {
		t.Fatalf("size=%d", layout.Size)
	}
}

func TestBindFrame(t *testing.T) {
	format := Format{
		Width:        17,
		Height:       9,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, layout.Size)

	frame, err := Bind(buffer, format)
	if err != nil {
		t.Fatal(err)
	}
	if frame.Layout.Size != layout.Size || len(frame.Y.Pix)+len(frame.U.Pix)+len(frame.V.Pix) != layout.Size {
		t.Fatalf("frame=%+v layout=%+v", frame, layout)
	}
	if frame.Y.Width != 17 || frame.U.Width != 9 || frame.V.Height != 5 {
		t.Fatalf("planes: Y=%+v U=%+v V=%+v", frame.Y, frame.U, frame.V)
	}
}

func TestBindRejectsShortBuffer(t *testing.T) {
	_, err := Bind(make([]byte, 1), Format{Width: 16, Height: 16, BitDepth: 8})
	if !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("Bind err=%v want %v", err, ErrShortBuffer)
	}
}

func TestFrameBindAllocs(t *testing.T) {
	format := Format{Width: 128, Height: 72, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, layout.Size)

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := Bind(buffer, format)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Bind allocated: %f", allocs)
	}
}

func BenchmarkBindFrame(b *testing.B) {
	format := Format{Width: 1920, Height: 1080, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	buffer := make([]byte, layout.Size)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = Bind(buffer, format)
	}
}
