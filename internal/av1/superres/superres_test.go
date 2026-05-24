package superres

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestUpscalePlaneMatchesNormativeRows(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		src      []uint16
		dstWidth int
		want     []uint16
	}{
		{
			name:     "8-bit",
			bitDepth: 8,
			src:      []uint16{0, 32, 64, 96, 128, 160, 192, 224},
			dstWidth: 13,
			want:     []uint16{0, 11, 33, 54, 73, 92, 112, 132, 151, 171, 191, 214, 226},
		},
		{
			name:     "10-bit",
			bitDepth: 10,
			src:      []uint16{0, 512, 1023, 400},
			dstWidth: 7,
			want:     []uint16{0, 104, 450, 901, 996, 642, 330},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint16, tt.dstWidth)
			srcPlane := frame.SamplePlane{Pix: tt.src, Stride: len(tt.src), Width: len(tt.src), Height: 1}
			dstPlane := frame.SamplePlane{Pix: dst, Stride: tt.dstWidth, Width: tt.dstWidth, Height: 1}
			if err := UpscalePlane(srcPlane, dstPlane, tt.bitDepth); err != nil {
				t.Fatal(err)
			}
			for i := range tt.want {
				if dst[i] != tt.want[i] {
					t.Fatalf("dst[%d]=%d want %d full=%v", i, dst[i], tt.want[i], dst)
				}
			}
		})
	}
}

func TestUpscalePlanePreservesConstantRows(t *testing.T) {
	src := []uint16{
		100, 100, 100, 100, 100,
		200, 200, 200, 200, 200,
	}
	dst := make([]uint16, 18)
	srcPlane := frame.SamplePlane{Pix: src, Stride: 5, Width: 5, Height: 2}
	dstPlane := frame.SamplePlane{Pix: dst, Stride: 9, Width: 9, Height: 2}
	if err := UpscalePlane(srcPlane, dstPlane, 8); err != nil {
		t.Fatal(err)
	}
	for x := 0; x < 9; x++ {
		if dst[x] != 100 {
			t.Fatalf("row0 dst[%d]=%d want 100 full=%v", x, dst[x], dst[:9])
		}
		if dst[9+x] != 200 {
			t.Fatalf("row1 dst[%d]=%d want 200 full=%v", x, dst[9+x], dst[9:])
		}
	}
}

func TestUpscalePlaneRejectsInvalidInputs(t *testing.T) {
	src := frame.SamplePlane{Pix: make([]uint16, 4), Stride: 4, Width: 4, Height: 1}
	dst := frame.SamplePlane{Pix: make([]uint16, 8), Stride: 8, Width: 8, Height: 1}
	if err := UpscalePlane(src, dst, 7); !errors.Is(err, frame.ErrInvalidPlane) {
		t.Fatalf("bitdepth err=%v want %v", err, frame.ErrInvalidPlane)
	}
	if err := UpscalePlane(src, src, 8); !errors.Is(err, frame.ErrInvalidPlane) {
		t.Fatalf("same-width err=%v want %v", err, frame.ErrInvalidPlane)
	}
	dst.Stride = 3
	if err := UpscalePlane(src, dst, 8); !errors.Is(err, frame.ErrInvalidPlane) {
		t.Fatalf("stride err=%v want %v", err, frame.ErrInvalidPlane)
	}
}
