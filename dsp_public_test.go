package goav1

import (
	"errors"
	"testing"

	internaldsp "github.com/thesyncim/goav1/internal/av1/dsp"
)

func TestPublicDSPPlaneBlockOps(t *testing.T) {
	src := publicPredictionPlane(6, 5, 2, 16)
	dst := publicPredictionPlane(7, 4, 2, 16)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setPublicFrameSample(src, 2, x, y, uint16(1000+y*10+x))
		}
	}
	if err := internaldsp.CopyPlaneBlock(dst, src, 2, 2, 1, 1, 2, 3, 2); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getPublicFrameSample(dst, 2, 2+x, 1+y)
			want := uint16(1000 + (2+y)*10 + 1 + x)
			if got != want {
				t.Fatalf("copy sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}

	residualPlane := publicPredictionPlane(4, 2, 1, 6)
	for y := range 2 {
		for x := range 4 {
			setPublicFrameSample(residualPlane, 1, x, y, 100)
		}
	}
	residual := []int16{
		-200, -1, 0, 200,
		20, -20, 155, 156,
	}
	internaldsp.AddResidualPlaneBlockTrusted(residualPlane.Pix, residualPlane.Stride, 1, 0xff, 4, 2, residual, 4)
	want := []uint16{0, 99, 100, 255, 120, 80, 255, 255}
	for y := range 2 {
		for x := range 4 {
			if got := getPublicFrameSample(residualPlane, 1, x, y); got != want[y*4+x] {
				t.Fatalf("residual sample(%d,%d)=%d want %d", x, y, got, want[y*4+x])
			}
		}
	}
}

func TestPublicDSPRejectsInvalid(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := internaldsp.CopyPlaneBlock(plane, plane, 1, 3, 0, 0, 0, 2, 1); !errors.Is(err, internaldsp.ErrInvalidBlock) {
		t.Fatalf("CopyPlaneBlock err=%v want %v", err, internaldsp.ErrInvalidBlock)
	}
}

func TestPublicDSPAllocs(t *testing.T) {
	src := publicPredictionPlane(16, 16, 1, 32)
	dst := publicPredictionPlane(16, 16, 1, 32)
	residual := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := internaldsp.CopyPlaneBlock(dst, src, 1, 0, 0, 0, 0, 16, 16); err != nil {
			t.Fatalf("CopyPlaneBlock err=%v", err)
		}
		internaldsp.AddResidualPlaneBlockTrusted(dst.Pix, dst.Stride, 1, 0xff, 16, 16, residual, 16)
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
