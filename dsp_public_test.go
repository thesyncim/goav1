package goav1_test

import (
	"errors"
	"slices"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDSPPlaneBlockOps(t *testing.T) {
	plane := publicPredictionPlane(7, 5, 1, 12)
	for i := range plane.Pix {
		plane.Pix[i] = 0xee
	}
	if err := av1.FillPlaneBlock(plane, 1, 2, 1, 3, 2, 0x2a); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			got := getPublicFrameSample(plane, 1, x, y)
			want := uint16(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = 0x2a
			}
			if got != want {
				t.Fatalf("fill sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}

	src := publicPredictionPlane(6, 5, 2, 16)
	dst := publicPredictionPlane(7, 4, 2, 16)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setPublicFrameSample(src, 2, x, y, uint16(1000+y*10+x))
		}
	}
	if err := av1.CopyPlaneBlock(dst, src, 2, 2, 1, 1, 2, 3, 2); err != nil {
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
	if err := av1.FillPlaneBlock(residualPlane, 1, 0, 0, 4, 2, 100); err != nil {
		t.Fatal(err)
	}
	residual := []int16{
		-200, -1, 0, 200,
		20, -20, 155, 156,
	}
	if err := av1.AddResidualPlaneBlock(residualPlane, 1, 8, 0, 0, 4, 2, residual, 4); err != nil {
		t.Fatal(err)
	}
	want := []uint16{0, 99, 100, 255, 120, 80, 255, 255}
	for y := range 2 {
		for x := range 4 {
			if got := getPublicFrameSample(residualPlane, 1, x, y); got != want[y*4+x] {
				t.Fatalf("residual sample(%d,%d)=%d want %d", x, y, got, want[y*4+x])
			}
		}
	}
}

func TestPublicDSPBlendAndMinMax(t *testing.T) {
	src0 := []uint16{
		0, 64, 128, 255,
		10, 20, 30, 40,
		200, 180, 160, 140,
		255, 0, 255, 0,
	}
	src1 := []uint16{
		255, 128, 64, 0,
		40, 30, 20, 10,
		0, 32, 64, 96,
		0, 255, 0, 255,
	}
	mask := []uint8{
		64, 32, 0, 16,
		48, 24, 8, 56,
		1, 63, 31, 33,
		0, 64, 32, 32,
	}
	dst := make([]uint16, 16)
	if err := av1.BlendA64Mask(dst, 4, src0, 4, src1, 4, mask, 4, 4, 4, false, false, 8); err != nil {
		t.Fatal(err)
	}
	want := []uint16{
		0, 96, 64, 64,
		18, 26, 21, 36,
		3, 178, 111, 119,
		0, 0, 128, 128,
	}
	if !slices.Equal(dst, want) {
		t.Fatalf("blend dst=%v want %v", dst, want)
	}

	var a [64]byte
	var b [64]byte
	for i := range b {
		b[i] = 255
	}
	b[17] = 7
	minDiff, maxDiff, err := av1.MinMaxAbsDiff8x8(a[:], 8, b[:], 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	if minDiff != 7 || maxDiff != 255 {
		t.Fatalf("8-bit min/max=%d/%d want 7/255", minDiff, maxDiff)
	}

	var a16 [64 * 2]byte
	var b16 [64 * 2]byte
	for i := range 64 {
		publicPut16(b16[:], i, 65535)
	}
	publicPut16(b16[:], 9, 13)
	minDiff, maxDiff, err = av1.MinMaxAbsDiff8x8(a16[:], 16, b16[:], 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	if minDiff != 13 || maxDiff != 65535 {
		t.Fatalf("16-bit min/max=%d/%d want 13/65535", minDiff, maxDiff)
	}
}

func TestPublicDSPRejectsInvalid(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := av1.FillPlaneBlock(plane, 3, 0, 0, 1, 1, 0); !errors.Is(err, av1.ErrDSPInvalidBlock) {
		t.Fatalf("FillPlaneBlock err=%v want %v", err, av1.ErrDSPInvalidBlock)
	}
	if err := av1.FillPlaneBlock(plane, 1, 0, 0, 1, 1, 256); !errors.Is(err, av1.ErrDSPInvalidBlock) {
		t.Fatalf("FillPlaneBlock value err=%v want %v", err, av1.ErrDSPInvalidBlock)
	}
	if err := av1.CopyPlaneBlock(plane, plane, 1, 3, 0, 0, 0, 2, 1); !errors.Is(err, av1.ErrDSPInvalidBlock) {
		t.Fatalf("CopyPlaneBlock err=%v want %v", err, av1.ErrDSPInvalidBlock)
	}
	if err := av1.AddResidualPlaneBlock(plane, 1, 10, 0, 0, 1, 1, []int16{0}, 1); !errors.Is(err, av1.ErrDSPInvalidBlock) {
		t.Fatalf("AddResidualPlaneBlock err=%v want %v", err, av1.ErrDSPInvalidBlock)
	}
	if err := av1.BlendA64Mask(make([]uint16, 16), 4, make([]uint16, 16), 4, make([]uint16, 16), 4, make([]uint8, 16), 4, 3, 4, false, false, 8); !errors.Is(err, av1.ErrDSPInvalidBlock) {
		t.Fatalf("BlendA64Mask err=%v want %v", err, av1.ErrDSPInvalidBlock)
	}
	if _, _, err := av1.MinMaxAbsDiff8x8(make([]byte, 64), 7, make([]byte, 64), 8, 1); !errors.Is(err, av1.ErrDSPInvalidBlock) {
		t.Fatalf("MinMaxAbsDiff8x8 err=%v want %v", err, av1.ErrDSPInvalidBlock)
	}
}

func TestPublicDSPAllocs(t *testing.T) {
	src := publicPredictionPlane(16, 16, 1, 32)
	dst := publicPredictionPlane(16, 16, 1, 32)
	residual := make([]int16, 16*16)
	blendDst := make([]uint16, 64*64)
	blend0 := make([]uint16, 64*64)
	blend1 := make([]uint16, 64*64)
	blendMask := make([]uint8, 64*64)
	var minmaxA [8 * 64]byte
	var minmaxB [8 * 64]byte
	for i := range blendMask {
		blendMask[i] = 32
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := av1.FillPlaneBlock(src, 1, 0, 0, 16, 16, 91); err != nil {
			t.Fatalf("FillPlaneBlock err=%v", err)
		}
		if err := av1.CopyPlaneBlock(dst, src, 1, 0, 0, 0, 0, 16, 16); err != nil {
			t.Fatalf("CopyPlaneBlock err=%v", err)
		}
		if err := av1.AddResidualPlaneBlock(dst, 1, 8, 0, 0, 16, 16, residual, 16); err != nil {
			t.Fatalf("AddResidualPlaneBlock err=%v", err)
		}
		if err := av1.BlendA64Mask(blendDst, 64, blend0, 64, blend1, 64, blendMask, 64, 64, 64, false, false, 12); err != nil {
			t.Fatalf("BlendA64Mask err=%v", err)
		}
		if _, _, err := av1.MinMaxAbsDiff8x8(minmaxA[:], 64, minmaxB[:], 64, 1); err != nil {
			t.Fatalf("MinMaxAbsDiff8x8 err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicPut16(dst []byte, index int, value uint16) {
	dst[index*2] = byte(value)
	dst[index*2+1] = byte(value >> 8)
}
