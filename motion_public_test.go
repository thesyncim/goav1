package goav1

import (
	"errors"
	"testing"

	internalmotion "github.com/thesyncim/goav1/internal/av1/motion"
)

func TestPublicMotionVectorReferenceOrigin(t *testing.T) {
	mv, err := internalmotion.FullpelVector(2, -1)
	if err != nil {
		t.Fatal(err)
	}
	if mv != (internalmotion.Vector{Row: -internalmotion.SubpelScale, Col: 2 * internalmotion.SubpelScale}) {
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
	refX, refY, err := internalmotion.FullpelReferenceOrigin(5, 6, mv)
	if err != nil {
		t.Fatal(err)
	}
	if refX != 7 || refY != 5 {
		t.Fatalf("origin=%d,%d want 7,5", refX, refY)
	}

	refX, refY, subX, subY, err := internalmotion.ReferenceOrigin(5, 6, internalmotion.Vector{Col: -1, Row: 5})
	if err != nil {
		t.Fatal(err)
	}
	if refX != 4 || refY != 6 || subX != 14 || subY != 10 {
		t.Fatalf("fractional origin=%d,%d sub=%d,%d want 4,6 sub=14,10", refX, refY, subX, subY)
	}
	refX, refY, subX, subY, err = internalmotion.ReferenceOriginSubsampled(8, 9, internalmotion.Vector{Col: -1, Row: -9}, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if refX != 7 || refY != 8 || subX != 15 || subY != 7 {
		t.Fatalf("subsampled origin=%d,%d sub=%d,%d want 7,8 sub=15,7", refX, refY, subX, subY)
	}
	if got := internalmotion.LowerPrecision(internalmotion.Vector{Row: 3, Col: -3}, false, false); got != (internalmotion.Vector{Row: 2, Col: -2}) {
		t.Fatalf("lower precision=%+v want row=2 col=-2", got)
	}
}

func TestPublicPredictInterPlaneBlock8Bit(t *testing.T) {
	src := publicPredictionPlane(7, 6, 1, 9)
	dst := publicPredictionPlane(7, 6, 1, 9)
	for i := range dst.Pix {
		dst.Pix[i] = 0xee
	}
	fillPublicMotionPlane(src, 1)
	mv, err := internalmotion.FullpelVector(-1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := internalmotion.PredictInterPlaneBlock(dst, src, 1, 2, 1, 3, 2, mv); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < dst.Height; y++ {
		for x := 0; x < dst.Width; x++ {
			got := getPublicFrameSample(dst, 1, x, y)
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

func TestPublicPredictInterPlaneBlockHighBitDepth(t *testing.T) {
	src := publicPredictionPlane(6, 5, 2, 7)
	dst := publicPredictionPlane(6, 5, 2, 7)
	fillPublicMotionPlane(src, 2)
	mv, err := internalmotion.FullpelVector(1, -1)
	if err != nil {
		t.Fatal(err)
	}
	if err := internalmotion.PredictInterPlaneBlockBitDepth(dst, src, 2, 12, 1, 2, 3, 2, mv); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getPublicFrameSample(dst, 2, 1+x, 2+y)
			want := uint16(1000 + (1+y)*10 + 2 + x)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicPredictInterPlaneBlockFractionalBilinear(t *testing.T) {
	src := publicPredictionPlane(16, 16, 1, 16)
	dst := publicPredictionPlane(4, 4, 1, 4)
	fillPublicMotionPlane(src, 1)
	filters := internalmotion.InterpFilters{X: internalmotion.InterpBilinear, Y: internalmotion.InterpBilinear}
	if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilter(dst, src, 1, 0, 0, 4, 4, 3, 2, 8, 0, filters); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getPublicFrameSample(dst, 1, x, y)
			want := uint16(10*(4+y) + 5 + x)
			if got != want {
				t.Fatalf("fractional sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicMotionRejectsInvalid(t *testing.T) {
	if _, _, err := (internalmotion.Vector{Col: 4}).FullpelOffset(); !errors.Is(err, internalmotion.ErrInvalidMotion) {
		t.Fatalf("fractional offset err=%v want %v", err, internalmotion.ErrInvalidMotion)
	}
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilter(plane, plane, 1, 0, 0, 0, 0, 1, 1, 16, 0, internalmotion.InterpFilters{X: internalmotion.InterpBilinear, Y: internalmotion.InterpBilinear}); !errors.Is(err, internalmotion.ErrInvalidMotion) {
		t.Fatalf("invalid subpel err=%v want %v", err, internalmotion.ErrInvalidMotion)
	}
	if err := internalmotion.PredictInterPlaneBlockWithFilter(plane, plane, 2, 0, 0, 1, 1, internalmotion.Vector{Col: 4}, internalmotion.InterpFilters{X: internalmotion.InterpBilinear, Y: internalmotion.InterpBilinear}); !errors.Is(err, internalmotion.ErrInvalidMotion) {
		t.Fatalf("fractional highbd without bitdepth err=%v want %v", err, internalmotion.ErrInvalidMotion)
	}
}

func TestPublicMotionAllocs(t *testing.T) {
	src := publicPredictionPlane(16, 16, 1, 16)
	dst := publicPredictionPlane(16, 16, 1, 16)
	fillPublicMotionPlane(src, 1)
	mv, err := internalmotion.FullpelVector(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	filters := internalmotion.InterpFilters{X: internalmotion.InterpBilinear, Y: internalmotion.InterpBilinear}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := internalmotion.PredictInterPlaneBlock(dst, src, 1, 0, 0, 8, 8, mv); err != nil {
			t.Fatalf("PredictInterPlaneBlock err=%v", err)
		}
		if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilter(dst, src, 1, 0, 0, 4, 4, 8, 8, 8, 0, filters); err != nil {
			t.Fatalf("PredictInterPlaneBlockFromOriginWithFilter err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func fillPublicMotionPlane(plane FramePlane, bytesPerSample int) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, bytesPerSample, x, y, uint16(1000*(bytesPerSample-1)+10*y+x))
		}
	}
}
