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
	refX, refY, subX, subY, err := ReferenceOrigin(5, 6, Vector{Col: -1, Row: 5})
	if err != nil {
		t.Fatal(err)
	}
	if refX != 4 || refY != 6 || subX != 14 || subY != 10 {
		t.Fatalf("fractional origin=%d,%d sub=%d,%d want 4,6 sub=14,10", refX, refY, subX, subY)
	}
	if _, _, err := (Vector{Col: 4}).FullpelOffset(); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("fractional offset err=%v want %v", err, ErrInvalidMotion)
	}
}

func TestLowerPrecisionMatchesLibaom(t *testing.T) {
	tests := []struct {
		name               string
		in                 Vector
		allowHighPrecision bool
		forceInteger       bool
		want               Vector
	}{
		{name: "high precision keeps eighth pel", in: Vector{Row: 3, Col: -5}, allowHighPrecision: true, want: Vector{Row: 3, Col: -5}},
		{name: "low precision moves odd components toward zero", in: Vector{Row: 3, Col: -3}, want: Vector{Row: 2, Col: -2}},
		{name: "low precision keeps even components", in: Vector{Row: 6, Col: -4}, want: Vector{Row: 6, Col: -4}},
		{name: "integer rounds positive over half", in: Vector{Row: 5, Col: 12}, allowHighPrecision: true, forceInteger: true, want: Vector{Row: 8, Col: 8}},
		{name: "integer rounds negative over half", in: Vector{Row: -5, Col: -12}, forceInteger: true, want: Vector{Row: -8, Col: -8}},
		{name: "integer ties round toward zero", in: Vector{Row: 4, Col: -4}, forceInteger: true, want: Vector{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LowerPrecision(tt.in, tt.allowHighPrecision, tt.forceInteger)
			if got != tt.want {
				t.Fatalf("LowerPrecision(%+v)=%+v want %+v", tt.in, got, tt.want)
			}
		})
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

func TestInterpFilterTablesMatchLibaom(t *testing.T) {
	tests := []struct {
		name      string
		filter    InterpFilter
		blockSize int
		subpel    int
		want      [filterTaps]int16
	}{
		{name: "regular 8 tap", filter: InterpEightTapRegular, blockSize: 8, subpel: 1, want: [filterTaps]int16{0, 2, -6, 126, 8, -2, 0, 0}},
		{name: "smooth 8 tap", filter: InterpEightTapSmooth, blockSize: 8, subpel: 8, want: [filterTaps]int16{0, -2, 14, 52, 52, 14, -2, 0}},
		{name: "sharp 8 tap", filter: InterpMultiTapSharp, blockSize: 8, subpel: 15, want: [filterTaps]int16{0, 2, -2, 8, 126, -6, 2, -2}},
		{name: "bilinear", filter: InterpBilinear, blockSize: 4, subpel: 8, want: [filterTaps]int16{0, 0, 0, 64, 64, 0, 0, 0}},
		{name: "regular 4 tap", filter: InterpEightTapRegular, blockSize: 4, subpel: 4, want: [filterTaps]int16{0, 0, -12, 110, 38, -8, 0, 0}},
		{name: "sharp uses regular 4 tap", filter: InterpMultiTapSharp, blockSize: 4, subpel: 4, want: [filterTaps]int16{0, 0, -12, 110, 38, -8, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := interpKernel(tt.filter, tt.blockSize, tt.subpel)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("kernel=%v want %v", got, tt.want)
			}
		})
	}
}

func TestPredictInterPlaneBlockFractionalXMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: 4, Row: 0}
	if err := PredictInterPlaneBlockWithFilter(dst, src, 1, 8, 8, 8, 8, mv, RegularFilters); err != nil {
		t.Fatal(err)
	}
	kernel, err := interpKernel(InterpEightTapRegular, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolveXRef(want, src, 8, 8, 8, 8, 8, 8, kernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockFractionalYMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: 0, Row: 6}
	filters := InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}
	if err := PredictInterPlaneBlockWithFilter(dst, src, 1, 8, 8, 8, 8, mv, filters); err != nil {
		t.Fatal(err)
	}
	kernel, err := interpKernel(InterpEightTapSmooth, 8, 12)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolveYRef(want, src, 8, 8, 8, 8, 8, 8, kernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockFractional2DMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: 3, Row: 5}
	filters := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}
	if err := PredictInterPlaneBlockWithFilter(dst, src, 1, 8, 8, 8, 8, mv, filters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(InterpMultiTapSharp, 8, 6)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(InterpEightTapSmooth, 8, 10)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolve2DRef(want, src, 8, 8, 8, 8, 8, 8, xKernel, yKernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockNegativeFractionalMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: -1, Row: -3}
	if err := PredictInterPlaneBlockWithFilter(dst, src, 1, 8, 8, 8, 8, mv, RegularFilters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(InterpEightTapRegular, 8, 14)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(InterpEightTapRegular, 8, 10)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolve2DRef(want, src, 7, 7, 8, 8, 8, 8, xKernel, yKernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockHighBitDepthFractionalMatchesLibaom(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		mv       Vector
		filters  InterpFilters
		ref      func(dst frame.Plane, src frame.Plane, max uint16, kernelX [filterTaps]int16, kernelY [filterTaps]int16)
	}{
		{
			name:     "x",
			bitDepth: 10,
			mv:       Vector{Col: 4, Row: 0},
			filters:  RegularFilters,
			ref: func(dst frame.Plane, src frame.Plane, max uint16, kernelX [filterTaps]int16, _ [filterTaps]int16) {
				libaomHighBDConvolveXRef(dst, src, max, 8, 8, 8, 8, 8, 8, kernelX)
			},
		},
		{
			name:     "y",
			bitDepth: 12,
			mv:       Vector{Col: 0, Row: 6},
			filters:  InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth},
			ref: func(dst frame.Plane, src frame.Plane, max uint16, _ [filterTaps]int16, kernelY [filterTaps]int16) {
				libaomHighBDConvolveYRef(dst, src, max, 8, 8, 8, 8, 8, 8, kernelY)
			},
		},
		{
			name:     "2d",
			bitDepth: 12,
			mv:       Vector{Col: 3, Row: 5},
			filters:  InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth},
			ref: func(dst frame.Plane, src frame.Plane, max uint16, kernelX [filterTaps]int16, kernelY [filterTaps]int16) {
				libaomHighBDConvolve2DRef(dst, src, 12, max, 8, 8, 8, 8, 8, 8, kernelX, kernelY)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src, _ := testPlane(32, 32, 2, 64)
			dst, _ := testPlane(32, 32, 2, 64)
			want, _ := testPlane(32, 32, 2, 64)
			max := uint16((1 << tt.bitDepth) - 1)
			fillHighBDMotionTestPlane(src, max)
			if err := PredictInterPlaneBlockWithFilterBitDepth(dst, src, 2, tt.bitDepth, 8, 8, 8, 8, tt.mv, tt.filters); err != nil {
				t.Fatal(err)
			}
			_, _, subX, subY, err := referenceOrigin(8, 8, tt.mv)
			if err != nil {
				t.Fatal(err)
			}
			xKernel, err := interpKernel(tt.filters.X, 8, subX)
			if err != nil {
				t.Fatal(err)
			}
			yKernel, err := interpKernel(tt.filters.Y, 8, subY)
			if err != nil {
				t.Fatal(err)
			}
			tt.ref(want, src, max, xKernel, yKernel)
			assertPlaneBlockEqual(t, dst, want, 2, 8, 8, 8, 8)
		})
	}
}

func TestPredictInterPlaneBlockFromOriginMatchesVectorPath(t *testing.T) {
	tests := []struct {
		name     string
		bps      int
		bitDepth uint8
		mv       Vector
		filters  InterpFilters
	}{
		{
			name:    "lowbd",
			bps:     1,
			mv:      Vector{Col: 3, Row: 5},
			filters: InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth},
		},
		{
			name:     "highbd",
			bps:      2,
			bitDepth: 10,
			mv:       Vector{Col: -1, Row: 6},
			filters:  InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src, _ := testPlane(32, 32, tt.bps, 32*tt.bps)
			vectorDst, _ := testPlane(32, 32, tt.bps, 32*tt.bps)
			scratchDst, _ := testPlane(8, 8, tt.bps, 8*tt.bps)
			if tt.bps == 1 {
				fillMotionTestPlane(src)
				if err := PredictInterPlaneBlockWithFilter(vectorDst, src, 1, 8, 8, 8, 8, tt.mv, tt.filters); err != nil {
					t.Fatal(err)
				}
			} else {
				fillHighBDMotionTestPlane(src, 0x3ff)
				if err := PredictInterPlaneBlockWithFilterBitDepth(vectorDst, src, 2, tt.bitDepth, 8, 8, 8, 8, tt.mv, tt.filters); err != nil {
					t.Fatal(err)
				}
			}
			refX, refY, subX, subY, err := referenceOrigin(8, 8, tt.mv)
			if err != nil {
				t.Fatal(err)
			}
			if tt.bps == 1 {
				err = PredictInterPlaneBlockFromOriginWithFilter(scratchDst, src, 1, 0, 0, refX, refY, 8, 8, subX, subY, tt.filters)
			} else {
				err = PredictInterPlaneBlockFromOriginWithFilterBitDepth(scratchDst, src, 2, tt.bitDepth, 0, 0, refX, refY, 8, 8, subX, subY, tt.filters)
			}
			if err != nil {
				t.Fatal(err)
			}
			assertPlaneBlocksEqualAt(t, scratchDst, 0, 0, vectorDst, 8, 8, tt.bps, 8, 8)
		})
	}
}

func TestPredictInterPlaneBlockFromOriginLibaomConvolvePort(t *testing.T) {
	tests := []struct {
		name    string
		size    libaomConvolveBlockSize
		subX    int
		subY    int
		filters InterpFilters
	}{
		{name: "x", size: libaomConvolveBlockSize{width: 4, height: 16}, subX: 7, filters: InterpFilters{X: InterpEightTapSmooth, Y: InterpEightTapRegular}},
		{name: "y", size: libaomConvolveBlockSize{width: 16, height: 4}, subY: 11, filters: InterpFilters{X: InterpEightTapRegular, Y: InterpBilinear}},
		{name: "2d-small", size: libaomConvolveBlockSize{width: 8, height: 8}, subX: 3, subY: 5, filters: InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}},
		{name: "2d-wide", size: libaomConvolveBlockSize{width: 64, height: 32}, subX: 15, subY: 1, filters: InterpFilters{X: InterpBilinear, Y: InterpMultiTapSharp}},
		{name: "2d-max", size: libaomConvolveBlockSize{width: 128, height: 128}, subX: 8, subY: 8, filters: InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := libaomConvolveInput8(tt.size)
			got, _ := testPlane(tt.size.width, tt.size.height, 1, tt.size.width)
			want, _ := testPlane(tt.size.width, tt.size.height, 1, tt.size.width)
			if err := PredictInterPlaneBlockFromOriginWithFilter(got, src, 1, 0, 0, libaomInputOrigin, libaomInputOrigin, tt.size.width, tt.size.height, tt.subX, tt.subY, tt.filters); err != nil {
				t.Fatal(err)
			}
			xKernel, err := interpKernel(tt.filters.X, tt.size.width, tt.subX)
			if err != nil {
				t.Fatal(err)
			}
			yKernel, err := interpKernel(tt.filters.Y, tt.size.height, tt.subY)
			if err != nil {
				t.Fatal(err)
			}
			switch {
			case tt.subX != 0 && tt.subY != 0:
				libaomConvolve2DRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, tt.size.width, tt.size.height, xKernel, yKernel)
			case tt.subX != 0:
				libaomConvolveXRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, tt.size.width, tt.size.height, xKernel)
			case tt.subY != 0:
				libaomConvolveYRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, tt.size.width, tt.size.height, yKernel)
			}
			assertLibaomConvolveEqual(t, tt.name, got, want, tt.size, tt.subX, tt.subY, tt.filters)
		})
	}
}

func TestPredictInterPlaneBlockFromOriginFullpelHighBitDepth(t *testing.T) {
	src, _ := testPlane(12, 12, 2, 24)
	got, _ := testPlane(4, 3, 2, 8)
	fillHighBDMotionTestPlane(src, 0xfff)

	if err := PredictInterPlaneBlockFromOrigin(got, src, 2, 0, 0, 3, 4, 4, 3, 0, 0); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 3; y++ {
		for x := 0; x < 4; x++ {
			if g, w := getSample(got, 2, x, y), getSample(src, 2, 3+x, 4+y); g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, g, w)
			}
		}
	}
}

func TestPredictInterPlaneBlockFromOriginRejectsInvalidInputs(t *testing.T) {
	src, _ := testPlane(16, 16, 1, 16)
	dst, _ := testPlane(8, 8, 1, 8)
	if err := PredictInterPlaneBlockFromOriginWithFilter(dst, src, 1, 0, 0, 4, 4, 4, 4, 16, 0, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad subX err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockFromOriginWithFilter(dst, src, 1, 0, 0, 4, 4, 4, 4, 0, -1, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad subY err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockFromOriginWithFilter(dst, src, 1, 0, 0, 4, 4, 4, 4, 0, 0, InterpFilters{X: interpFilterCount}); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad filter err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockFromOrigin(dst, src, 1, 0, 0, -1, 0, 4, 4, 0, 0); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("outside ref err=%v want %v", err, ErrInvalidMotion)
	}

	srcHigh, _ := testPlane(16, 16, 2, 32)
	dstHigh, _ := testPlane(8, 8, 2, 16)
	if err := PredictInterPlaneBlockFromOriginWithFilter(dstHigh, srcHigh, 2, 0, 0, 4, 4, 4, 4, 2, 0, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("implicit highbd fractional err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstHigh, srcHigh, 2, 8, 0, 0, 4, 4, 4, 4, 0, 0, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad highbd bitdepth err=%v want %v", err, ErrInvalidMotion)
	}
}

func TestPredictInterPlaneBlockFromOriginAllocs(t *testing.T) {
	t.Run("lowbd", func(t *testing.T) {
		src, _ := testPlane(32, 32, 1, 32)
		dst, _ := testPlane(8, 8, 1, 8)
		fillMotionTestPlane(src)
		allocs := testing.AllocsPerRun(1000, func() {
			if err := PredictInterPlaneBlockFromOriginWithFilter(dst, src, 1, 0, 0, 8, 8, 8, 8, 3, 5, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("explicit-origin inter prediction allocated: %f", allocs)
		}
	})
	t.Run("highbd", func(t *testing.T) {
		src, _ := testPlane(32, 32, 2, 64)
		dst, _ := testPlane(8, 8, 2, 16)
		fillHighBDMotionTestPlane(src, 0x3ff)
		allocs := testing.AllocsPerRun(1000, func() {
			if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 2, 10, 0, 0, 8, 8, 8, 8, 3, 5, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("explicit-origin highbd inter prediction allocated: %f", allocs)
		}
	})
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
	if err := PredictInterPlaneBlock(dst, src, 2, 1, 1, 1, 1, Vector{Col: 4}); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("highbd fractional err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockWithFilter(dst, src, 1, 1, 1, 1, 1, Vector{Col: 4}, InterpFilters{X: interpFilterCount}); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad filter err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockWithFilterBitDepth(dst, src, 2, 8, 1, 1, 1, 1, Vector{Col: 4}, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad highbd bitdepth err=%v want %v", err, ErrInvalidMotion)
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

func TestPredictInterPlaneBlockFractionalAllocs(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)
	mv := Vector{Col: 3, Row: 5}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictInterPlaneBlockWithFilter(dst, src, 1, 8, 8, 8, 8, mv, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("fractional inter prediction allocated: %f", allocs)
	}
}

func TestPredictInterPlaneBlockHighBitDepthFractionalAllocs(t *testing.T) {
	src, _ := testPlane(32, 32, 2, 64)
	dst, _ := testPlane(32, 32, 2, 64)
	fillHighBDMotionTestPlane(src, 0x3ff)
	mv := Vector{Col: 3, Row: 5}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictInterPlaneBlockWithFilterBitDepth(dst, src, 2, 10, 8, 8, 8, 8, mv, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("highbd fractional inter prediction allocated: %f", allocs)
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

func FuzzPredictInterPlaneBlockFromOriginFullpel(f *testing.F) {
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

		if err := PredictInterPlaneBlockFromOrigin(dst, src, bytesPerSample, dstX, dstY, refX, refY, blockW, blockH, 0, 0); err != nil {
			t.Fatalf("PredictInterPlaneBlockFromOrigin err=%v", err)
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

func BenchmarkPredictInterPlaneBlockFractional(b *testing.B) {
	src, _ := testPlane(64, 64, 1, 64)
	dst, _ := testPlane(64, 64, 1, 64)
	fillMotionTestPlane(src)
	mv := Vector{Col: 3, Row: 5}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = PredictInterPlaneBlockWithFilter(dst, src, 1, 8, 8, 32, 32, mv, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth})
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

func fillMotionTestPlane(plane frame.Plane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte((x*x + 3*y*y + 17*x + 11*y) & 0xff)
		}
	}
}

func fillHighBDMotionTestPlane(plane frame.Plane, max uint16) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 2, x, y, uint16((x*x+3*y*y+17*x+11*y)&int(max)))
		}
	}
}

func assertPlaneBlockEqual(t *testing.T, got frame.Plane, want frame.Plane, bytesPerSample int, x int, y int, width int, height int) {
	t.Helper()
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			g := getSample(got, bytesPerSample, x+col, y+row)
			w := getSample(want, bytesPerSample, x+col, y+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, w)
			}
		}
	}
}

func assertPlaneBlocksEqualAt(t *testing.T, got frame.Plane, gotX int, gotY int, want frame.Plane, wantX int, wantY int, bytesPerSample int, width int, height int) {
	t.Helper()
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			g := getSample(got, bytesPerSample, gotX+col, gotY+row)
			w := getSample(want, bytesPerSample, wantX+col, wantY+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want sample(%d,%d)=%d", gotX+col, gotY+row, g, wantX+col, wantY+row, w)
			}
		}
	}
}

func libaomHighBDConvolveXRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 0
			for k := 0; k < filterTaps; k++ {
				sum += int(kernel[k]) * int(getSample(src, 2, refX+x-fo+k, refY+y))
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, filterBits-round0Bits), max))
		}
	}
}

func libaomHighBDConvolveYRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 0
			for k := 0; k < filterTaps; k++ {
				sum += int(kernel[k]) * int(getSample(src, 2, refX+x, refY+y-fo+k))
			}
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func libaomHighBDConvolve2DRef(dst frame.Plane, src frame.Plane, bitDepth uint8, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	for y := 0; y < height+filterTaps-1; y++ {
		for x := 0; x < width; x++ {
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := 0; k < filterTaps; k++ {
				sum += int(xKernel[k]) * int(getSample(src, 2, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(libaomRoundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 1 << offsetBits
			for k := 0; k < filterTaps; k++ {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1Bits) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), max))
		}
	}
}

func libaomConvolveXRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 0
			for k := 0; k < filterTaps; k++ {
				sum += int(kernel[k]) * int(src.Pix[(refY+y)*src.Stride+refX+x-fo+k])
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func libaomConvolveYRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 0
			for k := 0; k < filterTaps; k++ {
				sum += int(kernel[k]) * int(src.Pix[(refY+y-fo+k)*src.Stride+refX+x])
			}
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(sum, filterBits)))
		}
	}
}

func libaomConvolve2DRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	for y := 0; y < height+filterTaps-1; y++ {
		for x := 0; x < width; x++ {
			sum := 1 << (8 + filterBits - 1)
			for k := 0; k < filterTaps; k++ {
				sum += int(xKernel[k]) * int(src.Pix[(refY-foY+y)*src.Stride+refX+x-foX+k])
			}
			im[y*imStride+x] = int16(libaomRoundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 1 << offsetBits
			for k := 0; k < filterTaps; k++ {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, bits)))
		}
	}
}

func libaomRoundPowerOfTwo(value int, bits int) int {
	if bits <= 0 {
		return value
	}
	return (value + (1 << (bits - 1))) >> bits
}

func libaomClipPixel(value int) uint8 {
	if value < 0 {
		return 0
	}
	if value > 255 {
		return 255
	}
	return uint8(value)
}

func libaomClipPixelHighBD(value int, max uint16) uint16 {
	if value < 0 {
		return 0
	}
	if value > int(max) {
		return max
	}
	return uint16(value)
}
