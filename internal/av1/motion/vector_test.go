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

func TestReferenceOriginSubsampledMatchesLibaomPositionRule(t *testing.T) {
	tests := []struct {
		name               string
		dstX, dstY         int
		mv                 Vector
		subsamplingX       bool
		subsamplingY       bool
		wantX, wantY       int
		wantSubX, wantSubY int
		compareLumaOrigin  bool
	}{
		{
			name:              "luma matches existing origin",
			dstX:              5,
			dstY:              6,
			mv:                Vector{Col: -1, Row: 5},
			wantX:             4,
			wantY:             6,
			wantSubX:          14,
			wantSubY:          10,
			compareLumaOrigin: true,
		},
		{
			name:         "subsampled positive fractional",
			dstX:         8,
			dstY:         9,
			mv:           Vector{Col: 3, Row: 5},
			subsamplingX: true,
			subsamplingY: true,
			wantX:        8,
			wantY:        9,
			wantSubX:     3,
			wantSubY:     5,
		},
		{
			name:         "subsampled negative fractional floors",
			dstX:         8,
			dstY:         9,
			mv:           Vector{Col: -1, Row: -9},
			subsamplingX: true,
			subsamplingY: true,
			wantX:        7,
			wantY:        8,
			wantSubX:     15,
			wantSubY:     7,
		},
		{
			name:         "mixed subsampling axes",
			dstX:         4,
			dstY:         7,
			mv:           Vector{Col: 7, Row: 7},
			subsamplingY: true,
			wantX:        4,
			wantY:        7,
			wantSubX:     14,
			wantSubY:     7,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotX, gotY, gotSubX, gotSubY, err := ReferenceOriginSubsampled(tt.dstX, tt.dstY, tt.mv, tt.subsamplingX, tt.subsamplingY)
			if err != nil {
				t.Fatal(err)
			}
			if gotX != tt.wantX || gotY != tt.wantY || gotSubX != tt.wantSubX || gotSubY != tt.wantSubY {
				t.Fatalf("origin=%d,%d sub=%d,%d want %d,%d sub=%d,%d", gotX, gotY, gotSubX, gotSubY, tt.wantX, tt.wantY, tt.wantSubX, tt.wantSubY)
			}
			if tt.compareLumaOrigin {
				refX, refY, subX, subY, err := ReferenceOrigin(tt.dstX, tt.dstY, tt.mv)
				if err != nil {
					t.Fatal(err)
				}
				if gotX != refX || gotY != refY || gotSubX != subX || gotSubY != subY {
					t.Fatalf("subsampled luma origin=%d,%d sub=%d,%d want ReferenceOrigin=%d,%d sub=%d,%d", gotX, gotY, gotSubX, gotSubY, refX, refY, subX, subY)
				}
			}
		})
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
	for y := range 2 {
		for x := range 3 {
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

func TestPredictInterPlaneBlockFullpelExtendsReferenceEdges(t *testing.T) {
	src, _ := testPlane(4, 4, 1, 4)
	dst, _ := testPlane(5, 3, 1, 5)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setSample(src, 1, x, y, uint16(10*y+x))
		}
	}
	if err := PredictInterPlaneBlockFromOrigin(dst, src, 1, 0, 0, -2, 2, 5, 3, 0, 0); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 5 {
			want := getSampleClamped(src, 1, x-2, y+2)
			if got := getSample(dst, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}

	srcHigh, _ := testPlane(4, 4, 2, 8)
	dstHigh, _ := testPlane(5, 3, 2, 10)
	for y := 0; y < srcHigh.Height; y++ {
		for x := 0; x < srcHigh.Width; x++ {
			setSample(srcHigh, 2, x, y, uint16(100+10*y+x))
		}
	}
	if err := PredictInterPlaneBlockFromOrigin(dstHigh, srcHigh, 2, 0, 0, 2, -2, 5, 3, 0, 0); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 5 {
			want := getSampleClamped(srcHigh, 2, x+2, y-2)
			if got := getSample(dstHigh, 2, x, y); got != want {
				t.Fatalf("highbd sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictInterPlaneBlockFractionalExtendsReferenceEdges8Bit(t *testing.T) {
	src, _ := testPlane(8, 8, 1, 8)
	got, _ := testPlane(4, 4, 1, 4)
	want, _ := testPlane(4, 4, 1, 4)
	fillMotionTestPlane(src)

	filters := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}
	if err := PredictInterPlaneBlockFromOriginWithFilter(got, src, 1, 0, 0, 0, 0, 4, 4, 3, 5, filters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(filters.X, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(filters.Y, 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolve2DClampedRef(want, src, 0, 0, 0, 0, 4, 4, xKernel, yKernel)
	assertPlaneBlockEqual(t, got, want, 1, 0, 0, 4, 4)
}

func TestPredictInterPlaneBlockOneDimensionalFractionalExtendsReferenceEdges8Bit(t *testing.T) {
	src, _ := testPlane(8, 8, 1, 8)
	fillMotionTestPlane(src)
	filters := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}

	t.Run("x", func(t *testing.T) {
		got, _ := testPlane(4, 4, 1, 4)
		want, _ := testPlane(4, 4, 1, 4)
		if err := PredictInterPlaneBlockFromOriginWithFilter(got, src, 1, 0, 0, 0, 2, 4, 4, 3, 0, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.X, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		libaomConvolveXClampedRef(want, src, 0, 2, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 1, 0, 0, 4, 4)
	})

	t.Run("y", func(t *testing.T) {
		got, _ := testPlane(4, 4, 1, 4)
		want, _ := testPlane(4, 4, 1, 4)
		if err := PredictInterPlaneBlockFromOriginWithFilter(got, src, 1, 0, 0, 2, 0, 4, 4, 0, 5, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.Y, 4, 5)
		if err != nil {
			t.Fatal(err)
		}
		libaomConvolveYClampedRef(want, src, 2, 0, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 1, 0, 0, 4, 4)
	})
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

func TestPredictInterPlaneBlockFractionalExtendsReferenceEdgesHighBD(t *testing.T) {
	src, _ := testPlane(8, 8, 2, 16)
	got, _ := testPlane(4, 4, 2, 8)
	want, _ := testPlane(4, 4, 2, 8)
	fillHighBDMotionTestPlane(src, 0x3ff)

	filters := InterpFilters{X: InterpEightTapSmooth, Y: InterpMultiTapSharp}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 2, 10, 0, 0, 5, 5, 4, 4, 9, 11, filters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(filters.X, 4, 9)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(filters.Y, 4, 11)
	if err != nil {
		t.Fatal(err)
	}
	libaomHighBDConvolve2DClampedRef(want, src, 10, 0x3ff, 5, 5, 0, 0, 4, 4, xKernel, yKernel)
	assertPlaneBlockEqual(t, got, want, 2, 0, 0, 4, 4)
}

func TestPredictInterPlaneBlockOneDimensionalFractionalExtendsReferenceEdgesHighBD(t *testing.T) {
	src, _ := testPlane(8, 8, 2, 16)
	fillHighBDMotionTestPlane(src, 0x3ff)
	filters := InterpFilters{X: InterpEightTapSmooth, Y: InterpMultiTapSharp}

	t.Run("x", func(t *testing.T) {
		got, _ := testPlane(4, 4, 2, 8)
		want, _ := testPlane(4, 4, 2, 8)
		if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 2, 10, 0, 0, 0, 2, 4, 4, 9, 0, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.X, 4, 9)
		if err != nil {
			t.Fatal(err)
		}
		libaomHighBDConvolveXClampedRef(want, src, 0x3ff, 0, 2, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 2, 0, 0, 4, 4)
	})

	t.Run("y", func(t *testing.T) {
		got, _ := testPlane(4, 4, 2, 8)
		want, _ := testPlane(4, 4, 2, 8)
		if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 2, 10, 0, 0, 2, 0, 4, 4, 0, 11, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.Y, 4, 11)
		if err != nil {
			t.Fatal(err)
		}
		libaomHighBDConvolveYClampedRef(want, src, 0x3ff, 2, 0, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 2, 0, 0, 4, 4)
	})
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
	for y := range 3 {
		for x := range 4 {
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
	if err := PredictInterPlaneBlockFromOrigin(dst, src, 1, 0, 0, -1, 0, 4, 4, 0, 0); err != nil {
		t.Fatalf("edge-extended fullpel err=%v", err)
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
	fillMotionTestPlane(src)
	if err := PredictInterPlaneBlock(dst, src, 1, 0, 0, 1, 1, Vector{Col: 4}); err != nil {
		t.Fatalf("fractional edge-extended err=%v", err)
	}
	mv, err := FullpelVector(-1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := PredictInterPlaneBlock(dst, src, 1, 0, 0, 1, 1, mv); err != nil {
		t.Fatalf("fullpel edge-extended err=%v", err)
	}
	if got, want := getSample(dst, 1, 0, 0), getSample(src, 1, 0, 0); got != want {
		t.Fatalf("fullpel clamped sample=%d want %d", got, want)
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
	if _, err := FullpelVector(int(maxInt16)/SubpelScale+1, 0); !errors.Is(err, ErrInvalidMotion) {
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
		for y := range blockH {
			for x := range blockW {
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
		for y := range blockH {
			for x := range blockW {
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
	for b.Loop() {
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
	for b.Loop() {
		_ = PredictInterPlaneBlock(dst, src, 1, 0, 0, 64, 64, mv)
	}
}

func BenchmarkPredictInterPlaneBlockFractional(b *testing.B) {
	src, _ := testPlane(64, 64, 1, 64)
	dst, _ := testPlane(64, 64, 1, 64)
	fillMotionTestPlane(src)
	mv := Vector{Col: 3, Row: 5}
	b.ReportAllocs()
	for b.Loop() {
		_ = PredictInterPlaneBlockWithFilter(dst, src, 1, 8, 8, 32, 32, mv, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth})
	}
}

// TestConvolveHighBDIdentityKernel asserts that running the High-BD
// convolvers with the subpel=0 identity kernel ({0,0,0,128,0,0,0,0})
// reproduces the input sample byte-for-byte at bd=10 (q32) and bd=12.
//
// This pins the dispatch + rounding shape for the simplest reproducible
// case: any drift here would imply a bias-shift or final-round bug that
// libaom does not have, because libaom's
// av1_highbd_convolve_{x,y,2d}_sr_c are bit-exact identity at the
// identity kernel by construction (see av1/common/convolve.c lines
// 689-790 in libaom-v3.13.1).
func TestConvolveHighBDIdentityKernel(t *testing.T) {
	identity := [filterTaps]int16{0, 0, 0, 128, 0, 0, 0, 0}
	for _, tc := range []struct {
		name     string
		bitDepth uint8
	}{
		{"bd=10", 10},
		{"bd=12", 12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			max, ok := highBDMax(tc.bitDepth)
			if !ok {
				t.Fatalf("highBDMax(%d) not supported", tc.bitDepth)
			}
			// Sized so that the full convolve window (block + filter
			// halo) fits inside the source plane without clamping.
			const stride = 64
			src, _ := testPlane(32, 32, 2, stride)
			fillHighBDMotionTestPlane(src, max)
			t.Run("X", func(t *testing.T) {
				dst, _ := testPlane(32, 32, 2, stride)
				convolveXHighBDImpl(dst, src, tc.bitDepth, max, 8, 8, 8, 8, 8, 8, identity)
				for row := range 8 {
					for col := range 8 {
						got := getSample(dst, 2, 8+col, 8+row)
						want := getSample(src, 2, 8+col, 8+row)
						if got != want {
							t.Fatalf("X identity sample(%d,%d)=%d want %d", col, row, got, want)
						}
					}
				}
			})
			t.Run("Y", func(t *testing.T) {
				dst, _ := testPlane(32, 32, 2, stride)
				convolveYHighBDImpl(dst, src, tc.bitDepth, max, 8, 8, 8, 8, 8, 8, identity)
				for row := range 8 {
					for col := range 8 {
						got := getSample(dst, 2, 8+col, 8+row)
						want := getSample(src, 2, 8+col, 8+row)
						if got != want {
							t.Fatalf("Y identity sample(%d,%d)=%d want %d", col, row, got, want)
						}
					}
				}
			})
			t.Run("2D", func(t *testing.T) {
				dst, _ := testPlane(32, 32, 2, stride)
				convolve2DHighBDImpl(dst, src, tc.bitDepth, max, 8, 8, 8, 8, 8, 8, identity, identity)
				for row := range 8 {
					for col := range 8 {
						got := getSample(dst, 2, 8+col, 8+row)
						want := getSample(src, 2, 8+col, 8+row)
						if got != want {
							t.Fatalf("2D identity sample(%d,%d)=%d want %d", col, row, got, want)
						}
					}
				}
			})
		})
	}
}

// TestConvolveHighBDQ10ClampedPath pins the *clamped* HighBD convolvers
// (the path that fires when the convolve halo overruns the reference
// edge) against a from-scratch libaom-shaped reference. Frame 1 of the
// q32 vector exercises this path at every block whose MV pulls the
// filter halo across the top/left frame edge — any drift in the edge
// extension would surface as a wave of ±N pixel diffs along the top
// and left of the frame.
func TestConvolveHighBDQ10ClampedPath(t *testing.T) {
	const bd = uint8(10)
	max := uint16((1 << bd) - 1)
	// Use a tiny source plane so the convolve halo overruns the edge
	// at every position; this forces the *Clamped paths exclusively.
	src, _ := testPlane(6, 6, 2, 12)
	fillHighBDMotionTestPlane(src, max)
	filters := []InterpFilters{
		{X: InterpEightTapRegular, Y: InterpEightTapRegular},
		{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
		{X: InterpBilinear, Y: InterpBilinear},
	}
	sizes := []libaomConvolveBlockSize{
		{width: 4, height: 4},
		{width: 8, height: 4},
		{width: 4, height: 8},
	}
	for _, sz := range sizes {
		for _, f := range filters {
			for _, sx := range []int{1, 7, 8, 15} {
				for _, sy := range []int{1, 7, 8, 15} {
					t.Run(libaomHighBDCaseName(sz, sx, sy, f)+"_clamped", func(t *testing.T) {
						got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						xKernel, err := interpKernel(f.X, sz.width, sx)
						if err != nil {
							t.Fatal(err)
						}
						yKernel, err := interpKernel(f.Y, sz.height, sy)
						if err != nil {
							t.Fatal(err)
						}
						convolve2DHighBDClampedImpl(got, src, bd, max, 0, 0, 0, 0, sz.width, sz.height, xKernel, yKernel)
						libaomVerbatimHighBD2DSRClamped(want, src, bd, max, 0, 0, 0, 0, sz.width, sz.height, xKernel, yKernel)
						for row := 0; row < sz.height; row++ {
							for col := 0; col < sz.width; col++ {
								g := getSample(got, 2, col, row)
								w := getSample(want, 2, col, row)
								if g != w {
									t.Fatalf("clamped size=%dx%d sub=(%d,%d) filters=(%d,%d) sample(%d,%d)=%d want %d",
										sz.width, sz.height, sx, sy, f.X, f.Y, col, row, g, w)
								}
							}
						}
					})
				}
			}
		}
	}
}

// TestConvolveHighBDQ10OneDimensional pins the X-only and Y-only
// HighBD convolve paths (subX=0 vs subY=0) against a from-scratch
// libaom-shaped reference. Frame 1 of q32 has many blocks that decode
// to a one-axis MV (e.g. cell.MV = (3, 0) or (0, 5)), and those land
// on the simpler convolveXHighBD / convolveYHighBD paths rather than
// the 2D path.
func TestConvolveHighBDQ10OneDimensional(t *testing.T) {
	const bd = uint8(10)
	max := uint16((1 << bd) - 1)
	src, _ := testPlane(32, 32, 2, 64)
	fillHighBDMotionTestPlane(src, max)
	filters := []InterpFilters{
		{X: InterpEightTapRegular, Y: InterpEightTapRegular},
		{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
		{X: InterpBilinear, Y: InterpBilinear},
	}
	sizes := []libaomConvolveBlockSize{
		{width: 4, height: 4},
		{width: 8, height: 8},
		{width: 16, height: 8},
		{width: 16, height: 16},
	}
	for _, sz := range sizes {
		for _, f := range filters {
			for _, sub := range []int{1, 4, 7, 8, 11, 15} {
				t.Run("X_"+libaomHighBDCaseName(sz, sub, 0, f), func(t *testing.T) {
					got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					xKernel, err := interpKernel(f.X, sz.width, sub)
					if err != nil {
						t.Fatal(err)
					}
					convolveXHighBDImpl(got, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel)
					libaomVerbatimHighBDXSR(want, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel)
					for row := 0; row < sz.height; row++ {
						for col := 0; col < sz.width; col++ {
							g := getSample(got, 2, col, row)
							w := getSample(want, 2, col, row)
							if g != w {
								t.Fatalf("X size=%dx%d sub=%d filter=%d sample(%d,%d)=%d want %d", sz.width, sz.height, sub, f.X, col, row, g, w)
							}
						}
					}
				})
				t.Run("Y_"+libaomHighBDCaseName(sz, 0, sub, f), func(t *testing.T) {
					got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					yKernel, err := interpKernel(f.Y, sz.height, sub)
					if err != nil {
						t.Fatal(err)
					}
					convolveYHighBDImpl(got, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, yKernel)
					libaomVerbatimHighBDYSR(want, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, yKernel)
					for row := 0; row < sz.height; row++ {
						for col := 0; col < sz.width; col++ {
							g := getSample(got, 2, col, row)
							w := getSample(want, 2, col, row)
							if g != w {
								t.Fatalf("Y size=%dx%d sub=%d filter=%d sample(%d,%d)=%d want %d", sz.width, sz.height, sub, f.Y, col, row, g, w)
							}
						}
					}
				})
			}
		}
	}
}

// libaomVerbatimHighBDXSR ports libaom av1_highbd_convolve_x_sr_c.
func libaomVerbatimHighBDXSR(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, xFilter [filterTaps]int16) {
	foHoriz := filterTaps/2 - 1
	bits := filterBits - round0Bits
	for y := range h {
		for x := range w {
			res := int32(0)
			for k := range filterTaps {
				res += int32(xFilter[k]) * int32(getSample(src, 2, refX+x-foHoriz+k, refY+y))
			}
			res = int32(libaomRoundPowerOfTwo(int(res), round0Bits))
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(int(res), bits), maxVal))
		}
	}
	_ = bd
}

// libaomVerbatimHighBDYSR ports libaom av1_highbd_convolve_y_sr_c.
func libaomVerbatimHighBDYSR(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, yFilter [filterTaps]int16) {
	foVert := filterTaps/2 - 1
	for y := range h {
		for x := range w {
			res := int32(0)
			for k := range filterTaps {
				res += int32(yFilter[k]) * int32(getSample(src, 2, refX+x, refY+y-foVert+k))
			}
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(int(res), filterBits), maxVal))
		}
	}
	_ = bd
}

// libaomVerbatimHighBD2DSRClamped is the clamped counterpart of
// libaomVerbatimHighBD2DSR: every reference load goes through edge
// replication (libaom's highbd_build_mc_border equivalent).
func libaomVerbatimHighBD2DSRClamped(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, xFilter [filterTaps]int16, yFilter [filterTaps]int16) {
	const imStride = maxBlockSize
	var imBlock [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	imH := h + filterTaps - 1
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	for y := range imH {
		for x := range w {
			sum := int32(1) << (uint32(bd) + uint32(filterBits) - 1)
			for k := range filterTaps {
				sum += int32(xFilter[k]) * int32(getSampleClamped(src, 2, refX+x-foHoriz+k, refY-foVert+y))
			}
			imBlock[y*imStride+x] = int16(libaomRoundPowerOfTwo(int(sum), round0Bits))
		}
	}
	offsetBits := int(bd) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range h {
		for x := range w {
			sum := int32(1) << uint32(offsetBits)
			for k := range filterTaps {
				sum += int32(yFilter[k]) * int32(imBlock[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(int(sum), round1Bits) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), maxVal))
		}
	}
}

// TestConvolveHighBDLibaomQ10MatchesAlgorithm pins the HighBD convolvers
// against an independent libaom-shaped reference implementation written
// inline (see libaomVerbatimHighBD2DSR) for a representative grid of
// block sizes, subpel positions, and switchable interpolation filters
// at bd=10. The reference is rewritten from scratch in this test (it
// does not call the libaom*Ref helpers above) so any algebra mistake in
// the production path that the helper might also have inherited (same
// formula, same bug) shows up here as a divergence.
func TestConvolveHighBDLibaomQ10MatchesAlgorithm(t *testing.T) {
	const bd = uint8(10)
	max := uint16((1 << bd) - 1)
	sizes := []libaomConvolveBlockSize{
		{width: 4, height: 4},
		{width: 4, height: 8},
		{width: 8, height: 4},
		{width: 8, height: 8},
		{width: 8, height: 16},
		{width: 16, height: 16},
		{width: 32, height: 32},
	}
	filters := []InterpFilters{
		{X: InterpEightTapRegular, Y: InterpEightTapRegular},
		{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
		{X: InterpBilinear, Y: InterpBilinear},
		{X: InterpEightTapRegular, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpBilinear},
	}
	subpels := []int{1, 2, 4, 7, 8, 11, 13, 15}
	for _, sz := range sizes {
		for _, f := range filters {
			for _, sx := range subpels {
				for _, sy := range subpels {
					t.Run(libaomHighBDCaseName(sz, sx, sy, f), func(t *testing.T) {
						src := libaomHighBDConvolveInput(sz, max)
						got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						xKernel, err := interpKernel(f.X, sz.width, sx)
						if err != nil {
							t.Fatal(err)
						}
						yKernel, err := interpKernel(f.Y, sz.height, sy)
						if err != nil {
							t.Fatal(err)
						}
						// Production call (anchored at the input
						// origin so the convolve halo lies inside
						// the padded source plane).
						convolve2DHighBDImpl(got, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel, yKernel)
						// Independent libaom-shaped reference,
						// rewritten verbatim from libaom-v3.13.1
						// av1/common/convolve.c
						// av1_highbd_convolve_2d_sr_c.
						libaomVerbatimHighBD2DSR(want, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel, yKernel)
						for row := 0; row < sz.height; row++ {
							for col := 0; col < sz.width; col++ {
								g := getSample(got, 2, col, row)
								w := getSample(want, 2, col, row)
								if g != w {
									t.Fatalf("size=%dx%d sub=(%d,%d) filters=(%d,%d) sample(%d,%d)=%d want %d",
										sz.width, sz.height, sx, sy, f.X, f.Y, col, row, g, w)
								}
							}
						}
					})
				}
			}
		}
	}
}

func libaomHighBDCaseName(sz libaomConvolveBlockSize, sx int, sy int, f InterpFilters) string {
	return libaomHighBDFormat(sz.width) + "x" + libaomHighBDFormat(sz.height) + "_sx" + libaomHighBDFormat(sx) + "_sy" + libaomHighBDFormat(sy) + "_fx" + libaomHighBDFormat(int(f.X)) + "_fy" + libaomHighBDFormat(int(f.Y))
}

func libaomHighBDFormat(v int) string {
	digits := "0123456789"
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var buf [12]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = digits[v%10]
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func libaomHighBDConvolveInput(size libaomConvolveBlockSize, max uint16) frame.Plane {
	const padding = libaomInputPadding
	plane, _ := testPlane(size.width+padding, size.height+padding, 2, (size.width+padding)*2)
	state := uint32(0xcafe)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			state = state*1664525 + 1013904223
			setSample(plane, 2, x, y, uint16(state>>16)&max)
		}
	}
	return plane
}

// libaomVerbatimHighBD2DSR is a from-scratch port of libaom-v3.13.1
// av1/common/convolve.c av1_highbd_convolve_2d_sr_c. It intentionally
// uses libaom's exact intermediate types (int16 im_block, int32
// accumulators) so the test cross-checks the production path against
// the reference algebra rather than against a Go helper that may share
// a refactoring bug.
func libaomVerbatimHighBD2DSR(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, xFilter [filterTaps]int16, yFilter [filterTaps]int16) {
	const imStride = maxBlockSize
	var imBlock [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	imH := h + filterTaps - 1
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	// Horizontal filter — bias is (1 << (bd + FILTER_BITS - 1)).
	for y := range imH {
		for x := range w {
			sum := int32(1) << (uint32(bd) + uint32(filterBits) - 1)
			for k := range filterTaps {
				sum += int32(xFilter[k]) * int32(getSample(src, 2, refX+x-foHoriz+k, refY-foVert+y))
			}
			imBlock[y*imStride+x] = int16(libaomRoundPowerOfTwo(int(sum), round0Bits))
		}
	}
	// Vertical filter — offset_bits = bd + 2 * FILTER_BITS - round_0.
	offsetBits := int(bd) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range h {
		for x := range w {
			sum := int32(1) << uint32(offsetBits)
			for k := range filterTaps {
				sum += int32(yFilter[k]) * int32(imBlock[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(int(sum), round1Bits) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), maxVal))
		}
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

func getSampleClamped(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	if x < 0 {
		x = 0
	} else if x >= plane.Width {
		x = plane.Width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= plane.Height {
		y = plane.Height - 1
	}
	return getSample(plane, bytesPerSample, x, y)
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
	for row := range height {
		for col := range width {
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
	for row := range height {
		for col := range width {
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
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSample(src, 2, refX+x-fo+k, refY+y))
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, filterBits-round0Bits), max))
		}
	}
}

func libaomHighBDConvolveYRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSample(src, 2, refX+x, refY+y-fo+k))
			}
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func libaomHighBDConvolveXClampedRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 2, refX+x-fo+k, refY+y))
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, filterBits-round0Bits), max))
		}
	}
}

func libaomHighBDConvolveYClampedRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 2, refX+x, refY+y-fo+k))
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
	// get_conv_params_no_round bumps round_0/round_1 at bd == 12.
	round0, round1 := highBDRoundBits(bitDepth)
	for y := 0; y < height+filterTaps-1; y++ {
		for x := range width {
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(getSample(src, 2, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(libaomRoundPowerOfTwo(sum, round0))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), max))
		}
	}
}

func libaomHighBDConvolve2DClampedRef(dst frame.Plane, src frame.Plane, bitDepth uint8, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	// get_conv_params_no_round bumps round_0/round_1 at bd == 12.
	round0, round1 := highBDRoundBits(bitDepth)
	for y := 0; y < height+filterTaps-1; y++ {
		for x := range width {
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(getSampleClamped(src, 2, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(libaomRoundPowerOfTwo(sum, round0))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), max))
		}
	}
}

func libaomConvolveXRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(src.Pix[(refY+y)*src.Stride+refX+x-fo+k])
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func libaomConvolveYRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(src.Pix[(refY+y-fo+k)*src.Stride+refX+x])
			}
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(sum, filterBits)))
		}
	}
}

func libaomConvolveXClampedRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 1, refX+x-fo+k, refY+y))
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func libaomConvolveYClampedRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 1, refX+x, refY+y-fo+k))
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
		for x := range width {
			sum := 1 << (8 + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(src.Pix[(refY-foY+y)*src.Stride+refX+x-foX+k])
			}
			im[y*imStride+x] = int16(libaomRoundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, bits)))
		}
	}
}

func libaomConvolve2DClampedRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	for y := 0; y < height+filterTaps-1; y++ {
		for x := range width {
			sum := 1 << (8 + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(getSampleClamped(src, 1, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int16(libaomRoundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
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
