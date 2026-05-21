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
