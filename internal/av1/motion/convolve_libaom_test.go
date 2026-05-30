package motion

import (
	"os"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

const (
	libaomInputPadding = 12
	libaomInputOrigin  = libaomInputPadding / 2
)

type libaomConvolveBlockSize struct {
	width  int
	height int
}

func TestLibaomConvolveParametersPort(t *testing.T) {
	sizes := libaomConvolveTestSizes()
	if len(sizes) != 27 {
		t.Fatalf("len(sizes)=%d want 27", len(sizes))
	}
	if sizes[0] != (libaomConvolveBlockSize{width: 2, height: 2}) {
		t.Fatalf("first size=%+v want 2x2", sizes[0])
	}
	if sizes[len(sizes)-1] != (libaomConvolveBlockSize{width: 128, height: 128}) {
		t.Fatalf("last size=%+v want 128x128", sizes[len(sizes)-1])
	}
}

func TestLibaomConvolveXRunTestPort(t *testing.T) {
	for _, size := range libaomConvolveTestSizes() {
		for subX := 1; subX < 16; subX++ {
			for _, filter := range libaomConvolveFilters() {
				runLibaomConvolveXCase(t, size, subX, filter)
			}
		}
	}
}

func TestLibaomConvolveYRunTestPort(t *testing.T) {
	for _, size := range libaomConvolveTestSizes() {
		for subY := 1; subY < 16; subY++ {
			for _, filter := range libaomConvolveFilters() {
				runLibaomConvolveYCase(t, size, subY, filter)
			}
		}
	}
}

func TestLibaomConvolve2DQuickPort(t *testing.T) {
	for _, size := range []libaomConvolveBlockSize{
		{width: 2, height: 2},
		{width: 4, height: 16},
		{width: 8, height: 8},
		{width: 16, height: 64},
		{width: 32, height: 16},
		{width: 64, height: 64},
		{width: 128, height: 128},
	} {
		for _, subX := range []int{1, 2, 7, 8, 15} {
			for _, subY := range []int{1, 2, 7, 8, 15} {
				for _, filters := range []InterpFilters{
					{X: InterpEightTapRegular, Y: InterpEightTapRegular},
					{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
					{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
					{X: InterpBilinear, Y: InterpBilinear},
					{X: InterpEightTapRegular, Y: InterpEightTapSmooth},
					{X: InterpMultiTapSharp, Y: InterpBilinear},
				} {
					runLibaomConvolve2DCase(t, size, subX, subY, filters)
				}
			}
		}
	}
}

func TestLibaomConvolve2DRunTestPort(t *testing.T) {
	if os.Getenv("GOAV1_FULL_LIBAOM_CONVOLVE") == "" {
		t.Skip("set GOAV1_FULL_LIBAOM_CONVOLVE=1 to run the full libaom 2D convolve matrix")
	}
	for _, size := range libaomConvolveTestSizes() {
		for subX := 1; subX < 16; subX++ {
			for subY := 1; subY < 16; subY++ {
				for _, filterX := range libaomConvolveFilters() {
					for _, filterY := range libaomConvolveFilters() {
						runLibaomConvolve2DCase(t, size, subX, subY, InterpFilters{X: filterX, Y: filterY})
					}
				}
			}
		}
	}
}

func runLibaomConvolveXCase(t *testing.T, size libaomConvolveBlockSize, subX int, filter InterpFilter) {
	t.Helper()
	src := libaomConvolveInput8(size)
	got, _ := testPlane(size.width, size.height, 1, size.width)
	want, _ := testPlane(size.width, size.height, 1, size.width)
	kernel, err := interpKernel(filter, size.width, subX)
	if err != nil {
		t.Fatalf("x %dx%d sub=%d filter=%d: %v", size.width, size.height, subX, filter, err)
	}
	convolveX8Impl(got, src, 0, 0, libaomInputOrigin, libaomInputOrigin, size.width, size.height, kernel)
	libaomConvolveXRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, size.width, size.height, kernel)
	assertLibaomConvolveEqual(t, "x", got, want, size, subX, 0, InterpFilters{X: filter})
}

func runLibaomConvolveYCase(t *testing.T, size libaomConvolveBlockSize, subY int, filter InterpFilter) {
	t.Helper()
	src := libaomConvolveInput8(size)
	got, _ := testPlane(size.width, size.height, 1, size.width)
	want, _ := testPlane(size.width, size.height, 1, size.width)
	kernel, err := interpKernel(filter, size.height, subY)
	if err != nil {
		t.Fatalf("y %dx%d sub=%d filter=%d: %v", size.width, size.height, subY, filter, err)
	}
	convolveY8Impl(got, src, 0, 0, libaomInputOrigin, libaomInputOrigin, size.width, size.height, kernel)
	libaomConvolveYRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, size.width, size.height, kernel)
	assertLibaomConvolveEqual(t, "y", got, want, size, 0, subY, InterpFilters{Y: filter})
}

func runLibaomConvolve2DCase(t *testing.T, size libaomConvolveBlockSize, subX int, subY int, filters InterpFilters) {
	t.Helper()
	src := libaomConvolveInput8(size)
	got, _ := testPlane(size.width, size.height, 1, size.width)
	want, _ := testPlane(size.width, size.height, 1, size.width)
	xKernel, err := interpKernel(filters.X, size.width, subX)
	if err != nil {
		t.Fatalf("2d %dx%d sub=%d,%d filters=%d,%d: %v", size.width, size.height, subX, subY, filters.X, filters.Y, err)
	}
	yKernel, err := interpKernel(filters.Y, size.height, subY)
	if err != nil {
		t.Fatalf("2d %dx%d sub=%d,%d filters=%d,%d: %v", size.width, size.height, subX, subY, filters.X, filters.Y, err)
	}
	convolve2D8Impl(got, src, 0, 0, libaomInputOrigin, libaomInputOrigin, size.width, size.height, xKernel, yKernel)
	libaomConvolve2DRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, size.width, size.height, xKernel, yKernel)
	assertLibaomConvolveEqual(t, "2d", got, want, size, subX, subY, filters)
}

func assertLibaomConvolveEqual(t *testing.T, name string, got frame.Plane, want frame.Plane, size libaomConvolveBlockSize, subX int, subY int, filters InterpFilters) {
	t.Helper()
	for y := 0; y < size.height; y++ {
		for x := 0; x < size.width; x++ {
			g := got.Pix[y*got.Stride+x]
			w := want.Pix[y*want.Stride+x]
			if g != w {
				t.Fatalf("%s %dx%d sub=%d,%d filters=%d,%d sample(%d,%d)=%d want %d", name, size.width, size.height, subX, subY, filters.X, filters.Y, x, y, g, w)
			}
		}
	}
}

func libaomConvolveInput8(size libaomConvolveBlockSize) frame.Plane {
	plane, _ := testPlane(size.width+libaomInputPadding, size.height+libaomInputPadding, 1, size.width+libaomInputPadding)
	var state uint32 = 0xbaba
	for i := range plane.Pix {
		state = state*1664525 + 1013904223
		plane.Pix[i] = byte(state >> 24)
	}
	return plane
}

func libaomConvolveFilters() []InterpFilter {
	return []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
}

func libaomConvolveTestSizes() []libaomConvolveBlockSize {
	return []libaomConvolveBlockSize{
		{width: 2, height: 2},
		{width: 2, height: 4},
		{width: 2, height: 8},
		{width: 4, height: 2},
		{width: 4, height: 4},
		{width: 4, height: 8},
		{width: 4, height: 16},
		{width: 8, height: 2},
		{width: 8, height: 4},
		{width: 8, height: 8},
		{width: 8, height: 16},
		{width: 8, height: 32},
		{width: 16, height: 4},
		{width: 16, height: 8},
		{width: 16, height: 16},
		{width: 16, height: 32},
		{width: 16, height: 64},
		{width: 32, height: 8},
		{width: 32, height: 16},
		{width: 32, height: 32},
		{width: 32, height: 64},
		{width: 64, height: 16},
		{width: 64, height: 32},
		{width: 64, height: 64},
		{width: 64, height: 128},
		{width: 128, height: 64},
		{width: 128, height: 128},
	}
}
