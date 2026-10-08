//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// TestScaleNearestGosimdBinding confirms the nearest-scale dispatch vars bind to
// the Go SIMD kernels wherever the CPU supports them.
func TestScaleNearestGosimdBinding(t *testing.T) {
	if !gosimdKernelsSupported() {
		t.Skip("Go SIMD kernels unsupported on this CPU")
	}
	for _, c := range []struct {
		name string
		fn   any
		want string
	}{
		{"scalePlaneNearestImpl", scalePlaneNearestImpl, "scalePlaneNearestSIMD"},
		{"scalePlaneNearest16Impl", scalePlaneNearest16Impl, "scalePlaneNearest16SIMD"},
	} {
		name := runtime.FuncForPC(reflect.ValueOf(c.fn).Pointer()).Name()
		if !strings.Contains(name, c.want) {
			t.Errorf("%s bound to %q, want %q", c.name, name, c.want)
		}
	}
}

// TestScalePlaneNearestSIMDMatchesPureGo covers both vectorized ratios with
// widths on and off the vector boundary, odd heights, offset strides, and the
// fallback ratio 3 that must route to the scalar path.
func TestScalePlaneNearestSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5CA1E))
	for _, ratio := range []int{2, 3, 4} {
		for _, dstW := range []int{1, 7, 8, 15, 16, 17, 31, 32, 40, 64, 100} {
			for _, dstH := range []int{1, 2, 5, 16} {
				srcW, srcH := dstW*ratio, dstH*ratio
				srcStride := srcW + rng.Intn(9)
				dstStride := dstW + rng.Intn(9)
				src := make([]byte, srcStride*srcH)
				fillScaleTestBytes(rng, src)
				got := make([]byte, dstStride*dstH)
				want := make([]byte, dstStride*dstH)
				scalePlaneNearestSIMD(got, dstStride, dstW, dstH, src, srcStride, srcW, srcH)
				scalePlaneNearestPureGo(want, dstStride, dstW, dstH, src, srcStride, srcW, srcH)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("ratio %d dst %dx%d: 8-bit output differs", ratio, dstW, dstH)
				}

				src16 := make([]uint16, srcStride*srcH)
				for i := range src16 {
					src16[i] = uint16(rng.Intn(65536))
				}
				got16 := make([]uint16, dstStride*dstH)
				want16 := make([]uint16, dstStride*dstH)
				scalePlaneNearest16SIMD(got16, dstStride, dstW, dstH, src16, srcStride, srcW, srcH)
				scalePlaneNearest16PureGo(want16, dstStride, dstW, dstH, src16, srcStride, srcW, srcH)
				if !reflect.DeepEqual(got16, want16) {
					t.Fatalf("ratio %d dst %dx%d: 16-bit output differs", ratio, dstW, dstH)
				}
			}
		}
	}
}

// fillScaleTestBytes fills b with random bytes, mixing full-range values with
// the 0 and 255 extremes.
func fillScaleTestBytes(rng *rand.Rand, b []byte) {
	for i := range b {
		switch rng.Intn(5) {
		case 0:
			b[i] = 0
		case 1:
			b[i] = 255
		default:
			b[i] = byte(rng.Intn(256))
		}
	}
}
