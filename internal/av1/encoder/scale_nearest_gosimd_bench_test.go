package encoder

import "testing"

// Benchmarks run through the nearest-scale dispatch vars, so they measure the
// kernel bound for the current build (Go SIMD, or scalar in default builds).

func benchScaleNearest8(b *testing.B, ratio int) {
	const dstW, dstH = 640, 360
	srcW, srcH := dstW*ratio, dstH*ratio
	src := make([]byte, srcW*srcH)
	for i := range src {
		src[i] = byte(i * 11)
	}
	dst := make([]byte, dstW*dstH)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		scalePlaneNearestImpl(dst, dstW, dstW, dstH, src, srcW, srcW, srcH)
	}
}

func benchScaleNearest16(b *testing.B, ratio int) {
	const dstW, dstH = 640, 360
	srcW, srcH := dstW*ratio, dstH*ratio
	src := make([]uint16, srcW*srcH)
	for i := range src {
		src[i] = uint16(i * 11)
	}
	dst := make([]uint16, dstW*dstH)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		scalePlaneNearest16Impl(dst, dstW, dstW, dstH, src, srcW, srcW, srcH)
	}
}

func BenchmarkScaleNearestDown2(b *testing.B)   { benchScaleNearest8(b, 2) }
func BenchmarkScaleNearestDown4(b *testing.B)   { benchScaleNearest8(b, 4) }
func BenchmarkScaleNearest16Down2(b *testing.B) { benchScaleNearest16(b, 2) }
func BenchmarkScaleNearest16Down4(b *testing.B) { benchScaleNearest16(b, 4) }
