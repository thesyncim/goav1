package encoder

import "testing"

func TestScalePlaneNearestMatchesPureGo(t *testing.T) {
	tests := []struct {
		name      string
		dstWidth  int
		dstHeight int
		srcWidth  int
		srcHeight int
		dstPad    int
		srcPad    int
	}{
		{name: "down2_tail", dstWidth: 19, dstHeight: 13, srcWidth: 38, srcHeight: 26, dstPad: 5, srcPad: 7},
		{name: "down2_vector", dstWidth: 32, dstHeight: 16, srcWidth: 64, srcHeight: 32, dstPad: 3, srcPad: 9},
		{name: "down4_tail", dstWidth: 17, dstHeight: 11, srcWidth: 68, srcHeight: 44, dstPad: 2, srcPad: 11},
		{name: "down4_vector", dstWidth: 32, dstHeight: 18, srcWidth: 128, srcHeight: 72, dstPad: 1, srcPad: 13},
		{name: "fallback_ratio", dstWidth: 23, dstHeight: 15, srcWidth: 41, srcHeight: 29, dstPad: 6, srcPad: 4},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dstStride := tc.dstWidth + tc.dstPad
			srcStride := tc.srcWidth + tc.srcPad
			src := make([]byte, srcStride*tc.srcHeight)
			for y := 0; y < tc.srcHeight; y++ {
				for x := 0; x < srcStride; x++ {
					src[y*srcStride+x] = byte((x*17 + y*31 + x*y*3 + 11) & 0xff)
				}
			}
			got := filledScaleDst(dstStride, tc.dstHeight)
			want := filledScaleDst(dstStride, tc.dstHeight)

			scalePlaneNearest(got, dstStride, tc.dstWidth, tc.dstHeight, src, srcStride, tc.srcWidth, tc.srcHeight)
			scalePlaneNearestPureGo(want, dstStride, tc.dstWidth, tc.dstHeight, src, srcStride, tc.srcWidth, tc.srcHeight)

			assertScalePlaneRowsEqual(t, got, want, dstStride, tc.dstWidth, tc.dstHeight)
			assertScalePlanePaddingPreserved(t, got, dstStride, tc.dstWidth, tc.dstHeight)
		})
	}
}

func TestScalePlaneNearest16MatchesPureGo(t *testing.T) {
	tests := []struct {
		name      string
		dstWidth  int
		dstHeight int
		srcWidth  int
		srcHeight int
		dstPad    int
		srcPad    int
	}{
		{name: "down2_tail", dstWidth: 13, dstHeight: 9, srcWidth: 26, srcHeight: 18, dstPad: 4, srcPad: 3},
		{name: "down2_vector", dstWidth: 24, dstHeight: 12, srcWidth: 48, srcHeight: 24, dstPad: 5, srcPad: 7},
		{name: "down4_tail", dstWidth: 11, dstHeight: 7, srcWidth: 44, srcHeight: 28, dstPad: 6, srcPad: 9},
		{name: "down4_vector", dstWidth: 24, dstHeight: 10, srcWidth: 96, srcHeight: 40, dstPad: 1, srcPad: 11},
		{name: "fallback_ratio", dstWidth: 19, dstHeight: 11, srcWidth: 33, srcHeight: 21, dstPad: 5, srcPad: 4},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dstStride := tc.dstWidth + tc.dstPad
			srcStride := tc.srcWidth + tc.srcPad
			src := make([]uint16, srcStride*tc.srcHeight)
			for y := 0; y < tc.srcHeight; y++ {
				for x := 0; x < srcStride; x++ {
					src[y*srcStride+x] = uint16((x*257 + y*769 + x*y*11 + 19) & 0x0fff)
				}
			}
			got := filledScaleDst16(dstStride, tc.dstHeight)
			want := filledScaleDst16(dstStride, tc.dstHeight)

			scalePlaneNearest16(got, dstStride, tc.dstWidth, tc.dstHeight, src, srcStride, tc.srcWidth, tc.srcHeight)
			scalePlaneNearest16PureGo(want, dstStride, tc.dstWidth, tc.dstHeight, src, srcStride, tc.srcWidth, tc.srcHeight)

			assertScalePlaneRows16Equal(t, got, want, dstStride, tc.dstWidth, tc.dstHeight)
			assertScalePlanePadding16Preserved(t, got, dstStride, tc.dstWidth, tc.dstHeight)
		})
	}
}

func BenchmarkScalePlaneNearestDown2_1080p(b *testing.B) {
	benchmarkScalePlaneNearest(b, 960, 540, 1920, 1080)
}

func BenchmarkScalePlaneNearestDown4_1080p(b *testing.B) {
	benchmarkScalePlaneNearest(b, 480, 270, 1920, 1080)
}

func BenchmarkScalePlaneNearest16Down2_1080p(b *testing.B) {
	benchmarkScalePlaneNearest16(b, 960, 540, 1920, 1080)
}

func BenchmarkScalePlaneNearest16Down4_1080p(b *testing.B) {
	benchmarkScalePlaneNearest16(b, 480, 270, 1920, 1080)
}

func benchmarkScalePlaneNearest(b *testing.B, dstWidth, dstHeight, srcWidth, srcHeight int) {
	dst := make([]byte, dstWidth*dstHeight)
	src := make([]byte, srcWidth*srcHeight)
	for i := range src {
		src[i] = byte((i*37 + i/11) & 0xff)
	}
	b.ReportAllocs()
	b.SetBytes(int64(dstWidth * dstHeight))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scalePlaneNearest(dst, dstWidth, dstWidth, dstHeight, src, srcWidth, srcWidth, srcHeight)
	}
}

func benchmarkScalePlaneNearest16(b *testing.B, dstWidth, dstHeight, srcWidth, srcHeight int) {
	dst := make([]uint16, dstWidth*dstHeight)
	src := make([]uint16, srcWidth*srcHeight)
	for i := range src {
		src[i] = uint16((i*37 + i/11) & 0x0fff)
	}
	b.ReportAllocs()
	b.SetBytes(int64(dstWidth * dstHeight * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scalePlaneNearest16(dst, dstWidth, dstWidth, dstHeight, src, srcWidth, srcWidth, srcHeight)
	}
}

func filledScaleDst(stride, height int) []byte {
	dst := make([]byte, stride*height)
	for i := range dst {
		dst[i] = 0xa5
	}
	return dst
}

func filledScaleDst16(stride, height int) []uint16 {
	dst := make([]uint16, stride*height)
	for i := range dst {
		dst[i] = 0xa5a5
	}
	return dst
}

func assertScalePlaneRowsEqual(t *testing.T, got []byte, want []byte, stride, width, height int) {
	t.Helper()
	for y := 0; y < height; y++ {
		goff := y * stride
		woff := y * stride
		for x := 0; x < width; x++ {
			if got[goff+x] != want[woff+x] {
				t.Fatalf("pixel[%d,%d]=%d want %d", x, y, got[goff+x], want[woff+x])
			}
		}
	}
}

func assertScalePlaneRows16Equal(t *testing.T, got []uint16, want []uint16, stride, width, height int) {
	t.Helper()
	for y := 0; y < height; y++ {
		goff := y * stride
		woff := y * stride
		for x := 0; x < width; x++ {
			if got[goff+x] != want[woff+x] {
				t.Fatalf("pixel[%d,%d]=%d want %d", x, y, got[goff+x], want[woff+x])
			}
		}
	}
}

func assertScalePlanePaddingPreserved(t *testing.T, got []byte, stride, width, height int) {
	t.Helper()
	for y := 0; y < height; y++ {
		for x := width; x < stride; x++ {
			if got[y*stride+x] != 0xa5 {
				t.Fatalf("padding[%d,%d]=%d want 0xa5", x, y, got[y*stride+x])
			}
		}
	}
}

func assertScalePlanePadding16Preserved(t *testing.T, got []uint16, stride, width, height int) {
	t.Helper()
	for y := 0; y < height; y++ {
		for x := width; x < stride; x++ {
			if got[y*stride+x] != 0xa5a5 {
				t.Fatalf("padding[%d,%d]=%d want 0xa5a5", x, y, got[y*stride+x])
			}
		}
	}
}
