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

func BenchmarkScalePlaneNearestDown2_1080p(b *testing.B) {
	benchmarkScalePlaneNearest(b, 960, 540, 1920, 1080)
}

func BenchmarkScalePlaneNearestDown4_1080p(b *testing.B) {
	benchmarkScalePlaneNearest(b, 480, 270, 1920, 1080)
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

func filledScaleDst(stride, height int) []byte {
	dst := make([]byte, stride*height)
	for i := range dst {
		dst[i] = 0xa5
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
