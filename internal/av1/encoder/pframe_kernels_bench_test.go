package encoder

import "testing"

func BenchmarkRealtimeAvg8x8Kernel(b *testing.B) {
	src := make([]byte, 8*64+8)
	for i := range src {
		src[i] = byte(i * 7)
	}
	b.ReportAllocs()
	b.ResetTimer()
	sink := 0
	for range b.N {
		sink += realtimeAvg8x8(src, 64)
	}
	benchSinkInt = sink
}

func BenchmarkRealtimeIntProRowKernel(b *testing.B) {
	const w, h, stride = 64, 64, 80
	ref := make([]byte, (h-1)*stride+w)
	for i := range ref {
		ref[i] = byte(i * 13)
	}
	dst := make([]int16, w)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		realtimeIntProRowInBoundsArch(dst, ref, stride, w, h, 5)
	}
	benchSinkInt = int(dst[0])
}

func BenchmarkRealtimeIntProColKernel(b *testing.B) {
	const w, h, stride = 64, 64, 80
	ref := make([]byte, (h-1)*stride+w)
	for i := range ref {
		ref[i] = byte(i * 13)
	}
	dst := make([]int16, h)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		realtimeIntProColInBoundsArch(dst, ref, stride, w, h, 5)
	}
	benchSinkInt = int(dst[0])
}

var benchSinkInt int
