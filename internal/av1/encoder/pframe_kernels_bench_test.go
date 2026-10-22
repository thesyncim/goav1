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

var benchSinkInt int
