package goav1_test

import "testing"

// This existing inter fixture executes both SGR radii at 12-bit depth.
func BenchmarkDecodeSGRHighBDProfileClip(b *testing.B) {
	benchmarkDecodeHighLevelProfileClip(b, "internal/av1/testvector/testdata/profiles/profile2-422-12bit-inter-cdef-restoration-160x128.ivf", 8)
}
