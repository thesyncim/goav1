package transform

import "testing"

// TestInverseDCTDCOnlySampleMatchesTransform checks dav1d's closed-form DC-only
// residual against the full 1D-kernel evaluation for every size, bit depth and
// dequantized DC value.
func TestInverseDCTDCOnlySampleMatchesTransform(t *testing.T) {
	scratch := make([]int32, 128)
	dims := []uint8{4, 8, 16, 32, 64}
	for _, w := range dims {
		for _, h := range dims {
			size := Size{Width: w, Height: h}
			if idx := sizeIndex(size); idx < 0 || !sizeValidTable[idx] {
				continue
			}
			for _, bd := range []uint8{8, 10, 12} {
				cfMax := int32(1)<<(7+bd) - 1
				step := int32(1)
				if bd > 8 {
					step = 7
				}
				for dc := -cfMax - 1; dc <= cfMax; dc += step {
					want, err := InverseDCTDCOnlySampleBitDepth(dc, scratch, size, bd)
					if err != nil {
						t.Fatal(err)
					}
					if got := InverseDCTDCOnlySample(dc, size, bd); got != want {
						t.Fatalf("%dx%d bd=%d dc=%d: got %d want %d", size.Width, size.Height, bd, dc, got, want)
					}
				}
			}
		}
	}
}
