package restoration

import "testing"

// TestSGRq63HighBitDepthConstantRegionPreservesInput pins the 10-bit
// loop-restoration SGRProj projection for a constant region under the
// extreme xqd / params seen on the libaom q63 first-frame vector. Before
// the fix in this directory ApplySelfguidedRestoration computed an extra
// 1<<(bit_depth-8) factor inside B[k] (it read the scaled local `b`
// instead of the raw bBuf[k] entry), so a flat 10-bit region collapsed to
// ~10/256 of the input value. The libaom reference (av1/common/restoration.c
// calculate_intermediate_result) uses the raw B[k] for the post-update.
//
// The test parameters mirror exactly what the q63 luma SGRProj first
// stripe reports: eps=9, xqd=[31,-32]. For a constant-input stripe (any
// input pixel value in range) the SGR self-guided smoothed image must
// satisfy flt == input<<RST_BITS so the projection is the identity. The
// expectation is that dst[i] equals the input pixel exactly for every
// sample.
func TestSGRq63HighBitDepthConstantRegionPreservesInput(t *testing.T) {
	const bitDepth = uint8(10)
	const width = 64
	const height = 8

	// Two representative 10-bit pixel values; the bug manifested at any
	// value once the bit-depth was >8, but we use both a "dim" value
	// (within the 8-bit numeric range) and a 4x-scaled "high-bit-depth"
	// value to lock in the fix across the q63 flat-region triage points.
	for _, value := range []uint16{154, 616} {
		t.Run("value="+itoa(int(value)), func(t *testing.T) {
			stride := width + 2*SGRProjBorderHorz
			origin := SGRProjBorderVert*stride + SGRProjBorderHorz
			src := make([]uint16, stride*(height+2*SGRProjBorderVert))
			for i := range src {
				src[i] = value
			}
			dst := make([]uint16, width*height)
			scratchLen, err := SelfguidedScratchLen(width, height)
			if err != nil {
				t.Fatal(err)
			}
			scratch := make([]int32, scratchLen)
			if err := ApplySelfguidedRestoration(src, stride, origin, dst, width, width, height, 9, [2]int{31, -32}, bitDepth, scratch); err != nil {
				t.Fatal(err)
			}
			for i, got := range dst {
				if got != value {
					t.Fatalf("dst[%d]=%d want %d (constant-region identity broke for bitDepth=%d xqd=[31,-32] eps=9)", i, got, value, bitDepth)
				}
			}
		})
	}
}

// itoa avoids fmt.Sprintf in a hot test path. It accepts the small
// non-negative integers used by the table-driven subtest above.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [16]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
