//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

// fillBorder sets every byte of dst to v: the memset of
// svt_aom_generate_padding's left and right borders. len(dst) is a multiple
// of 16 (the ME and pyramid borders are 80, 64 and 32 samples).
func fillBorder(dst []byte, v byte) {
	if len(dst)%16 != 0 {
		fillBorderScalar(dst, v)
		return
	}
	x := archsimd.BroadcastUint8x16(v)
	for i := 0; i+16 <= len(dst); i += 16 {
		x.StoreArray((*[16]uint8)(dst[i : i+16]))
	}
}
