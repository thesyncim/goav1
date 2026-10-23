//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

// memsetBytes sets every byte of dst to v: the memset of
// svt_aom_generate_padding's left and right borders and of the flat DC
// predictions. Go lowers a byte-store loop to one store per sample.
func memsetBytes(dst []byte, v byte) {
	x := archsimd.BroadcastUint8x16(v)
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		x.StoreArray((*[16]uint8)(dst[i : i+16]))
	}
	for ; i < len(dst); i++ {
		dst[i] = v
	}
}
