//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

// scaleRow8Down2 writes dst[x] = src[2x] for len(dst) a multiple of 32. Two
// ConcatEven pairs per iteration keep four loads in flight.
func scaleRow8Down2(dst, src []byte) {
	for x := 0; x < len(dst); x += 32 {
		s := (*[64]byte)(src[2*x:])
		a := archsimd.LoadUint8x16Array((*[16]uint8)(s[0:16]))
		b := archsimd.LoadUint8x16Array((*[16]uint8)(s[16:32]))
		c := archsimd.LoadUint8x16Array((*[16]uint8)(s[32:48]))
		d := archsimd.LoadUint8x16Array((*[16]uint8)(s[48:64]))
		a.ConcatEven(b).StoreArray((*[16]uint8)(dst[x : x+16]))
		c.ConcatEven(d).StoreArray((*[16]uint8)(dst[x+16 : x+32]))
	}
}

// scaleRow8Down4 writes dst[x] = src[4x] for len(dst) a multiple of 16. Two
// rounds of ConcatEven pick every 4th byte across a 64-byte window.
func scaleRow8Down4(dst, src []byte) {
	for x := 0; x < len(dst); x += 16 {
		s := src[4*x : 4*x+64]
		e1 := archsimd.LoadUint8x16(s[0:16]).ConcatEven(archsimd.LoadUint8x16(s[16:32]))
		e2 := archsimd.LoadUint8x16(s[32:48]).ConcatEven(archsimd.LoadUint8x16(s[48:64]))
		e1.ConcatEven(e2).Store(dst[x : x+16])
	}
}

// scaleRow16Down2 writes dst[x] = src[2x] for len(dst) a multiple of 16. Two
// ConcatEven pairs per iteration keep four loads in flight.
func scaleRow16Down2(dst, src []uint16) {
	for x := 0; x < len(dst); x += 16 {
		s := (*[32]uint16)(src[2*x:])
		a := archsimd.LoadUint16x8Array((*[8]uint16)(s[0:8]))
		b := archsimd.LoadUint16x8Array((*[8]uint16)(s[8:16]))
		c := archsimd.LoadUint16x8Array((*[8]uint16)(s[16:24]))
		d := archsimd.LoadUint16x8Array((*[8]uint16)(s[24:32]))
		a.ConcatEven(b).StoreArray((*[8]uint16)(dst[x : x+8]))
		c.ConcatEven(d).StoreArray((*[8]uint16)(dst[x+8 : x+16]))
	}
}

// scaleRow16Down4 writes dst[x] = src[4x] for len(dst) a multiple of 8.
func scaleRow16Down4(dst, src []uint16) {
	for x := 0; x < len(dst); x += 8 {
		s := src[4*x : 4*x+32]
		e1 := archsimd.LoadUint16x8(s[0:8]).ConcatEven(archsimd.LoadUint16x8(s[8:16]))
		e2 := archsimd.LoadUint16x8(s[16:24]).ConcatEven(archsimd.LoadUint16x8(s[24:32]))
		e1.ConcatEven(e2).Store(dst[x : x+8])
	}
}
