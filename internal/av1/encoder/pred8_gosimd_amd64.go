//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

func init() {
	if gosimdKernelsSupported() {
		pred8Impl = pred8AVX2
		pred8FullPelOnly = true
	}
}

// pred8AVX2 is the full-pel case of pred8SIMD (pred8_gosimd_arm64.go) on
// AVX2: svt_av1_convolve_2d_copy_sr_avx2 for predictions and VPSADBW for
// motion-search SADs. Sub-pel blocks stay on the motion package's dav1d
// put_8tap kernels (pred8FullPelOnly): an AVX2 port of the encoder's
// halved-tap sub-pel kernels measured 4-30x slower than those on x86.
// It clears the upper YMM state before returning.
func pred8AVX2(dst []byte, dstStride int, src []byte, srcStride int,
	ref []byte, refOff, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int {
	sad := pred8CopyAVX2(dst, dstStride, src, srcStride, ref[refOff:], refStride, w, h)
	archsimd.ClearAVXUpperBits()
	return sad
}

func pred8CopyAVX2(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refStride, w, h int) int {
	if src == nil {
		pred8CopyRowsAVX2(dst, dstStride, ref, refStride, w, h)
		return 0
	}
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	var acc archsimd.Uint64x2
	for y := 0; y < h; y++ {
		r, s := ref[y*refStride:y*refStride+w], src[y*srcStride:y*srcStride+w]
		s = s[:len(r)]
		for x := 0; x+16 <= len(r); x += 16 {
			acc = acc.Add(archsimd.LoadUint8x16Array((*[16]uint8)(r[x : x+16])).SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(s[x : x+16]))))
		}
	}
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// pred8CopyRowsAVX2 is svt_av1_convolve_2d_copy_sr_avx2: one loop per block
// width, each taking one bounds-checked window per row pair and array (row
// 2i at w[:n], row 2i+1 at w[stride:]).
func pred8CopyRowsAVX2(dst []byte, dstStride int, ref []byte, refStride, w, h int) {
	if uint(dstStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	switch w {
	case 8:
		dn, rn := dstStride+8, refStride+8
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			*(*[8]byte)(dw) = *(*[8]byte)(rw)
			*(*[8]byte)(dw[dstStride:]) = *(*[8]byte)(rw[refStride:])
		}
	case 16:
		dn, rn := dstStride+16, refStride+16
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			archsimd.LoadUint8x16Array((*[16]uint8)(rw)).StoreArray((*[16]uint8)(dw))
			archsimd.LoadUint8x16Array((*[16]uint8)(rw[refStride:])).StoreArray((*[16]uint8)(dw[dstStride:]))
		}
	case 32:
		dn, rn := dstStride+32, refStride+32
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			archsimd.LoadUint8x32Array((*[32]uint8)(rw)).StoreArray((*[32]uint8)(dw))
			archsimd.LoadUint8x32Array((*[32]uint8)(rw[refStride:])).StoreArray((*[32]uint8)(dw[dstStride:]))
		}
	case 64:
		dn, rn := dstStride+64, refStride+64
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			r0, d0 := (*[64]uint8)(rw), (*[64]uint8)(dw)
			r1, d1 := (*[64]uint8)(rw[refStride:]), (*[64]uint8)(dw[dstStride:])
			archsimd.LoadUint8x32Array((*[32]uint8)(r0[0:32])).StoreArray((*[32]uint8)(d0[0:32]))
			archsimd.LoadUint8x32Array((*[32]uint8)(r0[32:64])).StoreArray((*[32]uint8)(d0[32:64]))
			archsimd.LoadUint8x32Array((*[32]uint8)(r1[0:32])).StoreArray((*[32]uint8)(d1[0:32]))
			archsimd.LoadUint8x32Array((*[32]uint8)(r1[32:64])).StoreArray((*[32]uint8)(d1[32:64]))
		}
	default:
		for y := range h {
			d, r := dst[y*dstStride:y*dstStride+w], ref[y*refStride:y*refStride+w]
			d = d[:len(r)]
			x := 0
			for ; x+16 <= len(r); x += 16 {
				archsimd.LoadUint8x16Array((*[16]uint8)(r[x : x+16])).StoreArray((*[16]uint8)(d[x : x+16]))
			}
			if x+8 <= len(r) {
				*(*[8]byte)(d[x : x+8]) = *(*[8]byte)(r[x : x+8])
			}
		}
		return
	}
	if h&1 != 0 {
		// The block heights are even; an odd one finishes its last row.
		y := h - 1
		copy(dst[y*dstStride:y*dstStride+w], ref[y*refStride:y*refStride+w])
	}
}
