//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

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
func pred8AVX2(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int,
	ref unsafe.Pointer, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int {
	sad := pred8CopyAVX2(dst, dstStride, src, srcStride, ref, refStride, w, h)
	archsimd.ClearAVXUpperBits()
	return sad
}

// pred8Out stores sixteen output bytes or adds their SAD against src to acc.
func pred8Out(out archsimd.Uint8x16, dst unsafe.Pointer, dstOff int, src unsafe.Pointer, srcOff int, acc archsimd.Uint64x2) archsimd.Uint64x2 {
	if src != nil {
		return acc.Add(out.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, srcOff)))))
	}
	out.StoreArray((*[16]uint8)(unsafe.Add(dst, dstOff)))
	return acc
}

func pred8CopyAVX2(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, ref unsafe.Pointer, refStride, w, h int) int {
	if src == nil {
		pred8CopyRowsAVX2(dst, dstStride, ref, refStride, w, h)
		return 0
	}
	var acc archsimd.Uint64x2
	for y := 0; y < h; y++ {
		r := unsafe.Add(ref, y*refStride)
		for x := 0; x < w; x += 16 {
			acc = pred8Out(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, x))), nil, 0, src, y*srcStride+x, acc)
		}
	}
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// pred8CopyRowsAVX2 is svt_av1_convolve_2d_copy_sr_avx2: one loop per block
// width (the row pointers are formed by index: one past the last row would
// trip checkptr).
func pred8CopyRowsAVX2(dst unsafe.Pointer, dstStride int, ref unsafe.Pointer, refStride, w, h int) {
	switch w {
	case 8:
		for y := range h {
			*(*uint64)(unsafe.Add(dst, y*dstStride)) = *(*uint64)(unsafe.Add(ref, y*refStride))
		}
	case 16:
		for y := range h {
			archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(ref, y*refStride))).StoreArray((*[16]uint8)(unsafe.Add(dst, y*dstStride)))
		}
	case 32:
		for y := range h {
			archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(ref, y*refStride))).StoreArray((*[32]uint8)(unsafe.Add(dst, y*dstStride)))
		}
	case 64:
		for y := range h {
			r, d := unsafe.Add(ref, y*refStride), unsafe.Add(dst, y*dstStride)
			archsimd.LoadUint8x32Array((*[32]uint8)(r)).StoreArray((*[32]uint8)(d))
			archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(r, 32))).StoreArray((*[32]uint8)(unsafe.Add(d, 32)))
		}
	default:
		for y := range h {
			d, r := unsafe.Add(dst, y*dstStride), unsafe.Add(ref, y*refStride)
			x := 0
			for ; x+16 <= w; x += 16 {
				archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, x))).StoreArray((*[16]uint8)(unsafe.Add(d, x)))
			}
			if x < w {
				*(*uint64)(unsafe.Add(d, x)) = *(*uint64)(unsafe.Add(r, x))
			}
		}
	}
}
