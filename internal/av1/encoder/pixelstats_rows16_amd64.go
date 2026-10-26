//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// pixelStatsRows16Arch is pixelStatsRows16 as SVT-AV1's variance_kernel_avx2:
// each 16-sample row segment zero-extends to 16 int16 lanes (VPMOVZXBW), the
// difference squares and pair-sums straight into int32 lanes (VPMADDWD), and
// the signed sum accumulates in int16 lanes. Every 16-sample step adds one
// difference (|d| <= 255) to each int16 lane, so 128 steps fit before the
// flush to int32; the sse of a 64x64 block is at most 4096*65025 < 2^31.
//
// Like SVT's fixed-shape variance{16,32,64}_avx2, the 16-, 32- and 64-wide
// shapes step raw row pointers with no column loop and no per-row slicing,
// and their flush groups are constants (no division per call).
//
// The kernels run only behind gosimdKernelsSupported (AVX2).
func pixelStatsRows16Arch(src []byte, srcStride int, ref []byte, refStride int, w, h int) (uint32, int32) {
	if h <= 0 {
		return 0, 0
	}
	_ = src[(h-1)*srcStride+w-1]
	_ = ref[(h-1)*refStride+w-1]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var groupRows int
	switch w {
	case 16:
		groupRows = 128
	case 32:
		groupRows = 64
	case 64:
		groupRows = 32
	default:
		groupRows = max(1, 1024/w)
	}
	var sse, sum32 archsimd.Int32x8
	for r0 := 0; r0 < h; r0 += groupRows {
		var sum16 archsimd.Int16x16
		n := min(groupRows, h-r0)
		// Row offsets step as integers: a pointer stepped past the last row
		// would point one past the buffer, which the GC rejects if it is
		// still live at a safepoint.
		so, po := r0*srcStride, r0*refStride
		switch w {
		case 16:
			for range n {
				s, p := unsafe.Add(sp, so), unsafe.Add(rp, po)
				d := pixelDiff16(s, p)
				sum16 = sum16.Add(d)
				sse = sse.Add(d.DotProductPairs(d))
				so, po = so+srcStride, po+refStride
			}
		case 32:
			for range n {
				s, p := unsafe.Add(sp, so), unsafe.Add(rp, po)
				d0 := pixelDiff16(s, p)
				d1 := pixelDiff16(unsafe.Add(s, 16), unsafe.Add(p, 16))
				sum16 = sum16.Add(d0).Add(d1)
				sse = sse.Add(d0.DotProductPairs(d0)).Add(d1.DotProductPairs(d1))
				so, po = so+srcStride, po+refStride
			}
		case 64:
			for range n {
				s, p := unsafe.Add(sp, so), unsafe.Add(rp, po)
				d0 := pixelDiff16(s, p)
				d1 := pixelDiff16(unsafe.Add(s, 16), unsafe.Add(p, 16))
				d2 := pixelDiff16(unsafe.Add(s, 32), unsafe.Add(p, 32))
				d3 := pixelDiff16(unsafe.Add(s, 48), unsafe.Add(p, 48))
				sum16 = sum16.Add(d0).Add(d1).Add(d2).Add(d3)
				sse = sse.Add(d0.DotProductPairs(d0)).Add(d1.DotProductPairs(d1)).
					Add(d2.DotProductPairs(d2)).Add(d3.DotProductPairs(d3))
				so, po = so+srcStride, po+refStride
			}
		default:
			for range n {
				s, p := unsafe.Add(sp, so), unsafe.Add(rp, po)
				for c := 0; c < w; c += 16 {
					d := pixelDiff16(unsafe.Add(s, c), unsafe.Add(p, c))
					sum16 = sum16.Add(d)
					sse = sse.Add(d.DotProductPairs(d))
				}
				so, po = so+srcStride, po+refStride
			}
		}
		sum32 = sum32.Add(sum16.GetLo().ExtendToInt32()).Add(sum16.GetHi().ExtendToInt32())
	}
	sseOut, sumOut := uint32(reduceInt32x8(sse)), reduceInt32x8(sum32)
	// Return with clean upper ymm state (the compiler emits no vzeroupper).
	archsimd.ClearAVXUpperBits()
	return sseOut, sumOut
}

// pixelDiff16 is the sixteen int16 differences src-ref of the samples at s
// and p (VPMOVZXBW from memory, VPSUBW).
func pixelDiff16(s, p unsafe.Pointer) archsimd.Int16x16 {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(s)).ExtendToUint16().BitsToInt16()
	b := archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendToUint16().BitsToInt16()
	return a.Sub(b)
}

// reduceInt32x8 returns the sum of the eight lanes.
func reduceInt32x8(v archsimd.Int32x8) int32 {
	q := v.GetLo().Add(v.GetHi())
	return q.GetElem(0) + q.GetElem(1) + q.GetElem(2) + q.GetElem(3)
}
