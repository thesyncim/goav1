//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtDownsample2DArch is svtDownsample2D on AVX2: VPMADDUBSW by ones sums the
// horizontal sample pairs of each row into int16 lanes, the two rows add, and
// (sum + 2) >> 2 packs back to bytes, sixteen outputs per step.
func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	if !encAVX2 {
		svtDownsample2D(dst, dstStride, src, srcStride, w, h)
		return
	}
	ones := archsimd.BroadcastInt8x32(1)
	two := archsimd.BroadcastInt16x16(2)
	for y := 0; y < h; y++ {
		r0 := src[(2*y)*srcStride:]
		r1 := src[(2*y+1)*srcStride:]
		d := dst[y*dstStride : y*dstStride+w]
		x := 0
		if len(r0) >= 2*w && len(r1) >= 2*w {
			p0 := unsafe.Pointer(&r0[0])
			p1 := unsafe.Pointer(&r1[0])
			for ; x+16 <= w; x += 16 {
				a := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(p0, 2*x))).DotProductPairsSaturated(ones)
				b := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(p1, 2*x))).DotProductPairsSaturated(ones)
				packU16x16ToU8(a.Add(b).Add(two).ShiftAllRight(2).AsUint16x16()).StoreArray((*[16]uint8)(unsafe.Pointer(&d[x])))
			}
		}
		for ; x < w; x++ {
			s := uint32(r0[2*x]) + uint32(r0[2*x+1]) + uint32(r1[2*x]) + uint32(r1[2*x+1])
			d[x] = uint8((s + 2) >> 2)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// sad16xNx4 returns the SADs of the 16-wide, rows-tall source block against
// reference origins ref+0..ref+3 (svt_sad_loop_kernel's per-row position
// batch), VPSADBW per row and position.
func sad16xNx4(src []byte, srcStride int, ref []byte, refStride, rows int) (int, int, int, int) {
	_ = src[(rows-1)*srcStride+15]
	_ = ref[(rows-1)*refStride+18]
	if !encAVX2 || srcStride != refStride {
		var s [4]int
		for k := range s {
			for r := 0; r < rows; r++ {
				a := src[r*srcStride : r*srcStride+16]
				b := ref[r*refStride+k : r*refStride+k+16]
				for c := range a {
					s[k] += max(int(a[c])-int(b[c]), int(b[c])-int(a[c]))
				}
			}
		}
		return s[0], s[1], s[2], s[3]
	}
	return sad16ColsX4Ptr(unsafe.Pointer(&src[0]), unsafe.Pointer(&ref[0]), unsafe.Pointer(&ref[1]),
		unsafe.Pointer(&ref[2]), unsafe.Pointer(&ref[3]), srcStride, rows)
}

// The column-sum HME kernels serve NEON, which has no absolute difference;
// amd64 scores HME positions with VPSADBW (svtSAD16xhx8).
const svtHMEColSums = false

func svtColTotalsArch(cw *[svtHMEMaxCols]uint16, ref []byte, s2, rows, span int) {
	svtColTotals(cw, ref, s2, rows, span)
}

func svtColTotalsSlideArch(cw *[svtHMEMaxCols]uint16, out, in []byte, span int) {
	svtColTotalsSlide(cw, out, in, span)
}

func svtHMEEightColSum(t *[8]uint32, sp, rp, cwp unsafe.Pointer, s2, bw, rows int, srcSum uint32) {
	n := (rows-1)*s2 + bw + 8
	svtHMEEightColSumScalar(t, unsafe.Slice((*byte)(sp), n), s2, unsafe.Slice((*byte)(rp), n),
		unsafe.Slice((*uint16)(cwp), bw+8), bw, rows, srcSum)
}

func svtBlockSumSubArch(s []byte, s2, bw, rows int) uint32 {
	return svtBlockSumSub(s, s2, bw, rows)
}
