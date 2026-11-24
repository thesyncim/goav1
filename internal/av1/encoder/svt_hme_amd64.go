//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

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
			// One 32-byte window per input row and 16 outputs: j = 2x
			// steps by 32 so each window is one compare.
			r0, r1 := r0[:2*w], r1[:2*w]
			r1 = r1[:len(r0)]
			for j := 0; j+32 <= len(r0); j += 32 {
				a := archsimd.LoadUint8x32Array((*[32]uint8)(r0[j : j+32])).DotProductPairsSaturated(ones)
				b := archsimd.LoadUint8x32Array((*[32]uint8)(r1[j : j+32])).DotProductPairsSaturated(ones)
				x = j >> 1
				packU16x16ToU8(a.Add(b).Add(two).ShiftAllRight(2).AsUint16x16()).StoreArray((*[16]uint8)(d[x : x+16]))
			}
			x = w &^ 15
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
	return sad16ColsX4(src, ref, ref[1:], ref[2:], ref[3:], srcStride, rows)
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

func svtHMEColSumRow(t *[svtHMEMaxCols / 8][8]uint32, s []byte, s2 int, ref []byte, cw *[svtHMEMaxCols]uint16, bw, rows, w int, srcSum uint32) uint32 {
	return svtHMEColSumRowScalar(t, s, s2, ref, cw[:], bw, rows, w, srcSum)
}

func svtBlockSumSubArch(s []byte, s2, bw, rows int) uint32 {
	return svtBlockSumSub(s, s2, bw, rows)
}
