//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package encoder

import "unsafe"

func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	svtDownsample2D(dst, dstStride, src, srcStride, w, h)
}

func sad16xNx4(src []byte, srcStride int, ref []byte, refStride, rows int) (int, int, int, int) {
	var s [4]int
	for k := range s {
		for r := 0; r < rows; r++ {
			a := src[r*srcStride : r*srcStride+16]
			b := ref[r*refStride+k : r*refStride+k+16]
			for c := range a {
				d := int(a[c]) - int(b[c])
				if d < 0 {
					d = -d
				}
				s[k] += d
			}
		}
	}
	return s[0], s[1], s[2], s[3]
}

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
