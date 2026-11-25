// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
)

// cdefDist8 is the scalar svt_aom_compute_cdef_dist_8bit_c on targets
// without the Go SIMD kernel.
func cdefDist8(src []byte, sstride int, filtered []byte, fstride int, dlist []cdef.BlockPosition, bwLog2, bhLog2 int, subsampling int) uint64 {
	return computeCDEFDist8(src, sstride, filtered, fstride, dlist, bwLog2, bhLog2, subsampling)
}

// lpd1SampledSAD is the row-subsampled SAD the chroma detector uses
// (svt_nxm_sad_kernel called with stride << shift and height >> shift).
func lpd1SampledSAD(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	return lpd1SampledSADPureGo(src, ref, srcOff, refOff, srcStride, refStride, w, rows, step)
}

func boxInterpolateRow8(dst, s0, s1 []uint8, f int) { boxInterpolateRow(dst, s0, s1, f, 128) }

// boxColsCache and boxColsPlan are the arm64 SIMD column filter's table; the
// other builds run boxFilterCols.
type boxColsCache struct{}

type boxColsPlan struct{}

func (*boxColsCache) plan(sw, dw, x, dx, rows int) *boxColsPlan { return nil }

func (*boxColsPlan) filter(dst, s []uint8) {}

func svtSubpelVarianceArch(scratch *svtSubpelScratch, ref []byte, refStride int, src []byte, srcStride, bw, bh, xoff, yoff int) uint32 {
	svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
	_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, bw, srcStride, 0, 0, bw, bh)
	return v
}

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

func svtHMEColSumRow(t *[svtHMEMaxCols / 8][8]uint32, s []byte, s2 int, ref []byte, cw *[svtHMEMaxCols]uint16, bw, rows, w int, srcSum uint32) uint32 {
	return svtHMEColSumRowScalar(t, s, s2, ref, cw[:], bw, rows, w, srcSum)
}

func svtBlockSumSubArch(s []byte, s2, bw, rows int) uint32 {
	return svtBlockSumSub(s, s2, bw, rows)
}

func memsetBytes(dst []byte, v byte) { memsetBytesScalar(dst, v) }

func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	a := svtSAD16xhx8Scalar(src, srcStride, ref, refStride, rows)
	for k := range acc {
		acc[k] += a[k]
	}
}

// svtSAD32xhx8 is the 32-wide eight-position step: the two 16-column halves.
func svtSAD32xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	svtSAD16xhx8(acc, src, srcStride, ref, refStride, rows)
	svtSAD16xhx8(acc, src[16:], srcStride, ref[16:], refStride, rows)
}

func (me *realtimeSBMultiSizeME) eightSAD32x32_64x64(sad16 *[16][8]uint32, mvX, mvY int) {
	me.eightSAD32x32_64x64Generic(sad16, mvX, mvY)
}

func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	svtAllSAD8x8_16x16SubScalar(me, src, ref, stride, mvX, mvY, sad16)
}

func (me *realtimeSBMultiSizeME) sweepPoint1Arch(src []byte, stride int, ref []byte, srcBase, refBase, mvX, mvY int) {
	me.sweepSearchPoint1(src, stride, ref, stride, srcBase, refBase, mvX, mvY)
}
