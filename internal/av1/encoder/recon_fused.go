package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// reconFusedDCT reconstructs an 8-bit square DCT_DCT transform block through
// the transform package's fused inverse-transform-and-add kernels
// (itx_fused*), which write the bytes of the staged inverse transform plus
// addRawResidualToPredReconBlock. The coded coefficients lie in rows
// 0..activeRows-1 of the default scan prefix [0,eob). It reports false when
// no kernel covers the block (size, build, partial pred overlap, or a kernel
// guard), and the caller then runs the staged path.
func reconFusedDCT(dst []byte, stride int, pred []byte, predStride int, coeff []int32, geo txBlockGeometry, w, h int, activeRows int, eob int) bool {
	if w != h || activeRows <= 0 || rawPredReconNeedsCopy(dst, stride, pred, predStride, w, h) {
		return false
	}
	activeCols, ok := transform.DefaultScanActiveCols(geo.size, transform.Class2D, eob)
	if !ok || activeCols <= 0 {
		return false
	}
	switch w {
	case 8:
		return transform.InverseAddDCT8x8(dst, stride, pred, predStride, coeff, geo.coeffHeight, activeRows, activeCols)
	case 16:
		return transform.InverseAddDCT16x16(dst, stride, pred, predStride, coeff, geo.coeffHeight, activeRows, activeCols)
	case 32:
		if activeRows <= 8 && activeCols <= 8 {
			return transform.InverseAddDCT32x32Low8(dst, stride, pred, predStride, coeff, geo.coeffHeight)
		}
		if activeRows <= 16 && activeCols <= 16 {
			return transform.InverseAddDCT32x32Low16(dst, stride, pred, predStride, coeff, geo.coeffHeight, activeRows)
		}
	}
	return false
}
