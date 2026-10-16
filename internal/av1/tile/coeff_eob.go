package tile

// coeff_eob.go finds the end of block (one past the last non-zero coefficient
// in the default 2-D scan) for the specialized square coefficient writers.
// The scalar form walks the scan backwards from its end; the vector form
// takes the maximum of a raster-order inverse-scan table (scan index + 1)
// over the non-zero coefficients, which costs the same for every block
// instead of growing as the EOB shrinks.

var (
	coeffIScanP14x4Y2D   [16]int16
	coeffIScanP18x8Y2D   [64]int16
	coeffIScanP116x16Y2D [256]int16
	coeffIScanP132x32Y2D [1024]int16
)

func init() {
	for c := range coeffScanHot4x4Y2D {
		coeffIScanP14x4Y2D[coeffScanHot4x4Y2D[c].pos] = int16(c + 1)
	}
	for c := range coeffScanHot8x8Y2D {
		coeffIScanP18x8Y2D[coeffScanHot8x8Y2D[c].pos] = int16(c + 1)
	}
	for c := range coeffScanHot16x16Y2D {
		coeffIScanP116x16Y2D[coeffScanHot16x16Y2D[c].pos] = int16(c + 1)
	}
	for c := range coeffScanHot32x32Y2D {
		coeffIScanP132x32Y2D[coeffScanHot32x32Y2D[c].pos] = int16(c + 1)
	}
}

// txbEOBVectorImpl, when non-nil, returns the maximum of iscanP1 over the
// non-zero entries of coeffs (both n long, n a multiple of 8).
var txbEOBVectorImpl func(coeffs, iscanP1 []int16) int

func txbEOB16(coeffs *[16]int16) int {
	if txbEOBVectorImpl != nil {
		return txbEOBVectorImpl(coeffs[:], coeffIScanP14x4Y2D[:])
	}
	for c := 15; c >= 0; c-- {
		if coeffs[coeffScanHot4x4Y2D[c].pos] != 0 {
			return c + 1
		}
	}
	return 0
}

func txbEOB64(coeffs *[64]int16) int {
	if txbEOBVectorImpl != nil {
		return txbEOBVectorImpl(coeffs[:], coeffIScanP18x8Y2D[:])
	}
	for c := 63; c >= 0; c-- {
		if coeffs[coeffScanHot8x8Y2D[c].pos] != 0 {
			return c + 1
		}
	}
	return 0
}

func txbEOB256(coeffs *[256]int16) int {
	if txbEOBVectorImpl != nil {
		return txbEOBVectorImpl(coeffs[:], coeffIScanP116x16Y2D[:])
	}
	for c := 255; c >= 0; c-- {
		if coeffs[coeffScanHot16x16Y2D[c].pos] != 0 {
			return c + 1
		}
	}
	return 0
}

func txbEOB1024(coeffs *[1024]int16) int {
	if txbEOBVectorImpl != nil {
		return txbEOBVectorImpl(coeffs[:], coeffIScanP132x32Y2D[:])
	}
	for c := 1023; c >= 0; c-- {
		if coeffs[coeffScanHot32x32Y2D[c].pos] != 0 {
			return c + 1
		}
	}
	return 0
}
