//go:build !goav1_coeff_trace

package tile

func coeffTraceBlock(plane int, blkRow, blkCol int, txSize int, txClass int, eob int) {
}

func coeffTraceCoeff(c, pos, base, gol, level, sign int) {
}

func coeffTraceCulLevel(culLevel int) {
}
