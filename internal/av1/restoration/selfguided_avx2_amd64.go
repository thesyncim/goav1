// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

// AVX2-accelerated self-guided restoration box sums, mirroring the arm64 NEON
// port. The interior columns (where the 2r+1 horizontal window is fully in
// bounds) are summed 8 lanes at a time; the r edge columns on each side keep
// the scalar variable-bound reference. The per-pixel blend lives in
// selfguided_blend_avx2_amd64.go; the data-dependent LUT gather inside
// calculateIntermediate (xByXPlus1 / oneByX) stays scalar in this path.
//
// dav1d reference (algorithm shape, not bit-exact math): the sgr box3/box5
// sums live in third_party/upstream/dav1d/src/x86/looprestoration_avx2.asm
// (sgr_filter_5x5_8bpc / sgr_filter_3x3_8bpc). goav1 keeps the libaom integer
// lattice; the AVX2 here reproduces the exact pure-Go int32 box sums, so every
// output sample is bit-identical for 8/10/12-bit input.

// boxsumInteriorBandAVX2Asm sums the box-filter windows for the interior
// columns of one output row band. For each group of 8 interior columns it
// accumulates, in an 8-lane int32 register, the (squared) source values over
// all y0..y1 rows and all 2r+1 horizontal taps, then stores the eight results
// once. Bit-exact with the scalar reference: int32 addition is associative
// under two's-complement wraparound, so summing the identical set of (squared)
// values in a different grouping yields the same int32 result; VPMULLD's low
// 32 bits equal Go's int32 v*v.
//
//go:noescape
func boxsumInteriorBandAVX2Asm(src *int32, dst *int32, avxCols uintptr, radius uintptr, rows uintptr, srcStride uintptr, squared uintptr)

// boxsumAVX2 is the dispatch entry for the box sums. The horizontal extent is
// fixed (2r+1) for the interior columns [r, width-1-r]; those run through the
// AVX2 inner loop. The first r and last r columns have clamped windows and keep
// the scalar reference.
func boxsumAVX2(src []int32, srcOrigin int, width int, height int, srcStride int, radius int, squared bool, dst []int32, dstStride int) {
	interiorStart := radius
	interiorEnd := width - 1 - radius // inclusive
	interiorCount := interiorEnd - interiorStart + 1
	if width <= 0 || height <= 0 || radius < 0 || interiorCount < 8 {
		boxsum(src, srcOrigin, width, height, srcStride, radius, squared, dst, dstStride)
		return
	}
	sq := uintptr(0)
	if squared {
		sq = 1
	}
	for row := 0; row < height; row++ {
		y0 := maxInt(0, row-radius)
		y1 := minInt(height-1, row+radius)
		dstRow := dst[row*dstStride : row*dstStride+width]
		// Edge columns [0, r) keep the scalar variable-bound reference.
		for col := 0; col < interiorStart; col++ {
			dstRow[col] = boxsumCellAVX2(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
		// Interior columns [r, width-1-r] are summed 8 lanes at a time. For
		// these the horizontal window is exactly [col-r, col+r].
		avxCols := interiorCount &^ 7 // round down to multiple of 8
		if avxCols > 0 {
			base := srcOrigin + y0*srcStride + (interiorStart - radius)
			boxsumInteriorBandAVX2Asm(&src[base], &dstRow[interiorStart], uintptr(avxCols), uintptr(radius), uintptr(y1-y0+1), uintptr(srcStride), sq)
		}
		// Remaining interior columns and the trailing r edge columns: scalar.
		for col := interiorStart + avxCols; col < width; col++ {
			dstRow[col] = boxsumCellAVX2(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
	}
}

// boxsumCellAVX2 reproduces one scalar box-sum cell (used for the clamped edge
// columns). It mirrors boxsum exactly.
func boxsumCellAVX2(src []int32, srcOrigin, width, srcStride, radius int, squared bool, col, y0, y1 int) int32 {
	sum := int32(0)
	x0 := maxInt(0, col-radius)
	x1 := minInt(width-1, col+radius)
	if squared {
		for y := y0; y <= y1; y++ {
			base := srcOrigin + y*srcStride
			for _, v := range src[base+x0 : base+x1+1] {
				sum += v * v
			}
		}
	} else {
		for y := y0; y <= y1; y++ {
			base := srcOrigin + y*srcStride
			for _, v := range src[base+x0 : base+x1+1] {
				sum += v
			}
		}
	}
	return sum
}
