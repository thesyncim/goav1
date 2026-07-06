// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

// AVX2-accelerated final self-guided projection for 8-bit pixels, the amd64
// counterpart of selfguided_u8_neon_arm64.go (dav1d sgr_weighted2 8bpc role,
// libaom arithmetic). Per 8 pixels:
//
//	u  = s << SGRProjRstBits                  (VPMOVZXBD + VPSLLD $4)
//	v  = u << SGRProjPrjBits                  (VPSLLD $11 on s)
//	v += xq0*(f0-u) + xq1*(f1-u)              (VPSUBD + VPMULLD + VPADDD)
//	r  = (v + 1<<10) >> 11                    (VPADDD bias + VPSRAD $11)
//	w  = int32(int16(r))                      (VPSLLD $16 + VPSRAD $16)
//	d  = uint8(clampInt32(w, 0, 255))         (VPMAXSD/VPMINSD + packs)
//
// Every step wraps/shifts exactly like Go int32 arithmetic (VPADDD wraps in
// two's complement just as roundPowerOfTwo's addition does), so the output is
// bit-identical to sgrWeightedRowU8 for all inputs.

// sgrWeightedU8AVX2Ctx is the asm calling context for one projected output
// row. Field order and sizes are part of the ABI shared with
// selfguided_u8_avx2_amd64.s; do not reorder.
type sgrWeightedU8AVX2Ctx struct {
	dst  *uint8 // output row base (col 0)
	src  *uint8 // source row base (col 0)
	f0   *int32 // flt0 row base (col 0)
	f1   *int32 // flt1 row base (col 0)
	cols uintptr
	xq0  int32
	xq1  int32
	bias int32 // rounding bias 1 << (SGRProjPrjBits+SGRProjRstBits-1)
	maxv int32 // upper clamp (255)
}

//go:noescape
func sgrWeightedRowU8AVX2Asm(ctx *sgrWeightedU8AVX2Ctx)

// sgrWeightedRowU8AVX2 is the dispatch entry for the 8-bit final projection
// row. Full 8-wide column groups run through the AVX2 asm; the <8 column tail
// keeps the scalar reference.
func sgrWeightedRowU8AVX2(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	simd := width &^ 7
	if simd > 0 {
		ctx := sgrWeightedU8AVX2Ctx{
			dst:  &dst[0],
			src:  &src[0],
			f0:   &f0[0],
			f1:   &f1[0],
			cols: uintptr(simd),
			xq0:  xq0,
			xq1:  xq1,
			bias: 1 << (SGRProjPrjBits + SGRProjRstBits - 1),
			maxv: 255,
		}
		sgrWeightedRowU8AVX2Asm(&ctx)
	}
	if simd < width {
		sgrWeightedRowU8(dst[simd:], src[simd:], f0[simd:], f1[simd:], xq0, xq1)
	}
}
