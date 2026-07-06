// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

// NEON-accelerated final self-guided projection for 8-bit pixels, in the role
// of dav1d's sgr_weighted2_8bpc (src/looprestoration_tmpl.c sgr_weighted2 and
// its NEON port: weights applied to the two filtered planes, uint8 store) but
// keeping libaom's exact arithmetic. Per 8 pixels:
//
//	u  = s << SGRProjRstBits                      (USHLL #4 on the u8 samples)
//	v  = u << SGRProjPrjBits                      (SHL #7)
//	v += xq0*(f0-u) + xq1*(f1-u)                  (SUB + MLA, s32 lanes)
//	r  = roundPowerOfTwo(v, 11)                   (SRSHR #11)
//	w  = int32(int16(r))                          (XTN: truncate to 16 bits)
//	d  = uint8(clampInt32(w, 0, 255))             (SQXTUN: s16 -> u8 saturate)
//
// Bit-exactness with sgrWeightedRowU8: NEON s32 SUB/SHL/MLA wrap exactly like
// Go int32 arithmetic; SRSHR #11 computes (v + 1<<10) >> 11 without
// intermediate wrap, which matches roundPowerOfTwo for every v the SGR
// pipeline can produce (|v| < 2^24 for 8-bit input, nowhere near int32
// overflow); XTN keeps the low 16 bits, reproducing libaom's int16 cast; and
// SQXTUN saturates the signed 16-bit lanes to [0,255], identical to the final
// clamp. The unused-filter case needs no branches: DecodeSGRXQ pins the
// unused xq to 0 (see ApplySelfguidedRestorationU8).

// sgrWeightedU8Ctx is the asm calling context for one projected output row.
// Field order and sizes are part of the ABI shared with
// selfguided_u8_neon_arm64.s; do not reorder.
type sgrWeightedU8Ctx struct {
	dst  *uint8 // output row base (col 0)
	src  *uint8 // source row base (col 0)
	f0   *int32 // flt0 row base (col 0)
	f1   *int32 // flt1 row base (col 0)
	cols uintptr
	xq0  int32
	xq1  int32
}

//go:noescape
func sgrWeightedRowU8NEONAsm(ctx *sgrWeightedU8Ctx)

// sgrWeightedRowU8NEON is the dispatch entry for the 8-bit final projection
// row. Full 8-wide column groups run through the NEON asm; the <8 column tail
// keeps the scalar reference.
func sgrWeightedRowU8NEON(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	neon := width &^ 7
	if neon > 0 {
		ctx := sgrWeightedU8Ctx{
			dst:  &dst[0],
			src:  &src[0],
			f0:   &f0[0],
			f1:   &f1[0],
			cols: uintptr(neon),
			xq0:  xq0,
			xq1:  xq1,
		}
		sgrWeightedRowU8NEONAsm(&ctx)
	}
	if neon < width {
		sgrWeightedRowU8(dst[neon:], src[neon:], f0[neon:], f1[neon:], xq0, xq1)
	}
}
