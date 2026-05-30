// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

// AVX2-accelerated 8-bit narrow (four-sample) deblocking kernel. The .s file
// processes sixteen edge positions per 256-bit vector for horizontal edges,
// where the four taps (p1, p0, q0, q1) of sixteen adjacent positions are each a
// contiguous sixteen-byte load. Vertical edges (whose taps are strided one byte
// apart while positions advance by the row stride), high bit depth, and any
// tail shorter than sixteen positions route through the pure-Go reference.
//
// Bit-exactness with filter4EdgePureGo:
//   - all arithmetic runs in signed 16-bit lanes (VPMOVZXBW widen, sixteen int16
//     in one ymm); samples are 0..255 and the centred ps/qs values are
//     -128..127, all inside int16.
//   - needsFilter4 is reproduced lane-wise (VPABSW abs diffs, the *2 via VPSLLW
//     and /2 via VPSRAW, compared against the broadcast limit/blimit with
//     VPCMPGTW to form a pass mask).
//   - hev is reproduced lane-wise as a second mask.
//   - the clamps to [-128, 127] use VPMINSW/VPMAXSW against broadcast constants
//     and the >>3 / >>1 steps use VPSRAW arithmetic shifts, matching signedClamp
//     and the integer shifts in filter4Samples exactly.
//   - results are blended back with the original samples through the masks
//     (VPBLENDVB) so positions that fail needsFilter4 (or, for p1/q1, that are
//     hev) keep their original bytes, identical to the reference's conditional
//     writes. The sixteen int16 results are narrowed to sixteen contiguous bytes
//     with VPACKUSWB plus a VPERMQ lane fix-up.

// filter4AVX2Ctx is the asm calling context. Field order and sizes are part of
// the ABI shared with filter4_avx2_amd64.s; do not reorder.
type filter4AVX2Ctx struct {
	p1     *byte // pointer to first p1 sample (q0Base - 2*step)
	p0     *byte
	q0     *byte
	q1     *byte
	count  uintptr // number of 16-position groups to process
	limit  int64
	blimit int64
	hev    int64
}

//go:noescape
func filter4EdgeAVX2Asm(ctx *filter4AVX2Ctx)

// filter4EdgeAVX2 accelerates the 8-bit narrow filter for horizontal edges in
// groups of sixteen positions (outer == 1, so the taps form contiguous loads).
// Vertical edges (outer != 1) and any tail shorter than sixteen positions route
// through the pure-Go reference. Bit depth is fixed at 8 here (callers gate on
// bytesPerSample == 1), so center == 128 and the clamp range is [-128,127].
func filter4EdgeAVX2(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	groups := length / 16
	if outer != 1 || groups == 0 {
		filter4EdgePureGo(pix, q0Base, step, outer, length, params)
		return
	}
	ctx := filter4AVX2Ctx{
		p1:     &pix[q0Base-2*step],
		p0:     &pix[q0Base-step],
		q0:     &pix[q0Base],
		q1:     &pix[q0Base+step],
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
	}
	filter4EdgeAVX2Asm(&ctx)

	if rem := length - groups*16; rem > 0 {
		filter4EdgePureGo(pix, q0Base+groups*16*outer, step, outer, rem, params)
	}
}
