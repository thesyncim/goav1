// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

// AVX2-accelerated 10/12-bit narrow (four-sample) deblocking kernel. The .s file
// processes sixteen edge positions per 256-bit vector for horizontal edges,
// where the four taps (p1, p0, q0, q1) of sixteen adjacent two-byte positions
// are each a contiguous thirty-two-byte load. Vertical edges (whose taps are
// strided while positions advance by the row stride) and any tail shorter than
// sixteen positions route through the pure-Go reference.
//
// Bit-exactness with filter4Edge16PureGo:
//   - 10/12-bit samples (0..4095) and the centred ps/qs values (-2048..2047) fit
//     in signed int16, so all lane arithmetic runs in sixteen int16 lanes with no
//     widening step; loads/stores are plain VMOVDQU of the two-byte samples.
//   - needsFilter4 is reproduced lane-wise (VPABSW abs diffs, *2 via VPSLLW and
//     /2 via VPSRAW, compared against the broadcast limit/blimit with VPCMPGTW).
//   - hev is reproduced lane-wise as a second mask.
//   - the clamps to [min, max] and the +center offset use VPMINSW/VPMAXSW/VPADDW
//     against the broadcast center/min/max the context carries (so the same code
//     serves 10- and 12-bit), and the >>3 / >>1 steps use VPSRAW, matching
//     signedClamp and the integer shifts in filter4Samples exactly.
//   - results are blended back through the need (and, for p1/q1, need&~hev) masks
//     with VPBLENDVB so failing lanes keep their original words, identical to the
//     reference's conditional writes.

// filter4Edge16AVX2Ctx is the asm calling context. Field order and sizes are
// part of the ABI shared with filter4_16_avx2_amd64.s; do not reorder.
type filter4Edge16AVX2Ctx struct {
	p1     *byte // pointer to first p1 sample (q0Base - 2*step)
	p0     *byte
	q0     *byte
	q1     *byte
	count  uintptr // number of 16-position groups to process
	limit  int64
	blimit int64
	hev    int64
	center int64
	min    int64
	max    int64
}

//go:noescape
func filter4Edge16AVX2Asm(ctx *filter4Edge16AVX2Ctx)

// filter4Edge16AVX2 accelerates the 10/12-bit narrow filter for horizontal edges
// in groups of sixteen positions (outer == 2, so the taps form contiguous
// two-byte loads). Vertical edges (outer != 2) and any tail shorter than sixteen
// positions route through the pure-Go reference.
func filter4Edge16AVX2(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	groups := length / 16
	if outer != 2 || groups == 0 {
		filter4Edge16PureGo(pix, q0Base, step, outer, length, params)
		return
	}
	ctx := filter4Edge16AVX2Ctx{
		p1:     &pix[q0Base-2*step],
		p0:     &pix[q0Base-step],
		q0:     &pix[q0Base],
		q1:     &pix[q0Base+step],
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
		center: int64(params.center),
		min:    int64(params.min),
		max:    int64(params.max),
	}
	filter4Edge16AVX2Asm(&ctx)

	if rem := length - groups*16; rem > 0 {
		filter4Edge16PureGo(pix, q0Base+groups*16*outer, step, outer, rem, params)
	}
}
