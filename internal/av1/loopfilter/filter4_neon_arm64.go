// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package loopfilter

// NEON-accelerated 8-bit narrow (four-sample) deblocking kernel. The .s file
// processes eight edge positions per vector for horizontal edges, where the
// four taps (p1, p0, q0, q1) of eight adjacent positions are each a contiguous
// eight-byte load. Vertical edges (whose taps are strided one byte apart while
// positions advance by the row stride) and any tail shorter than eight
// positions route through the pure-Go reference, keeping the asm to a single
// contiguous-load code path and the byte-exactness contract easy to audit.
//
// Bit-exactness with filter4EdgePureGo:
//   - all arithmetic runs in signed 16-bit lanes; samples are 0..255 and the
//     centred ps/qs values are -128..127, all comfortably inside int16.
//   - needsFilter4 is reproduced lane-wise (abs diffs, the *2 and /2 terms via
//     shifts, compared against the broadcast limit/blimit) to form a mask.
//   - hev is reproduced lane-wise as a second mask.
//   - the clamps to [-128, 127] use SMIN/SMAX against broadcast constants and
//     the >>3 / >>1 steps use arithmetic right shifts, matching signedClamp and
//     the integer shifts in filter4Samples exactly.
//   - results are blended back with the original samples through the masks so
//     positions that fail needsFilter4 (or, for p1/q1, that are hev) keep their
//     original bytes, identical to the reference's conditional writes.

// filter4NEONCtx is the asm calling context. Field order and sizes are part of
// the ABI shared with filter4_neon_arm64.s; do not reorder.
type filter4NEONCtx struct {
	p1     *byte // pointer to first p1 sample (q0Base - 2*step)
	p0     *byte
	q0     *byte
	q1     *byte
	outer  uintptr // byte stride between successive positions along the edge
	count  uintptr // number of 8-position groups to process
	limit  int64
	blimit int64
	hev    int64
}

//go:noescape
func filter4EdgeNEONAsm(ctx *filter4NEONCtx)

func filter4EdgeNEON(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	// The NEON path handles horizontal edges only: there the per-position
	// stride (outer) is one byte, so eight positions form contiguous loads.
	// Vertical edges (outer == stride, step == 1) need a gather/transpose and
	// stay on the reference. Bit depth is fixed at 8 here (callers gate on
	// bytesPerSample == 1), so center == 128 and the clamp range is [-128,127].
	groups := length / 8
	if outer != 1 || groups == 0 {
		filter4EdgePureGo(pix, q0Base, step, outer, length, params)
		return
	}

	ctx := filter4NEONCtx{
		p1:     &pix[q0Base-2*step],
		p0:     &pix[q0Base-step],
		q0:     &pix[q0Base],
		q1:     &pix[q0Base+step],
		outer:  uintptr(outer),
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
	}
	filter4EdgeNEONAsm(&ctx)

	// Tail positions that do not fill a full vector run on the reference.
	if rem := length - groups*8; rem > 0 {
		filter4EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, params)
	}
}
