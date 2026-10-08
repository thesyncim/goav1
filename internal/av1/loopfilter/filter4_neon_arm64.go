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

// filter4Edge16WideNEONAsm processes sixteen horizontal edge positions per
// iteration (byte-domain kernel ported from dav1d loopfilter.S lpf_16_wd4); it
// shares the filter4NEONCtx layout with the eight-wide kernel above.
//
//go:noescape
func filter4Edge16WideNEONAsm(ctx *filter4NEONCtx)

// filter4VertNEONCtx is the asm calling context for the 8-bit vertical-edge
// kernel. A vertical edge has its four taps (p1,p0,q0,q1) contiguous in one row
// while successive positions step by the row stride, so the asm transposes via
// per-lane ld4/st4: base points at the first position's p1 byte and stride is
// the byte distance between adjacent positions. Field order is part of the ABI
// shared with filter4_neon_arm64.s; do not reorder.
type filter4VertNEONCtx struct {
	base   *byte   // pointer to p1 of the first position (q0Base - 2)
	stride uintptr // byte stride between successive positions along the edge
	count  uintptr // number of 8-position groups to process
	limit  int64
	blimit int64
	hev    int64
}

//go:noescape
func filter4VertNEONAsm(ctx *filter4VertNEONCtx)

// filter4VertNEON accelerates the 8-bit narrow filter for vertical edges, where
// the four taps are one byte apart (step == 1) and positions advance by the row
// stride. It transposes eight positions per group through ld4/st4 so the lane
// arithmetic is identical to the horizontal kernel. Any tail shorter than eight
// positions routes through the pure-Go reference.
func filter4VertNEON(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	groups := length / 8
	if step != 1 || groups == 0 {
		filter4EdgePureGo(pix, q0Base, step, outer, length, params)
		return
	}
	ctx := filter4VertNEONCtx{
		base:   &pix[q0Base-2],
		stride: uintptr(outer),
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
	}
	filter4VertNEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter4EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, params)
	}
}

// filter4Edge16NEONCtx is the asm calling context for the two-byte (10/12-bit)
// narrow kernel. Field order and sizes are part of the ABI shared with
// filter4_neon_arm64.s; do not reorder. The centre and clamp bounds are passed
// in (rather than fixed at the 8-bit values) so the same lane arithmetic serves
// both 10- and 12-bit planes.
type filter4Edge16NEONCtx struct {
	p1     *byte // pointer to first p1 sample (q0Base - 2*step)
	p0     *byte
	q0     *byte
	q1     *byte
	count  uintptr // number of 8-position groups to process
	limit  int64
	blimit int64
	hev    int64
	center int64
	min    int64
	max    int64
}

//go:noescape
func filter4Edge16NEONAsm(ctx *filter4Edge16NEONCtx)

// filter4Edge16NEON accelerates the 10/12-bit narrow filter for horizontal
// edges (contiguous two-byte taps). Vertical edges (strided taps) and any tail
// shorter than eight positions route through the pure-Go reference.
func filter4Edge16NEON(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	groups := length / 8
	if outer != 2 || groups == 0 {
		filter4Edge16PureGo(pix, q0Base, step, outer, length, params)
		return
	}
	ctx := filter4Edge16NEONCtx{
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
	filter4Edge16NEONAsm(&ctx)

	if rem := length - groups*8; rem > 0 {
		filter4Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, params)
	}
}

func filter4EdgeNEON(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	// Bit depth is fixed at 8 here (callers gate on bytesPerSample == 1), so
	// center == 128 and the clamp range is [-128,127]. Horizontal edges have a
	// one-byte per-position stride (outer == 1) so eight positions form
	// contiguous loads; vertical edges have one-byte taps (step == 1) and route
	// through the transposing ld4/st4 kernel.
	if outer != 1 || length < 8 {
		if step == 1 {
			filter4VertNEON(pix, q0Base, step, outer, length, params)
			return
		}
		filter4EdgePureGo(pix, q0Base, step, outer, length, params)
		return
	}

	// Horizontal 8-bit edge: run the byte-domain sixteen-wide kernel over the
	// bulk, then the eight-wide kernel for at most one residual group of eight,
	// then the pure-Go reference for the sub-eight tail.
	//
	// The sixteen-wide kernel evaluates needsFilter4 in the byte domain, where
	// |p0-q0|*2 + |p1-q1|/2 saturates at 255; that is bit-exact only while the
	// block limit stays below 255. Every 8-bit block limit the decoder derives
	// is a uint8 (BlockLimit = 2*(level+2)+limit <= 193), so the fast path always
	// engages in production; any out-of-range synthetic threshold falls back to
	// the exact eight-wide kernel below.
	done := 0
	if groups16 := length / 16; params.blimit < 255 && groups16 > 0 {
		ctx := filter4NEONCtx{
			p1:     &pix[q0Base-2*step],
			p0:     &pix[q0Base-step],
			q0:     &pix[q0Base],
			q1:     &pix[q0Base+step],
			outer:  uintptr(outer),
			count:  uintptr(groups16),
			limit:  int64(params.limit),
			blimit: int64(params.blimit),
			hev:    int64(params.hev),
		}
		filter4Edge16WideNEONAsm(&ctx)
		done = groups16 * 16
	}
	if rem := length - done; rem >= 8 {
		base := q0Base + done*outer
		ctx := filter4NEONCtx{
			p1:     &pix[base-2*step],
			p0:     &pix[base-step],
			q0:     &pix[base],
			q1:     &pix[base+step],
			outer:  uintptr(outer),
			count:  uintptr(rem / 8),
			limit:  int64(params.limit),
			blimit: int64(params.blimit),
			hev:    int64(params.hev),
		}
		filter4EdgeNEONAsm(&ctx)
		done += (rem / 8) * 8
	}
	if rem := length - done; rem > 0 {
		filter4EdgePureGo(pix, q0Base+done*outer, step, outer, rem, params)
	}
}
