// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package loopfilter

// NEON-accelerated 8-bit wide deblocking kernels (six- and eight-sample). Each
// processes eight edge positions per vector for horizontal edges, where every
// tap row of eight adjacent positions is a contiguous eight-byte load. Vertical
// edges (strided taps), any tail shorter than eight positions, and high bit
// depth all route through the pure-Go reference, keeping each asm path to a
// single contiguous-load code path.
//
// Bit-exactness with the pure-Go reference:
//   - all arithmetic runs in signed 16-bit lanes; samples are 0..255 and the
//     centred ps/qs values are -128..127, all inside int16. The flat-path
//     weighted sums (max p6*7 + ... for filter8/14) also fit in int16.
//   - needsFilter6/8 is reproduced lane-wise (abs diffs, *2 and /2 via shifts,
//     compared against broadcast limit/blimit) to form a need mask.
//   - the flat decision is reproduced lane-wise against the broadcast flat
//     threshold (scale, i.e. 1 at 8-bit) to form a flat mask.
//   - the not-flat lanes fall through to the exact filter4 narrow update; the
//     flat lanes use srshr #3 (round-to-nearest, ties up) on non-negative sums,
//     matching roundPowerOfTwo(v, 3) = (v + 4) >> 3 for v >= 0.
//   - results are blended back through need & flat masks so every conditional
//     write matches the reference's branch structure exactly.

// filter6NEONCtx is the asm calling context. Field order and sizes are part of
// the ABI shared with filter_wide_neon_arm64.s; do not reorder.
type filter6NEONCtx struct {
	p2     *byte // pointer to first p2 sample (q0Base - 3*step)
	p1     *byte
	p0     *byte
	q0     *byte
	q1     *byte
	q2     *byte
	count  uintptr // number of 8-position groups to process
	limit  int64
	blimit int64
	hev    int64
	thr    int64 // flat threshold (== scale; 1 at 8-bit)
}

// filter8NEONCtx is the asm calling context for the eight-sample kernel.
type filter8NEONCtx struct {
	p3     *byte // pointer to first p3 sample (q0Base - 4*step)
	p2     *byte
	p1     *byte
	p0     *byte
	q0     *byte
	q1     *byte
	q2     *byte
	q3     *byte
	count  uintptr
	limit  int64
	blimit int64
	hev    int64
	thr    int64
}

//go:noescape
func filter6EdgeNEONAsm(ctx *filter6NEONCtx)

//go:noescape
func filter8EdgeNEONAsm(ctx *filter8NEONCtx)

func filter6EdgeNEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 1 || groups == 0 {
		filter6EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter6NEONCtx{
		p2:     &pix[q0Base-3*step],
		p1:     &pix[q0Base-2*step],
		p0:     &pix[q0Base-step],
		q0:     &pix[q0Base],
		q1:     &pix[q0Base+step],
		q2:     &pix[q0Base+2*step],
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
		thr:    int64(scale),
	}
	filter6EdgeNEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter6EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

func filter8EdgeNEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 1 || groups == 0 {
		filter8EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter8NEONCtx{
		p3:     &pix[q0Base-4*step],
		p2:     &pix[q0Base-3*step],
		p1:     &pix[q0Base-2*step],
		p0:     &pix[q0Base-step],
		q0:     &pix[q0Base],
		q1:     &pix[q0Base+step],
		q2:     &pix[q0Base+2*step],
		q3:     &pix[q0Base+3*step],
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
		thr:    int64(scale),
	}
	filter8EdgeNEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter8EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
