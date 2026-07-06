// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package loopfilter

// NEON-accelerated 8-bit wide deblocking kernels (six-, eight- and fourteen-
// sample), for both horizontal and vertical edges. Horizontal kernels process
// eight positions per vector, where every tap row of eight adjacent positions
// is a contiguous eight-byte load. Vertical kernels (taps one byte apart,
// positions stride-separated) transpose eight positions per group through
// per-lane ld4/ld2 so the lane arithmetic is identical, then scatter the
// modified samples back; filter14's vertical path reuses the horizontal kernel
// through a NEON trn-ladder stack transpose (filter14_vtrn_neon_arm64.s). Any
// tail shorter than eight positions and all high bit depth route through the
// pure-Go reference.
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

// filter14NEONCtx is the asm calling context for the fourteen-sample kernel.
// It carries fourteen tap pointers (p6..q6) so the asm can load the full window
// for the flat8out wide path while still falling back to the filter8/filter4
// updates for lanes that are not flat-in-and-out.
type filter14NEONCtx struct {
	p6     *byte // pointer to first p6 sample (q0Base - 7*step)
	p5     *byte
	p4     *byte
	p3     *byte
	p2     *byte
	p1     *byte
	p0     *byte
	q0     *byte
	q1     *byte
	q2     *byte
	q3     *byte
	q4     *byte
	q5     *byte
	q6     *byte
	count  uintptr
	limit  int64
	blimit int64
	hev    int64
	thr    int64
}

// wideVertNEONCtx is the shared asm calling context for the vertical-edge wide
// kernels (filter6/8/14). A vertical edge has its taps contiguous within a row
// while successive positions step by the row stride, so each kernel transposes
// via per-lane ld4 (gathering taps into lane vectors) and scatters the modified
// samples back with per-lane single-byte stores. base points at the lowest tap
// of the first position (p2/p3/p6 respectively) and stride is the byte distance
// between adjacent positions. Field order is part of the ABI shared with
// filter_wide_neon_arm64.s; do not reorder.
type wideVertNEONCtx struct {
	base   *byte
	stride uintptr
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

//go:noescape
func filter14EdgeNEONAsm(ctx *filter14NEONCtx)

// wideVertTransposeCtx is the asm calling context for the vertical-edge
// transpose kernels (the 8-bit and 16-bit fourteen-sample variants and the
// 16-bit six/eight-sample variants all share the layout). src points at the
// lowest tap byte (p2/p3/p6 respectively) of the first position, stride is
// the byte distance between adjacent positions, scratch is the base of the
// horizontal-layout scratch (row stride 32 bytes at 8-bit, 64 at 16-bit), and
// count is the number of eight-position groups (1..4). Field order is part of
// the ABI shared with filter14_vtrn_neon_arm64.s and
// filter_wide16_vtrn_neon_arm64.s; do not reorder.
type wideVertTransposeCtx struct {
	src     *byte
	stride  uintptr
	scratch *byte
	count   uintptr
}

//go:noescape
func filter6VertNEONAsm(ctx *wideVertNEONCtx)

//go:noescape
func filter8VertNEONAsm(ctx *wideVertNEONCtx)

//go:noescape
func filter14VertGatherNEONAsm(ctx *wideVertTransposeCtx)

//go:noescape
func filter14VertScatterNEONAsm(ctx *wideVertTransposeCtx)

//go:noescape
func filter14Vert16GatherNEONAsm(ctx *wideVertTransposeCtx)

//go:noescape
func filter14Vert16ScatterNEONAsm(ctx *wideVertTransposeCtx)

// filter6VertNEON and filter8VertNEON accelerate the six/eight-sample filters
// along vertical edges (step == 1, positions stride-separated). Each transposes
// eight positions per group through ld4 so the lane arithmetic is identical to
// the corresponding horizontal kernel. Tails shorter than eight positions route
// through the pure-Go reference. The contexts are built and passed inline (no
// function-pointer indirection) so they stay stack-allocated.
func filter6VertNEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 1 || groups == 0 {
		filter6EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := wideVertNEONCtx{
		base:   &pix[q0Base-3], // p2 of the first position
		stride: uintptr(outer),
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
		thr:    int64(scale),
	}
	filter6VertNEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter6EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

func filter8VertNEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 1 || groups == 0 {
		filter8EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := wideVertNEONCtx{
		base:   &pix[q0Base-4], // p3 of the first position
		stride: uintptr(outer),
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
		thr:    int64(scale),
	}
	filter8VertNEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter8EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

// filter14VertBatchGroups is the number of eight-position groups gathered per
// scratch fill; the scratch row stride is 8*filter14VertBatchGroups bytes so a
// single filter14EdgeNEONAsm call (whose tap pointers advance eight bytes per
// group) processes the whole batch.
const filter14VertBatchGroups = 4

// filter14VertNEON accelerates the fourteen-sample filter along vertical edges
// by transposing batches of up to four eight-position groups into a contiguous
// scratch buffer laid out as a horizontal edge, running the validated
// horizontal filter14 NEON kernel on it (identical arithmetic, no separate
// vertical filter asm to audit), then scattering the modified samples back.
// The gather/scatter transposes run in NEON trn1/trn2 ladders
// (filter14_vtrn_neon_arm64.s, ported from dav1d src/arm/64/loopfilter.S
// lpf_h_16_16_neon); the scratch lives on the stack so the path stays
// allocation-free, and byte-exactness follows by construction because the
// filter kernel consumes exactly the bytes the scalar transpose used to feed
// it.
func filter14VertNEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 1 || groups == 0 {
		filter14EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	// scratch holds up to filter14VertBatchGroups groups of 8 positions laid
	// out as a horizontal edge: 14 tap rows of scratchStride contiguous bytes,
	// group g occupying bytes 8g..8g+7 of every row, so the horizontal kernel
	// sees outer == 1 and step == scratchStride.
	const scratchStride = 8 * filter14VertBatchGroups
	var scratch [14 * scratchStride]byte
	for g := 0; g < groups; g += filter14VertBatchGroups {
		n := groups - g
		if n > filter14VertBatchGroups {
			n = filter14VertBatchGroups
		}
		colBase := q0Base + g*8*outer
		ctx := wideVertTransposeCtx{
			src:     &pix[colBase-7], // p6 of the first position
			stride:  uintptr(outer),
			scratch: &scratch[0],
			count:   uintptr(n),
		}
		// Gather: scratch[t*scratchStride + 8b+j] = pix[colBase + (8b+j)*outer + (t-7)]
		// for t in 0..13, b in 0..n-1, j in 0..7.
		filter14VertGatherNEONAsm(&ctx)
		sq0 := 7 * scratchStride // q0 tap row
		filter14EdgeNEON(scratch[:], sq0, scratchStride, 1, n*8, scale, params)
		// Scatter back the twelve modifiable samples p5..q5 (tap rows 1..12).
		filter14VertScatterNEONAsm(&ctx)
	}
	if rem := length - groups*8; rem > 0 {
		filter14EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

func filter6EdgeNEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 1 || groups == 0 {
		if step == 1 {
			filter6VertNEON(pix, q0Base, step, outer, length, scale, params)
			return
		}
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
		if step == 1 {
			filter8VertNEON(pix, q0Base, step, outer, length, scale, params)
			return
		}
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

func filter14EdgeNEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 1 || groups == 0 {
		if step == 1 {
			filter14VertNEON(pix, q0Base, step, outer, length, scale, params)
			return
		}
		filter14EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter14NEONCtx{
		p6:     &pix[q0Base-7*step],
		p5:     &pix[q0Base-6*step],
		p4:     &pix[q0Base-5*step],
		p3:     &pix[q0Base-4*step],
		p2:     &pix[q0Base-3*step],
		p1:     &pix[q0Base-2*step],
		p0:     &pix[q0Base-step],
		q0:     &pix[q0Base],
		q1:     &pix[q0Base+step],
		q2:     &pix[q0Base+2*step],
		q3:     &pix[q0Base+3*step],
		q4:     &pix[q0Base+4*step],
		q5:     &pix[q0Base+5*step],
		q6:     &pix[q0Base+6*step],
		count:  uintptr(groups),
		limit:  int64(params.limit),
		blimit: int64(params.blimit),
		hev:    int64(params.hev),
		thr:    int64(scale),
	}
	filter14EdgeNEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter14EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
