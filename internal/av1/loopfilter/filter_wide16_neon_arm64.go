// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package loopfilter

// NEON-accelerated 10-bit (two-byte sample) eight-tap horizontal filtering and
// six/eight-tap vertical transpose helpers. Vertical edges transpose eight
// positions into stack scratch with shared NEON gather/scatter routines. The
// HBD6 horizontal assembly core has been replaced by Go SIMD.
//
// The remaining HBD8 horizontal kernel is limited to 10-bit because its
// partial sums must fit in int16. HBD6 and HBD14 horizontal filtering use Go
// SIMD when enabled and scalar Go otherwise; vertical SIMD shares the NEON
// gather/scatter routines. Short tails use the scalar reference.

// filter8Edge16NEONCtx is the asm calling context for the 10-bit eight-sample
// kernel. Field order and sizes are part of the ABI shared with
// filter_wide16_neon_arm64.s; do not reorder.
type filter8Edge16NEONCtx struct {
	p3     *byte
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
	center int64
	min    int64
	max    int64
}

//go:noescape
func filter8Edge16NEONAsm(ctx *filter8Edge16NEONCtx)

// highbdNEONCenter10 is the centre offset for 10-bit samples (1<<9). The 10-bit
// path is the only high bit depth accelerated by NEON; params.center identifies
// it without threading a separate bit-depth argument through the dispatch slot.
const highbdNEONCenter10 = 512

func filter8Edge16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if int(params.center) != highbdNEONCenter10 || groups == 0 {
		filter8Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	if outer != 2 {
		if step == 2 {
			filter8Vert16NEON(pix, q0Base, step, outer, length, scale, params)
			return
		}
		filter8Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter8Edge16NEONCtx{
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
		center: int64(params.center),
		min:    int64(params.min),
		max:    int64(params.max),
	}
	filter8Edge16NEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter8Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

func filter14Edge16PureGoFallback(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
}

// Vertical-edge kernels: taps are two bytes apart (step == 2) while positions
// advance by the row stride (outer). Each gathers batches of up to
// filter14VertBatchGroups eight-position groups into a contiguous 16-bit
// scratch laid out as a horizontal edge (tap rows wide16VertScratchStride
// bytes apart, positions two bytes apart), runs the pure-Go horizontal
// reference on the whole batch, then
// scatters the modifiable samples back. The gather/scatter transposes run in
// NEON .8h/.4s/.2d trn1/trn2 ladders (filter_wide16_vtrn_neon_arm64.s, ported
// from dav1d src/arm/64/loopfilter16.S lpf_h_8_8_neon / lpf_h_6_8_neon +
// util.S transpose_8x8h). The scratch lives on the stack so the paths stay
// allocation-free, and byte-exact with the reference follows by construction
// (the filter kernel consumes exactly the bytes the scalar transpose used to
// feed it).

// wide16VertScratchStride is the scratch row stride shared by the 16-bit
// vertical transpose kernels: 8 positions * 2 bytes per group. The stride is
// hardcoded in filter_wide16_vtrn_neon_arm64.s; keep them in sync.
const wide16VertScratchStride = 16 * filter14VertBatchGroups

//go:noescape
func filter6Vert16GatherNEONAsm(ctx *wideVertTransposeCtx)

//go:noescape
func filter6Vert16ScatterNEONAsm(ctx *wideVertTransposeCtx)

//go:noescape
func filter8Vert16GatherNEONAsm(ctx *wideVertTransposeCtx)

//go:noescape
func filter8Vert16ScatterNEONAsm(ctx *wideVertTransposeCtx)

func filter6Vert16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 2 || groups == 0 {
		filter6Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	var scratch [6 * wide16VertScratchStride]byte
	for g := 0; g < groups; g += filter14VertBatchGroups {
		n := groups - g
		if n > filter14VertBatchGroups {
			n = filter14VertBatchGroups
		}
		colBase := q0Base + g*8*outer
		ctx := wideVertTransposeCtx{
			src:     &pix[colBase-3*step], // p2 of the first position
			stride:  uintptr(outer),
			scratch: &scratch[0],
			count:   uintptr(n),
		}
		filter6Vert16GatherNEONAsm(&ctx)
		filter6Edge16PureGo(scratch[:], 3*wide16VertScratchStride, wide16VertScratchStride, 2, n*8, scale, params)
		// Scatter back the four modifiable samples p1..q1 (tap rows 1..4).
		filter6Vert16ScatterNEONAsm(&ctx)
	}
	if rem := length - groups*8; rem > 0 {
		filter6Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

func filter8Vert16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 2 || groups == 0 {
		filter8Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	var scratch [8 * wide16VertScratchStride]byte
	for g := 0; g < groups; g += filter14VertBatchGroups {
		n := groups - g
		if n > filter14VertBatchGroups {
			n = filter14VertBatchGroups
		}
		colBase := q0Base + g*8*outer
		ctx := wideVertTransposeCtx{
			src:     &pix[colBase-4*step], // p3 of the first position
			stride:  uintptr(outer),
			scratch: &scratch[0],
			count:   uintptr(n),
		}
		filter8Vert16GatherNEONAsm(&ctx)
		filter8Edge16NEON(scratch[:], 4*wide16VertScratchStride, wide16VertScratchStride, 2, n*8, scale, params)
		// Scatter back the six modifiable samples p2..q2 (tap rows 1..6).
		filter8Vert16ScatterNEONAsm(&ctx)
	}
	if rem := length - groups*8; rem > 0 {
		filter8Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

// filter14Vert16NEON batches up to filter14VertBatchGroups eight-position
// groups per scratch fill; the scratch row stride is 16*filter14VertBatchGroups
// bytes then the pure-Go horizontal reference processes each gathered batch. The gather/scatter transposes run in
// NEON .8h trn1/trn2 ladders (filter14_vtrn_neon_arm64.s, ported from dav1d
// src/arm/64/loopfilter16.S lpf_h_16_8_neon + util.S transpose_8x8h), the
// same shape the six/eight-sample vertical paths above use.
func filter14Vert16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 2 || groups == 0 {
		filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	const scratchStride = 16 * filter14VertBatchGroups
	var scratch [14 * scratchStride]byte
	for g := 0; g < groups; g += filter14VertBatchGroups {
		n := groups - g
		if n > filter14VertBatchGroups {
			n = filter14VertBatchGroups
		}
		colBase := q0Base + g*8*outer
		ctx := wideVertTransposeCtx{
			src:     &pix[colBase-7*step], // p6 of the first position
			stride:  uintptr(outer),
			scratch: &scratch[0],
			count:   uintptr(n),
		}
		filter14Vert16GatherNEONAsm(&ctx)
		filter14Edge16PureGoFallback(scratch[:], 7*scratchStride, scratchStride, 2, n*8, scale, params)
		// Scatter back the twelve modifiable samples p5..q5 (tap rows 1..12).
		filter14Vert16ScatterNEONAsm(&ctx)
	}
	if rem := length - groups*8; rem > 0 {
		filter14Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
