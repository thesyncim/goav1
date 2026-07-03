// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package loopfilter

// NEON-accelerated 10-bit (two-byte sample) wide deblocking kernels (six-,
// eight- and fourteen-sample). These mirror the 8-bit wide NEON kernels
// (filter_wide_neon_arm64.s / .go): the 8-bit kernels already run every step in
// signed 16-bit lanes after widening the byte loads, so the 10-bit variants are
// the same instruction stream with contiguous .8h loads/stores and the
// center/min/max clamp bounds taken from the context. Horizontal edges (taps
// row-separated, positions two bytes apart) load eight positions per vector as a
// contiguous sixteen-byte load; vertical edges (taps two bytes apart, positions
// stride-separated) transpose eight positions into a contiguous stack scratch,
// run the proven horizontal kernel, and scatter the modified samples back.
//
// Only 10-bit is routed here: at 10-bit every partial sum stays inside int16
// (max coefficient weight 16 for filter14 times max sample 1023 = 16368 <
// 32767), so the 8-bit lane layout is reused verbatim and is byte-exact with the
// pure-Go reference. 12-bit (whose fourteen-tap sums overflow int16) and any
// tail shorter than eight positions route through the pure-Go reference.

// filter6Edge16NEONCtx is the asm calling context for the 10-bit six-sample
// kernel. Field order and sizes are part of the ABI shared with
// filter_wide16_neon_arm64.s; do not reorder.
type filter6Edge16NEONCtx struct {
	p2     *byte
	p1     *byte
	p0     *byte
	q0     *byte
	q1     *byte
	q2     *byte
	count  uintptr
	limit  int64
	blimit int64
	hev    int64
	thr    int64
	center int64
	min    int64
	max    int64
}

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

// filter14Edge16NEONCtx is the asm calling context for the 10-bit
// fourteen-sample kernel. Field order and sizes are part of the ABI shared with
// filter_wide16_neon_arm64.s; do not reorder.
type filter14Edge16NEONCtx struct {
	p6     *byte
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
	center int64
	min    int64
	max    int64
}

//go:noescape
func filter6Edge16NEONAsm(ctx *filter6Edge16NEONCtx)

//go:noescape
func filter8Edge16NEONAsm(ctx *filter8Edge16NEONCtx)

//go:noescape
func filter14Edge16NEONAsm(ctx *filter14Edge16NEONCtx)

// highbdNEONCenter10 is the centre offset for 10-bit samples (1<<9). The 10-bit
// path is the only high bit depth accelerated by NEON; params.center identifies
// it without threading a separate bit-depth argument through the dispatch slot.
const highbdNEONCenter10 = 512

func filter6Edge16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if int(params.center) != highbdNEONCenter10 || groups == 0 {
		filter6Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	if outer != 2 {
		if step == 2 {
			filter6Vert16NEON(pix, q0Base, step, outer, length, scale, params)
			return
		}
		filter6Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter6Edge16NEONCtx{
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
		center: int64(params.center),
		min:    int64(params.min),
		max:    int64(params.max),
	}
	filter6Edge16NEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter6Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

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

func filter14Edge16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if int(params.center) != highbdNEONCenter10 || groups == 0 {
		filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	if outer != 2 {
		if step == 2 {
			filter14Vert16NEON(pix, q0Base, step, outer, length, scale, params)
			return
		}
		filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter14Edge16NEONCtx{
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
		center: int64(params.center),
		min:    int64(params.min),
		max:    int64(params.max),
	}
	filter14Edge16NEONAsm(&ctx)
	if rem := length - groups*8; rem > 0 {
		filter14Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

// Vertical-edge kernels: taps are two bytes apart (step == 2) while positions
// advance by the row stride (outer). Each transposes eight positions per group
// into a contiguous 16-bit scratch laid out as a horizontal edge (tap rows
// scratchStride bytes apart, positions two bytes apart), runs the proven
// horizontal kernel on it, then scatters the modifiable samples back. The
// scratch lives on the stack so the path stays allocation-free, and byte-exact
// with the reference follows by construction (identical inputs and arithmetic).

const wide16ScratchStride = 16 // 8 positions * 2 bytes

func filter6Vert16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	var scratch [6 * wide16ScratchStride]byte
	for g := 0; g < groups; g++ {
		colBase := q0Base + g*8*outer
		for j := 0; j < 8; j++ {
			src := colBase + j*outer - 3*step // p2 of position j
			for t := 0; t < 6; t++ {
				o := src + t*step
				d := t*wide16ScratchStride + j*2
				scratch[d] = pix[o]
				scratch[d+1] = pix[o+1]
			}
		}
		filter6Edge16NEON(scratch[:], 3*wide16ScratchStride, wide16ScratchStride, 2, 8, scale, params)
		for j := 0; j < 8; j++ {
			dst := colBase + j*outer - 3*step
			for t := 1; t <= 4; t++ { // p1..q1
				o := dst + t*step
				d := t*wide16ScratchStride + j*2
				pix[o] = scratch[d]
				pix[o+1] = scratch[d+1]
			}
		}
	}
	if rem := length - groups*8; rem > 0 {
		filter6Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

func filter8Vert16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	var scratch [8 * wide16ScratchStride]byte
	for g := 0; g < groups; g++ {
		colBase := q0Base + g*8*outer
		for j := 0; j < 8; j++ {
			src := colBase + j*outer - 4*step // p3 of position j
			for t := 0; t < 8; t++ {
				o := src + t*step
				d := t*wide16ScratchStride + j*2
				scratch[d] = pix[o]
				scratch[d+1] = pix[o+1]
			}
		}
		filter8Edge16NEON(scratch[:], 4*wide16ScratchStride, wide16ScratchStride, 2, 8, scale, params)
		for j := 0; j < 8; j++ {
			dst := colBase + j*outer - 4*step
			for t := 1; t <= 6; t++ { // p2..q2
				o := dst + t*step
				d := t*wide16ScratchStride + j*2
				pix[o] = scratch[d]
				pix[o+1] = scratch[d+1]
			}
		}
	}
	if rem := length - groups*8; rem > 0 {
		filter8Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

func filter14Vert16NEON(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	var scratch [14 * wide16ScratchStride]byte
	for g := 0; g < groups; g++ {
		colBase := q0Base + g*8*outer
		for j := 0; j < 8; j++ {
			src := colBase + j*outer - 7*step // p6 of position j
			for t := 0; t < 14; t++ {
				o := src + t*step
				d := t*wide16ScratchStride + j*2
				scratch[d] = pix[o]
				scratch[d+1] = pix[o+1]
			}
		}
		filter14Edge16NEON(scratch[:], 7*wide16ScratchStride, wide16ScratchStride, 2, 8, scale, params)
		for j := 0; j < 8; j++ {
			dst := colBase + j*outer - 7*step
			for t := 1; t <= 12; t++ { // p5..q5
				o := dst + t*step
				d := t*wide16ScratchStride + j*2
				pix[o] = scratch[d]
				pix[o+1] = scratch[d+1]
			}
		}
	}
	if rem := length - groups*8; rem > 0 {
		filter14Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
