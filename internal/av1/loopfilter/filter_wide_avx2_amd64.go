// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

// AVX2-accelerated 8-bit wide deblocking kernels (six-, eight- and fourteen-
// sample). The .s file processes sixteen horizontal edge positions per 256-bit
// vector, where every tap row of sixteen adjacent positions is a contiguous
// sixteen-byte load widened to sixteen int16 lanes with VPMOVZXBW. Vertical
// edges (whose taps are strided while positions advance by the row stride), high
// bit depth, and any tail shorter than sixteen positions route through the
// pure-Go reference — matching the horizontal-only scope of the AVX2 narrow
// (filter4) kernel.
//
// Bit-exactness with the pure-Go reference:
//   - all arithmetic runs in signed 16-bit lanes; 8-bit samples are 0..255 and
//     the centred ps/qs values are -128..127, and the flat-path weighted sums
//     (max p6*7 + ... == 4080 for filter14) all fit in int16.
//   - needsFilter6/8 is reproduced lane-wise (VPABSW abs diffs, *2 via VPSLLW
//     and /2 via VPSRAW, compared against the broadcast limit/blimit with
//     VPCMPGTW) to form a need mask.
//   - the flat / flat2 decisions are reproduced lane-wise against the broadcast
//     flat threshold (== scale; 1 at 8-bit) to form flat masks.
//   - not-flat lanes fall through to the exact filter4 narrow update (filter6),
//     the filter8 flat/narrow update (filter14's inner branch), etc.; the flat
//     lanes use roundPowerOfTwo(v, n) = (v + (1<<(n-1))) >> n on non-negative
//     sums, computed as VPADDW of the rounding bias then VPSRAW.
//   - every tap is blended back through the need / flat / flat2 / hev masks with
//     VPBLENDVB so each conditional write matches the reference's branch
//     structure exactly, then the sixteen int16 results are narrowed to sixteen
//     contiguous bytes with VPACKUSWB plus a VPERMQ lane fix-up.

// filter6AVX2Ctx is the asm calling context for the six-sample kernel. Field
// order and sizes are part of the ABI shared with filter_wide_avx2_amd64.s; do
// not reorder.
type filter6AVX2Ctx struct {
	p2     *byte // pointer to first p2 sample (q0Base - 3*step)
	p1     *byte
	p0     *byte
	q0     *byte
	q1     *byte
	q2     *byte
	count  uintptr // number of 16-position groups to process
	limit  int64
	blimit int64
	hev    int64
	thr    int64 // flat threshold (== scale; 1 at 8-bit)
}

// filter8AVX2Ctx is the asm calling context for the eight-sample kernel.
type filter8AVX2Ctx struct {
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

// filter14AVX2Ctx is the asm calling context for the fourteen-sample kernel. It
// carries fourteen tap pointers (p6..q6) so the asm can load the full window for
// the flat8out wide path while still falling back to the filter8/filter4 updates
// for lanes that are not flat-in-and-out.
type filter14AVX2Ctx struct {
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

//go:noescape
func filter6EdgeAVX2Asm(ctx *filter6AVX2Ctx)

//go:noescape
func filter8EdgeAVX2Asm(ctx *filter8AVX2Ctx)

//go:noescape
func filter14EdgeAVX2Asm(ctx *filter14AVX2Ctx)

// filter6EdgeAVX2 accelerates the 8-bit six-sample filter for horizontal edges
// in groups of sixteen positions (outer == 1, so the taps form contiguous
// loads). Vertical edges (outer != 1) and any tail shorter than sixteen
// positions route through the pure-Go reference.
func filter6EdgeAVX2(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 16
	if outer != 1 || groups == 0 {
		filter6EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter6AVX2Ctx{
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
	filter6EdgeAVX2Asm(&ctx)
	if rem := length - groups*16; rem > 0 {
		filter6EdgePureGo(pix, q0Base+groups*16*outer, step, outer, rem, scale, params)
	}
}

// filter8EdgeAVX2 accelerates the 8-bit eight-sample filter for horizontal edges
// in groups of sixteen positions. Vertical edges and short tails route through
// the pure-Go reference.
func filter8EdgeAVX2(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 16
	if outer != 1 || groups == 0 {
		filter8EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter8AVX2Ctx{
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
	filter8EdgeAVX2Asm(&ctx)
	if rem := length - groups*16; rem > 0 {
		filter8EdgePureGo(pix, q0Base+groups*16*outer, step, outer, rem, scale, params)
	}
}

// filter14EdgeAVX2 accelerates the 8-bit fourteen-sample filter for horizontal
// edges in groups of sixteen positions. Vertical edges and short tails route
// through the pure-Go reference.
func filter14EdgeAVX2(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 16
	if outer != 1 || groups == 0 {
		filter14EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	ctx := filter14AVX2Ctx{
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
	filter14EdgeAVX2Asm(&ctx)
	if rem := length - groups*16; rem > 0 {
		filter14EdgePureGo(pix, q0Base+groups*16*outer, step, outer, rem, scale, params)
	}
}
