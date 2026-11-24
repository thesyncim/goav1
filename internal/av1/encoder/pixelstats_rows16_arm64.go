//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

// pixelStatsRows16Arch is pixelStatsRows16 with NEON's widening multiply: the
// int16 differences square straight into int32 lanes (SMULL on the low and
// high halves) instead of being sign-extended first, and the signed sum
// accumulates in int16 lanes. A 16-sample step adds at most 2*255 to a
// lane, so 64 steps (1024 samples) fit int16 before the flush to int32:
// blocks of up to 1024 samples (16x16 through 32x32, 64x16) run one group,
// 64x64 four.
//
// archsimd has no USUBL, so the int16 differences come from the wrapping
// byte difference a-b and the byte mask b > a (0xFF where the difference is
// negative) interleaved as the low and high bytes: (a-b mod 256) + 0xFF00 is
// a-b mod 2^16. That is SUB, CMHI and two ZIPs per 16 samples instead of
// four UXTLs and two SUBs.
//
// Each step's terms are summed before they reach an accumulator, so every
// accumulator takes one add per step: the loop is throughput-bound rather
// than bound by an add chain.
//
// The 16- and 32-wide shapes (most calls) run row loops without a column
// loop; the totals reduce with ADDV (the sse of a 64x64 block is at most
// 4096*65025 < 2^31).
//
// The flush runs after each group of rows rather than on a per-step
// counter: the conditional reset in the column loop made the register
// allocator swap the int16 accumulator with its zero on every back edge.
//
// Rows are bounds-checked windows: the 16- and 32-wide shapes take one
// window per row pair (row 2i at w[:n], row 2i+1 at w[stride:]), the wider
// shapes one per row.
func pixelStatsRows16Arch(src []byte, srcStride int, ref []byte, refStride int, w, h int) (uint32, int32) {
	if h <= 0 {
		return 0, 0
	}
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(h) > rowsMax || uint(w) > rowsMax {
		panic(errKernelArgs)
	}
	var groupRows int
	switch w {
	case 16:
		groupRows = 64
	case 32:
		groupRows = 32
	case 64:
		groupRows = 16
	default:
		groupRows = max(1, 1024/w)
	}
	var sse0, sse1, sum32 archsimd.Int32x4
	for r0 := 0; r0 < h; r0 += groupRows {
		var sum16 archsimd.Int16x8
		r1 := min(r0+groupRows, h)
		switch w {
		case 16:
			// Pairs run from the group's first row; an odd tail row (only
			// in the last group) ends the block.
			sn, rn := srcStride+16, refStride+16
			gs, gr, n := src[r0*srcStride:], ref[r0*refStride:], r1-r0
			if uint(n) > rowsMax {
				panic(errKernelArgs)
			}
			for i := 0; i < n/2; i++ {
				so, ro := 2*i*srcStride, 2*i*refStride
				sw := gs[so : so+sn : so+sn]
				rw := gr[ro : ro+rn : ro+rn]
				dl, dh := pixDiff16((*[16]uint8)(sw), (*[16]uint8)(rw))
				el, eh := pixDiff16((*[16]uint8)(sw[srcStride:]), (*[16]uint8)(rw[refStride:]))
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
			}
			if n&1 != 0 {
				so, ro := (n-1)*srcStride, (n-1)*refStride
				dl, dh := pixDiff16((*[16]uint8)(gs[so:so+16]), (*[16]uint8)(gr[ro:ro+16]))
				sum16 = sum16.Add(dl.Add(dh))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
			}
		case 32:
			sn, rn := srcStride+32, refStride+32
			gs, gr, n := src[r0*srcStride:], ref[r0*refStride:], r1-r0
			if uint(n) > rowsMax {
				panic(errKernelArgs)
			}
			for i := 0; i < n/2; i++ {
				so, ro := 2*i*srcStride, 2*i*refStride
				sw := gs[so : so+sn : so+sn]
				rw := gr[ro : ro+rn : ro+rn]
				s0, p0 := (*[32]uint8)(sw), (*[32]uint8)(rw)
				s1, p1 := (*[32]uint8)(sw[srcStride:]), (*[32]uint8)(rw[refStride:])
				dl, dh := pixDiff16((*[16]uint8)(s0[0:16]), (*[16]uint8)(p0[0:16]))
				el, eh := pixDiff16((*[16]uint8)(s0[16:32]), (*[16]uint8)(p0[16:32]))
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
				dl, dh = pixDiff16((*[16]uint8)(s1[0:16]), (*[16]uint8)(p1[0:16]))
				el, eh = pixDiff16((*[16]uint8)(s1[16:32]), (*[16]uint8)(p1[16:32]))
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
			}
			if n&1 != 0 {
				so, ro := (n-1)*srcStride, (n-1)*refStride
				s0, p0 := (*[32]uint8)(gs[so:so+32]), (*[32]uint8)(gr[ro:ro+32])
				dl, dh := pixDiff16((*[16]uint8)(s0[0:16]), (*[16]uint8)(p0[0:16]))
				el, eh := pixDiff16((*[16]uint8)(s0[16:32]), (*[16]uint8)(p0[16:32]))
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
			}
		case 64:
			for r := r0; r < r1; r++ {
				so, ro := r*srcStride, r*refStride
				s0, p0 := (*[64]uint8)(src[so:so+64]), (*[64]uint8)(ref[ro:ro+64])
				dl, dh := pixDiff16((*[16]uint8)(s0[0:16]), (*[16]uint8)(p0[0:16]))
				el, eh := pixDiff16((*[16]uint8)(s0[16:32]), (*[16]uint8)(p0[16:32]))
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
				dl, dh = pixDiff16((*[16]uint8)(s0[32:48]), (*[16]uint8)(p0[32:48]))
				el, eh = pixDiff16((*[16]uint8)(s0[48:64]), (*[16]uint8)(p0[48:64]))
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
			}
		default:
			for r := r0; r < r1; r++ {
				s := src[r*srcStride : r*srcStride+w]
				p := ref[r*refStride : r*refStride+w]
				p = p[:len(s)]
				for c := 0; c+16 <= len(s); c += 16 {
					dl, dh := pixDiff16((*[16]uint8)(s[c:c+16]), (*[16]uint8)(p[c:c+16]))
					sum16 = sum16.Add(dl.Add(dh))
					sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
					sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
				}
			}
		}
		sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
	}
	return uint32(sse0.Add(sse1).ReduceSum()), sum32.ReduceSum()
}

// pixDiff16 is the sixteen int16 differences a-b, low and high halves:
// the wrapping byte difference interleaved with the borrow mask b > a.
func pixDiff16(a, b *[16]uint8) (lo, hi archsimd.Int16x8) {
	x, y := archsimd.LoadUint8x16Array(a), archsimd.LoadUint8x16Array(b)
	d := x.Sub(y)
	m := y.Greater(x).ToInt8x16().ToBits()
	return d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16(), d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
}
