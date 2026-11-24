//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

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
func pixelStatsRows16Arch(src []byte, srcStride int, ref []byte, refStride int, w, h int) (uint32, int32) {
	if h <= 0 {
		return 0, 0
	}
	_ = src[(h-1)*srcStride+w-1]
	_ = ref[(h-1)*refStride+w-1]
	// Strided rows inside the extents checked above; same shape as the SAD rows (checked rows +9-86%).
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
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
			for r := r0; r < r1; r++ {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, r*srcStride)))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(rp, r*refStride)))
				d := a.Sub(b)
				m := b.Greater(a).ToInt8x16().ToBits()
				dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
			}
		case 32:
			for r := r0; r < r1; r++ {
				s := unsafe.Add(sp, r*srcStride)
				p := unsafe.Add(rp, r*refStride)
				a := archsimd.LoadUint8x16Array((*[16]uint8)(s))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(p))
				d := a.Sub(b)
				m := b.Greater(a).ToInt8x16().ToBits()
				dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				a = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, 16)))
				b = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 16)))
				d = a.Sub(b)
				m = b.Greater(a).ToInt8x16().ToBits()
				el := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				eh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
			}
		default:
			for r := r0; r < r1; r++ {
				s := unsafe.Add(sp, r*srcStride)
				p := unsafe.Add(rp, r*refStride)
				for c := 0; c < w; c += 16 {
					a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, c)))
					b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, c)))
					d := a.Sub(b)
					m := b.Greater(a).ToInt8x16().ToBits()
					dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
					dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
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
