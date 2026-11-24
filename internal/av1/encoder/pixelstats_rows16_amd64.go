//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// pixelStatsRows16Arch is pixelStatsRows16 as SVT-AV1's variance_kernel_avx2:
// each 16-sample row segment zero-extends to 16 int16 lanes (VPMOVZXBW), the
// difference squares and pair-sums straight into int32 lanes (VPMADDWD), and
// the signed sum accumulates in int16 lanes. Every 16-sample step adds one
// difference (|d| <= 255) to each int16 lane, so 128 steps fit before the
// flush to int32; the sse of a 64x64 block is at most 4096*65025 < 2^31.
//
// Like SVT's fixed-shape variance{16,32,64}_avx2, the 16-, 32- and 64-wide
// shapes run with no column loop, and their flush groups are constants (no
// division per call). Rows are bounds-checked windows: the 16- and 32-wide
// shapes take one window per row pair (row 2i at w[:n], row 2i+1 at
// w[stride:]), the wider shapes one per row.
//
// The kernels run only behind gosimdKernelsSupported (AVX2).
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
		groupRows = 128
	case 32:
		groupRows = 64
	case 64:
		groupRows = 32
	default:
		groupRows = max(1, 1024/w)
	}
	var sse, sum32 archsimd.Int32x8
	for r0 := 0; r0 < h; r0 += groupRows {
		var sum16 archsimd.Int16x16
		n := min(groupRows, h-r0)
		if uint(n) > rowsMax {
			panic(errKernelArgs)
		}
		// The group's rows as windows of the block from its first row.
		gs, gr := src[r0*srcStride:], ref[r0*refStride:]
		switch w {
		case 16:
			sn, rn := srcStride+16, refStride+16
			for i := 0; i < n/2; i++ {
				so, ro := 2*i*srcStride, 2*i*refStride
				sw := gs[so : so+sn : so+sn]
				rw := gr[ro : ro+rn : ro+rn]
				d0 := pixelDiff16((*[16]uint8)(sw), (*[16]uint8)(rw))
				d1 := pixelDiff16((*[16]uint8)(sw[srcStride:]), (*[16]uint8)(rw[refStride:]))
				sum16 = sum16.Add(d0).Add(d1)
				sse = sse.Add(d0.DotProductPairs(d0)).Add(d1.DotProductPairs(d1))
			}
			if n&1 != 0 {
				so, ro := (n-1)*srcStride, (n-1)*refStride
				d := pixelDiff16((*[16]uint8)(gs[so:so+16]), (*[16]uint8)(gr[ro:ro+16]))
				sum16 = sum16.Add(d)
				sse = sse.Add(d.DotProductPairs(d))
			}
		case 32:
			sn, rn := srcStride+32, refStride+32
			for i := 0; i < n/2; i++ {
				so, ro := 2*i*srcStride, 2*i*refStride
				sw := gs[so : so+sn : so+sn]
				rw := gr[ro : ro+rn : ro+rn]
				s0, p0 := (*[32]uint8)(sw), (*[32]uint8)(rw)
				s1, p1 := (*[32]uint8)(sw[srcStride:]), (*[32]uint8)(rw[refStride:])
				d0 := pixelDiff16((*[16]uint8)(s0[0:16]), (*[16]uint8)(p0[0:16]))
				d1 := pixelDiff16((*[16]uint8)(s0[16:32]), (*[16]uint8)(p0[16:32]))
				d2 := pixelDiff16((*[16]uint8)(s1[0:16]), (*[16]uint8)(p1[0:16]))
				d3 := pixelDiff16((*[16]uint8)(s1[16:32]), (*[16]uint8)(p1[16:32]))
				sum16 = sum16.Add(d0).Add(d1).Add(d2).Add(d3)
				sse = sse.Add(d0.DotProductPairs(d0)).Add(d1.DotProductPairs(d1)).
					Add(d2.DotProductPairs(d2)).Add(d3.DotProductPairs(d3))
			}
			if n&1 != 0 {
				so, ro := (n-1)*srcStride, (n-1)*refStride
				s0, p0 := (*[32]uint8)(gs[so:so+32]), (*[32]uint8)(gr[ro:ro+32])
				d0 := pixelDiff16((*[16]uint8)(s0[0:16]), (*[16]uint8)(p0[0:16]))
				d1 := pixelDiff16((*[16]uint8)(s0[16:32]), (*[16]uint8)(p0[16:32]))
				sum16 = sum16.Add(d0).Add(d1)
				sse = sse.Add(d0.DotProductPairs(d0)).Add(d1.DotProductPairs(d1))
			}
		case 64:
			for r := 0; r < n; r++ {
				so, ro := r*srcStride, r*refStride
				s, p := (*[64]uint8)(gs[so:so+64]), (*[64]uint8)(gr[ro:ro+64])
				d0 := pixelDiff16((*[16]uint8)(s[0:16]), (*[16]uint8)(p[0:16]))
				d1 := pixelDiff16((*[16]uint8)(s[16:32]), (*[16]uint8)(p[16:32]))
				d2 := pixelDiff16((*[16]uint8)(s[32:48]), (*[16]uint8)(p[32:48]))
				d3 := pixelDiff16((*[16]uint8)(s[48:64]), (*[16]uint8)(p[48:64]))
				sum16 = sum16.Add(d0).Add(d1).Add(d2).Add(d3)
				sse = sse.Add(d0.DotProductPairs(d0)).Add(d1.DotProductPairs(d1)).
					Add(d2.DotProductPairs(d2)).Add(d3.DotProductPairs(d3))
			}
		default:
			for r := 0; r < n; r++ {
				s := gs[r*srcStride : r*srcStride+w]
				p := gr[r*refStride : r*refStride+w]
				p = p[:len(s)]
				for c := 0; c+16 <= len(s); c += 16 {
					d := pixelDiff16((*[16]uint8)(s[c:c+16]), (*[16]uint8)(p[c:c+16]))
					sum16 = sum16.Add(d)
					sse = sse.Add(d.DotProductPairs(d))
				}
			}
		}
		sum32 = sum32.Add(sum16.GetLo().ExtendToInt32()).Add(sum16.GetHi().ExtendToInt32())
	}
	sseOut, sumOut := uint32(reduceInt32x8(sse)), reduceInt32x8(sum32)
	// Return with clean upper ymm state (the compiler emits no vzeroupper).
	archsimd.ClearAVXUpperBits()
	return sseOut, sumOut
}

// pixelDiff16 is the sixteen int16 differences src-ref of the samples s
// and p (VPMOVZXBW from memory, VPSUBW).
func pixelDiff16(s, p *[16]uint8) archsimd.Int16x16 {
	a := archsimd.LoadUint8x16Array(s).ExtendToUint16().BitsToInt16()
	b := archsimd.LoadUint8x16Array(p).ExtendToUint16().BitsToInt16()
	return a.Sub(b)
}

// reduceInt32x8 returns the sum of the eight lanes.
func reduceInt32x8(v archsimd.Int32x8) int32 {
	q := v.GetLo().Add(v.GetHi())
	return q.GetElem(0) + q.GetElem(1) + q.GetElem(2) + q.GetElem(3)
}
