// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/unsafeutil"
	"math/bits"
	"simd/archsimd"
)

// cdefDist8 is SVT-AV1's svt_aom_compute_cdef_dist_8bit_neon_dotprod
// (ASM_NEON_DOTPROD/cdef_neon_dotprod.c mse_8xn_8bit_neon_dotprod) in Go
// SIMD for the 8x8 luma blocks: rows i and i+subsampling pack into one
// 16-byte vector, |s-d| squares into 16-bit lanes and accumulates into four
// 32-bit lanes (archsimd has no UDOT, so the dot product is widened). Other
// block shapes take the scalar reference.
func cdefDist8(src []byte, sstride int, filtered []byte, fstride int, dlist []cdef.BlockPosition, bwLog2, bhLog2 int, subsampling int) uint64 {
	if bwLog2 != 3 || bhLog2 != 3 || (subsampling != 1 && subsampling != 2 && subsampling != 4) {
		return computeCDEFDist8(src, sstride, filtered, fstride, dlist, bwLog2, bhLog2, subsampling)
	}
	if len(dlist) == 0 {
		return 0
	}
	if uint(sstride) > rowStrideMax || uint(fstride) > rowStrideMax {
		panic(errKernelArgs)
	}
	// Each row pair (i, i+subsampling) is one bounds-checked window: the
	// first row at w[:8], the second at w[sd:].
	sd, fd := subsampling*sstride, subsampling*fstride
	sn, fn := sd+8, fd+8
	pairs := 4 / subsampling
	acc := archsimd.BroadcastUint32x4(0)
	// Rows (i, i+subsampling) form a pair; pairs advance by twice that,
	// so subsampling 4 measures rows 0 and 4 only.
	for _, b := range dlist {
		so := int(b.BY)*8*sstride + int(b.BX)*8
		fo := int(b.BY)*8*fstride + int(b.BX)*8
		for k := 0; k < pairs; k++ {
			o, q := so+2*k*sd, fo+2*k*fd
			sw := src[o : o+sn : o+sn]
			fw := filtered[q : q+fn : q+fn]
			s := pack2((*[8]byte)(sw), (*[8]byte)(sw[sd:]))
			f := pack2((*[8]byte)(fw), (*[8]byte)(fw[fd:]))
			d := s.Max(f).Sub(s.Min(f))
			lo := d.ExtendLo8ToUint16()
			hi := d.HiToLo().ExtendLo8ToUint16()
			lo = lo.Mul(lo)
			hi = hi.Mul(hi)
			acc = acc.Add(lo.ExtendLo4ToUint32()).Add(lo.HiToLo().ExtendLo4ToUint32())
			acc = acc.Add(hi.ExtendLo4ToUint32()).Add(hi.HiToLo().ExtendLo4ToUint32())
		}
	}
	return uint64(acc.ReduceSum())
}

// gosimd_ops_arm64.go and gosimd_ops_amd64.go give the shared Go SIMD kernels
// one vocabulary for lane moves and widening squares. The arm64 forms use the
// NEON zip, extend and widening-multiply ops; the amd64 forms use PSHUFB
// (PermuteOrZero) and PMULDQ (MulWidenEven), which exist in AVX2 without
// AVX-512.

// widenHi8 zero-extends bytes 8..15 of v to 16-bit lanes.
func widenHi8(v archsimd.Uint8x16) archsimd.Uint16x8 {
	return v.HiToLo().ExtendLo8ToUint16()
}

// hiInt16 moves 16-bit lanes 4..7 of v into lanes 0..3.
func hiInt16(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.HiToLo()
}

// sqSumInt32 adds the exact squares of v's four int32 lanes to acc.
func sqSumInt32(acc archsimd.Int64x2, v archsimd.Int32x4) archsimd.Int64x2 {
	hi := v.HiToLo()
	return acc.Add(v.MulWidenLo(v)).Add(hi.MulWidenLo(hi))
}

// mulSumInt32 adds the exact products a[i]*b[i] of the four int32 lane pairs
// to acc.
func mulSumInt32(acc archsimd.Int64x2, a, b archsimd.Int32x4) archsimd.Int64x2 {
	return acc.Add(a.MulWidenLo(b)).Add(a.HiToLo().MulWidenLo(b.HiToLo()))
}

// gosimdKernelsSupported reports whether the Go-native SIMD kernels can run on
// this CPU. Advanced SIMD is baseline on arm64.
func gosimdKernelsSupported() bool { return true }

// lpd1SampledSAD is the row-subsampled SAD the chroma detector uses
// (svt_nxm_sad_kernel called with stride << shift and height >> shift), as
// svt_nxm_sad_kernel_helper_neon: 16-byte rows directly, 8-byte rows two to
// a vector. The detector samples at most 8 rows of at most 32 samples, so a
// uint16 lane gains at most 16*510.
func lpd1SampledSAD(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	if rows <= 0 || w <= 0 {
		return 0
	}
	ss, rs := step*srcStride, step*refStride
	if (w != 8 && w&15 != 0) || rows > 8 || w > 32 {
		return lpd1SampledSADPureGo(src, ref, srcOff, refOff, srcStride, refStride, w, rows, step)
	}
	if uint(ss) > rowStrideMax || uint(rs) > rowStrideMax {
		panic(errKernelArgs)
	}
	src, ref = src[srcOff:], ref[refOff:]
	var acc archsimd.Uint16x8
	if w == 8 {
		// One window per row pair: row 2i at w[:8], row 2i+1 at w[ss:].
		sn, rn := ss+8, rs+8
		for i := 0; i < rows/2; i++ {
			so, ro := 2*i*ss, 2*i*rs
			sw := src[so : so+sn : so+sn]
			rw := ref[ro : ro+rn : ro+rn]
			acc = absAcc16(acc, pack2((*[8]byte)(sw), (*[8]byte)(sw[ss:])), pack2((*[8]byte)(rw), (*[8]byte)(rw[rs:])))
		}
		total := sumU16(acc)
		if rows&1 != 0 {
			r := rows - 1
			s := src[r*ss : r*ss+8]
			d := ref[r*rs : r*rs+8]
			for c := range s {
				diff := int(s[c]) - int(d[c])
				total += max(diff, -diff)
			}
		}
		return total
	}
	for r := range rows {
		s, d := src[r*ss:r*ss+w], ref[r*rs:r*rs+w]
		for c := 0; c+16 <= len(s) && c+16 <= len(d); c += 16 {
			acc = absAcc16(acc, archsimd.LoadUint8x16Array((*[16]uint8)(s[c:c+16])), archsimd.LoadUint8x16Array((*[16]uint8)(d[c:c+16])))
		}
	}
	return sumU16(acc)
}

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
// The rows are unsafeutil grids checked once per call: the row windows
// (16 or 32 bytes, or 16-byte chunks of wider rows) cost no check each.
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
	// The rows are grids, checked once; each row group is a sub-grid whose
	// height bounds its row loop, so no row costs a check.
	if !unsafeutil.GridFit(len(src), 0, srcStride, w, h) || !unsafeutil.GridFit(len(ref), 0, refStride, w, h) {
		panic(errKernelArgs)
	}
	gsAll, grAll := unsafeutil.NewGrid(src, 0, srcStride, w, h), unsafeutil.NewGrid(ref, 0, refStride, w, h)
	var sse0, sse1, sum32 archsimd.Int32x4
	if w == 16 && h <= 64 {
		var sum16 archsimd.Int16x8
		for r := 0; r < gsAll.Height(); r++ {
			a := archsimd.LoadUint8x16Array(gsAll.Row16(r, 0))
			b := archsimd.LoadUint8x16Array(grAll.Row16(r, 0))
			d := a.Sub(b)
			m := b.Greater(a).ToInt8x16().ToBits()
			dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
			dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
			sum16 = sum16.Add(dl.Add(dh))
			sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
			sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
		}
		sum32 = sum16.ExtendLo4ToInt32().Add(sum16.HiToLo().ExtendLo4ToInt32())
		return uint32(sse0.Add(sse1).ReduceSum()), sum32.ReduceSum()
	}
	if w == 32 && h <= 32 {
		var sum16 archsimd.Int16x8
		for r := 0; r < gsAll.Height(); r++ {
			sw, rw := gsAll.Row32(r, 0), grAll.Row32(r, 0)
			a := archsimd.LoadUint8x16Array((*[16]uint8)(sw[0:16]))
			b := archsimd.LoadUint8x16Array((*[16]uint8)(rw[0:16]))
			d := a.Sub(b)
			m := b.Greater(a).ToInt8x16().ToBits()
			dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
			dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
			a = archsimd.LoadUint8x16Array((*[16]uint8)(sw[16:32]))
			b = archsimd.LoadUint8x16Array((*[16]uint8)(rw[16:32]))
			d = a.Sub(b)
			m = b.Greater(a).ToInt8x16().ToBits()
			el := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
			eh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
			sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
			sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
			sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
				Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
		}
		sum32 = sum16.ExtendLo4ToInt32().Add(sum16.HiToLo().ExtendLo4ToInt32())
		return uint32(sse0.Add(sse1).ReduceSum()), sum32.ReduceSum()
	}
	for r0 := 0; r0 < h; r0 += groupRows {
		var sum16 archsimd.Int16x8
		n := min(groupRows, h-r0)
		gs, gr := gsAll.Sub(r0, 0, w, n), grAll.Sub(r0, 0, w, n)
		switch w {
		case 16:
			for r := 0; r < gs.Height(); r++ {
				a := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gs, r, 0))
				b := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gr, r, 0))
				d := a.Sub(b)
				m := b.Greater(a).ToInt8x16().ToBits()
				dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
			}
		case 32:
			for r := 0; r < gs.Height(); r++ {
				sw, rw := unsafeutil.Row[[32]uint8](gs, r, 0), unsafeutil.Row[[32]uint8](gr, r, 0)
				a := archsimd.LoadUint8x16Array((*[16]uint8)(sw[0:16]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(rw[0:16]))
				d := a.Sub(b)
				m := b.Greater(a).ToInt8x16().ToBits()
				dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				a = archsimd.LoadUint8x16Array((*[16]uint8)(sw[16:32]))
				b = archsimd.LoadUint8x16Array((*[16]uint8)(rw[16:32]))
				d = a.Sub(b)
				m = b.Greater(a).ToInt8x16().ToBits()
				el := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				eh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh).Add(el.Add(eh)))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)).Add(el.MulWidenLo(el).Add(eh.MulWidenLo(eh))))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())).
					Add(el.HiToLo().MulWidenLo(el.HiToLo()).Add(eh.HiToLo().MulWidenLo(eh.HiToLo()))))
			}
		case 64:
			for r := 0; r < gs.Height(); r++ {
				sw, rw := gs.Row64(r, 0), gr.Row64(r, 0)
				a := archsimd.LoadUint8x16Array((*[16]uint8)(sw[0:16]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(rw[0:16]))
				d := a.Sub(b)
				m := b.Greater(a).ToInt8x16().ToBits()
				dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))

				a = archsimd.LoadUint8x16Array((*[16]uint8)(sw[16:32]))
				b = archsimd.LoadUint8x16Array((*[16]uint8)(rw[16:32]))
				d = a.Sub(b)
				m = b.Greater(a).ToInt8x16().ToBits()
				dl = d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh = d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))

				a = archsimd.LoadUint8x16Array((*[16]uint8)(sw[32:48]))
				b = archsimd.LoadUint8x16Array((*[16]uint8)(rw[32:48]))
				d = a.Sub(b)
				m = b.Greater(a).ToInt8x16().ToBits()
				dl = d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh = d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))

				a = archsimd.LoadUint8x16Array((*[16]uint8)(sw[48:64]))
				b = archsimd.LoadUint8x16Array((*[16]uint8)(rw[48:64]))
				d = a.Sub(b)
				m = b.Greater(a).ToInt8x16().ToBits()
				dl = d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
				dh = d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
				sum16 = sum16.Add(dl.Add(dh))
				sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
				sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
			}
		default:
			for r := 0; r < gs.Height(); r++ {
				for c := 0; c+16 <= w; c += 16 {
					a := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gs, r, c))
					b := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gr, r, c))
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

func init() {
	pred8Impl = pred8SIMD
}

// The AV1 8-tap kernels are all even, so the 8-bit passes accumulate halved
// taps in int16 lanes without overflow (the positive taps of any kernel sum
// to at most 92 after halving, so a sum stays below 92*255 + 34). With
// a = sum(k/2 * x):
//
//	convolve_x_sr: clip(round(round(2a, 3), 4)) = clip((a + 34) >> 6)
//	convolve_y_sr: clip(round(2a, 7))           = clip((a + 32) >> 6)
//	2-D first pass: round(16384 + 2a, 3)        = (a + 8194) >> 2
//
// and the 2-D second pass over the intermediate im (int32 lanes, full taps):
//
//	clip(round(2^19 + sum(k*im), 11) - 384) = clip((sum(k*im) - 261120) >> 11)
//
// (2^19 and 384<<11 are multiples of 2^11, so they fold into the bias).
//
// Every kernel reads and writes through bounds-checked windows: ref is the
// reference plane with the block origin at ref[refOff], dst and src are the
// destination and source blocks from their first sample.

// pred8Out16 stores the 16 output bytes of a row group at dst[do:] or folds
// their absolute difference with src[so:] into the SAD accumulator.
func pred8Out16(out archsimd.Uint8x16, dst []byte, do int, src []byte, so int, acc archsimd.Uint16x8) archsimd.Uint16x8 {
	if src != nil {
		return acc.Add(widen16(absDiffU8x16(out, archsimd.LoadUint8x16Array((*[16]uint8)(src[so:so+16])))))
	}
	out.StoreArray((*[16]uint8)(dst[do : do+16]))
	return acc
}

// pred8Pack16 saturates lo and hi into one vector, the high half as SQXTUN2
// (SetElem of the low 64 bits) rather than a second SQXTUN joined by ZIP1.
func pred8Pack16(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
	l := lo.SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
	return l.SetElem(1, hi.SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)).ToBits().ReshapeToUint8s()
}

// put64 stores v little-endian into b (one 8-byte store).
func put64(b *[8]byte, v uint64) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	b[4], b[5], b[6], b[7] = byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

// pred8Store8 stores the eight saturated bytes of v at dst[o:].
func pred8Store8(dst []byte, o int, v archsimd.Int16x8) {
	put64((*[8]byte)(dst[o:o+8]), v.SaturateToUint8().ReshapeToUint64s().GetElem(0))
}

func pred8SIMD(dst []byte, dstStride int, src []byte, srcStride int,
	ref []byte, refOff, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int {
	switch {
	case subX == 0 && subY == 0:
		return pred8Copy(dst, dstStride, src, srcStride, ref, refOff, refStride, w, h)
	case subY == 0:
		return pred8H(dst, dstStride, src, srcStride, ref, refOff, refStride, w, h, xk)
	case subX == 0:
		return pred8V(dst, dstStride, src, srcStride, ref, refOff, refStride, w, h, yk)
	}
	pred8IM(im[:], 64, ref, refOff-3*refStride, refStride, w, h+7, xk)
	return pred8VIM(dst, dstStride, src, srcStride, im[:], 64, w, h, yk)
}

func pred8Copy(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refOff, refStride, w, h int) int {
	ref = ref[refOff:]
	if src == nil {
		pred8CopyRows(dst, dstStride, ref, refStride, w, h)
		return 0
	}
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	var acc archsimd.Uint16x8
	sad := 0
	if h > 0 && w >= 16 && unsafeutil.GridFit(len(ref), 0, refStride, w, h) && unsafeutil.GridFit(len(src), 0, srcStride, w, h) {
		gr, gs := unsafeutil.NewGrid(ref, 0, refStride, w, h), unsafeutil.NewGrid(src, 0, srcStride, w, h)
		for y := 0; y < gr.Height(); y++ {
			for x := 0; x+16 <= w; x += 16 {
				acc = acc.Add(widen16(absDiffU8x16(archsimd.LoadUint8x16Array(gr.Row16(y, x)), archsimd.LoadUint8x16Array(gs.Row16(y, x)))))
			}
			if y&15 == 15 {
				sad += sumU16(acc)
				acc = archsimd.Uint16x8{}
			}
		}
		return sad + sumU16(acc)
	}
	for y := 0; y < h; y++ {
		r, s := ref[y*refStride:y*refStride+w], src[y*srcStride:y*srcStride+w]
		s = s[:len(r)]
		for x := 0; x+16 <= len(r); x += 16 {
			acc = acc.Add(widen16(absDiffU8x16(archsimd.LoadUint8x16Array((*[16]uint8)(r[x:x+16])), archsimd.LoadUint8x16Array((*[16]uint8)(s[x:x+16])))))
		}
		if y&15 == 15 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	return sad + sumU16(acc)
}

// pred8CopyRows is svt_av1_convolve_2d_copy_sr_neon: copyBlockRows, one
// loop per block width with a bounds-checked window per row pair.
func pred8CopyRows(dst []byte, dstStride int, ref []byte, refStride, w, h int) {
	copyBlockRows(dst, dstStride, ref, refStride, w, h)
}

// pred8H is convolve_x_sr: per output row, each tap's window is one 16-byte
// load whose low and high halves feed the two 8-lane accumulators. Rows run
// in pairs: each 16-column group of a pair reads one bounds-checked window
// holding both rows' 23 bytes (the eight taps at constant offsets), and
// stores through one window per row pair; the 8-column tail reads the same
// windows.
func pred8H(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refOff, refStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	ref = ref[refOff-3:]
	bias := archsimd.BroadcastInt16x8(34)
	sh := archsimd.BroadcastInt16x8(-6)
	var acc archsimd.Uint16x8
	sad := 0
	rn := refStride + 23
	if k[0] == 0 && k[7] == 0 {
		c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
		c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
		c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
		c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
		c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
		c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
		for i := 0; i < h/2; i++ {
			ro, do, so := 2*i*refStride, 2*i*dstStride, 2*i*srcStride
			x := 0
			for ; x+16 <= w; x += 16 {
				rw := ref[ro+x : ro+x+rn : ro+x+rn]
				var o0, o1 archsimd.Uint8x16
				{
					q := (*[23]uint8)(rw)
					v := archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
					a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
					a1 := widenHi8(v).BitsToInt16().MulAdd(c1, bias)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
					o0 = pred8Pack16(a0.Shift(sh), a1.Shift(sh))
				}
				{
					q := (*[23]uint8)(rw[refStride:])
					v := archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
					a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
					a1 := widenHi8(v).BitsToInt16().MulAdd(c1, bias)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
					v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
					a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
					a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
					o1 = pred8Pack16(a0.Shift(sh), a1.Shift(sh))
				}
				acc = pred8Out16x2(o0, o1, dst, do+x, dstStride, src, so+x, srcStride, acc)
			}
			if x < w {
				rw := ref[ro+x : ro+x+rn : ro+x+rn]
				{
					q := (*[23]uint8)(rw)
					a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
					pred8Store8(dst, do+x, a0.Shift(sh))
				}
				{
					q := (*[23]uint8)(rw[refStride:])
					a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
					a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
					pred8Store8(dst, do+dstStride+x, a0.Shift(sh))
				}
			}
			if src != nil && i&7 == 7 {
				sad += sumU16(acc)
				acc = archsimd.Uint16x8{}
			}
		}
		if h&1 != 0 {
			y := h - 1
			ro, do, so := y*refStride, y*dstStride, y*srcStride
			x := 0
			for ; x+16 <= w; x += 16 {
				q := (*[23]uint8)(ref[ro+x : ro+x+23])
				v := archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c1, bias)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, do+x, src, so+x, acc)
			}
			if x < w {
				q := (*[23]uint8)(ref[ro+x : ro+x+23])
				a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				pred8Store8(dst, do+x, a0.Shift(sh))
			}
		}
		return sad + sumU16(acc)
	}
	c0 := archsimd.BroadcastInt16x8(k[0] >> 1)
	c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
	c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
	c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
	c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
	c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
	c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
	c7 := archsimd.BroadcastInt16x8(k[7] >> 1)
	for i := 0; i < h/2; i++ {
		ro, do, so := 2*i*refStride, 2*i*dstStride, 2*i*srcStride
		x := 0
		for ; x+16 <= w; x += 16 {
			rw := ref[ro+x : ro+x+rn : ro+x+rn]
			var o0, o1 archsimd.Uint8x16
			{
				q := (*[23]uint8)(rw)
				v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
				o0 = pred8Pack16(a0.Shift(sh), a1.Shift(sh))
			}
			{
				q := (*[23]uint8)(rw[refStride:])
				v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
				o1 = pred8Pack16(a0.Shift(sh), a1.Shift(sh))
			}
			acc = pred8Out16x2(o0, o1, dst, do+x, dstStride, src, so+x, srcStride, acc)
		}
		if x < w {
			rw := ref[ro+x : ro+x+rn : ro+x+rn]
			{
				q := (*[23]uint8)(rw)
				a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				pred8Store8(dst, do+x, a0.Shift(sh))
			}
			{
				q := (*[23]uint8)(rw[refStride:])
				a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				pred8Store8(dst, do+dstStride+x, a0.Shift(sh))
			}
		}
		if src != nil && i&7 == 7 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	if h&1 != 0 {
		y := h - 1
		ro, do, so := y*refStride, y*dstStride, y*srcStride
		x := 0
		for ; x+16 <= w; x += 16 {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
			a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
			acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, do+x, src, so+x, acc)
		}
		if x < w {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			pred8Store8(dst, do+x, a0.Shift(sh))
		}
	}
	return sad + sumU16(acc)
}

// pred8Out16x2 is pred8Out16 for two rows (stride ds or ss apart), through
// one window per row pair.
func pred8Out16x2(o0, o1 archsimd.Uint8x16, dst []byte, do, ds int, src []byte, so, ss int, acc archsimd.Uint16x8) archsimd.Uint16x8 {
	if src != nil {
		n := ss + 16
		sw := src[so : so+n : so+n]
		acc = acc.Add(widen16(absDiffU8x16(o0, archsimd.LoadUint8x16Array((*[16]uint8)(sw)))))
		return acc.Add(widen16(absDiffU8x16(o1, archsimd.LoadUint8x16Array((*[16]uint8)(sw[ss:])))))
	}
	n := ds + 16
	dw := dst[do : do+n : do+n]
	o0.StoreArray((*[16]uint8)(dw))
	o1.StoreArray((*[16]uint8)(dw[ds:]))
	return acc
}

// pred8V is convolve_y_sr over 16-column (or a final 8-column) strips. The
// pinned libaom av1/common/arm/convolve_neon.c six-tap vertical kernel omits
// coefficients 0 and 7. Regular/smooth kernels have zero outer taps; fractional
// sharp kernels retain the full eight-tap path. Like libaom's
// convolve_y_sr_6tap/8tap_neon, a strip keeps its tap rows in registers and
// loads one new row per output row, each through one checked window.
func pred8V(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refOff, refStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	var acc archsimd.Uint16x8
	sad := 0
	if k[0] == 0 && k[7] == 0 {
		ref := ref[refOff-2*refStride:]
		c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
		c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
		c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
		c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
		c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
		c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
		bias := archsimd.BroadcastInt16x8(32)
		sh := archsimd.BroadcastInt16x8(-6)
		refW := (w + 15) &^ 15
		if h > 0 && w >= 8 && unsafeutil.GridFit(len(ref), 0, refStride, refW, h+5) &&
			(src != nil && unsafeutil.GridFit(len(src), 0, srcStride, w, h) || src == nil && unsafeutil.GridFit(len(dst), 0, dstStride, w, h)) {
			gr := unsafeutil.NewGrid(ref, 0, refStride, refW, h+5)
			var gd, gs unsafeutil.Grid[uint8]
			if src != nil {
				gs = unsafeutil.NewGrid(src, 0, srcStride, w, h)
			} else {
				gd = unsafeutil.NewGrid(dst, 0, dstStride, w, h)
			}
			for x := 0; x < w; x += 16 {
				r0 := archsimd.LoadUint8x16Array(gr.Row16(0, x))
				r1 := archsimd.LoadUint8x16Array(gr.Row16(1, x))
				r2 := archsimd.LoadUint8x16Array(gr.Row16(2, x))
				r3 := archsimd.LoadUint8x16Array(gr.Row16(3, x))
				r4 := archsimd.LoadUint8x16Array(gr.Row16(4, x))
				if x+16 <= w {
					if src != nil {
						for y := 0; y < h; y++ {
							r5 := archsimd.LoadUint8x16Array(gr.Row16(y+5, x))
							a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
							a1 := widenHi8(r0).BitsToInt16().MulAdd(c1, bias)
							a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
							a1 = widenHi8(r1).BitsToInt16().MulAdd(c2, a1)
							a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
							a1 = widenHi8(r2).BitsToInt16().MulAdd(c3, a1)
							a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
							a1 = widenHi8(r3).BitsToInt16().MulAdd(c4, a1)
							a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
							a1 = widenHi8(r4).BitsToInt16().MulAdd(c5, a1)
							a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
							a1 = widenHi8(r5).BitsToInt16().MulAdd(c6, a1)
							out := pred8Pack16(a0.Shift(sh), a1.Shift(sh))
							acc = acc.Add(widen16(absDiffU8x16(out, archsimd.LoadUint8x16Array(gs.Row16(y, x)))))
							r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
							if y&15 == 15 {
								sad += sumU16(acc)
								acc = archsimd.Uint16x8{}
							}
						}
					} else {
						for y := 0; y < h; y++ {
							r5 := archsimd.LoadUint8x16Array(gr.Row16(y+5, x))
							a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
							a1 := widenHi8(r0).BitsToInt16().MulAdd(c1, bias)
							a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
							a1 = widenHi8(r1).BitsToInt16().MulAdd(c2, a1)
							a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
							a1 = widenHi8(r2).BitsToInt16().MulAdd(c3, a1)
							a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
							a1 = widenHi8(r3).BitsToInt16().MulAdd(c4, a1)
							a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
							a1 = widenHi8(r4).BitsToInt16().MulAdd(c5, a1)
							a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
							a1 = widenHi8(r5).BitsToInt16().MulAdd(c6, a1)
							pred8Pack16(a0.Shift(sh), a1.Shift(sh)).StoreArray(gd.Row16(y, x))
							r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
						}
					}
					continue
				}
				for y := 0; y < h; y++ {
					r5 := archsimd.LoadUint8x16Array(gr.Row16(y+5, x))
					a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
					a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
					a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
					a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
					a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
					a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
					put64(gd.Row8(y, x), a0.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
					r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
				}
			}
			return sad + sumU16(acc)
		}
		for x := 0; x < w; x += 16 {
			// Strip x: the first 5 rows load up front, then each output row
			// loads its bottom tap row and the rows slide down one register.
			b := x
			r0 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b : b+16]))
			r1 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+1*refStride : b+1*refStride+16]))
			r2 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+2*refStride : b+2*refStride+16]))
			r3 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+3*refStride : b+3*refStride+16]))
			r4 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+4*refStride : b+4*refStride+16]))
			if x+16 <= w {
				for y := 0; y < h; y++ {
					o := (y+5)*refStride + x
					r5 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
					a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
					a1 := widenHi8(r0).BitsToInt16().MulAdd(c1, bias)
					a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
					a1 = widenHi8(r1).BitsToInt16().MulAdd(c2, a1)
					a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
					a1 = widenHi8(r2).BitsToInt16().MulAdd(c3, a1)
					a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
					a1 = widenHi8(r3).BitsToInt16().MulAdd(c4, a1)
					a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
					a1 = widenHi8(r4).BitsToInt16().MulAdd(c5, a1)
					a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
					a1 = widenHi8(r5).BitsToInt16().MulAdd(c6, a1)
					acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, y*dstStride+x, src, y*srcStride+x, acc)
					r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
					if src != nil && y&15 == 15 {
						sad += sumU16(acc)
						acc = archsimd.Uint16x8{}
					}
				}
				continue
			}
			for y := 0; y < h; y++ {
				o := (y+5)*refStride + x
				r5 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
				a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				pred8Store8(dst, y*dstStride+x, a0.Shift(sh))
				r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
			}
		}
		return sad + sumU16(acc)
	}
	ref = ref[refOff-3*refStride:]
	c0 := archsimd.BroadcastInt16x8(k[0] >> 1)
	c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
	c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
	c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
	c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
	c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
	c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
	c7 := archsimd.BroadcastInt16x8(k[7] >> 1)
	bias := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	for x := 0; x < w; x += 16 {
		// Strip x: the first 7 rows load up front, then each output row
		// loads its bottom tap row and the rows slide down one register.
		b := x
		r0 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b : b+16]))
		r1 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+1*refStride : b+1*refStride+16]))
		r2 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+2*refStride : b+2*refStride+16]))
		r3 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+3*refStride : b+3*refStride+16]))
		r4 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+4*refStride : b+4*refStride+16]))
		r5 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+5*refStride : b+5*refStride+16]))
		r6 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+6*refStride : b+6*refStride+16]))
		if x+16 <= w {
			for y := 0; y < h; y++ {
				o := (y+7)*refStride + x
				r7 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
				a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a1 := widenHi8(r0).BitsToInt16().MulAdd(c0, bias)
				a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a1 = widenHi8(r1).BitsToInt16().MulAdd(c1, a1)
				a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(r2).BitsToInt16().MulAdd(c2, a1)
				a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(r3).BitsToInt16().MulAdd(c3, a1)
				a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(r4).BitsToInt16().MulAdd(c4, a1)
				a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(r5).BitsToInt16().MulAdd(c5, a1)
				a0 = r6.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(r6).BitsToInt16().MulAdd(c6, a1)
				a0 = r7.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				a1 = widenHi8(r7).BitsToInt16().MulAdd(c7, a1)
				acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, y*dstStride+x, src, y*srcStride+x, acc)
				r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
				if src != nil && y&15 == 15 {
					sad += sumU16(acc)
					acc = archsimd.Uint16x8{}
				}
			}
			continue
		}
		for y := 0; y < h; y++ {
			o := (y+7)*refStride + x
			r7 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
			a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = r6.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = r7.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			pred8Store8(dst, y*dstStride+x, a0.Shift(sh))
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
		}
	}
	return sad + sumU16(acc)
}

// pred8IM is the 2-D first pass: rows -3..h+3 of the block into im (row y
// at y*imStride), as int16 (a + 8194) >> 2. ref[refOff] is the first row's
// origin sample; each 16-column group reads one window of the row.
func pred8IM(im []int16, imStride int, ref []byte, refOff, refStride, w, rows int, k *[8]int16) {
	if uint(imStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(rows) > rowsMax {
		panic(errKernelArgs)
	}
	if k[0] == 0 && k[7] == 0 {
		ref := ref[refOff-2:]
		c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
		c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
		c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
		c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
		c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
		c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
		bias := archsimd.BroadcastInt16x8(8194)
		sh := archsimd.BroadcastInt16x8(-2)
		for y := 0; y < rows; y++ {
			ro, io := y*refStride, y*imStride
			x := 0
			for ; x+16 <= w; x += 16 {
				q := (*[21]uint8)(ref[ro+x : ro+x+21])
				v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c1, bias)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				o := (*[16]int16)(im[io+x : io+x+16])
				a0.Shift(sh).StoreArray((*[8]int16)(o[0:8]))
				a1.Shift(sh).StoreArray((*[8]int16)(o[8:16]))
			}
			if x < w {
				q := (*[21]uint8)(ref[ro+x : ro+x+21])
				a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a0.Shift(sh).StoreArray((*[8]int16)(im[io+x : io+x+8]))
			}
		}
		return
	}
	ref = ref[refOff-3:]
	c0 := archsimd.BroadcastInt16x8(k[0] >> 1)
	c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
	c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
	c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
	c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
	c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
	c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
	c7 := archsimd.BroadcastInt16x8(k[7] >> 1)
	bias := archsimd.BroadcastInt16x8(8194)
	sh := archsimd.BroadcastInt16x8(-2)
	for y := 0; y < rows; y++ {
		ro, io := y*refStride, y*imStride
		x := 0
		for ; x+16 <= w; x += 16 {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
			a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
			o := (*[16]int16)(im[io+x : io+x+16])
			a0.Shift(sh).StoreArray((*[8]int16)(o[0:8]))
			a1.Shift(sh).StoreArray((*[8]int16)(o[8:16]))
		}
		if x < w {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a0.Shift(sh).StoreArray((*[8]int16)(im[io+x : io+x+8]))
		}
	}
}

// pred8VIM is the 2-D second pass over im in 8-column groups with int32
// accumulation. The pinned libaom av1/common/arm/convolve_neon.c
// (av1_convolve_2d_sr_neon) uses its six-tap kernel when the first and last
// coefficients are zero; the regular and smooth tables in av1/common/filter.h
// have that shape, while fractional sharp filters retain the eight-tap path.
// A strip keeps its tap rows in registers and loads one new im row per
// output row.
func pred8VIM(dst []byte, dstStride int, src []byte, srcStride int, im []int16, imStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(imStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	if k[0] != 0 || k[7] != 0 {
		return pred8VIM8(dst, dstStride, src, srcStride, im, imStride, w, h, k)
	}
	im = im[1*imStride:]
	c1 := archsimd.BroadcastInt32x4(int32(k[1]))
	c2 := archsimd.BroadcastInt32x4(int32(k[2]))
	c3 := archsimd.BroadcastInt32x4(int32(k[3]))
	c4 := archsimd.BroadcastInt32x4(int32(k[4]))
	c5 := archsimd.BroadcastInt32x4(int32(k[5]))
	c6 := archsimd.BroadcastInt32x4(int32(k[6]))
	bias := archsimd.BroadcastInt32x4(-261120)
	sh := archsimd.BroadcastInt32x4(-11)
	var acc archsimd.Uint16x8
	sad := 0
	for x := 0; x+8 <= w; x += 8 {
		// Strip x: the first 5 rows load up front, then each output row
		// loads its bottom tap row and the rows slide down one register.
		r0 := archsimd.LoadInt16x8Array((*[8]int16)(im[x : x+8]))
		r1 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+1*imStride : x+1*imStride+8]))
		r2 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+2*imStride : x+2*imStride+8]))
		r3 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+3*imStride : x+3*imStride+8]))
		r4 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+4*imStride : x+4*imStride+8]))
		for y := 0; y < h; y++ {
			o := (y+5)*imStride + x
			r5 := archsimd.LoadInt16x8Array((*[8]int16)(im[o : o+8]))
			lo := r0.ExtendLo4ToInt32().MulAdd(c1, bias)
			hi := r0.HiToLo().ExtendLo4ToInt32().MulAdd(c1, bias)
			lo = r1.ExtendLo4ToInt32().MulAdd(c2, lo)
			hi = r1.HiToLo().ExtendLo4ToInt32().MulAdd(c2, hi)
			lo = r2.ExtendLo4ToInt32().MulAdd(c3, lo)
			hi = r2.HiToLo().ExtendLo4ToInt32().MulAdd(c3, hi)
			lo = r3.ExtendLo4ToInt32().MulAdd(c4, lo)
			hi = r3.HiToLo().ExtendLo4ToInt32().MulAdd(c4, hi)
			lo = r4.ExtendLo4ToInt32().MulAdd(c5, lo)
			hi = r4.HiToLo().ExtendLo4ToInt32().MulAdd(c5, hi)
			lo = r5.ExtendLo4ToInt32().MulAdd(c6, lo)
			hi = r5.HiToLo().ExtendLo4ToInt32().MulAdd(c6, hi)
			r16 := lo.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s().
				InterleaveLo(hi.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
			out := r16.SaturateToUint8()
			if src != nil {
				// Eight output bytes in the low half; compare them with the
				// matching eight source bytes.
				so := y*srcStride + x
				var pair [16]uint8
				*(*[8]uint8)(pair[:8]) = *(*[8]uint8)(src[so : so+8])
				acc = acc.Add(absDiffU8x16(out, archsimd.LoadUint8x16Array(&pair)).ExtendLo8ToUint16())
			} else {
				do := y*dstStride + x
				put64((*[8]byte)(dst[do:do+8]), out.ReshapeToUint64s().GetElem(0))
			}
			r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
			if src != nil && y&15 == 15 {
				sad += sumU16(acc)
				acc = archsimd.Uint16x8{}
			}
		}
	}
	return sad + sumU16(acc)
}

// pred8VIM8 retains the full eight-tap path for kernels with nonzero outer
// coefficients.
func pred8VIM8(dst []byte, dstStride int, src []byte, srcStride int, im []int16, imStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(imStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	c0 := archsimd.BroadcastInt32x4(int32(k[0]))
	c1 := archsimd.BroadcastInt32x4(int32(k[1]))
	c2 := archsimd.BroadcastInt32x4(int32(k[2]))
	c3 := archsimd.BroadcastInt32x4(int32(k[3]))
	c4 := archsimd.BroadcastInt32x4(int32(k[4]))
	c5 := archsimd.BroadcastInt32x4(int32(k[5]))
	c6 := archsimd.BroadcastInt32x4(int32(k[6]))
	c7 := archsimd.BroadcastInt32x4(int32(k[7]))
	bias := archsimd.BroadcastInt32x4(-261120)
	sh := archsimd.BroadcastInt32x4(-11)
	var acc archsimd.Uint16x8
	sad := 0
	for x := 0; x+8 <= w; x += 8 {
		// Strip x: the first 7 rows load up front, then each output row
		// loads its bottom tap row and the rows slide down one register.
		r0 := archsimd.LoadInt16x8Array((*[8]int16)(im[x : x+8]))
		r1 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+1*imStride : x+1*imStride+8]))
		r2 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+2*imStride : x+2*imStride+8]))
		r3 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+3*imStride : x+3*imStride+8]))
		r4 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+4*imStride : x+4*imStride+8]))
		r5 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+5*imStride : x+5*imStride+8]))
		r6 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+6*imStride : x+6*imStride+8]))
		for y := 0; y < h; y++ {
			o := (y+7)*imStride + x
			r7 := archsimd.LoadInt16x8Array((*[8]int16)(im[o : o+8]))
			lo := r0.ExtendLo4ToInt32().MulAdd(c0, bias)
			hi := r0.HiToLo().ExtendLo4ToInt32().MulAdd(c0, bias)
			lo = r1.ExtendLo4ToInt32().MulAdd(c1, lo)
			hi = r1.HiToLo().ExtendLo4ToInt32().MulAdd(c1, hi)
			lo = r2.ExtendLo4ToInt32().MulAdd(c2, lo)
			hi = r2.HiToLo().ExtendLo4ToInt32().MulAdd(c2, hi)
			lo = r3.ExtendLo4ToInt32().MulAdd(c3, lo)
			hi = r3.HiToLo().ExtendLo4ToInt32().MulAdd(c3, hi)
			lo = r4.ExtendLo4ToInt32().MulAdd(c4, lo)
			hi = r4.HiToLo().ExtendLo4ToInt32().MulAdd(c4, hi)
			lo = r5.ExtendLo4ToInt32().MulAdd(c5, lo)
			hi = r5.HiToLo().ExtendLo4ToInt32().MulAdd(c5, hi)
			lo = r6.ExtendLo4ToInt32().MulAdd(c6, lo)
			hi = r6.HiToLo().ExtendLo4ToInt32().MulAdd(c6, hi)
			lo = r7.ExtendLo4ToInt32().MulAdd(c7, lo)
			hi = r7.HiToLo().ExtendLo4ToInt32().MulAdd(c7, hi)
			r16 := lo.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s().
				InterleaveLo(hi.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
			out := r16.SaturateToUint8()
			if src != nil {
				// Eight output bytes in the low half; compare them with the
				// matching eight source bytes.
				so := y*srcStride + x
				var pair [16]uint8
				*(*[8]uint8)(pair[:8]) = *(*[8]uint8)(src[so : so+8])
				acc = acc.Add(absDiffU8x16(out, archsimd.LoadUint8x16Array(&pair)).ExtendLo8ToUint16())
			} else {
				do := y*dstStride + x
				put64((*[8]byte)(dst[do:do+8]), out.ReshapeToUint64s().GetElem(0))
			}
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
			if src != nil && y&15 == 15 {
				sad += sumU16(acc)
				acc = archsimd.Uint16x8{}
			}
		}
	}
	return sad + sumU16(acc)
}

func init() {
	addRawResidualToPredReconBlockImpl = addRawResidualToPredReconBlockSIMD
}

// addRawResidualToPredReconBlockSIMD reads and writes through bounds-checked
// windows: one per row pair and array for the 4- and 8-wide blocks (row 2i
// at w[:n], row 2i+1 at w[stride:]), one per row for the wider ones.
func addRawResidualToPredReconBlockSIMD(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	if h <= 0 || (w != 4 && w&7 != 0) {
		addRawResidualToPredReconBlockPureGo(dst, stride, pred, predStride, raw, w, h)
		return
	}
	if uint(stride) > rowStrideMax || uint(predStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	switch w {
	case 4:
		rawPredRecon4(dst, stride, pred, predStride, raw, h)
	case 8:
		rawPredRecon8(dst, stride, pred, predStride, raw, h)
	default:
		rawPredReconWide(dst, stride, pred, predStride, raw, w, h)
	}
}

// Saturating v+8 differs from the int64 sum only near MaxInt32, where both
// rounded values saturate to 32767 on narrowing. Saturating pred+residual to
// int16 before the byte clamp likewise preserves the final [0, 255] result.
func rawPredReconRoundPair(a, b *[4]int32, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array(a).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	hi := archsimd.LoadInt32x4Array(b).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	return lo.ToBits().ReshapeToUint64s().InterleaveLo(hi.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

func rawPredReconRound(r *[8]int32, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	return rawPredReconRoundPair((*[4]int32)(r[0:4]), (*[4]int32)(r[4:8]), c8, sh4)
}

func rawPredReconAdd(d archsimd.Uint8x16, r archsimd.Int16x8) archsimd.Uint64x2 {
	return d.ReshapeToUint16s().BitsToInt16().AddSaturated(r).SaturateToUint8().ReshapeToUint64s()
}

// le32 is the little-endian uint32 of b (one 4-byte load).
func le32(b *[4]byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

// put32 stores v little-endian into b (one 4-byte store).
func put32(b *[4]byte, v uint32) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func rawPredRecon4(dst []byte, ds int, pred []byte, ps int, raw []int32, h int) {
	if uint(ds) > rowStrideMax || uint(ps) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	dn, pn := ds+4, ps+4
	for i := 0; i < h/2; i++ {
		po, do := 2*i*ps, 2*i*ds
		pw := pred[po : po+pn : po+pn]
		r := (*[8]int32)(raw[8*i : 8*i+8])
		rv := rawPredReconRound(r, c8, sh4)
		pv := archsimd.BroadcastUint64x2(uint64(le32((*[4]byte)(pw))) | uint64(le32((*[4]byte)(pw[ps:])))<<32).ReshapeToUint8s()
		out := rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0)
		dw := dst[do : do+dn : do+dn]
		put32((*[4]byte)(dw), uint32(out))
		put32((*[4]byte)(dw[ds:]), uint32(out>>32))
	}
	if h&1 != 0 {
		y := h - 1
		r := (*[4]int32)(raw[4*y : 4*y+4])
		rv := rawPredReconRoundPair(r, r, c8, sh4)
		pv := archsimd.BroadcastUint64x2(uint64(le32((*[4]byte)(pred[y*ps : y*ps+4])))).ReshapeToUint8s()
		put32((*[4]byte)(dst[y*ds:y*ds+4]), uint32(rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0)))
	}
}

func rawPredRecon8(dst []byte, ds int, pred []byte, ps int, raw []int32, h int) {
	if uint(ds) > rowStrideMax || uint(ps) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	dn, pn := ds+8, ps+8
	for i := 0; i < h/2; i++ {
		po, do := 2*i*ps, 2*i*ds
		pw := pred[po : po+pn : po+pn]
		r := (*[16]int32)(raw[16*i : 16*i+16])
		ra := rawPredReconRound((*[8]int32)(r[0:8]), c8, sh4)
		rb := rawPredReconRound((*[8]int32)(r[8:16]), c8, sh4)
		pair := [2]uint64{le64((*[8]byte)(pw)), le64((*[8]byte)(pw[ps:]))}
		pv := archsimd.LoadUint64x2Array(&pair).ReshapeToUint8s()
		dw := dst[do : do+dn : do+dn]
		put64((*[8]byte)(dw), rawPredReconAdd(pv.InterleaveLo(zero), ra).GetElem(0))
		put64((*[8]byte)(dw[ds:]), rawPredReconAdd(pv.InterleaveHi(zero), rb).GetElem(0))
	}
	if h&1 != 0 {
		y := h - 1
		r := rawPredReconRound((*[8]int32)(raw[8*y:8*y+8]), c8, sh4)
		pv := archsimd.BroadcastUint64x2(le64((*[8]byte)(pred[y*ps : y*ps+8]))).ReshapeToUint8s()
		put64((*[8]byte)(dst[y*ds:y*ds+8]), rawPredReconAdd(pv.InterleaveLo(zero), r).GetElem(0))
	}
}

func rawPredReconWide(dst []byte, ds int, pred []byte, ps int, raw []int32, w, h int) {
	if uint(ds) > rowStrideMax || uint(ps) > rowStrideMax || uint(h) > rowsMax || uint(w) > rowsMax {
		panic(errKernelArgs)
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	switch w {
	case 16:
		// One window per row pair and array; the raw rows are consecutive
		// 16-entry arrays.
		dn, pn := ds+16, ps+16
		for i := 0; i < h/2; i++ {
			po, do := 2*i*ps, 2*i*ds
			pw, dw := pred[po:po+pn:po+pn], dst[do:do+dn:do+dn]
			r := (*[32]int32)(raw[32*i : 32*i+32])
			rawPredRecon16((*[16]uint8)(dw), (*[16]uint8)(pw), (*[16]int32)(r[0:16]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(dw[ds:]), (*[16]uint8)(pw[ps:]), (*[16]int32)(r[16:32]), zero, c8, sh4)
		}
		if h&1 != 0 {
			y := h - 1
			rawPredRecon16((*[16]uint8)(dst[y*ds:y*ds+16]), (*[16]uint8)(pred[y*ps:y*ps+16]), (*[16]int32)(raw[16*y:16*y+16]), zero, c8, sh4)
		}
		return
	case 32:
		for y := 0; y < h; y++ {
			p := (*[32]uint8)(pred[y*ps : y*ps+32])
			d := (*[32]uint8)(dst[y*ds : y*ds+32])
			r := (*[32]int32)(raw[32*y : 32*y+32])
			rawPredRecon16((*[16]uint8)(d[0:16]), (*[16]uint8)(p[0:16]), (*[16]int32)(r[0:16]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[16:32]), (*[16]uint8)(p[16:32]), (*[16]int32)(r[16:32]), zero, c8, sh4)
		}
		return
	case 64:
		for y := 0; y < h; y++ {
			p := (*[64]uint8)(pred[y*ps : y*ps+64])
			d := (*[64]uint8)(dst[y*ds : y*ds+64])
			r := (*[64]int32)(raw[64*y : 64*y+64])
			rawPredRecon16((*[16]uint8)(d[0:16]), (*[16]uint8)(p[0:16]), (*[16]int32)(r[0:16]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[16:32]), (*[16]uint8)(p[16:32]), (*[16]int32)(r[16:32]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[32:48]), (*[16]uint8)(p[32:48]), (*[16]int32)(r[32:48]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[48:64]), (*[16]uint8)(p[48:64]), (*[16]int32)(r[48:64]), zero, c8, sh4)
		}
		return
	}
	for y := 0; y < h; y++ {
		p := pred[y*ps : y*ps+w]
		d := dst[y*ds : y*ds+w]
		r := raw[y*w : y*w+w]
		x := 0
		for ; x+16 <= w; x += 16 {
			rawPredRecon16((*[16]uint8)(d[x:x+16]), (*[16]uint8)(p[x:x+16]), (*[16]int32)(r[x:x+16]), zero, c8, sh4)
		}
		if x+8 <= w {
			rv := rawPredReconRound((*[8]int32)(r[x:x+8]), c8, sh4)
			pv := archsimd.BroadcastUint64x2(le64((*[8]byte)(p[x : x+8]))).ReshapeToUint8s()
			put64((*[8]byte)(d[x:x+8]), rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0))
		}
	}
}

// rawPredRecon16 writes 16 samples: d = clip(p + round(r)).
func rawPredRecon16(d, p *[16]uint8, r *[16]int32, zero archsimd.Uint8x16, c8, sh4 archsimd.Int32x4) {
	lo := rawPredReconRound((*[8]int32)(r[0:8]), c8, sh4)
	hi := rawPredReconRound((*[8]int32)(r[8:16]), c8, sh4)
	pv := archsimd.LoadUint8x16Array(p)
	a := rawPredReconAdd(pv.InterleaveLo(zero), lo)
	b := rawPredReconAdd(pv.InterleaveHi(zero), hi)
	a.InterleaveLo(b).ReshapeToUint8s().StoreArray(d)
}

// sadSIMDBound reports whether init binds the dispatch variables to the
// Go-native-SIMD kernels. Without NEON the portable scalar defaults from sad.go
// remain in place.
var sadSIMDBound = cpu.Detected.NEON

// init binds the dispatch variables to the NEON Go-native-SIMD kernels.
func init() {
	if sadSIMDBound {
		bindSIMDSAD()
	}
}

// --- lowercase wrappers (the hot-path dispatch surface) ---------------------

func sad8x8(src, ref []byte, stride int, limit int) int {
	return sad8x8Same(src, ref, stride)
}

func sad16x16(src, ref []byte, stride int) int {
	return sad16x16Same(src, ref, stride)
}

func sad32x32(src, ref []byte, stride int) int {
	return sad32x32SIMD(src, stride, ref, stride)
}

// sad64x64 is the direct 64-wide NEON-style leaf, not a four-call composition.
func sad64x64(src, ref []byte, stride int) int {
	return sad64x64SIMD(src, stride, ref, stride)
}

func sad8x8x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad8x8x4Step4SIMD(src, ref, stride)
}

func sad8x8x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x8x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad16x16x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad16x16x4Step4SIMD(src, ref, stride)
}

func sad32x32x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad32x32x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad32x32x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad32x32x4Step4SIMD(src, ref, stride)
}

func sad8x8Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad8x8SIMD(src, srcStride, ref, refStride)
}

func sad16x16Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad16x16SIMD(src, srcStride, ref, refStride)
}

func sad32x32Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad32x32SIMD(src, srcStride, ref, refStride)
}

func sad64x64Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad64x64SIMD(src, srcStride, ref, refStride)
}

func sad8x8CompoundAvgBlock(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	return sad8x8CompoundAvgSIMD(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
}

// absDiffU8x16 computes unsigned byte absolute differences using only the
// public archsimd API. The maximum is never below the minimum, so subtraction
// is exact without saturation.
func absDiffU8x16(a, b archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Max(b).Sub(a.Min(b))
}

// widen16 widens one 16-byte abs-diff vector to a Uint16x8 by adding its low
// and high halves lane-wise.
func widen16(absd archsimd.Uint8x16) archsimd.Uint16x8 {
	return absd.ExtendLo8ToUint16().Add(absd.HiToLo().ExtendLo8ToUint16())
}

// absAcc16 adds the absolute differences of one 16-byte row pair into acc. The
// row's widened sum is formed before it joins acc, so the loop-carried chain is
// one add per row. Each Uint16 lane gains at most 2*255 per call, so a lane is
// exact for up to 128 calls; every caller accumulates at most 64 rows.
func absAcc16(acc archsimd.Uint16x8, s, r archsimd.Uint8x16) archsimd.Uint16x8 {
	return acc.Add(widen16(absDiffU8x16(s, r)))
}

// sumU16 returns the exact sum of the eight lanes of acc. The lanes are widened
// to 32 bits after pairwise addition so the total cannot wrap. Each lane is at
// most 2*32*255 for the largest caller, so a pair fits in uint16.
func sumU16(acc archsimd.Uint16x8) int {
	return int(acc.ConcatAddPairs(acc).ExtendLo4ToUint32().ReduceSum())
}

// le64 is the little-endian uint64 of the eight bytes of b; the compiler
// combines the byte loads into one 8-byte load.
func le64(b *[8]byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// pack2 packs the 8-byte rows a (lanes 0..7) and b (lanes 8..15) into one
// vector, through a stack store and a vector load (cheaper on Apple cores
// than two general-register lane inserts).
func pack2(a, b *[8]byte) archsimd.Uint8x16 {
	rows := [2]uint64{le64(a), le64(b)}
	return archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s()
}

// The single-reference kernels take one bounds-checked window per row pair:
// w := s[o : o+stride+n : o+stride+n] holds row 2i at w[:n] and row 2i+1 at
// w[stride:], and the compiler proves both from the one window compare (its
// length folds to stride+n). Strides are bounded by rowStrideMax and row
// counts by rowsMax, so the offsets cannot overflow.

// sad8x8Rows is the 8x8 SAD with independent strides. A candidate's maximum
// total is 64*255 = 16,320, so the uint16 lane reduction is exact.
func sad8x8Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+8, refStride+8
	acc := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[refStride:]))
		acc = acc.Add(widen16(absDiffU8x16(s, r)))
	}
	return int(acc.ReduceSum())
}

// sad8x8x4Rows computes four 8x8 SADs of one source block against four
// reference origins sharing one stride. Each source row pair is packed once
// and reused across all four references.
func sad8x8x4Rows(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		w := src[o : o+n : o+n]
		s := pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))
		w = ref0[o : o+n : o+n]
		c0 = c0.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
		w = ref1[o : o+n : o+n]
		c1 = c1.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
		w = ref2[o : o+n : o+n]
		c2 = c2.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
		w = ref3[o : o+n : o+n]
		c3 = c3.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
	}
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}

// sad8x8x4Step4Rows is sad8x8x4Rows for the reference origins ref+0, ref+4,
// ref+8 and ref+12: one 20-byte reference window per row (two per row-pair
// window) holds all four candidates at constant offsets.
func sad8x8x4Step4Rows(src, ref []byte, stride int) (int, int, int, int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := stride+8, stride+20
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		w := src[o : o+sn : o+sn]
		s := pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))
		rw := ref[o : o+rn : o+rn]
		r0, r1 := (*[20]byte)(rw), (*[20]byte)(rw[stride:])
		c0 = c0.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[0:8]), (*[8]byte)(r1[0:8])))))
		c1 = c1.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[4:12]), (*[8]byte)(r1[4:12])))))
		c2 = c2.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[8:16]), (*[8]byte)(r1[8:16])))))
		c3 = c3.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[12:20]), (*[8]byte)(r1[12:20])))))
	}
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}

// sad8x8CompoundRows computes SAD(src, round((ref0+ref1)/2)) over one 8x8
// block. URHADD (Average) is exactly (a+b+1)>>1 per byte.
func sad8x8CompoundRows(src []byte, srcStride int, ref0 []byte, s0 int, ref1 []byte, s1 int) int {
	if uint(srcStride) > rowStrideMax || uint(s0) > rowStrideMax || uint(s1) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, n0, n1 := srcStride+8, s0+8, s1+8
	acc := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		so, o0, o1 := 2*i*srcStride, 2*i*s0, 2*i*s1
		sw := src[so : so+sn : so+sn]
		w0 := ref0[o0 : o0+n0 : o0+n0]
		w1 := ref1[o1 : o1+n1 : o1+n1]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		pred := pack2((*[8]byte)(w0), (*[8]byte)(w0[s0:])).Average(pack2((*[8]byte)(w1), (*[8]byte)(w1[s1:])))
		acc = acc.Add(widen16(absDiffU8x16(s, pred)))
	}
	return int(acc.ReduceSum())
}

// sad16ColsDual sums the SAD of one 16-column strip over h rows with
// independent strides. h is even (16, 32, or 64 in every caller). Even and
// odd rows feed separate accumulators so the two add chains overlap. Each
// lane gets at most 2*255 per row into its accumulator, so h <= 64 keeps the
// lanes exact.
func sad16ColsDual(src []byte, srcStride int, ref []byte, refStride int, h int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+16, refStride+16
	even := archsimd.BroadcastUint16x8(0)
	odd := archsimd.BroadcastUint16x8(0)
	for i := 0; i < h/2; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		// absAcc16 written out: its nested inlined helpers left five
		// inline-mark NOPs per row pair.
		s := archsimd.LoadUint8x16Array((*[16]uint8)(sw))
		r := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
		d := s.Max(r).Sub(s.Min(r))
		even = even.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
		s = archsimd.LoadUint8x16Array((*[16]uint8)(sw[srcStride:]))
		r = archsimd.LoadUint8x16Array((*[16]uint8)(rw[refStride:]))
		d = s.Max(r).Sub(s.Min(r))
		odd = odd.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
	}
	if h == 16 {
		// 16*16*255 = 65,280, so the horizontal uint16 sum is exact.
		return int(even.Add(odd).ReduceSum())
	}
	if h == 32 {
		// Each accumulator holds 16 rows and fits independently in uint16.
		return int(even.ReduceSum()) + int(odd.ReduceSum())
	}
	return sumU16(even) + sumU16(odd)
}

// sadAcc16 adds the absolute differences of the 16-byte rows s and r to acc,
// widened to uint16 lanes (each lane gains at most 2*255).
func sadAcc16(acc archsimd.Uint16x8, s, r *[16]uint8) archsimd.Uint16x8 {
	a, b := archsimd.LoadUint8x16Array(s), archsimd.LoadUint8x16Array(r)
	d := a.Max(b).Sub(a.Min(b))
	return acc.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
}

// sad32Rows is the 32x32 SAD with independent strides, one 32-byte window
// per row pair and array. Each of the four accumulators (column half by row
// parity) holds 16 rows, so its horizontal uint16 sum is exact.
func sad32Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+32, refStride+32
	var le, lo, re, ro archsimd.Uint16x8
	for i := 0; i < 16; i++ {
		so, oo := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[oo : oo+rn : oo+rn]
		s0, r0 := (*[32]uint8)(sw), (*[32]uint8)(rw)
		s1, r1 := (*[32]uint8)(sw[srcStride:]), (*[32]uint8)(rw[refStride:])
		le = sadAcc16(le, (*[16]uint8)(s0[0:16]), (*[16]uint8)(r0[0:16]))
		re = sadAcc16(re, (*[16]uint8)(s0[16:32]), (*[16]uint8)(r0[16:32]))
		lo = sadAcc16(lo, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		ro = sadAcc16(ro, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
	}
	return int(le.ReduceSum()) + int(lo.ReduceSum()) + int(re.ReduceSum()) + int(ro.ReduceSum())
}

// sad64Rows is the 64x64 SAD with independent strides, one 64-byte window
// per row pair and array. Each of the eight accumulators (16-column strip by
// row parity) holds 32 rows: a lane is at most 32*2*255 = 16,320, so the
// pairwise sumU16 reduction is exact.
func sad64Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+64, refStride+64
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Uint16x8
	for i := 0; i < 32; i++ {
		so, oo := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[oo : oo+rn : oo+rn]
		s0, r0 := (*[64]uint8)(sw), (*[64]uint8)(rw)
		s1, r1 := (*[64]uint8)(sw[srcStride:]), (*[64]uint8)(rw[refStride:])
		a0 = sadAcc16(a0, (*[16]uint8)(s0[0:16]), (*[16]uint8)(r0[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s0[16:32]), (*[16]uint8)(r0[16:32]))
		a2 = sadAcc16(a2, (*[16]uint8)(s0[32:48]), (*[16]uint8)(r0[32:48]))
		a3 = sadAcc16(a3, (*[16]uint8)(s0[48:64]), (*[16]uint8)(r0[48:64]))
		b0 = sadAcc16(b0, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		b1 = sadAcc16(b1, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
		b2 = sadAcc16(b2, (*[16]uint8)(s1[32:48]), (*[16]uint8)(r1[32:48]))
		b3 = sadAcc16(b3, (*[16]uint8)(s1[48:64]), (*[16]uint8)(r1[48:64]))
	}
	return sumU16(a0) + sumU16(a1) + sumU16(a2) + sumU16(a3) +
		sumU16(b0) + sumU16(b1) + sumU16(b2) + sumU16(b3)
}

// sad16x16Same is the 16x16 SAD of two blocks sharing one stride: one
// offset per row pair serves both windows.
func sad16x16Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 16
	even := archsimd.BroadcastUint16x8(0)
	odd := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		s := archsimd.LoadUint8x16Array((*[16]uint8)(sw))
		r := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
		d := s.Max(r).Sub(s.Min(r))
		even = even.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
		s = archsimd.LoadUint8x16Array((*[16]uint8)(sw[stride:]))
		r = archsimd.LoadUint8x16Array((*[16]uint8)(rw[stride:]))
		d = s.Max(r).Sub(s.Min(r))
		odd = odd.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
	}
	// 16*16*255 = 65,280, so the horizontal uint16 sum is exact.
	return int(even.Add(odd).ReduceSum())
}

// sad16x16Packed is the 16x16 SAD against a packed 16-stride reference
// block (the prediction scratch of the near-MV and compound checks): the
// reference rows sit at constant offsets in one array, so only the source
// pays a window per row pair.
func sad16x16Packed(src []byte, srcStride int, ref *[256]uint8) int {
	if uint(srcStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := srcStride + 16
	even := archsimd.BroadcastUint16x8(0)
	odd := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * srcStride
		sw := src[o : o+n : o+n]
		rw := (*[32]uint8)(ref[32*i : 32*i+32])
		s := archsimd.LoadUint8x16Array((*[16]uint8)(sw))
		r := archsimd.LoadUint8x16Array((*[16]uint8)(rw[0:16]))
		d := s.Max(r).Sub(s.Min(r))
		even = even.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
		s = archsimd.LoadUint8x16Array((*[16]uint8)(sw[srcStride:]))
		r = archsimd.LoadUint8x16Array((*[16]uint8)(rw[16:32]))
		d = s.Max(r).Sub(s.Min(r))
		odd = odd.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
	}
	return int(even.Add(odd).ReduceSum())
}

// sad8x8Same is the 8x8 SAD of two blocks sharing one stride.
func sad8x8Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	acc := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[stride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[stride:]))
		acc = acc.Add(widen16(absDiffU8x16(s, r)))
	}
	return int(acc.ReduceSum())
}

// sad16ColsX4 is sad16ColsDual for four references sharing one stride, as
// bounds-checked row windows (this shape measured neutral against the
// pointer form). Each source row is loaded once and reused across the four
// candidates. With stride and h checked against rowStrideMax and rowsMax,
// each row window is one bounds check.
func sad16ColsX4(src, ref0, ref1, ref2, ref3 []byte, stride int, h int) (int, int, int, int) {
	if uint(stride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	a0 := archsimd.BroadcastUint16x8(0)
	a1 := archsimd.BroadcastUint16x8(0)
	a2 := archsimd.BroadcastUint16x8(0)
	a3 := archsimd.BroadcastUint16x8(0)
	for row := 0; row < h; row++ {
		o := row * stride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[o : o+16]))
		a0 = absAcc16(a0, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref0[o:o+16])))
		a1 = absAcc16(a1, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref1[o:o+16])))
		a2 = absAcc16(a2, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref2[o:o+16])))
		a3 = absAcc16(a3, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref3[o:o+16])))
	}
	if h == 16 {
		return int(a0.ReduceSum()), int(a1.ReduceSum()), int(a2.ReduceSum()), int(a3.ReduceSum())
	}
	return sumU16(a0), sumU16(a1), sumU16(a2), sumU16(a3)
}

// boxInterpolateRow8 is InterpolateRow_C for 8-bit samples on NEON, as
// libyuv's InterpolateRow_NEON: (s0*(256-f) + s1*f + 128) >> 8 by widening
// multiply-accumulate, and the f == 128 HalfRow as a rounding average.
func boxInterpolateRow8(dst, s0, s1 []uint8, f int) {
	n := len(s0)
	s1, dst = s1[:n], dst[:n]
	x := 0
	if f == 128 {
		for ; x <= n-16; x += 16 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(s0[x:]))
			b := archsimd.LoadUint8x16Array((*[16]uint8)(s1[x:]))
			a.Average(b).StoreArray((*[16]uint8)(dst[x:]))
		}
	} else if n >= 16 {
		f0 := archsimd.BroadcastUint8x16(uint8(256 - f))
		f1 := archsimd.BroadcastUint8x16(uint8(f))
		bias := archsimd.BroadcastUint16x8(128)
		sh8 := archsimd.BroadcastInt16x8(-8)
		for ; x <= n-16; x += 16 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(s0[x:]))
			b := archsimd.LoadUint8x16Array((*[16]uint8)(s1[x:]))
			lo := a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).Shift(sh8).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
			hi := a.HiToLo().MulWidenLo(f0.HiToLo()).Add(b.HiToLo().MulWidenLo(f1.HiToLo())).Add(bias).Shift(sh8).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
			lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s().StoreArray((*[16]uint8)(dst[x:]))
		}
	}
	if x < n {
		boxInterpolateRow(dst[x:], s0[x:], s1[x:], f, 128)
	}
}

// libyuv's ScaleFilterCols_NEON gathers each output's two taps with ld2 lane
// loads and blends in 32-bit lanes. archsimd has no lane load, so this port
// takes the taps from TBL against the 32-byte source window a 16-output chunk
// can reach (the bilinear step is at most 2: 15 outputs advance at most 30
// samples). The gather indices and the 16-bit blend weights depend on the
// plane geometry only, never on the row, so they are built once per
// (sw, dw, x, dx) and every row of every plane with that geometry runs the
// same table.
//
// The blend is ScaleFilterCols_C's ARM BLENDER,
//
//	a + ((f*(b-a) + 0x8000) >> 16) == (g*a + f*b + 0x8000) >> 16, g = 65536-f,
//
// split into bytes f = 256*fh + fl, g = 256*gh + gl with gh = 255-fh and
// gl = 256-fl so every partial sum fits a 16-bit lane. With
// U = gh*a + fh*b and V = gl*a + fl*b the nested floor division gives
// (U + 128 + (V >> 8)) >> 8, at most 65408.
const (
	boxColsChunk   = 16  // outputs per chunk
	boxColsWindow  = 32  // source bytes a chunk reads
	boxColsEntry   = 192 // table bytes per chunk
	boxColsSlots   = 4   // cached geometries
	boxColsMinRows = 8   // rows that pay for building a table
	boxColsCoefOff = 64  // offset of the weights; the indices come first
)

type boxColsPos struct{ base, out int32 }

// boxColsPlan is the column table of one plane geometry. Per chunk it holds
// four 16-byte TBL index vectors (a and b taps against the window's low and
// high 16 bytes) and, for each 8-output half, the u16 weights gh, fh,
// gl = 256-fl and fl.
type boxColsPlan struct {
	sw, dw, x, dx int
	valid         bool
	pos           []boxColsPos
	tab           []uint8
}

// boxColsCache keeps the plans of the last few plane geometries; the U and V
// planes of a layer share one, so steady state builds nothing.
type boxColsCache struct {
	plans [boxColsSlots]boxColsPlan
	next  int
}

// plan returns the table for a column pass over an sw-sample row producing dw
// samples from x in steps of dx, or nil when the scalar filter must run: the
// row or output is shorter than one chunk, the step is outside [1, 2] (the
// tap window would not cover a chunk), or rows is too few to pay for a build.
func (c *boxColsCache) plan(sw, dw, x, dx, rows int) *boxColsPlan {
	if sw < boxColsWindow || dw < boxColsChunk || x < 0 || dx < 1<<16 || dx > 2<<16 || (x+(dw-1)*dx)>>16 >= sw {
		return nil
	}
	for i := range c.plans {
		if p := &c.plans[i]; p.valid && p.sw == sw && p.dw == dw && p.x == x && p.dx == dx {
			return p
		}
	}
	if rows < boxColsMinRows {
		return nil
	}
	p := &c.plans[c.next]
	c.next = (c.next + 1) % boxColsSlots
	p.build(sw, dw, x, dx)
	return p
}

func (p *boxColsPlan) build(sw, dw, x, dx int) {
	n := (dw + boxColsChunk - 1) / boxColsChunk
	if cap(p.pos) < n {
		p.pos = make([]boxColsPos, n)
		p.tab = make([]uint8, n*boxColsEntry)
	}
	p.pos, p.tab = p.pos[:n], p.tab[:n*boxColsEntry]
	p.sw, p.dw, p.x, p.dx, p.valid = sw, dw, x, dx, true
	for c := range p.pos {
		// The last chunk is pulled back to end at dw; it rewrites some of the
		// previous chunk's outputs with the same values.
		out := min(c*boxColsChunk, dw-boxColsChunk)
		xc := x + out*dx
		base := min(xc>>16, sw-boxColsWindow)
		p.pos[c] = boxColsPos{base: int32(base), out: int32(out)}
		t := p.tab[c*boxColsEntry : (c+1)*boxColsEntry]
		for i := 0; i < boxColsChunk; i++ {
			xx := xc + i*dx
			xi := xx >> 16
			f := xx & 0xffff
			ia := xi - base
			ib := min(xi+1, sw-1) - base // the right tap clamps to the row
			t[i], t[16+i] = uint8(ia), uint8(ia-16)
			t[32+i], t[48+i] = uint8(ib), uint8(ib-16)
			fh, fl := f>>8, f&255
			w := t[boxColsCoefOff+(i>>3)*64+(i&7)*2:]
			w[0] = uint8(255 - fh) // gh
			w[16] = uint8(fh)
			w[32], w[33] = uint8(256-fl), uint8((256-fl)>>8) // gl is 256 at fl == 0
			w[48] = uint8(fl)
		}
	}
}

// boxColsBlend is the blend of one 8-output half; q holds its weights.
func boxColsBlend(a, b archsimd.Uint16x8, q *[64]uint8, k128 archsimd.Uint16x8, sh8 archsimd.Int16x8) archsimd.Uint16x8 {
	gh := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ReshapeToUint16s()
	fh := archsimd.LoadUint8x16Array((*[16]uint8)(q[16:32])).ReshapeToUint16s()
	gl := archsimd.LoadUint8x16Array((*[16]uint8)(q[32:48])).ReshapeToUint16s()
	fl := archsimd.LoadUint8x16Array((*[16]uint8)(q[48:64])).ReshapeToUint16s()
	u := fh.MulAdd(b, gh.MulAdd(a, k128))
	v := fl.MulAdd(b, gl.Mul(a))
	return u.Add(v.Shift(sh8))
}

// filter is boxFilterCols for one row. Each chunk's table is one checked
// 192-byte array of the plan's table; its source window and its 16 outputs
// are one bounds-checked window each (build keeps them inside s[:sw] and
// dst[:dw]).
func (p *boxColsPlan) filter(dst, s []uint8) {
	dst, s = dst[:p.dw], s[:p.sw]
	zero := archsimd.BroadcastUint8x16(0)
	k128 := archsimd.BroadcastUint16x8(128)
	sh8 := archsimd.BroadcastInt16x8(-8)
	tab, chunks := p.tab, p.pos
	if len(chunks) > 1<<20 {
		panic(errKernelArgs)
	}
	for c, pos := range chunks {
		o := c * boxColsEntry
		t := (*[boxColsEntry]uint8)(tab[o : o+boxColsEntry])
		// Through uint32 the offsets are non-negative, so each window is
		// one compare.
		base, out := int(uint32(pos.base)), int(uint32(pos.out))
		w := (*[boxColsWindow]uint8)(s[base : base+boxColsWindow])
		s0 := archsimd.LoadUint8x16Array((*[16]uint8)(w[0:16]))
		s1 := archsimd.LoadUint8x16Array((*[16]uint8)(w[16:32]))
		a := s0.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(t[0:16]))).
			Or(s1.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(t[16:32]))))
		b := s0.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(t[32:48]))).
			Or(s1.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(t[48:64]))))
		lo := boxColsBlend(a.InterleaveLo(zero).ReshapeToUint16s(), b.InterleaveLo(zero).ReshapeToUint16s(), (*[64]uint8)(t[boxColsCoefOff:boxColsCoefOff+64]), k128, sh8)
		hi := boxColsBlend(a.InterleaveHi(zero).ReshapeToUint16s(), b.InterleaveHi(zero).ReshapeToUint16s(), (*[64]uint8)(t[boxColsCoefOff+64:boxColsCoefOff+128]), k128, sh8)
		lo.ReshapeToUint8s().ConcatOdd(hi.ReshapeToUint8s()).StoreArray((*[16]uint8)(dst[out : out+boxColsChunk]))
	}
}

// svtBilRowH is the first (horizontal) pass over the 16 samples at p[0:16]
// (the taps' second sample at p[1:17]).
func svtBilRowH(p *[17]uint8, xoff int, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8, sh7 archsimd.Int16x8) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))
	switch xoff {
	case 0:
		return a
	case 4:
		return a.Average(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])))
	}
	return svtBil16(a, archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])), f0, f1, bias, sh7)
}

// svtBil16 is var_filter_block2d_bil_w16's arithmetic on 16 lanes:
// (a*f0 + b*f1 + 64) >> 7. sh7 is the shift count (-7) built once by the
// caller: ShiftAllRight(7) would rebuild it (MOVD + INS + DUP) per use. The
// high half saturates as UQXTN2 (SetElem of the low 64 bits) rather than a
// second UQXTN joined by ZIP1.
// f0 and f1 are broadcasts, so their high halves equal them; naming the
// halves lets the high products fuse to UMULL2 instead of INS + UMULL.
func svtBil16(a, b, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8, sh7 archsimd.Int16x8) archsimd.Uint8x16 {
	lo := a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).Shift(sh7).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
	hi := a.HiToLo().MulWidenLo(f0.HiToLo()).Add(b.HiToLo().MulWidenLo(f1.HiToLo())).Add(bias).Shift(sh7).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
	return lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s()
}

// svtSubpelVarianceArch is vfp->svf: the bilinear prediction of
// svtBilinearPredict folded straight into the residual statistics of
// svt_aom_variance (variance_neon.c variance_large_neon's accumulation:
// widened differences, int16 sums flushed to int32, squares in int32 lanes)
// instead of being stored and re-read. Each row is one bounds-checked
// window per array.
func svtSubpelVarianceArch(scratch *svtSubpelScratch, ref []byte, refStride int, src []byte, srcStride, bw, bh, xoff, yoff int) uint32 {
	if bw&7 != 0 || bw > 64 || bh <= 0 ||
		!unsafeutil.GridFit(len(ref), 0, refStride, bw+16, bh+1) ||
		!unsafeutil.GridFit(len(src), 0, srcStride, max(bw, 16), bh) {
		svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
		_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, bw, srcStride, 0, 0, bw, bh)
		return v
	}
	fx0 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][0]))
	fx1 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][1]))
	fy0 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[yoff][0]))
	fy1 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[yoff][1]))
	bias := archsimd.BroadcastUint16x8(64)
	sh7 := archsimd.BroadcastInt16x8(-7)
	gr := unsafeutil.NewGrid(ref, 0, refStride, bw+16, bh+1)
	gs := unsafeutil.NewGrid(src, 0, srcStride, max(bw, 16), bh)
	var sse0, sse1, sum32 archsimd.Int32x4
	for x := 0; x < bw; x += 16 {
		wide := x+16 <= bw
		prev := svtBilRowH((*[17]uint8)(gr.Row24(0, x)[:17]), xoff, fx0, fx1, bias, sh7)
		for r := 0; r < bh; {
			group := min(r+32, bh)
			var sum16 archsimd.Int16x8
			if wide {
				switch yoff {
				case 0:
					for ; r < group; r++ {
						out := prev
						if r+1 < bh {
							prev = svtBilRowH((*[17]uint8)(gr.Row24(r+1, x)[:17]), xoff, fx0, fx1, bias, sh7)
						}
						sv := archsimd.LoadUint8x16Array(gs.Row16(r, x))
						d := out.Sub(sv)
						m := sv.Greater(out).ToInt8x16().ToBits()
						dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
						dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
						sum16 = sum16.Add(dl.Add(dh))
						sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
						sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
					}
				case 4:
					for ; r < group; r++ {
						cur := svtBilRowH((*[17]uint8)(gr.Row24(r+1, x)[:17]), xoff, fx0, fx1, bias, sh7)
						out := prev.Average(cur)
						prev = cur
						sv := archsimd.LoadUint8x16Array(gs.Row16(r, x))
						d := out.Sub(sv)
						m := sv.Greater(out).ToInt8x16().ToBits()
						dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
						dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
						sum16 = sum16.Add(dl.Add(dh))
						sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
						sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
					}
				default:
					for ; r < group; r++ {
						cur := svtBilRowH((*[17]uint8)(gr.Row24(r+1, x)[:17]), xoff, fx0, fx1, bias, sh7)
						out := svtBil16(prev, cur, fy0, fy1, bias, sh7)
						prev = cur
						sv := archsimd.LoadUint8x16Array(gs.Row16(r, x))
						d := out.Sub(sv)
						m := sv.Greater(out).ToInt8x16().ToBits()
						dl := d.InterleaveLo(m).ReshapeToUint16s().BitsToInt16()
						dh := d.InterleaveHi(m).ReshapeToUint16s().BitsToInt16()
						sum16 = sum16.Add(dl.Add(dh))
						sse0 = sse0.Add(dl.MulWidenLo(dl).Add(dh.MulWidenLo(dh)))
						sse1 = sse1.Add(dl.HiToLo().MulWidenLo(dl.HiToLo()).Add(dh.HiToLo().MulWidenLo(dh.HiToLo())))
					}
				}
			} else {
				for ; r < group; r++ {
					var out archsimd.Uint8x16
					switch yoff {
					case 0:
						out = prev
						if r+1 < bh {
							prev = svtBilRowH((*[17]uint8)(gr.Row24(r+1, x)[:17]), xoff, fx0, fx1, bias, sh7)
						}
					case 4:
						cur := svtBilRowH((*[17]uint8)(gr.Row24(r+1, x)[:17]), xoff, fx0, fx1, bias, sh7)
						out = prev.Average(cur)
						prev = cur
					default:
						cur := svtBilRowH((*[17]uint8)(gr.Row24(r+1, x)[:17]), xoff, fx0, fx1, bias, sh7)
						out = svtBil16(prev, cur, fy0, fy1, bias, sh7)
						prev = cur
					}
					sv := archsimd.LoadUint8x16Array(gs.Row16(r, x))
					dl := out.ExtendLo8ToUint16().BitsToInt16().Sub(sv.ExtendLo8ToUint16().BitsToInt16())
					sum16 = sum16.Add(dl)
					sse0 = sse0.Add(dl.MulWidenLo(dl))
					dl = dl.HiToLo()
					sse1 = sse1.Add(dl.MulWidenLo(dl))
				}
			}
			sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
		}
	}
	total, signed := reducePixelAcc(sse0.Add(sse1), sum32)
	return varianceFromStats(uint32(total), int32(signed), uint(bits.TrailingZeros(uint(bw*bh))))
}

// svtDownsample2DArch is svtDownsample2D on NEON: the 2x2 sums of 16 output
// samples are two row-widened adds folded by pairwise adds, then
// (sum + 2) >> 2 exactly as svt_aom_downsample_2d_c.
func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	two := archsimd.BroadcastUint16x8(2)
	// The count is built once: ShiftAllRight(2) rebuilds it (MOVD + INS + DUP)
	// at every use. The high half saturates as UQXTN2 rather than a second
	// UQXTN joined by ZIP1.
	sh2 := archsimd.BroadcastInt16x8(-2)
	for y := 0; y < h; y++ {
		r0 := src[(2*y)*srcStride:]
		r1 := src[(2*y+1)*srcStride:]
		d := dst[y*dstStride : y*dstStride+w]
		x := 0
		if len(r0) >= 2*w && len(r1) >= 2*w {
			// One 32-byte window per input row and 16 outputs: j = 2x
			// steps by 32 so each window is one compare.
			r0, r1 := r0[:2*w], r1[:2*w]
			r1 = r1[:len(r0)]
			for j := 0; j+32 <= len(r0); j += 32 {
				q0, q1 := (*[32]uint8)(r0[j:j+32]), (*[32]uint8)(r1[j:j+32])
				a := archsimd.LoadUint8x16Array((*[16]uint8)(q0[0:16]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(q1[0:16]))
				c := archsimd.LoadUint8x16Array((*[16]uint8)(q0[16:32]))
				e := archsimd.LoadUint8x16Array((*[16]uint8)(q1[16:32]))
				lo := a.ExtendLo8ToUint16().Add(b.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(a).Add(widenHi8(b)))
				hi := c.ExtendLo8ToUint16().Add(e.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(c).Add(widenHi8(e)))
				lf := lo.Add(two).Shift(sh2).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				hf := hi.Add(two).Shift(sh2).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				out := lf.SetElem(1, hf.GetElem(0)).ToBits().ReshapeToUint8s()
				x = j >> 1
				out.StoreArray((*[16]uint8)(d[x : x+16]))
			}
			x = w &^ 15
		}
		for ; x < w; x++ {
			s := uint32(r0[2*x]) + uint32(r0[2*x+1]) + uint32(r1[2*x]) + uint32(r1[2*x+1])
			d[x] = uint8((s + 2) >> 2)
		}
	}
}

// sad16xNx4 returns the SADs of the 16-wide, rows-tall source block against
// reference origins ref+0..ref+3 (svt_sad_loop_kernel's per-row position
// batch).
func sad16xNx4(src []byte, srcStride int, ref []byte, refStride, rows int) (int, int, int, int) {
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var c0, c1, c2, c3 archsimd.Uint16x8
	for r := 0; r < rows; r++ {
		so, ro := r*srcStride, r*refStride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[so : so+16]))
		p := (*[19]uint8)(ref[ro : ro+19])
		c0 = c0.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16])))))
		c1 = c1.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])))))
		c2 = c2.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18])))))
		c3 = c3.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19])))))
	}
	return sumU16(c0), sumU16(c1), sumU16(c2), sumU16(c3)
}

// svtSAD16xhx8 is the eight-position step of SVT-AV1's
// svt_sad_loop_kernel16xh_neon_dotprod / svt_sad_loop_kernel32xh_neon_dotprod
// (Source/Lib/ASM_NEON_DOTPROD/compute_sad_neon_dotprod.c, the
// search_area_width % 8 == 0 branch): the SADs of the 16-wide, rows-tall
// source block against ref+0..ref+7, as two sad16xhx4d batches that share the
// source row. rows <= 16.
//
// archsimd has no UABD/UDOT/UADALP, so each row's absolute differences are
// MAX-MIN and accumulate as uint16 lanes two ways: S sums the byte pairs as
// lo + 256*hi (wrapping) and H sums the high bytes. The pair sum lo + hi is
// S - 255*H modulo 2^16, exact because a position's total is at most
// 16*16*255 = 65280. The pairwise reduction is linear, so it runs on S and H
// before the single correction.
func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Uint16x8
	var h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Uint16x8
	// USHL by a hoisted -8: stock archsimd lowers ShiftAllRight to a
	// per-use DUP + USHL rather than USHR #8.
	hiShift := archsimd.BroadcastInt16x8(-8)
	// The source rows and the reference rows' 23-byte windows are grids,
	// checked once; the loop bound proves each row.
	// No rows add nothing to acc.
	if rows < 1 {
		return
	}
	if !unsafeutil.GridFit(len(src), 0, srcStride, 16, rows) || !unsafeutil.GridFit(len(ref), 0, refStride, 23, rows) {
		panic(errKernelArgs)
	}
	gs, gr := unsafeutil.NewGrid(src, 0, srcStride, 16, rows), unsafeutil.NewGrid(ref, 0, refStride, 23, rows)
	for r := 0; r < gs.Height(); r++ {
		// Written out rather than through load16 / absDiffU8x16: each inlined
		// helper call leaves an inline-mark NOP in the loop. One window per
		// row: the eight reference loads sit at constant offsets inside it.
		s := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gs, r, 0))
		p := unsafeutil.Row[[23]uint8](gr, r, 0)
		q := archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))
		d := s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s2, h2 = s2.Add(d), h2.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
	}
	// horizontal_add_4d_u32x4 for both batches at once: lane k of the
	// reduced vectors is position k.
	sv := s0.ConcatAddPairs(s1).ConcatAddPairs(s2.ConcatAddPairs(s3)).
		ConcatAddPairs(s4.ConcatAddPairs(s5).ConcatAddPairs(s6.ConcatAddPairs(s7)))
	hv := h0.ConcatAddPairs(h1).ConcatAddPairs(h2.ConcatAddPairs(h3)).
		ConcatAddPairs(h4.ConcatAddPairs(h5).ConcatAddPairs(h6.ConcatAddPairs(h7)))
	t := sv.Sub(hv.Shift(archsimd.BroadcastInt16x8(8))).Add(hv)
	lo := (*[4]uint32)(acc[0:4])
	hi := (*[4]uint32)(acc[4:8])
	archsimd.LoadUint32x4Array(lo).Add(t.ExtendLo4ToUint32()).StoreArray(lo)
	archsimd.LoadUint32x4Array(hi).Add(t.HiToLo().ExtendLo4ToUint32()).StoreArray(hi)
}

// svtSAD32xhx8 is the eight-position step of
// svt_sad_loop_kernel32xh_neon_dotprod (sad32xhx4d_neon_dotprod twice): the
// SADs of the 32-wide, rows-tall source block against ref+0..ref+7, both
// 16-column halves accumulating into the same byte-pair accumulators. The
// two in-register folds keep every uint16 lane at most 2*16*4*255 = 65280,
// so the S - 255*H correction is exact there and the last fold runs in
// uint32. rows <= 16.
func svtSAD32xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Uint16x8
	var h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Uint16x8
	hiShift := archsimd.BroadcastInt16x8(-8)
	// The source rows and the reference rows' 39-byte windows are grids,
	// checked once; the loop bound proves each row.
	// No rows add nothing to acc.
	if rows < 1 {
		return
	}
	if !unsafeutil.GridFit(len(src), 0, srcStride, 32, rows) || !unsafeutil.GridFit(len(ref), 0, refStride, 39, rows) {
		panic(errKernelArgs)
	}
	gs, gr := unsafeutil.NewGrid(src, 0, srcStride, 32, rows), unsafeutil.NewGrid(ref, 0, refStride, 39, rows)
	for r := 0; r < gs.Height(); r++ {
		sw := unsafeutil.Row[[32]uint8](gs, r, 0)
		rw := unsafeutil.Row[[39]uint8](gr, r, 0)
		for c := 0; c < 32; c += 16 {
			s := archsimd.LoadUint8x16Array((*[16]uint8)(sw[c : c+16]))
			p := (*[23]uint8)(rw[c : c+23])
			q := archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))
			d := s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s2, h2 = s2.Add(d), h2.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
		}
	}
	// Two folds in uint16 (lane pairs of positions 2j, 2j+1), the exact
	// correction, then the last fold in uint32.
	shl8 := archsimd.BroadcastInt16x8(8)
	sa := s0.ConcatAddPairs(s1).ConcatAddPairs(s2.ConcatAddPairs(s3))
	ha := h0.ConcatAddPairs(h1).ConcatAddPairs(h2.ConcatAddPairs(h3))
	sb := s4.ConcatAddPairs(s5).ConcatAddPairs(s6.ConcatAddPairs(s7))
	hb := h4.ConcatAddPairs(h5).ConcatAddPairs(h6.ConcatAddPairs(h7))
	ta := sa.Sub(ha.Shift(shl8)).Add(ha)
	tb := sb.Sub(hb.Shift(shl8)).Add(hb)
	lo := (*[4]uint32)(acc[0:4])
	hi := (*[4]uint32)(acc[4:8])
	archsimd.LoadUint32x4Array(lo).Add(ta.ExtendLo4ToUint32().ConcatAddPairs(ta.HiToLo().ExtendLo4ToUint32())).StoreArray(lo)
	archsimd.LoadUint32x4Array(hi).Add(tb.ExtendLo4ToUint32().ConcatAddPairs(tb.HiToLo().ExtendLo4ToUint32())).StoreArray(hi)
}

// svtHMEColSums enables svtHMESearchColSum (the MIN-based eight-position
// SAD) on NEON, where the absolute difference is MAX, MIN, SUB.
const svtHMEColSums = true

// svtColTotalsArch is svtColTotals sixteen columns at a time. Each row
// accumulates as wrapping uint16 byte pairs S (even column + 256 * odd
// column) plus the odd columns H, so a row costs one add and one
// shift-add; the even columns' totals are S - 256*H (each total is at most
// 16*255) and the two are zipped back into column order.
func svtColTotalsArch(cw *[svtHMEMaxCols]uint16, ref []byte, s2, rows, span int) {
	if uint(rows) > rowsMax || uint(s2) > rowStrideMax {
		panic(errKernelArgs)
	}
	hiShift := archsimd.BroadcastInt16x8(-8)
	loShift := archsimd.BroadcastInt16x8(8)
	for c := 0; c < span; c += 16 {
		var sv, hv archsimd.Uint16x8
		col := ref[c:]
		for r := 0; r < rows; r++ {
			o := r * s2
			v := archsimd.LoadUint8x16Array((*[16]uint8)(col[o : o+16])).ReshapeToUint16s()
			sv, hv = sv.Add(v), hv.Add(v.Shift(hiShift))
		}
		ev := sv.Sub(hv.Shift(loShift))
		ev.InterleaveLo(hv).StoreArray((*[8]uint16)(cw[c:]))
		ev.InterleaveHi(hv).StoreArray((*[8]uint16)(cw[c+8:]))
	}
}

// svtColTotalsSlideArch is svtColTotalsSlide sixteen columns at a time.
func svtColTotalsSlideArch(cw *[svtHMEMaxCols]uint16, out, in []byte, span int) {
	out = out[:span]
	in = in[:len(out)]
	for c := 0; c <= len(out)-16; c += 16 {
		o := archsimd.LoadUint8x16Array((*[16]uint8)(out[c:]))
		n := archsimd.LoadUint8x16Array((*[16]uint8)(in[c:]))
		lo := archsimd.LoadUint16x8Array((*[8]uint16)(cw[c:]))
		hi := archsimd.LoadUint16x8Array((*[8]uint16)(cw[c+8:]))
		lo = lo.Add(n.ExtendLo8ToUint16()).Sub(o.ExtendLo8ToUint16())
		hi = hi.Add(n.HiToLo().ExtendLo8ToUint16()).Sub(o.HiToLo().ExtendLo8ToUint16())
		lo.StoreArray((*[8]uint16)(cw[c:]))
		hi.StoreArray((*[8]uint16)(cw[c+8:]))
	}
}

// svtHMEColSumRow scores one HME search row: the positions x = 0..w-1 (w a
// multiple of 8, w + bw <= svtHMEMaxCols) of the bw-wide (16 or 32),
// rows-tall block s (stride s2) against ref[x:], eight at a time, from the
// source total srcSum and the row's reference column totals cw. It
// returns the smallest key t<<7 | x (t the SAD; t <= 16*32*255, x < 128): the
// row's minimum SAD and its first position. When t is not nil, t[x/8][k] is
// the SAD of position x+k (svtHMEEightColSumScalar).
//
// Per eight positions the sums of minima accumulate as svtSAD16xhx8's byte
// pairs (one MIN per row and position), and the reference window totals are
// the column totals' sliding sums: with C0 the first eight column totals and
// Cb the eight starting bw columns later, position k's window is
// T + sum(Cb[j] - C0[j], j < k), T the first window, as an exclusive prefix
// sum over int16 lanes (each term is at most 7*16*255 in magnitude). The
// positions loop inside the kernel so a search row is one call.
func svtHMEColSumRow(t *[svtHMEMaxCols / 8][8]uint32, s []byte, s2 int, ref []byte, cw *[svtHMEMaxCols]uint16, bw, rows, w int, srcSum uint32) uint32 {
	if uint(s2) > rowStrideMax || uint(rows) > rowsMax || uint(w) > svtHMEMaxCols || (bw != 16 && bw != 32) || w&7 != 0 || rows < 1 {
		panic(errKernelArgs)
	}
	// The block and the reference footprint ((rows-1)*s2 + bw + w bytes) are
	// checked once, as grids. For the eight positions from x the reference is
	// the (bw+8)-column view gx: each step's 24-byte window (the positions'
	// 16-byte windows at constant offsets, plus one spare byte still inside
	// gx) sits at a constant column of a row the loop bound proves, so a step
	// pays no check. Rows run as their own loop, one per block width.
	if !unsafeutil.GridFit(len(s), 0, s2, bw, rows) || !unsafeutil.GridFit(len(ref), 0, s2, bw+w, rows) {
		panic(errKernelArgs)
	}
	gs := unsafeutil.NewGrid(s, 0, s2, bw, rows)
	gr := unsafeutil.NewGrid(ref, 0, s2, bw+w, rows)
	// Only the shift count stays in a register across the row loop; the
	// reduction's constants load from one table each step (kept live, the
	// register allocator reloaded them inside the row loop).
	hiShift := archsimd.BroadcastInt16x8(-8)
	kc := &svtHMEKeyConsts
	key := ^uint32(0)
	for x := 0; x < w; x += 8 {
		gx := gr.Sub(0, x, bw+8, rows)
		// The first step seeds the accumulators, so no zero vector stays
		// live across the row loop.
		sv := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gs, 0, 0))
		p := unsafeutil.Row[[24]uint8](gx, 0, 0)
		var d archsimd.Uint16x8
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))).ReshapeToUint16s()
		s0, h0 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))).ReshapeToUint16s()
		s1, h1 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))).ReshapeToUint16s()
		s2v, h2 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))).ReshapeToUint16s()
		s3, h3 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))).ReshapeToUint16s()
		s4, h4 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))).ReshapeToUint16s()
		s5, h5 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))).ReshapeToUint16s()
		s6, h6 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))).ReshapeToUint16s()
		s7, h7 := d, d.Shift(hiShift)
		if bw == 16 {
			for r := 1; r < gx.Height(); r++ {
				sv = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gs, r, 0))
				p = unsafeutil.Row[[24]uint8](gx, r, 0)
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))).ReshapeToUint16s()
				s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))).ReshapeToUint16s()
				s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))).ReshapeToUint16s()
				s2v, h2 = s2v.Add(d), h2.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))).ReshapeToUint16s()
				s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))).ReshapeToUint16s()
				s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))).ReshapeToUint16s()
				s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))).ReshapeToUint16s()
				s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))).ReshapeToUint16s()
				s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
			}
		} else {
			for r := 0; r < gx.Height(); r++ {
				if r > 0 {
					sv = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gs, r, 0))
					p = unsafeutil.Row[[24]uint8](gx, r, 0)
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))).ReshapeToUint16s()
					s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))).ReshapeToUint16s()
					s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))).ReshapeToUint16s()
					s2v, h2 = s2v.Add(d), h2.Add(d.Shift(hiShift))
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))).ReshapeToUint16s()
					s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))).ReshapeToUint16s()
					s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))).ReshapeToUint16s()
					s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))).ReshapeToUint16s()
					s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
					d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))).ReshapeToUint16s()
					s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
				}
				sv = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](gs, r, 16))
				p = unsafeutil.Row[[24]uint8](gx, r, 16)
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))).ReshapeToUint16s()
				s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))).ReshapeToUint16s()
				s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))).ReshapeToUint16s()
				s2v, h2 = s2v.Add(d), h2.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))).ReshapeToUint16s()
				s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))).ReshapeToUint16s()
				s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))).ReshapeToUint16s()
				s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))).ReshapeToUint16s()
				s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
				d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))).ReshapeToUint16s()
				s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
			}
		}
		// Minima: two folds in uint16 (each lane at most 2*16*4*255), the
		// exact byte-pair correction S - 255*H as one MLA by 0xFF01, the
		// last fold in uint32.
		m255 := archsimd.LoadUint32x4Array((*[4]uint32)(kc[0:4])).ReshapeToUint16s()
		sa := s0.ConcatAddPairs(s1).ConcatAddPairs(s2v.ConcatAddPairs(s3))
		ha := h0.ConcatAddPairs(h1).ConcatAddPairs(h2.ConcatAddPairs(h3))
		sb := s4.ConcatAddPairs(s5).ConcatAddPairs(s6.ConcatAddPairs(s7))
		hb := h4.ConcatAddPairs(h5).ConcatAddPairs(h6.ConcatAddPairs(h7))
		ta := ha.MulAdd(m255, sa)
		tb := hb.MulAdd(m255, sb)
		mlo := ta.ExtendLo4ToUint32().ConcatAddPairs(ta.HiToLo().ExtendLo4ToUint32())
		mhi := tb.ExtendLo4ToUint32().ConcatAddPairs(tb.HiToLo().ExtendLo4ToUint32())

		// Reference window totals.
		var c0, win, cb archsimd.Uint16x8
		if bw == 32 {
			c := (*[40]uint16)(cw[x : x+40])
			c0 = archsimd.LoadUint16x8Array((*[8]uint16)(c[0:8]))
			win = c0.Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[8:16])))
			win = win.Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[16:24]))).Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[24:32])))
			cb = archsimd.LoadUint16x8Array((*[8]uint16)(c[32:40]))
		} else {
			c := (*[24]uint16)(cw[x : x+24])
			c0 = archsimd.LoadUint16x8Array((*[8]uint16)(c[0:8]))
			win = c0.Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[8:16])))
			cb = archsimd.LoadUint16x8Array((*[8]uint16)(c[16:24]))
		}
		// T in every uint32 lane (column-total lanes at most 4*16*255).
		tv := win.ExtendLo4ToUint32().Add(win.HiToLo().ExtendLo4ToUint32())
		tv = tv.ConcatAddPairs(tv)
		tv = tv.ConcatAddPairs(tv)
		// Exclusive prefix of Cb - C0: lane k gets the sum of lanes j < k.
		zero := archsimd.LoadUint32x4Array((*[4]uint32)(kc[16:20])).ReshapeToUint8s()
		e := cb.Sub(c0).ReshapeToUint8s()
		e = e.ConcatShiftBytesRight(zero, 14)
		e = e.ReshapeToUint16s().Add(e.ConcatShiftBytesRight(zero, 14).ReshapeToUint16s()).ReshapeToUint8s()
		e = e.ReshapeToUint16s().Add(e.ConcatShiftBytesRight(zero, 12).ReshapeToUint16s()).ReshapeToUint8s()
		e = e.ReshapeToUint16s().Add(e.ConcatShiftBytesRight(zero, 8).ReshapeToUint16s()).ReshapeToUint8s()
		ei := e.ReshapeToUint16s().BitsToInt16()
		// t - srcSum (signed: the reference windows minus twice the minima).
		tlo := tv.Add(ei.ExtendLo4ToInt32().ToBits()).Sub(mlo.Add(mlo)).BitsToInt32()
		thi := tv.Add(ei.HiToLo().ExtendLo4ToInt32().ToBits()).Sub(mhi.Add(mhi)).BitsToInt32()
		if t != nil {
			tx := &t[x/8]
			tlo.ToBits().StoreArray((*[4]uint32)(tx[0:4]))
			thi.ToBits().StoreArray((*[4]uint32)(tx[4:8]))
			for k := range tx {
				tx[k] += srcSum
			}
		}
		// The smallest (t - srcSum)<<7 + k, in int32 (|t - srcSum| <=
		// 16*32*255), then srcSum<<7 + x: adding the constants keeps the
		// order, and no srcSum or x vector is live across the row loop.
		k128 := archsimd.LoadUint32x4Array((*[4]uint32)(kc[4:8])).BitsToInt32()
		klo := archsimd.LoadUint32x4Array((*[4]uint32)(kc[8:12])).BitsToInt32()
		khi := archsimd.LoadUint32x4Array((*[4]uint32)(kc[12:16])).BitsToInt32()
		kmin := tlo.MulAdd(k128, klo).Min(thi.MulAdd(k128, khi)).ReduceMin()
		key = min(key, uint32(int(kmin)+int(srcSum)<<7+x))
	}
	return key
}

// svtHMEKeyConsts are svtHMEColSumRow's reduction constants: 0xFF01 in
// every uint16 lane, 128 in every uint32 lane, positions 0..7, zero.
var svtHMEKeyConsts = [20]uint32{
	0xFF01FF01, 0xFF01FF01, 0xFF01FF01, 0xFF01FF01,
	128, 128, 128, 128,
	0, 1, 2, 3, 4, 5, 6, 7,
	0, 0, 0, 0,
}

// svtBlockSumSubArch is svtBlockSumSub for bw a multiple of 16 (rows <= 16):
// byte pairs accumulate as wrapping uint16 S plus high bytes H, corrected
// as S - 255*H per lane (each lane's pair total is at most 2*2*16*255).
func svtBlockSumSubArch(s []byte, s2, bw, rows int) uint32 {
	if uint(rows) > rowsMax || uint(s2) > rowStrideMax || uint(bw) > rowsMax {
		panic(errKernelArgs)
	}
	hiShift := archsimd.BroadcastInt16x8(-8)
	var sa, ha archsimd.Uint16x8
	for r := 0; r < rows; r++ {
		row := s[r*s2 : r*s2+bw]
		for c := 0; c+16 <= len(row); c += 16 {
			v := archsimd.LoadUint8x16Array((*[16]uint8)(row[c : c+16])).ReshapeToUint16s()
			sa, ha = sa.Add(v), ha.Add(v.Shift(hiShift))
		}
	}
	t := sa.Sub(ha.Shift(archsimd.BroadcastInt16x8(8))).Add(ha)
	return t.ExtendLo4ToUint32().Add(t.HiToLo().ExtendLo4ToUint32()).ReduceSum()
}

// memsetBytes sets every byte of dst to v: the memset of
// svt_aom_generate_padding's left and right borders and of the flat DC
// predictions. Go lowers a byte-store loop to one store per sample.
func memsetBytes(dst []byte, v byte) {
	x := archsimd.BroadcastUint8x16(v)
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		x.StoreArray((*[16]uint8)(dst[i : i+16]))
	}
	for ; i < len(dst); i++ {
		dst[i] = v
	}
}

// svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16_neon_dotprod
// with sub_sad (Source/Lib/ASM_NEON_DOTPROD/sad_neon_dotprod.c) over the
// superblock at src[0] / ref[0]: for each 16x16 block (offsets[] order) and
// position k = 0..7 the four 8x8 SUB_SADs [TL TR BL BR], doubled (the
// dotprod kernel's mult = 2), keep each 8x8's first strict minimum (best SAD,
// best MV) and give p_eight_sad16x16[n][k], whose first strict minimum
// updates the 16x16 best.
//
// archsimd has no UABD/UDOT, so a row's MAX-MIN absolute differences
// accumulate as wrapping uint16 byte pairs S and high bytes H. The
// reduction runs on two positions per vector: pairwise adds fold each
// position's S and H to [TL TR BL BR] and pair positions 2j, 2j+1 in one
// uint16x8, and one MLA by 0xFF01 (-255) corrects S - 255*H to the
// undoubled SUB_SADs t (exact: t <= 32*255 = 8160). The 8x8 bests are the
// minimum over the eight positions of the key 8*t + k (<= 65287), which
// orders by SAD then position, so it is the first strict minimum the
// sequential compare-and-select keeps; it replaces the stored best only when
// 2*t is below it. The 16x16 sums (<= 2*4*8160) fold from the t vectors in
// uint16 and widen once.
//
// The rows are grid rows: see the footprint check below.
func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	// The superblock's even rows as grids, checked once: 32 rows of 64
	// source bytes and of 72 reference bytes (the eight positions' slack)
	// at stride s2. Each 16x16 block's eight rows are constant rows of a
	// sub-grid, read at their load sites from one base address, so neither
	// a row nor a position window costs a check.
	if !unsafeutil.GridFit(len(src), 0, s2, 64, 32) || !unsafeutil.GridFit(len(ref), 0, s2, 72, 32) {
		panic(errKernelArgs)
	}
	gs := unsafeutil.NewGrid(src, 0, s2, 64, 32)
	gr := unsafeutil.NewGrid(ref, 0, s2, 72, 32)
	hiShift := archsimd.BroadcastInt16x8(-8)
	m255 := archsimd.BroadcastUint16x8(0xFF01)
	eight := archsimd.BroadcastUint16x8(8)
	two := archsimd.BroadcastUint16x8(2)
	// Lanes 0-3 position 0, lanes 4-7 position 1.
	k01 := archsimd.LoadUint16x8Array(&[8]uint16{0, 0, 0, 0, 1, 1, 1, 1})
	keyShift := archsimd.BroadcastInt32x4(-2)
	keyMask := archsimd.BroadcastUint32x4(^uint32(1))
	posMask := archsimd.BroadcastUint32x4(7)
	mv0 := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY)).ReshapeToUint16s()
	eight32 := archsimd.BroadcastUint32x4(8)
	k03 := archsimd.LoadUint32x4Array(&[4]uint32{0, 1, 2, 3})
	k47 := archsimd.LoadUint32x4Array(&[4]uint32{4, 5, 6, 7})
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		bs := gs.Sub(y16/2, x16, 16, 8)
		br := gr.Sub(y16/2, x16, 24, 8)
		a0, a1 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 0, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 1, 0))
		a2, a3 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 2, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 3, 0))
		a4, a5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 4, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 5, 0))
		a6, a7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 6, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 7, 0))
		var tv [4]archsimd.Uint16x8
		var keyMin archsimd.Uint16x8
		kv := k01
		for j := range 4 {
			c := 2 * j
			q := archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 0, 0)[c : c+16]))
			d := a0.Max(q).Sub(a0.Min(q)).ReshapeToUint16s()
			st0, ht0 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 1, 0)[c : c+16]))
			d = a1.Max(q).Sub(a1.Min(q)).ReshapeToUint16s()
			st0, ht0 = st0.Add(d), ht0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 2, 0)[c : c+16]))
			d = a2.Max(q).Sub(a2.Min(q)).ReshapeToUint16s()
			st0, ht0 = st0.Add(d), ht0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 3, 0)[c : c+16]))
			d = a3.Max(q).Sub(a3.Min(q)).ReshapeToUint16s()
			st0, ht0 = st0.Add(d), ht0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 4, 0)[c : c+16]))
			d = a4.Max(q).Sub(a4.Min(q)).ReshapeToUint16s()
			sb0, hb0 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 5, 0)[c : c+16]))
			d = a5.Max(q).Sub(a5.Min(q)).ReshapeToUint16s()
			sb0, hb0 = sb0.Add(d), hb0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 6, 0)[c : c+16]))
			d = a6.Max(q).Sub(a6.Min(q)).ReshapeToUint16s()
			sb0, hb0 = sb0.Add(d), hb0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 7, 0)[c : c+16]))
			d = a7.Max(q).Sub(a7.Min(q)).ReshapeToUint16s()
			sb0, hb0 = sb0.Add(d), hb0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 0, 0)[c+1 : c+17]))
			d = a0.Max(q).Sub(a0.Min(q)).ReshapeToUint16s()
			st1, ht1 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 1, 0)[c+1 : c+17]))
			d = a1.Max(q).Sub(a1.Min(q)).ReshapeToUint16s()
			st1, ht1 = st1.Add(d), ht1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 2, 0)[c+1 : c+17]))
			d = a2.Max(q).Sub(a2.Min(q)).ReshapeToUint16s()
			st1, ht1 = st1.Add(d), ht1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 3, 0)[c+1 : c+17]))
			d = a3.Max(q).Sub(a3.Min(q)).ReshapeToUint16s()
			st1, ht1 = st1.Add(d), ht1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 4, 0)[c+1 : c+17]))
			d = a4.Max(q).Sub(a4.Min(q)).ReshapeToUint16s()
			sb1, hb1 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 5, 0)[c+1 : c+17]))
			d = a5.Max(q).Sub(a5.Min(q)).ReshapeToUint16s()
			sb1, hb1 = sb1.Add(d), hb1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 6, 0)[c+1 : c+17]))
			d = a6.Max(q).Sub(a6.Min(q)).ReshapeToUint16s()
			sb1, hb1 = sb1.Add(d), hb1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 7, 0)[c+1 : c+17]))
			d = a7.Max(q).Sub(a7.Min(q)).ReshapeToUint16s()
			sb1, hb1 = sb1.Add(d), hb1.Add(d.Shift(hiShift))
			sv := st0.ConcatAddPairs(sb0).ConcatAddPairs(st1.ConcatAddPairs(sb1))
			hv := ht0.ConcatAddPairs(hb0).ConcatAddPairs(ht1.ConcatAddPairs(hb1))
			t := hv.MulAdd(m255, sv)
			tv[j] = t
			key := t.MulAdd(eight, kv)
			if j == 0 {
				keyMin = key
			} else {
				keyMin = keyMin.Min(key)
			}
			kv = kv.Add(two)
		}
		// The 8x8 bests: lanes 0-3 hold the even positions' minimum keys,
		// lanes 4-7 the odd ones'.
		km := keyMin.Min(keyMin.HiToLo()).ExtendLo4ToUint32()
		sad := km.Shift(keyShift).And(keyMask)
		pu8 := 21 + n*4
		bp := (*[4]uint32)(me.bestSAD[pu8 : pu8+4])
		mp := (*[4]uint32)(me.bestMV[pu8 : pu8+4])
		best := archsimd.LoadUint32x4Array(bp)
		lt := sad.Less(best)
		sad.IfElse(lt, best).StoreArray(bp)
		mv := mv0.Add(km.And(posMask).ReshapeToUint16s()).ReshapeToUint32s()
		mv.IfElse(lt, archsimd.LoadUint32x4Array(mp)).StoreArray(mp)
		// The 16x16 sums: [p0 .. p7], undoubled, then doubled in uint16.
		u := tv[0].ConcatAddPairs(tv[1]).ConcatAddPairs(tv[2].ConcatAddPairs(tv[3]))
		u = u.Add(u)
		lo, hi := u.ExtendLo4ToUint32(), u.HiToLo().ExtendLo4ToUint32()
		out := &sad16[n]
		lo.StoreArray((*[4]uint32)(out[0:4]))
		hi.StoreArray((*[4]uint32)(out[4:8]))
		// update_best for the 16x16 PU, its first strict minimum: the
		// minimum key 8*sad + k (<= 8*65280 + 7), so no lane is read back
		// from the stores.
		key := lo.MulAdd(eight32, k03).Min(hi.MulAdd(eight32, k47)).ReduceMin()
		if m := key >> 3; m < me.bestSAD[5+n] {
			me.bestSAD[5+n] = m
			me.bestMV[5+n] = svtPackMV(mvX+int(key&7), mvY)
		}
	}
}

// svtSweepPoint1 is svtAllSAD8x8_16x16Sub for one search position, the
// svt_ext_sad_calculation_8x8_16x16 (sub_sad) remainder of
// open_loop_me_get_search_point_results_block: the four 8x8 SUB_SADs of each
// 16x16 block in one vector with a compare-and-select best update, then the
// 16x16, 32x32 and 64x64 sums. ref needs no slack past the 64x64 block.
// Rows are bounds-checked windows, one per row pair and array.
func svtSweepPoint1(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	n2 := s2 + 16
	hiShift := archsimd.BroadcastInt16x8(-8)
	loShift := archsimd.BroadcastInt16x8(8)
	mv := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY))
	var sad16 [16]uint32
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		off := y16*stride + x16
		var st, ht, sb, hb archsimd.Uint16x8
		for i := range 2 {
			o := off + 2*i*s2
			aw, rw := src[o:o+n2:o+n2], ref[o:o+n2:o+n2]
			p := archsimd.LoadUint8x16Array((*[16]uint8)(aw))
			q := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
			d := p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			p = archsimd.LoadUint8x16Array((*[16]uint8)(aw[s2:]))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(rw[s2:]))
			d = p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
		}
		for i := 2; i < 4; i++ {
			o := off + 2*i*s2
			aw, rw := src[o:o+n2:o+n2], ref[o:o+n2:o+n2]
			p := archsimd.LoadUint8x16Array((*[16]uint8)(aw))
			q := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
			d := p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			p = archsimd.LoadUint8x16Array((*[16]uint8)(aw[s2:]))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(rw[s2:]))
			d = p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
		}
		sv := st.ConcatAddPairs(sb)
		hv := ht.ConcatAddPairs(hb)
		sv = sv.ConcatAddPairs(sv)
		hv = hv.ConcatAddPairs(hv)
		sad := sv.Sub(hv.Shift(loShift)).Add(hv).ExtendLo4ToUint32()
		sad = sad.Add(sad)
		pu8 := 21 + n*4
		bp := (*[4]uint32)(me.bestSAD[pu8 : pu8+4])
		mp := (*[4]uint32)(me.bestMV[pu8 : pu8+4])
		best := archsimd.LoadUint32x4Array(bp)
		bestMV := archsimd.LoadUint32x4Array(mp)
		lt := sad.Less(best)
		sad.IfElse(lt, best).StoreArray(bp)
		mv.IfElse(lt, bestMV).StoreArray(mp)
		s := sad.ReduceSum()
		sad16[n] = s
		me.update(5+n, s, mvX, mvY)
	}
	var sad64 uint32
	for q := range 4 {
		s32 := sad16[q*4] + sad16[q*4+1] + sad16[q*4+2] + sad16[q*4+3]
		me.update(1+q, s32, mvX, mvY)
		sad64 += s32
	}
	me.update(0, sad64, mvX, mvY)
}

// sweepPoint1Arch searches one position: the vector kernel when the 64x64
// windows fit, else the generic path.
func (me *realtimeSBMultiSizeME) sweepPoint1Arch(src []byte, stride int, ref []byte, srcBase, refBase, mvX, mvY int) {
	if srcBase >= 0 && refBase >= 0 && srcBase+63*stride+64 <= len(src) && refBase+63*stride+64 <= len(ref) {
		svtSweepPoint1(me, src[srcBase:], ref[refBase:], stride, mvX, mvY)
		return
	}
	me.sweepSearchPoint1(src, stride, ref, stride, srcBase, refBase, mvX, mvY)
}
