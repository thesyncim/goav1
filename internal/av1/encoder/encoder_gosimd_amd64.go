// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"math/bits"
	"simd/archsimd"
)

// AVX2 forms of the encoder's small arm64-only kernels. VPSADBW sums the
// absolute differences of each 8-byte group into a 64-bit lane, so the SADs
// and byte sums below need no lane-overflow care; every kernel falls back to
// its scalar reference without AVX2.

// encAVX2 gates the kernels in this file.
var encAVX2 = gosimdKernelsSupported()

// lpd1SampledSAD is the row-subsampled SAD the chroma detector uses
// (svt_nxm_sad_kernel called with stride << shift and height >> shift), as
// svt_nxm_sad_kernel_helper_avx2: 16-byte rows directly, 8-byte rows two to a
// vector.
func lpd1SampledSAD(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	if rows <= 0 || w <= 0 {
		return 0
	}
	ss, rs := step*srcStride, step*refStride
	if !encAVX2 || (w != 8 && w&15 != 0) || rows > 8 || w > 32 {
		return lpd1SampledSADPureGo(src, ref, srcOff, refOff, srcStride, refStride, w, rows, step)
	}
	if uint(ss) > rowStrideMax || uint(rs) > rowStrideMax {
		panic(errKernelArgs)
	}
	src, ref = src[srcOff:], ref[refOff:]
	var acc archsimd.Uint64x2
	if w == 8 {
		// One window per row pair: row 2i at w[:8], row 2i+1 at w[ss:].
		sn, rn := ss+8, rs+8
		for i := 0; i < rows/2; i++ {
			so, ro := 2*i*ss, 2*i*rs
			sw := src[so : so+sn : so+sn]
			rw := ref[ro : ro+rn : ro+rn]
			acc = acc.Add(pack2((*[8]byte)(sw), (*[8]byte)(sw[ss:])).SumOf8AbsDiff(pack2((*[8]byte)(rw), (*[8]byte)(rw[rs:]))))
		}
		total := sumU64(acc)
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
			acc = acc.Add(archsimd.LoadUint8x16Array((*[16]uint8)(s[c : c+16])).SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(d[c : c+16]))))
		}
	}
	return sumU64(acc)
}

// memsetBytes sets every byte of dst to v: the memset of
// svt_aom_generate_padding's left and right borders and of the flat DC
// predictions. Go lowers a byte-store loop to one store per sample.
func memsetBytes(dst []byte, v byte) {
	i := 0
	if encAVX2 && len(dst) >= 16 {
		x := archsimd.BroadcastUint8x16(v)
		for ; i+16 <= len(dst); i += 16 {
			x.StoreArray((*[16]uint8)(dst[i : i+16]))
		}
	}
	for ; i < len(dst); i++ {
		dst[i] = v
	}
}

// cdefDist8 is SVT-AV1's svt_aom_compute_cdef_dist_8bit_avx2 for the 8x8
// luma blocks: rows i and i+subsampling pack into one 16-byte vector, widen to
// sixteen int16 lanes, and VPMADDWD squares and pair-sums the differences
// into int32 lanes. Other block shapes take the scalar reference.
func cdefDist8(src []byte, sstride int, filtered []byte, fstride int, dlist []cdef.BlockPosition, bwLog2, bhLog2 int, subsampling int) uint64 {
	if !encAVX2 || bwLog2 != 3 || bhLog2 != 3 || (subsampling != 1 && subsampling != 2 && subsampling != 4) {
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
	var acc archsimd.Int32x8
	var total uint64
	for n, b := range dlist {
		so := int(b.BY)*8*sstride + int(b.BX)*8
		fo := int(b.BY)*8*fstride + int(b.BX)*8
		for k := 0; k < pairs; k++ {
			o, q := so+2*k*sd, fo+2*k*fd
			sw := src[o : o+sn : o+sn]
			fw := filtered[q : q+fn : q+fn]
			s := pack2((*[8]byte)(sw), (*[8]byte)(sw[sd:])).ExtendToUint16().AsInt16x16()
			f := pack2((*[8]byte)(fw), (*[8]byte)(fw[fd:])).ExtendToUint16().AsInt16x16()
			d := s.Sub(f)
			acc = acc.Add(d.DotProductPairs(d))
		}
		// An int32 lane gains at most 4*2*65025 a block: flushing every 128
		// blocks keeps the eight-lane total below 2^31.
		if n&127 == 127 {
			total += uint64(uint32(reduceInt32x8(acc)))
			acc = archsimd.Int32x8{}
		}
	}
	total += uint64(uint32(reduceInt32x8(acc)))
	archsimd.ClearAVXUpperBits()
	return total
}

// gosimd_ops_amd64.go: see gosimd_ops_arm64.go for the shared contract.

// Byte-shuffle indices. A negative index selects zero.
var (
	hiBytesIdx   = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	hi16WordsIdx = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	oddDwordsIdx = [16]int8{4, 5, 6, 7, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1}
)

// widenHi8 zero-extends bytes 8..15 of v to 16-bit lanes.
func widenHi8(v archsimd.Uint8x16) archsimd.Uint16x8 {
	return v.PermuteOrZero(archsimd.LoadInt8x16Array(&hiBytesIdx)).ExtendLo8ToUint16()
}

// hiInt16 moves 16-bit lanes 4..7 of v into lanes 0..3 (upper lanes zero).
func hiInt16(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&hi16WordsIdx)).
		ReshapeToUint16s().BitsToInt16()
}

// sqSumInt32 adds the exact squares of v's four int32 lanes to acc. Even lanes
// widen directly; odd lanes are first moved into the even positions.
func sqSumInt32(acc archsimd.Int64x2, v archsimd.Int32x4) archsimd.Int64x2 {
	odd := v.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&oddDwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
	return acc.Add(v.MulWidenEven(v)).Add(odd.MulWidenEven(odd))
}

// mulSumInt32 adds the exact products a[i]*b[i] of the four int32 lane pairs
// to acc.
func mulSumInt32(acc archsimd.Int64x2, a, b archsimd.Int32x4) archsimd.Int64x2 {
	oddA := a.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&oddDwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
	oddB := b.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&oddDwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
	return acc.Add(a.MulWidenEven(b)).Add(oddA.MulWidenEven(oddB))
}

// packU16x16Idx keeps the low byte of each 16-bit lane in the low eight bytes
// of its 128-bit lane; packU16x16Lanes then joins the two lanes' halves.
var (
	packU16x16Idx   = [32]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
	packU16x16Lanes = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}
)

// packU16x16ToU8 narrows sixteen 16-bit lanes holding values 0..255 to bytes
// in lane order (VPSHUFB, then VPERMD across the 128-bit halves): amd64
// archsimd has no VPACKUSWB.
func packU16x16ToU8(v archsimd.Uint16x16) archsimd.Uint8x16 {
	b := v.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&packU16x16Idx))
	return b.AsUint32x8().Permute(archsimd.LoadUint32x8Array(&packU16x16Lanes)).AsUint8x32().GetLo()
}

// gosimdKernelsSupported reports whether the Go-native SIMD kernels can run on
// this CPU. The amd64 kernels use AVX2-width archsimd operations.
func gosimdKernelsSupported() bool { return archsimd.X86.AVX2() }

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

func init() {
	if gosimdKernelsSupported() {
		pred8Impl = pred8AVX2
		pred8FullPelOnly = true
	}
}

// pred8AVX2 is the full-pel case of pred8SIMD (pred8_gosimd_arm64.go) on
// AVX2: svt_av1_convolve_2d_copy_sr_avx2 for predictions and VPSADBW for
// motion-search SADs. Sub-pel blocks stay on the motion package's dav1d
// put_8tap kernels (pred8FullPelOnly): an AVX2 port of the encoder's
// halved-tap sub-pel kernels measured 4-30x slower than those on x86.
// It clears the upper YMM state before returning.
func pred8AVX2(dst []byte, dstStride int, src []byte, srcStride int,
	ref []byte, refOff, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int {
	sad := pred8CopyAVX2(dst, dstStride, src, srcStride, ref[refOff:], refStride, w, h)
	archsimd.ClearAVXUpperBits()
	return sad
}

func pred8CopyAVX2(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refStride, w, h int) int {
	if src == nil {
		pred8CopyRowsAVX2(dst, dstStride, ref, refStride, w, h)
		return 0
	}
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	var acc archsimd.Uint64x2
	for y := 0; y < h; y++ {
		r, s := ref[y*refStride:y*refStride+w], src[y*srcStride:y*srcStride+w]
		s = s[:len(r)]
		for x := 0; x+16 <= len(r); x += 16 {
			acc = acc.Add(archsimd.LoadUint8x16Array((*[16]uint8)(r[x : x+16])).SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(s[x : x+16]))))
		}
	}
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// pred8CopyRowsAVX2 is svt_av1_convolve_2d_copy_sr_avx2: one loop per block
// width, each taking one bounds-checked window per row pair and array (row
// 2i at w[:n], row 2i+1 at w[stride:]).
func pred8CopyRowsAVX2(dst []byte, dstStride int, ref []byte, refStride, w, h int) {
	if uint(dstStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	switch w {
	case 8:
		dn, rn := dstStride+8, refStride+8
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			*(*[8]byte)(dw) = *(*[8]byte)(rw)
			*(*[8]byte)(dw[dstStride:]) = *(*[8]byte)(rw[refStride:])
		}
	case 16:
		dn, rn := dstStride+16, refStride+16
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			archsimd.LoadUint8x16Array((*[16]uint8)(rw)).StoreArray((*[16]uint8)(dw))
			archsimd.LoadUint8x16Array((*[16]uint8)(rw[refStride:])).StoreArray((*[16]uint8)(dw[dstStride:]))
		}
	case 32:
		dn, rn := dstStride+32, refStride+32
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			archsimd.LoadUint8x32Array((*[32]uint8)(rw)).StoreArray((*[32]uint8)(dw))
			archsimd.LoadUint8x32Array((*[32]uint8)(rw[refStride:])).StoreArray((*[32]uint8)(dw[dstStride:]))
		}
	case 64:
		dn, rn := dstStride+64, refStride+64
		for i := 0; i < h/2; i++ {
			do, ro := 2*i*dstStride, 2*i*refStride
			dw, rw := dst[do:do+dn:do+dn], ref[ro:ro+rn:ro+rn]
			r0, d0 := (*[64]uint8)(rw), (*[64]uint8)(dw)
			r1, d1 := (*[64]uint8)(rw[refStride:]), (*[64]uint8)(dw[dstStride:])
			archsimd.LoadUint8x32Array((*[32]uint8)(r0[0:32])).StoreArray((*[32]uint8)(d0[0:32]))
			archsimd.LoadUint8x32Array((*[32]uint8)(r0[32:64])).StoreArray((*[32]uint8)(d0[32:64]))
			archsimd.LoadUint8x32Array((*[32]uint8)(r1[0:32])).StoreArray((*[32]uint8)(d1[0:32]))
			archsimd.LoadUint8x32Array((*[32]uint8)(r1[32:64])).StoreArray((*[32]uint8)(d1[32:64]))
		}
	default:
		for y := range h {
			d, r := dst[y*dstStride:y*dstStride+w], ref[y*refStride:y*refStride+w]
			d = d[:len(r)]
			x := 0
			for ; x+16 <= len(r); x += 16 {
				archsimd.LoadUint8x16Array((*[16]uint8)(r[x : x+16])).StoreArray((*[16]uint8)(d[x : x+16]))
			}
			if x+8 <= len(r) {
				*(*[8]byte)(d[x : x+8]) = *(*[8]byte)(r[x : x+8])
			}
		}
		return
	}
	if h&1 != 0 {
		// The block heights are even; an odd one finishes its last row.
		y := h - 1
		copy(dst[y*dstStride:y*dstStride+w], ref[y*refStride:y*refStride+w])
	}
}

func encShrU32x4(v archsimd.Uint32x4, _ archsimd.Int32x4, shift uint64) archsimd.Uint32x4 {
	return v.ShiftAllRight(shift)
}

func encSarI32x4(v archsimd.Int32x4, _ archsimd.Int32x4, shift uint64) archsimd.Int32x4 {
	return v.ShiftAllRight(shift)
}

func init() {
	// Keep the existing amd64 vector kernels until the separate-input fusion
	// has native generated-code and timing coverage on that architecture.
	if gosimdKernelsSupported() {
		addRawResidualToPredReconBlockImpl = addRawResidualToPredReconBlockCopied
	}
}

// sadSIMDBound reports whether init binds the SAD dispatch variables to the
// AVX2 Go-native-SIMD kernels. It gates on archsimd's own feature report rather
// than cpu.Detected.AVX2, which Rosetta reports as false although AVX2 executes.
// Without AVX2 the portable Go references stay in place, so the lowercase
// wrappers in sad_dispatch_default.go keep their scalar behavior.
var sadSIMDBound = archsimd.X86.AVX2()

// init binds the SAD dispatch variables to the AVX2 Go-native-SIMD kernels.
func init() {
	if sadSIMDBound {
		bindSIMDSAD()
	}
}

// le64 is the little-endian uint64 of the eight bytes of b; the compiler
// combines the byte loads into one 8-byte load.
func le64(b *[8]byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// pack2 packs the 8-byte rows a (lanes 0..7) and b (lanes 8..15) into one
// vector.
func pack2(a, b *[8]byte) archsimd.Uint8x16 {
	return archsimd.BroadcastUint64x2(le64(a)).SetElem(1, le64(b)).ReshapeToUint8s()
}

// sumU64 returns the exact sum of the two 64-bit lanes of acc.
func sumU64(acc archsimd.Uint64x2) int {
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// sadAcc16 adds the SAD of the 16-byte rows s and r to acc.
func sadAcc16(acc archsimd.Uint64x2, s, r *[16]uint8) archsimd.Uint64x2 {
	return acc.Add(archsimd.LoadUint8x16Array(s).SumOf8AbsDiff(archsimd.LoadUint8x16Array(r)))
}

// sad8x8Rows is the 8x8 SAD with independent strides.
func sad8x8Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+8, refStride+8
	acc := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[refStride:]))
		acc = acc.Add(s.SumOf8AbsDiff(r))
	}
	return sumU64(acc)
}

// sad8x8Same is the 8x8 SAD of two blocks sharing one stride.
func sad8x8Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	acc := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[stride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[stride:]))
		acc = acc.Add(s.SumOf8AbsDiff(r))
	}
	return sumU64(acc)
}

// sad8x8x4Rows computes four 8x8 SADs of one source block against four
// reference origins sharing one stride. Each source row pair is packed once
// and reused across all four references.
func sad8x8x4Rows(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	c0 := archsimd.BroadcastUint64x2(0)
	c1 := archsimd.BroadcastUint64x2(0)
	c2 := archsimd.BroadcastUint64x2(0)
	c3 := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		w := src[o : o+n : o+n]
		s := pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))
		w = ref0[o : o+n : o+n]
		c0 = c0.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
		w = ref1[o : o+n : o+n]
		c1 = c1.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
		w = ref2[o : o+n : o+n]
		c2 = c2.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
		w = ref3[o : o+n : o+n]
		c3 = c3.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
	}
	return sumU64(c0), sumU64(c1), sumU64(c2), sumU64(c3)
}

// sad8x8x4Step4Rows is sad8x8x4Rows for the reference origins ref+0, ref+4,
// ref+8 and ref+12: one 20-byte reference window per row (two per row-pair
// window) holds all four candidates at constant offsets.
func sad8x8x4Step4Rows(src, ref []byte, stride int) (int, int, int, int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := stride+8, stride+20
	c0 := archsimd.BroadcastUint64x2(0)
	c1 := archsimd.BroadcastUint64x2(0)
	c2 := archsimd.BroadcastUint64x2(0)
	c3 := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		w := src[o : o+sn : o+sn]
		s := pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))
		rw := ref[o : o+rn : o+rn]
		r0, r1 := (*[20]byte)(rw), (*[20]byte)(rw[stride:])
		c0 = c0.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(r0[0:8]), (*[8]byte)(r1[0:8]))))
		c1 = c1.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(r0[4:12]), (*[8]byte)(r1[4:12]))))
		c2 = c2.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(r0[8:16]), (*[8]byte)(r1[8:16]))))
		c3 = c3.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(r0[12:20]), (*[8]byte)(r1[12:20]))))
	}
	return sumU64(c0), sumU64(c1), sumU64(c2), sumU64(c3)
}

// sad8x8CompoundRows computes SAD(src, round((ref0+ref1)/2)) over one 8x8
// block. PAVGB (Average) is exactly (a+b+1)>>1 per byte.
func sad8x8CompoundRows(src []byte, srcStride int, ref0 []byte, s0 int, ref1 []byte, s1 int) int {
	if uint(srcStride) > rowStrideMax || uint(s0) > rowStrideMax || uint(s1) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, n0, n1 := srcStride+8, s0+8, s1+8
	acc := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		so, o0, o1 := 2*i*srcStride, 2*i*s0, 2*i*s1
		sw := src[so : so+sn : so+sn]
		w0 := ref0[o0 : o0+n0 : o0+n0]
		w1 := ref1[o1 : o1+n1 : o1+n1]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		pred := pack2((*[8]byte)(w0), (*[8]byte)(w0[s0:])).Average(pack2((*[8]byte)(w1), (*[8]byte)(w1[s1:])))
		acc = acc.Add(s.SumOf8AbsDiff(pred))
	}
	return sumU64(acc)
}

// sad16ColsDual sums the SAD of one 16-column strip over h rows with
// independent strides. h is even (16, 32, or 64 in every caller).
func sad16ColsDual(src []byte, srcStride int, ref []byte, refStride int, h int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+16, refStride+16
	even := archsimd.BroadcastUint64x2(0)
	odd := archsimd.BroadcastUint64x2(0)
	for i := 0; i < h/2; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		even = sadAcc16(even, (*[16]uint8)(sw), (*[16]uint8)(rw))
		odd = sadAcc16(odd, (*[16]uint8)(sw[srcStride:]), (*[16]uint8)(rw[refStride:]))
	}
	return sumU64(even.Add(odd))
}

// sad16x16Same is the 16x16 SAD of two blocks sharing one stride.
func sad16x16Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 16
	even := archsimd.BroadcastUint64x2(0)
	odd := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		even = sadAcc16(even, (*[16]uint8)(sw), (*[16]uint8)(rw))
		odd = sadAcc16(odd, (*[16]uint8)(sw[stride:]), (*[16]uint8)(rw[stride:]))
	}
	return sumU64(even.Add(odd))
}

// sad16x16Packed is the 16x16 SAD against a packed 16-stride reference
// block: the reference rows sit at constant offsets in one array.
func sad16x16Packed(src []byte, srcStride int, ref *[256]uint8) int {
	if uint(srcStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := srcStride + 16
	even := archsimd.BroadcastUint64x2(0)
	odd := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * srcStride
		sw := src[o : o+n : o+n]
		rw := (*[32]uint8)(ref[32*i : 32*i+32])
		even = sadAcc16(even, (*[16]uint8)(sw), (*[16]uint8)(rw[0:16]))
		odd = sadAcc16(odd, (*[16]uint8)(sw[srcStride:]), (*[16]uint8)(rw[16:32]))
	}
	return sumU64(even.Add(odd))
}

// sad32Rows is the 32x32 SAD with independent strides, one 32-byte window
// per row pair and array.
func sad32Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+32, refStride+32
	var a0, a1 archsimd.Uint64x2
	for i := 0; i < 16; i++ {
		so, oo := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[oo : oo+rn : oo+rn]
		s0, r0 := (*[32]uint8)(sw), (*[32]uint8)(rw)
		s1, r1 := (*[32]uint8)(sw[srcStride:]), (*[32]uint8)(rw[refStride:])
		a0 = sadAcc16(a0, (*[16]uint8)(s0[0:16]), (*[16]uint8)(r0[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s0[16:32]), (*[16]uint8)(r0[16:32]))
		a0 = sadAcc16(a0, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
	}
	return sumU64(a0.Add(a1))
}

// sad64Rows is the 64x64 SAD with independent strides, one 64-byte window
// per row pair and array.
func sad64Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+64, refStride+64
	var a0, a1, a2, a3 archsimd.Uint64x2
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
		a0 = sadAcc16(a0, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
		a2 = sadAcc16(a2, (*[16]uint8)(s1[32:48]), (*[16]uint8)(r1[32:48]))
		a3 = sadAcc16(a3, (*[16]uint8)(s1[48:64]), (*[16]uint8)(r1[48:64]))
	}
	return sumU64(a0.Add(a1).Add(a2.Add(a3)))
}

// sad16ColsX4 is sad16ColsDual for four references sharing one stride,
// as bounds-checked row windows. With stride and h checked against
// rowStrideMax and rowsMax, each row window is one bounds check.
func sad16ColsX4(src, ref0, ref1, ref2, ref3 []byte, stride int, h int) (int, int, int, int) {
	if uint(stride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	a0 := archsimd.BroadcastUint64x2(0)
	a1 := archsimd.BroadcastUint64x2(0)
	a2 := archsimd.BroadcastUint64x2(0)
	a3 := archsimd.BroadcastUint64x2(0)
	for row := 0; row < h; row++ {
		o := row * stride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[o : o+16]))
		a0 = a0.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref0[o : o+16]))))
		a1 = a1.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref1[o : o+16]))))
		a2 = a2.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref2[o : o+16]))))
		a3 = a3.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref3[o : o+16]))))
	}
	return sumU64(a0), sumU64(a1), sumU64(a2), sumU64(a3)
}

// boxInterpolateRow8 is InterpolateRow_C for 8-bit samples on AVX2, as
// libyuv's InterpolateRow_AVX2: (s0*(256-f) + s1*f + 128) >> 8 in 16-bit
// lanes (at most 255*256 + 128, so the unsigned lane never wraps), and the
// f == 128 HalfRow as a rounding average.
func boxInterpolateRow8(dst, s0, s1 []uint8, f int) {
	n := len(s0)
	s1, dst = s1[:n], dst[:n]
	x := 0
	if encAVX2 && n >= 16 {
		if f == 128 {
			for ; x <= n-16; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(s0[x:]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(s1[x:]))
				a.Average(b).StoreArray((*[16]uint8)(dst[x:]))
			}
		} else {
			f0 := archsimd.BroadcastUint16x16(uint16(256 - f))
			f1 := archsimd.BroadcastUint16x16(uint16(f))
			bias := archsimd.BroadcastUint16x16(128)
			for ; x <= n-16; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(s0[x:])).ExtendToUint16()
				b := archsimd.LoadUint8x16Array((*[16]uint8)(s1[x:])).ExtendToUint16()
				v := a.Mul(f0).Add(b.Mul(f1)).Add(bias).ShiftAllRight(8)
				packU16x16ToU8(v).StoreArray((*[16]uint8)(dst[x:]))
			}
			archsimd.ClearAVXUpperBits()
		}
	}
	if x < n {
		boxInterpolateRow(dst[x:], s0[x:], s1[x:], f, 128)
	}
}

// boxColsCache and boxColsPlan are the arm64 SIMD column filter's table; amd64
// runs boxFilterCols.
type boxColsCache struct{}

type boxColsPlan struct{}

func (*boxColsCache) plan(sw, dw, x, dx int, rows int) *boxColsPlan { return nil }

func (*boxColsPlan) filter(dst, s []uint8) {}

// svtSubpelAVX2 gates the AVX2 sub-pel variance kernel.
var svtSubpelAVX2 = gosimdKernelsSupported()

// svtSubpelVarianceArch is vfp->svf on AVX2, shaped like SVT-AV1's
// Source/Lib/ASM_AVX2/variance_avx2.c svt_aom_sub_pixel_variance{16,32,64}xh
// (filter_block_2rows / the half-pel _mm256_avg_epu8 branch): a zero phase
// skips its pass, the half-pel phase (taps 64/64) is a rounding average
// ((a+b+1)>>1 == (64a+64b+64)>>7), and the other phases multiply interleaved
// sample pairs by the packed tap pair with VPMADDUBSW and round by 7. Both
// passes keep 8-bit values, as the C reference does; they live here in int16
// lanes (sixteen columns of one row per vector, VPMOVZXBW), which is what the
// variance accumulation (variance_kernel_avx2: VPMADDWD squares, int16 sums)
// wants anyway. An 8-wide block runs the 16-column strip with the upper
// eight differences masked to zero.
func svtSubpelVarianceArch(scratch *svtSubpelScratch, ref []byte, refStride int, src []byte, srcStride, bw, bh, xoff, yoff int) uint32 {
	if !svtSubpelAVX2 || bw&7 != 0 || bw > 64 || bh <= 0 || bh > 64 || len(ref) < bh*refStride+bw+16 || len(src) < (bh-1)*srcStride+max(bw, 16) {
		svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
		_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, bw, srcStride, 0, 0, bw, bh)
		return v
	}
	tx := svtBilTapsAVX2(xoff)
	ty := svtBilTapsAVX2(yoff)
	bias := archsimd.BroadcastInt16x16(64)
	ones := archsimd.BroadcastInt16x16(1)
	mask := archsimd.BroadcastInt16x16(-1)
	if bw == 8 {
		mask = archsimd.LoadInt16x16Array(&svtSubpelLo8Mask)
	}
	if uint(refStride) > rowStrideMax || uint(srcStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	// Each row is one bounds-checked window per array.
	var sse, sum32 archsimd.Int32x8
	for x := 0; x < bw; x += 16 {
		prev := svtBilRowHAVX2((*[17]uint8)(ref[x:x+17]), xoff, tx, bias)
		var sum16 archsimd.Int16x16
		for r := 0; r < bh; r++ {
			po := (r+1)*refStride + x
			cur := svtBilRowHAVX2((*[17]uint8)(ref[po:po+17]), xoff, tx, bias)
			var out archsimd.Int16x16
			switch yoff {
			case 0:
				out = prev
			case 4:
				out = prev.AsUint16x16().Average(cur.AsUint16x16()).AsInt16x16()
			default:
				out = svtBilPairAVX2(prev, cur, ty, bias)
			}
			prev = cur
			so := r*srcStride + x
			s := archsimd.LoadUint8x16Array((*[16]uint8)(src[so : so+16])).ExtendToUint16().AsInt16x16()
			d := out.Sub(s).And(mask)
			sum16 = sum16.Add(d)
			sse = sse.Add(d.DotProductPairs(d))
		}
		// Each int16 lane holds at most 64 differences of magnitude <= 255.
		sum32 = sum32.Add(sum16.DotProductPairs(ones))
	}
	total := uint32(reduceInt32x8(sse))
	signed := reduceInt32x8(sum32)
	archsimd.ClearAVXUpperBits()
	return varianceFromStats(total, signed, uint(bits.TrailingZeros(uint(bw*bh))))
}

// svtSubpelLo8Mask keeps the eight low int16 lanes (an 8-wide block's columns).
var svtSubpelLo8Mask = [16]int16{-1, -1, -1, -1, -1, -1, -1, -1}

// svtBilTapsAVX2 is the VPMADDUBSW operand of a phase: the tap pair (f0, f1)
// as int8 bytes in every 16-bit lane. The non-zero, non-half phases have both
// taps in 16..112, so they fit int8 and no product sum saturates
// (255*128 < 2^15).
func svtBilTapsAVX2(off int) archsimd.Int8x32 {
	t := svtBilinearFilters2T[off&7]
	return archsimd.BroadcastInt16x16(int16(t[0] | t[1]<<8)).AsInt8x32()
}

// svtBilPairAVX2 is (a*f0 + b*f1 + 64) >> 7 on sixteen int16 lanes holding
// 8-bit values: the byte pairs (a, b) interleave as a | b<<8 per lane.
func svtBilPairAVX2(a, b archsimd.Int16x16, taps archsimd.Int8x32, bias archsimd.Int16x16) archsimd.Int16x16 {
	pairs := a.Or(b.ShiftAllLeft(8)).AsUint8x32()
	return pairs.DotProductPairsSaturated(taps).Add(bias).ShiftAllRight(7)
}

// svtBilRowHAVX2 is the first (horizontal) pass over the sixteen samples at
// p[0:16] (the taps' second sample at p[1:17]), as int16 lanes.
func svtBilRowHAVX2(p *[17]uint8, xoff int, taps archsimd.Int8x32, bias archsimd.Int16x16) archsimd.Int16x16 {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16])).ExtendToUint16()
	switch xoff {
	case 0:
		return a.AsInt16x16()
	case 4:
		return a.Average(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])).ExtendToUint16()).AsInt16x16()
	}
	b := archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])).ExtendToUint16()
	return svtBilPairAVX2(a.AsInt16x16(), b.AsInt16x16(), taps, bias)
}

// svtDownsample2DArch is svtDownsample2D on AVX2: VPMADDUBSW by ones sums the
// horizontal sample pairs of each row into int16 lanes, the two rows add, and
// (sum + 2) >> 2 packs back to bytes, sixteen outputs per step.
func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	if !encAVX2 {
		svtDownsample2D(dst, dstStride, src, srcStride, w, h)
		return
	}
	ones := archsimd.BroadcastInt8x32(1)
	two := archsimd.BroadcastInt16x16(2)
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
				a := archsimd.LoadUint8x32Array((*[32]uint8)(r0[j : j+32])).DotProductPairsSaturated(ones)
				b := archsimd.LoadUint8x32Array((*[32]uint8)(r1[j : j+32])).DotProductPairsSaturated(ones)
				x = j >> 1
				packU16x16ToU8(a.Add(b).Add(two).ShiftAllRight(2).AsUint16x16()).StoreArray((*[16]uint8)(d[x : x+16]))
			}
			x = w &^ 15
		}
		for ; x < w; x++ {
			s := uint32(r0[2*x]) + uint32(r0[2*x+1]) + uint32(r1[2*x]) + uint32(r1[2*x+1])
			d[x] = uint8((s + 2) >> 2)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// sad16xNx4 returns the SADs of the 16-wide, rows-tall source block against
// reference origins ref+0..ref+3 (svt_sad_loop_kernel's per-row position
// batch), VPSADBW per row and position.
func sad16xNx4(src []byte, srcStride int, ref []byte, refStride, rows int) (int, int, int, int) {
	if !encAVX2 || srcStride != refStride {
		var s [4]int
		for k := range s {
			for r := 0; r < rows; r++ {
				a := src[r*srcStride : r*srcStride+16]
				b := ref[r*refStride+k : r*refStride+k+16]
				for c := range a {
					s[k] += max(int(a[c])-int(b[c]), int(b[c])-int(a[c]))
				}
			}
		}
		return s[0], s[1], s[2], s[3]
	}
	return sad16ColsX4(src, ref, ref[1:], ref[2:], ref[3:], srcStride, rows)
}

// The column-sum HME kernels serve NEON, which has no absolute difference;
// amd64 scores HME positions with VPSADBW (svtSAD16xhx8).
const svtHMEColSums = false

func svtColTotalsArch(cw *[svtHMEMaxCols]uint16, ref []byte, s2, rows, span int) {
	svtColTotals(cw, ref, s2, rows, span)
}

func svtColTotalsSlideArch(cw *[svtHMEMaxCols]uint16, out, in []byte, span int) {
	svtColTotalsSlide(cw, out, in, span)
}

func svtHMEColSumRow(t *[svtHMEMaxCols / 8][8]uint32, s []byte, s2 int, ref []byte, cw *[svtHMEMaxCols]uint16, bw, rows, w int, srcSum uint32) uint32 {
	return svtHMEColSumRowScalar(t, s, s2, ref, cw[:], bw, rows, w, srcSum)
}

func svtBlockSumSubArch(s []byte, s2, bw, rows int) uint32 {
	return svtBlockSumSub(s, s2, bw, rows)
}

// svtSAD16xhx8 is the eight-position step of SVT-AV1's svt_sad_loop_kernel
// for 16/32-wide blocks (Source/Lib/ASM_AVX2/compute_sad_intrin_avx2.c and
// the NEON search_area_width % 8 == 0 branch): adds into acc the SADs of the 16-wide,
// rows-tall source block against ref+0..ref+7 with the source row shared.
// SumOf8AbsDiff is VPSADBW (SSE2/AVX2).
func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Uint64x2
	for r := 0; r < rows; r++ {
		// One window per row: the eight reference loads sit at constant
		// offsets inside it.
		so, ro := r*srcStride, r*refStride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[so : so+16]))
		p := (*[23]uint8)(ref[ro : ro+23])
		a0 = a0.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))))
		a1 = a1.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))))
		a2 = a2.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))))
		a3 = a3.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))))
		a4 = a4.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))))
		a5 = a5.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))))
		a6 = a6.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))))
		a7 = a7.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))))
	}
	acc[0] += uint32(sumU64(a0))
	acc[1] += uint32(sumU64(a1))
	acc[2] += uint32(sumU64(a2))
	acc[3] += uint32(sumU64(a3))
	acc[4] += uint32(sumU64(a4))
	acc[5] += uint32(sumU64(a5))
	acc[6] += uint32(sumU64(a6))
	acc[7] += uint32(sumU64(a7))
}

// svtSAD32xhx8 is the 32-wide eight-position step: the two 16-column halves.
func svtSAD32xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	svtSAD16xhx8(acc, src, srcStride, ref, refStride, rows)
	svtSAD16xhx8(acc, src[16:], srcStride, ref[16:], refStride, rows)
}

// svtEightSAD8x8Sub is svt_ext_all_sad_calculation_8x8_16x16_avx2's SUB_SAD
// pass (Source/Lib/ASM_AVX2/compute_sad_intrin_avx2.c) for one 16x16 block
// at eight positions. archsimd has no VMPSADBW or VPHMINPOSUW, so each
// position is one 256-bit VPSADBW per source row pair, in the
// svt_ext_sad_calculation_8x8_16x16_avx2_intrin shape: rows r and r+8 share
// a vector, the 64-bit lanes accumulate [TL TR BL BR] and VPACKSSDW folds
// them to four 32-bit SADs. The per-8x8 bests are a vector compare and blend
// per position in position order, which keeps the first strict minimum as
// _mm_minpos_epu16 does; sad16[k] is the four quadrants' sum. An 8x8 SUB_SAD
// is at most 2*32*255, so the signed pack is exact. The rows are
// bounds-checked windows taken once: the source's 16 bytes and the
// reference's 23 (the eight positions at constant offsets).
func svtEightSAD8x8Sub(src, ref []byte, stride int, best *[4]uint32, bestMV *[4]uint32, mvX, mvY int, sad16 *[8]uint32) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	s8 := 8 * stride
	a0 := sweepRows16(src, 0, s8)
	a1 := sweepRows16(src, s2, s2+s8)
	a2 := sweepRows16(src, 2*s2, 2*s2+s8)
	a3 := sweepRows16(src, 3*s2, 3*s2+s8)
	w0, w1 := (*[23]uint8)(ref[0:23]), (*[23]uint8)(ref[s8:s8+23])
	w2, w3 := (*[23]uint8)(ref[s2:s2+23]), (*[23]uint8)(ref[s2+s8:s2+s8+23])
	w4, w5 := (*[23]uint8)(ref[2*s2:2*s2+23]), (*[23]uint8)(ref[2*s2+s8:2*s2+s8+23])
	w6, w7 := (*[23]uint8)(ref[3*s2:3*s2+23]), (*[23]uint8)(ref[3*s2+s8:3*s2+s8+23])
	bestV := archsimd.LoadUint32x4Array(best)
	bestMVV := archsimd.LoadUint32x4Array(bestMV)
	for k := range 8 {
		acc := a0.SumOf8AbsDiff(sweepPos16(w0, w1, k)).
			Add(a1.SumOf8AbsDiff(sweepPos16(w2, w3, k))).
			Add(a2.SumOf8AbsDiff(sweepPos16(w4, w5, k))).
			Add(a3.SumOf8AbsDiff(sweepPos16(w6, w7, k)))
		acc = acc.ShiftAllLeft(1)
		v := acc.AsInt32x8()
		sad := v.GetLo().SaturateToInt16Concat(v.GetHi()).AsUint32x4()
		lt := sad.Less(bestV)
		bestV = sad.IfElse(lt, bestV)
		bestMVV = archsimd.BroadcastUint32x4(svtPackMV(mvX+k, mvY)).IfElse(lt, bestMVV)
		h := acc.GetLo().Add(acc.GetHi())
		sad16[k] = uint32(h.GetElem(0) + h.GetElem(1))
	}
	bestV.StoreArray(best)
	bestMVV.StoreArray(bestMV)
	archsimd.ClearAVXUpperBits()
}

// sweepRows16 joins the 16-byte rows of s at a and b into one vector.
func sweepRows16(s []byte, a, b int) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(s[a : a+16]))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(s[b : b+16])))
}

// sweepPos16 joins position k's 16 bytes of the reference row windows w and
// v into one vector.
func sweepPos16(w, v *[23]uint8, k int) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(w[k : k+16]))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(v[k : k+16])))
}

// svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16 (SUB_SAD)
// over the superblock at src[0] / ref[0]: the 16x16 blocks in
// p_eight_sad16x16 order, each scored at eight positions, then the 16x16
// bests.
func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		pu8 := 21 + n*4
		o := y16*stride + x16
		svtEightSAD8x8Sub(src[o:], ref[o:], stride, (*[4]uint32)(me.bestSAD[pu8:pu8+4]),
			(*[4]uint32)(me.bestMV[pu8:pu8+4]), mvX, mvY, &sad16[n])
		for k, s := range sad16[n] {
			me.update(5+n, s, mvX+k, mvY)
		}
	}
}

// svtSweepPoint1 is open_loop_me_get_search_point_results_block for one
// search position with sub_sad: per 16x16 block,
// svt_ext_sad_calculation_8x8_16x16_avx2_intrin (compute_sad_intrin_avx2.c)
// pairs source rows r and r+8 in one 256-bit VPSADBW, so the 64-bit lanes
// accumulate [TL TR BL BR]; VPACKSSDW folds them to four 32-bit SADs for the
// compare-and-blend 8x8 best update, and their sum updates the 16x16 best.
// The 32x32 and 64x64 SADs are the sums of the 16x16s
// (svt_ext_sad_calculation_32x32_64x64). ref needs no slack past the 64x64
// block. An 8x8 SUB_SAD is at most 2*32*255, so the signed pack is exact.
// Rows r and r+8 are one bounds-checked window per array.
func svtSweepPoint1(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	s8 := 8 * stride
	n8 := s8 + 16
	mv := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY))
	var sad16 [16]uint32
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		off := y16*stride + x16
		var acc archsimd.Uint64x4
		for row := range 4 {
			o := off + row*s2
			aw, rw := src[o:o+n8:o+n8], ref[o:o+n8:o+n8]
			p := archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(aw))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(aw[s8:])))
			q := archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(rw))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(rw[s8:])))
			acc = acc.Add(p.SumOf8AbsDiff(q))
		}
		acc = acc.ShiftAllLeft(1)
		v := acc.AsInt32x8()
		sad := v.GetLo().SaturateToInt16Concat(v.GetHi()).AsUint32x4()
		pu8 := 21 + n*4
		bp := (*[4]uint32)(me.bestSAD[pu8 : pu8+4])
		mp := (*[4]uint32)(me.bestMV[pu8 : pu8+4])
		best := archsimd.LoadUint32x4Array(bp)
		lt := sad.Less(best)
		sad.IfElse(lt, best).StoreArray(bp)
		mv.IfElse(lt, archsimd.LoadUint32x4Array(mp)).StoreArray(mp)
		h := acc.GetLo().Add(acc.GetHi())
		s := uint32(h.GetElem(0) + h.GetElem(1))
		sad16[n] = s
		me.update(5+n, s, mvX, mvY)
	}
	archsimd.ClearAVXUpperBits()
	var sad64 uint32
	for q := range 4 {
		s32 := sad16[q*4] + sad16[q*4+1] + sad16[q*4+2] + sad16[q*4+3]
		me.update(1+q, s32, mvX, mvY)
		sad64 += s32
	}
	me.update(0, sad64, mvX, mvY)
}

// sweepPoint1Arch searches one position: the AVX2 kernel when the 64x64
// windows fit, else the generic path.
func (me *realtimeSBMultiSizeME) sweepPoint1Arch(src []byte, stride int, ref []byte, srcBase, refBase, mvX, mvY int) {
	if encAVX2 && srcBase >= 0 && refBase >= 0 && srcBase+63*stride+64 <= len(src) && refBase+63*stride+64 <= len(ref) {
		svtSweepPoint1(me, src[srcBase:], ref[refBase:], stride, mvX, mvY)
		return
	}
	me.sweepSearchPoint1(src, stride, ref, stride, srcBase, refBase, mvX, mvY)
}
