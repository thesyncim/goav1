// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD compound (bidirectional) average / distance-weighted blend: the
// final step of compound inter prediction, folding two 16-bit CONV_BUF predictors
// into the 8-bit output.

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// blendMulAddU16Lo computes two unsigned 16x16 products in 32-bit lanes and
// adds them. The input weights are the AV1 distance weights and sum to 16, so
// the largest result is below 2^20 and does not wrap the Uint32 lanes.
func blendMulAddU16Lo(x, xWeight, y, yWeight archsimd.Uint16x8) archsimd.Uint32x4 {
	return x.MulWidenLo(xWeight).Add(y.MulWidenLo(yWeight))
}

// blendMulAddU16Hi does the same for the high four lanes. The public API only
// exposes low-half widening multiply, so move those high lanes down first.
func blendMulAddU16Hi(x, xWeight, y, yWeight archsimd.Uint16x8) archsimd.Uint32x4 {
	return x.HiToLo().MulWidenLo(xWeight.HiToLo()).Add(y.HiToLo().MulWidenLo(yWeight.HiToLo()))
}

// blendShiftNarrowU16 applies the arithmetic >>8 and unsigned-saturating
// int32-to-uint16 conversion used by SQSHRUN. Narrowing keeps its four results
// in the low 64 bits and clears the upper half.
func blendShiftNarrowU16(x archsimd.Int32x4) archsimd.Uint16x8 {
	return x.ShiftAllRight(8).SaturateToUint16()
}

// blendJoinLowU16 places four low uint16 lanes from each narrowing result into
// one vector. InterleaveLo on the uint64 views joins the two low 64-bit halves.
func blendJoinLowU16(lo, hi archsimd.Uint16x8) archsimd.Uint16x8 {
	return lo.ReshapeToUint64s().InterleaveLo(hi.ReshapeToUint64s()).ReshapeToUint16s()
}

// blendJoinLowU8 joins the low eight bytes from each narrowing result.
func blendJoinLowU8(lo, hi archsimd.Uint8x16) archsimd.Uint8x16 {
	return lo.ReshapeToUint64s().InterleaveLo(hi.ReshapeToUint64s()).ReshapeToUint8s()
}

// blendCompoundAvg8GoSIMD is the Go-native SIMD form of blendCompoundAvg8PureGo.
// Per pixel: tmp = src0*fwd + src1*bck; tmp >>= 4 (DIST_PRECISION_BITS);
// tmp -= roundOffset; dst = clip[0,255](roundPowerOfTwo(tmp, roundBits)).
//
// Each 8-lane group is the asm's exact op sequence: UMULL/UMULL2 for the first
// product (fresh destination, no zero-accumulator copy), UMLAL/UMLAL2 to fold in
// the second (unsigned -- CONV_BUF spans the full uint16 range, so a signed SMLAL
// would diverge once the top bit is set), one bias ADD on the UNSHIFTED
// accumulator, then a single SQSHRUN/SQSHRUN2 (>>8) that fuses both rounding
// shifts + the int32->uint16 clamp. The bias folds -roundOffset + roundBias AND
// collapses the two >>4 shifts into one >>8: ((A>>4) - roundOffset + 8) >> 4 ==
// (A - BIAS) >> 8 exactly, because A's dropped low-4 bits (<16) can never cross a
// 256 boundary. BIAS = 16*roundOffset - 128.
//
// The main loop is 16-wide: two independent 8-lane chains (columns 0..7 and
// 8..15) interleave to hide the ~7-deep per-chain latency, and the two results
// pack into one Uint8x16 for a single 16-byte store. Only the 8-bit path
// (roundBits==4) is SIMD-covered; every other shape falls back to the NEON asm.
func blendCompoundAvg8GoSIMD(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	if roundBits != 4 || width < 8 || width%8 != 0 ||
		len(src0) < width*height || len(src1) < width*height {
		blendCompoundAvg8NEON(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	const distBits = 4 // DIST_PRECISION_BITS; roundBits is guaranteed 4 here too.
	fwdV := archsimd.BroadcastUint16x8(uint16(fwdOffset))
	bckV := archsimd.BroadcastUint16x8(uint16(bckOffset))
	// biasV = ((1<<(roundBits-1)) - roundOffset) << distBits == -(16*roundOffset - 128).
	biasV := archsimd.BroadcastInt32x4(int32(((1 << (roundBits - 1)) - roundOffset) << distBits))

	const u16 = 2
	w16 := width &^ 15 // 16-wide-covered columns; the rest (a trailing 8) is the remainder.
	s0base := unsafe.Pointer(&src0[0])
	s1base := unsafe.Pointer(&src1[0])
	dbase := unsafe.Pointer(&dst.Pix[dstY*dst.Stride+dstX])
	for y := 0; y < height; y++ {
		s0p := unsafe.Add(s0base, y*width*u16)
		s1p := unsafe.Add(s1base, y*width*u16)
		dp := unsafe.Add(dbase, y*dst.Stride)
		for x := 0; x < w16; x += 16 {
			s0a := archsimd.LoadUint16x8Array((*[8]uint16)(s0p))
			s0b := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(s0p, 8*u16)))
			s1a := archsimd.LoadUint16x8Array((*[8]uint16)(s1p))
			s1b := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(s1p, 8*u16)))

			accLoA := blendMulAddU16Lo(s0a, fwdV, s1a, bckV) // cols 0..3
			accHiA := blendMulAddU16Hi(s0a, fwdV, s1a, bckV) // cols 4..7
			accLoB := blendMulAddU16Lo(s0b, fwdV, s1b, bckV) // cols 8..11
			accHiB := blendMulAddU16Hi(s0b, fwdV, s1b, bckV) // cols 12..15

			rLoA := accLoA.BitsToInt32().Add(biasV)
			rHiA := accHiA.BitsToInt32().Add(biasV)
			rLoB := accLoB.BitsToInt32().Add(biasV)
			rHiB := accHiB.BitsToInt32().Add(biasV)

			pa := blendJoinLowU16(blendShiftNarrowU16(rLoA), blendShiftNarrowU16(rHiA)) // px 0..7
			pb := blendJoinLowU16(blendShiftNarrowU16(rLoB), blendShiftNarrowU16(rHiB)) // px 8..15
			// Clamp both groups to [0,255] and join their low eight bytes.
			out := blendJoinLowU8(pa.SaturateToUint8(), pb.SaturateToUint8())
			out.StoreArray((*[16]uint8)(dp))

			// Do not form a one-past pointer after the final vector group: checkptr
			// rejects that arithmetic even though the pointer is not dereferenced.
			if x+16 < w16 || width&8 != 0 {
				s0p = unsafe.Add(s0p, 16*u16)
				s1p = unsafe.Add(s1p, 16*u16)
				dp = unsafe.Add(dp, 16)
			}
		}
		if width&8 != 0 { // trailing 8-column group (only width==8 mod 16).
			s0 := archsimd.LoadUint16x8Array((*[8]uint16)(s0p))
			s1 := archsimd.LoadUint16x8Array((*[8]uint16)(s1p))
			accLo := blendMulAddU16Lo(s0, fwdV, s1, bckV)
			accHi := blendMulAddU16Hi(s0, fwdV, s1, bckV)
			rLo := accLo.BitsToInt32().Add(biasV)
			rHi := accHi.BitsToInt32().Add(biasV)
			narrowed := blendJoinLowU16(blendShiftNarrowU16(rLo), blendShiftNarrowU16(rHi))
			out := narrowed.SaturateToUint8()
			convStore8U8(dp, out)
		}
	}
}
