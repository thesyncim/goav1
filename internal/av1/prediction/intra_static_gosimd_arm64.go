// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

// Go-native SIMD PAETH and SMOOTH intra static predictors for arm64. Each
// predictor has three kernel shapes that mirror the shapes the NEON asm covered:
//   - 8-bit, width a multiple of 8 (the common case),
//   - 8-bit, width 4 (two rows packed per 8-lane vector, even height only),
//   - 16-bit (high bit depth), width a multiple of 8.
// The routers at the bottom of the file select the kernel for a block shape and
// fall back to the *PureGo reference for everything else. Every kernel is
// byte-identical to the reference (see intra_static_gosimd_arm64_test.go).
//
// Arithmetic bounds that make the narrow lanes exact:
//   - PAETH: base = top+left-topLeft and the abs diffs fit int16 for 8-bit
//     samples; for 12-bit samples they still fit (|base| <= 2*4095).
//   - SMOOTH_V and SMOOTH_H are convex combinations of two 8-bit samples with
//     weights that sum to 256, so the rounded result (w*a + (256-w)*b + 128) is
//     at most 255*256+128 = 65408 and fits uint16 exactly.
//   - SMOOTH (2D) is V+H where V and H are each such convex combinations, so
//     V+H+256 can exceed 16 bits. As dav1d's ipred_smooth_8bpc_neon, V is
//     bottom*256 + (top-bottom)*w_ver and H is right*256 + (left-right)*w_hor,
//     each one multiply-accumulate exact mod 2^16. V carries a +255 bias
//     (V+255 <= 65535), so the rounded halving add avg(V+255, H) is
//     floor((V+H+256)/2) and its high byte is (V+H+256)>>9 exactly.
//   - 16-bit SMOOTH products reach 4095*255 and need int32 lanes.

import (
	"encoding/binary"
	"simd/archsimd"
)

// storeU16x8 writes the eight uint16 lanes of v to dst[0:16], little-endian:
// one vector store (the lanes are already in memory order on arm64).
func storeU16x8(dst []byte, v archsimd.Int16x8) {
	v.ToBits().ReshapeToUint8s().StoreArray((*[16]byte)(dst[:16]))
}

// paethW4SIMD is the 8-bit kernel for width 4 and even height. Lanes 0..3 hold
// row r and lanes 4..7 hold row r+1, so one vector covers two rows.
func paethW4SIMD(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	height := block.height
	aV := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]}).ConvertToInt16()
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	tl2V := archsimd.BroadcastInt16x8(int16(aboveLeft) << 1)
	for row := 0; row < height; row += 2 {
		lV := archsimd.LoadInt16x8Array(&[8]int16{
			int16(left[row]), int16(left[row]), int16(left[row]), int16(left[row]),
			int16(left[row+1]), int16(left[row+1]), int16(left[row+1]), int16(left[row+1]),
		})
		pTop := lV.Sub(tlV).Abs()
		out := paethInt16(aV, lV, tlV, lV.Sub(tl2V), pTop)
		packed := out.ConvertToUint16().SaturateToUint8().ReshapeToUint32s()
		stride := block.stride
		putU32(block.pix, row*stride, packed.GetElem(0))
		putU32(block.pix, (row+1)*stride, packed.GetElem(1))
	}
}

// smooth8W4SIMD is the 8-bit 2D SMOOTH kernel for width 4 and even height, two
// rows per vector (lanes 0..3 row r, lanes 4..7 row r+1).
func smooth8W4SIMD(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	height := block.height
	tmb := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]}).Sub(archsimd.BroadcastUint16x8(belowPred))
	wWV := archsimd.LoadUint16x8Array(&[8]uint16{weightsW[0], weightsW[1], weightsW[2], weightsW[3], weightsW[0], weightsW[1], weightsW[2], weightsW[3]})
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	for row := 0; row < height; row += 2 {
		wHV := archsimd.LoadUint16x8Array(&[8]uint16{
			weightsH[row], weightsH[row], weightsH[row], weightsH[row],
			weightsH[row+1], weightsH[row+1], weightsH[row+1], weightsH[row+1],
		})
		l0, l1 := left[row]-rightPred, left[row+1]-rightPred
		lmr := archsimd.LoadUint16x8Array(&[8]uint16{l0, l0, l0, l0, l1, l1, l1, l1})
		avg := tmb.MulAdd(wHV, vBase).Average(wWV.MulAdd(lmr, hBase)).ReshapeToUint8s()
		out := avg.ConcatOdd(avg).ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// smoothSplat holds every smooth weight broadcast to eight lanes, so a row's
// vertical weight is one vector load, as dav1d's ld4r/ld2r of weights_ver.
var smoothSplat = func() (t [len(smoothWeights)][8]uint16) {
	for i, w := range smoothWeights {
		t[i] = [8]uint16{w, w, w, w, w, w, w, w}
	}
	return t
}()

// splat8 broadcasts each lane of v to a whole vector through three zip
// levels: the per-row left broadcasts dav1d's ld4r/ld2r take from memory.
func splat8(v archsimd.Uint16x8) (b0, b1, b2, b3, b4, b5, b6, b7 archsimd.Uint16x8) {
	lo, hi := v.InterleaveLo(v), v.InterleaveHi(v)
	q0, q1, q2, q3 := lo.InterleaveLo(lo), lo.InterleaveHi(lo), hi.InterleaveLo(hi), hi.InterleaveHi(hi)
	return q0.InterleaveLo(q0), q0.InterleaveHi(q0), q1.InterleaveLo(q1), q1.InterleaveHi(q1),
		q2.InterleaveLo(q2), q2.InterleaveHi(q2), q3.InterleaveLo(q3), q3.InterleaveHi(q3)
}

// smoothRow is one 8-pixel smooth row before narrowing: the high byte of each
// lane is the predicted sample.
func smoothRow(tmb, wv, vBase, wh, lmr, hBase archsimd.Uint16x8) archsimd.Uint16x8 {
	return tmb.MulAdd(wv, vBase).Average(wh.MulAdd(lmr, hBase))
}

// putRows8 narrows two smoothRow results and stores them as rows off and
// off+stride.
func putRows8(pix []byte, off int, stride int, a0, a1 archsimd.Uint16x8) {
	out := a0.ReshapeToUint8s().ConcatOdd(a1.ReshapeToUint8s()).ReshapeToUint64s()
	binary.LittleEndian.PutUint64(pix[off:], out.GetElem(0))
	binary.LittleEndian.PutUint64(pix[off+stride:], out.GetElem(1))
}

// smoothW8NEON is the 8-bit 2D SMOOTH kernel for width 8: dav1d's
// ipred_smooth w8 loop, eight rows per iteration with the left samples
// broadcast from one vector and the vertical weights loaded pre-broadcast.
// The pre-broadcast weights are those of the block height, so a block
// clipped below its weight table takes the two-row loop.
func smoothW8NEON(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	height := block.height
	stride := block.stride
	tmb := archsimd.LoadUint16x8Array((*[8]uint16)(above)).Sub(archsimd.BroadcastUint16x8(belowPred))
	wh := archsimd.LoadUint16x8Array((*[8]uint16)(weightsW))
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	pix := block.pix
	if height%8 != 0 || len(weightsH) != height {
		weightsH, left = weightsH[:height], left[:height]
		for row := 0; row+1 < height; row += 2 {
			a0 := smoothRow(tmb, archsimd.BroadcastUint16x8(weightsH[row]), vBase, wh, archsimd.BroadcastUint16x8(left[row]-rightPred), hBase)
			a1 := smoothRow(tmb, archsimd.BroadcastUint16x8(weightsH[row+1]), vBase, wh, archsimd.BroadcastUint16x8(left[row+1]-rightPred), hBase)
			putRows8(pix, row*stride, stride, a0, a1)
		}
		return
	}
	rightV := archsimd.BroadcastUint16x8(rightPred)
	splat := smoothSplat[height-4:][:height]
	left = left[:height]
	for row := 0; row+8 <= height; row += 8 {
		l0, l1, l2, l3, l4, l5, l6, l7 := splat8(archsimd.LoadUint16x8Array((*[8]uint16)(left[row:])).Sub(rightV))
		w := splat[row:][:8]
		off := row * stride
		putRows8(pix, off, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[0]), vBase, wh, l0, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[1]), vBase, wh, l1, hBase))
		putRows8(pix, off+2*stride, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[2]), vBase, wh, l2, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[3]), vBase, wh, l3, hBase))
		putRows8(pix, off+4*stride, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[4]), vBase, wh, l4, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[5]), vBase, wh, l5, hBase))
		putRows8(pix, off+6*stride, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[6]), vBase, wh, l6, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[7]), vBase, wh, l7, hBase))
	}
}

// smoothWideNEON is the 8-bit 2D SMOOTH kernel for width a multiple of 16:
// per row, sixteen columns per narrowed store. Unclipped blocks whose height
// is a multiple of 8 run smoothW16NEON per sixteen columns.
func smoothWideNEON(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	width, height := block.width, block.height
	if height%8 == 0 && len(weightsH) == height {
		for c := 0; c+16 <= width; c += 16 {
			smoothW16NEON(block.pix[c:], block.stride, height, weightsW[c:], above[c:], left, belowPred, rightPred)
		}
		return
	}
	blocks := min(width/8, 8)
	bottomV := archsimd.BroadcastUint16x8(belowPred)
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	var tmb, wh [8]archsimd.Uint16x8
	for b := 0; b < blocks; b++ {
		tmb[b] = archsimd.LoadUint16x8Array((*[8]uint16)(above[b*8:])).Sub(bottomV)
		wh[b] = archsimd.LoadUint16x8Array((*[8]uint16)(weightsW[b*8:]))
	}
	weightsH, left = weightsH[:height], left[:height]
	for row := range height {
		smoothWideRow(block.pix[row*block.stride:], &tmb, &wh, blocks, archsimd.BroadcastUint16x8(weightsH[row]), vBase, archsimd.BroadcastUint16x8(left[row]-rightPred), hBase)
	}
}

// smoothW16NEON is the 8-bit 2D SMOOTH kernel for sixteen columns of an
// unclipped block whose height is a multiple of 8, the top and weight vectors
// held in registers; wider blocks run it per sixteen columns.
func smoothW16NEON(pix []byte, stride int, height int, weightsW []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	bottomV := archsimd.BroadcastUint16x8(belowPred)
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	rightV := archsimd.BroadcastUint16x8(rightPred)
	above, weightsW = above[:16], weightsW[:16]
	tmb0 := archsimd.LoadUint16x8Array((*[8]uint16)(above)).Sub(bottomV)
	tmb1 := archsimd.LoadUint16x8Array((*[8]uint16)(above[8:])).Sub(bottomV)
	wh0 := archsimd.LoadUint16x8Array((*[8]uint16)(weightsW))
	wh1 := archsimd.LoadUint16x8Array((*[8]uint16)(weightsW[8:]))
	splat := smoothSplat[height-4:][:height]
	left = left[:height]
	for row := 0; row+8 <= height; row += 8 {
		l0, l1, l2, l3, l4, l5, l6, l7 := splat8(archsimd.LoadUint16x8Array((*[8]uint16)(left[row:])).Sub(rightV))
		w := splat[row:][:8]
		off := row * stride
		smoothRow16(pix[off:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[0]), vBase, wh0, wh1, l0, hBase)
		smoothRow16(pix[off+stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[1]), vBase, wh0, wh1, l1, hBase)
		smoothRow16(pix[off+2*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[2]), vBase, wh0, wh1, l2, hBase)
		smoothRow16(pix[off+3*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[3]), vBase, wh0, wh1, l3, hBase)
		smoothRow16(pix[off+4*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[4]), vBase, wh0, wh1, l4, hBase)
		smoothRow16(pix[off+5*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[5]), vBase, wh0, wh1, l5, hBase)
		smoothRow16(pix[off+6*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[6]), vBase, wh0, wh1, l6, hBase)
		smoothRow16(pix[off+7*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[7]), vBase, wh0, wh1, l7, hBase)
	}
}

// smoothRow16 writes one 16-pixel row of smoothW16NEON.
func smoothRow16(dst []byte, tmb0, tmb1, wv, vBase, wh0, wh1, lmr, hBase archsimd.Uint16x8) {
	a0 := smoothRow(tmb0, wv, vBase, wh0, lmr, hBase)
	a1 := smoothRow(tmb1, wv, vBase, wh1, lmr, hBase)
	a0.ReshapeToUint8s().ConcatOdd(a1.ReshapeToUint8s()).StoreArray((*[16]uint8)(dst))
}

// smoothWideRow writes one row of smoothWideNEON.
func smoothWideRow(dst []byte, tmb, wh *[8]archsimd.Uint16x8, blocks int, wv, vBase, lmr, hBase archsimd.Uint16x8) {
	for b := 0; b+1 < blocks; b += 2 {
		a0 := smoothRow(tmb[b], wv, vBase, wh[b], lmr, hBase)
		a1 := smoothRow(tmb[b+1], wv, vBase, wh[b+1], lmr, hBase)
		a0.ReshapeToUint8s().ConcatOdd(a1.ReshapeToUint8s()).StoreArray((*[16]uint8)(dst[b*8:]))
	}
}

// smoothVerticalW4SIMD is the 8-bit SMOOTH_V kernel for width 4, even height.
func smoothVerticalW4SIMD(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	height := block.height
	aV := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]})
	belowV := archsimd.BroadcastUint16x8(belowPred)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	for row := 0; row < height; row += 2 {
		wV := archsimd.LoadUint16x8Array(&[8]uint16{
			weights[row], weights[row], weights[row], weights[row],
			weights[row+1], weights[row+1], weights[row+1], weights[row+1],
		})
		cb := archsimd.BroadcastUint16x8(smoothScale).Sub(wV).Mul(belowV).Add(roundV)
		out := wV.Mul(aV).Add(cb).Shift(shr8U16).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// smoothHorizontalW4SIMD is the 8-bit SMOOTH_H kernel for width 4, even height.
func smoothHorizontalW4SIMD(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	height := block.height
	wWV := archsimd.LoadUint16x8Array(&[8]uint16{weights[0], weights[1], weights[2], weights[3], weights[0], weights[1], weights[2], weights[3]})
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	cwRV := archsimd.BroadcastUint16x8(smoothScale).Sub(wWV).Mul(archsimd.BroadcastUint16x8(rightPred)).Add(roundV)
	for row := 0; row < height; row += 2 {
		lV := archsimd.LoadUint16x8Array(&[8]uint16{
			left[row], left[row], left[row], left[row],
			left[row+1], left[row+1], left[row+1], left[row+1],
		})
		out := wWV.Mul(lV).Add(cwRV).Shift(shr8U16).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// paeth16SIMD is the 16-bit PAETH kernel for width a multiple of 8. Samples are
// at most 12 bits, so the int16 lanes hold base and the distances exactly.
func paeth16SIMD(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	width, height := block.width, block.height
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		pTop := archsimd.BroadcastInt16x8(absDiffInt16(left[row], aboveLeft))
		lmtV := archsimd.BroadcastInt16x8(int16(left[row]) - int16(aboveLeft)*2)
		dst := block.pix[row*block.stride:][:width*2]
		for col := 0; col < width; col += 8 {
			storeU16x8(dst[col*2:], paethInt16(loadU16x8(above[col:]), lV, tlV, lmtV, pTop))
		}
	}
}

// smooth16Cols holds the per-column 16-bit SMOOTH terms of one 8-column
// block as int32 lanes (low and high four columns): top - bottom, and the
// column weight scaled by 2^7.
type smooth16Cols struct {
	tmb [8][2]archsimd.Int32x4
	ww  [8][2]archsimd.Int32x4
}

// smooth16SIMD is the 16-bit 2D SMOOTH kernel for width a multiple of 8, in
// the factored form of the 8-bit kernels:
//
//	pred = (256*bottom + wH*(top - bottom) + 256*right + wW*(left - right) + 256) >> 9
//
// evaluated as int32 multiply-accumulates (MLA) with the constant terms in
// the accumulator seed and every term scaled by 2^7, so the >> 9 is the high
// halves of the int32 sums (one UZP2 for eight samples). The scaled sum is at
// most 2^7 * (512*4095 + 256) < 2^31 and its partial sums are non-negative
// combinations of the same bound, so the arithmetic is exact.
func smooth16SIMD(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	bottomV := archsimd.BroadcastInt16x8(int16(belowPred))
	seed := archsimd.BroadcastInt32x4((256*int32(belowPred) + 256*int32(rightPred) + 256) << 7)
	var cols smooth16Cols
	for b := 0; b < blocks; b++ {
		d := loadU16x8(above[b*8:]).Sub(bottomV)
		cols.tmb[b] = [2]archsimd.Int32x4{d.ExtendLo4ToInt32(), d.HiToLo().ExtendLo4ToInt32()}
		w := loadU16x8(weightsW[b*8:]).ShiftAllLeft(7)
		cols.ww[b] = [2]archsimd.Int32x4{w.ExtendLo4ToInt32(), w.HiToLo().ExtendLo4ToInt32()}
	}
	weightsH, left = weightsH[:height], left[:height]
	for row := range height {
		wH := archsimd.BroadcastInt32x4(int32(weightsH[row]) << 7)
		lmr := archsimd.BroadcastInt32x4(int32(left[row]) - int32(rightPred))
		dst := block.pix[row*block.stride:][:width*2]
		for b := range cols.tmb[:blocks] {
			lo := cols.ww[b][0].MulAdd(lmr, seed)
			hi := cols.ww[b][1].MulAdd(lmr, seed)
			lo = cols.tmb[b][0].MulAdd(wH, lo)
			hi = cols.tmb[b][1].MulAdd(wH, hi)
			lo.ToBits().ReshapeToUint16s().ConcatOdd(hi.ToBits().ReshapeToUint16s()).ReshapeToUint8s().StoreArray((*[16]byte)(dst[b*16:]))
		}
	}
}

// smoothVertical16SIMD is the 16-bit SMOOTH_V kernel for width a multiple of 8.
func smoothVertical16SIMD(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	width, height := block.width, block.height
	scale32 := archsimd.BroadcastInt32x4(smoothScale)
	belowV := archsimd.BroadcastInt32x4(int32(belowPred))
	round := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	for row := 0; row < height; row++ {
		wH := int16(weights[row])
		wHV := archsimd.BroadcastInt16x8(wH)
		cb := scale32.Sub(archsimd.BroadcastInt32x4(int32(wH))).Mul(belowV).Add(round)
		dst := block.pix[row*block.stride:][:width*2]
		for col := 0; col < width; col += 8 {
			aV := loadU16x8(above[col:])
			loOut := wHV.MulWidenLo(aV).Add(cb).Shift(shr8I32)
			hiOut := wHV.MulWidenLo(aV.HiToLo()).Add(cb).Shift(shr8I32)
			storeU16x8(dst[col*2:], cflTruncateInt32PairToInt16(loOut, hiOut))
		}
	}
}

// smoothHorizontal16SIMD is the 16-bit SMOOTH_H kernel for width a multiple of 8.
func smoothHorizontal16SIMD(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	rightV := archsimd.BroadcastInt32x4(int32(rightPred))
	scale32 := archsimd.BroadcastInt32x4(smoothScale)
	round := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	var wW [8]archsimd.Int16x8
	var cwRLo, cwRHi [8]archsimd.Int32x4
	for b := 0; b < blocks; b++ {
		w := loadU16x8(weights[b*8:])
		wW[b] = w
		cwRLo[b] = scale32.Sub(w.ExtendLo4ToInt32()).Mul(rightV).Add(round)
		cwRHi[b] = scale32.Sub(w.HiToLo().ExtendLo4ToInt32()).Mul(rightV).Add(round)
	}
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		dst := block.pix[row*block.stride:][:width*2]
		for b := 0; b < blocks; b++ {
			loOut := wW[b].MulWidenLo(lV).Add(cwRLo[b]).Shift(shr8I32)
			hiOut := wW[b].HiToLo().MulWidenLo(lV).Add(cwRHi[b]).Shift(shr8I32)
			storeU16x8(dst[b*16:], cflTruncateInt32PairToInt16(loOut, hiOut))
		}
	}
}

// predictPaethSIMD routes a PAETH block to the kernel for its shape; shapes no
// kernel covers use the pure-Go reference.
func predictPaethSIMD(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		paeth8SIMD(block, above, left, aboveLeft)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		paethW4SIMD(block, above, left, aboveLeft)
	case bytesPerSample == 2 && block.width%8 == 0:
		paeth16SIMD(block, above, left, aboveLeft)
	default:
		predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
	}
}

// predictSmoothSIMD routes a 2D SMOOTH block to the kernel for its shape.
func predictSmoothSIMD(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width == 8 && block.height%2 == 0:
		smoothW8NEON(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 1 && block.width%16 == 0 && block.width <= 64:
		smoothWideNEON(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 1 && block.width%8 == 0:
		smooth8SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smooth8W4SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smooth16SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	default:
		predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
	}
}

// predictSmoothVerticalSIMD routes a SMOOTH_V block to the kernel for its shape.
func predictSmoothVerticalSIMD(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		smoothVertical8SIMD(block, weights, above, belowPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smoothVerticalW4SIMD(block, weights, above, belowPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smoothVertical16SIMD(block, weights, above, belowPred)
	default:
		predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
	}
}

// predictSmoothHorizontalSIMD routes a SMOOTH_H block to the kernel for its shape.
func predictSmoothHorizontalSIMD(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		smoothHorizontal8SIMD(block, weights, left, rightPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smoothHorizontalW4SIMD(block, weights, left, rightPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smoothHorizontal16SIMD(block, weights, left, rightPred)
	default:
		predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
	}
}
