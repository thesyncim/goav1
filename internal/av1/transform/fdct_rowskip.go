// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

// fdct_rowskip.go is the forward DCT_DCT the encoder's skip decision runs:
// it computes only the coefficient rows that can survive quantization.
//
// The 2-D transform is a column pass followed by a row pass over each
// column-pass output row k (one vertical frequency). Every row-pass output of
// row k is a combination of that row's column-pass values T[k][j] whose
// effective weights are cos((2j+1)l*pi/2N) (cos(pi/4) for l=0) — the
// libaom 1-D DCT carries gain sqrt(N/2) over the orthonormal basis, whose
// entries are at most sqrt(2/N) — so they are at most 1 in magnitude up to
// the quantized cospi constants. On top of that comes the network's
// rounding: half_btf rounds to nearest, and propagating that error through
// the rotations and butterflies (gain |w0|+|w1| <= sqrt(2) per rotation,
// error sums at each add) bounds the accumulated error by 1.21 for the
// 8-point network, 2.83 for the 16-point one and 7.05 for the 32-point one.
// With S = sum_j |T[k][j]| that gives
//
//	|C[k][l]| <= S + err < S + S>>6 + slack
//
// for slack = ceil(err) + 1 (the +1 absorbs the floor of S>>6, which also
// covers the constants' quantization noise, below 1/200 here). When that
// bound is below the quantizer's zero bin, every coefficient of the row
// quantizes to zero and its row pass is skipped. TestForwardDCTRowBound pins
// the weights and the rounding error of each network.
//
// The encoder only needs those rows' coefficients for its skip RD terms,
// where an all-zero coefficient adds the same c*c to the skip and to the code
// distortion; the two cancel in the comparison, so the decision is unchanged
// by leaving them out.

const (
	fwdRowSkipSlack8  = 3
	fwdRowSkipSlack16 = 4
	fwdRowSkipSlack32 = 9
)

// fwdRowSkipLive reports whether a row whose column-pass magnitudes sum to s
// may produce a row-pass coefficient at or above thr.
func fwdRowSkipLive(s, thr, slack int32) bool {
	return s+s>>6+slack >= thr
}

// The row-skipping kernels. Builds without one compute the whole transform,
// which is always a valid answer.
var (
	forwardDCT8x8RowSkipImpl   = forwardDCT8x8RowSkipFull
	forwardDCT16x16RowSkipImpl = forwardDCT16x16RowSkipFull
	forwardDCT32x32RowSkipImpl = forwardDCT32x32RowSkipFull
)

// ForwardDCT16x16RowSkip8BitTrusted computes the 16x16 DCT_DCT forward
// transform of an 8-bit residual block (residual stride 16, coefficient
// stride 16, decoder layout coeff[col*16+row]) for a caller that quantizes it
// with zero bins thr0 (the first column-pass row, which holds DC: the smaller
// of the DC and AC zero bins) and thr (every other row). Each group of four
// column-pass rows (bit g covers rows 4g..4g+3) whose coefficients provably
// all fall below their zero bin may be skipped; the result sets bit g for each
// group it computed exactly. Skipped groups read as zero coefficients when
// any group was computed. A zero result means every coefficient quantizes to
// zero and coeff was not written.
func ForwardDCT16x16RowSkip8BitTrusted(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	return forwardDCT16x16RowSkipImpl(coeff, residual, thr0, thr)
}

// ForwardDCT8x8RowSkip8BitTrusted is ForwardDCT16x16RowSkip8BitTrusted for
// 8x8 blocks (strides 8; bits 0 and 1 cover rows 0..3 and 4..7).
func ForwardDCT8x8RowSkip8BitTrusted(coeff *[64]int32, residual *[64]int16, thr0, thr int32) uint8 {
	return forwardDCT8x8RowSkipImpl(coeff, residual, thr0, thr)
}

// ForwardDCT32x32RowSkip8BitTrusted is ForwardDCT16x16RowSkip8BitTrusted for
// 32x32 blocks (strides 32; bit g covers rows 4g..4g+3 of eight groups).
func ForwardDCT32x32RowSkip8BitTrusted(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	return forwardDCT32x32RowSkipImpl(coeff, residual, thr0, thr)
}

func forwardDCT8x8RowSkipFull(coeff *[64]int32, residual *[64]int16, _, _ int32) uint8 {
	forwardDCT8x8Trusted8BitImpl(coeff[:], 8, residual[:], 8)
	return 0x3
}

func forwardDCT16x16RowSkipFull(coeff *[256]int32, residual *[256]int16, _, _ int32) uint8 {
	forwardDCT16x16Trusted8BitImpl(coeff[:], 16, residual[:], 16)
	return 0xf
}

func forwardDCT32x32RowSkipFull(coeff *[1024]int32, residual *[1024]int16, _, _ int32) uint8 {
	forwardDCT32x32Trusted8BitImpl(coeff[:], 32, residual[:], 32)
	return 0xff
}

// fwdRowSkipMask returns the live four-row groups of an n-wide row-major
// column-pass output buf.
func fwdRowSkipMask(buf []int32, n int, thr0, thr, slack int32) uint8 {
	var mask uint8
	for r := range n {
		s := int32(0)
		for _, v := range buf[r*n : r*n+n] {
			s += max(v, -v)
		}
		t := thr
		if r == 0 {
			t = thr0
		}
		if fwdRowSkipLive(s, t, slack) {
			mask |= 1 << (r >> 2)
		}
	}
	return mask
}

// forwardDCT8x8RowSkipPureGo is the scalar statement of the row-skipping 8x8
// kernel (forwardDCT8x8PureGo with the per-group bound).
func forwardDCT8x8RowSkipPureGo(coeff *[64]int32, residual *[64]int16, thr0, thr int32) uint8 {
	var buf [64]int32 // column-pass output, row-major buf[r*8+c]
	var in, out [8]int32
	for c := range 8 {
		for r := range 8 {
			in[r] = int32(residual[r*8+c]) << 2 // shift[0] = 2
		}
		fwdDCT8(&in, &out)
		fwdRoundShift1(out[:]) // shift[1] = -1
		for r := range 8 {
			buf[r*8+c] = out[r]
		}
	}
	mask := fwdRowSkipMask(buf[:], 8, thr0, thr, fwdRowSkipSlack8)
	if mask == 0 {
		return 0
	}
	for r := range 8 {
		if mask&(1<<(r>>2)) == 0 {
			for c := range 8 {
				coeff[c*8+r] = 0
			}
			continue
		}
		copy(in[:], buf[r*8:r*8+8])
		fwdDCT8(&in, &out)
		for c := range 8 { // shift[2] = 0; output[c*rows + r]
			coeff[c*8+r] = out[c]
		}
	}
	return mask
}

// forwardDCT16x16RowSkipPureGo is the scalar statement of the row-skipping
// kernel (forwardDCT16x16PureGo with the per-group bound).
func forwardDCT16x16RowSkipPureGo(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	var buf [256]int32 // column-pass output, row-major buf[r*16+c]
	var in, out [16]int32
	for c := range 16 {
		for r := range 16 {
			in[r] = int32(residual[r*16+c]) << 2 // shift[0] = 2
		}
		fwdDCT16(&in, &out, &fwdCospi13, 13)
		fwdRoundShift2(out[:]) // shift[1] = -2
		for r := range 16 {
			buf[r*16+c] = out[r]
		}
	}
	mask := fwdRowSkipMask(buf[:], 16, thr0, thr, fwdRowSkipSlack16)
	if mask == 0 {
		return 0
	}
	for r := range 16 {
		if mask&(1<<(r>>2)) == 0 {
			for c := range 16 {
				coeff[c*16+r] = 0
			}
			continue
		}
		copy(in[:], buf[r*16:r*16+16])
		fwdDCT16(&in, &out, &fwdCospi12, 12)
		for c := range 16 { // shift[2] = 0; output[c*rows + r]
			coeff[c*16+r] = out[c]
		}
	}
	return mask
}

// forwardDCT32x32RowSkipPureGo is the scalar statement of the row-skipping
// 32x32 kernel (forwardDCT32x32PureGo with the per-group bound).
func forwardDCT32x32RowSkipPureGo(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	var buf [1024]int32 // column-pass output, row-major buf[r*32+c]
	var in, out [32]int32
	for c := range 32 {
		for r := range 32 {
			in[r] = int32(residual[r*32+c]) << 2 // shift[0] = 2
		}
		fwdDCT32(&in, &out, &fwdCospi12, 12)
		fwdRoundShift4(out[:]) // shift[1] = -4
		for r := range 32 {
			buf[r*32+c] = out[r]
		}
	}
	mask := fwdRowSkipMask(buf[:], 32, thr0, thr, fwdRowSkipSlack32)
	if mask == 0 {
		return 0
	}
	for r := range 32 {
		if mask&(1<<(r>>2)) == 0 {
			for c := range 32 {
				coeff[c*32+r] = 0
			}
			continue
		}
		copy(in[:], buf[r*32:r*32+32])
		fwdDCT32(&in, &out, &fwdCospi12, 12)
		for c := range 32 { // shift[2] = 0; output[c*rows + r]
			coeff[c*32+r] = out[c]
		}
	}
	return mask
}
