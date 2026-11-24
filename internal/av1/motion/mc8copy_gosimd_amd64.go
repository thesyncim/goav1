// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// The put copies (dav1d put_avx2 / put_16bpc_avx2): an in-plane w x h block
// moved as vector rows over footprint-checked Grid views instead of one
// memmove per row. unsafeutil.CopyRect is not used here: on amd64 the
// compiler lowers its 32- and 64-byte array copies to runtime.memmove calls,
// so the rows go through archsimd loads and stores instead (VMOVDQU ymm, then
// VZEROUPPER at exit as the AVX2 kernels do).

// mc8CopyFast is the 8-bit put copy for an in-plane w x h block with w a
// multiple of 4. It returns false when the source leaves the plane (the
// clamped copy runs).
func mc8CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
		refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < ref.Width {
		return false
	}
	dOff, sOff := dstY*dst.Stride+dstX, refY*ref.Stride+refX
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, w, h) {
		return false
	}
	mc8CopyGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, w, h))
	return true
}

// mc16CopyFast is the 16bpc put copy: an in-plane w x h block of a
// 2-byte-sample plane, w a multiple of 4, copied as rows of 2w bytes. It
// returns false when the source leaves the plane (the clamped copy runs).
func mc16CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width ||
		refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < 2*ref.Width {
		return false
	}
	dOff, sOff := dstY*dst.Stride+2*dstX, refY*ref.Stride+2*refX
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, 2*w, h) {
		return false
	}
	mc8CopyGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, 2*w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, 2*w, h))
	return true
}

// mc8CopyGrid copies the rows of sg onto dg, two views of the same shape
// whose width is a multiple of 4 bytes: 4- and 8-byte rows as scalar moves,
// 16-byte rows as one xmm load and store, wider rows in 32-byte ymm pieces
// (with a 16-byte tail for widths that are 16 mod 32). Every row index is
// the loop variable, bounded by the views' common height, and every column
// a constant or a strip view's, so the rows pay no checks.
func mc8CopyGrid(dg, sg unsafeutil.Grid[uint8]) {
	h, n := dg.Height(), dg.Width()
	if !sg.Fits(0, 0, n, h) {
		return
	}
	switch {
	case n == 4:
		for r := 0; r < h; r++ {
			*dg.Row4(r, 0) = *sg.Row4(r, 0)
		}
	case n == 8:
		for r := 0; r < h; r++ {
			*dg.Row8(r, 0) = *sg.Row8(r, 0)
		}
	case n == 16:
		for r := 0; r < h; r++ {
			archsimd.LoadUint8x16Array(sg.Row16(r, 0)).StoreArray(dg.Row16(r, 0))
		}
	case n == 32:
		for r := 0; r < h; r++ {
			archsimd.LoadUint8x32Array(sg.Row32(r, 0)).StoreArray(dg.Row32(r, 0))
		}
		archsimd.ClearAVXUpperBits()
	case n == 64:
		for r := 0; r < h; r++ {
			s, d := sg.Row64(r, 0), dg.Row64(r, 0)
			a := archsimd.LoadUint8x32Array((*[32]byte)(s[0:32]))
			b := archsimd.LoadUint8x32Array((*[32]byte)(s[32:64]))
			a.StoreArray((*[32]byte)(d[0:32]))
			b.StoreArray((*[32]byte)(d[32:64]))
		}
		archsimd.ClearAVXUpperBits()
	default:
		// Wider rows (and widths that are not powers of two): 64-byte strips,
		// then 32-, 16-, 8- and 4-byte ones.
		c := 0
		for ; c+64 <= n; c += 64 {
			mc8CopyGrid(dg.Sub(0, c, 64, h), sg.Sub(0, c, 64, h))
		}
		for _, k := range [...]int{32, 16, 8, 4} {
			if c+k <= n {
				mc8CopyGrid(dg.Sub(0, c, k, h), sg.Sub(0, c, k, h))
				c += k
			}
		}
	}
}
