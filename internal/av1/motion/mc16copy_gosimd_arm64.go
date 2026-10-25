// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// mc16CopyFast is the 16bpc put copy (dav1d mc16.S put_neon): an in-plane
// w x h block of a 2-byte-sample plane, w a multiple of 4, moved as vector
// rows by mc8Copy over 2w bytes instead of one memmove per row. It returns
// false when the source leaves the plane (the clamped copy runs).
func mc16CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width {
		return false
	}
	if refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < 2*ref.Width {
		return false
	}
	dOff := dstY*dst.Stride + 2*dstX
	sOff := refY*ref.Stride + 2*refX
	if dOff+(h-1)*dst.Stride+2*w > len(dst.Pix) || sOff+(h-1)*ref.Stride+2*w > len(ref.Pix) {
		return false
	}
	mc8Copy(unsafe.Pointer(&dst.Pix[dOff]), dst.Stride, unsafe.Pointer(&ref.Pix[sOff]), ref.Stride, 2*w, h)
	return true
}
