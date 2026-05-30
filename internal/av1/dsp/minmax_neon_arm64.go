// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

// NEON-accelerated MinMaxAbsDiff8x8. The .s file implements the 8-row inner
// loop for both the 8-bit (uabd v.8b + uminv/umaxv) and 16-bit (uabd v.8h)
// shapes. The Go wrapper here performs the exact same bounds and stride
// validation as minMaxAbsDiff8x8PureGo so the error behaviour is identical,
// then hands the validated row pointers to the kernel.
//
// Bit-exactness with minMaxAbsDiff8x8PureGo:
//   - uabd computes |a-b| per lane, matching absDiff8/absDiff16.
//   - the running umin/umax fold is order-independent and covers all 64
//     samples, so the horizontal uminv/umaxv reduction equals the scalar
//     min/max scan in the reference.

//go:noescape
func minMaxAbsDiff8x8NEON8Asm(a *byte, aStride uintptr, b *byte, bStride uintptr, out *uint32)

//go:noescape
func minMaxAbsDiff8x8NEON16Asm(a *byte, aStride uintptr, b *byte, bStride uintptr, out *uint32)

func minMaxAbsDiff8x8NEON(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return 0, 0, ErrInvalidBlock
	}
	rowBytes := 8 * bytesPerSample
	if aStride < rowBytes || bStride < rowBytes ||
		!byteBlockFits(len(a), aStride, rowBytes, 8) ||
		!byteBlockFits(len(b), bStride, rowBytes, 8) {
		return 0, 0, ErrInvalidBlock
	}

	var out [2]uint32
	switch bytesPerSample {
	case 1:
		minMaxAbsDiff8x8NEON8Asm(&a[0], uintptr(aStride), &b[0], uintptr(bStride), &out[0])
	case 2:
		minMaxAbsDiff8x8NEON16Asm(&a[0], uintptr(aStride), &b[0], uintptr(bStride), &out[0])
	}
	return uint16(out[0]), uint16(out[1]), nil
}
