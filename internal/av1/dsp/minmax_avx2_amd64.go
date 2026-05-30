// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

// AVX2-accelerated MinMaxAbsDiff8x8. The .s file implements the 8-row inner
// loop for both the 8-bit (VPMINUB/VPMAXUB byte lanes) and 16-bit
// (VPMINUW/VPMAXUW word lanes) shapes. The Go wrapper performs the exact same
// bounds and stride validation as minMaxAbsDiff8x8PureGo so the error
// behaviour is identical, then hands the validated row pointers to the kernel.
//
// Bit-exactness with minMaxAbsDiff8x8PureGo:
//   - the per-lane absolute difference is computed as max(a,b)-min(a,b)
//     (VPMAXUB/VPMINUB then VPSUBB; VPMAXUW/VPMINUW then VPSUBW), matching
//     absDiff8/absDiff16.
//   - the running per-lane min/max fold is order-independent and covers all
//     64 samples, so the horizontal reduction equals the scalar min/max scan
//     in the reference.

//go:noescape
func minMaxAbsDiff8x8AVX2_8(a *byte, aStride uintptr, b *byte, bStride uintptr, out *uint32)

//go:noescape
func minMaxAbsDiff8x8AVX2_16(a *byte, aStride uintptr, b *byte, bStride uintptr, out *uint32)

func minMaxAbsDiff8x8AVX2(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
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
		minMaxAbsDiff8x8AVX2_8(&a[0], uintptr(aStride), &b[0], uintptr(bStride), &out[0])
	case 2:
		minMaxAbsDiff8x8AVX2_16(&a[0], uintptr(aStride), &b[0], uintptr(bStride), &out[0])
	}
	return uint16(out[0]), uint16(out[1]), nil
}
