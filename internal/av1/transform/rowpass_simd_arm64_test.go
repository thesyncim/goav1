// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// row2TestImpls returns the arm64 adapters directly so the differential test
// always exercises every arm64 kernel for bit-exactness, even the ones the
// dispatcher leaves unbound for performance reasons.
func row2TestImpls() []row2TestFunc {
	return []row2TestFunc{
		{"DCT8", dct8Size, inverseDCT8Row2SIMDAdapter, inverseDCT8Row2PureGo},
		{"DCT16", dct16Size, inverseDCT16Row2SIMDAdapter, inverseDCT16Row2PureGo},
		{"DCT32", dct32Size, inverseDCT32Row2SIMDAdapter, inverseDCT32Row2PureGo},
	}
}

// row4TestImpls returns the arm64 four-row adapters directly so the
// differential test always exercises the four-row kernels for bit-exactness.
func row4TestImpls() []row4TestFunc {
	return []row4TestFunc{
		{"DCT32Row4", dct32Size, inverseDCT32Row4SIMDAdapter, inverseDCT32Row4PureGo},
		{"DCT64Row4", dct64Size, inverseDCT64Row4Impl, inverseDCT64Row4PureGo},
		{"ADST16Row4", adst16Size, inverseADST16Row4SIMDAdapter, inverseADST16Row4PureGo},
		{"ADST16Row4Flip", adst16Size, inverseADST16Row4FlipSIMDAdapter, inverseADST16Row4FlipPureGo},
	}
}
