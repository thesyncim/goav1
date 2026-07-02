// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

// row2TestImpls returns the NEON adapters directly so the differential test
// always exercises every NEON kernel for bit-exactness, even the ones the
// dispatcher leaves unbound for performance reasons.
func row2TestImpls() []row2TestFunc {
	return []row2TestFunc{
		{"DCT4", dct4Size, inverseDCT4Row2NEONAdapter, inverseDCT4Row2PureGo},
		{"DCT8", dct8Size, inverseDCT8Row2NEONAdapter, inverseDCT8Row2PureGo},
		{"DCT16", dct16Size, inverseDCT16Row2NEONAdapter, inverseDCT16Row2PureGo},
		{"DCT32", dct32Size, inverseDCT32Row2NEONAdapter, inverseDCT32Row2PureGo},
		{"DCT64", dct64Size, inverseDCT64Row2NEONAdapter, inverseDCT64Row2PureGo},
		{"ADST4", adst4Size, inverseADST4Row2NEONAdapter, inverseADST4Row2PureGo},
		{"ADST8", adst8Size, inverseADST8Row2NEONAdapter, inverseADST8Row2PureGo},
	}
}

// row4TestImpls returns the NEON four-row adapters directly so the
// differential test always exercises the four-row kernels for bit-exactness.
func row4TestImpls() []row4TestFunc {
	return []row4TestFunc{
		{"DCT32Row4", dct32Size, inverseDCT32Row4NEONAdapter, inverseDCT32Row4PureGo},
		{"DCT64Row4", dct64Size, inverseDCT64Row4NEONAdapter, inverseDCT64Row4PureGo},
	}
}
