// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

// colPass2TestFuncs returns the NEON adapters directly so the differential test
// always exercises every NEON column kernel for bit-exactness.
func colPass2TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT8", dct8Size, inverseDCT8Col2NEONAdapter, inverseDCT8Col2PureGo},
		{"DCT16", dct16Size, inverseDCT16Col2NEONAdapter, inverseDCT16Col2PureGo},
		{"DCT32", dct32Size, inverseDCT32Col2NEONAdapter, inverseDCT32Col2PureGo},
	}
}

// colPass4TestFuncs returns the NEON four-column adapters directly so the
// differential test always exercises the four-column kernels for
// bit-exactness.
func colPass4TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT32Col4", dct32Size, inverseDCT32Col4NEONAdapter, inverseDCT32Col4PureGo},
		{"DCT64Col4", dct64Size, inverseDCT64Col4NEONAdapter, inverseDCT64Col4PureGo},
	}
}
