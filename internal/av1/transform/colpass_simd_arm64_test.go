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
	}
}
