// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

// colPass2TestFuncs returns the AVX2 column adapters directly so the
// differential test always exercises the AVX2 column kernels for bit-exactness,
// independent of whether the dispatcher binds them.
func colPass2TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT8", dct8Size, inverseDCT8Col2AVX2Adapter, inverseDCT8Col2PureGo},
		{"DCT16", dct16Size, inverseDCT16Col2AVX2Adapter, inverseDCT16Col2PureGo},
	}
}
