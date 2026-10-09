// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

// colPass2TestFuncs uses the live dispatch slots for DCT8/16; the Go SIMD
// dispatch binding is checked separately in dct2lane_gosimd_amd64_test.go.
func colPass2TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT8", dct8Size, inverseDCT8Col2Impl, inverseDCT8Col2PureGo},
		{"DCT16", dct16Size, inverseDCT16Col2Impl, inverseDCT16Col2PureGo},
	}
}

func colPass4TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT16Col4", dct16Size, inverseDCT16Col4Impl, inverseDCT16Col4PureGo},
		{"DCT32Col4", dct32Size, inverseDCT32Col4Impl, inverseDCT32Col4PureGo},
		{"DCT64Col4", dct64Size, inverseDCT64Col4Impl, inverseDCT64Col4PureGo},
	}
}
