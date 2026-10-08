// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

// row2TestImpls uses the live dispatch slots for DCT8/16 and the pure-Go slots
// for the remaining row2 transforms. The Go SIMD dispatch binding is checked
// separately in dct2lane_gosimd_amd64_test.go.
func row2TestImpls() []row2TestFunc {
	return []row2TestFunc{
		{"DCT4", dct4Size, inverseDCT4Row2Impl, inverseDCT4Row2PureGo},
		{"DCT8", dct8Size, inverseDCT8Row2Impl, inverseDCT8Row2PureGo},
		{"DCT16", dct16Size, inverseDCT16Row2Impl, inverseDCT16Row2PureGo},
		{"ADST4", adst4Size, inverseADST4Row2Impl, inverseADST4Row2PureGo},
		{"ADST8", adst8Size, inverseADST8Row2Impl, inverseADST8Row2PureGo},
	}
}

// row4TestImpls returns the AVX2 four-row adapters directly so the differential
// test always exercises the AVX2 kernels for bit-exactness, independent of
// whether the dispatcher binds them (Rosetta 2 does not advertise AVX2 but still
// executes the instructions).
func row4TestImpls() []row4TestFunc {
	return []row4TestFunc{
		{"DCT32Row4", dct32Size, inverseDCT32Row4AVX2Adapter, inverseDCT32Row4PureGo},
		{"DCT64Row4", dct64Size, inverseDCT64Row4AVX2Adapter, inverseDCT64Row4PureGo},
	}
}
