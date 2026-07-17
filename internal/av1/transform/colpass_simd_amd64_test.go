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

// colPass4TestFuncs returns the AVX2 four-column adapters directly so the
// differential test always exercises the AVX2 kernels for bit-exactness,
// independent of whether the dispatcher binds them (Rosetta 2 does not advertise
// AVX2 but still executes the instructions).
func colPass4TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT32Col4", dct32Size, inverseDCT32Col4AVX2Adapter, inverseDCT32Col4PureGo},
		{"DCT64Col4", dct64Size, inverseDCT64Col4AVX2Adapter, inverseDCT64Col4PureGo},
		{"Identity4Col4", 4, inverseIdentity4Col4Impl, inverseIdentity4Col4PureGo},
		{"Identity8Col4", 8, inverseIdentity8Col4Impl, inverseIdentity8Col4PureGo},
		{"Identity16Col4", 16, inverseIdentity16Col4Impl, inverseIdentity16Col4PureGo},
		{"Identity32Col4", 32, inverseIdentity32Col4Impl, inverseIdentity32Col4PureGo},
	}
}
