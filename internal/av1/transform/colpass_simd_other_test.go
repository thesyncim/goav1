// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || purego

package transform

// colPass2TestFuncs returns the live dispatch slots on architectures without
// NEON column kernels; there the slots are the pure-Go references and the test
// degenerates to a self-consistency check.
func colPass2TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT8", dct8Size, inverseDCT8Col2Impl, inverseDCT8Col2PureGo},
		{"DCT16", dct16Size, inverseDCT16Col2Impl, inverseDCT16Col2PureGo},
	}
}

// colPass4TestFuncs returns the live four-column dispatch slots; without a
// four-column NEON kernel they are the pure-Go references and the test
// degenerates to a self-consistency check.
func colPass4TestFuncs() []col2TestFunc {
	return []col2TestFunc{
		{"DCT32Col4", dct32Size, inverseDCT32Col4Impl, inverseDCT32Col4PureGo},
		{"DCT64Col4", dct64Size, inverseDCT64Col4Impl, inverseDCT64Col4PureGo},
	}
}
