// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || purego

package transform

// row2TestImpls returns the live dispatch slots on architectures without NEON
// kernels; there the slots are the pure-Go references and the test degenerates
// to a self-consistency check.
func row2TestImpls() []row2TestFunc {
	return []row2TestFunc{
		{"DCT4", dct4Size, inverseDCT4Row2Impl, inverseDCT4Row2PureGo},
		{"DCT8", dct8Size, inverseDCT8Row2Impl, inverseDCT8Row2PureGo},
		{"DCT16", dct16Size, inverseDCT16Row2Impl, inverseDCT16Row2PureGo},
		{"ADST4", adst4Size, inverseADST4Row2Impl, inverseADST4Row2PureGo},
		{"ADST8", adst8Size, inverseADST8Row2Impl, inverseADST8Row2PureGo},
	}
}

// row4TestImpls returns the live four-row dispatch slots; without a four-row
// NEON kernel they are the pure-Go references and the test degenerates to a
// self-consistency check.
func row4TestImpls() []row4TestFunc {
	return []row4TestFunc{
		{"DCT32Row4", dct32Size, inverseDCT32Row4Impl, inverseDCT32Row4PureGo},
		{"DCT64Row4", dct64Size, inverseDCT64Row4Impl, inverseDCT64Row4PureGo},
	}
}
