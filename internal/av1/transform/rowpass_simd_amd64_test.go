// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

// row2TestImpls returns the AVX2 adapters directly so the differential test
// always exercises the AVX2 kernels for bit-exactness, independent of whether
// the dispatcher binds them (cpu.Detected.AVX2 is false on hosts that do not
// advertise AVX2, e.g. Rosetta 2, which nonetheless still executes the
// instructions). The lengths without an AVX2 kernel fall back to the live
// dispatch slots (pure-Go), degenerating to a self-consistency check.
func row2TestImpls() []row2TestFunc {
	return []row2TestFunc{
		{"DCT4", dct4Size, inverseDCT4Row2Impl, inverseDCT4Row2PureGo},
		{"DCT8", dct8Size, inverseDCT8Row2AVX2Adapter, inverseDCT8Row2PureGo},
		{"DCT16", dct16Size, inverseDCT16Row2AVX2Adapter, inverseDCT16Row2PureGo},
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
