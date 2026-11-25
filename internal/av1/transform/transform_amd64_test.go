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

func row4TestImpls() []row4TestFunc {
	return []row4TestFunc{
		{"DCT32Row4", dct32Size, inverseDCT32Row4Impl, inverseDCT32Row4PureGo},
		{"DCT64Row4", dct64Size, inverseDCT64Row4Impl, inverseDCT64Row4PureGo},
	}
}
