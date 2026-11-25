//go:build (!amd64 && !arm64) || purego || (arm64 && !goexperiment.simd)

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
