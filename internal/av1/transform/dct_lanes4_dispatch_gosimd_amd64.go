//go:build goexperiment.simd && amd64 && !purego

package transform

import "simd/archsimd"

func init() {
	if archsimd.X86.AVX2() {
		inverseDCT32Col4Impl = inverseDCT32Lanes4ColSIMD
		inverseDCT32Row4Impl = inverseDCT32Lanes4RowSIMD
		inverseDCT64Col4Impl = inverseDCT64Col4SIMD
		inverseDCT64Row4Impl = inverseDCT64Row4SIMD
	}
}
