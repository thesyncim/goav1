//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package encoder

// sad8x4x4 computes four 8x4 SADs of one source block against four reference
// origins sharing stride (svt_aom_compute8x4_sad_kernel_c).
func sad8x4x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x4PureGo(src, ref0, stride), sad8x4PureGo(src, ref1, stride),
		sad8x4PureGo(src, ref2, stride), sad8x4PureGo(src, ref3, stride)
}
