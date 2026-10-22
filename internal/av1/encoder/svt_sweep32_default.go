//go:build !(goexperiment.simd && !purego && (arm64 || amd64))

package encoder

func (me *realtimeSBMultiSizeME) eightSAD32x32_64x64(sad16 *[16][8]uint32, mvX, mvY int) {
	me.eightSAD32x32_64x64Generic(sad16, mvX, mvY)
}
