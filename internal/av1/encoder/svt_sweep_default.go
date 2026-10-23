//go:build !(goexperiment.simd && !purego && (arm64 || amd64))

package encoder

func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	svtAllSAD8x8_16x16SubScalar(me, src, ref, stride, mvX, mvY, sad16)
}

func (me *realtimeSBMultiSizeME) sweepPoint1Arch(src []byte, stride int, ref []byte, srcBase, refBase, mvX, mvY int) {
	me.sweepSearchPoint1(src, stride, ref, stride, srcBase, refBase, mvX, mvY)
}
