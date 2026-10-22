//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

// sweepSearchPoint8 has no portable batch; the four-position path runs.
func (me *realtimeSBMultiSizeME) sweepSearchPoint8(src, ref []byte, stride, srcBase, refBase, mvX, mvY int) bool {
	return false
}
