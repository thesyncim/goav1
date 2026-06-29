//go:build !arm64 || purego

package encoder

func buildQuarterPlaneArch(dst []byte, src []byte, stride, qw, qh int) {
	buildQuarterPlanePureGo(dst, src, stride, qw, qh)
}
