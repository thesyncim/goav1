//go:build !arm64 || purego

package encoder

func realtimeIntProRowInBoundsArch(dst []int16, ref []byte, stride, projWidth, projHeight, normFactor int) {
	realtimeIntProRowInBoundsPureGo(dst, ref, stride, projWidth, projHeight, normFactor)
}

func realtimeIntProColInBoundsArch(dst []int16, ref []byte, stride, projWidth, projHeight, normFactor int) {
	realtimeIntProColInBoundsPureGo(dst, ref, stride, projWidth, projHeight, normFactor)
}
