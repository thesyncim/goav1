//go:build arm64 && !purego

package encoder

import "unsafe"

// realtimeIntProNEONCtx carries libaom's aom_int_pro_row_neon /
// aom_int_pro_col_neon arguments. Field offsets are mirrored by
// pframe_intpro_arm64.s.
type realtimeIntProNEONCtx struct {
	Dst    unsafe.Pointer
	Ref    unsafe.Pointer
	Stride int64
	Width  int64
	Height int64
}

//go:noescape
func realtimeIntProRowNEONAsm(ctx *realtimeIntProNEONCtx)

//go:noescape
func realtimeIntProColNEONAsm(ctx *realtimeIntProNEONCtx)

func realtimeIntProRowInBoundsArch(dst []int16, ref []byte, stride, projWidth, projHeight, normFactor int) {
	// The row kernel sums projHeight source bytes per column into 16-bit lanes;
	// the u16 lane overflows once projHeight*255 exceeds 65535, i.e. past
	// projHeight == 257. Beyond that the lane accumulator diverges from the
	// pure-Go int reference, so fall back to pure-Go.
	if projWidth == 0 || projHeight == 0 ||
		projWidth&15 != 0 || projHeight&3 != 0 || normFactor != 5 ||
		projHeight > 257 {
		realtimeIntProRowInBoundsPureGo(dst, ref, stride, projWidth, projHeight, normFactor)
		return
	}
	_ = dst[projWidth-1]
	_ = ref[(projHeight-1)*stride+projWidth-1]
	ctx := realtimeIntProNEONCtx{
		Dst:    unsafe.Pointer(&dst[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
		Width:  int64(projWidth),
		Height: int64(projHeight),
	}
	realtimeIntProRowNEONAsm(&ctx)
}

func realtimeIntProColInBoundsArch(dst []int16, ref []byte, stride, projWidth, projHeight, normFactor int) {
	// The column kernel accumulates projWidth/16 pairwise byte sums per 16-bit
	// lane; the u16 lane overflows once (projWidth/16)*510 exceeds 65535, i.e.
	// past projWidth == 2048. Beyond that the lane accumulator diverges from the
	// pure-Go int reference, so fall back to pure-Go.
	if projWidth == 0 || projHeight == 0 ||
		projWidth&15 != 0 || projHeight&3 != 0 || normFactor != 5 ||
		projWidth > 2056 {
		realtimeIntProColInBoundsPureGo(dst, ref, stride, projWidth, projHeight, normFactor)
		return
	}
	_ = dst[projHeight-1]
	_ = ref[(projHeight-1)*stride+projWidth-1]
	ctx := realtimeIntProNEONCtx{
		Dst:    unsafe.Pointer(&dst[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
		Width:  int64(projWidth),
		Height: int64(projHeight),
	}
	realtimeIntProColNEONAsm(&ctx)
}
