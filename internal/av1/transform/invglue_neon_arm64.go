//go:build arm64 && !purego

package transform

import "unsafe"

// invGlueCtx carries the in-place round-and-clamp kernel arguments; offsets
// are mirrored by #define in invglue_neon_arm64.s.
type invGlueCtx struct {
	Ptr    unsafe.Pointer
	N      int64
	NShift int64
	Min    int64
	Max    int64
}

//go:noescape
func clampRoundNEONAsm(ctx *invGlueCtx)

// invStoreCtx carries the narrowing-store kernel arguments.
type invStoreCtx struct {
	Dst       unsafe.Pointer
	DstStride int64
	Src       unsafe.Pointer
	Width     int64
	Height    int64
}

//go:noescape
func narrowStoreNEONAsm(ctx *invStoreCtx)

// clampRoundImpl rounds-and-shifts then clamps scratch in place.
var clampRoundImpl = clampRoundNEON

func clampRoundNEON(scratch []int32, shift int, min, max int32) {
	if len(scratch)%8 != 0 {
		clampRoundPureGo(scratch, shift, min, max)
		return
	}
	ctx := invGlueCtx{
		Ptr:    unsafe.Pointer(&scratch[0]),
		N:      int64(len(scratch)),
		NShift: -int64(shift),
		Min:    int64(min),
		Max:    int64(max),
	}
	clampRoundNEONAsm(&ctx)
}

// narrowStoreImpl writes the final residual rows.
var narrowStoreImpl = narrowStoreNEON

func narrowStoreNEON(dst []int16, dstStride int, scratch []int32, width, height int) {
	if width != 4 && width%8 != 0 {
		narrowStorePureGo(dst, dstStride, scratch, width, height)
		return
	}
	ctx := invStoreCtx{
		Dst:       unsafe.Pointer(&dst[0]),
		DstStride: int64(dstStride),
		Src:       unsafe.Pointer(&scratch[0]),
		Width:     int64(width),
		Height:    int64(height),
	}
	narrowStoreNEONAsm(&ctx)
}
