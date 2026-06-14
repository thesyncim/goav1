//go:build arm64 && !purego

package encoder

import "unsafe"

// rdNEONCtx carries the residual-extraction kernel arguments; offsets are
// mirrored by #define in rdstats_neon_arm64.s.
type rdNEONCtx struct {
	Src        unsafe.Pointer
	SrcStride  int64
	Pred       unsafe.Pointer
	PredStride int64
	Dst        unsafe.Pointer
	W          int64
	H          int64
}

//go:noescape
func residualNEONAsm(ctx *rdNEONCtx)

// rdStatsNEONCtx carries the RD-statistics kernel arguments. NShift is the
// negated transform scale (SSHL by a negative count shifts right).
type rdStatsNEONCtx struct {
	Tran    unsafe.Pointer
	QCoeff  unsafe.Pointer
	Count   int64
	Step    int64
	NShift  int64
	DSkip   int64
	DCode   int64
	Rate    int64
	NonZero int64
}

// blockErrorNEONCtx carries SVT-shaped coefficient block-error arguments.
type blockErrorNEONCtx struct {
	Coeff   unsafe.Pointer
	DQCoeff unsafe.Pointer
	Count   int64
	Error   int64
	SSZ     int64
}

//go:noescape
func rdStatsNEONAsm(ctx *rdStatsNEONCtx)

//go:noescape
func blockErrorNEONAsm(ctx *blockErrorNEONCtx)

// residualBlockImpl extracts the prediction residual; the NEON kernel covers
// the 8/16/32-wide shapes the coder produces.
var residualBlockImpl = residualBlockNEON

func residualBlockNEON(dst []int16, srcPlane []byte, srcOff, stride int, pred []byte, predStride, w, h int) {
	if w < 8 {
		// The kernel's narrowest row form is eight columns.
		residualBlockPureGo(dst, srcPlane, srcOff, stride, pred, predStride, w, h)
		return
	}
	ctx := rdNEONCtx{
		Src:        unsafe.Pointer(&srcPlane[srcOff]),
		SrcStride:  int64(stride),
		Pred:       unsafe.Pointer(&pred[0]),
		PredStride: int64(predStride),
		Dst:        unsafe.Pointer(&dst[0]),
		W:          int64(w),
		H:          int64(h),
	}
	residualNEONAsm(&ctx)
}

// rdStatsBlockImpl accumulates the skip-decision statistics with the AC step
// applied to every lane; the caller redoes the DC coefficient.
var rdStatsBlockImpl = rdStatsBlockNEON

func rdStatsBlockNEON(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (dskip, dcode int64, rate int64, allZero bool) {
	ctx := rdStatsNEONCtx{
		Tran:   unsafe.Pointer(&tran[0]),
		QCoeff: unsafe.Pointer(&qcoeff[0]),
		Count:  int64(count),
		Step:   int64(step),
		NShift: -int64(ts),
	}
	rdStatsNEONAsm(&ctx)
	return ctx.DSkip, ctx.DCode, ctx.Rate << 9, ctx.NonZero == 0
}

func blockError(coeff []int32, dqcoeff []int32, count int) (err int64, ssz int64) {
	return blockErrorNEON(coeff, dqcoeff, count)
}

func blockErrorNEON(coeff []int32, dqcoeff []int32, count int) (err int64, ssz int64) {
	if count < 8 || count&7 != 0 {
		return blockErrorPureGo(coeff, dqcoeff, count)
	}
	ctx := blockErrorNEONCtx{
		Coeff:   unsafe.Pointer(&coeff[0]),
		DQCoeff: unsafe.Pointer(&dqcoeff[0]),
		Count:   int64(count),
	}
	blockErrorNEONAsm(&ctx)
	return ctx.Error, ctx.SSZ
}
