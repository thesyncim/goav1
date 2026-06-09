package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// mode_write.go holds the forwards of the block mode-symbol readers needed to
// assemble an intra keyframe: skip_transform, luma intra mode, intra angle
// delta, chroma intra mode, and the CfL alphas. Each writer mirrors its reader
// exactly — same context derivation, same CDF selection, same conditional
// no-symbol cases — so encoder and decoder CDF state adapt in lockstep.

// WriteSkipTransform codes skip_transform for one block, the forward of
// ReadSkipTransform. When skip_mode or the segment forces skip the decoder
// derives skip without reading a symbol, so the writer codes nothing and
// requires skip to be true.
func WriteSkipTransform(w *entropy.Writer, cdfs *BlockModeCDFs, ctx *BlockModeContext, req BlockModeRequest, skipMode bool, skip bool) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	if skipMode || segmentForcesSkip(req) {
		if !skip {
			return ErrInvalidDecodeState
		}
		return nil
	}
	context, err := ctx.SkipContext(int(req.X4), int(req.Y4))
	if err != nil {
		return err
	}
	cdf, err := cdfs.SkipCDF(context)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, boolToSym(skip))
	return nil
}

// WriteLumaIntraMode codes one luma intra prediction mode, the forward of
// ReadLumaIntraMode: keyframes use the above/left keyframe mode context,
// inter/switch frames the block-size-class y_mode CDF.
func WriteLumaIntraMode(w *entropy.Writer, cdfs *IntraModeCDFs, ctx *BlockModeContext, req LumaIntraModeRequest, mode IntraMode) error {
	if w == nil || !mode.Valid() {
		return ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok {
		return ErrInvalidDecodeState
	}
	var cdf *entropy.CDF
	var err error
	if frameTypeIsInterOrSwitch(req.FrameType) {
		cdf, err = cdfs.YModeCDF(int(yModeSizeContext[req.Size]))
	} else {
		above, left, ctxErr := ctx.KeyframeYModeContext(int(req.X4), int(req.Y4))
		if ctxErr != nil {
			return ctxErr
		}
		cdf, err = cdfs.KeyframeYModeCDF(above, left)
	}
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, int(mode))
	return nil
}

// WriteIntraAngleDelta codes the optional directional intra angle delta, the
// forward of ReadIntraAngleDelta. Blocks that do not code the delta (small or
// non-directional) require delta == 0 and code nothing.
func WriteIntraAngleDelta(w *entropy.Writer, cdfs *IntraModeCDFs, req IntraAngleDeltaRequest, delta int8) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	write, err := shouldReadIntraAngleDelta(req.Size, req.Mode)
	if err != nil {
		return err
	}
	if !write {
		if delta != 0 {
			return ErrInvalidDecodeState
		}
		return nil
	}
	if delta < -AngleDeltaMax || delta > AngleDeltaMax {
		return ErrInvalidDecodeState
	}
	cdf, err := cdfs.AngleDeltaCDF(req.Mode)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, int(delta)+AngleDeltaMax)
	return nil
}

// WriteCFLAlphas codes the CfL joint sign and per-plane alpha indices, the
// forward of ReadCFLAlphas. alpha.Index packs the U index in bits 7:4 and the V
// index in bits 3:0; a plane whose sign is zero codes no alpha symbol and its
// packed index nibble must be zero.
func WriteCFLAlphas(w *entropy.Writer, cdfs *IntraModeCDFs, alpha CFLAlphaResult) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	jointSign := int(alpha.JointSign)
	if jointSign < 0 || jointSign >= CFLJointSigns {
		return ErrInvalidDecodeState
	}
	signCDF, err := cdfs.CFLSignCDF()
	if err != nil {
		return err
	}
	w.WriteCDF(signCDF, jointSign)
	alphaU := int(alpha.Index >> 4)
	alphaV := int(alpha.Index & 0xF)
	if cflSignU(jointSign) != cflSignZero {
		cdf, err := cdfs.CFLAlphaCDF(cflContextU(jointSign))
		if err != nil {
			return err
		}
		w.WriteCDF(cdf, alphaU)
	} else if alphaU != 0 {
		return ErrInvalidDecodeState
	}
	if cflSignV(jointSign) != cflSignZero {
		cdf, err := cdfs.CFLAlphaCDF(cflContextV(jointSign))
		if err != nil {
			return err
		}
		w.WriteCDF(cdf, alphaV)
	} else if alphaV != 0 {
		return ErrInvalidDecodeState
	}
	return nil
}

// WriteChromaIntraMode codes the UV intra mode and, for CfL, the alpha signs and
// indices, the forward of ReadChromaIntraMode.
func WriteChromaIntraMode(w *entropy.Writer, cdfs *IntraModeCDFs, req ChromaIntraModeRequest, mode ChromaIntraMode, alpha CFLAlphaResult) error {
	if w == nil || !req.LumaMode.Valid() || !mode.Valid() {
		return ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok {
		return ErrInvalidDecodeState
	}
	if !req.CFLAllowed && mode == ChromaIntraModeCFL {
		return ErrInvalidDecodeState
	}
	cdf, err := cdfs.UVModeCDF(req.CFLAllowed, req.LumaMode)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, int(mode))
	if mode != ChromaIntraModeCFL {
		return nil
	}
	return WriteCFLAlphas(w, cdfs, alpha)
}
