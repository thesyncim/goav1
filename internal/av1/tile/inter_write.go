package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// inter_write.go holds the forwards of the single-reference inter mode-symbol
// readers needed for a P-frame: the new/global/ref-mv mode cascade, the DRL
// index, and the motion-vector joint/component residual. Each writer mirrors
// its reader's CDF selection and conditional no-symbol cases exactly. Ported
// against libaom av1/encoder/encodemv.c (encode_mv_component) and
// bitstream.c (write_inter_mode / write_drl_idx).

// WriteSingleInterMode codes one single-reference inter mode, the forward of
// ReadSingleInterMode's new->global->ref cascade.
func WriteSingleInterMode(w *entropy.Writer, cdfs *InterModeCDFs, modeContext uint16, mode InterMode) error {
	if w == nil || !mode.Valid() {
		return ErrInvalidDecodeState
	}
	newCtx := int(modeContext & newMVMask)
	cdf, err := cdfs.NewMVCDF(newCtx)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, boolToSym(mode != InterModeNewMV))
	if mode == InterModeNewMV {
		return nil
	}

	globalCtx := int((modeContext >> globalMVOffset) & globalMVMask)
	cdf, err = cdfs.GlobalMVCDF(globalCtx)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, boolToSym(mode != InterModeGlobalMV))
	if mode == InterModeGlobalMV {
		return nil
	}

	refCtx := int((modeContext >> refMVOffset) & refMVMask)
	cdf, err = cdfs.RefMVCDF(refCtx)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, boolToSym(mode != InterModeNearestMV))
	return nil
}

// WriteDRLIndex codes libaom's ref_mv_idx, the forward of ReadDRLIndex: NEWMV
// walks stack slots 0..1 and NEARMV slots 1..2, emitting one continuation bit
// per visited slot while the stack still has candidates.
func WriteDRLIndex(w *entropy.Writer, cdfs *InterModeCDFs, req DRLRequest, refMVIndex int) error {
	if w == nil || req.RefMVCount > MaxRefMVStackSize || refMVIndex < 0 {
		return ErrInvalidDecodeState
	}
	if req.Compound {
		if !req.CompoundMode.Valid() {
			return ErrInvalidDecodeState
		}
	} else if !req.Mode.Valid() {
		return ErrInvalidDecodeState
	}
	refMVCount := int(req.RefMVCount)

	if req.usesNewMV() {
		for idx := range 2 {
			if refMVCount <= idx+1 {
				if refMVIndex != minInt(idx, refMVIndex) || refMVIndex > idx {
					return ErrInvalidDecodeState
				}
				return nil
			}
			bit := refMVIndex > idx
			if err := writeDRLBit(w, cdfs, req.Contexts[idx], bit); err != nil {
				return err
			}
			if !bit {
				return nil
			}
		}
		return nil
	}
	if req.usesNearMV() {
		for idx := 1; idx < 3; idx++ {
			if refMVCount <= idx+1 {
				if refMVIndex > idx-1 {
					return ErrInvalidDecodeState
				}
				return nil
			}
			bit := refMVIndex > idx-1
			if err := writeDRLBit(w, cdfs, req.Contexts[idx], bit); err != nil {
				return err
			}
			if !bit {
				return nil
			}
		}
		return nil
	}
	if refMVIndex != 0 {
		return ErrInvalidDecodeState
	}
	return nil
}

func writeDRLBit(w *entropy.Writer, cdfs *InterModeCDFs, context uint8, bit bool) error {
	cdf, err := cdfs.DRLCDF(int(context))
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, boolToSym(bit))
	return nil
}

// WriteMotionVector codes one motion vector against its reference predictor:
// the joint symbol, then the row and column component residuals that the
// joint marks non-zero. The forward of ReadMotionVector.
func WriteMotionVector(w *entropy.Writer, cdfs *MVCDFs, mv motion.Vector, ref motion.Vector, precision MVSubpelPrecision) error {
	if w == nil || !precision.Valid() {
		return ErrInvalidDecodeState
	}
	diffRow := int32(mv.Row) - int32(ref.Row)
	diffCol := int32(mv.Col) - int32(ref.Col)
	joint := MVJointZero
	switch {
	case diffRow != 0 && diffCol != 0:
		joint = MVJointBoth
	case diffRow != 0:
		joint = MVJointVertical
	case diffCol != 0:
		joint = MVJointHorizontal
	}
	cdf, err := cdfs.JointCDF()
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, int(joint))
	if joint.HasVertical() {
		if err := WriteMVComponentDiff(w, &cdfs.Components[mvComponentRow], precision, diffRow); err != nil {
			return err
		}
	}
	if joint.HasHorizontal() {
		if err := WriteMVComponentDiff(w, &cdfs.Components[mvComponentCol], precision, diffCol); err != nil {
			return err
		}
	}
	return nil
}

// WriteMVComponentDiff codes one non-zero MV component residual, the forward
// of ReadMVComponentDiff / libaom encode_mv_component: sign, magnitude class,
// class0 bin or per-class integer bits, then the fractional and high-precision
// bits the precision enables. Residuals whose low bits are not representable
// at the given precision (fraction != 3 or hp != 1 when those symbols are not
// coded) are rejected, mirroring the decoder's implied defaults.
func WriteMVComponentDiff(w *entropy.Writer, cdfs *MVComponentCDFs, precision MVSubpelPrecision, diff int32) error {
	if w == nil || !precision.Valid() || diff == 0 {
		return ErrInvalidDecodeState
	}
	sign := diff < 0
	mag := diff
	if sign {
		mag = -mag
	}
	z := int(mag) - 1
	mvClass := 0
	for c := MVClasses - 1; c > 0; c-- {
		if z >= MVClass0Size<<(c+2) {
			mvClass = c
			break
		}
	}
	offset := z
	if mvClass > 0 {
		offset = z - MVClass0Size<<(mvClass+2)
	}
	integerPart := offset >> 3
	fraction := (offset >> 1) & 3
	highPrecision := offset & 1

	if !precision.UsesSubpel() && fraction != 3 {
		return ErrInvalidDecodeState
	}
	if !precision.UsesHighPrecision() && highPrecision != 1 {
		return ErrInvalidDecodeState
	}

	signCDF, err := cdfs.SignCDF()
	if err != nil {
		return err
	}
	w.WriteCDF(signCDF, boolToSym(sign))
	classesCDF, err := cdfs.ClassesCDF()
	if err != nil {
		return err
	}
	w.WriteCDF(classesCDF, mvClass)

	if mvClass == 0 {
		class0CDF, err := cdfs.Class0CDF()
		if err != nil {
			return err
		}
		w.WriteCDF(class0CDF, integerPart)
	} else {
		n := mvClass + MVClass0Bits - 1
		for i := range n {
			bitCDF, err := cdfs.BitCDF(i)
			if err != nil {
				return err
			}
			w.WriteCDF(bitCDF, (integerPart>>i)&1)
		}
	}
	if precision.UsesSubpel() {
		var fpCDF *entropy.CDF
		if mvClass == 0 {
			fpCDF, err = cdfs.Class0FPCDF(integerPart)
		} else {
			fpCDF, err = cdfs.FPCDF()
		}
		if err != nil {
			return err
		}
		w.WriteCDF(fpCDF, fraction)
		if precision.UsesHighPrecision() {
			var hpCDF *entropy.CDF
			if mvClass == 0 {
				hpCDF, err = cdfs.Class0HPCDF()
			} else {
				hpCDF, err = cdfs.HPCDF()
			}
			if err != nil {
				return err
			}
			w.WriteCDF(hpCDF, highPrecision)
		}
	}
	return nil
}

// BlockHasTopRight reports whether the reference-MV scan may use the top-right
// neighbor for this block, exposed for the encoder so its
// ReferenceMVStackRequest matches the decode loop's blockHasTopRight exactly.
func BlockHasTopRight(sbSizeMIB uint8, block BlockVisit) bool {
	return blockHasTopRight(sbSizeMIB, block)
}
