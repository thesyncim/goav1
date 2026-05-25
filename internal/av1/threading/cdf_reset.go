package threading

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func resetCDFCount(cdf *entropy.CDF) error {
	if cdf == nil || cdf.Symbols() == 0 {
		return nil
	}
	return cdf.ResetCount()
}

func (s *FrameWorkTileResidualCDFStorage) ResetCDFCounts() error {
	if s == nil {
		return ErrInvalidBatch
	}
	for i := range s.Partition.Partition {
		for j := range s.Partition.Partition[i] {
			if err := resetCDFCount(&s.Partition.Partition[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range s.Mode.Skip {
		if err := resetCDFCount(&s.Mode.Skip[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&s.Mode.SkipMode[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&s.Mode.SegmentPred[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&s.Mode.SegmentID[i]); err != nil {
			return err
		}
	}
	if err := resetIntraModeCDFCounts(&s.Intra); err != nil {
		return err
	}
	if err := resetInterRefCDFCounts(&s.InterRef); err != nil {
		return err
	}
	if err := resetInterModeCDFCounts(&s.InterMode); err != nil {
		return err
	}
	if err := resetMVCDFCounts(&s.MV); err != nil {
		return err
	}
	for i := range s.Interp.Switchable {
		if err := resetCDFCount(&s.Interp.Switchable[i]); err != nil {
			return err
		}
	}
	for i := range s.Motion.MotionMode {
		if err := resetCDFCount(&s.Motion.MotionMode[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&s.Motion.OBMC[i]); err != nil {
			return err
		}
	}
	if err := resetCompoundBlendCDFCounts(&s.Blend); err != nil {
		return err
	}
	for i := range s.Transform.TxSize {
		for j := range s.Transform.TxSize[i] {
			if err := resetCDFCount(&s.Transform.TxSize[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range s.Transform.TxPartition {
		for j := range s.Transform.TxPartition[i] {
			if err := resetCDFCount(&s.Transform.TxPartition[i][j]); err != nil {
				return err
			}
		}
	}
	if err := resetTransformTypeCDFCounts(&s.TransformType); err != nil {
		return err
	}
	if err := resetCoeffCDFCounts(&s.Coeff); err != nil {
		return err
	}
	if err := resetCDFCount(&s.DeltaQ); err != nil {
		return err
	}
	if err := resetCDFCount(&s.DeltaLF); err != nil {
		return err
	}
	for i := range s.DeltaLFMulti {
		if err := resetCDFCount(&s.DeltaLFMulti[i]); err != nil {
			return err
		}
	}
	if err := resetCDFCount(&s.RestorationSwitchable); err != nil {
		return err
	}
	if err := resetCDFCount(&s.RestorationWiener); err != nil {
		return err
	}
	return resetCDFCount(&s.RestorationSGRProj)
}

func resetIntraModeCDFCounts(cdfs *tile.IntraModeCDFs) error {
	for i := range cdfs.Intra {
		if err := resetCDFCount(&cdfs.Intra[i]); err != nil {
			return err
		}
	}
	if err := resetCDFCount(&cdfs.Intrabc); err != nil {
		return err
	}
	for i := range cdfs.YMode {
		if err := resetCDFCount(&cdfs.YMode[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.KeyframeYMode {
		for j := range cdfs.KeyframeYMode[i] {
			if err := resetCDFCount(&cdfs.KeyframeYMode[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.UVMode {
		for j := range cdfs.UVMode[i] {
			if err := resetCDFCount(&cdfs.UVMode[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.AngleDelta {
		if err := resetCDFCount(&cdfs.AngleDelta[i]); err != nil {
			return err
		}
	}
	if err := resetCDFCount(&cdfs.CFLSign); err != nil {
		return err
	}
	for i := range cdfs.CFLAlpha {
		if err := resetCDFCount(&cdfs.CFLAlpha[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.FilterIntra {
		if err := resetCDFCount(&cdfs.FilterIntra[i]); err != nil {
			return err
		}
	}
	if err := resetCDFCount(&cdfs.FilterIntraMode); err != nil {
		return err
	}
	for i := range cdfs.PaletteYMode {
		for j := range cdfs.PaletteYMode[i] {
			if err := resetCDFCount(&cdfs.PaletteYMode[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.PaletteUVMode {
		if err := resetCDFCount(&cdfs.PaletteUVMode[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.PaletteYSize {
		if err := resetCDFCount(&cdfs.PaletteYSize[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&cdfs.PaletteUVSize[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.PaletteYColorIndex {
		for j := range cdfs.PaletteYColorIndex[i] {
			if err := resetCDFCount(&cdfs.PaletteYColorIndex[i][j]); err != nil {
				return err
			}
			if err := resetCDFCount(&cdfs.PaletteUVColorIndex[i][j]); err != nil {
				return err
			}
		}
	}
	return nil
}

func resetInterRefCDFCounts(cdfs *tile.InterRefCDFs) error {
	for i := range cdfs.CompInter {
		if err := resetCDFCount(&cdfs.CompInter[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.CompRefType {
		if err := resetCDFCount(&cdfs.CompRefType[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.SingleRef {
		for j := range cdfs.SingleRef[i] {
			if err := resetCDFCount(&cdfs.SingleRef[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.CompFwdRef {
		for j := range cdfs.CompFwdRef[i] {
			if err := resetCDFCount(&cdfs.CompFwdRef[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.CompBwdRef {
		for j := range cdfs.CompBwdRef[i] {
			if err := resetCDFCount(&cdfs.CompBwdRef[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.UniCompRef {
		for j := range cdfs.UniCompRef[i] {
			if err := resetCDFCount(&cdfs.UniCompRef[i][j]); err != nil {
				return err
			}
		}
	}
	return nil
}

func resetInterModeCDFCounts(cdfs *tile.InterModeCDFs) error {
	for i := range cdfs.NewMV {
		if err := resetCDFCount(&cdfs.NewMV[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.GlobalMV {
		if err := resetCDFCount(&cdfs.GlobalMV[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.RefMV {
		if err := resetCDFCount(&cdfs.RefMV[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.DRL {
		if err := resetCDFCount(&cdfs.DRL[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.Compound {
		if err := resetCDFCount(&cdfs.Compound[i]); err != nil {
			return err
		}
	}
	return nil
}

func resetMVCDFCounts(cdfs *tile.MVCDFs) error {
	if err := resetCDFCount(&cdfs.Joint); err != nil {
		return err
	}
	for i := range cdfs.Components {
		component := &cdfs.Components[i]
		if err := resetCDFCount(&component.Classes); err != nil {
			return err
		}
		for j := range component.Class0FP {
			if err := resetCDFCount(&component.Class0FP[j]); err != nil {
				return err
			}
		}
		if err := resetCDFCount(&component.FP); err != nil {
			return err
		}
		if err := resetCDFCount(&component.Sign); err != nil {
			return err
		}
		if err := resetCDFCount(&component.Class0HP); err != nil {
			return err
		}
		if err := resetCDFCount(&component.HP); err != nil {
			return err
		}
		if err := resetCDFCount(&component.Class0); err != nil {
			return err
		}
		for j := range component.Bits {
			if err := resetCDFCount(&component.Bits[j]); err != nil {
				return err
			}
		}
	}
	return nil
}

func resetCompoundBlendCDFCounts(cdfs *tile.CompoundBlendCDFs) error {
	for i := range cdfs.CompoundIndex {
		if err := resetCDFCount(&cdfs.CompoundIndex[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&cdfs.CompoundGroup[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.CompoundType {
		if err := resetCDFCount(&cdfs.CompoundType[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&cdfs.WedgeIndex[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&cdfs.WedgeInterIntra[i]); err != nil {
			return err
		}
	}
	for i := range cdfs.InterIntra {
		if err := resetCDFCount(&cdfs.InterIntra[i]); err != nil {
			return err
		}
		if err := resetCDFCount(&cdfs.InterIntraModeCDF[i]); err != nil {
			return err
		}
	}
	return nil
}

func resetTransformTypeCDFCounts(cdfs *tile.TransformTypeCDFs) error {
	for i := range cdfs.Intra {
		for j := range cdfs.Intra[i] {
			for k := range cdfs.Intra[i][j] {
				if err := resetCDFCount(&cdfs.Intra[i][j][k]); err != nil {
					return err
				}
			}
		}
	}
	for i := range cdfs.Inter {
		for j := range cdfs.Inter[i] {
			if err := resetCDFCount(&cdfs.Inter[i][j]); err != nil {
				return err
			}
		}
	}
	return nil
}

func resetCoeffCDFCounts(cdfs *tile.CoeffCDFs) error {
	for i := range cdfs.TXBSkip {
		for j := range cdfs.TXBSkip[i] {
			if err := resetCDFCount(&cdfs.TXBSkip[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.EOBExtra {
		for j := range cdfs.EOBExtra[i] {
			for k := range cdfs.EOBExtra[i][j] {
				if err := resetCDFCount(&cdfs.EOBExtra[i][j][k]); err != nil {
					return err
				}
			}
		}
	}
	for i := range cdfs.DCSign {
		for j := range cdfs.DCSign[i] {
			if err := resetCDFCount(&cdfs.DCSign[i][j]); err != nil {
				return err
			}
		}
	}
	for i := range cdfs.CoeffBR {
		for j := range cdfs.CoeffBR[i] {
			for k := range cdfs.CoeffBR[i][j] {
				if err := resetCDFCount(&cdfs.CoeffBR[i][j][k]); err != nil {
					return err
				}
			}
		}
	}
	for i := range cdfs.CoeffBase {
		for j := range cdfs.CoeffBase[i] {
			for k := range cdfs.CoeffBase[i][j] {
				if err := resetCDFCount(&cdfs.CoeffBase[i][j][k]); err != nil {
					return err
				}
			}
		}
	}
	for i := range cdfs.CoeffBaseEOB {
		for j := range cdfs.CoeffBaseEOB[i] {
			for k := range cdfs.CoeffBaseEOB[i][j] {
				if err := resetCDFCount(&cdfs.CoeffBaseEOB[i][j][k]); err != nil {
					return err
				}
			}
		}
	}
	for i := range cdfs.EOBFlag16 {
		for j := range cdfs.EOBFlag16[i] {
			if err := resetCDFCount(&cdfs.EOBFlag16[i][j]); err != nil {
				return err
			}
			if err := resetCDFCount(&cdfs.EOBFlag32[i][j]); err != nil {
				return err
			}
			if err := resetCDFCount(&cdfs.EOBFlag64[i][j]); err != nil {
				return err
			}
			if err := resetCDFCount(&cdfs.EOBFlag128[i][j]); err != nil {
				return err
			}
			if err := resetCDFCount(&cdfs.EOBFlag256[i][j]); err != nil {
				return err
			}
			if err := resetCDFCount(&cdfs.EOBFlag512[i][j]); err != nil {
				return err
			}
			if err := resetCDFCount(&cdfs.EOBFlag1024[i][j]); err != nil {
				return err
			}
		}
	}
	return nil
}
