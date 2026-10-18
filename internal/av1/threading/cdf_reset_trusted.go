package threading

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// resetCDFCountsTrusted is restricted to private decoder-owned state after a
// successful decode with no caller hooks. Defaults and entropy adaptation
// preserve each CDF shape; unused CDFs stay zero. The public reset continues to
// validate writable Values slices. Keep this walk in step with ResetCDFCounts.
func (s *FrameWorkTileResidualCDFStorage) resetCDFCountsTrusted() {
	for i := range s.Partition.Partition {
		for j := range s.Partition.Partition[i] {
			entropy.ResetCDFCountTrusted(&s.Partition.Partition[i][j])
		}
	}
	for i := range s.Mode.Skip {
		entropy.ResetCDFCountTrusted(&s.Mode.Skip[i])
		entropy.ResetCDFCountTrusted(&s.Mode.SkipMode[i])
		entropy.ResetCDFCountTrusted(&s.Mode.SegmentPred[i])
		entropy.ResetCDFCountTrusted(&s.Mode.SegmentID[i])
	}
	resetIntraModeCDFCountsTrusted(&s.Intra)
	resetInterRefCDFCountsTrusted(&s.InterRef)
	resetInterModeCDFCountsTrusted(&s.InterMode)
	resetMVCDFCountsTrusted(&s.MV)
	for i := range s.Interp.Switchable {
		entropy.ResetCDFCountTrusted(&s.Interp.Switchable[i])
	}
	for i := range s.Motion.MotionMode {
		entropy.ResetCDFCountTrusted(&s.Motion.MotionMode[i])
		entropy.ResetCDFCountTrusted(&s.Motion.OBMC[i])
	}
	resetCompoundBlendCDFCountsTrusted(&s.Blend)
	for i := range s.Transform.TxSize {
		for j := range s.Transform.TxSize[i] {
			entropy.ResetCDFCountTrusted(&s.Transform.TxSize[i][j])
		}
	}
	for i := range s.Transform.TxPartition {
		for j := range s.Transform.TxPartition[i] {
			entropy.ResetCDFCountTrusted(&s.Transform.TxPartition[i][j])
		}
	}
	resetTransformTypeCDFCountsTrusted(&s.TransformType)
	resetCoeffCDFCountsTrusted(&s.Coeff)
	entropy.ResetCDFCountTrusted(&s.DeltaQ)
	entropy.ResetCDFCountTrusted(&s.DeltaLF)
	for i := range s.DeltaLFMulti {
		entropy.ResetCDFCountTrusted(&s.DeltaLFMulti[i])
	}
	entropy.ResetCDFCountTrusted(&s.RestorationSwitchable)
	entropy.ResetCDFCountTrusted(&s.RestorationWiener)
	entropy.ResetCDFCountTrusted(&s.RestorationSGRProj)
}

func resetIntraModeCDFCountsTrusted(cdfs *tile.IntraModeCDFs) {
	for i := range cdfs.Intra {
		entropy.ResetCDFCountTrusted(&cdfs.Intra[i])
	}
	entropy.ResetCDFCountTrusted(&cdfs.Intrabc)
	for i := range cdfs.YMode {
		entropy.ResetCDFCountTrusted(&cdfs.YMode[i])
	}
	for i := range cdfs.KeyframeYMode {
		for j := range cdfs.KeyframeYMode[i] {
			entropy.ResetCDFCountTrusted(&cdfs.KeyframeYMode[i][j])
		}
	}
	for i := range cdfs.UVMode {
		for j := range cdfs.UVMode[i] {
			entropy.ResetCDFCountTrusted(&cdfs.UVMode[i][j])
		}
	}
	for i := range cdfs.AngleDelta {
		entropy.ResetCDFCountTrusted(&cdfs.AngleDelta[i])
	}
	entropy.ResetCDFCountTrusted(&cdfs.CFLSign)
	for i := range cdfs.CFLAlpha {
		entropy.ResetCDFCountTrusted(&cdfs.CFLAlpha[i])
	}
	for i := range cdfs.FilterIntra {
		entropy.ResetCDFCountTrusted(&cdfs.FilterIntra[i])
	}
	entropy.ResetCDFCountTrusted(&cdfs.FilterIntraMode)
	for i := range cdfs.PaletteYMode {
		for j := range cdfs.PaletteYMode[i] {
			entropy.ResetCDFCountTrusted(&cdfs.PaletteYMode[i][j])
		}
	}
	for i := range cdfs.PaletteUVMode {
		entropy.ResetCDFCountTrusted(&cdfs.PaletteUVMode[i])
	}
	for i := range cdfs.PaletteYSize {
		entropy.ResetCDFCountTrusted(&cdfs.PaletteYSize[i])
		entropy.ResetCDFCountTrusted(&cdfs.PaletteUVSize[i])
	}
	for i := range cdfs.PaletteYColorIndex {
		for j := range cdfs.PaletteYColorIndex[i] {
			entropy.ResetCDFCountTrusted(&cdfs.PaletteYColorIndex[i][j])
			entropy.ResetCDFCountTrusted(&cdfs.PaletteUVColorIndex[i][j])
		}
	}
}

func resetInterRefCDFCountsTrusted(cdfs *tile.InterRefCDFs) {
	for i := range cdfs.CompInter {
		entropy.ResetCDFCountTrusted(&cdfs.CompInter[i])
	}
	for i := range cdfs.CompRefType {
		entropy.ResetCDFCountTrusted(&cdfs.CompRefType[i])
	}
	for i := range cdfs.SingleRef {
		for j := range cdfs.SingleRef[i] {
			entropy.ResetCDFCountTrusted(&cdfs.SingleRef[i][j])
		}
	}
	for i := range cdfs.CompFwdRef {
		for j := range cdfs.CompFwdRef[i] {
			entropy.ResetCDFCountTrusted(&cdfs.CompFwdRef[i][j])
		}
	}
	for i := range cdfs.CompBwdRef {
		for j := range cdfs.CompBwdRef[i] {
			entropy.ResetCDFCountTrusted(&cdfs.CompBwdRef[i][j])
		}
	}
	for i := range cdfs.UniCompRef {
		for j := range cdfs.UniCompRef[i] {
			entropy.ResetCDFCountTrusted(&cdfs.UniCompRef[i][j])
		}
	}
}

func resetInterModeCDFCountsTrusted(cdfs *tile.InterModeCDFs) {
	for i := range cdfs.NewMV {
		entropy.ResetCDFCountTrusted(&cdfs.NewMV[i])
	}
	for i := range cdfs.GlobalMV {
		entropy.ResetCDFCountTrusted(&cdfs.GlobalMV[i])
	}
	for i := range cdfs.RefMV {
		entropy.ResetCDFCountTrusted(&cdfs.RefMV[i])
	}
	for i := range cdfs.DRL {
		entropy.ResetCDFCountTrusted(&cdfs.DRL[i])
	}
	for i := range cdfs.Compound {
		entropy.ResetCDFCountTrusted(&cdfs.Compound[i])
	}
}

func resetMVCDFCountsTrusted(cdfs *tile.MVCDFs) {
	entropy.ResetCDFCountTrusted(&cdfs.Joint)
	for i := range cdfs.Components {
		component := &cdfs.Components[i]
		entropy.ResetCDFCountTrusted(&component.Classes)
		for j := range component.Class0FP {
			entropy.ResetCDFCountTrusted(&component.Class0FP[j])
		}
		entropy.ResetCDFCountTrusted(&component.FP)
		entropy.ResetCDFCountTrusted(&component.Sign)
		entropy.ResetCDFCountTrusted(&component.Class0HP)
		entropy.ResetCDFCountTrusted(&component.HP)
		entropy.ResetCDFCountTrusted(&component.Class0)
		for j := range component.Bits {
			entropy.ResetCDFCountTrusted(&component.Bits[j])
		}
	}
}

func resetCompoundBlendCDFCountsTrusted(cdfs *tile.CompoundBlendCDFs) {
	for i := range cdfs.CompoundIndex {
		entropy.ResetCDFCountTrusted(&cdfs.CompoundIndex[i])
		entropy.ResetCDFCountTrusted(&cdfs.CompoundGroup[i])
	}
	for i := range cdfs.CompoundType {
		entropy.ResetCDFCountTrusted(&cdfs.CompoundType[i])
		entropy.ResetCDFCountTrusted(&cdfs.WedgeIndex[i])
		entropy.ResetCDFCountTrusted(&cdfs.WedgeInterIntra[i])
	}
	for i := range cdfs.InterIntra {
		entropy.ResetCDFCountTrusted(&cdfs.InterIntra[i])
		entropy.ResetCDFCountTrusted(&cdfs.InterIntraModeCDF[i])
	}
}

func resetTransformTypeCDFCountsTrusted(cdfs *tile.TransformTypeCDFs) {
	for i := range cdfs.Intra {
		for j := range cdfs.Intra[i] {
			for k := range cdfs.Intra[i][j] {
				entropy.ResetCDFCountTrusted(&cdfs.Intra[i][j][k])
			}
		}
	}
	for i := range cdfs.Inter {
		for j := range cdfs.Inter[i] {
			entropy.ResetCDFCountTrusted(&cdfs.Inter[i][j])
		}
	}
}

func resetCoeffCDFCountsTrusted(cdfs *tile.CoeffCDFs) {
	for i := range cdfs.TXBSkip {
		for j := range cdfs.TXBSkip[i] {
			entropy.ResetCDFCountTrusted(&cdfs.TXBSkip[i][j])
		}
	}
	for i := range cdfs.EOBExtra {
		for j := range cdfs.EOBExtra[i] {
			for k := range cdfs.EOBExtra[i][j] {
				entropy.ResetCDFCountTrusted(&cdfs.EOBExtra[i][j][k])
			}
		}
	}
	for i := range cdfs.DCSign {
		for j := range cdfs.DCSign[i] {
			entropy.ResetCDFCountTrusted(&cdfs.DCSign[i][j])
		}
	}
	for i := range cdfs.CoeffBR {
		for j := range cdfs.CoeffBR[i] {
			for k := range cdfs.CoeffBR[i][j] {
				entropy.ResetCDFCountTrusted(&cdfs.CoeffBR[i][j][k])
			}
		}
	}
	for i := range cdfs.CoeffBase {
		for j := range cdfs.CoeffBase[i] {
			for k := range cdfs.CoeffBase[i][j] {
				entropy.ResetCDFCountTrusted(&cdfs.CoeffBase[i][j][k])
			}
		}
	}
	for i := range cdfs.CoeffBaseEOB {
		for j := range cdfs.CoeffBaseEOB[i] {
			for k := range cdfs.CoeffBaseEOB[i][j] {
				entropy.ResetCDFCountTrusted(&cdfs.CoeffBaseEOB[i][j][k])
			}
		}
	}
	for i := range cdfs.EOBFlag16 {
		for j := range cdfs.EOBFlag16[i] {
			entropy.ResetCDFCountTrusted(&cdfs.EOBFlag16[i][j])
			entropy.ResetCDFCountTrusted(&cdfs.EOBFlag32[i][j])
			entropy.ResetCDFCountTrusted(&cdfs.EOBFlag64[i][j])
			entropy.ResetCDFCountTrusted(&cdfs.EOBFlag128[i][j])
			entropy.ResetCDFCountTrusted(&cdfs.EOBFlag256[i][j])
			entropy.ResetCDFCountTrusted(&cdfs.EOBFlag512[i][j])
			entropy.ResetCDFCountTrusted(&cdfs.EOBFlag1024[i][j])
		}
	}
}
