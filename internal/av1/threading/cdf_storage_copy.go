package threading

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// copyCDFStorage is *dst = *src written as one copy per CDF array (each a
// memmove) plus the single-CDF fields. TinyGo lowers the plain assignment of
// this ~56 KB struct, and of any member struct over 1 KB, through a heap
// temporary, which made every frame allocate several of them; copy does not.
// TestCopyCDFStorageCoversEveryField pins that every field is copied.
func copyCDFStorage(dst, src *FrameWorkTileResidualCDFStorage) {
	copy(dst.Partition.Partition[:], src.Partition.Partition[:])

	copy(dst.Mode.Skip[:], src.Mode.Skip[:])
	copy(dst.Mode.SkipMode[:], src.Mode.SkipMode[:])
	copy(dst.Mode.SegmentPred[:], src.Mode.SegmentPred[:])
	copy(dst.Mode.SegmentID[:], src.Mode.SegmentID[:])

	copyIntraModeCDFs(&dst.Intra, &src.Intra)

	copy(dst.InterRef.CompInter[:], src.InterRef.CompInter[:])
	copy(dst.InterRef.CompRefType[:], src.InterRef.CompRefType[:])
	copy(dst.InterRef.SingleRef[:], src.InterRef.SingleRef[:])
	copy(dst.InterRef.CompFwdRef[:], src.InterRef.CompFwdRef[:])
	copy(dst.InterRef.CompBwdRef[:], src.InterRef.CompBwdRef[:])
	copy(dst.InterRef.UniCompRef[:], src.InterRef.UniCompRef[:])

	copy(dst.InterMode.NewMV[:], src.InterMode.NewMV[:])
	copy(dst.InterMode.GlobalMV[:], src.InterMode.GlobalMV[:])
	copy(dst.InterMode.RefMV[:], src.InterMode.RefMV[:])
	copy(dst.InterMode.DRL[:], src.InterMode.DRL[:])
	copy(dst.InterMode.Compound[:], src.InterMode.Compound[:])

	dst.MV.Joint = src.MV.Joint
	copy(dst.MV.Components[:], src.MV.Components[:])
	dst.DV.Joint = src.DV.Joint
	copy(dst.DV.Components[:], src.DV.Components[:])

	copy(dst.Interp.Switchable[:], src.Interp.Switchable[:])

	copy(dst.Motion.MotionMode[:], src.Motion.MotionMode[:])
	copy(dst.Motion.OBMC[:], src.Motion.OBMC[:])

	copy(dst.Blend.CompoundIndex[:], src.Blend.CompoundIndex[:])
	copy(dst.Blend.CompoundGroup[:], src.Blend.CompoundGroup[:])
	copy(dst.Blend.CompoundType[:], src.Blend.CompoundType[:])
	copy(dst.Blend.WedgeIndex[:], src.Blend.WedgeIndex[:])
	copy(dst.Blend.InterIntra[:], src.Blend.InterIntra[:])
	copy(dst.Blend.WedgeInterIntra[:], src.Blend.WedgeInterIntra[:])
	copy(dst.Blend.InterIntraModeCDF[:], src.Blend.InterIntraModeCDF[:])

	copy(dst.Transform.TxSize[:], src.Transform.TxSize[:])
	copy(dst.Transform.TxPartition[:], src.Transform.TxPartition[:])

	copy(dst.TransformType.Intra[:], src.TransformType.Intra[:])
	copy(dst.TransformType.Inter[:], src.TransformType.Inter[:])

	dst.Coeff.CopyFrom(&src.Coeff)

	dst.DeltaQ = src.DeltaQ
	dst.DeltaLF = src.DeltaLF
	copy(dst.DeltaLFMulti[:], src.DeltaLFMulti[:])
	dst.RestorationSwitchable = src.RestorationSwitchable
	dst.RestorationWiener = src.RestorationWiener
	dst.RestorationSGRProj = src.RestorationSGRProj
}

func copyIntraModeCDFs(dst, src *tile.IntraModeCDFs) {
	copy(dst.Intra[:], src.Intra[:])
	dst.Intrabc = src.Intrabc
	copy(dst.YMode[:], src.YMode[:])
	copy(dst.KeyframeYMode[:], src.KeyframeYMode[:])
	copy(dst.UVMode[:], src.UVMode[:])
	copy(dst.AngleDelta[:], src.AngleDelta[:])
	dst.CFLSign = src.CFLSign
	copy(dst.CFLAlpha[:], src.CFLAlpha[:])
	copy(dst.FilterIntra[:], src.FilterIntra[:])
	dst.FilterIntraMode = src.FilterIntraMode
	copy(dst.PaletteYMode[:], src.PaletteYMode[:])
	copy(dst.PaletteUVMode[:], src.PaletteUVMode[:])
	copy(dst.PaletteYSize[:], src.PaletteYSize[:])
	copy(dst.PaletteUVSize[:], src.PaletteUVSize[:])
	copy(dst.PaletteYColorIndex[:], src.PaletteYColorIndex[:])
	copy(dst.PaletteUVColorIndex[:], src.PaletteUVColorIndex[:])
}

// defaultCDFStorage caches the default CDF storage per coefficient
// q-context; every other table in the storage is independent of the
// quantizer, so one entry per context covers all base q indexes.
var defaultCDFStorage [tile.CoeffQContexts]struct {
	once    sync.Once
	storage FrameWorkTileResidualCDFStorage
	err     error
}

func defaultTileResidualCDFStorage(baseQIndex uint8) (*FrameWorkTileResidualCDFStorage, error) {
	d := &defaultCDFStorage[tile.CoeffQContext(baseQIndex)]
	d.once.Do(func() { d.err = d.storage.buildDefault(baseQIndex) })
	return &d.storage, d.err
}
