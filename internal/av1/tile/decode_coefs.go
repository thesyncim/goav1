package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coefLevelsLen is dav1d's t->scratch.levels size.
const coefLevelsLen = entropy.CoefLevelsLen

// coefSizeTable is, per transform size, the TxfmInfo and scan dav1d's
// decode_coefs reads (entropy.(*Reader).DecodeCoefs).
var coefSizeTable [transformSizeCount]entropy.CoefSize

func init() {
	for size := range transformSizeCount {
		dims := &transformDimensions[size]
		geo := &coeffGeometryTable[size]
		if !geo.valid {
			panic("tile: transform size without coefficient geometry")
		}
		loCtx := uint8(0)
		if geo.txWide {
			loCtx = 1
		} else if geo.txTall {
			loCtx = 2
		}
		scan := coeffScanTable[size][transform.Class2D]
		slh := log2Coef[geo.scanHeight] - 2
		coefSizeTable[size] = entropy.CoefSize{
			Scan:   scan,
			Scan2D: entropy.BuildCoefScan2D(scan, slh, loCtx),
			Size:   uint8(size),
			W4:     dims.W4,
			H4:     dims.H4,
			Min:    dims.Min,
			Max:    dims.Max,
			Ctx:    geo.txCtx,
			Slw:    log2Coef[geo.scanWidth] - 2,
			Slh:    slh,
			LoCtx:  loCtx,
		}
	}
}

// log2Coef maps a coefficient scan dimension (4, 8, 16 or 32) to its log2.
var log2Coef = [33]uint8{4: 2, 8: 3, 16: 4, 32: 5}
