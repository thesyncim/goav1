package tile

import "github.com/thesyncim/goav1/internal/av1/transform"

type BlockCoeffCDFs struct {
	Transform *TransformCDFs
	Coeff     *CoeffCDFs
}

type BlockCoeffScratch struct {
	Coeff LumaCoeffTreeScratch
	// txtpMap is dav1d's t->scratch.txtp_map.
	txtpMap [MaxBlockModeSlots][MaxBlockModeSlots]transform.Type
}

// BlockCoeffBlock is one transform block as the reconstruction consumes it:
// its plane, position and type, and the decode result.
type BlockCoeffBlock struct {
	Plane     uint8
	Block     TransformBlock
	Transform transform.Type

	Result TXBDecodeResult
}

// BlockCoeffResult is what the block loop keeps of a block's coefficient
// decode: its transform sizes.
type BlockCoeffResult struct {
	Tree TransformTreeResult
}

const (
	maxCoeffScanLen    = 32 * 32
	maxCoeffScratchLen = (32 + txPadHorizontal) * (32 + txPadHorizontal)
)

type LumaCoeffTreeScratch struct {
	// CoefLevels is dav1d's t->scratch.levels for decodeCoefTokens, cleared
	// per TXB by the reader.
	CoefLevels [coefLevelsLen]uint8
}
