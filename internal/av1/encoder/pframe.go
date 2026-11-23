package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// pframe.go assembles the encoder's first inter frame: a "repeat" P-frame in
// which every block is a skipped single-reference GLOBALMV block predicting
// from LAST with the identity global motion (zero MV), so the decoded frame
// equals the reference reconstruction exactly. The per-block symbol order and
// context marks mirror the decoder's inter path in tile/block_loop.go
// decodeBlockVisit: skip_transform, prefix mark, is-inter flag, references,
// the shared reference-MV stack for the mode context, the inter mode cascade,
// DRL (no symbols for GLOBALMV), the inter motion/filter marks, and the
// skipped block's coefficient entropy-context reset.
//
// Scope: 8-bit 4:2:0, dimensions multiples of 64, one 64x64 block per
// superblock, single tile, LAST-only references, no post-filters.

// repeatPFrameHeader builds the inter frame header plus the reference state the
// decoder holds after the keyframe (RefreshFrameFlags 0xff seeded every slot).
// interTileInfo builds the inter-frame tile layout with the requested number
// of uniform tile columns (log2), clamped to the legal range for the frame
// size. Multiple columns let the encoder run one entropy coder per tile in
// parallel; each tile starts from the frame-initial CDFs exactly as the
// decoder does.
func interTileInfo(width, height int, log2Cols uint8) (TileInfo, error) {
	return interTileInfoForSequence(losslessKeyframeSequence(width, height), width, height, log2Cols)
}

func interTileInfoForSequence(seq SequenceHeader, width, height int, log2Cols uint8) (TileInfo, error) {
	derived, err := deriveEncoderTileInfo(seq, uint32(width), uint32(height), TileInfo{})
	if err != nil {
		return TileInfo{}, err
	}
	if log2Cols < derived.MinLog2Cols {
		log2Cols = derived.MinLog2Cols
	}
	if log2Cols > derived.MaxLog2Cols {
		log2Cols = derived.MaxLog2Cols
	}
	tiles := TileInfo{
		RefreshContext:      true,
		UniformSpacing:      true,
		SBCols:              derived.SBCols,
		SBRows:              derived.SBRows,
		MinLog2Cols:         derived.MinLog2Cols,
		MaxLog2Cols:         derived.MaxLog2Cols,
		MaxLog2Rows:         derived.MaxLog2Rows,
		MinLog2Tiles:        derived.MinLog2Tiles,
		Log2Cols:            log2Cols,
		InterpolationFilter: InterpolationEightTap,
	}
	minRows := uint8(0)
	if derived.MinLog2Tiles > tiles.Log2Cols {
		minRows = derived.MinLog2Tiles - tiles.Log2Cols
	}
	tiles.MinLog2Rows = minRows
	tiles.Log2Rows = minRows
	fillUniformTileStarts(&tiles, derived)
	if tiles.Log2Cols != 0 || tiles.Log2Rows != 0 {
		tiles.TileSizeBytes = 4
	}
	return tiles, nil
}

func repeatPFrameHeader(width, height int, qIndex uint8, refreshFlags uint8) (InterFrameHeaderParams, parser.ReferenceState) {
	refs := referenceStateForFrame(width, height)
	base := losslessKeyframeHeader(width, height)
	tiles := base.Tile
	tiles.InterpolationFilter = InterpolationEightTap
	allLossless := qIndex == 0
	header := InterFrameHeaderParams{
		Prefix: FrameHeaderPrefix{
			FrameType:          FrameHeaderTypeInter,
			ShowFrame:          true,
			ShowableFrame:      true,
			ErrorResilientMode: true,
			PrimaryRefFrame:    EncoderPrimaryRefNone, // default CDFs each frame
		},
		Size: InterFrameSize{
			UpscaledWidth:       uint32(width),
			Height:              uint32(height),
			RenderWidth:         uint32(width),
			RenderHeight:        uint32(height),
			SuperResDenominator: 8,
			RefreshFrameFlags:   refreshFlags,
			// GOLDEN names slot 1 (a periodic anchor the streaming encoder
			// refreshes); every other name stays on slot 0 (LAST).
			RefFrameIdx: [7]uint8{0, 0, 0, 1, 0, 0, 0},
		},
		Tile:         tiles,
		Quantization: QuantizationParams{BaseQIdx: qIndex},
		LoopFilter: LoopFilterParams{
			Deltas: defaultLoopFilterDeltas(),
		},
		TransformRef: TransformReferenceParams{
			TransformMode: TransformModeSwitchable,
			ReferenceMode: ReferenceModeSingle,
		},
		FrameMode:    FrameModeParams{},
		GlobalMotion: DefaultGlobalMotionParams(),
		CDEF:         CDEFParams{Damping: 3},
		AllLossless:  allLossless,
	}
	if allLossless {
		header.LoopFilter.ModeRefDeltaEnabled = true
		header.LoopFilter.ModeRefDeltaUpdate = true
		header.TransformRef.TransformMode = TransformMode4x4Only
		header.CDEF = CDEFParams{}
	}
	return header, refs
}

func referenceStateForFrame(width, height int) parser.ReferenceState {
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := range parser.RefFrames {
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			GlobalMotion: defaultGlobal,
			Size: parser.FrameSize{
				CodedWidth:          uint32(width),
				UpscaledWidth:       uint32(width),
				Height:              uint32(height),
				RenderWidth:         uint32(width),
				RenderHeight:        uint32(height),
				SuperResDenominator: 8,
			},
		}
	}
	return refs
}
