package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
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

// EncodeRepeatPFrame encodes a P-frame that reproduces the previous frame's
// reconstruction bit for bit. width/height must match the reference frame.
func EncodeRepeatPFrame(width, height int, qIndex uint8) ([]byte, error) {
	if width <= 0 || height <= 0 || width%8 != 0 || height%8 != 0 {
		return nil, fmt.Errorf("encoder: repeat P-frame requires multiple-of-8 dimensions, got %dx%d", width, height)
	}
	tilePayload, err := encodeRepeatPFrameTile(width, height)
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}

	seq := losslessKeyframeSequence(width, height)
	header, refs := repeatPFrameHeader(width, height, qIndex, 0x01)
	header.References = &refs

	tdSize := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter})
	frameSize, err := LowOverheadInterFrameHeaderOBUSize(seq, header, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("size inter header: %w", err)
	}
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, err
	}

	out := make([]byte, 0, tdSize+frameSize+groupOBUSize)
	out, err = AppendLowOverheadOBU(out, OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		return nil, err
	}
	out, err = AppendLowOverheadInterFrameHeaderOBU(out, seq, header, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("append inter header: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, nil
}

// repeatPFrameHeader builds the inter frame header plus the reference state the
// decoder holds after the keyframe (RefreshFrameFlags 0xff seeded every slot).
// interTileInfo builds the inter-frame tile layout with the requested number
// of uniform tile columns (log2), clamped to the legal range for the frame
// size. Multiple columns let the encoder run one entropy coder per tile in
// parallel; each tile starts from the frame-initial CDFs exactly as the
// decoder does.
func interTileInfo(width, height int, log2Cols uint8) (TileInfo, error) {
	seq := losslessKeyframeSequence(width, height)
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
	base := losslessKeyframeHeader(width, height)
	tiles := base.Tile
	tiles.InterpolationFilter = InterpolationEightTap
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
	}
	return header, refs
}

// encodeRepeatPFrameTile codes one skipped GLOBALMV LAST block per superblock.
func encodeRepeatPFrameTile(width, height int) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	var modeCDFs tile.BlockModeCDFs
	var intraCDFs tile.IntraModeCDFs
	var refCDFs tile.InterRefCDFs
	var interModeCDFs tile.InterModeCDFs
	if err := partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := modeCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := intraCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := refCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := interModeCDFs.InitDefault(); err != nil {
		return nil, err
	}

	color := parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	w := entropy.NewWriter(make([]byte, 0, 1<<14))

	miCols := uint16(width / 4)
	miRows := uint16(height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miCols) + sbSizeMIB - 1) / sbSizeMIB

	var scratch tile.BlockLoopScratch
	carrier := &tile.BlockLoopContextCarrier{
		Above: make([]tile.BlockLoopRootAboveContext, rootCols),
	}
	walkReq := tile.BlockWalkRequest{
		Root:     tile.BlockLevel64x64,
		MIColEnd: miCols,
		MIRowEnd: miRows,
	}
	// All blocks 8x8, matching the residual P-frame path: every leaf fits
	// fully inside multiple-of-8 frames.
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}

	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		modeCtx := &scratch.Mode

		// 1) skip_transform = 1 (no residual; skip_mode is disabled so no
		// symbol precedes it).
		prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
		if err := tile.WriteSkipTransform(&w, &modeCDFs, modeCtx, prefixReq, false, true); err != nil {
			return fmt.Errorf("skip: %w", err)
		}
		if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{SkipTransform: true}); err != nil {
			return fmt.Errorf("mark prefix: %w", err)
		}

		// 2) is-inter flag, then the LAST single reference.
		if err := tile.WriteIntraFlag(&w, &intraCDFs, modeCtx, tile.IntraFlagRequest{
			FrameType: parser.FrameTypeInter,
			X4:        block.X4, Y4: block.Y4,
			HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		}, false); err != nil {
			return fmt.Errorf("intra flag: %w", err)
		}
		if err := tile.WriteInterReferences(&w, &refCDFs, modeCtx, tile.InterReferenceRequest{
			Size:          block.Size,
			ReferenceMode: parser.ReferenceModeSingle,
			X4:            block.X4, Y4: block.Y4,
			HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		}, refs); err != nil {
			return fmt.Errorf("references: %w", err)
		}

		// 3) the shared reference-MV stack supplies the mode context exactly
		// as the decoder derives it.
		stackReq := tile.ReferenceMVStackRequest{
			MICol:          block.MICol,
			MIRow:          block.MIRow,
			TileMIColStart: walkReq.MIColStart,
			TileMIRowStart: walkReq.MIRowStart,
			TileMIColEnd:   walkReq.MIColEnd,
			TileMIRowEnd:   walkReq.MIRowEnd,
			FrameMIRows:    miRows,
			FrameMICols:    miCols,
			Size:           block.Size,
			References:     refs,
			X4:             block.X4,
			Y4:             block.Y4,
			HaveTop:        block.HaveTop,
			HaveLeft:       block.HaveLeft,
			HaveTopRight:   tile.BlockHasTopRight(sbSizeMIB, block),
		}
		stack, err := modeCtx.BuildReferenceMVStack(stackReq)
		if err != nil {
			return fmt.Errorf("build ref mv stack: %w", err)
		}
		modeResult := tile.InterModeResult{Mode: tile.InterModeGlobalMV}
		if err := tile.WriteSingleInterMode(&w, &interModeCDFs, stack.ModeContext, tile.InterModeGlobalMV); err != nil {
			return fmt.Errorf("inter mode: %w", err)
		}
		drlReq, err := stack.Stack.DRLRequestForMode(modeResult)
		if err != nil {
			return fmt.Errorf("drl request: %w", err)
		}
		if err := tile.WriteDRLIndex(&w, &interModeCDFs, drlReq, 0); err != nil {
			return fmt.Errorf("drl: %w", err)
		}
		// 4) GLOBALMV with identity global motion: zero MV, no residual
		// symbols, fixed EIGHTTAP filters (non-switchable, no symbols).

		// 5) context marks, mirroring the decoder's motion-decoded path.
		hasChroma := true // 8x8 blocks at 4:2:0 always carry chroma
		motionResult := tile.InterMotionResult{
			References: refs,
			Mode:       modeResult,
		}
		if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), motionResult, hasChroma); err != nil {
			return fmt.Errorf("mark inter motion: %w", err)
		}
		if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), refs, motion.InterpFilters{}); err != nil {
			return fmt.Errorf("mark inter filters: %w", err)
		}

		// 6) skipped block: reset the coefficient entropy contexts per plane
		// exactly as the decoder's residual path does for skip blocks.
		if err := scratch.CoeffCtx.ResetBlock(0, block.Size, int(block.X4), int(block.Y4)); err != nil {
			return fmt.Errorf("reset luma coeff ctx: %w", err)
		}
		chromaBlock, err := tile.PlaneBlockSize(block.Size, color, 1)
		if err != nil {
			return err
		}
		for plane := 1; plane <= 2; plane++ {
			if err := scratch.CoeffCtx.ResetBlock(plane, chromaBlock, int(block.X4)/2, int(block.Y4)/2); err != nil {
				return fmt.Errorf("reset chroma %d coeff ctx: %w", plane, err)
			}
		}
		return nil
	}
	if err := tile.WalkBlockLoopWrite(&w, &partCDFs, &scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}
