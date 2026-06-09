package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// keyframe.go assembles a complete decodable LOSSLESS all-intra keyframe
// temporal unit. It is the first end-to-end encode path: the tile symbol stream
// mirrors the decoder's keyframe block loop exactly (tile/block_loop.go
// decodeBlockVisit order) using the round-trip-verified writers, and the
// headers come from this package's byte-verified OBU emitters.
//
// Scope (first milestone, extended next): 8-bit 4:2:0, 64x64 frame (one 64x64
// superblock, single tile), PARTITION_NONE, DC intra everywhere, qindex 0
// (lossless: WHT 4x4 transforms, recon == source so prediction neighbors come
// straight from the source plane).

// SourceFrame420 is one caller-owned 8-bit 4:2:0 source picture.
type SourceFrame420 struct {
	Y, U, V               []byte
	YStride, ChromaStride int
	Width, Height         int
}

// EncodeLosslessKeyframe64x64 encodes src (which must be exactly 64x64) as one
// low-overhead temporal unit: temporal delimiter, sequence header, complete
// lossless keyframe header, and a single-tile tile group carrying the coded
// blocks. The returned bytes decode in the goav1 decoder to a frame that is
// bit-exactly src.
func EncodeLosslessKeyframe64x64(src SourceFrame420) ([]byte, error) {
	if src.Width != 64 || src.Height != 64 {
		return nil, fmt.Errorf("encoder: only 64x64 frames supported, got %dx%d", src.Width, src.Height)
	}
	tilePayload, err := encodeLosslessKeyframeTile(src)
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}

	seq := losslessKeyframeSequence(src.Width, src.Height)
	header := losslessKeyframeHeader(src.Width, src.Height)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, fmt.Errorf("size header TU: %w", err)
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

	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, nil
}

func losslessKeyframeSequence(width, height int) SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        uint32(width),
		MaxFrameHeight:       uint32(height),
		Use128x128Superblock: false,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func losslessKeyframeHeader(width, height int) IntraFrameHeaderParams {
	return IntraFrameHeaderParams{
		Prefix: FrameHeaderPrefix{
			FrameType:          FrameHeaderTypeKey,
			ShowFrame:          true,
			ErrorResilientMode: true,
			ForceIntegerMV:     true, // inferred 1 for key/intra frames
			PrimaryRefFrame:    EncoderPrimaryRefNone,
		},
		Size: IntraFrameSize{
			UpscaledWidth:       uint32(width),
			Height:              uint32(height),
			RenderWidth:         uint32(width),
			RenderHeight:        uint32(height),
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		Tile: TileInfo{
			RefreshContext: true,
			SBCols:         1,
			SBRows:         1,
			Cols:           1,
			Rows:           1,
			ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
			RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
		},
		Quantization: QuantizationParams{BaseQIdx: 0},
		LoopFilter: LoopFilterParams{
			ModeRefDeltaEnabled: true,
			ModeRefDeltaUpdate:  true,
			Deltas:              defaultLoopFilterDeltas(),
		},
		TransformRef: TransformReferenceParams{
			TransformMode: TransformMode4x4Only,
			ReferenceMode: ReferenceModeSingle,
		},
		AllLossless: true,
	}
}

// encodeLosslessKeyframeTile codes the single 64x64-superblock tile: one
// PARTITION_NONE 64x64 DC-intra block whose residual is coded as 4x4 WHT
// transform blocks, mirroring the decoder's keyframe block loop symbol order.
func encodeLosslessKeyframeTile(src SourceFrame420) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	var modeCDFs tile.BlockModeCDFs
	var intraCDFs tile.IntraModeCDFs
	var coeffCDFs tile.CoeffCDFs
	if err := partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := modeCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := intraCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := coeffCDFs.InitDefault(0); err != nil {
		return nil, err
	}
	var partCtx tile.PartitionContext
	var modeCtx tile.BlockModeContext
	var coeffCtx tile.CoeffEntropyContext

	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := transform.FillDefaultScan(scan, inverse, transform.Size{Width: 4, Height: 4}, transform.Class2D); err != nil {
		return nil, err
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize4x4)
	if err != nil {
		return nil, err
	}
	levels := make([]uint8, scratchLen)

	w := entropy.NewWriter(make([]byte, 0, 1<<16))

	walkReq := tile.BlockWalkRequest{
		Root:     tile.BlockLevel64x64,
		MIColEnd: 16,
		MIRowEnd: 16,
	}
	decide := func(level tile.BlockLevel, ctx int, haveRight, haveBottom bool) (tile.Partition, error) {
		return tile.PartitionNone, nil
	}
	visit := func(block tile.BlockVisit) error {
		return encodeLosslessKeyframeBlock(&w, src, block,
			&modeCDFs, &intraCDFs, &coeffCDFs, &modeCtx, &coeffCtx, scan, levels)
	}
	if _, err := tile.WalkBlocksWrite(&w, &partCDFs, &partCtx, walkReq, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

// encodeLosslessKeyframeBlock codes one keyframe block in the decoder's
// decodeBlockVisit symbol order: skip_transform, mode-context mark, luma DC
// mode, chroma DC mode, then the residual transform blocks (all luma 4x4 TXBs
// in raster order, then U, then V — the <=64px single-unit residual order).
func encodeLosslessKeyframeBlock(w *entropy.Writer, src SourceFrame420, block tile.BlockVisit,
	modeCDFs *tile.BlockModeCDFs, intraCDFs *tile.IntraModeCDFs, coeffCDFs *tile.CoeffCDFs,
	modeCtx *tile.BlockModeContext, coeffCtx *tile.CoeffEntropyContext,
	scan []int16, levels []uint8) error {

	if block.Size != tile.BlockSize64x64 || block.MICol != 0 || block.MIRow != 0 {
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}

	// 1) skip_transform = 0 (residual present).
	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(w, modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	// 2) CDEF disabled (lossless), no symbol. 3) Mark the prefix exactly as the
	// decoder does after reading it.
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	// 4) deltas disabled, no symbols. 5) keyframe intra: no intrabc symbol
	// (allow_intrabc=0), luma mode then chroma mode. DC is non-directional so
	// no angle deltas; palette and filter-intra are gated off by the headers.
	if err := tile.WriteLumaIntraMode(w, intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
	}, tile.IntraModeDC); err != nil {
		return fmt.Errorf("luma mode: %w", err)
	}
	if err := tile.WriteChromaIntraMode(w, intraCDFs, tile.ChromaIntraModeRequest{
		Size: block.Size, LumaMode: tile.IntraModeDC, CFLAllowed: false,
	}, tile.ChromaIntraModeDC, tile.CFLAlphaResult{}); err != nil {
		return fmt.Errorf("chroma mode: %w", err)
	}

	// 6) residual: 4x4 WHT TXBs. Luma 16x16 grid raster, then U, then V.
	for ty := range 16 {
		for tx := range 16 {
			if err := encodeLosslessTXB(w, coeffCDFs, coeffCtx, tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: tile.BlockSize64x64,
				Size:       tile.TransformSize4x4,
				X4:         uint8(tx),
				Y4:         uint8(ty),
			}, src.Y, src.YStride, tx*4, ty*4, scan, levels); err != nil {
				return fmt.Errorf("luma txb (%d,%d): %w", tx, ty, err)
			}
		}
	}
	for plane := 1; plane <= 2; plane++ {
		data := src.U
		if plane == 2 {
			data = src.V
		}
		for ty := range 8 {
			for tx := range 8 {
				if err := encodeLosslessTXB(w, coeffCDFs, coeffCtx, tile.CoeffContextRequest{
					Plane:      uint8(plane),
					PlaneBlock: tile.BlockSize32x32,
					Size:       tile.TransformSize4x4,
					X4:         uint8(tx),
					Y4:         uint8(ty),
				}, data, src.ChromaStride, tx*4, ty*4, scan, levels); err != nil {
					return fmt.Errorf("chroma %d txb (%d,%d): %w", plane, tx, ty, err)
				}
			}
		}
	}
	return nil
}

// encodeLosslessTXB DC-predicts one 4x4 transform block from the source plane
// (lossless: reconstruction equals source, so source neighbors are the
// decoder's reconstructed neighbors), forward-WHT-transforms the residual,
// quantizes at qindex 0 (exact >>2 of the x4-scaled WHT output), and codes the
// coefficients through the carrier-context writer.
func encodeLosslessTXB(w *entropy.Writer, cdfs *tile.CoeffCDFs, ctx *tile.CoeffEntropyContext,
	ctxReq tile.CoeffContextRequest, plane []byte, stride, px, py int,
	scan []int16, levels []uint8) error {

	dc := dcPredict4x4(plane, stride, px, py)

	var residual [16]int16
	for r := range 4 {
		row := (py+r)*stride + px
		for c := range 4 {
			residual[r*4+c] = int16(plane[row+c]) - int16(dc)
		}
	}
	var wht [16]int32
	if err := transform.ForwardWHT4x4(wht[:], 4, residual[:], 4); err != nil {
		return err
	}
	var coeffs [16]int16
	for i, v := range wht {
		coeffs[i] = int16(v >> 2) // qindex 0: dequant 4 restores the x4 WHT scale
	}
	_, err := tile.WriteCoefficientsTXBWithContext(w, cdfs, ctx, ctxReq, transform.Class2D, coeffs[:], scan, levels)
	return err
}

// dcPredict4x4 is the decoder's DC predictor (prediction.dcPrediction) for one
// 4x4 block at pixel (px,py): the rounded mean of the 4 above and 4 left
// neighbors that exist, or 128 when neither edge does.
func dcPredict4x4(plane []byte, stride, px, py int) uint8 {
	sum := 0
	count := 0
	if py > 0 {
		row := (py-1)*stride + px
		for i := range 4 {
			sum += int(plane[row+i])
		}
		count += 4
	}
	if px > 0 {
		col := py*stride + px - 1
		for i := range 4 {
			sum += int(plane[col+i*stride])
		}
		count += 4
	}
	if count == 0 {
		return 128
	}
	return uint8((sum + count/2) / count)
}
