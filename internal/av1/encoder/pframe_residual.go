package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// pframe_residual.go is the encoder's first real temporal-compression path: a
// P-frame of zero-motion GLOBALMV LAST blocks with coded residuals. Each 8x8
// block predicts from the reference reconstruction at the same position (the
// identity global motion needs no interpolation), transforms and quantizes the
// temporal residual exactly like the lossy keyframe path, and reconstructs
// through the decoder's own dequant + inverse transform. Motion estimation
// (NEWMV + subpel interpolation) extends this path next.

// EncodePFrame encodes src as an inter frame predicting from ref (the previous
// frame's reconstruction) at the given qindex, returning the temporal unit and
// the encoder reconstruction the decoder must reproduce exactly.
func EncodePFrame(src SourceFrame420, ref SourceFrame420, qIndex uint8) ([]byte, SourceFrame420, error) {
	if src.Width != ref.Width || src.Height != ref.Height {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: source %dx%d does not match reference %dx%d", src.Width, src.Height, ref.Width, ref.Height)
	}
	if src.Width <= 0 || src.Height <= 0 || src.Width%64 != 0 || src.Height%64 != 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: P-frame requires multiple-of-64 dimensions, got %dx%d", src.Width, src.Height)
	}
	if qIndex == 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: qindex 0 lossless inter coding is not supported")
	}
	recon := SourceFrame420{
		Y:            make([]byte, len(src.Y)),
		U:            make([]byte, len(src.U)),
		V:            make([]byte, len(src.V)),
		YStride:      src.YStride,
		ChromaStride: src.ChromaStride,
		Width:        src.Width,
		Height:       src.Height,
	}
	tilePayload, err := encodePFrameTile(src, ref, &recon, qIndex)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("encode tile: %w", err)
	}

	seq := losslessKeyframeSequence(src.Width, src.Height)
	header, refState := repeatPFrameHeader(src.Width, src.Height, qIndex)
	header.References = &refState

	out, err := assembleInterTU(seq, header, tilePayload)
	if err != nil {
		return nil, SourceFrame420{}, err
	}
	return out, recon, nil
}

// assembleInterTU wraps one coded tile into a TD + inter frame header + tile
// group temporal unit.
func assembleInterTU(seq SequenceHeader, header InterFrameHeaderParams, tilePayload []byte) ([]byte, error) {
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
	frameSize, err := LowOverheadInterFrameHeaderOBUSize(seq, header, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("size inter header: %w", err)
	}
	tdSize := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter})
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

func encodePFrameTile(src SourceFrame420, ref SourceFrame420, recon *SourceFrame420, qIndex uint8) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	var refCDFs tile.InterRefCDFs
	var interModeCDFs tile.InterModeCDFs
	if err := partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := refCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := interModeCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &lossyEncodeState{qIndex: qIndex, color: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}}
	if err := st.modeCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := st.intraCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := st.coeffCDFs.InitDefault(qIndex); err != nil {
		return nil, err
	}
	if err := st.txCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := st.mvCDFs.InitDefault(); err != nil {
		return nil, err
	}
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.Plane(plane))
		if err != nil {
			return nil, err
		}
		*dst = q
	}
	st.scan8 = make([]int16, 64)
	inverse8 := make([]int16, 64)
	if err := transform.FillDefaultScan(st.scan8, inverse8, transform.Size{Width: 8, Height: 8}, transform.Class2D); err != nil {
		return nil, err
	}
	st.scan4 = make([]int16, 16)
	inverse4 := make([]int16, 16)
	if err := transform.FillDefaultScan(st.scan4, inverse4, transform.Size{Width: 4, Height: 4}, transform.Class2D); err != nil {
		return nil, err
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize8x8)
	if err != nil {
		return nil, err
	}
	st.levels = make([]uint8, scratchLen)
	st.invScratch = make([]int32, 64)

	w := entropy.NewWriter(make([]byte, 0, 1<<18))
	st.w = &w

	miCols := uint16(src.Width / 4)
	miRows := uint16(src.Height / 4)
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
	decide := func(level tile.BlockLevel, ctx int, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}

	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodePBlock(src, ref, recon, block, scratch, &refCDFs, &interModeCDFs, refs, walkReq, miCols, miRows)
	}
	if err := tile.WalkBlockLoopWrite(&w, &partCDFs, &scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

// encodePBlock codes one 8x8 zero-motion inter block with residual: the inter
// mode symbols in the decoder's order, then the luma TX_8X8 residual (with the
// inter tx_type symbol after txb_skip) and two chroma TX_4X4 residuals against
// the reference reconstruction.
func (st *lossyEncodeState) encodePBlock(src, ref SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	refCDFs *tile.InterRefCDFs, interModeCDFs *tile.InterModeCDFs, refs tile.InterReferencesResult,
	walkReq tile.BlockWalkRequest, miCols, miRows uint16) error {

	if block.Size != tile.BlockSize8x8 {
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	// Motion estimation first: the skip decision needs the motion-compensated
	// residual. Full-pel, even offsets (chroma stays at integer positions).
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	dx, dy := fullPelDiamondSearch(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, 8)

	// Quantize all three transform blocks up front; a block whose residual
	// quantizes to zero everywhere is coded as skip (no residual symbols, the
	// reconstruction is the prediction itself).
	var lumaQ, uQ, vQ [64]int16
	lumaZero := st.prepareInterTXB(src.Y, ref.Y, src.YStride, lumaPX, lumaPY, 8, dx, dy, st.yQuant, &lumaQ)
	uZero := st.prepareInterTXB(src.U, ref.U, src.ChromaStride, lumaPX/2, lumaPY/2, 4, dx/2, dy/2, st.uQuant, &uQ)
	vZero := st.prepareInterTXB(src.V, ref.V, src.ChromaStride, lumaPX/2, lumaPY/2, 4, dx/2, dy/2, st.vQuant, &vQ)
	skip := lumaZero && uZero && vZero

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, skip); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{SkipTransform: skip}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	if err := tile.WriteIntraFlag(st.w, &st.intraCDFs, modeCtx, tile.IntraFlagRequest{
		FrameType: parser.FrameTypeInter,
		X4:        block.X4, Y4: block.Y4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
	}, false); err != nil {
		return fmt.Errorf("intra flag: %w", err)
	}
	if err := tile.WriteInterReferences(st.w, refCDFs, modeCtx, tile.InterReferenceRequest{
		Size:          block.Size,
		ReferenceMode: parser.ReferenceModeSingle,
		X4:            block.X4, Y4: block.Y4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
	}, refs); err != nil {
		return fmt.Errorf("references: %w", err)
	}

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
		HaveTopRight:   tile.BlockHasTopRight(16, block),
	}
	stack, err := modeCtx.BuildReferenceMVStack(stackReq)
	if err != nil {
		return fmt.Errorf("build ref mv stack: %w", err)
	}

	mode := tile.InterModeGlobalMV
	if dx != 0 || dy != 0 {
		mode = tile.InterModeNewMV
	}
	modeResult := tile.InterModeResult{Mode: mode}
	if err := tile.WriteSingleInterMode(st.w, interModeCDFs, stack.ModeContext, mode); err != nil {
		return fmt.Errorf("inter mode: %w", err)
	}
	drlReq, err := stack.Stack.DRLRequestForMode(modeResult)
	if err != nil {
		return fmt.Errorf("drl request: %w", err)
	}
	if err := tile.WriteDRLIndex(st.w, interModeCDFs, drlReq, 0); err != nil {
		return fmt.Errorf("drl: %w", err)
	}
	mv := motion.Vector{}
	if mode == tile.InterModeNewMV {
		mvRefs, err := stack.Stack.ResolveInterMVReferences(modeResult, 0, false, false)
		if err != nil {
			return fmt.Errorf("resolve mv references: %w", err)
		}
		mv = motion.Vector{Row: int16(dy * 8), Col: int16(dx * 8)}
		if err := tile.WriteMotionVector(st.w, &st.mvCDFs, mv, mvRefs.Residual[0], tile.MVSubpelLow); err != nil {
			return fmt.Errorf("motion vector: %w", err)
		}
	}

	hasChroma := true // 8x8 blocks at 4:2:0 carry chroma
	motionResult := tile.InterMotionResult{References: refs, Mode: modeResult}
	motionResult.MV[0] = mv
	if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), motionResult, hasChroma); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), refs, motion.InterpFilters{}); err != nil {
		return fmt.Errorf("mark inter filters: %w", err)
	}

	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma plane block: %w", err)
	}
	if skip {
		// No residual symbols; reset the coefficient entropy contexts per
		// plane as the decoder's residual path does for skip blocks, and the
		// reconstruction is the motion-compensated prediction.
		if err := coeffCtx.ResetBlock(0, block.Size, int(block.X4), int(block.Y4)); err != nil {
			return fmt.Errorf("reset luma coeff ctx: %w", err)
		}
		for plane := 1; plane <= 2; plane++ {
			if err := coeffCtx.ResetBlock(plane, chromaBlock, int(block.X4)/2, int(block.Y4)/2); err != nil {
				return fmt.Errorf("reset chroma %d coeff ctx: %w", plane, err)
			}
		}
		copyPredBlock(recon.Y, ref.Y, src.YStride, lumaPX, lumaPY, 8, dx, dy)
		copyPredBlock(recon.U, ref.U, src.ChromaStride, lumaPX/2, lumaPY/2, 4, dx/2, dy/2)
		copyPredBlock(recon.V, ref.V, src.ChromaStride, lumaPX/2, lumaPY/2, 4, dx/2, dy/2)
		return nil
	}

	// Residual: luma TX_8X8 with the inter tx_type symbol, then chroma TX_4X4.
	txTypeReq := tile.InterTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		QIndexKnown: true,
		QIndex:      st.qIndex,
	}
	afterSkip := func() error {
		return tile.WriteInterTransformType(st.w, &st.txCDFs, txTypeReq, transform.TypeDCTDCT)
	}
	if err := st.finishInterTXB(recon.Y, ref.Y, src.YStride, lumaPX, lumaPY, 8, dx, dy, st.yQuant, &lumaQ, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, st.scan8, afterSkip); err != nil {
		return fmt.Errorf("luma txb: %w", err)
	}
	for plane := 1; plane <= 2; plane++ {
		rdata, refData, qc := recon.U, ref.U, &uQ
		q := st.uQuant
		if plane == 2 {
			rdata, refData, qc = recon.V, ref.V, &vQ
			q = st.vQuant
		}
		if err := st.finishInterTXB(rdata, refData, src.ChromaStride, lumaPX/2, lumaPY/2, 4, dx/2, dy/2, q, qc, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       tile.TransformSize4x4,
			X4:         block.X4 / 2,
			Y4:         block.Y4 / 2,
		}, coeffCtx, st.scan4, nil); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
	}
	return nil
}

// copyPredBlock copies the motion-compensated prediction into the recon plane
// for skipped blocks.
func copyPredBlock(reconPlane, refPlane []byte, stride, px, py, n, dx, dy int) {
	for r := range n {
		copy(reconPlane[(py+r)*stride+px:(py+r)*stride+px+n], refPlane[(py+r+dy)*stride+px+dx:(py+r+dy)*stride+px+dx+n])
	}
}

// prepareInterTXB computes the quantized coefficients of one square n x n
// motion-compensated residual block and reports whether they are all zero.
func (st *lossyEncodeState) prepareInterTXB(srcPlane, refPlane []byte, stride, px, py, n, dx, dy int, q quantize.Quantizer, qcoeff *[64]int16) bool {
	var residual [64]int16
	for r := range n {
		row := (py+r)*stride + px
		refRow := (py+r+dy)*stride + px + dx
		for c := range n {
			residual[r*n+c] = int16(srcPlane[row+c]) - int16(refPlane[refRow+c])
		}
	}
	var tran [64]int32
	switch n {
	case 4:
		if err := transform.ForwardDCT4x4(tran[:16], 4, residual[:16], 4); err != nil {
			return false
		}
	case 8:
		if err := transform.ForwardDCT8x8(tran[:64], 8, residual[:64], 8); err != nil {
			return false
		}
	default:
		return false
	}
	if err := quantize.QuantizeBlockScaled(qcoeff[:n*n], n, tran[:n*n], n, n, n, q, 0); err != nil {
		return false
	}
	for _, v := range qcoeff[:n*n] {
		if v != 0 {
			return false
		}
	}
	return true
}

// finishInterTXB codes the prepared coefficients and rebuilds the recon block
// through the decoder's dequant + inverse transform.
func (st *lossyEncodeState) finishInterTXB(reconPlane, refPlane []byte, stride, px, py, n, dx, dy int, q quantize.Quantizer, qcoeff *[64]int16,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error) error {

	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff[:n*n], scan, st.levels, afterSkip); err != nil {
		return err
	}
	var dq [64]int32
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n*n], n, qcoeff[:n*n], n, n, n, q, 0, 8); err != nil {
		return err
	}
	var res [64]int16
	if err := transform.InverseDCTBlock(res[:n*n], n, dq[:n*n], n, st.invScratch[:n*n], transform.Size{Width: uint8(n), Height: uint8(n)}); err != nil {
		return err
	}
	for r := range n {
		row := (py+r)*stride + px
		refRow := (py+r+dy)*stride + px + dx
		for c := range n {
			v := int(refPlane[refRow+c]) + int(res[r*n+c])
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			reconPlane[row+c] = uint8(v)
		}
	}
	return nil
}

// fullPelDiamondSearch finds the even full-pel offset (dx, dy) within an 8px
// window that minimizes the luma SAD of the n x n block at (px, py) against
// the reference plane, keeping the offset window fully inside the frame. Even
// offsets keep chroma prediction at integer positions at 4:2:0. A small
// diamond refinement around the best raster candidate keeps the search cheap.
func fullPelDiamondSearch(src, ref []byte, stride, width, height, px, py, n int) (int, int) {
	sad := func(dx, dy int) int {
		total := 0
		for r := range n {
			row := (py+r)*stride + px
			refRow := (py+r+dy)*stride + px + dx
			for c := range n {
				d := int(src[row+c]) - int(ref[refRow+c])
				if d < 0 {
					d = -d
				}
				total += d
			}
		}
		return total
	}
	clampLo := func(v, lo int) int {
		if v < lo {
			return lo
		}
		return v
	}
	clampHi := func(v, hi int) int {
		if v > hi {
			return hi
		}
		return v
	}
	minDX := clampLo(-8, -px)
	maxDX := clampHi(8, width-n-px)
	minDY := clampLo(-8, -py)
	maxDY := clampHi(8, height-n-py)

	bestDX, bestDY := 0, 0
	bestSAD := sad(0, 0)
	// Coarse even-step raster, then a +-2 even diamond refinement.
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		for dx := minDX &^ 1; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sad(dx, dy); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	for _, cand := range [4][2]int{{bestDX + 2, bestDY}, {bestDX - 2, bestDY}, {bestDX, bestDY + 2}, {bestDX, bestDY - 2}} {
		dx, dy := cand[0], cand[1]
		if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
			continue
		}
		if s := sad(dx, dy); s < bestSAD {
			bestSAD, bestDX, bestDY = s, dx, dy
		}
	}
	return bestDX, bestDY
}
