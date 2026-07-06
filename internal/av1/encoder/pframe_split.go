package encoder

// pframe_split.go implements the decision/write split for the lossy P-frame
// tile coder, the libaom two-phase encode shape:
//
//   - av1/encoder/encodeframe.c encode_sb(): mode search, prediction,
//     transform/quantize, and reconstruction run first and store per-block
//     MB_MODE_INFO + quantized coefficients without touching the bitstream;
//   - av1/encoder/bitstream.c write_modes()/av1_write_coeffs_txb(): a serial
//     walk over the stored mode-info grid replays the decisions through the
//     adaptive-CDF symbol writers.
//
// SVT-AV1 splits identically (MODE DECISION / ENC-DEC over EncDecSegments vs
// the serial per-tile ENTROPY CODING stage, Source/Lib/Codec
// enc_dec_process.c / entropy_coding_process.c). The adaptive CDFs make the
// WRITE order-serial; everything before it is order-free given the neighbor
// state, which is what the SB-row wavefront (stage b) exploits.
//
// The decision pass records, per leaf block, exactly what libaom saves for
// pack_bs: the mode/reference/motion decision, the resolved ref-MV outputs
// (MB_MODE_INFO_EXT_FRAME: mode_context + the DRL/ref-MV data, saved by
// av1_copy_mbmi_ext_to_mbmi_ext_frame), the transform plan, and the quantized
// coefficients (the token buffer of av1/encoder/tokenize.c in coefficient
// form). The write pass replays them through the existing tile writers and
// re-marks the mode/coefficient contexts exactly as the fused path did, so
// the produced tile payload is byte-identical to the fused single-pass coder.

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	pblockInter uint8 = iota
	pblockIntra
)

// pblockRecord is one leaf block's stored decision, the goav1 equivalent of
// libaom's MB_MODE_INFO + MB_MODE_INFO_EXT_FRAME for the pack phase.
type pblockRecord struct {
	kind    uint8
	skip    bool
	splitTX bool

	refs        tile.InterReferencesResult
	modeResult  tile.InterModeResult
	drlIndex    uint8
	modeContext uint16
	drlReq      tile.DRLRequest
	mvRefs      tile.InterMVReferenceSet
	motion      tile.InterMotionResult
	filters     motion.InterpFilters
	txType      transform.Type
	txPlan      realtimeInterTXPlan

	intraMode  tile.IntraMode
	angleDelta int8
}

// pframeSplitRecord is the preallocated frame-level decision store for one
// tile: leaf-block records, the partition decisions in walk order, and the
// quantized-coefficient arena, all reused across frames (capacity is the
// tile's worst case, so the steady state allocates nothing).
type pframeSplitRecord struct {
	blocks []pblockRecord
	parts  []tile.Partition
	coeffs []int16

	blockCur int
	partCur  int
	coeffCur int
}

// reset clears the record for a new frame, reserving worst-case capacity for
// a tile of tileMI4W x tileMI4H MI units so the hot path never reallocates:
// one record per 8x8 leaf, the full interior partition tree, and one
// coefficient per pixel across all planes (coeff counts never exceed pixel
// counts; 4:2:0 uses 1.5x luma, other samplings up to 3x).
func (rec *pframeSplitRecord) reset(tileMI4W, tileMI4H int, color parser.ColorConfig) {
	// True upper bounds, not content estimates: the record must never grow on
	// a detailed frame after a black Prewarm frame under-fills it (that growth
	// would allocate on the first live frame). A 64x64 superblock's partition
	// walk visits at most 1+4+16+64 = 85 square nodes (down to 4x4), and its
	// leaves are at most one per 8x8 mode-info quad.
	maxBlocks := ((tileMI4W + 1) / 2) * ((tileMI4H + 1) / 2)
	sbs := ((tileMI4W + 15) / 16) * ((tileMI4H + 15) / 16)
	maxParts := sbs*85 + 8
	lumaPix := tileMI4W * 4 * tileMI4H * 4
	maxCoeffs := lumaPix
	if !color.MonoChrome {
		chroma := lumaPix
		if color.SubsamplingX {
			chroma /= 2
		}
		if color.SubsamplingY {
			chroma /= 2
		}
		maxCoeffs += 2 * chroma
	}
	if cap(rec.blocks) < maxBlocks {
		rec.blocks = make([]pblockRecord, 0, maxBlocks)
	}
	if cap(rec.parts) < maxParts {
		rec.parts = make([]tile.Partition, 0, maxParts)
	}
	if cap(rec.coeffs) < maxCoeffs {
		rec.coeffs = make([]int16, 0, maxCoeffs)
	}
	rec.blocks = rec.blocks[:0]
	rec.parts = rec.parts[:0]
	rec.coeffs = rec.coeffs[:0]
	rec.rewind()
}

// rewind restarts the replay cursors for the write pass.
func (rec *pframeSplitRecord) rewind() {
	rec.blockCur = 0
	rec.partCur = 0
	rec.coeffCur = 0
}

func (rec *pframeSplitRecord) appendBlock() *pblockRecord {
	if len(rec.blocks) < cap(rec.blocks) {
		rec.blocks = rec.blocks[:len(rec.blocks)+1]
	} else {
		rec.blocks = append(rec.blocks, pblockRecord{})
	}
	r := &rec.blocks[len(rec.blocks)-1]
	*r = pblockRecord{}
	return r
}

func (rec *pframeSplitRecord) nextBlock() (*pblockRecord, error) {
	if rec.blockCur >= len(rec.blocks) {
		return nil, fmt.Errorf("encoder: split replay ran past %d recorded blocks", len(rec.blocks))
	}
	r := &rec.blocks[rec.blockCur]
	rec.blockCur++
	return r, nil
}

func (rec *pframeSplitRecord) appendPart(p tile.Partition) {
	rec.parts = append(rec.parts, p)
}

func (rec *pframeSplitRecord) nextPart() (tile.Partition, error) {
	if rec.partCur >= len(rec.parts) {
		return 0, fmt.Errorf("encoder: split replay ran past %d recorded partitions", len(rec.parts))
	}
	p := rec.parts[rec.partCur]
	rec.partCur++
	return p, nil
}

// grabCoeffs reserves the next n coefficients of the arena for the decision
// pass to fill. Growth beyond the reserved worst case extends capacity (cold
// path only).
func (rec *pframeSplitRecord) grabCoeffs(n int) []int16 {
	off := len(rec.coeffs)
	for cap(rec.coeffs) < off+n {
		rec.coeffs = append(rec.coeffs[:cap(rec.coeffs)], 0)
	}
	rec.coeffs = rec.coeffs[:off+n]
	return rec.coeffs[off : off+n]
}

// nextCoeffs consumes the next n coefficients during write replay.
func (rec *pframeSplitRecord) nextCoeffs(n int) ([]int16, error) {
	if rec.coeffCur+n > len(rec.coeffs) {
		return nil, fmt.Errorf("encoder: split replay ran past %d recorded coefficients", len(rec.coeffs))
	}
	c := rec.coeffs[rec.coeffCur : rec.coeffCur+n]
	rec.coeffCur += n
	return c, nil
}

// pframeBlockDims returns the pixel dimensions of a coded P-frame leaf block.
func pframeBlockDims(size tile.BlockSize) (int, int, bool) {
	switch size {
	case tile.BlockSize8x8:
		return 8, 8, true
	case tile.BlockSize16x16:
		return 16, 16, true
	case tile.BlockSize32x32:
		return 32, 32, true
	case tile.BlockSize64x64:
		return 64, 64, true
	case tile.BlockSize64x32:
		return 64, 32, true
	case tile.BlockSize64x16:
		return 64, 16, true
	case tile.BlockSize32x64:
		return 32, 64, true
	case tile.BlockSize16x64:
		return 16, 64, true
	case tile.BlockSize16x8:
		return 16, 8, true
	case tile.BlockSize8x16:
		return 8, 16, true
	case tile.BlockSize32x16:
		return 32, 16, true
	case tile.BlockSize16x32:
		return 16, 32, true
	}
	return 0, 0, false
}

// interLumaTXAndScan mirrors the fused residual phase's largest-TX selection.
func (st *lossyEncodeState) interLumaTXAndScan(bw, bh int) (tile.TransformSize, []int16) {
	lumaTX, lumaScan := tile.TransformSize8x8, st.scan8
	switch {
	case bw == 16 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize16x16, st.scan16
	case bw == 32 && bh == 32:
		lumaTX, lumaScan = tile.TransformSize32x32, st.scan32
	case bw == 16 && bh == 8:
		lumaTX, lumaScan = tile.TransformSize16x8, st.scan16x8
	case bw == 8 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize8x16, st.scan8x16
	case bw == 32 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize32x16, st.scan32x16
	case bw == 16 && bh == 32:
		lumaTX, lumaScan = tile.TransformSize16x32, st.scan16x32
	case bw == 64 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize64x16, st.scan64x16
	case bw == 16 && bh == 64:
		lumaTX, lumaScan = tile.TransformSize16x64, st.scan16x64
	}
	return lumaTX, lumaScan
}

func (st *lossyEncodeState) interChromaTXAndScan(blockSize tile.BlockSize) (tile.TransformSize, []int16, error) {
	chromaTX, err := tile.MaxTransformSize(blockSize, st.color, 1)
	if err != nil {
		return 0, nil, fmt.Errorf("chroma transform size: %w", err)
	}
	chromaScan, ok := st.scanForTransformSize(chromaTX)
	if !ok {
		chromaScan, ok = st.scanForThinTransformSize(chromaTX)
	}
	if !ok {
		return 0, nil, fmt.Errorf("encoder: unsupported chroma transform %d", chromaTX)
	}
	return chromaTX, chromaScan, nil
}

func (st *lossyEncodeState) scanForLeafTX(leafTX tile.TransformSize) ([]int16, error) {
	scan, ok := st.scanForTransformSize(leafTX)
	if !ok {
		scan, ok = st.scanForThinTransformSize(leafTX)
	}
	if !ok {
		return nil, fmt.Errorf("encoder: unsupported realtime luma transform %d", leafTX)
	}
	return scan, nil
}

// interTXBResult derives the writer's TXB outcome (all-zero / EOB) from the
// quantized coefficients alone: eob is the scan index of the last non-zero
// coefficient plus one, exactly as WriteCoefficientsTXB computes it before
// coding txb_skip. This lets the decision pass reconstruct without writing.
func interTXBResult(qcoeff []int16, scan []int16, coeffCount int) tile.TXBDecodeResult {
	eob := 0
	for c := coeffCount - 1; c >= 0; c-- {
		if qcoeff[scan[c]] != 0 {
			eob = c + 1
			break
		}
	}
	if eob == 0 {
		return tile.TXBDecodeResult{AllZero: true}
	}
	return tile.TXBDecodeResult{EOB: uint16(eob)}
}

// reconRecordedTXB copies the prepared coefficients into the record arena and
// reconstructs the block through the decoder-exact dequant/inverse path,
// without emitting any symbols.
func (st *lossyEncodeState) reconRecordedTXB(rec *pframeSplitRecord, reconPlane, pred []byte, predStride, stride, px, py, w, h int,
	q quantize.Quantizer, qcoeff []int16, size tile.TransformSize, scan []int16, txType transform.Type) error {
	geo, err := txBlockGeometryForTransformSize(size)
	if err != nil {
		return err
	}
	if !geo.matchesDimensions(w, h) || len(qcoeff) < geo.coeffCount {
		return tile.ErrInvalidDecodeState
	}
	qcoeff = qcoeff[:geo.coeffCount]
	copy(rec.grabCoeffs(geo.coeffCount), qcoeff)
	return st.reconInterTXBResult(reconPlane, pred, predStride, stride, px, py, w, h, q, qcoeff, geo, scan, txType, interTXBResult(qcoeff, scan, geo.coeffCount))
}

// writeRecordedTXB emits one recorded TXB through the fused path's exact
// entropy dispatch, consuming its coefficients from the record arena.
func (st *lossyEncodeState) writeRecordedTXB(rec *pframeSplitRecord, w, h int, ctxReq tile.CoeffContextRequest,
	coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, txType transform.Type) error {
	geo, err := txBlockGeometryForTransformSize(ctxReq.Size)
	if err != nil {
		return err
	}
	if !geo.matchesDimensions(w, h) {
		return tile.ErrInvalidDecodeState
	}
	qcoeff, err := rec.nextCoeffs(geo.coeffCount)
	if err != nil {
		return err
	}
	_, err = st.emitInterTXBTyped(w, h, qcoeff, ctxReq, coeffCtx, scan, afterSkip, txType)
	return err
}

// decidePBlock is the decision half of encodePBlock: motion estimation,
// reference/mode/filter selection, prediction, transform/quantize, the skip
// decision, and reconstruction — everything except symbol writing. It marks
// the mode context exactly as the fused path so the ref-MV stacks and filter
// contexts of later blocks see the serial coding order's neighbor state, and
// stores the block's decision in rec for the serial write pass.
func (st *lossyEncodeState) decidePBlock(src, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	referenceMode parser.ReferenceMode, walkReq tile.BlockWalkRequest, miCols, miRows uint16, rec *pframeSplitRecord) error {

	bw, bh, ok := pframeBlockDims(block.Size)
	if !ok {
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	n := bw // square tier size; rect blocks take dedicated paths below
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	hasChroma := !st.color.MonoChrome
	modeCtx := &scratch.Mode
	cbw, cbh := 0, 0
	chromaBlock := tile.BlockSize4x4
	chromaPX, chromaPY := 0, 0
	chromaWidth, chromaHeight := 0, 0
	if hasChroma {
		var err error
		chromaBlock, err = tile.PlaneBlockSize(block.Size, st.color, 1)
		if err != nil {
			return fmt.Errorf("chroma plane block: %w", err)
		}
		cbw, cbh, err = planeBlockPixels(chromaBlock)
		if err != nil {
			return fmt.Errorf("chroma plane block dimensions: %w", err)
		}
		chromaPX = chromaXForColor(lumaPX, st.color)
		chromaPY = chromaYForColor(lumaPY, st.color)
		chromaWidth = chromaWidthForColor(src.Width, st.color)
		chromaHeight = chromaHeightForColor(src.Height, st.color)
	}

	// Motion estimation, identical to the fused path (see encodePBlock).
	scaledReference := ref.Width != src.Width || ref.Height != src.Height
	var mv motion.Vector
	fullSAD := -1
	sadEpoch := st.sadCacheEpoch
	if scaledReference {
		if err := predictIntoScaled(st.sadScratch[:bw*bh], ref.Y, ref.YStride, ref.Width, ref.Height, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false, &st.scaledScratch); err != nil {
			return fmt.Errorf("scaled reference luma probe: %w", err)
		}
		fullSAD = sadRectDualBlock(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, st.sadScratch[:bw*bh], bw, bw, bh)
	} else {
		switch {
		case bw != bh:
			if bw >= 32 || bh >= 32 {
				i0 := (lumaPY/16)*st.grid16Cols + lumaPX/16
				i1 := i0 + 1
				if bh > bw {
					i1 = i0 + st.grid16Cols
				}
				if sadCacheValid(st.sad16Grid[i0], sadEpoch) && sadCacheValid(st.sad16Grid[i1], sadEpoch) {
					mv = st.mv16Grid[i0]
					fullSAD = sadCacheValue(st.sad16Grid[i0]) + sadCacheValue(st.sad16Grid[i1])
				}
				break
			}
			i0 := (lumaPY/8)*st.grid8Cols + lumaPX/8
			i1 := i0 + 1
			if bh > bw {
				i1 = i0 + st.grid8Cols
			}
			if sadCacheValid(st.sad8Grid[i0], sadEpoch) && sadCacheValid(st.sad8Grid[i1], sadEpoch) {
				mv = st.mv8Grid[i0]
				fullSAD = sadCacheValue(st.sad8Grid[i0]) + sadCacheValue(st.sad8Grid[i1])
			}
		case n == 64:
			idx := (lumaPY/64)*st.grid64Cols + lumaPX/64
			if sadCacheValid(st.sad64Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv64Grid[idx], sadCacheValue(st.sad64Grid[idx])
			}
		case n == 32:
			idx := (lumaPY/32)*st.grid32Cols + lumaPX/32
			if sadCacheValid(st.sad32Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv32Grid[idx], sadCacheValue(st.sad32Grid[idx])
			}
		case n == 16:
			idx := (lumaPY/16)*st.grid16Cols + lumaPX/16
			if sadCacheValid(st.sad16Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv16Grid[idx], sadCacheValue(st.sad16Grid[idx])
			}
		default:
			if idx := (lumaPY/8)*st.grid8Cols + lumaPX/8; sadCacheValid(st.sad8Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv8Grid[idx], sadCacheValue(st.sad8Grid[idx])
			}
		}
		if fullSAD < 0 {
			seedDX, seedDY, reach := 0, 0, fullPelReach
			if st.hme != nil {
				var trusted bool
				seedDX, seedDY, trusted = st.hme.seedAt(lumaPX, lumaPY)
				if trusted {
					reach = fullPelReachTrusted
				}
			}
			dx, dy, sad := 0, 0, 0
			if bw == bh {
				dx, dy, sad = fullPelDiamondSearchSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, seedDX, seedDY, reach)
			} else {
				dx, dy, sad = fullPelDiamondSearchRectSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, seedDX, seedDY, reach)
			}
			mv, fullSAD = motion.Vector{Row: int16(dy * 8), Col: int16(dx * 8)}, sad
		}
		if st.allowSubpelRefinement() && bw == bh && fullSAD > n*n*2 {
			fullMV := mv
			subpelStop := st.realtimeSubpelStopForBlock(src, lumaPX, lumaPY, n)
			mv, fullSAD = st.subpelRefineWithStop(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, mv, fullSAD, subpelStop)
			if subpelStop != realtimeSubpelStopFull && fullSAD > n*n*2 && (fullMV.Row != 0 || fullMV.Col != 0) {
				zeroSAD := sadBlock(src.Y, ref.Y, lumaPY*src.YStride+lumaPX, lumaPY*src.YStride+lumaPX, src.YStride, n, 1<<30)
				if zeroSAD < fullSAD*2 {
					if zmv, zsad := st.subpelRefineWithStop(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, motion.Vector{}, zeroSAD, subpelStop); zsad < fullSAD {
						mv, fullSAD = zmv, zsad
					}
				}
			}
		}
	}

	// Reference selection (GOLDEN probe / compound), identical to the fused path.
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	refPlanes := ref
	compound := false
	var compoundMV [2]motion.Vector
	goldenEligible := (bw == 8 && bh == 8) ||
		(bw == 16 && bh == 16) || (bw == 32 && bh == 32) ||
		(bw == 16 && bh == 8) || (bw == 8 && bh == 16) ||
		(bw == 32 && bh == 16) || (bw == 16 && bh == 32) ||
		(bw == 64 && bh == 16) || (bw == 16 && bh == 64)
	if !scaledReference && golden != nil && golden.Y != nil && goldenEligible && fullSAD > bw*bh*4 {
		lastMV, lastSAD := mv, fullSAD
		var gmv motion.Vector
		gsad := 1 << 30
		compoundCapable := bw == bh && bw <= 16
		goldenRefPruned := false
		if compoundCapable {
			gdx, gdy, s := fullPelDiamondSearch(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, bw)
			gmv, gsad = motion.Vector{Row: int16(gdy * 8), Col: int16(gdx * 8)}, s
			// dist_based_ref_pruning: drop GOLDEN's subpel + compound MD work once
			// its full-pel distortion is >30% worse than LAST (perform_md_reference_pruning).
			goldenRefPruned = int64(gsad)*100 > int64(lastSAD)*(100+goldenRefPruneMaxDevPct)
			if st.allowSubpelRefinement() && gsad > bw*bh*2 && !goldenRefPruned {
				gmv, gsad = st.subpelRefineWithStop(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, gmv, gsad, st.realtimeSubpelStopForBlock(src, lumaPX, lumaPY, bw))
			}
		} else {
			base := lumaPY*src.YStride + lumaPX
			gsad = sadRectBlock(src.Y, golden.Y, base, base, src.YStride, bw, bh, lastSAD)
		}
		if compoundCapable && !goldenRefPruned && referenceMode == parser.ReferenceModeSelect && gsad+32 >= lastSAD && gsad <= lastSAD+bw*bh*4 {
			if err := predictCompoundInto(st.sadScratch[:bw*bh], ref.Y, ref.YStride, golden.Y, golden.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, lastMV, gmv, false, false, &st.compBuf0, &st.compBuf1, &st.compScratch); err == nil {
				srcBlock := src.Y[lumaPY*src.YStride+lumaPX:]
				var compoundSAD int
				if bw == 8 {
					compoundSAD = sad8x8Dual(srcBlock, src.YStride, st.sadScratch[:64], 8)
				} else {
					compoundSAD = sad16x16Dual(srcBlock, src.YStride, st.sadScratch[:256], 16)
				}
				compoundBias := 64 + 12*st.sadPerBit
				if compoundSAD+compoundBias < fullSAD {
					compound = true
					compoundMV = [2]motion.Vector{lastMV, gmv}
					fullSAD = compoundSAD
				}
			}
		}
		if !compound && gsad+32 < fullSAD {
			refs.Ref[0] = tile.ReferenceFrameGolden
			refPlanes = *golden
			mv, fullSAD = gmv, gsad
		}
		if compound {
			refs = tile.InterReferencesResult{
				Ref:      [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameGolden},
				Compound: true,
				Unidir:   true,
			}
		}
	}

	// Intra fallback (8x8 leaves only), identical to the fused path.
	if bw == 8 && bh == 8 {
		dc := dcPredictN(recon.Y, src.YStride, lumaPX, lumaPY, 8, block.HaveTop, block.HaveLeft)
		intraSAD := 0
		for r := range 8 {
			row := (lumaPY+r)*src.YStride + lumaPX
			for c := range 8 {
				d := int(src.Y[row+c]) - int(dc)
				if d < 0 {
					d = -d
				}
				intraSAD += d
			}
		}
		if fullSAD > intraSAD+32 {
			return st.decideIntraPBlock(src, recon, block, scratch, rec)
		}
	}

	// Reference-MV stack and priced mode choice, identical to the fused path.
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
		ForceIntegerMV: st.forceIntegerMV,
	}
	stack, err := modeCtx.BuildReferenceMVStack(stackReq)
	if err != nil {
		return fmt.Errorf("build ref mv stack: %w", err)
	}
	modeResult := tile.InterModeResult{Mode: tile.InterModeGlobalMV}
	if refs.Compound {
		modeResult = tile.InterModeResult{Compound: true, CompoundMode: tile.CompoundInterModeNewNew}
		if compoundMV[0] == (motion.Vector{}) && compoundMV[1] == (motion.Vector{}) {
			modeResult.CompoundMode = tile.CompoundInterModeGlobalGlobal
		}
	}
	drlIndex := 0
	if !refs.Compound {
		mds0Picked := false
		mds0Gate := fullSAD*4 > bw*bh
		if st.mds0Level == 2 {
			mds0Gate = mds0Gate && fullSAD <= bw*bh*4
		}
		if st.mds0Level != 0 && mds0Gate && !scaledReference && refPlanes.Width == src.Width && refPlanes.Height == src.Height &&
			st.realtimeContentStateForBlock(lumaPX, lumaPY).sourceSADNonRD != realtimeSourceSADZero {
			if cand, ok := st.mds0PickInterMode(src, refPlanes.Y, refPlanes.YStride, &stack, lumaPX, lumaPY, bw, bh, mv); ok {
				modeResult.Mode = cand.mode
				mv = cand.mv
				drlIndex = int(cand.drl)
				mds0Picked = true
			}
		}
		if !mds0Picked {
			modeResult.Mode = st.classifyInterMode(src, refPlanes, &stack, lumaPX, lumaPY, bw, bh, &mv, &fullSAD)
		}
	}
	// Switchable interpolation filter decision (SVT IFS_MDS3, see pframe_ifs.go).
	blockFilters := motion.InterpFilters{}
	if st.interpSearch || st.interpShadow {
		ifsReq := tile.InterpFilterRequest{
			FrameFilter:      parser.InterpolationSwitchable,
			EnableDualFilter: st.interpDual,
			Size:             block.Size,
			References:       refs,
			Mode:             modeResult,
			MotionMode:       tile.MotionModeTranslation,
			X4:               block.X4,
			Y4:               block.Y4,
			HaveTop:          block.HaveTop,
			HaveLeft:         block.HaveLeft,
		}
		if !refs.Compound && tile.InterpNeeded(ifsReq) &&
			refPlanes.Width == src.Width && refPlanes.Height == src.Height {
			searched := st.interpolationFilterSearch(src, refPlanes.Y, refPlanes.YStride, modeCtx, ifsReq, lumaPX, lumaPY, bw, bh, mv)
			st.interpUsed[searched.Y]++
			if st.interpSearch {
				blockFilters = searched
				if st.decisionStats != nil {
					st.decisionStats.noteInterpFilter(int(searched.Y))
				}
			}
		}
	}
	// Materialize the three plane predictions, identical to the fused path.
	if refs.Compound {
		if err := predictCompoundInto(st.predY[:bw*bh], ref.Y, ref.YStride, golden.Y, golden.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, compoundMV[0], compoundMV[1], false, false, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
			return fmt.Errorf("predict compound luma: %w", err)
		}
		if hasChroma {
			if err := predictCompoundInto(st.predU[:cbw*cbh], ref.U, ref.ChromaStride, golden.U, golden.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, compoundMV[0], compoundMV[1], st.color.SubsamplingX, st.color.SubsamplingY, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
				return fmt.Errorf("predict compound u: %w", err)
			}
			if err := predictCompoundInto(st.predV[:cbw*cbh], ref.V, ref.ChromaStride, golden.V, golden.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, compoundMV[0], compoundMV[1], st.color.SubsamplingX, st.color.SubsamplingY, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
				return fmt.Errorf("predict compound v: %w", err)
			}
		}
	} else {
		if refPlanes.Width != src.Width || refPlanes.Height != src.Height {
			refChromaW := chromaWidthForColor(refPlanes.Width, st.color)
			refChromaH := chromaHeightForColor(refPlanes.Height, st.color)
			if err := predictIntoScaled(st.predY[:bw*bh], refPlanes.Y, refPlanes.YStride, refPlanes.Width, refPlanes.Height, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false, &st.scaledScratch); err != nil {
				return fmt.Errorf("predict scaled luma: %w", err)
			}
			if hasChroma {
				if err := predictIntoScaled(st.predU[:cbw*cbh], refPlanes.U, refPlanes.ChromaStride, refChromaW, refChromaH, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, &st.scaledScratch); err != nil {
					return fmt.Errorf("predict scaled u: %w", err)
				}
				if err := predictIntoScaled(st.predV[:cbw*cbh], refPlanes.V, refPlanes.ChromaStride, refChromaW, refChromaH, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, &st.scaledScratch); err != nil {
					return fmt.Errorf("predict scaled v: %w", err)
				}
			}
		} else {
			if err := predictIntoFilters(st.predY[:bw*bh], refPlanes.Y, refPlanes.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false, blockFilters, st.scaledScratch.Conv()); err != nil {
				return fmt.Errorf("predict luma: %w", err)
			}
			if hasChroma {
				if err := predictIntoFilters(st.predU[:cbw*cbh], refPlanes.U, refPlanes.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, blockFilters, st.scaledScratch.Conv()); err != nil {
					return fmt.Errorf("predict u: %w", err)
				}
				if err := predictIntoFilters(st.predV[:cbw*cbh], refPlanes.V, refPlanes.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, blockFilters, st.scaledScratch.Conv()); err != nil {
					return fmt.Errorf("predict v: %w", err)
				}
			}
		}
	}

	// Quantize all planes and take the rate-priced skip decision, identical to
	// the fused path.
	skip := fullSAD*4 <= bw*bh
	splitTX := false
	var txPlan realtimeInterTXPlan
	useRealtimeTXPlan := videoColorIs420(st.color)
	if useRealtimeTXPlan {
		sse, variance := realtimeInterResidualSSEVariance(src.Y, st.predY[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
		var err error
		txLevel := realtimeTXSizeLevelBasedOnQstep(src.Width, src.Height, st.effortLevel)
		txPlan, err = realtimeInterTXPlanForBlock(block.Size, st.qIndex, st.yQuant.AC, txLevel, sse, variance)
		if err != nil {
			return err
		}
	}
	dctRdD := int64(0)
	if !skip {
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		var lumaZero bool
		if txPlan.Variable() {
			lumaZero = true
			leafArea := txPlan.leafSize * txPlan.leafSize
			if err := txPlan.ForEachLeaf(func(i, dx, dy int) error {
				if !st.prepareInterTXB(src.Y, st.predY[dy*bw+dx:], bw, src.YStride, lumaPX+dx, lumaPY+dy, txPlan.leafSize, txPlan.leafSize, st.yQuant, st.lumaQ2[i*leafArea:(i+1)*leafArea]) {
					lumaZero = false
				}
				return nil
			}); err != nil {
				return fmt.Errorf("prepare realtime luma tx: %w", err)
			}
		} else {
			lumaZero = st.prepareInterTXB(src.Y, st.predY[:bw*bh], bw, src.YStride, lumaPX, lumaPY, bw, bh, st.yQuant, st.lumaQ[:bw*bh])
		}
		if hasChroma {
			uZero := st.prepareInterTXB(src.U, st.predU[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.uQuant, st.uQ[:cbw*cbh])
			vZero := st.prepareInterTXB(src.V, st.predV[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.vQuant, st.vQ[:cbw*cbh])
			skip = lumaZero && uZero && vZero
		} else {
			skip = lumaZero
		}
		dctRdD = st.rdDcode
		if !skip {
			rdCode := ((st.rdRcode*st.rdMult + 256) >> 9) + (st.rdDcode << 7)
			rdSkip := st.rdDskip << 7
			if rdSkip <= rdCode {
				skip = true
			}
		}
		if !skip && txPlan.Variable() {
			splitTX = true
		}
	}
	if !useRealtimeTXPlan {
		skip = false
		splitTX = false
	}

	txType := transform.TypeDCTDCT
	if !skip && !splitTX && bw == 8 && bh == 8 && st.qIndex <= 96 && videoColorIs420(st.color) {
		txType = st.chooseInter8x8TXType(src, lumaPX, lumaPY, dctRdD)
	}

	// Record the decision: everything the serial write pass consumes, the
	// libaom MB_MODE_INFO(+_EXT_FRAME) save for pack_bs.
	r := rec.appendBlock()
	r.kind = pblockInter
	r.skip = skip
	r.splitTX = splitTX
	r.refs = refs
	r.modeResult = modeResult
	r.drlIndex = uint8(drlIndex)
	r.modeContext = stack.ModeContext
	r.drlReq, err = stack.Stack.DRLRequestForMode(modeResult)
	if err != nil {
		return fmt.Errorf("drl request: %w", err)
	}
	if !interModeResultUsesGlobalOnly(modeResult) {
		r.mvRefs, err = stack.Stack.ResolveInterMVReferences(modeResult, drlIndex, false, st.forceIntegerMV)
		if err != nil {
			return fmt.Errorf("resolve mv references: %w", err)
		}
	}
	motionResult := tile.InterMotionResult{References: refs, Mode: modeResult}
	if refs.Compound {
		motionResult.MV = compoundMV
	} else {
		motionResult.MV[0] = mv
	}
	r.motion = motionResult
	r.filters = blockFilters
	r.txType = txType
	if splitTX {
		r.txPlan = txPlan
	}
	// Zero-motion accounting (libaom update_state(), av1/encoder/
	// encodeframe_utils.c) belongs to the decision pass.
	if refs.Ref[0] == tile.ReferenceFrameLast {
		zmv := motionResult.MV[0]
		if zmv.Row > -8 && zmv.Row < 8 && zmv.Col > -8 && zmv.Col < 8 {
			st.cntZeroMV += int(block.VisibleW4) * int(block.VisibleH4)
		}
	}

	// Mark the mode context in the fused path's order so later blocks' stacks
	// and filter searches read identical neighbor state.
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{SkipTransform: skip}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), motionResult, hasChroma); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), refs, blockFilters); err != nil {
		return fmt.Errorf("mark inter filters: %w", err)
	}
	if refs.Compound {
		if err := modeCtx.MarkCompoundBlend(block.Size, int(block.X4), int(block.Y4), tile.CompoundBlendResult{
			Type:          tile.CompoundTypeAverage,
			CompoundIndex: 1,
		}); err != nil {
			return fmt.Errorf("mark compound blend: %w", err)
		}
	}

	if skip {
		// Reconstruction is the motion-compensated prediction.
		copyPredScratch(recon.Y, st.predY[:bw*bh], src.YStride, lumaPX, lumaPY, bw, bh)
		if hasChroma {
			copyPredScratch(recon.U, st.predU[:cbw*cbh], src.ChromaStride, chromaPX, chromaPY, cbw, cbh)
			copyPredScratch(recon.V, st.predV[:cbw*cbh], src.ChromaStride, chromaPX, chromaPY, cbw, cbh)
		}
		if st.decisionStats != nil {
			st.decisionStats.noteInterBlock(block.Size, true, false, refs, modeResult, transform.TypeDCTDCT)
		}
		return nil
	}

	// Store the coefficients and reconstruct, in the exact TXB order the write
	// pass replays: luma (whole block or realtime-split leaves), then chroma.
	if splitTX {
		childScan, err := st.scanForLeafTX(txPlan.leafTX)
		if err != nil {
			return err
		}
		leafArea := txPlan.leafSize * txPlan.leafSize
		if err := txPlan.ForEachLeaf(func(i, dx, dy int) error {
			return st.reconRecordedTXB(rec, recon.Y, st.predY[dy*bw+dx:], bw, src.YStride, lumaPX+dx, lumaPY+dy, txPlan.leafSize, txPlan.leafSize, st.yQuant, st.lumaQ2[i*leafArea:(i+1)*leafArea], txPlan.leafTX, childScan, transform.TypeDCTDCT)
		}); err != nil {
			return fmt.Errorf("luma realtime tx: %w", err)
		}
	} else {
		lumaTX, lumaScan := st.interLumaTXAndScan(bw, bh)
		if err := st.reconRecordedTXB(rec, recon.Y, st.predY[:bw*bh], bw, src.YStride, lumaPX, lumaPY, bw, bh, st.yQuant, st.lumaQ[:bw*bh], lumaTX, lumaScan, txType); err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
	}
	if hasChroma {
		chromaTX, chromaScan, err := st.interChromaTXAndScan(block.Size)
		if err != nil {
			return err
		}
		chromaTxType := transform.TypeDCTDCT
		if !splitTX && bw == 8 && bh == 8 {
			chromaTxType = txType
		}
		for plane := 1; plane <= 2; plane++ {
			rdata, pred, qc := recon.U, st.predU[:cbw*cbh], st.uQ[:cbw*cbh]
			q := st.uQuant
			if plane == 2 {
				rdata, pred, qc = recon.V, st.predV[:cbw*cbh], st.vQ[:cbw*cbh]
				q = st.vQuant
			}
			if err := st.reconRecordedTXB(rec, rdata, pred, cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, q, qc, chromaTX, chromaScan, chromaTxType); err != nil {
				return fmt.Errorf("chroma %d txb: %w", plane, err)
			}
		}
	}
	if st.decisionStats != nil {
		st.decisionStats.noteInterBlock(block.Size, false, splitTX, refs, modeResult, txType)
		st.noteInterTXBSizes(block.Size, splitTX, txPlan, bw, bh)
	}
	return nil
}

// writePBlock is the entropy half of encodePBlock: it replays one recorded
// block through the same tile writers, in the same order, with the same
// context marks, as the fused path (libaom write_modes_b in
// av1/encoder/bitstream.c).
func (st *lossyEncodeState) writePBlock(block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	refCDFs *tile.InterRefCDFs, interModeCDFs *tile.InterModeCDFs, interpCDFs *tile.InterpFilterCDFs,
	referenceMode parser.ReferenceMode, rec *pframeSplitRecord) error {

	r, err := rec.nextBlock()
	if err != nil {
		return err
	}
	if r.kind == pblockIntra {
		return st.writeIntraPBlock(block, scratch, r, rec)
	}
	bw, bh, ok := pframeBlockDims(block.Size)
	if !ok {
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	hasChroma := !st.color.MonoChrome
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx
	cbw, cbh := 0, 0
	chromaBlock := tile.BlockSize4x4
	chromaX4, chromaY4 := uint8(0), uint8(0)
	if hasChroma {
		var err error
		chromaBlock, err = tile.PlaneBlockSize(block.Size, st.color, 1)
		if err != nil {
			return fmt.Errorf("chroma plane block: %w", err)
		}
		cbw, cbh, err = planeBlockPixels(chromaBlock)
		if err != nil {
			return fmt.Errorf("chroma plane block dimensions: %w", err)
		}
		chromaX4 = chromaX4ForColor(block.X4, st.color)
		chromaY4 = chromaY4ForColor(block.Y4, st.color)
	}

	refs := r.refs
	modeResult := r.modeResult
	skip := r.skip

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
		ReferenceMode: referenceMode,
		X4:            block.X4, Y4: block.Y4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
	}, refs); err != nil {
		return fmt.Errorf("references: %w", err)
	}

	if err := tile.WriteBlockInterMode(st.w, interModeCDFs, tile.InterModeRequest{
		Compound:    refs.Compound,
		ModeContext: r.modeContext,
	}, modeResult); err != nil {
		return fmt.Errorf("inter mode: %w", err)
	}
	if err := tile.WriteDRLIndex(st.w, interModeCDFs, r.drlReq, int(r.drlIndex)); err != nil {
		return fmt.Errorf("drl: %w", err)
	}
	if err := tile.WriteInterMotion(st.w, &st.mvCDFs, tile.InterMotionRequest{
		References:   refs,
		Mode:         modeResult,
		ReferenceMVs: r.mvRefs,
		Precision:    tile.MVPrecision(false, st.forceIntegerMV),
	}, r.motion); err != nil {
		return fmt.Errorf("motion vector: %w", err)
	}

	if st.interpSearch {
		ifsReq := tile.InterpFilterRequest{
			FrameFilter:      parser.InterpolationSwitchable,
			EnableDualFilter: st.interpDual,
			Size:             block.Size,
			References:       refs,
			Mode:             modeResult,
			MotionMode:       tile.MotionModeTranslation,
			X4:               block.X4,
			Y4:               block.Y4,
			HaveTop:          block.HaveTop,
			HaveLeft:         block.HaveLeft,
		}
		if _, err := tile.WriteInterpFilters(st.w, interpCDFs, modeCtx, ifsReq, r.filters); err != nil {
			return fmt.Errorf("interp filters: %w", err)
		}
	}

	if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), r.motion, hasChroma); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), refs, r.filters); err != nil {
		return fmt.Errorf("mark inter filters: %w", err)
	}
	if refs.Compound {
		if err := modeCtx.MarkCompoundBlend(block.Size, int(block.X4), int(block.Y4), tile.CompoundBlendResult{
			Type:          tile.CompoundTypeAverage,
			CompoundIndex: 1,
		}); err != nil {
			return fmt.Errorf("mark compound blend: %w", err)
		}
	}

	var treeRes tile.TransformTreeResult
	if r.splitTX {
		treeRes = r.txPlan.tree
	}
	lfTree, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: st.transformMode,
		Inter: true, SkipTransform: skip,
	}, treeRes)
	if err != nil {
		return fmt.Errorf("transform tree: %w", err)
	}
	if st.lfMap != nil {
		lfMode := loopfilter.ModeDeltaClassZero
		if !interModeResultUsesGlobalOnly(modeResult) {
			lfMode = loopfilter.ModeDeltaClassMotion
		}
		if err := markLoopFilterBlock(st.lfMap, block, lfTree, skip, false, uint8(refs.Ref[0])+1, lfMode); err != nil {
			return fmt.Errorf("mark loop filter: %w", err)
		}
	}

	if skip {
		if err := coeffCtx.ResetBlock(0, block.Size, int(block.X4), int(block.Y4)); err != nil {
			return fmt.Errorf("reset luma coeff ctx: %w", err)
		}
		if hasChroma {
			for plane := 1; plane <= 2; plane++ {
				if err := coeffCtx.ResetBlock(plane, chromaBlock, int(chromaX4), int(chromaY4)); err != nil {
					return fmt.Errorf("reset chroma %d coeff ctx: %w", plane, err)
				}
			}
		}
		return nil
	}

	// Residual replay: same TXB order and same writers as the fused path.
	if r.splitTX {
		childScan, err := st.scanForLeafTX(r.txPlan.leafTX)
		if err != nil {
			return err
		}
		st.interTxTypeReq.Size = r.txPlan.leafTX
		st.interTxType = transform.TypeDCTDCT
		if err := r.txPlan.ForEachLeaf(func(i, dx, dy int) error {
			return st.writeRecordedTXB(rec, r.txPlan.leafSize, r.txPlan.leafSize, tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: block.Size,
				Size:       r.txPlan.leafTX,
				X4:         block.X4 + uint8(dx/4),
				Y4:         block.Y4 + uint8(dy/4),
			}, coeffCtx, childScan, st.afterSkipInter, transform.TypeDCTDCT)
		}); err != nil {
			return fmt.Errorf("luma realtime tx: %w", err)
		}
	} else {
		lumaTX, lumaScan := st.interLumaTXAndScan(bw, bh)
		st.interTxTypeReq.Size = lumaTX
		st.interTxType = r.txType
		if err := st.writeRecordedTXB(rec, bw, bh, tile.CoeffContextRequest{
			Plane:      0,
			PlaneBlock: block.Size,
			Size:       lumaTX,
			X4:         block.X4,
			Y4:         block.Y4,
		}, coeffCtx, lumaScan, st.afterSkipInter, r.txType); err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
	}
	if hasChroma {
		chromaTX, chromaScan, err := st.interChromaTXAndScan(block.Size)
		if err != nil {
			return err
		}
		chromaTxType := transform.TypeDCTDCT
		if !r.splitTX && bw == 8 && bh == 8 {
			chromaTxType = r.txType
		}
		for plane := 1; plane <= 2; plane++ {
			if err := st.writeRecordedTXB(rec, cbw, cbh, tile.CoeffContextRequest{
				Plane:      uint8(plane),
				PlaneBlock: chromaBlock,
				Size:       chromaTX,
				X4:         chromaX4,
				Y4:         chromaY4,
			}, coeffCtx, chromaScan, nil, chromaTxType); err != nil {
				return fmt.Errorf("chroma %d txb: %w", plane, err)
			}
		}
	}
	return nil
}

// decideIntraPBlock is the decision half of encodeIntraPBlock (the 8x8 intra
// fallback): mode selection against the reconstructed edges, quantization of
// all three planes into the record arena, and reconstruction.
func (st *lossyEncodeState) decideIntraPBlock(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch, rec *pframeSplitRecord) error {
	modeCtx := &scratch.Mode
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	pred := st.predY[:64]
	mode := selectIntraMode8(src.Y, recon.Y, src.YStride, lumaPX, lumaPY, block.HaveTop, block.HaveLeft, pred)
	if !st.trialReady {
		if err := st.trialCDFs.InitDefault(st.qIndex); err != nil {
			return err
		}
		st.trialReady = true
	}
	mode, angleDelta := st.improveIntraModeDirectional(src, recon, block, mode, pred, lumaPX, lumaPY, 8)

	r := rec.appendBlock()
	r.kind = pblockIntra
	r.intraMode = mode
	r.angleDelta = int8(angleDelta)

	// Context marks in the fused path's order.
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("intra mark prefix: %w", err)
	}
	if err := modeCtx.MarkIntra(block.Size, int(block.X4), int(block.Y4), true, mode); err != nil {
		return fmt.Errorf("mark intra: %w", err)
	}
	hasChroma := !st.color.MonoChrome
	if hasChroma {
		if err := modeCtx.MarkChromaIntra(block.Size, int(block.X4), int(block.Y4), true, tile.ChromaIntraModeDC); err != nil {
			return fmt.Errorf("mark chroma intra: %w", err)
		}
	}
	if st.allowScreenContentTools {
		// writeNoPaletteMode's context marks without its symbols.
		if err := modeCtx.MarkPaletteY(block.Size, int(block.X4), int(block.Y4), tile.PaletteModeResult{}); err != nil {
			return fmt.Errorf("mark palette y: %w", err)
		}
		if err := modeCtx.MarkPaletteUV(block.Size, int(block.X4), int(block.Y4), tile.PaletteModeResult{}); err != nil {
			return fmt.Errorf("mark palette uv: %w", err)
		}
	}

	// Luma: quantize against the chosen prediction, record, reconstruct.
	if err := st.reconRecordedIntraTXB(rec, recon.Y, src.Y, src.YStride, lumaPX, lumaPY, 8, 8, st.yQuant, tile.TransformSize8x8, pred); err != nil {
		return fmt.Errorf("intra luma txb: %w", err)
	}
	if !hasChroma {
		if st.decisionStats != nil {
			st.decisionStats.noteIntraBlock(block.Size)
		}
		return nil
	}
	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma plane block: %w", err)
	}
	cw, ch, err := planeBlockPixels(chromaBlock)
	if err != nil {
		return fmt.Errorf("chroma plane block dimensions: %w", err)
	}
	chromaTX, err := tile.MaxTransformSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma transform size: %w", err)
	}
	chromaX := chromaXForColor(lumaPX, st.color)
	chromaY := chromaYForColor(lumaPY, st.color)
	for plane := 1; plane <= 2; plane++ {
		data, rdata := src.U, recon.U
		q := st.uQuant
		if plane == 2 {
			data, rdata = src.V, recon.V
			q = st.vQuant
		}
		// DC prediction from the reconstructed edges, as encodeTXBAvailRect.
		dc := dcPredictRect(rdata, src.ChromaStride, chromaX, chromaY, cw, ch, block.HaveTop, block.HaveLeft)
		cpred := st.predU[:cw*ch]
		for i := range cpred {
			cpred[i] = dc
		}
		if err := st.reconRecordedIntraTXB(rec, rdata, data, src.ChromaStride, chromaX, chromaY, cw, ch, q, chromaTX, cpred); err != nil {
			return fmt.Errorf("intra chroma %d txb: %w", plane, err)
		}
	}
	if st.decisionStats != nil {
		st.decisionStats.noteIntraBlock(block.Size)
	}
	return nil
}

// reconRecordedIntraTXB is the write-free half of encodeTXBPredRect: residual,
// forward DCT, quantize into the record arena, then the decoder-exact
// dequant + inverse + add reconstruction (full-coefficient dequant, as the
// fused intra path does).
func (st *lossyEncodeState) reconRecordedIntraTXB(rec *pframeSplitRecord, reconPlane, srcPlane []byte, stride, px, py, w, h int,
	q quantize.Quantizer, size tile.TransformSize, pred []byte) error {
	geo, err := txBlockGeometryForTransformSize(size)
	if err != nil {
		return err
	}
	if !geo.matchesDimensions(w, h) {
		return tile.ErrInvalidDecodeState
	}
	n := geo.sampleCount
	cn := geo.coeffCount
	residual := &st.resScratch
	residualBlockImpl(residual[:n], srcPlane, py*stride+px, stride, pred, w, w, h)
	tran := &st.tranScratch
	if err := forwardDCTBlock(tran[:cn], residual[:n], w, h); err != nil {
		return err
	}
	qcoeff := rec.grabCoeffs(cn)
	if err := quantize.QuantizeBlockScaledB(qcoeff, geo.coeffHeight, tran[:cn], geo.coeffHeight, geo.coeffWidth, geo.coeffHeight, q, geo.txScale); err != nil {
		return err
	}
	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:cn], geo.coeffHeight, qcoeff, geo.coeffHeight, geo.coeffWidth, geo.coeffHeight, q, geo.txScale, 8); err != nil {
		return err
	}
	res := &st.invResidual
	if err := transform.InverseDCTBlock(res[:n], w, dq[:cn], geo.coeffHeight, st.invScratch[:n], geo.size); err != nil {
		return err
	}
	for r := range h {
		row := (py+r)*stride + px
		for c := range w {
			v := int(pred[r*w+c]) + int(res[r*w+c])
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

// writeIntraPBlock is the entropy half of encodeIntraPBlock, replaying the
// recorded mode and coefficients through the same writers and marks.
func (st *lossyEncodeState) writeIntraPBlock(block tile.BlockVisit, scratch *tile.BlockLoopScratch, r *pblockRecord, rec *pframeSplitRecord) error {
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("intra skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("intra mark prefix: %w", err)
	}
	modeCtx.TxNeighborValid = false
	if int(block.X4) < tile.MaxBlockModeSlots && int(block.Y4) < tile.MaxBlockModeSlots {
		modeCtx.TxNeighborValid = true
		modeCtx.TxAboveNeighborIntra = modeCtx.AboveIntra[block.X4]
		modeCtx.TxAboveNeighborBlockSize = modeCtx.AboveBlockSize[block.X4]
		modeCtx.TxLeftNeighborIntra = modeCtx.LeftIntra[block.Y4]
		modeCtx.TxLeftNeighborBlockSize = modeCtx.LeftBlockSize[block.Y4]
	}
	if err := tile.WriteIntraFlag(st.w, &st.intraCDFs, modeCtx, tile.IntraFlagRequest{
		FrameType: parser.FrameTypeInter,
		X4:        block.X4, Y4: block.Y4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
	}, true); err != nil {
		return fmt.Errorf("intra flag: %w", err)
	}
	mode := r.intraMode
	if err := tile.WriteLumaIntraMode(st.w, &st.intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		FrameType: parser.FrameTypeInter,
		Size:      block.Size, X4: block.X4, Y4: block.Y4,
	}, mode); err != nil {
		return fmt.Errorf("intra luma mode: %w", err)
	}
	if err := modeCtx.MarkIntra(block.Size, int(block.X4), int(block.Y4), true, mode); err != nil {
		return fmt.Errorf("mark intra: %w", err)
	}
	if err := tile.WriteIntraAngleDelta(st.w, &st.intraCDFs, tile.IntraAngleDeltaRequest{
		Size: block.Size, Mode: mode,
	}, r.angleDelta); err != nil {
		return fmt.Errorf("intra angle delta: %w", err)
	}
	hasChroma := !st.color.MonoChrome
	if hasChroma {
		cflAllowed, err := tile.ChromaIntraCFLAllowed(block.Size, st.color, false)
		if err != nil {
			return fmt.Errorf("cfl allowed: %w", err)
		}
		if err := tile.WriteChromaIntraMode(st.w, &st.intraCDFs, tile.ChromaIntraModeRequest{
			Size: block.Size, LumaMode: mode, CFLAllowed: cflAllowed,
		}, tile.ChromaIntraModeDC, tile.CFLAlphaResult{}); err != nil {
			return fmt.Errorf("intra chroma mode: %w", err)
		}
		if err := modeCtx.MarkChromaIntra(block.Size, int(block.X4), int(block.Y4), true, tile.ChromaIntraModeDC); err != nil {
			return fmt.Errorf("mark chroma intra: %w", err)
		}
		if err := st.writeNoPaletteMode(modeCtx, block, mode, tile.ChromaIntraModeDC, true); err != nil {
			return err
		}
	} else if err := st.writeNoPaletteMode(modeCtx, block, mode, tile.ChromaIntraModeDC, false); err != nil {
		return err
	}

	lfTree, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: st.transformMode,
	}, tile.TransformTreeResult{Y: tile.TransformSize8x8})
	if err != nil {
		return fmt.Errorf("intra transform tree: %w", err)
	}
	if st.lfMap != nil {
		if err := markLoopFilterBlock(st.lfMap, block, lfTree, false, true, 0, loopfilter.ModeDeltaClassZero); err != nil {
			return fmt.Errorf("mark intra loop filter: %w", err)
		}
	}

	st.intraTxTypeReq.Mode = mode
	lumaCoeffs, err := rec.nextCoeffs(64)
	if err != nil {
		return err
	}
	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, transform.Class2D, lumaCoeffs, st.scan8, st.levels, st.afterSkipIntra); err != nil {
		return fmt.Errorf("intra luma txb: %w", err)
	}
	if !hasChroma {
		return nil
	}
	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma plane block: %w", err)
	}
	chromaTX, err := tile.MaxTransformSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma transform size: %w", err)
	}
	chromaScan, ok := st.scanForTransformSize(chromaTX)
	if !ok {
		chromaScan, ok = st.scanForThinTransformSize(chromaTX)
	}
	if !ok {
		return fmt.Errorf("encoder: unsupported chroma transform %d", chromaTX)
	}
	chromaGeo, err := txBlockGeometryForTransformSize(chromaTX)
	if err != nil {
		return err
	}
	chromaX4 := chromaX4ForColor(block.X4, st.color)
	chromaY4 := chromaY4ForColor(block.Y4, st.color)
	for plane := 1; plane <= 2; plane++ {
		qcoeff, err := rec.nextCoeffs(chromaGeo.coeffCount)
		if err != nil {
			return err
		}
		if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       chromaTX,
			X4:         chromaX4,
			Y4:         chromaY4,
		}, transform.Class2D, qcoeff, chromaScan, st.levels, nil); err != nil {
			return fmt.Errorf("intra chroma %d txb: %w", plane, err)
		}
	}
	return nil
}
