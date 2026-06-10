package encoder

import (
	"fmt"
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/threading"
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
	if src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: P-frame requires multiple-of-8 dimensions, got %dx%d", src.Width, src.Height)
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
	header, refState := repeatPFrameHeader(src.Width, src.Height, qIndex, 0x01)
	header.References = &refState

	out, err := assembleInterTU(seq, header, []TilePayload{{Data: tilePayload}}, 0)
	if err != nil {
		return nil, SourceFrame420{}, err
	}
	return out, recon, nil
}

// assembleInterTU wraps one coded tile into a TD + inter frame header + tile
// group temporal unit.
func assembleInterTU(seq SequenceHeader, header InterFrameHeaderParams, tilePayloads []TilePayload, temporalID uint8) ([]byte, error) {
	endTile := uint16(len(tilePayloads) - 1)
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, endTile, tilePayloads)
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, endTile, tilePayloads)
	if err != nil {
		return nil, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, TemporalID: temporalID, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, err
	}
	frameSize, err := LowOverheadInterFrameHeaderOBUSize(seq, header, temporalID, 0)
	if err != nil {
		return nil, fmt.Errorf("size inter header: %w", err)
	}
	tdSize := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter})
	out := make([]byte, 0, tdSize+frameSize+groupOBUSize)
	out, err = AppendLowOverheadOBU(out, OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		return nil, err
	}
	out, err = AppendLowOverheadInterFrameHeaderOBU(out, seq, header, temporalID, 0)
	if err != nil {
		return nil, fmt.Errorf("append inter header: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, nil
}

// frameCDFs is one frame's complete saved symbol-context state - what the
// decoder keeps from the context-update tile and reloads when the next
// frame names it through primary_ref_frame. It reuses the decoder's own
// storage type so the save semantics (notably the symbol-counter reset of
// av1_reset_cdf_symbol_counters) are shared, and families this encoder
// never codes persist across the chain exactly as the decoder's do.
type frameCDFs = threading.FrameWorkTileResidualCDFStorage

// pframeCoder is the reusable per-stream P-frame coding state: CDFs, scratch,
// the superblock context carrier, and the entropy output buffer, so steady-
// state frame encoding allocates nothing beyond the temporal-unit assembly.
type pframeCoder struct {
	st        lossyEncodeState
	partCDFs  tile.PartitionCDFs
	refCDFs   tile.InterRefCDFs
	modeCDFs  tile.InterModeCDFs
	scratch   tile.BlockLoopScratch
	carrier   tile.BlockLoopContextCarrier
	writerBuf []byte
}

// reset (re)initializes the per-frame CDF and quantizer state. Buffers are
// allocated on first use and reused afterwards. When prev is non-nil the
// symbol contexts chain from it - the saved state of the frame named by
// primary_ref_frame - instead of the defaults, exactly as the decoder loads
// them.
func (pc *pframeCoder) reset(qIndex uint8, rootCols int, prev *frameCDFs) error {
	st := &pc.st
	st.qIndex = qIndex
	st.color = parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	if prev != nil {
		pc.partCDFs = prev.Partition
		pc.refCDFs = prev.InterRef
		pc.modeCDFs = prev.InterMode
		st.modeCDFs = prev.Mode
		st.intraCDFs = prev.Intra
		st.coeffCDFs = prev.Coeff
		st.txCDFs = prev.TransformType
		st.treeCDFs = prev.Transform
		st.mvCDFs = prev.MV
	} else {
		if err := pc.partCDFs.InitDefault(); err != nil {
			return err
		}
		if err := pc.refCDFs.InitDefault(); err != nil {
			return err
		}
		if err := pc.modeCDFs.InitDefault(); err != nil {
			return err
		}
		if err := st.modeCDFs.InitDefault(); err != nil {
			return err
		}
		if err := st.intraCDFs.InitDefault(); err != nil {
			return err
		}
		if err := st.coeffCDFs.InitDefault(qIndex); err != nil {
			return err
		}
		if err := st.txCDFs.InitDefault(); err != nil {
			return err
		}
		if err := st.treeCDFs.InitDefault(); err != nil {
			return err
		}
		if err := st.mvCDFs.InitDefault(); err != nil {
			return err
		}
	}
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.Plane(plane))
		if err != nil {
			return err
		}
		*dst = q
	}
	if st.scan8 == nil {
		st.scan8 = make([]int16, 64)
		inverse8 := make([]int16, 64)
		if err := transform.FillDefaultScan(st.scan8, inverse8, transform.Size{Width: 8, Height: 8}, transform.Class2D); err != nil {
			return err
		}
		st.scan4 = make([]int16, 16)
		inverse4 := make([]int16, 16)
		if err := transform.FillDefaultScan(st.scan4, inverse4, transform.Size{Width: 4, Height: 4}, transform.Class2D); err != nil {
			return err
		}
		st.scan16 = make([]int16, 256)
		inverse16 := make([]int16, 256)
		if err := transform.FillDefaultScan(st.scan16, inverse16, transform.Size{Width: 16, Height: 16}, transform.Class2D); err != nil {
			return err
		}
		st.scan32 = make([]int16, 1024)
		inverse32 := make([]int16, 1024)
		if err := transform.FillDefaultScan(st.scan32, inverse32, transform.Size{Width: 32, Height: 32}, transform.Class2D); err != nil {
			return err
		}
		scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize32x32)
		if err != nil {
			return err
		}
		st.levels = make([]uint8, scratchLen)
		st.invScratch = make([]int32, 1024)
	}
	if cap(pc.writerBuf) == 0 {
		pc.writerBuf = make([]byte, 1<<18)
	}
	if len(pc.carrier.Above) < rootCols {
		pc.carrier.Above = make([]tile.BlockLoopRootAboveContext, rootCols)
	}
	pc.carrier.Left = tile.BlockLoopRootLeftContext{}
	for i := range pc.carrier.Above[:rootCols] {
		pc.carrier.Above[i] = tile.BlockLoopRootAboveContext{}
	}
	return nil
}

func encodePFrameTile(src SourceFrame420, ref SourceFrame420, recon *SourceFrame420, qIndex uint8) ([]byte, error) {
	var pc pframeCoder
	return pc.encodeTile(src, ref, nil, recon, qIndex, nil, 0, uint16(src.Width/4))
}

// exportCDFs snapshots the coder's adapted symbol contexts - the frame-end
// state the decoder saves from the context-update tile - and resets the
// symbol counters the way the decoder's save path does.
func (pc *pframeCoder) exportCDFs(dst *frameCDFs) error {
	dst.Partition = pc.partCDFs
	dst.InterRef = pc.refCDFs
	dst.InterMode = pc.modeCDFs
	dst.Mode = pc.st.modeCDFs
	dst.Intra = pc.st.intraCDFs
	dst.Coeff = pc.st.coeffCDFs
	dst.TransformType = pc.st.txCDFs
	dst.Transform = pc.st.treeCDFs
	dst.MV = pc.st.mvCDFs
	return dst.ResetCDFCounts()
}

// encodeTile codes one P-frame tile covering MI columns [miColStart,
// miColEnd) reusing the coder's buffers. Bounds must be superblock-aligned;
// the full-frame single-tile case passes [0, miCols).
func (pc *pframeCoder) encodeTile(src SourceFrame420, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, qIndex uint8, prev *frameCDFs, miColStart, miColEnd uint16) ([]byte, error) {
	miCols := uint16(src.Width / 4)
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB
	if err := pc.reset(qIndex, rootCols, prev); err != nil {
		return nil, err
	}
	st := &pc.st

	w := entropy.NewWriter(pc.writerBuf[:0])
	st.w = &w
	st.interTxTypeReq = tile.InterTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		QIndexKnown: true,
		QIndex:      qIndex,
	}
	st.afterSkipInter = func() error {
		return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, transform.TypeDCTDCT)
	}
	st.intraTxTypeReq = tile.IntraTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		Mode:        tile.IntraModeDC,
		QIndexKnown: true,
		QIndex:      qIndex,
	}
	st.afterSkipIntra = func() error {
		return tile.WriteIntraTransformType(st.w, &st.txCDFs, st.intraTxTypeReq, transform.TypeDCTDCT)
	}

	// av1_compute_rd_mult_based_on_qindex, inter shape at 8-bit: the DC step
	// squared scaled by the rate multiplier; RDCOST pairs it with rates in
	// 512-units-per-bit and distortion shifted by RDDIV_BITS.
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	// av1's sad-per-bit lut formula: full-pel motion search prices one bit
	// of side information at this many SAD units (q is the dc step over 4).
	st.sadPerBit = int(0.0418*(dcq/4) + 2.4107)

	scratch := &pc.scratch
	carrier := &pc.carrier
	walkReq := tile.BlockWalkRequest{
		Root:       tile.BlockLevel64x64,
		MIColStart: miColStart,
		MIColEnd:   miColEnd,
		MIRowEnd:   miRows,
	}
	st.grid16Cols = (int(miCols) + 3) / 4
	grid16Rows := (int(miRows) + 3) / 4
	if len(st.mv16Grid) < st.grid16Cols*grid16Rows {
		st.mv16Grid = make([]motion.Vector, st.grid16Cols*grid16Rows)
		st.sad16Grid = make([]int32, st.grid16Cols*grid16Rows)
	}
	st.grid8Cols = (int(miCols) + 1) / 2
	grid8Rows := (int(miRows) + 1) / 2
	if len(st.mv8Grid) < st.grid8Cols*grid8Rows {
		st.mv8Grid = make([]motion.Vector, st.grid8Cols*grid8Rows)
		st.sad8Grid = make([]int32, st.grid8Cols*grid8Rows)
	}
	st.grid32Cols = (int(miCols) + 7) / 8
	grid32Rows := (int(miRows) + 7) / 8
	if len(st.mv32Grid) < st.grid32Cols*grid32Rows {
		st.mv32Grid = make([]motion.Vector, st.grid32Cols*grid32Rows)
		st.sad32Grid = make([]int32, st.grid32Cols*grid32Rows)
	}
	for i := range st.sad8Grid {
		st.sad8Grid[i] = -1
	}
	for i := range st.sad16Grid[:st.grid16Cols*grid16Rows] {
		st.sad16Grid[i] = -1
	}
	// mergeBias16 is the extra full-pel SAD a merged 16x16 block may carry
	// over the four independent 8x8 searches and still be coded as one block:
	// the saved mode/MV syntax of three blocks outweighs a small residual
	// increase at realtime rates. mergeBias32 plays the same role one tier up
	// (a 32x32 merge saves up to three 16x16 blocks' syntax).
	const mergeBias16 = 64
	const mergeBias32 = 192
	// evaluate16 fills the 16x16 and child 8x8 motion grids for the 16x16
	// region at (px, py) — searching only on first use — and returns the
	// merged SAD and the sum of the four child SADs.
	evaluate16 := func(px, py int) (int, int) {
		idx16 := (py/16)*st.grid16Cols + px/16
		if st.sad16Grid[idx16] < 0 {
			dx16, dy16, sad16 := fullPelDiamondSearch(src.Y, ref.Y, src.YStride, src.Width, src.Height, px, py, 16)
			st.mv16Grid[idx16] = motion.Vector{Row: int16(dy16 * 8), Col: int16(dx16 * 8)}
			st.sad16Grid[idx16] = int32(sad16)
			for _, off := range [4][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
				cx, cy := px+off[0], py+off[1]
				dx, dy, sad := fullPelDiamondSearch(src.Y, ref.Y, src.YStride, src.Width, src.Height, cx, cy, 8)
				idx8 := (cy/8)*st.grid8Cols + cx/8
				st.mv8Grid[idx8] = motion.Vector{Row: int16(dy * 8), Col: int16(dx * 8)}
				st.sad8Grid[idx8] = int32(sad)
			}
		}
		sum8 := 0
		for _, off := range [4][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
			sum8 += int(st.sad8Grid[((py+off[1])/8)*st.grid8Cols+(px+off[0])/8])
		}
		return int(st.sad16Grid[idx16]), sum8
	}
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		if !haveRight || !haveBottom {
			return tile.PartitionSplit, nil
		}
		px, py := int(miCol)*4, int(miRow)*4
		switch level {
		case tile.BlockLevel32x32:
			// haveRight/haveBottom only certify the half extents; a 32x32
			// leaf must lie fully inside the frame for the unclipped
			// residual path (overhanging nodes split into contained 16s).
			if px+32 > src.Width || py+32 > src.Height {
				return tile.PartitionSplit, nil
			}
			// Merge signal without a 32x32 search: when all four 16x16
			// children settled on the same full-pel vector, one SAD probe of
			// the merged block at that vector decides; disagreeing children
			// mean real sub-block motion and the node splits.
			childCost := 0
			agree := true
			var mv32 motion.Vector
			for i, off := range [4][2]int{{0, 0}, {16, 0}, {0, 16}, {16, 16}} {
				sad16, sum8 := evaluate16(px+off[0], py+off[1])
				childCost += min(sad16, sum8)
				cmv := st.mv16Grid[((py+off[1])/16)*st.grid16Cols+(px+off[0])/16]
				if i == 0 {
					mv32 = cmv
				} else if cmv != mv32 {
					agree = false
				}
			}
			if !agree {
				return tile.PartitionSplit, nil
			}
			dx, dy := int(mv32.Col)/8, int(mv32.Row)/8
			base := py*src.YStride + px
			refBase := (py+dy)*src.YStride + px + dx
			if py+dy < 0 || px+dx < 0 || py+dy+32 > src.Height || px+dx+32 > src.Width {
				return tile.PartitionSplit, nil
			}
			sad32 := sadBlock(src.Y, ref.Y, base, refBase, src.YStride, 32, 1<<30)
			if sad32 <= childCost+mergeBias32 {
				idx32 := (py/32)*st.grid32Cols + px/32
				st.mv32Grid[idx32] = mv32
				st.sad32Grid[idx32] = int32(sad32)
				return tile.PartitionNone, nil
			}
			return tile.PartitionSplit, nil
		case tile.BlockLevel16x16:
			sad16, sum8 := evaluate16(px, py)
			if sad16 <= sum8+mergeBias16 {
				return tile.PartitionNone, nil
			}
			return tile.PartitionSplit, nil
		}
		return tile.PartitionSplit, nil
	}

	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodePBlock(src, ref, golden, recon, block, scratch, &pc.refCDFs, &pc.modeCDFs, walkReq, miCols, miRows)
	}
	if err := tile.WalkBlockLoopWrite(&w, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

// encodePBlock codes one inter block (8x8 or 16x16) with residual: the inter
// mode symbols in the decoder's order, then the largest-TX luma residual (with
// the inter tx_type symbol after txb_skip) and two chroma residuals against
// the reference reconstruction.
func (st *lossyEncodeState) encodePBlock(src, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	refCDFs *tile.InterRefCDFs, interModeCDFs *tile.InterModeCDFs,
	walkReq tile.BlockWalkRequest, miCols, miRows uint16) error {

	var n int
	switch block.Size {
	case tile.BlockSize8x8:
		n = 8
	case tile.BlockSize16x16:
		n = 16
	case tile.BlockSize32x32:
		n = 32
	default:
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	cn := n / 2
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	// Motion estimation first: the skip decision needs the motion-compensated
	// residual. The partition decider already ran the full-pel searches and
	// cached them; fall back to a fresh search for blocks it never scored
	// (frame-edge nodes reach the leaf without a 16x16 decision). Subpel
	// refinement through the decoder's own convolve when the match is
	// imperfect.
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	var mv motion.Vector
	fullSAD := -1
	switch {
	case n == 32:
		idx := (lumaPY/32)*st.grid32Cols + lumaPX/32
		mv, fullSAD = st.mv32Grid[idx], int(st.sad32Grid[idx])
	case n == 16:
		idx := (lumaPY/16)*st.grid16Cols + lumaPX/16
		if st.sad16Grid[idx] >= 0 {
			mv, fullSAD = st.mv16Grid[idx], int(st.sad16Grid[idx])
		}
	default:
		if idx := (lumaPY/8)*st.grid8Cols + lumaPX/8; st.sad8Grid[idx] >= 0 {
			mv, fullSAD = st.mv8Grid[idx], int(st.sad8Grid[idx])
		}
	}
	if fullSAD < 0 {
		dx, dy, sad := fullPelDiamondSearch(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n)
		mv, fullSAD = motion.Vector{Row: int16(dy * 8), Col: int16(dx * 8)}, sad
	}
	if fullSAD > n*n*2 {
		fullMV := mv
		mv, fullSAD = st.subpelRefine(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, mv, fullSAD)
		// Periodic textures can alias the full-pel raster into a distant
		// basin; a second refinement seeded at zero motion recovers the
		// near-origin subpel optimum when it is better.
		if fullSAD > n*n*2 && (fullMV.Row != 0 || fullMV.Col != 0) {
			zeroSAD := sadBlock(src.Y, ref.Y, lumaPY*src.YStride+lumaPX, lumaPY*src.YStride+lumaPX, src.YStride, n, 1<<30)
			if zmv, zsad := st.subpelRefine(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, motion.Vector{}, zeroSAD); zsad < fullSAD {
				mv, fullSAD = zmv, zsad
			}
		}
	}

	// Reference selection (8x8 leaves only): when LAST left a poor match,
	// probe the GOLDEN anchor; occluded-then-revealed content predicts from
	// the older anchor when the previous frame cannot. Merged leaves stay on
	// LAST (their child agreement was measured against it).
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	refPlanes := ref
	if n == 8 && golden != nil && golden.Y != nil && fullSAD > 8*8*4 {
		gdx, gdy, gsad := fullPelDiamondSearch(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, 8)
		gmv := motion.Vector{Row: int16(gdy * 8), Col: int16(gdx * 8)}
		if gsad > 8*8*2 {
			gmv, gsad = st.subpelRefine(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, 8, gmv, gsad)
		}
		// The anchor must win clearly: LAST keeps the cheaper ref symbol and
		// denser MV predictors.
		if gsad+32 < fullSAD {
			refs.Ref[0] = tile.ReferenceFrameGolden
			refPlanes = *golden
			mv, fullSAD = gmv, gsad
		}
	}

	// Intra fallback (8x8 leaves only): on scene changes and occlusions the
	// best motion match is worse than predicting flat from the already-coded
	// neighbors; compare the motion SAD against a DC-prediction SAD over the
	// source and take the cheaper block type. Larger leaves only exist where
	// child vectors agreed, which presumes the motion model holds.
	if n == 8 {
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
		// Bias toward inter: skip blocks and merged neighbors are cheaper to
		// code, so intra must win clearly.
		if fullSAD > intraSAD+32 {
			return st.encodeIntraPBlock(src, recon, block, scratch)
		}
	}

	// The reference-MV stack and the mode choice must precede prediction:
	// the priced decision may move the coded vector onto a stack predictor,
	// and the materialized prediction, residuals, and reconstruction all
	// have to follow the vector that is actually signaled. The stack reads
	// only motion context this block has not yet marked, so building it
	// before the prefix symbols matches the decoder's later view exactly.
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
	// Mode choice by signaling cost: zero motion keeps the short GLOBALMV
	// cascade; a vector the predictor stack already names codes as
	// NEARESTMV/NEARMV with no motion residual at all; everything else pays
	// for NEWMV plus the joint and component symbols.
	mode := tile.InterModeGlobalMV
	if mv.Row != 0 || mv.Col != 0 {
		mode = tile.InterModeNewMV
		if predRefs, err := stack.Stack.ResolveInterMVReferences(tile.InterModeResult{Mode: tile.InterModeNearestMV}, 0, false, false); err == nil {
			switch mv {
			case predRefs.Nearest[0]:
				mode = tile.InterModeNearestMV
			case predRefs.Near[0]:
				mode = tile.InterModeNearMV
			default:
				nearest := predRefs.Nearest[0]
				if nearest.Row != 0 || nearest.Col != 0 {
					dr := int(mv.Row) - int(nearest.Row)
					if dr < 0 {
						dr = -dr
					}
					dc := int(mv.Col) - int(nearest.Col)
					if dc < 0 {
						dc = -dc
					}
					if dr <= 16 && dc <= 16 {
						mvBits := 4 + bits.Len(uint(dr)) + bits.Len(uint(dc))
						if err := predictInto(st.sadScratch[:n*n], refPlanes.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, nearest, false, false); err == nil {
							nearSAD := 0
							for r := 0; r < n; r += 8 {
								for c := 0; c < n; c += 8 {
									nearSAD += sad8x8DualImpl(src.Y[(lumaPY+r)*src.YStride+lumaPX+c:], src.YStride, st.sadScratch[r*n+c:], n)
								}
							}
							// Two extra cascade symbols pick NEARESTMV.
							if nearSAD+2*st.sadPerBit < fullSAD+mvBits*st.sadPerBit {
								mode = tile.InterModeNearestMV
								mv = nearest
								fullSAD = nearSAD
							}
						}
					}
				}
			}
		}
	}
	// Materialize the three plane predictions with the decoder's convolve so
	// residual coding and reconstruction agree with the decoder bit for bit.
	if err := predictInto(st.predY[:n*n], refPlanes.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, mv, false, false); err != nil {
		return fmt.Errorf("predict luma: %w", err)
	}
	cw, chh := src.Width/2, src.Height/2
	if err := predictInto(st.predU[:cn*cn], refPlanes.U, src.ChromaStride, cw, chh, lumaPX/2, lumaPY/2, cn, mv, true, true); err != nil {
		return fmt.Errorf("predict u: %w", err)
	}
	if err := predictInto(st.predV[:cn*cn], refPlanes.V, src.ChromaStride, cw, chh, lumaPX/2, lumaPY/2, cn, mv, true, true); err != nil {
		return fmt.Errorf("predict v: %w", err)
	}

	// Quantize all three transform blocks up front; a block whose residual
	// quantizes to zero everywhere is coded as skip (no residual symbols, the
	// reconstruction is the prediction itself). A near-perfect match skips
	// the proof: with the luma SAD at a quarter sample per pixel, no
	// realtime quantizer step keeps a coefficient, so the transforms are
	// pure overhead (skip is an encoder choice the decoder honors either
	// way, so this cannot affect parity).
	skip := fullSAD*4 <= n*n
	splitTX := false
	if !skip {
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		lumaZero := st.prepareInterTXB(src.Y, st.predY[:n*n], n, src.YStride, lumaPX, lumaPY, n, st.yQuant, st.lumaQ[:n*n])
		lumaRdD, lumaRdR := st.rdDcode, st.rdRcode
		uZero := st.prepareInterTXB(src.U, st.predU[:cn*cn], cn, src.ChromaStride, lumaPX/2, lumaPY/2, cn, st.uQuant, st.uQ[:cn*cn])
		vZero := st.prepareInterTXB(src.V, st.predV[:cn*cn], cn, src.ChromaStride, lumaPX/2, lumaPY/2, cn, st.vQuant, st.vQ[:cn*cn])
		skip = lumaZero && uZero && vZero
		if !skip {
			// Rate-priced skip (RDCOST shapes): code when distortion saved
			// outweighs the coefficient rate at the working quantizer.
			rdCode := ((st.rdRcode*st.rdMult + 256) >> 9) + (st.rdDcode << 7)
			rdSkip := st.rdDskip << 7
			if rdSkip <= rdCode {
				skip = true
			}
		}
		if !skip && n == 8 && !lumaZero {
			// One-level var-tx split RD: quantize the same luma residual as
			// four quadrant 4x4 TXBs and split when their coded rate-
			// distortion beats the whole 8x8 transform - energy isolated in
			// one quadrant smears across the large DCT but codes as a
			// single small tree (the other children are one txb_skip each,
			// and an 8x8 split to 4x4 costs no extra tree symbols).
			// Measured at 16/32 the split never paid for its four extra
			// child symbols, so only the 8x8 class is evaluated.
			cN := n / 2
			st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
			for i := range 4 {
				dy, dx := (i>>1)*cN, (i&1)*cN
				st.prepareInterTXB(src.Y, st.predY[dy*n+dx:], n, src.YStride, lumaPX+dx, lumaPY+dy, cN, st.yQuant, st.lumaQ2[i*cN*cN:(i+1)*cN*cN])
			}
			costFull := ((lumaRdR*st.rdMult + 256) >> 9) + (lumaRdD << 7)
			costSplit := ((st.rdRcode*st.rdMult + 256) >> 9) + (st.rdDcode << 7)
			if costSplit < costFull {
				splitTX = true
			}
		}
	}

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
	if mode == tile.InterModeNewMV {
		mvRefs, err := stack.Stack.ResolveInterMVReferences(modeResult, 0, false, false)
		if err != nil {
			return fmt.Errorf("resolve mv references: %w", err)
		}
		if err := tile.WriteMotionVector(st.w, &st.mvCDFs, mv, mvRefs.Residual[0], tile.MVSubpelLow); err != nil {
			return fmt.Errorf("motion vector: %w", err)
		}
	}

	hasChroma := true // all coded inter sizes (8x8..32x32) at 4:2:0 carry chroma
	motionResult := tile.InterMotionResult{References: refs, Mode: modeResult}
	motionResult.MV[0] = mv
	if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), motionResult, hasChroma); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), refs, motion.InterpFilters{}); err != nil {
		return fmt.Errorf("mark inter filters: %w", err)
	}

	// Residual phase starts with the var-tx tree, exactly where the decoder
	// reads it: skip blocks mark contexts without symbols, coded blocks write
	// one split decision per node.
	var treeRes tile.TransformTreeResult
	if splitTX {
		treeRes.Split[0] = 1
	}
	if err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: parser.TransformModeSwitchable,
		Inter: true, SkipTransform: skip,
	}, treeRes); err != nil {
		return fmt.Errorf("transform tree: %w", err)
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
		copyPredScratch(recon.Y, st.predY[:n*n], src.YStride, lumaPX, lumaPY, n)
		copyPredScratch(recon.U, st.predU[:cn*cn], src.ChromaStride, lumaPX/2, lumaPY/2, cn)
		copyPredScratch(recon.V, st.predV[:cn*cn], src.ChromaStride, lumaPX/2, lumaPY/2, cn)
		return nil
	}

	// Residual: largest-TX luma with the inter tx_type symbol, then chroma.
	lumaTX, lumaScan := tile.TransformSize8x8, st.scan8
	chromaTX, chromaScan := tile.TransformSize4x4, st.scan4
	switch n {
	case 16:
		lumaTX, lumaScan = tile.TransformSize16x16, st.scan16
		chromaTX, chromaScan = tile.TransformSize8x8, st.scan8
	case 32:
		lumaTX, lumaScan = tile.TransformSize32x32, st.scan32
		chromaTX, chromaScan = tile.TransformSize16x16, st.scan16
	}
	if splitTX {
		// The quadrant TXBs replay in the decoder's recursive order, which
		// is raster for a square one-level split.
		childTX, childScan := tile.TransformSize4x4, st.scan4
		switch n {
		case 16:
			childTX, childScan = tile.TransformSize8x8, st.scan8
		case 32:
			childTX, childScan = tile.TransformSize16x16, st.scan16
		}
		st.interTxTypeReq.Size = childTX
		cN := n / 2
		for i := range 4 {
			dy, dx := (i>>1)*cN, (i&1)*cN
			if err := st.finishInterTXB(recon.Y, st.predY[dy*n+dx:], n, src.YStride, lumaPX+dx, lumaPY+dy, cN, st.yQuant, st.lumaQ2[i*cN*cN:(i+1)*cN*cN], tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: block.Size,
				Size:       childTX,
				X4:         block.X4 + uint8(dx/4),
				Y4:         block.Y4 + uint8(dy/4),
			}, coeffCtx, childScan, st.afterSkipInter); err != nil {
				return fmt.Errorf("luma child txb %d: %w", i, err)
			}
		}
	} else {
		st.interTxTypeReq.Size = lumaTX
		if err := st.finishInterTXB(recon.Y, st.predY[:n*n], n, src.YStride, lumaPX, lumaPY, n, st.yQuant, st.lumaQ[:n*n], tile.CoeffContextRequest{
			Plane:      0,
			PlaneBlock: block.Size,
			Size:       lumaTX,
			X4:         block.X4,
			Y4:         block.Y4,
		}, coeffCtx, lumaScan, st.afterSkipInter); err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
	}
	for plane := 1; plane <= 2; plane++ {
		rdata, pred, qc := recon.U, st.predU[:cn*cn], st.uQ[:cn*cn]
		q := st.uQuant
		if plane == 2 {
			rdata, pred, qc = recon.V, st.predV[:cn*cn], st.vQ[:cn*cn]
			q = st.vQuant
		}
		if err := st.finishInterTXB(rdata, pred, cn, src.ChromaStride, lumaPX/2, lumaPY/2, cn, q, qc, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       chromaTX,
			X4:         block.X4 / 2,
			Y4:         block.Y4 / 2,
		}, coeffCtx, chromaScan, nil); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
	}
	return nil
}

// encodeIntraPBlock codes one 8x8 intra block inside an inter frame: skip,
// the is_inter flag (intra), DC luma and chroma modes through the inter-frame
// y_mode CDFs, the decoder's MarkIntra/MarkChromaIntra context updates, then
// the same DC-predicted TXB pipeline the lossy keyframe path uses.
func (st *lossyEncodeState) encodeIntraPBlock(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("intra skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("intra mark prefix: %w", err)
	}
	// The intra tx-size context reads the pre-mark neighbor flags; take the
	// decoder's per-block snapshot before MarkIntra overwrites the shared
	// above/left slots with this block's own.
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
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	pred := st.predY[:64]
	mode := selectIntraMode8(src.Y, recon.Y, src.YStride, lumaPX, lumaPY, block.HaveTop, block.HaveLeft, pred)
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
	}, 0); err != nil {
		return fmt.Errorf("intra angle delta: %w", err)
	}
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

	if err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: parser.TransformModeSwitchable,
	}, tile.TransformTreeResult{Y: tile.TransformSize8x8}); err != nil {
		return fmt.Errorf("intra transform tree: %w", err)
	}

	st.intraTxTypeReq.Mode = mode
	if err := st.encodeTXBPred(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, 8, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, st.scan8, st.afterSkipIntra, pred); err != nil {
		return fmt.Errorf("intra luma txb: %w", err)
	}
	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma plane block: %w", err)
	}
	for plane := 1; plane <= 2; plane++ {
		data, rdata := src.U, recon.U
		q := st.uQuant
		if plane == 2 {
			data, rdata = src.V, recon.V
			q = st.vQuant
		}
		if err := st.encodeTXBAvail(rdata, data, src.ChromaStride, lumaPX/2, lumaPY/2, 4, q, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       tile.TransformSize4x4,
			X4:         block.X4 / 2,
			Y4:         block.Y4 / 2,
		}, coeffCtx, st.scan4, nil, block.HaveTop, block.HaveLeft); err != nil {
			return fmt.Errorf("intra chroma %d txb: %w", plane, err)
		}
	}
	return nil
}

// predictInto runs the decoder's inter prediction (8-tap convolve, fixed
// EIGHTTAP filters) for one n x n block of plane at frame position (px, py)
// with motion vector mv, writing into dst (stride n). Full-pel vectors reduce
// to copies inside the kernel, so one path serves both cases bit-exactly.
func predictInto(dst []byte, refPlane []byte, stride, width, height, px, py, n int, mv motion.Vector, ssX, ssY bool) error {
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, ssX, ssY)
	if err != nil {
		return err
	}
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	ref := frame.Plane{Pix: refPlane, Stride: stride, Width: width, Height: height}
	return motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstPlane, ref, 1, 8, 0, 0, refX, refY, n, n, subX, subY, motion.InterpFilters{})
}

// subpelRefine improves a full-pel luma motion vector in two bounded
// stages. Stage one probes the four half-pel neighbors with the decoder's
// exact convolve - integer SAD surfaces cannot localize a subpel optimum on
// detailed content, so these four predictions are the irreducible cost.
// Stage two fits a parabola per axis through the EXACT half-pel SADs and
// verifies only the quarter-pel positions surrounding the fitted minimum,
// at most two more predictions. Six convolves bound the search against the
// eight to sixteen of a greedy descent; the coded prediction always goes
// through the exact convolve later, so search shape cannot affect parity.
func (st *lossyEncodeState) subpelRefine(src, refPlane []byte, stride, width, height, px, py, n int, mv motion.Vector, bestSAD int) (motion.Vector, int) {
	// The probes sit within one pixel of the full-pel start, so geometry
	// validation hoists into the prober; blocks near the frame edge fall
	// back to the fully validated predictor per probe.
	st.prober.Init(frame.Plane{
		Pix: refPlane, Stride: stride, Width: width, Height: height,
	}, px+int(mv.Col)>>3, py+int(mv.Row)>>3, n)
	startMV := mv
	exact := func(cand motion.Vector) int {
		if !st.prober.Predict(st.sadScratch[:n*n], motion.Vector{Row: cand.Row - startMV.Row, Col: cand.Col - startMV.Col}) {
			if err := predictInto(st.sadScratch[:n*n], refPlane, stride, width, height, px, py, n, cand, false, false); err != nil {
				return -1
			}
		}
		base := py*stride + px
		if n == 8 {
			return sad8x8DualImpl(src[base:], stride, st.sadScratch[:], 8)
		}
		s := 0
		for r := range n {
			row := base + r*stride
			for c := range n {
				d := int(src[row+c]) - int(st.sadScratch[r*n+c])
				if d < 0 {
					d = -d
				}
				s += d
			}
		}
		return s
	}
	start := mv
	center := bestSAD
	// Stage 1: exact half-pel cross.
	var half [4]int // left, right, up, down
	offs := [4]motion.Vector{
		{Row: start.Row, Col: start.Col - 4},
		{Row: start.Row, Col: start.Col + 4},
		{Row: start.Row - 4, Col: start.Col},
		{Row: start.Row + 4, Col: start.Col},
	}
	for i, cand := range offs {
		s := exact(cand)
		half[i] = s
		if s >= 0 && s < bestSAD {
			bestSAD, mv = s, cand
		}
	}
	// Stage 2: per-axis parabola through the exact half-pel SADs locates the
	// quarter-pel minimum; verify its surrounding even-1/8 positions.
	quarterAxis := func(sl, sr int) int {
		if sl < 0 || sr < 0 {
			return 0
		}
		den := sl + sr - 2*center
		if den <= 0 {
			return 0
		}
		est := (sl - sr) * 2 / den // half-pel steps are 4 eighths
		if est > 4 {
			est = 4
		}
		if est < -4 {
			est = -4
		}
		return est
	}
	estX := quarterAxis(half[0], half[1])
	estY := quarterAxis(half[2], half[3])
	for _, e := range [2][2]int{{estX &^ 1, estY &^ 1}, {(estX + 1) &^ 1, (estY + 1) &^ 1}} {
		if e[0] == 0 && e[1] == 0 {
			continue
		}
		cand := motion.Vector{Row: start.Row + int16(e[1]), Col: start.Col + int16(e[0])}
		if cand == mv || cand == start {
			continue
		}
		if s := exact(cand); s >= 0 && s < bestSAD {
			bestSAD, mv = s, cand
		}
	}
	return mv, bestSAD
}

// txScaleForSize is get_tx_scale for the square transforms the encoder
// emits: TX_32X32 dequantizes with one extra right shift, smaller sizes none.
func txScaleForSize(n int) uint8 {
	if n >= 32 {
		return 1
	}
	return 0
}

// sadBlock is the generic n x n SAD with the 8x8 kernel fast path; limit is
// an early-exit hint as in sad8x8Impl.
func sadBlock(src, ref []byte, base, refBase, stride, n, limit int) int {
	switch n {
	case 8:
		return sad8x8Impl(src[base:], ref[refBase:], stride, limit)
	case 16:
		return sad16x16Impl(src[base:], ref[refBase:], stride)
	case 32:
		return sad16x16Impl(src[base:], ref[refBase:], stride) +
			sad16x16Impl(src[base+16:], ref[refBase+16:], stride) +
			sad16x16Impl(src[base+16*stride:], ref[refBase+16*stride:], stride) +
			sad16x16Impl(src[base+16*stride+16:], ref[refBase+16*stride+16:], stride)
	}
	total := 0
	for r := range n {
		row := base + r*stride
		refRow := refBase + r*stride
		for c := range n {
			d := int(src[row+c]) - int(ref[refRow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
		if total >= limit {
			return total
		}
	}
	return total
}

// copyPredScratch copies a materialized n x n prediction into the recon plane
// for skipped blocks.
func copyPredScratch(reconPlane, pred []byte, stride, px, py, n int) {
	for r := range n {
		copy(reconPlane[(py+r)*stride+px:(py+r)*stride+px+n], pred[r*n:r*n+n])
	}
}

// prepareInterTXB computes the quantized coefficients of one square n x n
// motion-compensated residual block (pred holds the prediction at stride n)
// and reports whether they are all zero.
// prepareInterTXB computes the quantized coefficients of one square n x n
// motion-compensated residual block and reports whether they are all zero,
// accumulating the block's rate-distortion terms for the skip decision:
// transform-domain distortion when coded (sum of squared dequantization
// errors), when skipped (sum of squared coefficients), and a coarse
// coefficient rate estimate in 512-units-per-bit.
func (st *lossyEncodeState) prepareInterTXB(srcPlane, pred []byte, predStride, stride, px, py, n int, q quantize.Quantizer, qcoeff []int16) bool {
	residual := &st.resScratch
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			residual[r*n+c] = int16(srcPlane[row+c]) - int16(pred[r*predStride+c])
		}
	}
	tran := &st.tranScratch
	switch n {
	case 4:
		if err := transform.ForwardDCT4x4(tran[:16], 4, residual[:16], 4); err != nil {
			return false
		}
	case 8:
		if err := transform.ForwardDCT8x8(tran[:64], 8, residual[:64], 8); err != nil {
			return false
		}
	case 16:
		if err := transform.ForwardDCT16x16(tran[:256], 16, residual[:256], 16); err != nil {
			return false
		}
	case 32:
		if err := transform.ForwardDCT32x32(tran[:1024], 32, residual[:1024], 32); err != nil {
			return false
		}
	default:
		return false
	}
	ts := txScaleForSize(n)
	if err := quantize.QuantizeBlockScaledB(qcoeff, n, tran[:n*n], n, n, n, q, ts); err != nil {
		return false
	}
	allZero := true
	for i, v := range qcoeff {
		c := int64(tran[i])
		st.rdDskip += c * c
		if v == 0 {
			st.rdDcode += c * c
			continue
		}
		allZero = false
		step := int64(q.AC)
		if i == 0 {
			step = int64(q.DC)
		}
		dq := (int64(v) * step) >> ts
		e := c - dq
		st.rdDcode += e * e
		level := v
		if level < 0 {
			level = -level
		}
		// Coarse rate: two bits of map overhead plus the level magnitude
		// class, in 512-units-per-bit.
		st.rdRcode += int64(2+bits.Len16(uint16(level))) << 9
	}
	return allZero
}

// finishInterTXB codes the prepared coefficients and rebuilds the recon block
// from the prediction through the decoder's dequant + inverse transform.
func (st *lossyEncodeState) finishInterTXB(reconPlane, pred []byte, predStride, stride, px, py, n int, q quantize.Quantizer, qcoeff []int16,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error) error {

	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip); err != nil {
		return err
	}
	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n*n], n, qcoeff, n, n, n, q, txScaleForSize(n), 8); err != nil {
		return err
	}
	res := &st.invResidual
	if err := transform.InverseDCTBlock(res[:n*n], n, dq[:n*n], n, st.invScratch[:n*n], transform.Size{Width: uint8(n), Height: uint8(n)}); err != nil {
		return err
	}
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			v := int(pred[r*predStride+c]) + int(res[r*n+c])
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
func fullPelDiamondSearch(src, ref []byte, stride, width, height, px, py, n int) (int, int, int) {
	// sad dispatches to the architecture SAD kernel; limit is an early-exit
	// hint the kernel may ignore, so callers compare the return value.
	sad := func(dx, dy, limit int) int {
		base := py*stride + px
		refBase := (py+dy)*stride + px + dx
		switch n {
		case 8:
			return sad8x8Impl(src[base:], ref[refBase:], stride, limit)
		case 16:
			return sad16x16Impl(src[base:], ref[refBase:], stride)
		case 32:
			return sad16x16Impl(src[base:], ref[refBase:], stride) +
				sad16x16Impl(src[base+16:], ref[refBase+16:], stride) +
				sad16x16Impl(src[base+16*stride:], ref[refBase+16*stride:], stride) +
				sad16x16Impl(src[base+16*stride+16:], ref[refBase+16*stride+16:], stride)
		}
		total := 0
		for r := range n {
			row := base + r*stride
			refRow := refBase + r*stride
			for c := range n {
				d := int(src[row+c]) - int(ref[refRow+c])
				if d < 0 {
					d = -d
				}
				total += d
			}
			if total >= limit {
				return total
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
	bestSAD := sad(0, 0, 1<<30)
	// Static-content fast path: when zero motion is already a near-perfect
	// match (below ~2 per sample, the quantizer's noise floor at realtime
	// qindexes), searching cannot pay for its own cost.
	if bestSAD <= n*n*2 {
		return 0, 0, bestSAD
	}
	// Coarse even-step raster with row-granular early exit, then a +-2 even
	// diamond refinement.
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		for dx := minDX &^ 1; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sad(dx, dy, bestSAD); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	for _, cand := range [4][2]int{{bestDX + 2, bestDY}, {bestDX - 2, bestDY}, {bestDX, bestDY + 2}, {bestDX, bestDY - 2}} {
		dx, dy := cand[0], cand[1]
		if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
			continue
		}
		if s := sad(dx, dy, bestSAD); s < bestSAD {
			bestSAD, bestDX, bestDY = s, dx, dy
		}
	}
	return bestDX, bestDY, bestSAD
}
