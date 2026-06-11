package encoder

import (
	"fmt"
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
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

	var groupScratch, outScratch []byte
	out, err := assembleInterTU(seq, header, []TilePayload{{Data: tilePayload}}, 0, &groupScratch, &outScratch)
	if err != nil {
		return nil, SourceFrame420{}, err
	}
	return out, recon, nil
}

// assembleInterTU wraps one coded tile into a TD + inter frame header + tile
// group temporal unit.
func assembleInterTU(seq SequenceHeader, header InterFrameHeaderParams, tilePayloads []TilePayload, temporalID uint8, groupScratch, outScratch *[]byte) ([]byte, error) {
	endTile := uint16(len(tilePayloads) - 1)
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, endTile, tilePayloads)
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	if cap(*groupScratch) < groupSize {
		*groupScratch = make([]byte, 0, groupSize+groupSize/2)
	}
	group := (*groupScratch)[:0]
	group, err = AppendTileGroupPayload(group, header.Tile, 0, endTile, tilePayloads)
	if err != nil {
		return nil, fmt.Errorf("append tile group: %w", err)
	}
	*groupScratch = group
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
	total := tdSize + frameSize + groupOBUSize
	if cap(*outScratch) < total {
		*outScratch = make([]byte, 0, total+total/2)
	}
	out := (*outScratch)[:0]
	*outScratch = out
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
	st                   lossyEncodeState
	partCDFs             tile.PartitionCDFs
	refCDFs              tile.InterRefCDFs
	modeCDFs             tile.InterModeCDFs
	scratch              tile.BlockLoopScratch
	carrier              tile.BlockLoopContextCarrier
	writerBuf            []byte
	writer               entropy.Writer
	decisionStats        EncoderDecisionStats
	decisionStatsEnabled bool
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
	if err := st.initScans(); err != nil {
		return err
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

// initScans allocates the coder's scan tables and coefficient scratch on
// first use; later calls are free, so both the inter and keyframe tile
// paths share one set per coder.
func (st *lossyEncodeState) initScans() error {
	if st.scan8 != nil {
		return nil
	}
	for _, sq := range []struct {
		dst *[]int16
		n   uint8
	}{
		{&st.scan4, 4}, {&st.scan8, 8}, {&st.scan16, 16}, {&st.scan32, 32},
	} {
		*sq.dst = make([]int16, int(sq.n)*int(sq.n))
		inv := make([]int16, int(sq.n)*int(sq.n))
		if err := transform.FillDefaultScan(*sq.dst, inv, transform.Size{Width: sq.n, Height: sq.n}, transform.Class2D); err != nil {
			return err
		}
	}
	for _, rs := range []struct {
		dst  *[]int16
		w, h uint8
	}{
		{&st.scan16x8, 16, 8}, {&st.scan8x16, 8, 16},
		{&st.scan8x4, 8, 4}, {&st.scan4x8, 4, 8},
		{&st.scan32x16, 32, 16}, {&st.scan16x32, 16, 32},
	} {
		*rs.dst = make([]int16, int(rs.w)*int(rs.h))
		inv := make([]int16, int(rs.w)*int(rs.h))
		if err := transform.FillDefaultScan(*rs.dst, inv, transform.Size{Width: rs.w, Height: rs.h}, transform.Class2D); err != nil {
			return err
		}
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize32x32)
	if err != nil {
		return err
	}
	st.levels = make([]uint8, scratchLen)
	st.invScratch = make([]int32, 1024)
	return nil
}

func encodePFrameTile(src SourceFrame420, ref SourceFrame420, recon *SourceFrame420, qIndex uint8) ([]byte, error) {
	var pc pframeCoder
	return pc.encodeTile(src, ref, nil, recon, qIndex, nil, parser.ReferenceModeSingle, 0, uint16(src.Width/4))
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
func (pc *pframeCoder) encodeTile(src SourceFrame420, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, qIndex uint8, prev *frameCDFs, referenceMode parser.ReferenceMode, miColStart, miColEnd uint16) ([]byte, error) {
	miCols := uint16(src.Width / 4)
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB
	if err := pc.reset(qIndex, rootCols, prev); err != nil {
		return nil, err
	}
	st := &pc.st
	st.decisionStats = nil
	if pc.decisionStatsEnabled {
		pc.decisionStats.Reset()
		pc.decisionStats.Tiles = 1
		st.decisionStats = &pc.decisionStats
	}

	pc.writer.Reset(pc.writerBuf[:0])
	st.w = &pc.writer
	st.interTxTypeReq = tile.InterTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		QIndexKnown: true,
		QIndex:      qIndex,
	}
	st.interTxType = transform.TypeDCTDCT
	st.intraTxTypeReq = tile.IntraTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		Mode:        tile.IntraModeDC,
		QIndexKnown: true,
		QIndex:      qIndex,
	}
	// The after-skip hooks close over the stable state pointer only, so
	// they persist across frames instead of allocating per tile.
	if st.afterSkipInter == nil {
		st.afterSkipInter = func() error {
			return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, st.interTxType)
		}
		st.afterSkipIntra = func() error {
			return tile.WriteIntraTransformType(st.w, &st.txCDFs, st.intraTxTypeReq, transform.TypeDCTDCT)
		}
	}

	// The mode and partition searches trial-code against throwaway
	// contexts; they re-arm lazily on first use each frame. The buffer
	// exists from the first tile so no frame mid-stream pays it.
	st.trialReady = false
	if cap(st.trialBuf) == 0 {
		st.trialBuf = make([]byte, 1<<14)
	}
	st.keyMIColEnd = uint32(miColEnd)
	st.keyMIRowEnd = uint32(miRows)
	st.keyVisW, st.keyVisH = src.Width, src.Height

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
	st.grid64Cols = (int(miCols) + 15) / 16
	grid64Rows := (int(miRows) + 15) / 16
	if len(st.mv64Grid) < st.grid64Cols*grid64Rows {
		st.mv64Grid = make([]motion.Vector, st.grid64Cols*grid64Rows)
		st.sad64Grid = make([]int32, st.grid64Cols*grid64Rows)
	}
	for i := range st.sad64Grid[:st.grid64Cols*grid64Rows] {
		st.sad64Grid[i] = -1
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
	// A 64x64 merge saves up to three 32-tier block headers and the extra
	// partition symbols beneath them.
	const mergeBias64 = 576
	// evaluate16 fills the 16x16 and child 8x8 motion grids for the 16x16
	// region at (px, py) — searching only on first use — and returns the
	// merged SAD and the sum of the four child SADs.
	// evaluate16Merged searches only the merged 16x16; the children search
	// on demand so blocks the dead-zone bar merges outright never pay for
	// four child searches they cannot use.
	evaluate16Merged := func(px, py int) int {
		idx16 := (py/16)*st.grid16Cols + px/16
		if st.sad16Grid[idx16] < 0 {
			seedDX, seedDY, reach := 0, 0, fullPelReach
			if st.hme != nil {
				var trusted bool
				seedDX, seedDY, trusted = st.hme.seedAt(px, py)
				if trusted {
					reach = fullPelReachTrusted
				}
			}
			dx16, dy16, sad16 := fullPelDiamondSearchSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, px, py, 16, seedDX, seedDY, reach)
			if (seedDX > reach || seedDX < -reach || seedDY > reach || seedDY < -reach) && sad16 > 16*16*2 {
				// The seeded window excludes the fine zero neighborhood
				// when the regional vector is large; blocks that disagree
				// with their region (mover boundaries) refall back to the
				// zero-centered window when the seeded match stays poor.
				if zx, zy, zsad := fullPelDiamondSearchSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, px, py, 16, 0, 0, fullPelReach); zsad < sad16 {
					dx16, dy16, sad16 = zx, zy, zsad
				}
			}
			st.mv16Grid[idx16] = motion.Vector{Row: int16(dy16 * 8), Col: int16(dx16 * 8)}
			st.sad16Grid[idx16] = int32(sad16)
		}
		return int(st.sad16Grid[idx16])
	}
	children8 := func(px, py int) int {
		seedDX, seedDY, reach := 0, 0, fullPelReach
		if st.hme != nil {
			var trusted bool
			seedDX, seedDY, trusted = st.hme.seedAt(px, py)
			if trusted {
				reach = fullPelReachTrusted
			}
		}
		sum8 := 0
		for _, off := range [4][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
			cx, cy := px+off[0], py+off[1]
			idx8 := (cy/8)*st.grid8Cols + cx/8
			if st.sad8Grid[idx8] < 0 {
				dx, dy, sad := fullPelDiamondSearchSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, cx, cy, 8, seedDX, seedDY, reach)
				st.mv8Grid[idx8] = motion.Vector{Row: int16(dy * 8), Col: int16(dx * 8)}
				st.sad8Grid[idx8] = int32(sad)
			}
			sum8 += int(st.sad8Grid[idx8])
		}
		return sum8
	}
	evaluate16 := func(px, py int) (int, int) {
		return evaluate16Merged(px, py), children8(px, py)
	}
	decideCore := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		if !haveRight || !haveBottom {
			return tile.PartitionSplit, nil
		}
		px, py := int(miCol)*4, int(miRow)*4
		switch level {
		case tile.BlockLevel64x64:
			// One tier above the 32 merge with the same shape: all sixteen
			// 16x16 descendants must settle on one full-pel vector, then a
			// single merged probe prices the whole superblock as one block.
			if px+64 > src.Width || py+64 > src.Height {
				return tile.PartitionSplit, nil
			}
			childCost := 0
			var mv64 motion.Vector
			for i, off := range [16][2]int{
				{0, 0}, {16, 0}, {32, 0}, {48, 0},
				{0, 16}, {16, 16}, {32, 16}, {48, 16},
				{0, 32}, {16, 32}, {32, 32}, {48, 32},
				{0, 48}, {16, 48}, {32, 48}, {48, 48},
			} {
				sad16, sum8 := evaluate16(px+off[0], py+off[1])
				childCost += min(sad16, sum8)
				cmv := st.mv16Grid[((py+off[1])/16)*st.grid16Cols+(px+off[0])/16]
				if i == 0 {
					mv64 = cmv
				} else if cmv != mv64 {
					return tile.PartitionSplit, nil
				}
			}
			dx, dy := int(mv64.Col)/8, int(mv64.Row)/8
			if py+dy < 0 || px+dx < 0 || py+dy+64 > src.Height || px+dx+64 > src.Width {
				return tile.PartitionSplit, nil
			}
			base := py*src.YStride + px
			refBase := (py+dy)*src.YStride + px + dx
			sad64 := 0
			for _, q := range [4][2]int{{0, 0}, {32, 0}, {0, 32}, {32, 32}} {
				qoff := q[1]*src.YStride + q[0]
				sad64 += sadBlock(src.Y, ref.Y, base+qoff, refBase+qoff, src.YStride, 32, 1<<30)
			}
			if sad64 <= childCost+mergeBias64 {
				idx64 := (py/64)*st.grid64Cols + px/64
				st.mv64Grid[idx64] = mv64
				st.sad64Grid[idx64] = int32(sad64)
				return tile.PartitionNone, nil
			}
			return tile.PartitionSplit, nil
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
			// Rect halves one tier up from the 16-tier shape: a 32x16
			// (16x32) half coded with one vector saves two blocks' syntax
			// and gains the whole-half transform.
			i0 := (py/16)*st.grid16Cols + px/16
			idx := [4]int{i0, i0 + 1, i0 + st.grid16Cols, i0 + st.grid16Cols + 1}
			const halfBias32 = 128
			const halfSADGate32 = 32 * 16 * 2
			halfSAD := func(cx, cy int, mv motion.Vector) int {
				dx, dy := int(mv.Col)/8, int(mv.Row)/8
				if cy+dy < 0 || cx+dx < 0 || cy+dy+16 > src.Height || cx+dx+16 > src.Width {
					return 1 << 30
				}
				return sadBlock(src.Y, ref.Y, cy*src.YStride+cx, (cy+dy)*src.YStride+cx+dx, src.YStride, 16, 1<<30)
			}
			tryHalf := func(ia, ib int, ax, ay, bx, by int) bool {
				mva := st.mv16Grid[idx[ia]]
				mvb := st.mv16Grid[idx[ib]]
				sa, sb := int(st.sad16Grid[idx[ia]]), int(st.sad16Grid[idx[ib]])
				best, bestA, bestB := mva, sa, halfSAD(bx, by, mva)
				if mvb != mva {
					ca, cb := halfSAD(ax, ay, mvb), sb
					if ca+cb < bestA+bestB {
						best, bestA, bestB = mvb, ca, cb
					}
				}
				if bestA+bestB > sa+sb+halfBias32 || bestA+bestB > halfSADGate32 {
					return false
				}
				st.mv16Grid[idx[ia]], st.mv16Grid[idx[ib]] = best, best
				st.sad16Grid[idx[ia]], st.sad16Grid[idx[ib]] = int32(bestA), int32(bestB)
				return true
			}
			if tryHalf(0, 1, px, py, px+16, py) && tryHalf(2, 3, px, py+16, px+16, py+16) {
				return tile.PartitionH, nil
			}
			if tryHalf(0, 2, px, py, px, py+16) && tryHalf(1, 3, px+16, py, px+16, py+16) {
				return tile.PartitionV, nil
			}
			return tile.PartitionSplit, nil
		case tile.BlockLevel16x16:
			sad16 := evaluate16Merged(px, py)
			// Inside the quantizer's dead zone both shapes code almost
			// nothing and headers favor the merge outright - and the four
			// child searches never need to run.
			skipBar := 16 * 16 * int(st.yQuant.AC) / 24
			if sad16 <= skipBar {
				return tile.PartitionNone, nil
			}
			sum8 := children8(px, py)
			margin := sad16 - (sum8 + mergeBias16)
			if margin <= 0 {
				return tile.PartitionNone, nil
			}
			// Rect halves: a 16x8 (8x16) half coded with one vector saves a
			// block's mode/MV syntax and gains the whole-half transform.
			// Each half probes its two child vectors and merges when the
			// best whole-half SAD stays within a one-block syntax bias of
			// the children's independent matches - the merge logic one tier
			// below 16, with the same fixed-bias shape. The absolute gate
			// keeps halves whose children would have wanted subpel
			// refinement coded as separate 8x8s instead.
			i0 := (py/8)*st.grid8Cols + px/8
			idx := [4]int{i0, i0 + 1, i0 + st.grid8Cols, i0 + st.grid8Cols + 1}
			const halfBias = 32
			const halfSADGate = 16 * 8 * 2
			childSAD := func(cx, cy int, mv motion.Vector) int {
				dx, dy := int(mv.Col)/8, int(mv.Row)/8
				if cy+dy < 0 || cx+dx < 0 || cy+dy+8 > src.Height || cx+dx+8 > src.Width {
					return 1 << 30
				}
				return sadBlock(src.Y, ref.Y, cy*src.YStride+cx, (cy+dy)*src.YStride+cx+dx, src.YStride, 8, 1<<30)
			}
			// tryHalf probes the half spanning children ia/ib (child pixel
			// origins given) and on success rewrites the pair's grid slots
			// with the shared vector so the leaf coder reads them directly.
			tryHalf := func(ia, ib int, ax, ay, bx, by int) bool {
				mva := st.mv8Grid[idx[ia]]
				mvb := st.mv8Grid[idx[ib]]
				sa, sb := int(st.sad8Grid[idx[ia]]), int(st.sad8Grid[idx[ib]])
				best, bestA, bestB := mva, sa, childSAD(bx, by, mva)
				if mvb != mva {
					ca, cb := childSAD(ax, ay, mvb), sb
					if ca+cb < bestA+bestB {
						best, bestA, bestB = mvb, ca, cb
					}
				}
				if bestA+bestB > sa+sb+halfBias || bestA+bestB > halfSADGate {
					return false
				}
				st.mv8Grid[idx[ia]], st.mv8Grid[idx[ib]] = best, best
				st.sad8Grid[idx[ia]], st.sad8Grid[idx[ib]] = int32(bestA), int32(bestB)
				return true
			}
			if tryHalf(0, 1, px, py, px+8, py) && tryHalf(2, 3, px, py+8, px+8, py+8) {
				return tile.PartitionH, nil
			}
			if tryHalf(0, 2, px, py, px, py+8) && tryHalf(1, 3, px+8, py, px+8, py+8) {
				return tile.PartitionV, nil
			}
			return tile.PartitionSplit, nil
		}
		return tile.PartitionSplit, nil
	}
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		partition, err := decideCore(level, ctx, miCol, miRow, haveRight, haveBottom)
		if err == nil && st.decisionStats != nil {
			st.decisionStats.notePartition(level, partition)
		}
		return partition, err
	}

	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodePBlock(src, ref, golden, recon, block, scratch, &pc.refCDFs, &pc.modeCDFs, referenceMode, walkReq, miCols, miRows)
	}
	if err := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return pc.writer.Finish()
}

// encodePBlock codes one inter block (8x8 or 16x16) with residual: the inter
// mode symbols in the decoder's order, then the largest-TX luma residual (with
// the inter tx_type symbol after txb_skip) and two chroma residuals against
// the reference reconstruction.
func (st *lossyEncodeState) encodePBlock(src, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	refCDFs *tile.InterRefCDFs, interModeCDFs *tile.InterModeCDFs,
	referenceMode parser.ReferenceMode, walkReq tile.BlockWalkRequest, miCols, miRows uint16) error {

	var bw, bh int
	switch block.Size {
	case tile.BlockSize8x8:
		bw, bh = 8, 8
	case tile.BlockSize16x16:
		bw, bh = 16, 16
	case tile.BlockSize32x32:
		bw, bh = 32, 32
	case tile.BlockSize64x64:
		bw, bh = 64, 64
	case tile.BlockSize16x8:
		bw, bh = 16, 8
	case tile.BlockSize8x16:
		bw, bh = 8, 16
	case tile.BlockSize32x16:
		bw, bh = 32, 16
	case tile.BlockSize16x32:
		bw, bh = 16, 32
	default:
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	n := bw // square tier size; rect blocks take dedicated paths below
	cbw, cbh := bw/2, bh/2
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
	case bw != bh:
		// Rect halves exist only where the partition decider proved the two
		// child vectors equal; the shared vector and the summed child SADs
		// describe the half exactly.
		if bw >= 32 || bh >= 32 {
			i0 := (lumaPY/16)*st.grid16Cols + lumaPX/16
			i1 := i0 + 1
			if bh > bw {
				i1 = i0 + st.grid16Cols
			}
			if st.sad16Grid[i0] >= 0 && st.sad16Grid[i1] >= 0 {
				mv = st.mv16Grid[i0]
				fullSAD = int(st.sad16Grid[i0]) + int(st.sad16Grid[i1])
			}
			break
		}
		i0 := (lumaPY/8)*st.grid8Cols + lumaPX/8
		i1 := i0 + 1
		if bh > bw {
			i1 = i0 + st.grid8Cols
		}
		if st.sad8Grid[i0] >= 0 && st.sad8Grid[i1] >= 0 {
			mv = st.mv8Grid[i0]
			fullSAD = int(st.sad8Grid[i0]) + int(st.sad8Grid[i1])
		}
	case n == 64:
		idx := (lumaPY/64)*st.grid64Cols + lumaPX/64
		if st.sad64Grid[idx] >= 0 {
			mv, fullSAD = st.mv64Grid[idx], int(st.sad64Grid[idx])
		}
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
		if bw != bh {
			return fmt.Errorf("encoder: rect block %+v without scored children", block)
		}
		seedDX, seedDY, reach := 0, 0, fullPelReach
		if st.hme != nil {
			var trusted bool
			seedDX, seedDY, trusted = st.hme.seedAt(lumaPX, lumaPY)
			if trusted {
				reach = fullPelReachTrusted
			}
		}
		dx, dy, sad := fullPelDiamondSearchSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, seedDX, seedDY, reach)
		mv, fullSAD = motion.Vector{Row: int16(dy * 8), Col: int16(dx * 8)}, sad
	}
	if bw == bh && fullSAD > n*n*2 {
		fullMV := mv
		mv, fullSAD = st.subpelRefine(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, mv, fullSAD)
		// Periodic textures can alias the full-pel raster into a distant
		// basin; a second refinement seeded at zero motion recovers the
		// near-origin subpel optimum when it is better.
		// The zero probe is cheap; the second subpel refinement only pays
		// when zero is already competitive with the searched vector.
		if fullSAD > n*n*2 && (fullMV.Row != 0 || fullMV.Col != 0) {
			zeroSAD := sadBlock(src.Y, ref.Y, lumaPY*src.YStride+lumaPX, lumaPY*src.YStride+lumaPX, src.YStride, n, 1<<30)
			if zeroSAD < fullSAD*2 {
				if zmv, zsad := st.subpelRefine(src.Y, ref.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, n, motion.Vector{}, zeroSAD); zsad < fullSAD {
					mv, fullSAD = zmv, zsad
				}
			}
		}
	}

	// Reference selection: when LAST left a poor match, probe the GOLDEN
	// anchor; occluded-then-revealed content predicts from the older anchor
	// when the previous frame cannot. Compound stays limited to 8x8 leaves.
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	refPlanes := ref
	compound := false
	var compoundMV [2]motion.Vector
	goldenEligible := (bw == 8 && bh == 8) ||
		(bw == 16 && bh == 16) || (bw == 32 && bh == 32) ||
		(bw == 16 && bh == 8) || (bw == 8 && bh == 16) ||
		(bw == 32 && bh == 16) || (bw == 16 && bh == 32)
	if golden != nil && golden.Y != nil && goldenEligible && fullSAD > bw*bh*4 {
		lastMV, lastSAD := mv, fullSAD
		var gmv motion.Vector
		gsad := 1 << 30
		if bw == 8 {
			gdx, gdy, s := fullPelDiamondSearch(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, 8)
			gmv, gsad = motion.Vector{Row: int16(gdy * 8), Col: int16(gdx * 8)}, s
			if gsad > 8*8*2 {
				gmv, gsad = st.subpelRefine(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, 8, gmv, gsad)
			}
		} else {
			base := lumaPY*src.YStride + lumaPX
			gsad = sadRectBlock(src.Y, golden.Y, base, base, src.YStride, bw, bh, lastSAD)
		}
		if bw == 8 && referenceMode == parser.ReferenceModeSelect && gsad+32 >= lastSAD && gsad <= lastSAD+8*8*4 {
			if err := predictCompoundInto(st.sadScratch[:64], ref.Y, ref.YStride, golden.Y, golden.YStride, src.Width, src.Height, lumaPX, lumaPY, 8, 8, lastMV, gmv, false, false, &st.compBuf0, &st.compBuf1, &st.compScratch); err == nil {
				compoundSAD := sad8x8DualImpl(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, st.sadScratch[:64], 8)
				compoundBias := 64 + 12*st.sadPerBit
				if compoundSAD+compoundBias < fullSAD {
					compound = true
					compoundMV = [2]motion.Vector{lastMV, gmv}
					fullSAD = compoundSAD
				}
			}
		}
		// The anchor must win clearly: LAST keeps the cheaper ref symbol and
		// denser MV predictors.
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

	// Intra fallback (8x8 leaves only): on scene changes and occlusions the
	// best motion match is worse than predicting flat from the already-coded
	// neighbors; compare the motion SAD against a DC-prediction SAD over the
	// source and take the cheaper block type. Larger leaves only exist where
	// child vectors agreed, which presumes the motion model holds.
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
	// for NEWMV plus the joint and component symbols. Compound is deliberately
	// conservative in this first encoder path: only a strong LAST+GOLDEN
	// predictor win reaches here, so code the actual pair directly.
	modeResult := tile.InterModeResult{Mode: tile.InterModeGlobalMV}
	if refs.Compound {
		modeResult = tile.InterModeResult{Compound: true, CompoundMode: tile.CompoundInterModeNewNew}
		if compoundMV[0] == (motion.Vector{}) && compoundMV[1] == (motion.Vector{}) {
			modeResult.CompoundMode = tile.CompoundInterModeGlobalGlobal
		}
	} else {
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
					// Pricing the predictor swap needs a prediction and a SAD;
					// blocks already in skip territory keep NEWMV without it.
					if (nearest.Row != 0 || nearest.Col != 0) && fullSAD > bw*bh {
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
							if err := predictInto(st.sadScratch[:bw*bh], refPlanes.Y, refPlanes.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, nearest, false, false); err == nil {
								nearSAD := 0
								for r := 0; r < bh; r += 8 {
									for c := 0; c < bw; c += 8 {
										nearSAD += sad8x8DualImpl(src.Y[(lumaPY+r)*src.YStride+lumaPX+c:], src.YStride, st.sadScratch[r*bw+c:], bw)
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
		modeResult.Mode = mode
	}
	// Materialize the three plane predictions with the decoder's convolve so
	// residual coding and reconstruction agree with the decoder bit for bit.
	halfW, halfH := src.Width/2, src.Height/2
	if refs.Compound {
		if err := predictCompoundInto(st.predY[:bw*bh], ref.Y, ref.YStride, golden.Y, golden.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, compoundMV[0], compoundMV[1], false, false, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
			return fmt.Errorf("predict compound luma: %w", err)
		}
		if err := predictCompoundInto(st.predU[:cbw*cbh], ref.U, ref.ChromaStride, golden.U, golden.ChromaStride, halfW, halfH, lumaPX/2, lumaPY/2, cbw, cbh, compoundMV[0], compoundMV[1], true, true, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
			return fmt.Errorf("predict compound u: %w", err)
		}
		if err := predictCompoundInto(st.predV[:cbw*cbh], ref.V, ref.ChromaStride, golden.V, golden.ChromaStride, halfW, halfH, lumaPX/2, lumaPY/2, cbw, cbh, compoundMV[0], compoundMV[1], true, true, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
			return fmt.Errorf("predict compound v: %w", err)
		}
	} else {
		if err := predictInto(st.predY[:bw*bh], refPlanes.Y, refPlanes.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false); err != nil {
			return fmt.Errorf("predict luma: %w", err)
		}
		if err := predictInto(st.predU[:cbw*cbh], refPlanes.U, refPlanes.ChromaStride, halfW, halfH, lumaPX/2, lumaPY/2, cbw, cbh, mv, true, true); err != nil {
			return fmt.Errorf("predict u: %w", err)
		}
		if err := predictInto(st.predV[:cbw*cbh], refPlanes.V, refPlanes.ChromaStride, halfW, halfH, lumaPX/2, lumaPY/2, cbw, cbh, mv, true, true); err != nil {
			return fmt.Errorf("predict v: %w", err)
		}
	}

	// Quantize all three transform blocks up front; a block whose residual
	// quantizes to zero everywhere is coded as skip (no residual symbols, the
	// reconstruction is the prediction itself). A near-perfect match skips
	// the proof: with the luma SAD at a quarter sample per pixel, no
	// realtime quantizer step keeps a coefficient, so the transforms are
	// pure overhead (skip is an encoder choice the decoder honors either
	// way, so this cannot affect parity).
	skip := fullSAD*4 <= bw*bh
	splitTX := false
	if !skip {
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		var lumaZero bool
		if bw == 64 {
			// No 64-point transform in the coder: the luma residual always
			// codes as a one-level split into four 32x32 quadrant TXBs.
			lumaZero = true
			for i := range 4 {
				dy, dx := (i>>1)*32, (i&1)*32
				if !st.prepareInterTXB(src.Y, st.predY[dy*64+dx:], 64, src.YStride, lumaPX+dx, lumaPY+dy, 32, 32, st.yQuant, st.lumaQ2[i*1024:(i+1)*1024]) {
					lumaZero = false
				}
			}
		} else {
			lumaZero = st.prepareInterTXB(src.Y, st.predY[:bw*bh], bw, src.YStride, lumaPX, lumaPY, bw, bh, st.yQuant, st.lumaQ[:bw*bh])
		}
		lumaRdD, lumaRdR := st.rdDcode, st.rdRcode
		uZero := st.prepareInterTXB(src.U, st.predU[:cbw*cbh], cbw, src.ChromaStride, lumaPX/2, lumaPY/2, cbw, cbh, st.uQuant, st.uQ[:cbw*cbh])
		vZero := st.prepareInterTXB(src.V, st.predV[:cbw*cbh], cbw, src.ChromaStride, lumaPX/2, lumaPY/2, cbw, cbh, st.vQuant, st.vQ[:cbw*cbh])
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
		if !skip && bw == 64 {
			splitTX = true
		}
		if !skip && bw == 8 && bh == 8 && !lumaZero {
			// One-level var-tx split RD: quantize the same luma residual as
			// four quadrant 4x4 TXBs and split when their coded rate-
			// distortion beats the whole 8x8 transform.
			cN := n / 2
			st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
			for i := range 4 {
				dy, dx := (i>>1)*cN, (i&1)*cN
				st.prepareInterTXB(src.Y, st.predY[dy*n+dx:], n, src.YStride, lumaPX+dx, lumaPY+dy, cN, cN, st.yQuant, st.lumaQ2[i*cN*cN:(i+1)*cN*cN])
			}
			costFull := ((lumaRdR*st.rdMult + 256) >> 9) + (lumaRdD << 7)
			costSplit := ((st.rdRcode*st.rdMult + 256) >> 9) + (st.rdDcode << 7)
			if costSplit < costFull {
				splitTX = true
			}
		}
		if !skip && bw == 16 && bh == 16 && !lumaZero && st.qIndex <= 160 &&
			(st.hme == nil || st.hme.staticFraction() <= 192) {
			// Static scenes skip the sixteen split: their adapted symbol
			// contexts sit furthest from the throwaway trial state, and
			// split mispricing there costs more than the occasional win.
			// One-level var-tx split RD: the prediction is final here, so
			// the coarse model prices both shapes first and the real
			// coefficient coder arbitrates only the ambiguous band -
			// energy isolated in one quadrant smears across the large DCT
			// but codes as a single small tree.
			cN := n / 2
			coarseFull := ((lumaRdR*st.rdMult + 256) >> 9) + lumaRdD<<7
			st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
			for i := range 4 {
				dy, dx := (i>>1)*cN, (i&1)*cN
				st.prepareInterTXB(src.Y, st.predY[dy*n+dx:], n, src.YStride, lumaPX+dx, lumaPY+dy, cN, cN, st.yQuant, st.lumaQ2[i*cN*cN:(i+1)*cN*cN])
			}
			coarseSplit := ((st.rdRcode*st.rdMult + 256) >> 9) + st.rdDcode<<7
			splitD := st.rdDcode
			diff := coarseSplit - coarseFull
			if diff < 0 {
				diff = -diff
			}
			switch {
			case coarseSplit >= coarseFull && diff*4 > coarseFull:
				// Clear keep: the whole transform wins outright. Splits
				// never pass on the coarse model alone - it overprices
				// dense whole-block coefficients, and a wrong split costs
				// more than a missed one.
			case st.armTrial():
				costFull := st.trialTXBBits(tile.CoeffPlaneY, st.lumaQ[:n*n], n) + lumaRdD<<7
				costSplit := splitD << 7
				for i := range 4 {
					costSplit += st.trialTXBBits(tile.CoeffPlaneY, st.lumaQ2[i*cN*cN:(i+1)*cN*cN], cN)
				}
				if costSplit < costFull {
					splitTX = true
				}
			}
		}
	}

	txType := transform.TypeDCTDCT
	if !skip && !splitTX && bw == 8 && bh == 8 {
		txType = st.chooseInter8x8TXType(src, lumaPX, lumaPY)
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
		ReferenceMode: referenceMode,
		X4:            block.X4, Y4: block.Y4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
	}, refs); err != nil {
		return fmt.Errorf("references: %w", err)
	}

	if err := tile.WriteBlockInterMode(st.w, interModeCDFs, tile.InterModeRequest{
		Compound:    refs.Compound,
		ModeContext: stack.ModeContext,
	}, modeResult); err != nil {
		return fmt.Errorf("inter mode: %w", err)
	}
	drlReq, err := stack.Stack.DRLRequestForMode(modeResult)
	if err != nil {
		return fmt.Errorf("drl request: %w", err)
	}
	if err := tile.WriteDRLIndex(st.w, interModeCDFs, drlReq, 0); err != nil {
		return fmt.Errorf("drl: %w", err)
	}
	var mvRefs tile.InterMVReferenceSet
	if !interModeResultUsesGlobalOnly(modeResult) {
		mvRefs, err = stack.Stack.ResolveInterMVReferences(modeResult, 0, false, false)
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
	if err := tile.WriteInterMotion(st.w, &st.mvCDFs, tile.InterMotionRequest{
		References:   refs,
		Mode:         modeResult,
		ReferenceMVs: mvRefs,
		Precision:    tile.MVSubpelLow,
	}, motionResult); err != nil {
		return fmt.Errorf("motion vector: %w", err)
	}

	hasChroma := true // all coded inter sizes (8x8..32x32) at 4:2:0 carry chroma
	if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), motionResult, hasChroma); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), refs, motion.InterpFilters{}); err != nil {
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

	// Residual phase starts with the var-tx tree, exactly where the decoder
	// reads it: skip blocks mark contexts without symbols, coded blocks write
	// one split decision per node.
	var treeRes tile.TransformTreeResult
	if splitTX {
		treeRes.Split[0] = 1
	}
	lfTree, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: parser.TransformModeSwitchable,
		Inter: true, SkipTransform: skip,
	}, treeRes)
	if err != nil {
		return fmt.Errorf("transform tree: %w", err)
	}
	if st.lfMap != nil {
		// libaom's mode_lf_lut: every inter mode is the motion delta class
		// except the pure global-motion modes.
		lfMode := loopfilter.ModeDeltaClassZero
		if !interModeResultUsesGlobalOnly(modeResult) {
			lfMode = loopfilter.ModeDeltaClassMotion
		}
		if err := markLoopFilterBlock(st.lfMap, block, lfTree, skip, false, uint8(refs.Ref[0])+1, lfMode); err != nil {
			return fmt.Errorf("mark loop filter: %w", err)
		}
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
		copyPredScratch(recon.Y, st.predY[:bw*bh], src.YStride, lumaPX, lumaPY, bw, bh)
		copyPredScratch(recon.U, st.predU[:cbw*cbh], src.ChromaStride, lumaPX/2, lumaPY/2, cbw, cbh)
		copyPredScratch(recon.V, st.predV[:cbw*cbh], src.ChromaStride, lumaPX/2, lumaPY/2, cbw, cbh)
		if st.decisionStats != nil {
			st.decisionStats.noteInterBlock(block.Size, true, false, refs, modeResult, transform.TypeDCTDCT)
		}
		return nil
	}

	// Residual: largest-TX luma with the inter tx_type symbol, then chroma.
	lumaTX, lumaScan := tile.TransformSize8x8, st.scan8
	chromaTX, chromaScan := tile.TransformSize4x4, st.scan4
	switch {
	case bw == 16 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize16x16, st.scan16
		chromaTX, chromaScan = tile.TransformSize8x8, st.scan8
	case bw == 32 && bh == 32:
		lumaTX, lumaScan = tile.TransformSize32x32, st.scan32
		chromaTX, chromaScan = tile.TransformSize16x16, st.scan16
	case bw == 64 && bh == 64:
		chromaTX, chromaScan = tile.TransformSize32x32, st.scan32
	case bw == 16 && bh == 8:
		lumaTX, lumaScan = tile.TransformSize16x8, st.scan16x8
		chromaTX, chromaScan = tile.TransformSize8x4, st.scan8x4
	case bw == 8 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize8x16, st.scan8x16
		chromaTX, chromaScan = tile.TransformSize4x8, st.scan4x8
	case bw == 32 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize32x16, st.scan32x16
		chromaTX, chromaScan = tile.TransformSize16x8, st.scan16x8
	case bw == 16 && bh == 32:
		lumaTX, lumaScan = tile.TransformSize16x32, st.scan16x32
		chromaTX, chromaScan = tile.TransformSize8x16, st.scan8x16
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
		case 64:
			childTX, childScan = tile.TransformSize32x32, st.scan32
		}
		st.interTxTypeReq.Size = childTX
		st.interTxType = transform.TypeDCTDCT
		cN := n / 2
		for i := range 4 {
			dy, dx := (i>>1)*cN, (i&1)*cN
			if err := st.finishInterTXB(recon.Y, st.predY[dy*n+dx:], n, src.YStride, lumaPX+dx, lumaPY+dy, cN, cN, st.yQuant, st.lumaQ2[i*cN*cN:(i+1)*cN*cN], tile.CoeffContextRequest{
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
		st.interTxType = txType
		if err := st.finishInterTXBTyped(recon.Y, st.predY[:bw*bh], bw, src.YStride, lumaPX, lumaPY, bw, bh, st.yQuant, st.lumaQ[:bw*bh], tile.CoeffContextRequest{
			Plane:      0,
			PlaneBlock: block.Size,
			Size:       lumaTX,
			X4:         block.X4,
			Y4:         block.Y4,
		}, coeffCtx, lumaScan, st.afterSkipInter, txType); err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
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
		if err := st.finishInterTXBTyped(rdata, pred, cbw, src.ChromaStride, lumaPX/2, lumaPY/2, cbw, cbh, q, qc, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       chromaTX,
			X4:         block.X4 / 2,
			Y4:         block.Y4 / 2,
		}, coeffCtx, chromaScan, nil, chromaTxType); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
	}
	if st.decisionStats != nil {
		st.decisionStats.noteInterBlock(block.Size, false, splitTX, refs, modeResult, txType)
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
	if !st.trialReady {
		if err := st.trialCDFs.InitDefault(st.qIndex); err != nil {
			return err
		}
		if cap(st.trialBuf) == 0 {
			st.trialBuf = make([]byte, 1<<14)
		}
		st.trialReady = true
	}
	mode, angleDelta := func() (tile.IntraMode, int) {
		return st.improveIntraModeDirectional(src, recon, block, mode, pred, lumaPX, lumaPY, 8)
	}()
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
	}, int8(angleDelta)); err != nil {
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

	lfTree, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: parser.TransformModeSwitchable,
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
	if st.decisionStats != nil {
		st.decisionStats.noteIntraBlock(block.Size)
	}
	return nil
}

// predictInto runs the decoder's inter prediction (8-tap convolve, fixed
// EIGHTTAP filters) for one bw x bh block of plane at frame position (px, py)
// with motion vector mv, writing into dst (stride bw). Full-pel vectors reduce
// to copies inside the kernel, so one path serves both cases bit-exactly.
func predictInto(dst []byte, refPlane []byte, stride, width, height, px, py, bw, bh int, mv motion.Vector, ssX, ssY bool) error {
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, ssX, ssY)
	if err != nil {
		return err
	}
	dstPlane := frame.Plane{Pix: dst, Stride: bw, Width: bw, Height: bh}
	ref := frame.Plane{Pix: refPlane, Stride: stride, Width: width, Height: height}
	return motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstPlane, ref, 1, 8, 0, 0, refX, refY, bw, bh, subX, subY, motion.InterpFilters{})
}

func predictCompoundInto(dst []byte, ref0Plane []byte, stride0 int, ref1Plane []byte, stride1 int, width, height, px, py, bw, bh int,
	mv0, mv1 motion.Vector, ssX, ssY bool, buf0, buf1 *motion.CompoundConvBuf, scratch *motion.CompoundConvolveScratch) error {
	ref0X, ref0Y, sub0X, sub0Y, err := motion.ReferenceOriginSubsampled(px, py, mv0, ssX, ssY)
	if err != nil {
		return err
	}
	ref1X, ref1Y, sub1X, sub1Y, err := motion.ReferenceOriginSubsampled(px, py, mv1, ssX, ssY)
	if err != nil {
		return err
	}
	ref0 := frame.Plane{Pix: ref0Plane, Stride: stride0, Width: width, Height: height}
	ref1 := frame.Plane{Pix: ref1Plane, Stride: stride1, Width: width, Height: height}
	if err := motion.PredictInterCompoundRefToConvBufWithScratch(buf0, ref0, 1, 8, ref0X, ref0Y, bw, bh, sub0X, sub0Y, motion.InterpFilters{}, scratch); err != nil {
		return err
	}
	if err := motion.PredictInterCompoundRefToConvBufWithScratch(buf1, ref1, 1, 8, ref1X, ref1Y, bw, bh, sub1X, sub1Y, motion.InterpFilters{}, scratch); err != nil {
		return err
	}
	dstPlane := frame.Plane{Pix: dst, Stride: bw, Width: bw, Height: bh}
	return motion.BlendCompoundAvg(dstPlane, buf0, buf1, 1, 8, 0, 0, bw, bh, 8, 8)
}

// compoundGoldenLikely samples a bounded set of luma 8x8 blocks before the
// tile pass decides frame-level reference syntax. Single LAST/GOLDEN remains
// available in ReferenceModeSingle; Select is only worth its frame-wide inter
// reference cost when LAST and GOLDEN are locally competitive and their average
// prediction beats LAST by the same kind of margin the leaf selector requires.
func compoundGoldenLikely(st *lossyEncodeState, src, ref SourceFrame420, golden *SourceFrame420) bool {
	if st == nil || golden == nil || golden.Y == nil || ref.Y == nil || src.Width < 8 || src.Height < 8 {
		return false
	}
	if ref.Width != src.Width || ref.Height != src.Height || golden.Width != src.Width || golden.Height != src.Height {
		return false
	}

	const (
		blockSize    = 8
		maxSamples   = 8
		compoundBias = 96
	)
	step := 16
	for ((src.Width+step-1)/step)*((src.Height+step-1)/step) > maxSamples {
		step *= 2
	}
	for py := 0; py+blockSize <= src.Height; py += step {
		for px := 0; px+blockSize <= src.Width; px += step {
			seedDX, seedDY, reach := 0, 0, fullPelReach
			if st.hme != nil {
				var trusted bool
				seedDX, seedDY, trusted = st.hme.seedAt(px, py)
				if trusted {
					reach = fullPelReachTrusted
				}
			}
			ldx, ldy, lastSAD := fullPelDiamondSearchSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, px, py, blockSize, seedDX, seedDY, reach)
			if lastSAD <= blockSize*blockSize*4 {
				continue
			}
			gdx, gdy, gsad := fullPelDiamondSearch(src.Y, golden.Y, src.YStride, src.Width, src.Height, px, py, blockSize)
			if gsad+32 < lastSAD || gsad > lastSAD+blockSize*blockSize*4 {
				continue
			}
			compoundSAD := sad8x8CompoundAvg(src.Y, ref.Y, golden.Y, src.YStride, ref.YStride, golden.YStride, px, py, ldx, ldy, gdx, gdy)
			if compoundSAD+compoundBias < lastSAD {
				return true
			}
		}
	}
	return false
}

func sad8x8CompoundAvg(src, ref0, ref1 []byte, srcStride, stride0, stride1, px, py, dx0, dy0, dx1, dy1 int) int {
	total := 0
	for r := range 8 {
		srcOff := (py+r)*srcStride + px
		ref0Off := (py+dy0+r)*stride0 + px + dx0
		ref1Off := (py+dy1+r)*stride1 + px + dx1
		for c := range 8 {
			pred := (int(ref0[ref0Off+c]) + int(ref1[ref1Off+c]) + 1) >> 1
			d := int(src[srcOff+c]) - pred
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func interModeResultUsesGlobalOnly(mode tile.InterModeResult) bool {
	if mode.Compound {
		return mode.CompoundMode == tile.CompoundInterModeGlobalGlobal
	}
	return mode.Mode == tile.InterModeGlobalMV
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
			if err := predictInto(st.sadScratch[:n*n], refPlane, stride, width, height, px, py, n, n, cand, false, false); err != nil {
				return -1
			}
		}
		base := py*stride + px
		s := 0
		for r := 0; r < n; r += 8 {
			for c := 0; c < n; c += 8 {
				s += sad8x8DualImpl(src[base+r*stride+c:], stride, st.sadScratch[r*n+c:], n)
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

func sadRectBlock(src, ref []byte, base, refBase, stride, bw, bh, limit int) int {
	if bw == bh {
		return sadBlock(src, ref, base, refBase, stride, bw, limit)
	}
	total := 0
	for r := range bh {
		row := base + r*stride
		refRow := refBase + r*stride
		for c := range bw {
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
func copyPredScratch(reconPlane, pred []byte, stride, px, py, w, h int) {
	for r := range h {
		copy(reconPlane[(py+r)*stride+px:(py+r)*stride+px+w], pred[r*w:r*w+w])
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
func (st *lossyEncodeState) prepareInterTXB(srcPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16) bool {
	return st.prepareInterTXBTyped(srcPlane, pred, predStride, stride, px, py, w, h, q, qcoeff, transform.TypeDCTDCT)
}

func (st *lossyEncodeState) prepareInterTXBTyped(srcPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16, txType transform.Type) bool {
	residual := &st.resScratch
	residualBlockImpl(residual[:w*h], srcPlane, py*stride+px, stride, pred, predStride, w, h)
	n := w * h
	tran := &st.tranScratch
	if err := forwardTransformBlock(tran[:n], residual[:n], st.dqScratch[:n], w, h, txType); err != nil {
		return false
	}
	ts := txScaleForSize(max(w, h))
	if err := quantize.QuantizeBlockScaledB(qcoeff, h, tran[:n], h, h, w, q, ts); err != nil {
		return false
	}
	// The vector statistics apply the AC step to every lane; the DC
	// coefficient's coded distortion is then corrected with the DC step.
	dskip, dcode, rate, allZero := rdStatsBlockImpl(tran[:n], qcoeff[:n], n, q.AC, ts)
	if v := qcoeff[0]; v != 0 {
		c := int64(tran[0])
		eAC := c - ((int64(v) * int64(q.AC)) >> ts)
		eDC := c - ((int64(v) * int64(q.DC)) >> ts)
		dcode += eDC*eDC - eAC*eAC
	}
	st.rdDskip += dskip
	st.rdDcode += dcode
	st.rdRcode += rate
	return allZero
}

func (st *lossyEncodeState) chooseInter8x8TXType(src SourceFrame420, lumaPX, lumaPY int) transform.Type {
	if !st.armTrial() {
		return transform.TypeDCTDCT
	}
	bestType := transform.TypeDCTDCT
	bestCost := int64(1 << 62)
	saveDcode, saveDskip, saveRcode := st.rdDcode, st.rdDskip, st.rdRcode
	baseTrialCDFs := st.trialCDFs
	tmpY := st.lumaQ2[:64]
	tmpU := st.lumaQ2[64:80]
	tmpV := st.lumaQ2[80:96]
	for _, typ := range [...]transform.Type{
		transform.TypeDCTDCT,
		transform.TypeADSTDCT,
		transform.TypeDCTADST,
		transform.TypeADSTADST,
		transform.TypeIDTX,
	} {
		st.trialCDFs = baseTrialCDFs
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		lumaZero := st.prepareInterTXBTyped(src.Y, st.predY[:64], 8, src.YStride, lumaPX, lumaPY, 8, 8, st.yQuant, tmpY, typ)
		if typ != transform.TypeDCTDCT && lumaZero {
			continue
		}
		st.prepareInterTXBTyped(src.U, st.predU[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.uQuant, tmpU, typ)
		st.prepareInterTXBTyped(src.V, st.predV[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.vQuant, tmpV, typ)
		cost := st.rdDcode << 7
		cost += st.trialTXBBitsInter(tmpY, 8, tile.TransformSize8x8, typ)
		cost += st.trialTXBBits(tile.CoeffPlaneUV, tmpU, 4)
		cost += st.trialTXBBits(tile.CoeffPlaneUV, tmpV, 4)
		if cost < bestCost {
			bestCost = cost
			bestType = typ
			copy(st.lumaQ[:64], tmpY)
			copy(st.uQ[:16], tmpU)
			copy(st.vQ[:16], tmpV)
		}
	}
	st.rdDcode, st.rdDskip, st.rdRcode = saveDcode, saveDskip, saveRcode
	st.trialCDFs = baseTrialCDFs
	return bestType
}

// residualBlockPureGo is the portable residual extraction.
func residualBlockPureGo(dst []int16, srcPlane []byte, srcOff, stride int, pred []byte, predStride, w, h int) {
	for r := range h {
		row := srcOff + r*stride
		for c := range w {
			dst[r*w+c] = int16(srcPlane[row+c]) - int16(pred[r*predStride+c])
		}
	}
}

// rdStatsBlockPureGo is the portable skip-decision statistics accumulator,
// with the AC step applied to every coefficient (callers correct DC).
func rdStatsBlockPureGo(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (dskip, dcode int64, rate int64, allZero bool) {
	allZero = true
	for i := 0; i < count; i++ {
		c := int64(tran[i])
		dskip += c * c
		v := qcoeff[i]
		if v == 0 {
			dcode += c * c
			continue
		}
		allZero = false
		dq := (int64(v) * int64(step)) >> ts
		e := c - dq
		dcode += e * e
		level := v
		if level < 0 {
			level = -level
		}
		// Coarse rate: two bits of map overhead plus the level magnitude
		// class, in 512-units-per-bit.
		rate += int64(2+bits.Len16(uint16(level))) << 9
	}
	return dskip, dcode, rate, allZero
}

// finishInterTXB codes the prepared coefficients and rebuilds the recon block
// from the prediction through the decoder's dequant + inverse transform.
func (st *lossyEncodeState) finishInterTXB(reconPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error) error {
	return st.finishInterTXBTyped(reconPlane, pred, predStride, stride, px, py, w, h, q, qcoeff, ctxReq, coeffCtx, scan, afterSkip, transform.TypeDCTDCT)
}

func (st *lossyEncodeState) finishInterTXBTyped(reconPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, txType transform.Type) error {
	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip); err != nil {
		return err
	}
	n := w * h
	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n], h, qcoeff, h, h, w, q, txScaleForSize(max(w, h)), 8); err != nil {
		return err
	}
	res := &st.invResidual
	if err := transform.InverseBlock(res[:n], w, dq[:n], h, st.invScratch[:n], transform.Size{Width: uint8(w), Height: uint8(h)}, txType); err != nil {
		return err
	}
	for r := range h {
		row := (py+r)*stride + px
		for c := range w {
			v := int(pred[r*predStride+c]) + int(res[r*w+c])
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

// forwardTransformBlock dispatches the forward transform for every coded
// DCT_DCT shape and the small square hybrid transforms enabled in the realtime
// inter tx_type selector.
func forwardTransformBlock(tran []int32, residual []int16, scratch []int32, w, h int, typ transform.Type) error {
	if typ == transform.TypeDCTDCT {
		return forwardDCTBlock(tran, residual, w, h)
	}
	return transform.ForwardBlock(tran, h, residual, w, scratch, transform.Size{Width: uint8(w), Height: uint8(h)}, typ)
}

// forwardDCTBlock dispatches the forward DCT_DCT for every coded transform
// shape (squares and the factor-two rectangles).
// armTrial readies the throwaway trial contexts for this frame's quantizer.
func (st *lossyEncodeState) armTrial() bool {
	if st.trialReady {
		return true
	}
	if err := st.trialCDFs.InitDefault(st.qIndex); err != nil {
		return false
	}
	if cap(st.trialBuf) == 0 {
		st.trialBuf = make([]byte, 1<<14)
	}
	st.trialReady = true
	return true
}

// interHeaderCost prices one extra inter block's prefix symbols (mode,
// reference, vector residual) under the frame multiplier.
func (st *lossyEncodeState) interHeaderCost() int64 {
	return (int64(24) << 9) * st.rdMult >> 9
}

// trialInterCost prices coding one motion-compensated block exactly: the
// true transform-quantize pass for distortion and the real coefficient coder
// for bits. The prediction lands in sadScratch.
func (st *lossyEncodeState) trialInterCost(src SourceFrame420, ref SourceFrame420, px, py, n int, mv motion.Vector) int64 {
	pred := st.sadScratch[:n*n]
	if err := predictInto(pred, ref.Y, src.YStride, src.Width, src.Height, px, py, n, n, mv, false, false); err != nil {
		return 1 << 59
	}
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, pred, n, src.YStride, px, py, n, n, st.yQuant, st.lumaQ2[:n*n])
	cost := st.trialTXBBits(tile.CoeffPlaneY, st.lumaQ2[:n*n], n) + st.rdDcode<<7
	// The chroma transform structure differs between the shapes too: one
	// large block against four small ones per plane.
	cn := n / 2
	halfW, halfH := src.Width/2, src.Height/2
	for plane := 1; plane <= 2; plane++ {
		data, rdata, q := src.U, ref.U, st.uQuant
		qc := st.uQ[:cn*cn]
		if plane == 2 {
			data, rdata, q = src.V, ref.V, st.vQuant
			qc = st.vQ[:cn*cn]
		}
		cpred := st.sadScratch[n*n : n*n+cn*cn]
		if err := predictInto(cpred, rdata, src.ChromaStride, halfW, halfH, px/2, py/2, cn, cn, mv, true, true); err != nil {
			return 1 << 59
		}
		st.rdDcode = 0
		st.prepareInterTXB(data, cpred, cn, src.ChromaStride, px/2, py/2, cn, cn, q, qc)
		cost += st.trialTXBBits(tile.CoeffPlaneUV, qc, cn) + st.rdDcode<<7
	}
	return cost
}

// trialInterMergeWins prices the merged 16x16 against its four 8x8 children
// with real bits; the merge keeps three blocks' header savings.
func (st *lossyEncodeState) trialInterMergeWins(src SourceFrame420, ref SourceFrame420, px, py int) bool {
	if !st.armTrial() {
		return false
	}
	children := int64(0)
	for _, off := range [4][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
		cx, cy := px+off[0], py+off[1]
		idx8 := (cy/8)*st.grid8Cols + cx/8
		children += st.trialInterCost(src, ref, cx, cy, 8, st.mv8Grid[idx8]) + st.interHeaderCost()
	}
	idx16 := (py/16)*st.grid16Cols + px/16
	merged := st.trialInterCost(src, ref, px, py, 16, st.mv16Grid[idx16]) + st.interHeaderCost()
	// The trial models full-pel prediction without chroma, so it only
	// overrides the calibrated SAD rule on a decisive margin.
	return merged*16 <= children*15
}

func forwardDCTBlock(tran []int32, residual []int16, w, h int) error {
	switch {
	case w == 4 && h == 4:
		return transform.ForwardDCT4x4(tran, 4, residual, 4)
	case w == 8 && h == 8:
		return transform.ForwardDCT8x8(tran, 8, residual, 8)
	case w == 16 && h == 16:
		return transform.ForwardDCT16x16(tran, 16, residual, 16)
	case w == 32 && h == 32:
		return transform.ForwardDCT32x32(tran, 32, residual, 32)
	case w == 16 && h == 8:
		return transform.ForwardDCT16x8(tran, 8, residual, 16)
	case w == 8 && h == 16:
		return transform.ForwardDCT8x16(tran, 16, residual, 8)
	case w == 8 && h == 4:
		return transform.ForwardDCT8x4(tran, 4, residual, 8)
	case w == 4 && h == 8:
		return transform.ForwardDCT4x8(tran, 8, residual, 4)
	case w == 32 && h == 16:
		return transform.ForwardDCT32x16(tran, 16, residual, 32)
	case w == 16 && h == 32:
		return transform.ForwardDCT16x32(tran, 32, residual, 16)
	}
	return transform.ErrInvalidTransform
}

// fullPelDiamondSearch finds the even full-pel offset (dx, dy) within an 8px
// window that minimizes the luma SAD of the n x n block at (px, py) against
// the reference plane, keeping the offset window fully inside the frame. Even
// offsets keep chroma prediction at integer positions at 4:2:0. A small
// diamond refinement around the best raster candidate keeps the search cheap.
func fullPelDiamondSearch(src, ref []byte, stride, width, height, px, py, n int) (int, int, int) {
	return fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, n, 0, 0, fullPelReach)
}

// fullPelReach is the raster half-window around the seed; trusted seeds
// (clean coarse matches, see hmeTrustRegionSAD) narrow it to the seed's own
// four-pel quantization step.
const (
	fullPelReach        = 8
	fullPelReachTrusted = 4
)

// fullPelDiamondSearchSeeded recenters the raster window on a coarse-search
// seed (the hierarchical pre-pass vector), extending reach to the seed
// +-reach px while always probing zero motion first.
func fullPelDiamondSearchSeeded(src, ref []byte, stride, width, height, px, py, n int, seedDX, seedDY, reach int) (int, int, int) {
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
	seedDX = clampHi(clampLo(seedDX, -px), width-n-px) &^ 1
	seedDY = clampHi(clampLo(seedDY, -py), height-n-py) &^ 1
	minDX := clampLo(seedDX-reach, -px)
	maxDX := clampHi(seedDX+reach, width-n-px)
	minDY := clampLo(seedDY-reach, -py)
	maxDY := clampHi(seedDY+reach, height-n-py)

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
