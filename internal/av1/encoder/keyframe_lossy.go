package encoder

import (
	"sync"

	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// keyframe_lossy.go assembles a decodable NON-lossless all-intra keyframe: the
// first compressed encode path. Unlike the lossless encoder, prediction
// neighbors come from the encoder's own reconstruction loop (predict from
// recon, transform+quantize the residual, dequantize+inverse-transform exactly
// as the decoder does, and store recon), so decoder output equals the encoder
// reconstruction bit-for-bit while the source is only approximated.
//
// Scope: 8-bit 4:2:0, dimensions multiples of 8, all blocks 8x8 under
// TX_MODE_LARGEST (luma TX_8X8 + chroma TX_4X4, no tx_size symbols), DC intra,
// DCT_DCT with the tx_type symbol coded per luma TXB, single tile, no
// post-filters (decode output == recon).

// EncodeKeyframe encodes src at the given base qindex (1..255) and returns the
// temporal unit together with the encoder-side reconstruction the decoder must
// reproduce exactly.
func EncodeKeyframe(src SourceFrame420, qIndex uint8) ([]byte, SourceFrame420, error) {
	return encodeKeyframeFiltered(src, qIndex, nil, 0, 0)
}

// encodeKeyframeFiltered encodes the keyframe and, when in-loop filtering is
// active for this size, runs the deblocking pass over the reconstruction
// through lf (allocating a frame-local applier when the caller has none).
func encodeKeyframeFiltered(src SourceFrame420, qIndex uint8, lf *loopFilterApplier, renderW, renderH int) ([]byte, SourceFrame420, error) {
	if src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: frame dimensions must be positive multiples of 8, got %dx%d", src.Width, src.Height)
	}
	if qIndex == 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: qindex 0 is the lossless path; use EncodeLosslessKeyframe")
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
	lfLevel := uint8(0)
	if src.Width*src.Height <= loopFilterMaxArea {
		lfLevel = filterLevelFromQIndex(qIndex, true)
	}
	var lfMap *threading.FrameWorkLoopFilterMap
	if lfLevel > 0 {
		if lf == nil {
			lf = &loopFilterApplier{}
		}
		if !lf.bound {
			if err := lf.init(src.Width, src.Height); err != nil {
				return nil, SourceFrame420{}, fmt.Errorf("loop filter init: %w", err)
			}
		}
		if err := lf.reset(); err != nil {
			return nil, SourceFrame420{}, fmt.Errorf("loop filter reset: %w", err)
		}
		lfMap = &lf.filtMap
	}
	seq := losslessKeyframeSequence(src.Width, src.Height)
	header := lossyKeyframeHeader(src.Width, src.Height, qIndex)
	if renderW > 0 && (renderW != src.Width || renderH != src.Height) {
		header.Size.RenderWidth = uint32(renderW)
		header.Size.RenderHeight = uint32(renderH)
		header.Size.HaveRenderSize = true
	}
	if lfLevel > 0 {
		header.LoopFilter.LevelY = [2]uint8{lfLevel, lfLevel}
		header.LoopFilter.LevelU = lfLevel
		header.LoopFilter.LevelV = lfLevel
	}
	if log2 := defaultTileColsLog2(src.Width); log2 > 0 {
		tiles, err := interTileInfo(src.Width, src.Height, log2)
		if err != nil {
			return nil, SourceFrame420{}, fmt.Errorf("tile info: %w", err)
		}
		tiles.InterpolationFilter = 0 // intra headers carry no filter field
		header.Tile = tiles
	}
	nTiles := int(header.Tile.Cols)
	miCols := uint16(src.Width / 4)
	bounds := func(t int) (uint16, uint16) {
		c0 := header.Tile.ColStartSB[t] * 16
		c1 := header.Tile.ColStartSB[t+1] * 16
		if c1 > miCols {
			c1 = miCols
		}
		return c0, c1
	}
	payloads := make([]TilePayload, nTiles)
	var wg sync.WaitGroup
	errs := make([]error, nTiles)
	for t := 1; t < nTiles; t++ {
		c0, c1 := bounds(t)
		wg.Add(1)
		go func(t int, c0, c1 uint16) {
			defer wg.Done()
			data, err := encodeKeyframeTile(src, &recon, qIndex, c0, c1, lfMap)
			if err != nil {
				errs[t] = err
				return
			}
			payloads[t].Data = data
		}(t, c0, c1)
	}
	c0, c1 := bounds(0)
	tile0, tile0Err := encodeKeyframeTile(src, &recon, qIndex, c0, c1, lfMap)
	wg.Wait()
	if tile0Err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("encode tile 0: %w", tile0Err)
	}
	payloads[0].Data = tile0
	for t := 1; t < nTiles; t++ {
		if errs[t] != nil {
			return nil, SourceFrame420{}, fmt.Errorf("encode tile %d: %w", t, errs[t])
		}
	}

	if lfLevel > 0 {
		if err := lf.apply(&recon, parser.LoopFilterParams{
			LevelY: [2]uint8{lfLevel, lfLevel},
			LevelU: lfLevel,
			LevelV: lfLevel,
		}); err != nil {
			return nil, SourceFrame420{}, fmt.Errorf("loop filter apply: %w", err)
		}
	}
	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("size header TU: %w", err)
	}
	endTile := uint16(nTiles - 1)
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, endTile, payloads)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, endTile, payloads)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, SourceFrame420{}, err
	}
	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, recon, nil
}

func lossyKeyframeHeader(width, height int, qIndex uint8) IntraFrameHeaderParams {
	header := losslessKeyframeHeader(width, height)
	header.Quantization = QuantizationParams{BaseQIdx: qIndex}
	header.AllLossless = false
	header.LoopFilter = LoopFilterParams{
		Deltas: defaultLoopFilterDeltas(),
	}
	if lvl := filterLevelFromQIndex(qIndex, true); lvl > 0 {
		header.LoopFilter.LevelY = [2]uint8{lvl, lvl}
		header.LoopFilter.LevelU = lvl
		header.LoopFilter.LevelV = lvl
	}
	header.TransformRef = TransformReferenceParams{
		TransformMode: TransformModeLargest,
		ReferenceMode: ReferenceModeSingle,
	}
	return header
}

// lossyEncodeState carries the per-tile coding state of the non-lossless
// keyframe encoder.
type lossyEncodeState struct {
	w         *entropy.Writer
	modeCDFs  tile.BlockModeCDFs
	intraCDFs tile.IntraModeCDFs
	coeffCDFs tile.CoeffCDFs
	txCDFs    tile.TransformTypeCDFs
	treeCDFs  tile.TransformCDFs
	mvCDFs    tile.MVCDFs

	qIndex uint8
	yQuant quantize.Quantizer
	uQuant quantize.Quantizer
	vQuant quantize.Quantizer

	scan8, scan4, scan16, scan32 []int16
	scan16x8, scan8x16           []int16
	scan8x4, scan4x8             []int16
	levels                       []uint8
	invScratch                   []int32
	color                        parser.ColorConfig

	// Per-block scratch reused across blocks so the hot encode loop stays
	// allocation-free: quantized coefficients per plane (sized for the largest
	// coded TXB, 16x16 luma / 8x8 chroma) and the prebuilt after-skip tx_type
	// hook (a closure built once per tile, not per block).
	lumaQ          [1024]int16
	lumaQ2         [1024]int16
	uQ, vQ         [256]int16
	interTxTypeReq tile.InterTransformTypeRequest
	afterSkipInter func() error
	intraTxTypeReq tile.IntraTransformTypeRequest
	afterSkipIntra func() error

	// Motion-compensated prediction scratch, filled per block through the
	// decoder's own convolve so subpel predictions match bit for bit.
	predY      [1024]byte
	predU      [256]byte
	predV      [256]byte
	sadScratch [1024]byte

	// Transform/quant scratch for the inter TXB pipeline (residual in,
	// forward transform out, dequant + inverse residual back), state-owned so
	// the per-block helpers stay allocation-free at 16x16 sizes.
	resScratch  [1024]int16
	tranScratch [1024]int32
	dqScratch   [1024]int32
	invResidual [1024]int16

	// rdMult prices coefficient rate against transform-domain distortion for
	// the skip decision (av1_compute_rd_mult's inter shape at the working
	// quantizer); rdTXB accumulates the current block's code-vs-skip terms.
	rdMult    int64
	rdDcode   int64
	rdDskip   int64
	rdRcode   int64
	sadPerBit int

	// prober hoists the subpel refinement's geometry validation per block;
	// its convolve scratch is reused across blocks.
	prober motion.LumaSubpelProber

	// lfMap, when non-nil, collects per-MI loop-filter records for the
	// frame-level deblocking pass (the decoder's own map type, filled with
	// the same per-block syntax the decoder records).
	lfMap *threading.FrameWorkLoopFilterMap

	// hme, when non-nil, supplies the coarse-search seeds that recenter the
	// full-pel refinement windows.
	hme *hmeState

	// Per-frame motion partition grids filled by the 16x16 partition decider:
	// the merged 16x16 full-pel result, and the child 8x8 full-pel results so
	// split leaves do not repeat the search. sad < 0 marks an empty slot.
	mv16Grid   []motion.Vector
	sad16Grid  []int32
	grid16Cols int
	mv8Grid    []motion.Vector
	sad8Grid   []int32
	grid8Cols  int
	mv32Grid   []motion.Vector
	sad32Grid  []int32
	grid32Cols int
}

func encodeKeyframeTile(src SourceFrame420, recon *SourceFrame420, qIndex uint8, miColStart, miColEnd uint16, lfMap *threading.FrameWorkLoopFilterMap) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	if err := partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &lossyEncodeState{qIndex: qIndex, color: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}, lfMap: lfMap}
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
	st.scan16 = make([]int16, 256)
	inverse16 := make([]int16, 256)
	if err := transform.FillDefaultScan(st.scan16, inverse16, transform.Size{Width: 16, Height: 16}, transform.Class2D); err != nil {
		return nil, err
	}
	st.scan32 = make([]int16, 1024)
	inverse32 := make([]int16, 1024)
	if err := transform.FillDefaultScan(st.scan32, inverse32, transform.Size{Width: 32, Height: 32}, transform.Class2D); err != nil {
		return nil, err
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize32x32)
	if err != nil {
		return nil, err
	}
	st.levels = make([]uint8, scratchLen)
	st.invScratch = make([]int32, 1024)

	w := entropy.NewWriter(make([]byte, 0, 1<<18))
	st.w = &w

	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB

	var scratch tile.BlockLoopScratch
	carrier := &tile.BlockLoopContextCarrier{
		Above: make([]tile.BlockLoopRootAboveContext, rootCols),
	}
	walkReq := tile.BlockWalkRequest{
		Root:       tile.BlockLevel64x64,
		MIColStart: miColStart,
		MIColEnd:   miColEnd,
		MIRowEnd:   miRows,
	}
	// Blocks are 8x8 by default; a 16x16 node fully inside the frame merges
	// when its best whole-block intra prediction is already tight (under two
	// SAD units per pixel), which keeps merge quality protecting the
	// reconstruction the way the inter tiers do.
	mergeIntra := func(px, py, n int) bool {
		mode := selectIntraModeN(src.Y, recon.Y, src.YStride, px, py, n, py > int(walkReq.MIRowStart)*4, px > int(walkReq.MIColStart)*4, st.predY[:n*n])
		_ = mode
		sad := 0
		for r := range n {
			row := (py+r)*src.YStride + px
			for c := range n {
				d := int(src.Y[row+c]) - int(st.predY[r*n+c])
				if d < 0 {
					d = -d
				}
				sad += d
			}
		}
		return sad <= n*n*2
	}
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		if level == tile.BlockLevel32x32 && haveRight && haveBottom {
			px, py := int(miCol)*4, int(miRow)*4
			if px+32 <= src.Width && py+32 <= src.Height && mergeIntra(px, py, 32) {
				return tile.PartitionNone, nil
			}
			return tile.PartitionSplit, nil
		}
		if level == tile.BlockLevel16x16 && haveRight && haveBottom {
			px, py := int(miCol)*4, int(miRow)*4
			if px+16 <= src.Width && py+16 <= src.Height && mergeIntra(px, py, 16) {
				return tile.PartitionNone, nil
			}
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodeBlock(src, recon, block, scratch)
	}
	if err := tile.WalkBlockLoopWrite(&w, &partCDFs, &scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

// encodeBlock codes one 8x8 DC-intra block: mode symbols in the decoder's
// order, then the luma TX_8X8 residual (with the tx_type symbol after
// txb_skip) and the two chroma TX_4X4 residuals, reconstructing each transform
// block before the next so later predictions see decoder-identical neighbors.
func (st *lossyEncodeState) encodeBlock(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
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

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if st.lfMap != nil {
		yTX, err := tile.MaxTransformSize(block.Size, parser.ColorConfig{}, 0)
		if err != nil {
			return err
		}
		uvTX, err := tile.MaxTransformSize(block.Size, st.color, 1)
		if err != nil {
			return err
		}
		if err := markLoopFilterBlock(st.lfMap, block, tile.TransformTreeResult{Y: yTX, UV: uvTX, HasUV: true},
			false, true, 0, loopfilter.ModeDeltaClassZero); err != nil {
			return fmt.Errorf("mark loop filter: %w", err)
		}
	}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}

	// Luma mode selection by prediction SAD against the reconstructed edges.
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	pred := st.predY[:n*n]
	mode := selectIntraModeN(src.Y, recon.Y, src.YStride, lumaPX, lumaPY, n, block.HaveTop, block.HaveLeft, pred)

	if err := tile.WriteLumaIntraMode(st.w, &st.intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
	}, mode); err != nil {
		return fmt.Errorf("luma mode: %w", err)
	}
	if err := modeCtx.MarkIntra(block.Size, int(block.X4), int(block.Y4), true, mode); err != nil {
		return fmt.Errorf("mark intra: %w", err)
	}
	if err := tile.WriteIntraAngleDelta(st.w, &st.intraCDFs, tile.IntraAngleDeltaRequest{
		Size: block.Size, Mode: mode,
	}, 0); err != nil {
		return fmt.Errorf("angle delta: %w", err)
	}
	cflAllowed, err := tile.ChromaIntraCFLAllowed(block.Size, st.color, false)
	if err != nil {
		return fmt.Errorf("cfl allowed: %w", err)
	}
	if err := tile.WriteChromaIntraMode(st.w, &st.intraCDFs, tile.ChromaIntraModeRequest{
		Size: block.Size, LumaMode: mode, CFLAllowed: cflAllowed,
	}, tile.ChromaIntraModeDC, tile.CFLAlphaResult{}); err != nil {
		return fmt.Errorf("chroma mode: %w", err)
	}
	if err := modeCtx.MarkChromaIntra(block.Size, int(block.X4), int(block.Y4), true, tile.ChromaIntraModeDC); err != nil {
		return fmt.Errorf("mark chroma intra: %w", err)
	}

	// Largest-TX luma with the tx_type symbol between txb_skip and eob,
	// then the half-size chroma TXBs.
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
	txTypeReq := tile.IntraTransformTypeRequest{
		Size:        lumaTX,
		Mode:        mode,
		QIndexKnown: true,
		QIndex:      st.qIndex,
	}
	afterSkip := func() error {
		return tile.WriteIntraTransformType(st.w, &st.txCDFs, txTypeReq, transform.TypeDCTDCT)
	}
	if err := st.encodeTXBPred(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, n, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       lumaTX,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, lumaScan, afterSkip, pred); err != nil {
		return fmt.Errorf("luma txb: %w", err)
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
		if err := st.encodeTXBAvail(rdata, data, src.ChromaStride, lumaPX/2, lumaPY/2, cn, q, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       chromaTX,
			X4:         block.X4 / 2,
			Y4:         block.Y4 / 2,
		}, coeffCtx, chromaScan, nil, block.HaveTop, block.HaveLeft); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
	}
	return nil
}

// encodeTXB codes one square n x n transform block: DC-predict from the recon
// plane, forward transform the source residual, quantize, write the
// coefficients, then reconstruct through the decoder's own dequant + inverse
// transform so recon matches the decoder bit-for-bit.
func (st *lossyEncodeState) encodeTXB(reconPlane []byte, srcPlane []byte, stride int, px, py, n int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error) error {
	return st.encodeTXBAvail(reconPlane, srcPlane, stride, px, py, n, q, ctxReq, coeffCtx, scan, afterSkip, py > 0, px > 0)
}

// encodeTXBAvail is encodeTXB with explicit neighbor availability: inside a
// tile the left/top edges follow the tile boundary, not the frame's. The
// prediction is flat DC from the available reconstructed edges.
func (st *lossyEncodeState) encodeTXBAvail(reconPlane []byte, srcPlane []byte, stride int, px, py, n int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, haveTop, haveLeft bool) error {

	dc := dcPredictN(reconPlane, stride, px, py, n, haveTop, haveLeft)
	pred := st.predU[:n*n] // free during intra TXB coding at n <= 8
	for i := range pred {
		pred[i] = dc
	}
	return st.encodeTXBPred(reconPlane, srcPlane, stride, px, py, n, q, ctxReq, coeffCtx, scan, afterSkip, pred)
}

// encodeTXBPred codes one square n x n transform block against an arbitrary
// already-materialized prediction (stride n): forward transform the source
// residual, quantize, write the coefficients, then reconstruct through the
// decoder's own dequant + inverse transform.
func (st *lossyEncodeState) encodeTXBPred(reconPlane []byte, srcPlane []byte, stride int, px, py, n int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, pred []byte) error {

	residual := &st.resScratch
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			residual[r*n+c] = int16(srcPlane[row+c]) - int16(pred[r*n+c])
		}
	}
	tran := &st.tranScratch
	qcoeff := &st.lumaQ
	switch n {
	case 4:
		if err := transform.ForwardDCT4x4(tran[:16], 4, residual[:16], 4); err != nil {
			return err
		}
	case 8:
		if err := transform.ForwardDCT8x8(tran[:64], 8, residual[:64], 8); err != nil {
			return err
		}
	case 16:
		if err := transform.ForwardDCT16x16(tran[:256], 16, residual[:256], 16); err != nil {
			return err
		}
	case 32:
		if err := transform.ForwardDCT32x32(tran[:1024], 32, residual[:1024], 32); err != nil {
			return err
		}
	default:
		return fmt.Errorf("encoder: unsupported txb size %d", n)
	}
	if err := quantize.QuantizeBlockScaledB(qcoeff[:n*n], n, tran[:n*n], n, n, n, q, txScaleForSize(n)); err != nil {
		return err
	}
	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff[:n*n], scan, st.levels, afterSkip); err != nil {
		return err
	}

	// Reconstruct exactly as the decoder will: dequantize, inverse transform,
	// add to the prediction, clip to 8 bits.
	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n*n], n, qcoeff[:n*n], n, n, n, q, txScaleForSize(n), 8); err != nil {
		return err
	}
	res := &st.invResidual
	if err := transform.InverseDCTBlock(res[:n*n], n, dq[:n*n], n, st.invScratch[:n*n], transform.Size{Width: uint8(n), Height: uint8(n)}); err != nil {
		return err
	}
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			v := int(pred[r*n+c]) + int(res[r*n+c])
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

// dcPredictN is the decoder's DC predictor for one n x n block at pixel
// (px,py) of plane: the rounded mean of the n above and n left reconstructed
// neighbors that are available, or 128 when neither edge is. Availability is
// the caller's tile-relative HaveTop/HaveLeft, not raw frame position.
func dcPredictN(plane []byte, stride, px, py, n int, haveTop, haveLeft bool) uint8 {
	sum := 0
	count := 0
	if haveTop && py > 0 {
		row := (py-1)*stride + px
		for i := range n {
			sum += int(plane[row+i])
		}
		count += n
	}
	if haveLeft && px > 0 {
		col := py*stride + px - 1
		for i := range n {
			sum += int(plane[col+i*stride])
		}
		count += n
	}
	if count == 0 {
		return 128
	}
	return uint8((sum + count/2) / count)
}

// selectIntraModeN chooses the luma intra mode of one n x n block by
// prediction SAD against the reconstructed edges and fills pred (stride n)
// with the chosen prediction. DC competes always; the exact-angle vertical
// and horizontal copies need their edge (angles 90 and 180 predict without
// edge filtering, so the copies match the decoder bit for bit). DC keeps
// ties: it codes no angle-delta symbol. SMOOTH was built and measured a
// wash at 8x8 - DC plus a two-coefficient DCT residual already covers
// gradients - and removed.
func selectIntraModeN(srcPlane, reconPlane []byte, stride, px, py, n int, haveTop, haveLeft bool, pred []byte) tile.IntraMode {
	dc := dcPredictN(reconPlane, stride, px, py, n, haveTop, haveLeft)
	sadDC, sadV, sadH := 0, 1<<30, 1<<30
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			d := int(srcPlane[row+c]) - int(dc)
			if d < 0 {
				d = -d
			}
			sadDC += d
		}
	}
	above := (py-1)*stride + px
	if haveTop {
		sadV = 0
		for r := range n {
			row := (py+r)*stride + px
			for c := range n {
				d := int(srcPlane[row+c]) - int(reconPlane[above+c])
				if d < 0 {
					d = -d
				}
				sadV += d
			}
		}
	}
	if haveLeft {
		sadH = 0
		for r := range n {
			row := (py+r)*stride + px
			left := int(reconPlane[row-1])
			for c := range n {
				d := int(srcPlane[row+c]) - left
				if d < 0 {
					d = -d
				}
				sadH += d
			}
		}
	}
	switch {
	case sadV+16 < sadDC && sadV <= sadH:
		for r := range n {
			copy(pred[r*n:r*n+n], reconPlane[above:above+n])
		}
		return tile.IntraModeVertical
	case sadH+16 < sadDC:
		for r := range n {
			v := reconPlane[(py+r)*stride+px-1]
			for c := range n {
				pred[r*n+c] = v
			}
		}
		return tile.IntraModeHorizontal
	default:
		for i := range n * n {
			pred[i] = dc
		}
		return tile.IntraModeDC
	}
}

// selectIntraMode8 is selectIntraModeN at the 8x8 fallback size.
func selectIntraMode8(srcPlane, reconPlane []byte, stride, px, py int, haveTop, haveLeft bool, pred []byte) tile.IntraMode {
	return selectIntraModeN(srcPlane, reconPlane, stride, px, py, 8, haveTop, haveLeft, pred)
}
