package encoder

import (
	"fmt"
	"sync"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/prediction"
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
	return encodeKeyframeFiltered(src, qIndex, nil, 0, 0, nil, nil, nil)
}

// EncodeMonochromeKeyframe encodes src at the given base qindex (1..255) as a
// native AV1 monochrome non-lossless keyframe and returns the encoder-side
// monochrome reconstruction the decoder must reproduce exactly.
func EncodeMonochromeKeyframe(src SourceFrameMono, qIndex uint8) ([]byte, SourceFrameMono, error) {
	if err := validateSourceFrameMono(src); err != nil {
		return nil, SourceFrameMono{}, err
	}
	if qIndex == 0 {
		return nil, SourceFrameMono{}, fmt.Errorf("encoder: qindex 0 is the lossless path; use EncodeLosslessMonochromeKeyframe")
	}
	recon := SourceFrameMono{
		Y:       make([]byte, len(src.Y)),
		YStride: src.YStride,
		Width:   src.Width,
		Height:  src.Height,
	}

	var pc pframeCoder
	tilePayload, err := pc.encodeMonochromeKeyframeTile(src, &recon, qIndex, 0, uint16(src.Width/4))
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("encode tile: %w", err)
	}

	seq := lossyMonochromeKeyframeSequence(src.Width, src.Height)
	header := lossyMonochromeKeyframeHeaderForSequence(seq, src.Width, src.Height, qIndex)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("size header TU: %w", err)
	}

	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, SourceFrameMono{}, err
	}

	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, recon, nil
}

// EncodeKeyframeWithSequenceMax encodes src as a shown keyframe while keeping
// the sequence header max dimensions at maxWidth x maxHeight. This is useful
// for scalable streams whose lower spatial layers are smaller than the shared
// sequence maximum.
func EncodeKeyframeWithSequenceMax(src SourceFrame420, qIndex uint8, maxWidth, maxHeight int) ([]byte, SourceFrame420, error) {
	return encodeKeyframeFilteredTiles(src, qIndex, nil, 0, 0, nil, nil, nil, defaultTileColsLog2(src.Width), keyframeTileOptions{
		sequenceMaxWidth:  maxWidth,
		sequenceMaxHeight: maxHeight,
	})
}

// encodeKeyframeFiltered encodes the keyframe and, when in-loop filtering is
// active for this size, runs the deblocking pass over the reconstruction
// through lf (allocating a frame-local applier when the caller has none).
// Streaming callers pass their reusable reconstruction buffer and tile-coder
// pool so periodic keyframes allocate nothing; one-shot callers pass nil for
// both.
func encodeKeyframeFiltered(src SourceFrame420, qIndex uint8, lf *loopFilterApplier, renderW, renderH int, reconBuf *SourceFrame420, tilePC func(t int) *pframeCoder, cdefApp *cdefApplier) ([]byte, SourceFrame420, error) {
	return encodeKeyframeFilteredTiles(src, qIndex, lf, renderW, renderH, reconBuf, tilePC, cdefApp, defaultTileColsLog2(src.Width), keyframeTileOptions{})
}

// encodeKeyframeFilteredTiles is encodeKeyframeFiltered with the stream's
// tile-column split, which must match the coder pool the caller sized.
func encodeKeyframeFilteredTiles(src SourceFrame420, qIndex uint8, lf *loopFilterApplier, renderW, renderH int, reconBuf *SourceFrame420, tilePC func(t int) *pframeCoder, cdefApp *cdefApplier, tileColsLog2 uint8, tileOpts keyframeTileOptions) ([]byte, SourceFrame420, error) {
	if src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: frame dimensions must be positive multiples of 8, got %dx%d", src.Width, src.Height)
	}
	if qIndex == 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: qindex 0 is the lossless path; use EncodeLosslessKeyframe")
	}
	seqWidth, seqHeight := src.Width, src.Height
	if tileOpts.sequenceMaxWidth != 0 || tileOpts.sequenceMaxHeight != 0 {
		if tileOpts.sequenceMaxWidth <= 0 || tileOpts.sequenceMaxHeight <= 0 ||
			tileOpts.sequenceMaxWidth < src.Width || tileOpts.sequenceMaxHeight < src.Height {
			return nil, SourceFrame420{}, fmt.Errorf("encoder: invalid sequence max %dx%d for coded frame %dx%d", tileOpts.sequenceMaxWidth, tileOpts.sequenceMaxHeight, src.Width, src.Height)
		}
		seqWidth, seqHeight = tileOpts.sequenceMaxWidth, tileOpts.sequenceMaxHeight
	}
	var recon SourceFrame420
	if reconBuf != nil && reconBuf.Y != nil {
		recon = *reconBuf
	} else {
		recon = SourceFrame420{
			Y:            make([]byte, len(src.Y)),
			U:            make([]byte, len(src.U)),
			V:            make([]byte, len(src.V)),
			YStride:      src.YStride,
			ChromaStride: src.ChromaStride,
			Width:        src.Width,
			Height:       src.Height,
		}
		if reconBuf != nil {
			*reconBuf = recon
		}
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
	seq := losslessKeyframeSequence(seqWidth, seqHeight)
	if tileOpts.stream != nil {
		seq = tileOpts.stream.sequenceHeader(seqWidth, seqHeight)
	}
	header := lossyKeyframeHeaderForSequence(seq, src.Width, src.Height, qIndex)
	if tileOpts.stream != nil {
		tileOpts.stream.applyContentHintToFrameHeader(&header.Prefix)
	}
	if renderW > 0 && (renderW != src.Width || renderH != src.Height) {
		header.Size.RenderWidth = uint32(renderW)
		header.Size.RenderHeight = uint32(renderH)
		header.Size.HaveRenderSize = true
	}
	if lfLevel > 0 {
		header.LoopFilter.LevelY = [2]uint8{lfLevel, lfLevel}
		header.LoopFilter.LevelU = lfLevel
		header.LoopFilter.LevelV = lfLevel
		header.CDEF = cdefHeaderParams(qIndex, true)
		if cdefApp == nil {
			cdefApp = &cdefApplier{}
		}
		if !cdefApp.bound {
			if err := cdefApp.init(src.Width, src.Height, cdefParserParams(header.CDEF)); err != nil {
				return nil, SourceFrame420{}, fmt.Errorf("cdef init: %w", err)
			}
		}
	}
	if log2 := tileColsLog2; log2 > 0 {
		tiles, err := interTileInfoForSequence(seq, src.Width, src.Height, log2)
		if err != nil {
			return nil, SourceFrame420{}, fmt.Errorf("tile info: %w", err)
		}
		tiles.InterpolationFilter = 0 // intra headers carry no filter field
		header.Tile = tiles
	}
	nTiles := int(header.Tile.Cols)
	miCols := uint16(src.Width / 4)
	payloads := tileOpts.payloads
	if cap(payloads) < nTiles {
		payloads = make([]TilePayload, nTiles)
	} else {
		payloads = payloads[:nTiles]
		for i := range payloads {
			payloads[i] = TilePayload{}
		}
	}
	errs := tileOpts.errs
	if cap(errs) < nTiles {
		errs = make([]error, nTiles)
	} else {
		errs = errs[:nTiles]
		for i := range errs {
			errs[i] = nil
		}
	}
	req := keyframeTileRun{
		src:                     src,
		recon:                   &recon,
		qIndex:                  qIndex,
		allowScreenContentTools: header.Prefix.AllowScreenContentTools,
		tile:                    header.Tile,
		miCols:                  miCols,
		lfMap:                   lfMap,
		payloads:                payloads,
		errs:                    errs,
	}
	var err error
	if tileOpts.stream != nil {
		err = tileOpts.stream.runKeyframeTileWorkers(req)
	} else {
		err = runKeyframeTilesDefault(req, tilePC)
	}
	if err != nil {
		return nil, SourceFrame420{}, err
	}

	if lfLevel > 0 {
		if err := lf.apply(&recon, parser.LoopFilterParams{
			LevelY: [2]uint8{lfLevel, lfLevel},
			LevelU: lfLevel,
			LevelV: lfLevel,
		}); err != nil {
			return nil, SourceFrame420{}, fmt.Errorf("loop filter apply: %w", err)
		}
		if err := cdefApp.apply(&recon, cdefParserParams(header.CDEF), &lf.filtMap); err != nil {
			return nil, SourceFrame420{}, fmt.Errorf("cdef apply: %w", err)
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

type keyframeTileOptions struct {
	payloads          []TilePayload
	errs              []error
	stream            *VideoEncoder
	sequenceMaxWidth  int
	sequenceMaxHeight int
}

type keyframeTileRun struct {
	src                     SourceFrame420
	recon                   *SourceFrame420
	qIndex                  uint8
	allowScreenContentTools bool
	tile                    TileInfo
	miCols                  uint16
	lfMap                   *threading.FrameWorkLoopFilterMap
	payloads                []TilePayload
	errs                    []error
}

func runKeyframeTilesDefault(req keyframeTileRun, tilePC func(t int) *pframeCoder) error {
	nTiles := len(req.payloads)
	var wg sync.WaitGroup
	for t := 1; t < nTiles; t++ {
		c0, c1 := tileColBounds(req.tile, t, req.miCols)
		pc := &pframeCoder{}
		if tilePC != nil {
			pc = tilePC(t)
		}
		wg.Add(1)
		go func(t int, c0, c1 uint16, pc *pframeCoder) {
			defer wg.Done()
			data, err := pc.encodeKeyframeTileWithOptions(req.src, req.recon, req.qIndex, c0, c1, req.lfMap, req.allowScreenContentTools)
			if err != nil {
				req.errs[t] = err
				return
			}
			req.payloads[t].Data = data
		}(t, c0, c1, pc)
	}
	c0, c1 := tileColBounds(req.tile, 0, req.miCols)
	pc0 := &pframeCoder{}
	if tilePC != nil {
		pc0 = tilePC(0)
	}
	tile0, tile0Err := pc0.encodeKeyframeTileWithOptions(req.src, req.recon, req.qIndex, c0, c1, req.lfMap, req.allowScreenContentTools)
	wg.Wait()
	if tile0Err != nil {
		return fmt.Errorf("encode tile 0: %w", tile0Err)
	}
	req.payloads[0].Data = tile0
	for t := 1; t < nTiles; t++ {
		if req.errs[t] != nil {
			return fmt.Errorf("encode tile %d: %w", t, req.errs[t])
		}
	}
	return nil
}

func lossyKeyframeHeader(width, height int, qIndex uint8) IntraFrameHeaderParams {
	return lossyKeyframeHeaderForSequence(losslessKeyframeSequence(width, height), width, height, qIndex)
}

func lossyKeyframeHeaderForSequence(seq SequenceHeader, width, height int, qIndex uint8) IntraFrameHeaderParams {
	header := losslessKeyframeHeaderForSequence(seq, width, height)
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
	header.CDEF = CDEFParams{Damping: 3}
	return header
}

func lossyMonochromeKeyframeSequence(width, height int) SequenceHeader {
	seq := losslessMonochromeKeyframeSequence(width, height)
	seq.EnableCDEF = false
	return seq
}

func lossyMonochromeKeyframeHeaderForSequence(seq SequenceHeader, width, height int, qIndex uint8) IntraFrameHeaderParams {
	header := losslessKeyframeHeaderForSequence(seq, width, height)
	header.Quantization = QuantizationParams{BaseQIdx: qIndex}
	header.AllLossless = false
	header.LoopFilter = LoopFilterParams{
		Deltas: defaultLoopFilterDeltas(),
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

	qIndex                  uint8
	forceIntegerMV          bool
	allowScreenContentTools bool
	sadCacheEpoch           uint32
	yQuant                  quantize.Quantizer
	uQuant                  quantize.Quantizer
	vQuant                  quantize.Quantizer

	scan8, scan4, scan16, scan32 []int16
	scan16x8, scan8x16           []int16
	scan32x16, scan16x32         []int16
	scan8x4, scan4x8             []int16
	levels                       []uint8
	trialCDFs                    tile.CoeffCDFs
	trialReady                   bool
	trial8x8CDFs                 coeffTrial8x8Snapshot
	intraEdgeScratch             threading.FrameWorkIntraPredictionScratch
	keyMIColEnd, keyMIRowEnd     uint32
	keyVisW, keyVisH             int
	invScratch                   []int32
	color                        parser.ColorConfig

	// Per-block scratch reused across blocks so the hot encode loop stays
	// allocation-free: quantized coefficients per plane (sized for the largest
	// coded TXB, 16x16 luma / 8x8 chroma) and the prebuilt after-skip tx_type
	// hook (a closure built once per tile, not per block).
	lumaQ          [4096]int16
	lumaQ2         [4096]int16
	uQ, vQ         [1024]int16
	interTxTypeReq tile.InterTransformTypeRequest
	interTxType    transform.Type
	afterSkipInter func() error
	intraTxTypeReq tile.IntraTransformTypeRequest
	afterSkipIntra func() error

	// Motion-compensated prediction scratch, filled per block through the
	// decoder's own convolve so subpel predictions match bit for bit.
	predY         [4096]byte
	predY16       [4096]uint16
	predU         [1024]byte
	predV         [1024]byte
	sadScratch    [4096]byte
	compBuf0      motion.CompoundConvBuf
	compBuf1      motion.CompoundConvBuf
	compScratch   motion.CompoundConvolveScratch
	scaledScratch motion.ScaledConvolveScratch

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

	// decisionStats is nil on the production hot path. When diagnostics are
	// explicitly enabled, it points at the owning tile coder's frame-local
	// counter buffer.
	decisionStats *EncoderDecisionStats

	// Per-frame motion partition grids filled by the 16x16 partition decider:
	// the merged 16x16 full-pel result, and the child 8x8 full-pel results so
	// split leaves do not repeat the search. SAD entries are tagged by
	// sadCacheEpoch to avoid bulk-invalidating the grids every tile.
	mv16Grid   []motion.Vector
	sad16Grid  []uint32
	grid16Cols int
	mv8Grid    []motion.Vector
	sad8Grid   []uint32
	grid8Cols  int
	mv32Grid   []motion.Vector
	sad32Grid  []uint32
	grid32Cols int
	mv64Grid   []motion.Vector
	sad64Grid  []uint32
	grid64Cols int
}

func encodeKeyframeTile(src SourceFrame420, recon *SourceFrame420, qIndex uint8, miColStart, miColEnd uint16, lfMap *threading.FrameWorkLoopFilterMap) ([]byte, error) {
	var pc pframeCoder
	return pc.encodeKeyframeTile(src, recon, qIndex, miColStart, miColEnd, lfMap)
}

// encodeKeyframeTile codes one keyframe tile reusing the coder's buffers:
// the symbol contexts reinitialize to the keyframe defaults in place, and
// the scans, scratch planes, context carrier, and writer buffer carry over
// from inter coding, so a streaming keyframe allocates nothing per tile.
func (pc *pframeCoder) encodeKeyframeTile(src SourceFrame420, recon *SourceFrame420, qIndex uint8, miColStart, miColEnd uint16, lfMap *threading.FrameWorkLoopFilterMap) ([]byte, error) {
	return pc.encodeKeyframeTileWithOptions(src, recon, qIndex, miColStart, miColEnd, lfMap, false)
}

func (pc *pframeCoder) encodeKeyframeTileWithOptions(src SourceFrame420, recon *SourceFrame420, qIndex uint8, miColStart, miColEnd uint16, lfMap *threading.FrameWorkLoopFilterMap, allowScreenContentTools bool) ([]byte, error) {
	if err := pc.partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &pc.st
	st.qIndex = qIndex
	st.forceIntegerMV = false
	st.allowScreenContentTools = allowScreenContentTools
	st.color = parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	st.lfMap = lfMap
	st.hme = nil
	st.decisionStats = nil
	if pc.decisionStatsEnabled {
		pc.decisionStats.Reset()
		pc.decisionStats.Tiles = 1
		st.decisionStats = &pc.decisionStats
	}
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
	// The merge decisions price candidates with the real coefficient coder:
	// trial TXBs run through WriteCoefficientsTXB against this throwaway
	// context set, and distortion pairs with rate under the inter coder's
	// multiplier shape.
	if err := st.trialCDFs.InitDefault(qIndex); err != nil {
		return nil, err
	}
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	st.keyMIColEnd = uint32(miColEnd)
	st.keyMIRowEnd = uint32(src.Height / 4)
	st.keyVisW, st.keyVisH = src.Width, src.Height
	if err := st.initScans(); err != nil {
		return nil, err
	}

	if cap(pc.writerBuf) == 0 {
		pc.writerBuf = make([]byte, 1<<18)
	}
	w := entropy.NewWriter(pc.writerBuf[:0])
	st.w = &w

	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB

	scratch := &pc.scratch
	if len(pc.carrier.Above) < rootCols {
		pc.carrier.Above = make([]tile.BlockLoopRootAboveContext, rootCols)
	}
	pc.carrier.Left = tile.BlockLoopRootLeftContext{}
	for i := range pc.carrier.Above[:rootCols] {
		pc.carrier.Above[i] = tile.BlockLoopRootAboveContext{}
	}
	carrier := &pc.carrier
	walkReq := tile.BlockWalkRequest{
		Root:       tile.BlockLevel64x64,
		MIColStart: miColStart,
		MIColEnd:   miColEnd,
		MIRowEnd:   miRows,
	}
	// intraRDCost prices one candidate exactly: best intra prediction, the
	// true transform-quantize pass for distortion, and the real coefficient
	// coder for rate (trial-coded against throwaway contexts, identical for
	// every candidate). Whole-block candidates predict from the coded
	// reconstruction; child estimates predict from the source - their
	// neighbors are not reconstructed yet, and lossy reconstructions track
	// the source closely enough for a split decision. Keys are rare, so the
	// trial coding prices in microseconds per node.
	intraRDCost := func(plane []byte, px, py, n int) (int64, int64) {
		haveTop := py > int(walkReq.MIRowStart)*4
		haveLeft := px > int(walkReq.MIColStart)*4
		selectIntraModeN(src.Y, plane, src.YStride, px, py, n, haveTop, haveLeft, st.predY[:n*n])
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		st.prepareInterTXB(src.Y, st.predY[:n*n], n, src.YStride, px, py, n, n, st.yQuant, st.lumaQ2[:n*n])
		lumaD := st.rdDcode
		cost := st.trialTXBBits(tile.CoeffPlaneY, st.lumaQ2[:n*n], n) + lumaD<<7
		// Merge candidates differ in chroma transform structure too: one
		// large block against four small ones.
		cost += st.trialChromaCost(src, recon, &plane[0] == &recon.Y[0], px, py, n, haveTop, haveLeft)
		return cost, lumaD
	}
	// intraHeaderCost is the priced overhead of one extra block's prefix
	// symbols (partition, mode, angle, chroma).
	intraHeaderCost := (int64(20) << 9) * st.rdMult >> 9
	decideCore := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		px, py := int(miCol)*4, int(miRow)*4
		if level == tile.BlockLevel32x32 && haveRight && haveBottom {
			// The 32-tier keeps the near-perfect gate: trial pricing
			// misjudges the largest transform even with chroma included
			// (the throwaway contexts diverge most from the real tile
			// state at that size).
			if px+32 > src.Width || py+32 > src.Height {
				return tile.PartitionSplit, nil
			}
			selectIntraModeN(src.Y, recon.Y, src.YStride, px, py, 32, py > int(walkReq.MIRowStart)*4, px > int(walkReq.MIColStart)*4, st.predY[:32*32])
			sad := 0
			for r := range 32 {
				row := (py+r)*src.YStride + px
				for c := range 32 {
					d := int(src.Y[row+c]) - int(st.predY[r*32+c])
					if d < 0 {
						d = -d
					}
					sad += d
				}
			}
			if sad <= 32*32*2 {
				return tile.PartitionNone, nil
			}
			return tile.PartitionSplit, nil
		}
		if level == tile.BlockLevel16x16 && haveRight && haveBottom {
			if px+16 > src.Width || py+16 > src.Height {
				return tile.PartitionSplit, nil
			}
			child := int64(0)
			for _, o8 := range [4][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
				cc, _ := intraRDCost(src.Y, px+o8[0], py+o8[1], 8)
				child += cc + intraHeaderCost
			}
			mc, _ := intraRDCost(recon.Y, px, py, 16)
			if mc+intraHeaderCost <= child {
				return tile.PartitionNone, nil
			}
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
		return st.encodeBlock(src, recon, block, scratch)
	}
	if err := tile.WalkBlockLoopWrite(&w, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

func (pc *pframeCoder) encodeMonochromeKeyframeTile(src SourceFrameMono, recon *SourceFrameMono, qIndex uint8, miColStart, miColEnd uint16) ([]byte, error) {
	if err := pc.partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &pc.st
	st.qIndex = qIndex
	st.forceIntegerMV = false
	st.allowScreenContentTools = false
	st.color = parser.ColorConfig{BitDepth: 8, MonoChrome: true, SubsamplingX: true, SubsamplingY: true}
	st.lfMap = nil
	st.hme = nil
	st.decisionStats = nil
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
	q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.Plane(0))
	if err != nil {
		return nil, err
	}
	st.yQuant = q
	if err := st.trialCDFs.InitDefault(qIndex); err != nil {
		return nil, err
	}
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	st.keyMIColEnd = uint32(miColEnd)
	st.keyMIRowEnd = uint32(src.Height / 4)
	st.keyVisW, st.keyVisH = src.Width, src.Height
	if err := st.initScans(); err != nil {
		return nil, err
	}

	if cap(pc.writerBuf) == 0 {
		pc.writerBuf = make([]byte, 1<<18)
	}
	w := entropy.NewWriter(pc.writerBuf[:0])
	st.w = &w

	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB

	scratch := &pc.scratch
	if len(pc.carrier.Above) < rootCols {
		pc.carrier.Above = make([]tile.BlockLoopRootAboveContext, rootCols)
	}
	pc.carrier.Left = tile.BlockLoopRootLeftContext{}
	for i := range pc.carrier.Above[:rootCols] {
		pc.carrier.Above[i] = tile.BlockLoopRootAboveContext{}
	}
	carrier := &pc.carrier
	walkReq := tile.BlockWalkRequest{
		Root:       tile.BlockLevel64x64,
		MIColStart: miColStart,
		MIColEnd:   miColEnd,
		MIRowEnd:   miRows,
	}
	intraRDCost := func(plane []byte, px, py, n int) (int64, int64) {
		haveTop := py > int(walkReq.MIRowStart)*4
		haveLeft := px > int(walkReq.MIColStart)*4
		selectIntraModeN(src.Y, plane, src.YStride, px, py, n, haveTop, haveLeft, st.predY[:n*n])
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		st.prepareInterTXB(src.Y, st.predY[:n*n], n, src.YStride, px, py, n, n, st.yQuant, st.lumaQ2[:n*n])
		lumaD := st.rdDcode
		cost := st.trialTXBBits(tile.CoeffPlaneY, st.lumaQ2[:n*n], n) + lumaD<<7
		return cost, lumaD
	}
	intraHeaderCost := (int64(16) << 9) * st.rdMult >> 9
	decideCore := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		px, py := int(miCol)*4, int(miRow)*4
		if level == tile.BlockLevel32x32 && haveRight && haveBottom {
			if px+32 > src.Width || py+32 > src.Height {
				return tile.PartitionSplit, nil
			}
			selectIntraModeN(src.Y, recon.Y, src.YStride, px, py, 32, py > int(walkReq.MIRowStart)*4, px > int(walkReq.MIColStart)*4, st.predY[:32*32])
			sad := 0
			for r := range 32 {
				row := (py+r)*src.YStride + px
				for c := range 32 {
					d := int(src.Y[row+c]) - int(st.predY[r*32+c])
					if d < 0 {
						d = -d
					}
					sad += d
				}
			}
			if sad <= 32*32*2 {
				return tile.PartitionNone, nil
			}
			return tile.PartitionSplit, nil
		}
		if level == tile.BlockLevel16x16 && haveRight && haveBottom {
			if px+16 > src.Width || py+16 > src.Height {
				return tile.PartitionSplit, nil
			}
			child := int64(0)
			for _, o8 := range [4][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
				cc, _ := intraRDCost(src.Y, px+o8[0], py+o8[1], 8)
				child += cc + intraHeaderCost
			}
			mc, _ := intraRDCost(recon.Y, px, py, 16)
			if mc+intraHeaderCost <= child {
				return tile.PartitionNone, nil
			}
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
		return st.encodeMonochromeBlock(src, recon, block, scratch)
	}
	if err := tile.WalkBlockLoopWrite(&w, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
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
	mode, angleDelta := st.improveIntraModeDirectional(src, recon, block, mode, pred, lumaPX, lumaPY, n)

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
	}, int8(angleDelta)); err != nil {
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
	if err := st.writeNoPaletteMode(modeCtx, block, mode, tile.ChromaIntraModeDC, true); err != nil {
		return err
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
	if st.decisionStats != nil {
		st.decisionStats.noteIntraBlock(block.Size)
	}
	return nil
}

func (st *lossyEncodeState) encodeMonochromeBlock(src SourceFrameMono, recon *SourceFrameMono, block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
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
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}

	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	pred := st.predY[:n*n]
	mode := selectIntraModeN(src.Y, recon.Y, src.YStride, lumaPX, lumaPY, n, block.HaveTop, block.HaveLeft, pred)
	src420 := SourceFrame420{Y: src.Y, YStride: src.YStride, Width: src.Width, Height: src.Height}
	recon420 := SourceFrame420{Y: recon.Y, YStride: recon.YStride, Width: recon.Width, Height: recon.Height}
	mode, angleDelta := st.improveIntraModeDirectional(src420, &recon420, block, mode, pred, lumaPX, lumaPY, n)

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
	}, int8(angleDelta)); err != nil {
		return fmt.Errorf("angle delta: %w", err)
	}
	if err := st.writeNoPaletteMode(modeCtx, block, mode, tile.ChromaIntraModeDC, false); err != nil {
		return err
	}

	lumaTX, lumaScan := tile.TransformSize8x8, st.scan8
	switch n {
	case 16:
		lumaTX, lumaScan = tile.TransformSize16x16, st.scan16
	case 32:
		lumaTX, lumaScan = tile.TransformSize32x32, st.scan32
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
	if st.decisionStats != nil {
		st.decisionStats.noteIntraBlock(block.Size)
	}
	return nil
}

func (st *lossyEncodeState) writeNoPaletteMode(modeCtx *tile.BlockModeContext, block tile.BlockVisit, lumaMode tile.IntraMode, chromaMode tile.ChromaIntraMode, chromaModeValid bool) error {
	if !st.allowScreenContentTools {
		return nil
	}
	hasChroma := tile.HasChromaBlock(tile.TransformTreeRequest{
		Size: block.Size,
		X4:   block.X4,
		Y4:   block.Y4,
	}, st.color)
	if err := tile.WriteNoPaletteMode(st.w, &st.intraCDFs, modeCtx, tile.PaletteModeRequest{
		AllowScreenContentTools: st.allowScreenContentTools,
		Size:                    block.Size,
		LumaMode:                lumaMode,
		X4:                      block.X4,
		Y4:                      block.Y4,
		HaveTop:                 block.HaveTop,
		HaveLeft:                block.HaveLeft,
		BitDepth:                st.color.BitDepth,
		Color:                   st.color,
		ChromaMode:              chromaMode,
		ChromaModeValid:         chromaModeValid,
		HasChroma:               hasChroma,
	}); err != nil {
		return fmt.Errorf("palette: %w", err)
	}
	if err := modeCtx.MarkPaletteY(block.Size, int(block.X4), int(block.Y4), tile.PaletteModeResult{}); err != nil {
		return fmt.Errorf("mark palette y: %w", err)
	}
	if err := modeCtx.MarkPaletteUV(block.Size, int(block.X4), int(block.Y4), tile.PaletteModeResult{}); err != nil {
		return fmt.Errorf("mark palette uv: %w", err)
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

// keyDirectionalModes are the diagonal candidates and their base angles;
// vertical and horizontal already compete in the flat selector.
var keyDirectionalModes = [6]struct {
	mode  tile.IntraMode
	angle int
}{
	{tile.IntraModeD45, 45},
	{tile.IntraModeD67, 67},
	{tile.IntraModeD113, 113},
	{tile.IntraModeD135, 135},
	{tile.IntraModeD157, 157},
	{tile.IntraModeD203, 203},
}

// improveIntraModeDirectional tries the diagonal intra modes against the flat
// selector's winner. Candidate predictions come from the decoder's own edge
// construction and directional predictors, so a chosen mode reconstructs
// bit-exactly; a SAD pre-filter keeps the exact-rate trials to the two most
// promising angles. The streams code no intra edge filter, so the edges are
// the raw reconstructed neighbors throughout.
func (st *lossyEncodeState) improveIntraModeDirectional(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, mode tile.IntraMode, pred []byte, px, py, n int) (tile.IntraMode, int) {
	if !block.HaveTop || !block.HaveLeft || n != 8 {
		// Larger blocks exist only where the merge gates proved the flat
		// prediction tight; the diagonal search pays on detail blocks.
		return mode, 0
	}
	// Diagonals only pay where the flat winner predicts poorly; a tight
	// flat match skips the search and keeps keyframe latency in budget.
	flatSAD := 0
	for r := 0; r < n; r++ {
		row := (py+r)*src.YStride + px
		for c := 0; c < n; c++ {
			d := int(src.Y[row+c]) - int(pred[r*n+c])
			if d < 0 {
				d = -d
			}
			flatSAD += d
		}
	}
	if flatSAD <= n*n*4 {
		return mode, 0
	}
	reconPlane := frame.Plane{Pix: recon.Y, Stride: src.YStride, Width: recon.Width, Height: recon.Height}
	cand := st.sadScratch[:n*n]
	candPlane := frame.Plane{Pix: cand, Stride: n, Width: n, Height: n}
	type scored struct {
		idx int
		sad int
	}
	best1, best2 := scored{-1, 1 << 30}, scored{-1, 1 << 30}
	predictAngle := func(angle int) bool {
		edges, err := threading.BuildLumaDirectionalEdges(reconPlane, 8, px, py, n, n, angle, block, &st.intraEdgeScratch, 16, st.keyMIColEnd, st.keyMIRowEnd, false, false, st.keyVisW, st.keyVisH)
		if err != nil {
			return false
		}
		return prediction.PredictDirectionalIntraPlaneBlock(candPlane, 1, 8, 0, 0, n, n, angle, edges) == nil
	}
	candSAD := func() int {
		sad := 0
		for r := 0; r < n; r++ {
			row := (py+r)*src.YStride + px
			for c := 0; c < n; c++ {
				d := int(src.Y[row+c]) - int(cand[r*n+c])
				if d < 0 {
					d = -d
				}
				sad += d
			}
		}
		return sad
	}
	predict := func(i int) bool { return predictAngle(keyDirectionalModes[i].angle) }
	for i := range keyDirectionalModes {
		if !predict(i) {
			continue
		}
		sad := candSAD()
		if sad < best1.sad {
			best2 = best1
			best1 = scored{i, sad}
		} else if sad < best2.sad {
			best2 = scored{i, sad}
		}
	}
	if best1.idx < 0 || best1.sad >= flatSAD {
		// No diagonal even matches the flat winner's prediction; skip the
		// exact-rate trials entirely.
		return mode, 0
	}
	baseCost := st.trialLumaCost(src.Y, pred, src.YStride, px, py, n)
	// A directional mode also codes an angle-delta symbol the flat modes
	// skip; two bits priced under the frame multiplier covers it.
	angleCost := (int64(2) << 9) * st.rdMult >> 9
	bestMode, bestCost := mode, baseCost
	bestIdx := -1
	for _, cnd := range [2]scored{best1, best2} {
		if cnd.idx < 0 {
			continue
		}
		if !predict(cnd.idx) {
			continue
		}
		cost := st.trialLumaCost(src.Y, cand, src.YStride, px, py, n) + angleCost
		if cost < bestCost {
			bestMode, bestCost, bestIdx = keyDirectionalModes[cnd.idx].mode, cost, cnd.idx
			copy(pred, cand)
		}
	}
	if bestIdx < 0 {
		return bestMode, 0
	}
	// Angle refinement around the winning diagonal: three-degree steps,
	// SAD-picked, one exact-rate trial; a finer angle prices one extra
	// bit per step away from the base.
	base := keyDirectionalModes[bestIdx].angle
	bestDelta := 0
	dSAD, dPick := 1<<30, 0
	for delta := -3; delta <= 3; delta++ {
		if delta == 0 || !predictAngle(base+3*delta) {
			continue
		}
		if sad := candSAD(); sad < dSAD {
			dSAD, dPick = sad, delta
		}
	}
	if dPick != 0 && predictAngle(base+3*dPick) {
		extra := (int64(2+abs(dPick)) << 9) * st.rdMult >> 9
		if cost := st.trialLumaCost(src.Y, cand, src.YStride, px, py, n) + angleCost + extra; cost < bestCost {
			bestDelta = dPick
			copy(pred, cand)
		}
	}
	return bestMode, bestDelta
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// trialLumaCost prices coding pred against the source block exactly: the
// true transform-quantize pass for distortion and the real coefficient coder
// for bits, trial-coded against the throwaway context set.
func (st *lossyEncodeState) trialLumaCost(srcPlane, pred []byte, stride, px, py, n int) int64 {
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(srcPlane, pred, n, stride, px, py, n, n, st.yQuant, st.lumaQ2[:n*n])
	return st.trialTXBBits(tile.CoeffPlaneY, st.lumaQ2[:n*n], n) + st.rdDcode<<7
}

// trialTXBBits trial-codes one quantized transform block and returns the
// priced rate term.
func (st *lossyEncodeState) trialTXBBits(plane tile.CoeffPlaneType, qcoeff []int16, n int) int64 {
	if n == 4 {
		if plane == tile.CoeffPlaneY {
			return st.trialTXBBitsY4x4((*[16]int16)(qcoeff))
		}
		if plane == tile.CoeffPlaneUV {
			return st.trialTXBBitsUV4x4((*[16]int16)(qcoeff))
		}
	}
	if n == 8 {
		if plane == tile.CoeffPlaneY {
			return st.trialTXBBitsY8x8((*[64]int16)(qcoeff))
		}
		if plane == tile.CoeffPlaneUV {
			return st.trialTXBBitsUV8x8((*[64]int16)(qcoeff))
		}
	}
	if n == 16 {
		if plane == tile.CoeffPlaneY {
			return st.trialTXBBitsY16x16((*[256]int16)(qcoeff))
		}
		if plane == tile.CoeffPlaneUV {
			return st.trialTXBBitsUV16x16((*[256]int16)(qcoeff))
		}
	}
	if n == 32 {
		if plane == tile.CoeffPlaneY {
			return st.trialTXBBitsY32x32((*[1024]int16)(qcoeff))
		}
		if plane == tile.CoeffPlaneUV {
			return st.trialTXBBitsUV32x32((*[1024]int16)(qcoeff))
		}
	}
	size, scan := tile.TransformSize4x4, st.scan4
	switch n {
	case 8:
		size, scan = tile.TransformSize8x8, st.scan8
	case 16:
		size, scan = tile.TransformSize16x16, st.scan16
	case 32:
		size, scan = tile.TransformSize32x32, st.scan32
	}
	tw := entropy.NewCountingWriter()
	base := tw.Tell()
	if _, err := tile.WriteCoefficientsTXB(&tw, &st.trialCDFs, tile.TXBEncodeRequest{
		Size: size, Plane: plane, Class: transform.Class2D,
	}, qcoeff[:n*n], scan, st.levels); err != nil {
		return 1 << 59
	}
	bits := int64(tw.Tell() - base)
	return st.txbBitsToRate(int(bits))
}

func (st *lossyEncodeState) trialTXBBitsY4x4(qcoeff *[16]int16) int64 {
	_, bits := tile.CountCoefficientsTXB4x4Y2DTrustedArray(&st.trialCDFs, qcoeff)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) trialTXBBitsUV4x4(qcoeff *[16]int16) int64 {
	_, bits := tile.CountCoefficientsTXB4x4UV2DTrustedArray(&st.trialCDFs, qcoeff)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) trialTXBBitsY8x8(qcoeff *[64]int16) int64 {
	_, bits := tile.CountCoefficientsTXB8x8Y2DTrustedArray(&st.trialCDFs, qcoeff, nil, 0)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) trialTXBBitsUV8x8(qcoeff *[64]int16) int64 {
	_, bits := tile.CountCoefficientsTXB8x8UV2DTrustedArray(&st.trialCDFs, qcoeff)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) trialTXBBitsY16x16(qcoeff *[256]int16) int64 {
	_, bits := tile.CountCoefficientsTXB16x16Y2DTrustedArray(&st.trialCDFs, qcoeff)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) trialTXBBitsUV16x16(qcoeff *[256]int16) int64 {
	_, bits := tile.CountCoefficientsTXB16x16UV2DTrustedArray(&st.trialCDFs, qcoeff)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) trialTXBBitsY32x32(qcoeff *[1024]int16) int64 {
	_, bits := tile.CountCoefficientsTXB32x32Y2DTrustedArray(&st.trialCDFs, qcoeff)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) trialTXBBitsUV32x32(qcoeff *[1024]int16) int64 {
	_, bits := tile.CountCoefficientsTXB32x32UV2DTrustedArray(&st.trialCDFs, qcoeff)
	return st.txbBitsToRate(bits)
}

func (st *lossyEncodeState) txbBitsToRate(bits int) int64 {
	return ((int64(bits)<<9)*st.rdMult + 256) >> 9
}

func (st *lossyEncodeState) trialTXBBitsInter(qcoeff []int16, n int, size tile.TransformSize, typ transform.Type) int64 {
	if n == 8 && size == tile.TransformSize8x8 {
		return st.trialTXBBitsInter8x8((*[64]int16)(qcoeff), typ)
	}
	if n == 16 && size == tile.TransformSize16x16 {
		return st.trialTXBBitsInter16x16((*[256]int16)(qcoeff), typ)
	}
	if n == 32 && size == tile.TransformSize32x32 {
		return st.trialTXBBitsInter32x32((*[1024]int16)(qcoeff), typ)
	}

	tw := entropy.NewCountingWriter()
	scan := st.scan4
	var txCDF *entropy.CDF
	txSymbol := 0
	if st.qIndex == 0 {
		if typ != transform.TypeDCTDCT {
			return 1 << 59
		}
	} else {
		set, err := tile.ExtTXSetTypeFor(size, true, false)
		if err != nil {
			return 1 << 59
		}
		symbols, err := tile.ExtTXTypeCount(set)
		if err != nil {
			return 1 << 59
		}
		if symbols <= 1 {
			if typ != transform.TypeDCTDCT {
				return 1 << 59
			}
		} else {
			index, err := tile.ExtTXSetIndex(size, true, false)
			if err != nil {
				return 1 << 59
			}
			square, err := tile.TransformSizeSquare(size)
			if err != nil {
				return 1 << 59
			}
			txCDF, err = st.txCDFs.InterCDF(index, square, symbols)
			if err != nil {
				return 1 << 59
			}
			txSymbol, err = tile.ExtTXSymbolForType(set, typ)
			if err != nil {
				return 1 << 59
			}
		}
	}
	base := tw.Tell()
	afterSkip := func() error {
		if txCDF == nil {
			return nil
		}
		saved := *txCDF
		tw.WriteCDF(txCDF, txSymbol)
		*txCDF = saved
		return nil
	}
	if _, err := tile.WriteCoefficientsTXB(&tw, &st.trialCDFs, tile.TXBEncodeRequest{
		Size: size, Plane: tile.CoeffPlaneY, Class: transform.Class2D, AfterSkip: afterSkip,
	}, qcoeff[:n*n], scan, st.levels); err != nil {
		return 1 << 59
	}
	bits := int64(tw.Tell() - base)
	return ((bits<<9)*st.rdMult + 256) >> 9
}

func (st *lossyEncodeState) trialTXBBitsInter8x8(qcoeff *[64]int16, typ transform.Type) int64 {
	txCDF, txSymbol, ok := st.inter8x8TXCDFAndSymbol(typ)
	if !ok {
		return 1 << 59
	}
	_, bits := tile.CountCoefficientsTXB8x8Y2DTrustedArray(&st.trialCDFs, qcoeff, txCDF, txSymbol)
	rate := int64(bits)
	return ((rate<<9)*st.rdMult + 256) >> 9
}

func (st *lossyEncodeState) trialTXBBitsInter16x16(qcoeff *[256]int16, typ transform.Type) int64 {
	txCDF, txSymbol, ok := st.interTXCDFAndSymbol(tile.TransformSize16x16, typ)
	if !ok {
		return 1 << 59
	}
	_, bits := tile.CountCoefficientsTXB16x16Y2DTrustedWithTXTypeArray(&st.trialCDFs, qcoeff, txCDF, txSymbol)
	rate := int64(bits)
	return ((rate<<9)*st.rdMult + 256) >> 9
}

func (st *lossyEncodeState) trialTXBBitsInter32x32(qcoeff *[1024]int16, typ transform.Type) int64 {
	txCDF, txSymbol, ok := st.interTXCDFAndSymbol(tile.TransformSize32x32, typ)
	if !ok {
		return 1 << 59
	}
	_, bits := tile.CountCoefficientsTXB32x32Y2DTrustedWithTXTypeArray(&st.trialCDFs, qcoeff, txCDF, txSymbol)
	rate := int64(bits)
	return ((rate<<9)*st.rdMult + 256) >> 9
}

// trialChromaCost prices the two DC-predicted chroma transform blocks of an
// intra block whose luma corner sits at (px, py) with luma size n. Merge
// candidates differ in chroma transform structure - one large block against
// four small ones - so the split decision must see that rate too. Children
// estimates predict from the source plane like their luma halves.
func (st *lossyEncodeState) trialChromaCost(src SourceFrame420, recon *SourceFrame420, useRecon bool, px, py, n int, haveTop, haveLeft bool) int64 {
	cn := n / 2
	cx, cy := px/2, py/2
	total := int64(0)
	for plane := 1; plane <= 2; plane++ {
		data, rdata, q := src.U, recon.U, st.uQuant
		predBuf, qc := st.predU[:cn*cn], st.uQ[:cn*cn]
		if plane == 2 {
			data, rdata, q = src.V, recon.V, st.vQuant
			predBuf, qc = st.predV[:cn*cn], st.vQ[:cn*cn]
		}
		edgePlane := rdata
		if !useRecon {
			edgePlane = data
		}
		dc := dcPredictN(edgePlane, src.ChromaStride, cx, cy, cn, haveTop, haveLeft)
		for i := range predBuf {
			predBuf[i] = dc
		}
		st.rdDcode = 0
		st.prepareInterTXB(data, predBuf, cn, src.ChromaStride, cx, cy, cn, cn, q, qc)
		total += st.trialTXBBits(tile.CoeffPlaneUV, qc, cn) + st.rdDcode<<7
	}
	return total
}

// selectIntraMode8 is selectIntraModeN at the 8x8 fallback size.
func selectIntraMode8(srcPlane, reconPlane []byte, stride, px, py int, haveTop, haveLeft bool, pred []byte) tile.IntraMode {
	return selectIntraModeN(srcPlane, reconPlane, stride, px, py, 8, haveTop, haveLeft, pred)
}
