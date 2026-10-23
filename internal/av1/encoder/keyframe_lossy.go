package encoder

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
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

// KeyframeEncoder is a reusable keyframe-only encoder. It keeps the same
// writer, tile-coder, reconstruction, and filter scratch between calls as the
// streaming VideoEncoder, so repeated forced keyframes do not allocate once
// Prewarm has run. Returned temporal units and reconstructions alias
// encoder-owned storage and remain valid until the next Encode call.
type KeyframeEncoder struct {
	enc *VideoEncoder
}

// NewKeyframeEncoder creates a reusable keyframe encoder for same-sized 8-bit
// 4:2:0 frames. qIndex must be non-zero; SetQIndex can update it later without
// rebuilding the scratch state.
func NewKeyframeEncoder(width, height int, qIndex uint8) (*KeyframeEncoder, error) {
	enc, err := NewVideoEncoder(width, height, qIndex)
	if err != nil {
		return nil, err
	}
	return &KeyframeEncoder{enc: enc}, nil
}

// Prewarm sizes the reusable scratch and worker pool before the first externally
// visible Encode call.
func (e *KeyframeEncoder) Prewarm() error {
	if e == nil || e.enc == nil {
		return fmt.Errorf("encoder: nil keyframe encoder")
	}
	return e.enc.Prewarm()
}

// Close waits for background filter work and releases persistent workers.
func (e *KeyframeEncoder) Close() error {
	if e == nil || e.enc == nil {
		return nil
	}
	err := e.enc.Close()
	e.enc = nil
	return err
}

// SetQIndex updates the fixed keyframe qindex used by later Encode calls.
func (e *KeyframeEncoder) SetQIndex(qIndex uint8) error {
	if e == nil || e.enc == nil {
		return fmt.Errorf("encoder: nil keyframe encoder")
	}
	return e.enc.SetQIndex(qIndex)
}

// Encode emits src as a shown keyframe and returns the encoder-side
// reconstruction a conformant decoder must reproduce exactly.
func (e *KeyframeEncoder) Encode(src SourceFrame420) ([]byte, SourceFrame420, error) {
	if e == nil || e.enc == nil {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: nil keyframe encoder")
	}
	tu, key, err := e.enc.Encode(src, true)
	if err != nil {
		return nil, SourceFrame420{}, err
	}
	if !key {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: forced keyframe was not coded as keyframe")
	}
	return tu, e.enc.Recon(), nil
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
	out, err = AppendLowOverheadIntraFrameTemporalUnit(out, seq, header, groupOBU.Payload)
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("append header TU: %w", err)
	}
	return out, recon, nil
}

// EncodeKeyframeWithSequenceMax encodes src as a shown keyframe while keeping
// the sequence header max dimensions at maxWidth x maxHeight. This is useful
// for scalable streams whose lower spatial layers are smaller than the shared
// sequence maximum.
func EncodeKeyframeWithSequenceMax(src SourceFrame420, qIndex uint8, maxWidth, maxHeight int) ([]byte, SourceFrame420, error) {
	return encodeKeyframeFilteredTiles(src, qIndex, nil, 0, 0, nil, nil, nil, 0, keyframeTileOptions{
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
	return encodeKeyframeFilteredTiles(src, qIndex, lf, renderW, renderH, reconBuf, tilePC, cdefApp, 0, keyframeTileOptions{})
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
	if reconBuf != nil && reconBuf.Y != nil {
		return encodeKeyframeFilteredTilesInto(src, qIndex, lf, renderW, renderH, reconBuf, tilePC, cdefApp, tileColsLog2, tileOpts, seqWidth, seqHeight)
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
	if reconBuf != nil {
		*reconBuf = recon
		return encodeKeyframeFilteredTilesInto(src, qIndex, lf, renderW, renderH, reconBuf, tilePC, cdefApp, tileColsLog2, tileOpts, seqWidth, seqHeight)
	}
	return encodeKeyframeFilteredTilesInto(src, qIndex, lf, renderW, renderH, &recon, tilePC, cdefApp, tileColsLog2, tileOpts, seqWidth, seqHeight)
}

func encodeKeyframeFilteredTilesInto(src SourceFrame420, qIndex uint8, lf *loopFilterApplier, renderW, renderH int, recon *SourceFrame420, tilePC func(t int) *pframeCoder, cdefApp *cdefApplier, tileColsLog2 uint8, tileOpts keyframeTileOptions, seqWidth, seqHeight int) ([]byte, SourceFrame420, error) {
	if recon == nil || recon.Y == nil {
		recon = &SourceFrame420{
			Y:            make([]byte, len(src.Y)),
			U:            make([]byte, len(src.U)),
			V:            make([]byte, len(src.V)),
			YStride:      src.YStride,
			ChromaStride: src.ChromaStride,
			Width:        src.Width,
			Height:       src.Height,
		}
	}
	color := parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	if tileOpts.stream != nil {
		color = tileOpts.stream.parserColorConfig()
	}
	lfLevel := uint8(0)
	if videoColorSupportsInLoopFilters(color) && src.Width*src.Height <= loopFilterMaxArea {
		lfLevel = filterLevelFromQIndex(qIndex, true)
	}
	seq := losslessKeyframeSequence(seqWidth, seqHeight)
	if tileOpts.stream != nil {
		seq = tileOpts.stream.sequenceHeader(seqWidth, seqHeight)
	}
	// SVT-AV1 rtc CDEF on the I slice (cdef_frame.go): cdef_search_level 5,
	// searched after the tiles, whose symbol streams are recorded.
	screen := tileOpts.stream != nil && tileOpts.stream.svtScreen()
	cdefPlan := planSVTCDEF(videoColorSupportsInLoopFilters(color) && seq.EnableCDEF && src.Width*src.Height <= loopFilterMaxArea,
		true, true, true, screen, svtCDEFRefStrengths{}, 0)
	var lfMap *threading.FrameWorkLoopFilterMap
	if lfLevel > 0 || cdefPlan.on {
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
	header := lossyKeyframeHeaderForSequence(seq, src.Width, src.Height, qIndex)
	if tileOpts.stream != nil {
		tileOpts.stream.applyContentHintToFrameHeader(&header.Prefix)
	}
	if renderW > 0 && (renderW != src.Width || renderH != src.Height) {
		header.Size.RenderWidth = uint32(renderW)
		header.Size.RenderHeight = uint32(renderH)
		header.Size.HaveRenderSize = true
	}
	// The video encoder follows SVT-AV1 preset 12 rtc: no deblocking
	// (get_dlf_level_rtc returns dlf_level 0), CDEF still runs.
	dlf := lfLevel
	if tileOpts.stream != nil {
		dlf = svtRTCDeblockLevel(lfLevel)
		header.LoopFilter.LevelY = [2]uint8{}
		header.LoopFilter.LevelU, header.LoopFilter.LevelV = 0, 0
	}
	if lfLevel > 0 || cdefPlan.on {
		header.LoopFilter.LevelY = [2]uint8{dlf, dlf}
		header.LoopFilter.LevelU = dlf
		header.LoopFilter.LevelV = dlf
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
		recon:                   recon,
		qIndex:                  qIndex,
		color:                   color,
		allowScreenContentTools: header.Prefix.AllowScreenContentTools,
		tile:                    header.Tile,
		miCols:                  miCols,
		lfMap:                   lfMap,
		payloads:                payloads,
		errs:                    errs,
	}
	// coders lists the tile coders in tile order for the cdef_idx replay.
	var coders []*pframeCoder
	var logs *svtCDEFLogs
	if cdefPlan.search {
		coders = keyframeTileCoders(tileOpts.stream, &tilePC, nTiles)
		if tileOpts.stream != nil {
			logs = &tileOpts.stream.cdefLogs
		} else {
			logs = new(svtCDEFLogs)
		}
		logs.arm(coders)
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

	var cdefIdx []uint8
	if cdefPlan.on {
		var searcher *cdefSearcher
		if tileOpts.stream != nil {
			searcher = &tileOpts.stream.cdefSearch
		} else {
			searcher = new(cdefSearcher)
		}
		cdefPlan.lambda = svtCDEFFullLambda(qIndex, true, false, 150)
		header.CDEF, cdefIdx = cdefPlan.decide(searcher, &src, recon, lfMap, qIndex)
		if cdefPlan.search {
			err := logs.replay(payloads, coders, header.Tile, miCols, src.Height, cdefIdx, header.CDEF.Bits)
			logs.disarm(coders)
			if err != nil {
				return nil, SourceFrame420{}, fmt.Errorf("cdef replay: %w", err)
			}
		}
	}
	if tileOpts.stream != nil {
		ref := svtCDEFRefFromParams(header.CDEF)
		tileOpts.stream.cdefRefLast, tileOpts.stream.cdefRefT1 = ref, ref
	}
	if lfLevel > 0 || cdefPlan.on {
		lfParams := parser.LoopFilterParams{
			LevelY: [2]uint8{dlf, dlf},
			LevelU: dlf,
			LevelV: dlf,
		}
		applyLF := lf.apply
		applyCDEF := cdefApp.apply
		if tileOpts.stream != nil && tileOpts.stream.singleThread {
			// singleThread forces a single keyframe tile (SetTileColumns(1)),
			// so the whole-frame mask build matches the edge-list pass exactly.
			applyLF = lf.applySerialMasks
			applyCDEF = cdefApp.applySerial
		}
		if dlf > 0 {
			if err := applyLF(recon, lfParams); err != nil {
				return nil, SourceFrame420{}, fmt.Errorf("loop filter apply: %w", err)
			}
		}
		if err := applyCDEF(recon, cdefParserParams(header.CDEF), &lf.filtMap, cdefIdx); err != nil {
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
	var group []byte
	if tileOpts.groupScratch != nil {
		if cap(*tileOpts.groupScratch) < groupSize {
			*tileOpts.groupScratch = make([]byte, 0, groupSize+groupSize/2)
		}
		group = (*tileOpts.groupScratch)[:0]
	} else {
		group = make([]byte, 0, groupSize)
	}
	group, err = AppendTileGroupPayload(group, header.Tile, 0, endTile, payloads)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("append tile group: %w", err)
	}
	if tileOpts.groupScratch != nil {
		*tileOpts.groupScratch = group
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, SourceFrame420{}, err
	}
	total := headerSize + groupOBUSize
	var out []byte
	if tileOpts.outScratch != nil {
		if cap(*tileOpts.outScratch) < total {
			*tileOpts.outScratch = make([]byte, 0, total+total/2)
		}
		out = (*tileOpts.outScratch)[:0]
	} else {
		out = make([]byte, 0, total)
	}
	out, err = AppendLowOverheadIntraFrameTemporalUnit(out, seq, header, groupOBU.Payload)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("append header TU: %w", err)
	}
	if tileOpts.outScratch != nil {
		*tileOpts.outScratch = out
	}
	return out, *recon, nil
}

type keyframeTileOptions struct {
	payloads          []TilePayload
	errs              []error
	groupScratch      *[]byte
	outScratch        *[]byte
	stream            *VideoEncoder
	sequenceMaxWidth  int
	sequenceMaxHeight int
}

type keyframeTileRun struct {
	src                     SourceFrame420
	recon                   *SourceFrame420
	qIndex                  uint8
	color                   parser.ColorConfig
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
			data, err := pc.encodeKeyframeTileWithColorOptions(req.src, req.recon, req.qIndex, c0, c1, req.lfMap, req.allowScreenContentTools, req.color)
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
	tile0, tile0Err := pc0.encodeKeyframeTileWithColorOptions(req.src, req.recon, req.qIndex, c0, c1, req.lfMap, req.allowScreenContentTools, req.color)
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
	if seq.EnableCDEF && videoColorSupportsInLoopFilters(parserColorConfig(seq.ColorConfig)) {
		header.CDEF = CDEFParams{Damping: 3}
	} else {
		header.LoopFilter = LoopFilterParams{Deltas: defaultLoopFilterDeltas()}
	}
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
	// zmvCtrl is the picture's zero-mv static-superblock exit controls
	// (zeromv_skip.go).
	zmvCtrl   zmvCtrl
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
	effortLevel             int8
	sadCacheEpoch           uint32
	yQuant                  quantize.Quantizer
	uQuant                  quantize.Quantizer
	vQuant                  quantize.Quantizer

	scan8, scan4, scan16, scan32 []int16
	scan16x8, scan8x16           []int16
	scan32x16, scan16x32         []int16
	scan8x4, scan4x8             []int16
	scan4x16, scan16x4           []int16
	scan8x32, scan32x8           []int16
	scan16x64, scan64x16         []int16
	levels                       []uint8
	levels32Zeroed               [1296]uint8
	trialCDFs                    tile.CoeffCDFs
	trialReady                   bool
	// RDOQ state (rdoq.go): the coefficient rate tables of the trial CDFs,
	// the qindex they were built for, the dequantized-coefficient scratch,
	// and 1 + the plane type of the inter TXB being prepared (0 when RDOQ
	// is not armed for it).
	// rdoqShared points at a process-wide immutable default-CDF rate table
	// (rdoq.go), nil when rdoqOwn holds the tables; struct copies keep both
	// meanings intact.
	rdoqShared     *rdoqCosts
	rdoqOwn        rdoqCosts
	rdoqReady      bool
	rdoqQIndex     uint8
	rdoqDQ         [1024]int32
	rdoqArm        int8
	rdoqBlock      bool
	trial8x8CDFs   coeffTrial8x8Snapshot
	trialIntraSnap coeffTrial8x8Snapshot
	// trialBlockLocal keeps the throwaway trial contexts at their frame-start
	// (armed default) state across blocks: every trial-priced decision reads
	// the same frozen tables, the shape SVT/libaom use for realtime mode
	// pricing (md_rate_estimation.c / av1_fill_coeff_costs freeze rate tables
	// at frame start). Inter tiles set it so decisions are independent of
	// coding order — required for the SB-row wavefront to be byte-identical
	// to the serial walk. Keyframe paths leave it false (unchanged behavior).
	trialBlockLocal          bool
	// keyTrialPristine, when set (keyframe tiles), is the frame-start trial
	// context set that restoreKeyTrial puts back the contexts keyTrialDirty
	// marks after each decision unit (keyframe_split.go). Shared read-only by
	// the wavefront lanes.
	keyTrialPristine *tile.CoeffCDFs
	keyTrialDirty    uint16
	intraEdgeScratch         threading.FrameWorkIntraPredictionScratch
	keyMIColEnd, keyMIRowEnd uint32
	keyVisW, keyVisH         int
	invScratch               []int32
	color                    parser.ColorConfig
	transformMode            parser.TransformMode

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
	predY          [4096]byte
	predY16        [4096]uint16
	predY16Bytes   [8192]byte
	refY16Bytes    []byte
	refU16Bytes    []byte
	refV16Bytes    []byte
	goldenY16Bytes []byte
	goldenU16Bytes []byte
	goldenV16Bytes []byte
	predU          [1024]byte
	predV          [1024]byte
	sadScratch     [4096]byte
	// mds0Scratch is MDS0's second candidate buffer (pframe_mds0.go).
	mds0Scratch   [4096]byte
	compBuf0      motion.CompoundConvBuf
	compBuf1      motion.CompoundConvBuf
	compScratch   motion.CompoundConvolveScratch
	scaledScratch motion.ScaledConvolveScratch

	// Transform/quant scratch for the inter TXB pipeline (residual in,
	// forward transform out, dequant + inverse residual back), state-owned so
	// the per-block helpers stay allocation-free at 16x16 sizes.
	resScratch  [4096]int16
	tranScratch [1024]int32
	dqScratch   [4096]int32
	invResidual [4096]int16

	// rdMult prices coefficient rate against transform-domain distortion for
	// the skip decision (av1_compute_rd_mult's inter shape at the working
	// quantizer); rdTXB accumulates the current block's code-vs-skip terms.
	rdMult    int64
	rdDcode   int64
	rdDskip   int64
	rdRcode   int64
	sadPerBit int

	// lpd1Tx holds the frame's light-PD1 TX shortcut controls (SVT-AV1
	// set_lpd1_tx_ctrls; see pframe_lpd1_tx.go). Zero value = disabled.
	lpd1Tx lpd1TxCtrls
	// pfN4 arms the N4_SHAPE partial-frequency forward transform for the
	// following inter TXB (svt_aom_full_loop_chroma_light_pd1).
	pfN4 bool
	// intraLumaDCT marks an intra luma TXB whose after-skip hook is
	// afterSkipIntra (DCT_DCT under st.intraTxTypeReq), so writeIntraTXB may
	// code its tx_type through the specialized writers.
	intraLumaDCT bool
	// txbZero records which of the block's luma, Cb and Cr TXBs quantized to
	// all-zero levels (svtInterCoeffCost then skips their end-of-block scan).
	txbZero [3]bool
	// svtDistShift scales the skip decision's TXB distortions like SVT's
	// light-PD1 full loop (svtTXDist).
	svtDistShift bool

	// cntZeroMV accumulates the MI (4x4) area of inter blocks predicting
	// from LAST with a sub-pixel-magnitude vector, exactly as libaom's
	// update_state() counts cnt_zeromv (av1/encoder/encodeframe_utils.c).
	// Reset per tile, summed across tiles after every inter frame into the
	// avg_frame_low_motion statistic driving adaptive golden refresh.
	cntZeroMV int

	// subpelCache is the lane-owned motion-search scratch (subpel_cache.go).
	subpelCache *subpelProbeCache
	// mePredY says predY holds LAST's refined ME prediction (pframe_motion.go).
	mePredY mePredYKey
	// predStats and predYKey carry the luma prediction statistics of the
	// current inter block between its stages (pred_stats.go).
	predStats lumaPredStats
	predYKey  lumaPredKey

	// lfMap, when non-nil, collects per-MI loop-filter records for the
	// frame-level deblocking pass (the decoder's own map type, filled with
	// the same per-block syntax the decoder records).
	lfMap         *threading.FrameWorkLoopFilterMap
	lfMaskBuilder *threading.FrameWorkLoopFilterOrderedBuilder

	// hme, when non-nil, supplies the coarse-search seeds that recenter the
	// full-pel refinement windows.
	hme *hmeState

	// mds0Level arms the light-PD1 MDS0 inter candidate decision, mirroring
	// SVT-AV1 preset 12's temporal-layer modulation (enc_mode_config.c:
	// pic_lpd1_lvl = is_base ? 4 : 7): 0 disables it, 1 is the base-layer
	// (reference frame) setting, 2 is the droppable-leaf setting that caps
	// the loop to the moderate-distortion band the way SVT's
	// lpd1_detector_post_pd0 steers leaf SBs by block difficulty.
	// mds0Rates freezes the mode and DRL symbol costs from the frame-start
	// CDFs (SVT's md_rate_est_ctx), so the decision cannot feed back into
	// its own pricing mid-frame.
	mds0Level uint8
	mds0Rates tile.InterModeRateTables
	// mds0IntraRates prices the DC_PRED candidate light-PD1 injects next to
	// the inter candidates (pframe_mds0.go mds0IntraWins); mds0IntraOK
	// reports that the frame-start tables were built. mds0BestCost and
	// mds0BestVar carry the inter winner's RD cost and residual variance.
	mds0IntraRates tile.IntraDCRateTables
	// mds0IntraModeRates extends mds0IntraRates to the screen-content intra
	// candidates (pframe_intra_sc.go); mdMEDist is the block's md_me_dist,
	// the smallest post-sub-pel ME cost (^0 when no sub-pel search ran).
	mds0IntraModeRates svtIntraModeRates
	mdMEDist           uint64
	// intraEdgeAbove / intraEdgeLeft hold one block's resolved intra edges
	// and intraPredScratch a candidate prediction; intraUVTxType is the
	// implicit chroma transform type prepareIntraTXB and emitIntraTXB use.
	intraEdgeAbove, intraEdgeLeft [64]uint16
	intraPredScratch              [2][64 * 64]byte
	intraUVTxType                 transform.Type
	// pal is the screen-content keyframe palette search scratch (palette_sc.go),
	// allocated on first use.
	pal *paletteSearch

	// skipRateCDFs prices the inter skip decision's coded alternative
	// (pframe_lpd1_tx.go exactInterCoeffCost), restarting from skipRateBase
	// at every superblock skipRateSB names; skipRateReady arms the base
	// lazily each frame.
	skipRateCDFs  tile.CoeffCDFs
	skipRateBase  tile.CoeffCDFs
	skipRateSB    int
	skipRateDirty uint16
	skipRateReady bool

	// sbQIndex holds the per-64x64 superblock qindex of an inter frame
	// under cyclic refresh (cyclic_refresh.go), frame-wide in raster order
	// with sbQCols columns; nil codes every superblock at the frame q.
	// quantQ is the qindex the quantizers and lambda currently hold;
	// deltaQCur is the delta_q writer's running CurrentQIndex and deltaQCDF
	// the adaptive delta_q_abs CDF.
	// sbQPending/sbQPendingCols hand the next tile encode its grid; the
	// tile consumes them so no other inter path inherits a stale grid.
	sbQPending     []uint8
	sbQPendingCols int
	sbQIndex       []uint8
	sbQCols        int
	quantQ         uint8
	baseQAC        int32
	deltaQCur      uint8
	deltaQCDF      entropy.CDF
	mds0IntraOK    bool
	mds0BestCost   int64
	// mds0BestRate is the MDS0 winner's luma rate (fast_luma_rate, 512 per
	// bit; -1 when MDS0 did not pick the block's mode), the input of
	// the light-PD1 TX bypass.
	mds0BestRate int64
	// mds0Intra carries the DC candidate's MDS0 statistics when it won the
	// block (mds0IntraWins): the inputs of the intra full loop's TX
	// shortcuts (pframe_lpd1_intra.go).
	mds0Intra mds0IntraStats
	// refPads holds the padded copies of the frame's references (refpad.go),
	// nil on paths that read the planes directly.
	refPads *refPads
	// lpd1 carries the picture's light-PD1 level setup and statistics
	// (pframe_lpd1_level.go); lpd1Pending hands the next tile encode its
	// setup the way sbQPending does. lpd1Dec caches the current SB's
	// decision, lpd1TileSBCol0 is the tile's first SB column, and
	// lpd1SkipBits the frame-frozen skip_fac_bits of skip_coeff_ctx 0.
	lpd1           lpd1PictureSetup
	lpd1Pending    lpd1PictureSetup
	lpd1Dec        lpd1SBDecision
	lpd1TileSBCol0 int
	lpd1SkipBits   [2]int64
	// lpd1NbrSkipCtx is the block's skip context: 2 when the left and
	// above blocks are both available and coded skip (get_perform_tx_flag).
	lpd1NbrSkipCtx int
	// mds0BestSSE is the MDS0 winner's regular-filter luma SSE (full_dist)
	// and lpd1IsInterBits its is_inter flag rate (intra_inter_fac_bits),
	// part of fast_luma_rate.
	mds0BestSSE     uint32
	lpd1IsInterBits int64
	mds0BestVar     int64

	// depthRemovalLevel arms SVT-AV1's depth-removal signal chain for inter
	// frames (set_depth_removal_level_controls, enc_mode_config.c; see
	// pframe_depth_removal.go): 0 disables it (standalone tile paths), the
	// video encoder sets the preset-12 ladder level (is_base ? 9 : 14 at
	// 1080p) the same way mds0Level is plumbed. depthRemovalIsRef mirrors
	// me_ctx->is_ref (true when this frame will be referenced).
	depthRemovalLevel uint8
	depthRemovalIsRef bool

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

	// pd0Grid holds the per-SB VERY_LIGHT_PD0 trees (pframe_vlpd0.go),
	// pd0Rates the frame-frozen partition rates they price with and
	// pd0EdgeRef the edge-replicated reference block of off-frame vectors.
	pd0Grid    []svtPD0SB
	pd0Cols    int
	pd0Rates   svtPD0PartitionRates
	pd0EdgeRef [64 * 64]byte

	// refmvsScratch is the tile scratch whose dav1d refmvs rows the current
	// serial walk bound (bindRefmvs), or nil.
	refmvsScratch *tile.BlockLoopScratch
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
	return pc.encodeKeyframeTileWithColorOptions(src, recon, qIndex, miColStart, miColEnd, lfMap, allowScreenContentTools, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true})
}

func (pc *pframeCoder) encodeKeyframeTileWithColorOptions(src SourceFrame420, recon *SourceFrame420, qIndex uint8, miColStart, miColEnd uint16, lfMap *threading.FrameWorkLoopFilterMap, allowScreenContentTools bool, color parser.ColorConfig) ([]byte, error) {
	if err := pc.partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &pc.st
	st.qIndex = qIndex
	st.forceIntegerMV = false
	st.allowScreenContentTools = allowScreenContentTools
	st.color = color
	st.lfMap = lfMap
	st.lfMaskBuilder = nil
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
	st.trialBlockLocal = false
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
	pc.resetWriter()
	pc.armCDEFLog()
	st.w = &pc.writer

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
	// Decisions price against the frame-start trial contexts
	// (keyframe_split.go), so they are independent of coding order.
	pc.keyTrialPristine = st.trialCDFs
	st.keyTrialPristine = &pc.keyTrialPristine
	st.keyTrialDirty = 0
	defer func() { st.keyTrialPristine = nil }()

	// Decision pass, then the serial entropy pass replaying it
	// (keyframe_split.go); the decision pass runs as the SB-row wavefront
	// when the single-tile coder has lanes.
	tileMI4W := int(miColEnd - miColStart)
	rows := (int(miRows) + sbSizeMIB - 1) / sbSizeMIB
	if cap(pc.splitRowRecs) < rows {
		next := make([]pframeSplitRecord, rows)
		copy(next, pc.splitRowRecs)
		pc.splitRowRecs = next
	}
	pc.splitRowRecs = pc.splitRowRecs[:rows]
	for r := range pc.splitRowRecs {
		rowMI4H := min(int(miRows)-r*sbSizeMIB, sbSizeMIB)
		pc.splitRowRecs[r].reset(tileMI4W, rowMI4H, st.color)
		if allowScreenContentTools {
			pc.splitRowRecs[r].reservePalMaps(tileMI4W * 4 * rowMI4H * 4)
		}
	}
	lanes := min(pc.wavefrontLanes, rows)
	singleTile := miColStart == 0 && int(miColEnd)*4 >= src.Width
	pipelined := pc.wavefront != nil && lanes >= 2 && singleTile && st.decisionStats == nil
	if pipelined {
		pc.wavefront.startKey(pc, lanes, src, recon, walkReq, miRows, rows, rootCols)
	} else {
		decideRecord := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
			partition, err := st.keyDecidePartition(src, recon, walkReq, level, miCol, miRow, haveRight, haveBottom)
			if err == nil {
				if st.decisionStats != nil {
					st.decisionStats.notePartition(level, partition)
				}
				pc.splitRowRecs[int(miRow)/sbSizeMIB].appendPart(partition)
			}
			return partition, err
		}
		visitDecide := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
			return st.decideKeyBlock(src, recon, block, scratch, &pc.splitRowRecs[int(block.MIRow)/sbSizeMIB])
		}
		if err := tile.WalkBlockLoopDecide(scratch, carrier, walkReq, sbSizeMIB, decideRecord, visitDecide); err != nil {
			return nil, err
		}
	}
	pc.resetCarrierEdges(rootCols)
	decideReplay := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		row := int(miRow) / sbSizeMIB
		if pipelined {
			if err := pc.wavefront.waitRowDecided(row); err != nil {
				return 0, err
			}
		}
		return pc.splitRowRecs[row].nextPart()
	}
	visitWrite := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		row := int(block.MIRow) / sbSizeMIB
		if pipelined {
			if err := pc.wavefront.waitRowDecided(row); err != nil {
				return err
			}
		}
		return st.writeKeyBlock(block, scratch, &pc.splitRowRecs[row])
	}
	writeErr := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decideReplay, visitWrite)
	if pipelined {
		if writeErr != nil {
			pc.wavefront.abort()
		}
		if err := pc.wavefront.join(pc); err != nil && writeErr == nil {
			writeErr = err
		}
	}
	if writeErr != nil {
		return nil, writeErr
	}
	return pc.writer.Finish()
}

func (pc *pframeCoder) encodeMonochromeKeyframeTile(src SourceFrameMono, recon *SourceFrameMono, qIndex uint8, miColStart, miColEnd uint16) ([]byte, error) {
	return pc.encodeMonochromeKeyframeTileWithOptions(src, recon, qIndex, miColStart, miColEnd, false)
}

func (pc *pframeCoder) encodeMonochromeKeyframeTileWithOptions(src SourceFrameMono, recon *SourceFrameMono, qIndex uint8, miColStart, miColEnd uint16, allowScreenContentTools bool) ([]byte, error) {
	if err := pc.partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &pc.st
	st.qIndex = qIndex
	st.forceIntegerMV = false
	st.allowScreenContentTools = allowScreenContentTools
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
	st.trialBlockLocal = false
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
	pc.resetWriter()
	st.w = &pc.writer

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
	if err := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return pc.writer.Finish()
}

// encodeMonochromeBlock codes one monochrome intra leaf: mode symbols in the
// decoder's order, then the luma residual, reconstructing each transform
// block before the next so later predictions see decoder-identical neighbors.
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
	mode, angleDelta := st.improveIntraModeDirectional(src420, &recon420, block, modeCtx, mode, pred, lumaPX, lumaPY, n)

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
	hasChroma := tile.HasChromaBlockAt(block.Size, int(block.X4), int(block.Y4), st.color)
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
	memsetBytes(pred, dc)
	return st.encodeTXBPred(reconPlane, srcPlane, stride, px, py, n, q, ctxReq, coeffCtx, scan, afterSkip, pred)
}

func (st *lossyEncodeState) encodeTXBAvailRect(reconPlane []byte, srcPlane []byte, stride int, px, py, w, h int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, haveTop, haveLeft bool) error {

	dc := dcPredictRect(reconPlane, stride, px, py, w, h, haveTop, haveLeft)
	pred := st.predU[:w*h]
	memsetBytes(pred, dc)
	return st.encodeTXBPredRect(reconPlane, srcPlane, stride, px, py, w, h, q, ctxReq, coeffCtx, scan, afterSkip, pred)
}

// encodeTXBPred codes one square n x n transform block against an arbitrary
// already-materialized prediction (stride n): forward transform the source
// residual, quantize, write the coefficients, then reconstruct through the
// decoder's own dequant + inverse transform.
func (st *lossyEncodeState) encodeTXBPred(reconPlane []byte, srcPlane []byte, stride int, px, py, n int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, pred []byte) error {
	return st.encodeTXBPredRect(reconPlane, srcPlane, stride, px, py, n, n, q, ctxReq, coeffCtx, scan, afterSkip, pred)
}

// encodeTXBPredRect codes one w x h DCT_DCT transform block against a
// materialized prediction (stride w) in the shape of SVT-AV1's light-PD1 TX
// path (perform_dct_dct_tx_light_pd1 then the inv_transform_recon8bit
// reconstruction): residual kernel, forward transform, quantize, write, then
// the decoder-exact dequant + inverse + add of reconInterTXBResult driven by
// the written end of block. A 64-point extent codes its low 32 coefficients.
func (st *lossyEncodeState) encodeTXBPredRect(reconPlane []byte, srcPlane []byte, stride int, px, py, w, h int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, pred []byte) error {
	geo, _, err := st.prepareIntraTXB(srcPlane, stride, px, py, w, h, q, ctxReq.Size, pred, st.lumaQ[:])
	if err != nil {
		return err
	}
	return st.emitIntraTXB(reconPlane, pred, stride, px, py, w, h, q, ctxReq, coeffCtx, scan, afterSkip, st.lumaQ[:geo.coeffCount], geo)
}

// prepareIntraTXB is the transform/quantize half of encodeTXBPredRect: the
// residual kernel, the trusted forward transform and the quantizer into
// qcoeff. It reports whether every level is zero.
func (st *lossyEncodeState) prepareIntraTXB(srcPlane []byte, stride int, px, py, w, h int, q quantize.Quantizer,
	size tile.TransformSize, pred []byte, qcoeff []int16) (txBlockGeometry, bool, error) {
	geo, err := txBlockGeometryForTransformSize(size)
	if err != nil {
		return geo, false, err
	}
	if !geo.matchesDimensions(w, h) || len(qcoeff) < geo.coeffCount {
		return geo, false, tile.ErrInvalidDecodeState
	}
	n := geo.sampleCount
	cn := geo.coeffCount
	residual := &st.resScratch
	residualBlockImpl(residual[:n], srcPlane, py*stride+px, stride, pred, w, w, h)
	tran := &st.tranScratch
	if err := forwardTransformBlock8BitResidualTrusted(tran[:cn], residual[:n], st.dqScratch[:n], w, h, st.intraUVTxType); err != nil {
		return geo, false, err
	}
	if st.pfN4 {
		// N4_SHAPE (the light-PD1 chroma shortcut): only the low quarter.
		svtPFN4Zero(tran[:cn], w, h, geo.coeffWidth, geo.coeffHeight)
	}
	if err := quantize.QuantizeBlockScaledB(qcoeff[:cn], geo.coeffHeight, tran[:cn], geo.coeffHeight, geo.coeffWidth, geo.coeffHeight, q, geo.txScale); err != nil {
		return geo, false, err
	}
	return geo, levelsAllZero(qcoeff[:cn]), nil
}

// emitIntraTXB is the write/reconstruct half of encodeTXBPredRect: the
// coefficient writer, then the decoder-exact reconstruction driven by the
// written end of block.
func (st *lossyEncodeState) emitIntraTXB(reconPlane, pred []byte, stride int, px, py, w, h int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error,
	qcoeff []int16, geo txBlockGeometry) error {
	res, err := st.writeIntraTXB(w, h, qcoeff, ctxReq, coeffCtx, scan, afterSkip)
	if err != nil {
		return err
	}
	return st.reconInterTXBResult(reconPlane, pred, w, stride, px, py, w, h, q, qcoeff, geo, scan, st.intraUVTxType, res)
}

// writeIntraTXB codes one intra DCT_DCT TXB through the size-specialized
// trusted-context coefficient writers the inter path uses (the per-TXB shape
// of SVT's av1_write_coeffs_txb_1d), falling back to the generic writer for
// the other shapes. Luma carries its intra tx_type symbol (afterSkip set).
func (st *lossyEncodeState) writeIntraTXB(w, h int, qcoeff []int16, ctxReq tile.CoeffContextRequest,
	coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error) (tile.TXBDecodeResult, error) {
	var res tile.TXBDecodeResult
	blockDims, ok := ctxReq.PlaneBlock.Dimensions()
	if !ok {
		return res, tile.ErrInvalidDecodeState
	}
	written := false
	if ctxReq.Plane == 0 && afterSkip != nil && st.intraLumaDCT && w == h && (w == 8 || w == 16 || w == 32) {
		if txCDF, txSymbol, ok := st.intraTXCDFAndSymbol(st.intraTxTypeReq); ok {
			written = true
			switch w {
			case 8:
				c := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 2, H4: 2}, blockDims)
				res = tile.WriteCoefficientsTXB8x8Y2DContextTrustedArray(st.w, &st.coeffCDFs, (*[64]int16)(qcoeff), c.TXBSkipContext, c.DCSignContext, txCDF, txSymbol)
			case 16:
				c := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 4, H4: 4}, blockDims)
				res = tile.WriteCoefficientsTXB16x16Y2DContextTrustedArray(st.w, &st.coeffCDFs, (*[256]int16)(qcoeff), c.TXBSkipContext, c.DCSignContext, txCDF, txSymbol)
			default:
				c := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 8, H4: 8}, blockDims)
				res = tile.WriteCoefficientsTXB32x32Y2DContextTrustedZeroedLevels(st.w, &st.coeffCDFs, (*[1024]int16)(qcoeff), st.levels32Zeroed[:], c.TXBSkipContext, c.DCSignContext, txCDF, txSymbol)
			}
		}
	} else if ctxReq.Plane != 0 && afterSkip == nil && w == h && (w == 4 || w == 8 || w == 16 || w == 32) {
		written = true
		switch w {
		case 4:
			c := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 1, H4: 1}, blockDims)
			res = tile.WriteCoefficientsTXB4x4UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[16]int16)(qcoeff), c.TXBSkipContext, c.DCSignContext)
		case 8:
			c := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 2, H4: 2}, blockDims)
			res = tile.WriteCoefficientsTXB8x8UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[64]int16)(qcoeff), c.TXBSkipContext, c.DCSignContext)
		case 16:
			c := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 4, H4: 4}, blockDims)
			res = tile.WriteCoefficientsTXB16x16UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[256]int16)(qcoeff), c.TXBSkipContext, c.DCSignContext)
		default:
			c := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 8, H4: 8}, blockDims)
			res = tile.WriteCoefficientsTXB32x32UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[1024]int16)(qcoeff), c.TXBSkipContext, c.DCSignContext)
		}
	}
	if !written {
		return tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip)
	}
	if err := coeffCtx.MarkTXB(ctxReq, res); err != nil {
		return res, err
	}
	return res, nil
}

// intraTXCDFAndSymbol is WriteIntraTransformType's symbol selection for a
// DCT_DCT intra TXB: the CDF and symbol to code, or a nil CDF when the set
// signals nothing. ok=false leaves the caller on the generic writer.
func (st *lossyEncodeState) intraTXCDFAndSymbol(req tile.IntraTransformTypeRequest) (*entropy.CDF, int, bool) {
	if req.SkipTransform || req.Lossless || (req.QIndexKnown && req.QIndex == 0) {
		return nil, 0, true
	}
	set, err := tile.ExtTXSetTypeFor(req.Size, false, req.ReducedTXSet)
	if err != nil {
		return nil, 0, false
	}
	symbols, err := tile.ExtTXTypeCount(set)
	if err != nil {
		return nil, 0, false
	}
	if symbols <= 1 {
		return nil, 0, true
	}
	index, err := tile.ExtTXSetIndex(req.Size, false, req.ReducedTXSet)
	if err != nil {
		return nil, 0, false
	}
	square, err := tile.TransformSizeSquare(req.Size)
	if err != nil {
		return nil, 0, false
	}
	cdf, err := st.txCDFs.IntraCDF(index, square, req.Mode, symbols)
	if err != nil {
		return nil, 0, false
	}
	symbol, err := tile.ExtTXSymbolForType(set, transform.TypeDCTDCT)
	if err != nil {
		return nil, 0, false
	}
	return cdf, symbol, true
}

// dcPredictN is the decoder's DC predictor for one n x n block at pixel
// (px,py) of plane: the rounded mean of the n above and n left reconstructed
// neighbors that are available, or 128 when neither edge is. Availability is
// the caller's tile-relative HaveTop/HaveLeft, not raw frame position.
func dcPredictN(plane []byte, stride, px, py, n int, haveTop, haveLeft bool) uint8 {
	return dcPredictRect(plane, stride, px, py, n, n, haveTop, haveLeft)
}

func dcPredictRect(plane []byte, stride, px, py, w, h int, haveTop, haveLeft bool) uint8 {
	sum := 0
	count := 0
	if haveTop && py > 0 {
		row := (py-1)*stride + px
		for i := range w {
			sum += int(plane[row+i])
		}
		count += w
	}
	if haveLeft && px > 0 {
		col := py*stride + px - 1
		for i := range h {
			sum += int(plane[col+i*stride])
		}
		count += h
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
	if haveTop {
		sadV = 0
	}
	if haveLeft {
		sadH = 0
	}
	above := (py-1)*stride + px
	for r := range n {
		row := (py+r)*stride + px
		left := 0
		if haveLeft {
			left = int(reconPlane[row-1])
		}
		for c := range n {
			src := int(srcPlane[row+c])
			d := src - int(dc)
			if d < 0 {
				d = -d
			}
			sadDC += d
			if haveTop {
				d = src - int(reconPlane[above+c])
				if d < 0 {
					d = -d
				}
				sadV += d
			}
			if haveLeft {
				d = src - left
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
func (st *lossyEncodeState) improveIntraModeDirectional(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, modeCtx *tile.BlockModeContext, mode tile.IntraMode, pred []byte, px, py, n int) (tile.IntraMode, int) {
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
	if st.trialBlockLocal {
		// Restore the trial contexts after this block's exact-rate pricing so
		// every block prices against the frame-start tables (see
		// trialBlockLocal). The within-block adaptation across the 1-3 trials
		// below is deterministic and kept.
		st.trialIntraSnap.save(&st.trialCDFs)
		defer st.trialIntraSnap.restore(&st.trialCDFs)
	}
	// The intra edge filter strength reads whether a neighbour uses a smooth
	// mode (get_filt_type), before this block marks its own mode.
	smoothNeighbor, err := modeCtx.IntraEdgeSmoothNeighbor(int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft)
	if err != nil {
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
		edges, err := threading.BuildLumaDirectionalEdges(reconPlane, 8, px, py, n, n, angle, block, &st.intraEdgeScratch, 16, st.keyMIColEnd, st.keyMIRowEnd, encoderIntraEdgeFilter, smoothNeighbor, st.keyVisW, st.keyVisH)
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
	st.noteKeyTrial(plane, n)
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
	cn := n >> chromaShiftX(st.color)
	ch := n >> chromaShiftY(st.color)
	if cn != ch {
		return 0
	}
	cx, cy := chromaXForColor(px, st.color), chromaYForColor(py, st.color)
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

// levelsAllZero reports whether every level is zero, four at a time (every
// coded area is a multiple of 16 levels).
func levelsAllZero(q []int16) bool {
	w := unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(q))), len(q)/4)
	var acc uint64
	for _, v := range w {
		acc |= v
	}
	for _, v := range q[len(w)*4:] {
		acc |= uint64(uint16(v))
	}
	return acc == 0
}
