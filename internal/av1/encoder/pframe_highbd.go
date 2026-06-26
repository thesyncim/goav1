package encoder

import (
	"fmt"

	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// EncodeHighBitDepthMonochromePFrame encodes src as a native 10/12-bit
// monochrome inter frame predicting from ref, returning the temporal unit and
// the encoder reconstruction the decoder must reproduce exactly.
func EncodeHighBitDepthMonochromePFrame(src SourceFrameMono16, ref SourceFrameMono16, qIndex uint8) ([]byte, SourceFrameMono16, error) {
	if err := validateSourceFrameMono16(src); err != nil {
		return nil, SourceFrameMono16{}, err
	}
	if err := validateSourceFrameMono16(ref); err != nil {
		return nil, SourceFrameMono16{}, err
	}
	if src.Width != ref.Width || src.Height != ref.Height {
		return nil, SourceFrameMono16{}, fmt.Errorf("encoder: source %dx%d does not match reference %dx%d", src.Width, src.Height, ref.Width, ref.Height)
	}
	if src.BitDepth != ref.BitDepth {
		return nil, SourceFrameMono16{}, fmt.Errorf("encoder: source bit depth %d does not match reference bit depth %d", src.BitDepth, ref.BitDepth)
	}
	if qIndex == 0 {
		return nil, SourceFrameMono16{}, fmt.Errorf("encoder: qindex 0 lossless inter coding is not supported")
	}
	recon := SourceFrameMono16{
		Y:        make([]uint16, len(src.Y)),
		YStride:  src.YStride,
		Width:    src.Width,
		Height:   src.Height,
		BitDepth: src.BitDepth,
	}
	seq := lossyHighBitDepthMonochromeKeyframeSequence(src.Width, src.Height, src.BitDepth)
	header, refState := repeatPFrameHeader(src.Width, src.Height, qIndex, 0x01)
	header.References = &refState
	header.CDEF = CDEFParams{}
	var pc pframeCoder
	tilePayload, err := pc.encodeHighBitDepthMonochromePFrameTile(src, ref, &recon, qIndex, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, 0, uint16(src.Width/4))
	if err != nil {
		return nil, SourceFrameMono16{}, fmt.Errorf("encode tile: %w", err)
	}

	var groupScratch, outScratch []byte
	out, err := assembleInterTU(seq, header, []TilePayload{{Data: tilePayload}}, 0, &groupScratch, &outScratch)
	if err != nil {
		return nil, SourceFrameMono16{}, err
	}
	return out, recon, nil
}

// EncodeHighBitDepth420PFrame encodes src as a native 10/12-bit 4:2:0 AV1
// inter frame predicting from ref, returning the temporal unit and the
// encoder reconstruction a conformant decoder must reproduce exactly.
func EncodeHighBitDepth420PFrame(src SourceFrame42016, ref SourceFrame42016, qIndex uint8) ([]byte, SourceFrame42016, error) {
	if err := validateSourceFrame42016(src); err != nil {
		return nil, SourceFrame42016{}, err
	}
	if err := validateSourceFrame42016(ref); err != nil {
		return nil, SourceFrame42016{}, err
	}
	if src.Width != ref.Width || src.Height != ref.Height {
		return nil, SourceFrame42016{}, fmt.Errorf("encoder: source %dx%d does not match reference %dx%d", src.Width, src.Height, ref.Width, ref.Height)
	}
	if src.BitDepth != ref.BitDepth {
		return nil, SourceFrame42016{}, fmt.Errorf("encoder: source bit depth %d does not match reference bit depth %d", src.BitDepth, ref.BitDepth)
	}
	if qIndex == 0 {
		return nil, SourceFrame42016{}, fmt.Errorf("encoder: qindex 0 lossless inter coding is not supported")
	}
	recon := SourceFrame42016{
		Y:            make([]uint16, len(src.Y)),
		U:            make([]uint16, len(src.U)),
		V:            make([]uint16, len(src.V)),
		YStride:      src.YStride,
		ChromaStride: src.ChromaStride,
		Width:        src.Width,
		Height:       src.Height,
		BitDepth:     src.BitDepth,
	}
	seq := lossyHighBitDepth420KeyframeSequence(src.Width, src.Height, src.BitDepth)
	header, refState := repeatPFrameHeader(src.Width, src.Height, qIndex, 0x01)
	header.References = &refState
	header.CDEF = CDEFParams{}
	var pc pframeCoder
	tilePayload, err := pc.encodeHighBitDepth420PFrameTile(src, ref, &recon, qIndex, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, 0, uint16(src.Width/4))
	if err != nil {
		return nil, SourceFrame42016{}, fmt.Errorf("encode tile: %w", err)
	}

	var groupScratch, outScratch []byte
	out, err := assembleInterTU(seq, header, []TilePayload{{Data: tilePayload}}, 0, &groupScratch, &outScratch)
	if err != nil {
		return nil, SourceFrame42016{}, err
	}
	return out, recon, nil
}

func (pc *pframeCoder) encodeHighBitDepthMonochromePFrameTile(src SourceFrameMono16, ref SourceFrameMono16, recon *SourceFrameMono16, qIndex uint8, forceIntegerMV bool, allowScreenContentTools bool, miColStart, miColEnd uint16) ([]byte, error) {
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB
	color := parser.ColorConfig{BitDepth: src.BitDepth, MonoChrome: true, SubsamplingX: true, SubsamplingY: true}
	if err := pc.reset(qIndex, rootCols, nil, color); err != nil {
		return nil, err
	}
	st := &pc.st
	st.forceIntegerMV = forceIntegerMV
	st.allowScreenContentTools = allowScreenContentTools
	st.lfMap = nil
	st.hme = nil
	st.decisionStats = nil

	pc.writer.Reset(pc.writerBuf[:0])
	st.w = &pc.writer
	st.interTxTypeReq = tile.InterTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		QIndexKnown: true,
		QIndex:      qIndex,
	}
	st.interTxType = transform.TypeDCTDCT
	if st.afterSkipInter == nil {
		st.afterSkipInter = func() error {
			return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, st.interTxType)
		}
		st.afterSkipIntra = func() error {
			return tile.WriteIntraTransformType(st.w, &st.txCDFs, st.intraTxTypeReq, transform.TypeDCTDCT)
		}
	}

	refPlane, err := st.highBitDepthMonoReferencePlane(ref)
	if err != nil {
		return nil, err
	}

	scratch := &pc.scratch
	carrier := &pc.carrier
	walkReq := tile.BlockWalkRequest{
		Root:       tile.BlockLevel64x64,
		MIColStart: miColStart,
		MIColEnd:   miColEnd,
		MIRowEnd:   miRows,
	}
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodeHighBitDepthMonochromePBlock(src, refPlane, recon, block, scratch, &pc.refCDFs, &pc.modeCDFs, walkReq, uint16(src.Width/4), miRows)
	}
	if err := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return pc.writer.Finish()
}

func (pc *pframeCoder) encodeHighBitDepth420PFrameTile(src SourceFrame42016, ref SourceFrame42016, recon *SourceFrame42016, qIndex uint8, forceIntegerMV bool, allowScreenContentTools bool, miColStart, miColEnd uint16) ([]byte, error) {
	return pc.encodeHighBitDepthColorPFrameTile(src, ref, recon, qIndex, forceIntegerMV, allowScreenContentTools, parser.ColorConfig{BitDepth: src.BitDepth, SubsamplingX: true, SubsamplingY: true}, miColStart, miColEnd)
}

func (pc *pframeCoder) encodeHighBitDepthColorPFrameTile(src SourceFrame42016, ref SourceFrame42016, recon *SourceFrame42016, qIndex uint8, forceIntegerMV bool, allowScreenContentTools bool, color parser.ColorConfig, miColStart, miColEnd uint16) ([]byte, error) {
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB
	if err := pc.reset(qIndex, rootCols, nil, color); err != nil {
		return nil, err
	}
	st := &pc.st
	st.forceIntegerMV = forceIntegerMV
	st.allowScreenContentTools = allowScreenContentTools
	st.lfMap = nil
	st.hme = nil
	st.decisionStats = nil

	pc.writer.Reset(pc.writerBuf[:0])
	st.w = &pc.writer
	st.interTxTypeReq = tile.InterTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		QIndexKnown: true,
		QIndex:      qIndex,
	}
	st.interTxType = transform.TypeDCTDCT
	if st.afterSkipInter == nil {
		st.afterSkipInter = func() error {
			return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, st.interTxType)
		}
		st.afterSkipIntra = func() error {
			return tile.WriteIntraTransformType(st.w, &st.txCDFs, st.intraTxTypeReq, transform.TypeDCTDCT)
		}
	}

	refPlanes, err := st.highBitDepthColorReferencePlanes(ref, color)
	if err != nil {
		return nil, err
	}

	scratch := &pc.scratch
	carrier := &pc.carrier
	walkReq := tile.BlockWalkRequest{
		Root:       tile.BlockLevel64x64,
		MIColStart: miColStart,
		MIColEnd:   miColEnd,
		MIRowEnd:   miRows,
	}
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodeHighBitDepth420PBlock(src, refPlanes, recon, block, scratch, &pc.refCDFs, &pc.modeCDFs, walkReq, uint16(src.Width/4), miRows)
	}
	if err := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return pc.writer.Finish()
}

func (st *lossyEncodeState) encodeHighBitDepthMonochromePBlock(src SourceFrameMono16, ref av1frame.Plane, recon *SourceFrameMono16, block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	refCDFs *tile.InterRefCDFs, interModeCDFs *tile.InterModeCDFs, walkReq tile.BlockWalkRequest, miCols, miRows uint16) error {
	if block.Size != tile.BlockSize8x8 {
		return fmt.Errorf("encoder: unexpected high-bit-depth monochrome P block %+v", block)
	}
	const n = 8
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4

	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
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
	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
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
	if err := tile.WriteBlockInterMode(st.w, interModeCDFs, tile.InterModeRequest{
		Compound:    false,
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
	motionResult := tile.InterMotionResult{References: refs, Mode: modeResult}
	if err := tile.WriteInterMotion(st.w, &st.mvCDFs, tile.InterMotionRequest{
		References: refs,
		Mode:       modeResult,
		Precision:  tile.MVPrecision(false, st.forceIntegerMV),
	}, motionResult); err != nil {
		return fmt.Errorf("motion vector: %w", err)
	}
	if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), motionResult, false); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), refs, motion.InterpFilters{}); err != nil {
		return fmt.Errorf("mark inter filters: %w", err)
	}

	lfTree, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: parser.TransformModeSwitchable,
		Inter: true,
	}, tile.TransformTreeResult{})
	if err != nil {
		return fmt.Errorf("transform tree: %w", err)
	}
	if st.lfMap != nil {
		if err := markLoopFilterBlock(st.lfMap, block, lfTree, false, false, uint8(refs.Ref[0])+1, loopfilter.ModeDeltaClassZero); err != nil {
			return fmt.Errorf("mark loop filter: %w", err)
		}
	}

	pred := st.predY16[:n*n]
	if err := predictIntoScaledHighBitDepthMono16(pred, st.predY16Bytes[:n*n*2], ref, src.BitDepth, src.Width, src.Height, lumaPX, lumaPY, n, n, motion.Vector{}, &st.scaledScratch); err != nil {
		return fmt.Errorf("high-bit-depth monochrome prediction: %w", err)
	}
	st.interTxTypeReq.Size = tile.TransformSize8x8
	st.interTxType = transform.TypeDCTDCT
	if err := st.encodeTXBPred16(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, n, src.BitDepth, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, st.scan8, st.afterSkipInter, pred); err != nil {
		return fmt.Errorf("luma txb: %w", err)
	}
	if st.decisionStats != nil {
		st.decisionStats.noteInterBlock(block.Size, false, false, refs, modeResult, transform.TypeDCTDCT)
	}
	return nil
}

func (st *lossyEncodeState) encodeHighBitDepth420PBlock(src SourceFrame42016, refs [3]av1frame.Plane, recon *SourceFrame42016, block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	refCDFs *tile.InterRefCDFs, interModeCDFs *tile.InterModeCDFs, walkReq tile.BlockWalkRequest, miCols, miRows uint16) error {
	if block.Size != tile.BlockSize8x8 {
		return fmt.Errorf("encoder: unexpected high-bit-depth color P block %+v", block)
	}
	const n = 8
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4

	interRefs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
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
		References:     interRefs,
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
	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
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
	}, interRefs); err != nil {
		return fmt.Errorf("references: %w", err)
	}
	if err := tile.WriteBlockInterMode(st.w, interModeCDFs, tile.InterModeRequest{
		Compound:    false,
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
	motionResult := tile.InterMotionResult{References: interRefs, Mode: modeResult}
	if err := tile.WriteInterMotion(st.w, &st.mvCDFs, tile.InterMotionRequest{
		References: interRefs,
		Mode:       modeResult,
		Precision:  tile.MVPrecision(false, st.forceIntegerMV),
	}, motionResult); err != nil {
		return fmt.Errorf("motion vector: %w", err)
	}
	if err := modeCtx.MarkInterMotion(block.Size, int(block.X4), int(block.Y4), motionResult, true); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	if err := modeCtx.MarkInterFilters(block.Size, int(block.X4), int(block.Y4), interRefs, motion.InterpFilters{}); err != nil {
		return fmt.Errorf("mark inter filters: %w", err)
	}

	lfTree, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: parser.TransformModeSwitchable,
		Inter: true,
	}, tile.TransformTreeResult{})
	if err != nil {
		return fmt.Errorf("transform tree: %w", err)
	}
	if st.lfMap != nil {
		if err := markLoopFilterBlock(st.lfMap, block, lfTree, false, false, uint8(interRefs.Ref[0])+1, loopfilter.ModeDeltaClassZero); err != nil {
			return fmt.Errorf("mark loop filter: %w", err)
		}
	}

	pred := st.predY16[:n*n]
	if err := predictIntoScaledHighBitDepthMono16(pred, st.predY16Bytes[:n*n*2], refs[0], src.BitDepth, src.Width, src.Height, lumaPX, lumaPY, n, n, motion.Vector{}, &st.scaledScratch); err != nil {
		return fmt.Errorf("high-bit-depth color luma prediction: %w", err)
	}
	st.interTxTypeReq.Size = tile.TransformSize8x8
	st.interTxType = transform.TypeDCTDCT
	if err := st.encodeTXBPred16(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, n, src.BitDepth, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, st.scan8, st.afterSkipInter, pred); err != nil {
		return fmt.Errorf("luma txb: %w", err)
	}

	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma plane block: %w", err)
	}
	cw, ch, err := planeBlockPixels(chromaBlock)
	if err != nil {
		return fmt.Errorf("chroma block dimensions: %w", err)
	}
	chromaTX, err := tile.MaxTransformSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma transform size: %w", err)
	}
	chromaScan, ok := st.scanForTransformSize(chromaTX)
	if !ok {
		return fmt.Errorf("encoder: unsupported chroma transform %d", chromaTX)
	}
	chromaX := chromaXForColor(lumaPX, st.color)
	chromaY := chromaYForColor(lumaPY, st.color)
	chromaX4 := chromaX4ForColor(block.X4, st.color)
	chromaY4 := chromaY4ForColor(block.Y4, st.color)
	chromaWidth, chromaHeight := chromaWidthForColor(src.Width, st.color), chromaHeightForColor(src.Height, st.color)
	for plane := 1; plane <= 2; plane++ {
		data, rdata := src.U, recon.U
		q := st.uQuant
		if plane == 2 {
			data, rdata = src.V, recon.V
			q = st.vQuant
		}
		chromaPred := st.predY16[:cw*ch]
		if err := predictIntoScaledHighBitDepthMono16(chromaPred, st.predY16Bytes[:cw*ch*2], refs[plane], src.BitDepth, chromaWidth, chromaHeight, chromaX, chromaY, cw, ch, motion.Vector{}, &st.scaledScratch); err != nil {
			return fmt.Errorf("high-bit-depth color chroma %d prediction: %w", plane, err)
		}
		if err := st.encodeTXBPredRect16(rdata, data, src.ChromaStride, chromaX, chromaY, cw, ch, src.BitDepth, q, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       chromaTX,
			X4:         chromaX4,
			Y4:         chromaY4,
		}, coeffCtx, chromaScan, nil, chromaPred); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
	}
	if st.decisionStats != nil {
		st.decisionStats.noteInterBlock(block.Size, false, false, interRefs, modeResult, transform.TypeDCTDCT)
	}
	return nil
}

func (st *lossyEncodeState) highBitDepthMonoReferencePlane(ref SourceFrameMono16) (av1frame.Plane, error) {
	if err := validateSourceFrameMono16(ref); err != nil {
		return av1frame.Plane{}, err
	}
	return highBitDepthReferencePlane(ref.Y, ref.YStride, ref.Width, ref.Height, &st.refY16Bytes)
}

func (st *lossyEncodeState) highBitDepth420ReferencePlanes(ref SourceFrame42016) ([3]av1frame.Plane, error) {
	return st.highBitDepthColorReferencePlanes(ref, parser.ColorConfig{BitDepth: ref.BitDepth, SubsamplingX: true, SubsamplingY: true})
}

func (st *lossyEncodeState) highBitDepthColorReferencePlanes(ref SourceFrame42016, color parser.ColorConfig) ([3]av1frame.Plane, error) {
	if err := validateSourceFrameColor16(ref, color, videoColorLabel(color)); err != nil {
		return [3]av1frame.Plane{}, err
	}
	chromaWidth, chromaHeight := chromaWidthForColor(ref.Width, color), chromaHeightForColor(ref.Height, color)
	y, err := highBitDepthReferencePlane(ref.Y, ref.YStride, ref.Width, ref.Height, &st.refY16Bytes)
	if err != nil {
		return [3]av1frame.Plane{}, err
	}
	u, err := highBitDepthReferencePlane(ref.U, ref.ChromaStride, chromaWidth, chromaHeight, &st.refU16Bytes)
	if err != nil {
		return [3]av1frame.Plane{}, err
	}
	v, err := highBitDepthReferencePlane(ref.V, ref.ChromaStride, chromaWidth, chromaHeight, &st.refV16Bytes)
	if err != nil {
		return [3]av1frame.Plane{}, err
	}
	return [3]av1frame.Plane{y, u, v}, nil
}

func highBitDepthReferencePlane(samples []uint16, stride, width, height int, scratch *[]byte) (av1frame.Plane, error) {
	strideBytes := stride * 2
	need := (height-1)*strideBytes + width*2
	if cap(*scratch) < need {
		*scratch = make([]byte, need)
	}
	buf := (*scratch)[:need]
	for y := 0; y < height; y++ {
		srcRow := samples[y*stride : y*stride+width]
		dstRow := buf[y*strideBytes : y*strideBytes+width*2]
		for x, sample := range srcRow {
			dstRow[2*x] = byte(sample)
			dstRow[2*x+1] = byte(sample >> 8)
		}
	}
	return av1frame.Plane{
		Pix:    buf,
		Stride: strideBytes,
		Width:  width,
		Height: height,
	}, nil
}

func predictIntoScaledHighBitDepthMono16(dst []uint16, dstBytes []byte, ref av1frame.Plane, bitDepth uint8, curWidth, curHeight, px, py, bw, bh int, mv motion.Vector, scratch *motion.ScaledConvolveScratch) error {
	if len(dst) < bw*bh || len(dstBytes) < bw*bh*2 {
		return ErrInvalidFrame
	}
	dstPlane := av1frame.Plane{
		Pix:    dstBytes[:bw*bh*2],
		Stride: bw * 2,
		Width:  bw,
		Height: bh,
	}
	filters := motion.RegularFilters
	if ref.Width == curWidth && ref.Height == curHeight {
		refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, false, false)
		if err != nil {
			return err
		}
		if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstPlane, ref, 2, bitDepth, 0, 0, refX, refY, bw, bh, subX, subY, filters); err != nil {
			return err
		}
	} else {
		sf, err := motion.NewScaleFactors(ref.Width, ref.Height, curWidth, curHeight)
		if err != nil {
			return err
		}
		startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(px, py, mv, false, false)
		if err != nil {
			return err
		}
		xTable, err := motion.SubpelKernelTableFor(filters.X, bw)
		if err != nil {
			return err
		}
		yTable, err := motion.SubpelKernelTableFor(filters.Y, bh)
		if err != nil {
			return err
		}
		if err := motion.ConvolveScale2DHighBDClampedWithScratch(dstPlane, ref, bitDepth, 0, 0, bw, bh, startX, xStep, startY, yStep, xTable, yTable, scratch); err != nil {
			return err
		}
	}
	for i := 0; i < bw*bh; i++ {
		dst[i] = uint16(dstBytes[2*i]) | uint16(dstBytes[2*i+1])<<8
	}
	return nil
}
