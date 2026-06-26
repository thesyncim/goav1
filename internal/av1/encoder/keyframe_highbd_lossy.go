package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// EncodeHighBitDepthMonochromeKeyframe encodes src at qIndex 1..255 as one
// native 10/12-bit AV1 monochrome non-lossless keyframe and returns the
// encoder-side reconstruction a conformant decoder must reproduce exactly.
func EncodeHighBitDepthMonochromeKeyframe(src SourceFrameMono16, qIndex uint8) ([]byte, SourceFrameMono16, error) {
	if err := validateSourceFrameMono16(src); err != nil {
		return nil, SourceFrameMono16{}, err
	}
	if qIndex == 0 {
		return nil, SourceFrameMono16{}, fmt.Errorf("encoder: qindex 0 is the lossless path; use EncodeLosslessHighBitDepthMonochromeKeyframe")
	}
	recon := SourceFrameMono16{
		Y:        make([]uint16, len(src.Y)),
		YStride:  src.YStride,
		Width:    src.Width,
		Height:   src.Height,
		BitDepth: src.BitDepth,
	}

	var pc pframeCoder
	tilePayload, err := pc.encodeHighBitDepthMonochromeKeyframeTile(src, &recon, qIndex, 0, uint16(src.Width/4))
	if err != nil {
		return nil, SourceFrameMono16{}, fmt.Errorf("encode tile: %w", err)
	}

	seq := lossyHighBitDepthMonochromeKeyframeSequence(src.Width, src.Height, src.BitDepth)
	header := lossyMonochromeKeyframeHeaderForSequence(seq, src.Width, src.Height, qIndex)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, SourceFrameMono16{}, fmt.Errorf("size header TU: %w", err)
	}

	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, SourceFrameMono16{}, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, SourceFrameMono16{}, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, SourceFrameMono16{}, err
	}

	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, SourceFrameMono16{}, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, SourceFrameMono16{}, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, recon, nil
}

func lossyHighBitDepthMonochromeKeyframeSequence(width, height int, bitDepth uint8) SequenceHeader {
	seq := losslessHighBitDepthMonochromeKeyframeSequence(width, height, bitDepth)
	seq.EnableCDEF = false
	return seq
}

func (pc *pframeCoder) encodeHighBitDepthMonochromeKeyframeTile(src SourceFrameMono16, recon *SourceFrameMono16, qIndex uint8, miColStart, miColEnd uint16) ([]byte, error) {
	return pc.encodeHighBitDepthMonochromeKeyframeTileWithOptions(src, recon, qIndex, miColStart, miColEnd, false)
}

func (pc *pframeCoder) encodeHighBitDepthMonochromeKeyframeTileWithOptions(src SourceFrameMono16, recon *SourceFrameMono16, qIndex uint8, miColStart, miColEnd uint16, allowScreenContentTools bool) ([]byte, error) {
	if err := pc.partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &pc.st
	st.qIndex = qIndex
	st.forceIntegerMV = false
	st.allowScreenContentTools = allowScreenContentTools
	st.color = parser.ColorConfig{BitDepth: src.BitDepth, MonoChrome: true, SubsamplingX: true, SubsamplingY: true}
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
	q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, src.BitDepth, quantize.PlaneY)
	if err != nil {
		return nil, err
	}
	st.yQuant = q
	if err := st.initScans(); err != nil {
		return nil, err
	}

	if cap(pc.writerBuf) == 0 {
		pc.writerBuf = make([]byte, 1<<18)
	}
	pc.writer.Reset(pc.writerBuf[:0])
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
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodeHighBitDepthMonochromeBlock(src, recon, block, scratch)
	}
	if err := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return pc.writer.Finish()
}

func (st *lossyEncodeState) encodeHighBitDepthMonochromeBlock(src SourceFrameMono16, recon *SourceFrameMono16, block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
	if block.Size != tile.BlockSize8x8 {
		return fmt.Errorf("encoder: unexpected high-bit-depth monochrome block %+v", block)
	}
	const n = 8
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
	mode := tile.IntraModeDC
	pred := st.predY16[:n*n]
	dc := dcPredictN16(recon.Y, recon.YStride, lumaPX, lumaPY, n, block.HaveTop, block.HaveLeft, src.BitDepth)
	for i := range pred {
		pred[i] = dc
	}

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
	if err := st.writeNoPaletteMode(modeCtx, block, mode, tile.ChromaIntraModeDC, false); err != nil {
		return fmt.Errorf("palette: %w", err)
	}

	txTypeReq := tile.IntraTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		Mode:        mode,
		QIndexKnown: true,
		QIndex:      st.qIndex,
	}
	afterSkip := func() error {
		return tile.WriteIntraTransformType(st.w, &st.txCDFs, txTypeReq, transform.TypeDCTDCT)
	}
	if err := st.encodeTXBPred16(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, n, src.BitDepth, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, st.scan8, afterSkip, pred); err != nil {
		return fmt.Errorf("luma txb: %w", err)
	}
	if st.decisionStats != nil {
		st.decisionStats.noteIntraBlock(block.Size)
	}
	return nil
}

func (st *lossyEncodeState) encodeTXBPred16(reconPlane []uint16, srcPlane []uint16, stride int, px, py, n int, bitDepth uint8, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, pred []uint16) error {

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

	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n*n], n, qcoeff[:n*n], n, n, n, q, txScaleForSize(n), bitDepth); err != nil {
		return err
	}
	res := &st.invResidual
	size := transform.Size{Width: uint8(n), Height: uint8(n)}
	if err := transform.InverseBlockBitDepth(res[:n*n], n, dq[:n*n], n, st.invScratch[:n*n], size, transform.TypeDCTDCT, bitDepth); err != nil {
		return err
	}
	maxSample := int((1 << bitDepth) - 1)
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			v := int(pred[r*n+c]) + int(res[r*n+c])
			if v < 0 {
				v = 0
			} else if v > maxSample {
				v = maxSample
			}
			reconPlane[row+c] = uint16(v)
		}
	}
	return nil
}
