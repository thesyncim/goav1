package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// EncodeHighBitDepth420Keyframe encodes src at qIndex 1..255 as one native
// 10/12-bit 4:2:0 AV1 keyframe and returns the encoder-side reconstruction a
// conformant decoder must reproduce exactly.
func EncodeHighBitDepth420Keyframe(src SourceFrame42016, qIndex uint8) ([]byte, SourceFrame42016, error) {
	if err := validateSourceFrame42016(src); err != nil {
		return nil, SourceFrame42016{}, err
	}
	if qIndex == 0 {
		return nil, SourceFrame42016{}, fmt.Errorf("encoder: qindex 0 is the lossless path; high-bit-depth 4:2:0 lossless keyframes are not implemented")
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

	var pc pframeCoder
	tilePayload, err := pc.encodeHighBitDepth420KeyframeTile(src, &recon, qIndex, 0, uint16(src.Width/4))
	if err != nil {
		return nil, SourceFrame42016{}, fmt.Errorf("encode tile: %w", err)
	}

	seq := lossyHighBitDepth420KeyframeSequence(src.Width, src.Height, src.BitDepth)
	header := lossyKeyframeHeaderForSequence(seq, src.Width, src.Height, qIndex)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, SourceFrame42016{}, fmt.Errorf("size header TU: %w", err)
	}
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, SourceFrame42016{}, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, SourceFrame42016{}, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, SourceFrame42016{}, err
	}

	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, SourceFrame42016{}, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, SourceFrame42016{}, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, recon, nil
}

func lossyHighBitDepth420KeyframeSequence(width, height int, bitDepth uint8) SequenceHeader {
	seq := losslessKeyframeSequence(width, height)
	seq.ColorConfig.BitDepth = bitDepth
	seq.EnableCDEF = false
	if bitDepth == 12 {
		seq.Profile = Profile2
	}
	return seq
}

func (pc *pframeCoder) encodeHighBitDepth420KeyframeTile(src SourceFrame42016, recon *SourceFrame42016, qIndex uint8, miColStart, miColEnd uint16) ([]byte, error) {
	if err := pc.partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	st := &pc.st
	st.qIndex = qIndex
	st.forceIntegerMV = false
	st.allowScreenContentTools = false
	st.color = parser.ColorConfig{BitDepth: src.BitDepth, SubsamplingX: true, SubsamplingY: true}
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
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, src.BitDepth, quantize.Plane(plane))
		if err != nil {
			return nil, err
		}
		*dst = q
	}
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
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return st.encodeHighBitDepth420Block(src, recon, block, scratch)
	}
	if err := tile.WalkBlockLoopWrite(&w, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

func (st *lossyEncodeState) encodeHighBitDepth420Block(src SourceFrame42016, recon *SourceFrame42016, block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
	if block.Size != tile.BlockSize8x8 {
		return fmt.Errorf("encoder: unexpected high-bit-depth 4:2:0 block %+v", block)
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
	for plane := 1; plane <= 2; plane++ {
		data, rdata := src.U, recon.U
		q := st.uQuant
		if plane == 2 {
			data, rdata = src.V, recon.V
			q = st.vQuant
		}
		if err := st.encodeTXBAvailRect16(rdata, data, src.ChromaStride, chromaX, chromaY, cw, ch, src.BitDepth, q, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       chromaTX,
			X4:         chromaX4,
			Y4:         chromaY4,
		}, coeffCtx, chromaScan, nil, block.HaveTop, block.HaveLeft); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
	}
	return nil
}

func (st *lossyEncodeState) encodeTXBAvailRect16(reconPlane []uint16, srcPlane []uint16, stride int, px, py, w, h int, bitDepth uint8, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, haveTop, haveLeft bool) error {

	dc := dcPredictRect16(reconPlane, stride, px, py, w, h, haveTop, haveLeft, bitDepth)
	pred := st.predY16[:w*h]
	for i := range pred {
		pred[i] = dc
	}
	return st.encodeTXBPredRect16(reconPlane, srcPlane, stride, px, py, w, h, bitDepth, q, ctxReq, coeffCtx, scan, afterSkip, pred)
}

func (st *lossyEncodeState) encodeTXBPredRect16(reconPlane []uint16, srcPlane []uint16, stride int, px, py, w, h int, bitDepth uint8, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, pred []uint16) error {

	n := w * h
	residual := &st.resScratch
	for r := range h {
		row := (py+r)*stride + px
		for c := range w {
			residual[r*w+c] = int16(srcPlane[row+c]) - int16(pred[r*w+c])
		}
	}
	tran := &st.tranScratch
	if err := forwardDCTBlock(tran[:n], residual[:n], w, h); err != nil {
		return err
	}
	qcoeff := &st.lumaQ
	scale := txScaleForSize(max(w, h))
	if err := quantize.QuantizeBlockScaledB(qcoeff[:n], h, tran[:n], h, w, h, q, scale); err != nil {
		return err
	}
	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff[:n], scan, st.levels, afterSkip); err != nil {
		return err
	}

	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n], h, qcoeff[:n], h, w, h, q, scale, bitDepth); err != nil {
		return err
	}
	res := &st.invResidual
	size := transform.Size{Width: uint8(w), Height: uint8(h)}
	if err := transform.InverseBlockBitDepth(res[:n], w, dq[:n], h, st.invScratch[:n], size, transform.TypeDCTDCT, bitDepth); err != nil {
		return err
	}
	maxSample := int((1 << bitDepth) - 1)
	for r := range h {
		row := (py+r)*stride + px
		for c := range w {
			v := int(pred[r*w+c]) + int(res[r*w+c])
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
