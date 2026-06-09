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
	tilePayload, err := encodeKeyframeTile(src, &recon, qIndex)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("encode tile: %w", err)
	}

	seq := losslessKeyframeSequence(src.Width, src.Height)
	header := lossyKeyframeHeader(src.Width, src.Height, qIndex)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("size header TU: %w", err)
	}
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
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

	qIndex uint8
	yQuant quantize.Quantizer
	uQuant quantize.Quantizer
	vQuant quantize.Quantizer

	scan8, scan4 []int16
	levels       []uint8
	invScratch   []int32
	color        parser.ColorConfig
}

func encodeKeyframeTile(src SourceFrame420, recon *SourceFrame420, qIndex uint8) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	if err := partCDFs.InitDefault(); err != nil {
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
	// All blocks 8x8: split every level above BlockLevel8x8, PARTITION_NONE at
	// the 8x8 level.
	decide := func(level tile.BlockLevel, ctx int, haveRight, haveBottom bool) (tile.Partition, error) {
		if level == tile.BlockLevel8x8 {
			return tile.PartitionNone, nil
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
	if block.Size != tile.BlockSize8x8 {
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
	if err := tile.WriteLumaIntraMode(st.w, &st.intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
	}, tile.IntraModeDC); err != nil {
		return fmt.Errorf("luma mode: %w", err)
	}
	cflAllowed, err := tile.ChromaIntraCFLAllowed(block.Size, st.color, false)
	if err != nil {
		return fmt.Errorf("cfl allowed: %w", err)
	}
	if err := tile.WriteChromaIntraMode(st.w, &st.intraCDFs, tile.ChromaIntraModeRequest{
		Size: block.Size, LumaMode: tile.IntraModeDC, CFLAllowed: cflAllowed,
	}, tile.ChromaIntraModeDC, tile.CFLAlphaResult{}); err != nil {
		return fmt.Errorf("chroma mode: %w", err)
	}

	// Luma TX_8X8 with the tx_type symbol between txb_skip and eob.
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	txTypeReq := tile.IntraTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		Mode:        tile.IntraModeDC,
		QIndexKnown: true,
		QIndex:      st.qIndex,
	}
	afterSkip := func() error {
		return tile.WriteIntraTransformType(st.w, &st.txCDFs, txTypeReq, transform.TypeDCTDCT)
	}
	if err := st.encodeTXB(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, 8, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, st.scan8, afterSkip); err != nil {
		return fmt.Errorf("luma txb: %w", err)
	}

	// Chroma TX_4X4, one per plane (8x8 luma block at 4:2:0).
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
		if err := st.encodeTXB(rdata, data, src.ChromaStride, lumaPX/2, lumaPY/2, 4, q, tile.CoeffContextRequest{
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

// encodeTXB codes one square n x n transform block: DC-predict from the recon
// plane, forward transform the source residual, quantize, write the
// coefficients, then reconstruct through the decoder's own dequant + inverse
// transform so recon matches the decoder bit-for-bit.
func (st *lossyEncodeState) encodeTXB(reconPlane []byte, srcPlane []byte, stride int, px, py, n int, q quantize.Quantizer,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error) error {

	dc := dcPredictN(reconPlane, stride, px, py, n)

	var residual [64]int16
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			residual[r*n+c] = int16(srcPlane[row+c]) - int16(dc)
		}
	}
	var tran [64]int32
	var qcoeff [64]int16
	switch n {
	case 4:
		if err := transform.ForwardDCT4x4(tran[:16], 4, residual[:16], 4); err != nil {
			return err
		}
	case 8:
		if err := transform.ForwardDCT8x8(tran[:64], 8, residual[:64], 8); err != nil {
			return err
		}
	default:
		return fmt.Errorf("encoder: unsupported txb size %d", n)
	}
	if err := quantize.QuantizeBlockScaled(qcoeff[:n*n], n, tran[:n*n], n, n, n, q, 0); err != nil {
		return err
	}
	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff[:n*n], scan, st.levels, afterSkip); err != nil {
		return err
	}

	// Reconstruct exactly as the decoder will: dequantize, inverse transform,
	// add to the prediction, clip to 8 bits.
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
		for c := range n {
			v := int(dc) + int(res[r*n+c])
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
// neighbors that exist, or 128 when neither edge does.
func dcPredictN(plane []byte, stride, px, py, n int) uint8 {
	sum := 0
	count := 0
	if py > 0 {
		row := (py-1)*stride + px
		for i := range n {
			sum += int(plane[row+i])
		}
		count += n
	}
	if px > 0 {
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
