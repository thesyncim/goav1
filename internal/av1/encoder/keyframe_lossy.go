package encoder

import (
	"sync"

	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
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
	seq := losslessKeyframeSequence(src.Width, src.Height)
	header := lossyKeyframeHeader(src.Width, src.Height, qIndex)
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
			data, err := encodeKeyframeTile(src, &recon, qIndex, c0, c1)
			if err != nil {
				errs[t] = err
				return
			}
			payloads[t].Data = data
		}(t, c0, c1)
	}
	c0, c1 := bounds(0)
	tile0, tile0Err := encodeKeyframeTile(src, &recon, qIndex, c0, c1)
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
	mvCDFs    tile.MVCDFs

	qIndex uint8
	yQuant quantize.Quantizer
	uQuant quantize.Quantizer
	vQuant quantize.Quantizer

	scan8, scan4, scan16, scan32 []int16
	levels                       []uint8
	invScratch                   []int32
	color                        parser.ColorConfig

	// Per-block scratch reused across blocks so the hot encode loop stays
	// allocation-free: quantized coefficients per plane (sized for the largest
	// coded TXB, 16x16 luma / 8x8 chroma) and the prebuilt after-skip tx_type
	// hook (a closure built once per tile, not per block).
	lumaQ          [1024]int16
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

func encodeKeyframeTile(src SourceFrame420, recon *SourceFrame420, qIndex uint8, miColStart, miColEnd uint16) ([]byte, error) {
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
	// All blocks 8x8: split every level above BlockLevel8x8, PARTITION_NONE at
	// the 8x8 level.
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
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

	// Luma mode selection by prediction SAD against the reconstructed edges.
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	pred := st.predY[:64]
	mode := selectIntraMode8(src.Y, recon.Y, src.YStride, lumaPX, lumaPY, block.HaveTop, block.HaveLeft, pred)

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

	// Luma TX_8X8 with the tx_type symbol between txb_skip and eob.
	txTypeReq := tile.IntraTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		Mode:        mode,
		QIndexKnown: true,
		QIndex:      st.qIndex,
	}
	afterSkip := func() error {
		return tile.WriteIntraTransformType(st.w, &st.txCDFs, txTypeReq, transform.TypeDCTDCT)
	}
	if err := st.encodeTXBPred(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, 8, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       tile.TransformSize8x8,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, st.scan8, afterSkip, pred); err != nil {
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
		if err := st.encodeTXBAvail(rdata, data, src.ChromaStride, lumaPX/2, lumaPY/2, 4, q, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       tile.TransformSize4x4,
			X4:         block.X4 / 2,
			Y4:         block.Y4 / 2,
		}, coeffCtx, st.scan4, nil, block.HaveTop, block.HaveLeft); err != nil {
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
	default:
		return fmt.Errorf("encoder: unsupported txb size %d", n)
	}
	if err := quantize.QuantizeBlockScaledFP(qcoeff[:n*n], n, tran[:n*n], n, n, n, q, 0); err != nil {
		return err
	}
	if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff[:n*n], scan, st.levels, afterSkip); err != nil {
		return err
	}

	// Reconstruct exactly as the decoder will: dequantize, inverse transform,
	// add to the prediction, clip to 8 bits.
	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n*n], n, qcoeff[:n*n], n, n, n, q, 0, 8); err != nil {
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

// selectIntraMode8 chooses the luma intra mode of one 8x8 block by
// prediction SAD against the reconstructed edges and fills pred (stride 8)
// with the chosen prediction. DC competes always; the exact-angle vertical
// and horizontal copies need their edge (angles 90 and 180 predict without
// edge filtering, so the copies match the decoder bit for bit). DC keeps
// ties: it codes no angle-delta symbol. SMOOTH was built and measured a
// wash at 8x8 - DC plus a two-coefficient DCT residual already covers
// gradients - and removed.
func selectIntraMode8(srcPlane, reconPlane []byte, stride, px, py int, haveTop, haveLeft bool, pred []byte) tile.IntraMode {
	dc := dcPredictN(reconPlane, stride, px, py, 8, haveTop, haveLeft)
	sadDC, sadV, sadH := 0, 1<<30, 1<<30
	for r := range 8 {
		row := (py+r)*stride + px
		for c := range 8 {
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
		for r := range 8 {
			row := (py+r)*stride + px
			for c := range 8 {
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
		for r := range 8 {
			row := (py+r)*stride + px
			left := int(reconPlane[row-1])
			for c := range 8 {
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
		for r := range 8 {
			copy(pred[r*8:r*8+8], reconPlane[above:above+8])
		}
		return tile.IntraModeVertical
	case sadH+16 < sadDC:
		for r := range 8 {
			v := reconPlane[(py+r)*stride+px-1]
			for c := range 8 {
				pred[r*8+c] = v
			}
		}
		return tile.IntraModeHorizontal
	default:
		for i := range 64 {
			pred[i] = dc
		}
		return tile.IntraModeDC
	}
}
