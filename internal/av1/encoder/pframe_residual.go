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

// EncodeMonochromePFrame encodes src as a native AV1 monochrome inter frame
// predicting from ref, returning the temporal unit and the encoder
// reconstruction the decoder must reproduce exactly.
func EncodeMonochromePFrame(src SourceFrameMono, ref SourceFrameMono, qIndex uint8) ([]byte, SourceFrameMono, error) {
	if err := validateSourceFrameMono(src); err != nil {
		return nil, SourceFrameMono{}, err
	}
	if err := validateSourceFrameMono(ref); err != nil {
		return nil, SourceFrameMono{}, err
	}
	if src.Width != ref.Width || src.Height != ref.Height {
		return nil, SourceFrameMono{}, fmt.Errorf("encoder: source %dx%d does not match reference %dx%d", src.Width, src.Height, ref.Width, ref.Height)
	}
	if qIndex == 0 {
		return nil, SourceFrameMono{}, fmt.Errorf("encoder: qindex 0 lossless inter coding is not supported")
	}
	recon := SourceFrameMono{
		Y:       make([]byte, len(src.Y)),
		YStride: src.YStride,
		Width:   src.Width,
		Height:  src.Height,
	}
	var pc pframeCoder
	tilePayload, err := pc.encodeMonochromeTile(src, ref, nil, &recon, qIndex, nil, parser.ReferenceModeSingle, 0, uint16(src.Width/4))
	if err != nil {
		return nil, SourceFrameMono{}, fmt.Errorf("encode tile: %w", err)
	}

	seq := lossyMonochromeKeyframeSequence(src.Width, src.Height)
	header, refState := repeatPFrameHeader(src.Width, src.Height, qIndex, 0x01)
	header.References = &refState
	header.CDEF = CDEFParams{}

	var groupScratch, outScratch []byte
	out, err := assembleInterTU(seq, header, []TilePayload{{Data: tilePayload}}, 0, &groupScratch, &outScratch)
	if err != nil {
		return nil, SourceFrameMono{}, err
	}
	return out, recon, nil
}

// EncodeScaledReferencePFrame encodes src as an inter frame whose LAST
// reference has different coded dimensions. The returned temporal unit assumes
// the decoder already holds that smaller or larger reference frame in slot 0.
func EncodeScaledReferencePFrame(src SourceFrame420, ref SourceFrame420, qIndex uint8) ([]byte, SourceFrame420, error) {
	if src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: scaled-reference P-frame requires multiple-of-8 source dimensions, got %dx%d", src.Width, src.Height)
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Width%8 != 0 || ref.Height%8 != 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: scaled-reference P-frame requires multiple-of-8 reference dimensions, got %dx%d", ref.Width, ref.Height)
	}
	if qIndex == 0 {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: qindex 0 lossless inter coding is not supported")
	}
	if _, err := motion.NewScaleFactors(ref.Width, ref.Height, src.Width, src.Height); err != nil {
		return nil, SourceFrame420{}, fmt.Errorf("encoder: invalid scaled reference: %w", err)
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
	refState = referenceStateForFrame(ref.Width, ref.Height)
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

const (
	sadCacheBits      = 20
	sadCacheValueMask = (uint32(1) << sadCacheBits) - 1
	sadCacheMaxEpoch  = (uint32(1) << (32 - sadCacheBits)) - 1
)

func sadCachePack(epoch uint32, sad int) uint32 {
	return epoch<<sadCacheBits | uint32(sad)
}

func sadCacheValid(v uint32, epoch uint32) bool {
	return epoch != 0 && v>>sadCacheBits == epoch
}

func sadCacheValue(v uint32) int {
	return int(v & sadCacheValueMask)
}

func (st *lossyEncodeState) beginSADCacheFrame() {
	st.sadCacheEpoch++
	if st.sadCacheEpoch <= sadCacheMaxEpoch {
		return
	}
	clear(st.sad8Grid)
	clear(st.sad16Grid)
	clear(st.sad32Grid)
	clear(st.sad64Grid)
	st.sadCacheEpoch = 1
}

// reset (re)initializes the per-frame CDF and quantizer state. Buffers are
// allocated on first use and reused afterwards. When prev is non-nil the
// symbol contexts chain from it - the saved state of the frame named by
// primary_ref_frame - instead of the defaults, exactly as the decoder loads
// them.
func (pc *pframeCoder) reset(qIndex uint8, rootCols int, prev *frameCDFs, color parser.ColorConfig) error {
	st := &pc.st
	st.qIndex = qIndex
	st.color = color
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
		if color.MonoChrome && plane != 0 {
			*dst = quantize.Quantizer{}
			continue
		}
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, color.BitDepth, quantize.Plane(plane))
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

func (st *lossyEncodeState) scanForTransformSize(size tile.TransformSize) ([]int16, bool) {
	switch size {
	case tile.TransformSize4x4:
		return st.scan4, true
	case tile.TransformSize8x8:
		return st.scan8, true
	case tile.TransformSize16x16:
		return st.scan16, true
	case tile.TransformSize32x32:
		return st.scan32, true
	case tile.TransformSize16x8:
		return st.scan16x8, true
	case tile.TransformSize8x16:
		return st.scan8x16, true
	case tile.TransformSize32x16:
		return st.scan32x16, true
	case tile.TransformSize16x32:
		return st.scan16x32, true
	case tile.TransformSize8x4:
		return st.scan8x4, true
	case tile.TransformSize4x8:
		return st.scan4x8, true
	default:
		return nil, false
	}
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
	return pc.encodeTileWithOptions(src, ref, golden, recon, qIndex, prev, referenceMode, false, false, miColStart, miColEnd)
}

func (pc *pframeCoder) encodeTileWithOptions(src SourceFrame420, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, qIndex uint8, prev *frameCDFs, referenceMode parser.ReferenceMode, forceIntegerMV bool, allowScreenContentTools bool, miColStart, miColEnd uint16) ([]byte, error) {
	return pc.encodeTileWithOptionsColor(src, ref, golden, recon, qIndex, prev, referenceMode, forceIntegerMV, allowScreenContentTools, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}, miColStart, miColEnd)
}

func (pc *pframeCoder) encodeMonochromeTile(src SourceFrameMono, ref SourceFrameMono, golden *SourceFrameMono, recon *SourceFrameMono, qIndex uint8, prev *frameCDFs, referenceMode parser.ReferenceMode, miColStart, miColEnd uint16) ([]byte, error) {
	return pc.encodeMonochromeTileWithOptions(src, ref, golden, recon, qIndex, prev, referenceMode, false, false, miColStart, miColEnd)
}

func (pc *pframeCoder) encodeMonochromeTileWithOptions(src SourceFrameMono, ref SourceFrameMono, golden *SourceFrameMono, recon *SourceFrameMono, qIndex uint8, prev *frameCDFs, referenceMode parser.ReferenceMode, forceIntegerMV bool, allowScreenContentTools bool, miColStart, miColEnd uint16) ([]byte, error) {
	src420 := SourceFrame420{Y: src.Y, YStride: src.YStride, Width: src.Width, Height: src.Height}
	ref420 := SourceFrame420{Y: ref.Y, YStride: ref.YStride, Width: ref.Width, Height: ref.Height}
	recon420 := SourceFrame420{Y: recon.Y, YStride: recon.YStride, Width: recon.Width, Height: recon.Height}
	var golden420 *SourceFrame420
	if golden != nil {
		g := SourceFrame420{Y: golden.Y, YStride: golden.YStride, Width: golden.Width, Height: golden.Height}
		golden420 = &g
	}
	return pc.encodeTileWithOptionsColor(src420, ref420, golden420, &recon420, qIndex, prev, referenceMode, forceIntegerMV, allowScreenContentTools, parser.ColorConfig{BitDepth: 8, MonoChrome: true, SubsamplingX: true, SubsamplingY: true}, miColStart, miColEnd)
}

func (pc *pframeCoder) encodeTileWithOptionsColor(src SourceFrame420, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, qIndex uint8, prev *frameCDFs, referenceMode parser.ReferenceMode, forceIntegerMV bool, allowScreenContentTools bool, color parser.ColorConfig, miColStart, miColEnd uint16) ([]byte, error) {
	miCols := uint16(src.Width / 4)
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB
	if err := pc.reset(qIndex, rootCols, prev, color); err != nil {
		return nil, err
	}
	st := &pc.st
	st.forceIntegerMV = forceIntegerMV
	st.allowScreenContentTools = allowScreenContentTools
	st.transformMode = parser.TransformModeSwitchable
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
	// contexts; they re-arm CDFs lazily on first use each frame.
	st.trialReady = false
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
	scaledReference := ref.Width != src.Width || ref.Height != src.Height

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
		st.sad16Grid = make([]uint32, st.grid16Cols*grid16Rows)
	}
	st.grid8Cols = (int(miCols) + 1) / 2
	grid8Rows := (int(miRows) + 1) / 2
	if len(st.mv8Grid) < st.grid8Cols*grid8Rows {
		st.mv8Grid = make([]motion.Vector, st.grid8Cols*grid8Rows)
		st.sad8Grid = make([]uint32, st.grid8Cols*grid8Rows)
	}
	st.grid32Cols = (int(miCols) + 7) / 8
	grid32Rows := (int(miRows) + 7) / 8
	if len(st.mv32Grid) < st.grid32Cols*grid32Rows {
		st.mv32Grid = make([]motion.Vector, st.grid32Cols*grid32Rows)
		st.sad32Grid = make([]uint32, st.grid32Cols*grid32Rows)
	}
	st.grid64Cols = (int(miCols) + 15) / 16
	grid64Rows := (int(miRows) + 15) / 16
	if len(st.mv64Grid) < st.grid64Cols*grid64Rows {
		st.mv64Grid = make([]motion.Vector, st.grid64Cols*grid64Rows)
		st.sad64Grid = make([]uint32, st.grid64Cols*grid64Rows)
	}
	st.beginSADCacheFrame()
	sadEpoch := st.sadCacheEpoch
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
		sad16Entry := st.sad16Grid[idx16]
		if !sadCacheValid(sad16Entry, sadEpoch) {
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
			sad16Entry = sadCachePack(sadEpoch, sad16)
			st.sad16Grid[idx16] = sad16Entry
		}
		return sadCacheValue(sad16Entry)
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
			sad8Entry := st.sad8Grid[idx8]
			if !sadCacheValid(sad8Entry, sadEpoch) {
				dx, dy, sad := fullPelDiamondSearchSeeded(src.Y, ref.Y, src.YStride, src.Width, src.Height, cx, cy, 8, seedDX, seedDY, reach)
				st.mv8Grid[idx8] = motion.Vector{Row: int16(dy * 8), Col: int16(dx * 8)}
				sad8Entry = sadCachePack(sadEpoch, sad)
				st.sad8Grid[idx8] = sad8Entry
			}
			sum8 += sadCacheValue(sad8Entry)
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
		if !videoColorIs420(color) {
			return tile.PartitionSplit, nil
		}
		if scaledReference {
			return tile.PartitionSplit, nil
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
			sad64 := sad64x64(src.Y[base:], ref.Y[refBase:], src.YStride)
			if sad64 <= childCost+mergeBias64 {
				idx64 := (py/64)*st.grid64Cols + px/64
				st.mv64Grid[idx64] = mv64
				st.sad64Grid[idx64] = sadCachePack(sadEpoch, sad64)
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
				st.sad32Grid[idx32] = sadCachePack(sadEpoch, sad32)
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
				sa, sb := sadCacheValue(st.sad16Grid[idx[ia]]), sadCacheValue(st.sad16Grid[idx[ib]])
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
				st.sad16Grid[idx[ia]], st.sad16Grid[idx[ib]] = sadCachePack(sadEpoch, bestA), sadCachePack(sadEpoch, bestB)
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
				sa, sb := sadCacheValue(st.sad8Grid[idx[ia]]), sadCacheValue(st.sad8Grid[idx[ib]])
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
				st.sad8Grid[idx[ia]], st.sad8Grid[idx[ib]] = sadCachePack(sadEpoch, bestA), sadCachePack(sadEpoch, bestB)
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
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	hasChroma := !st.color.MonoChrome
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx
	cbw, cbh := 0, 0
	chromaBlock := tile.BlockSize4x4
	chromaPX, chromaPY := 0, 0
	chromaX4, chromaY4 := uint8(0), uint8(0)
	chromaWidth, chromaHeight := 0, 0
	if hasChroma {
		var err error
		chromaBlock, err = tile.PlaneBlockSize(block.Size, st.color, 1)
		if err != nil {
			return fmt.Errorf("chroma plane block: %w", err)
		}
		cbw, cbh, err = planeBlockPixels(chromaBlock)
		if err != nil {
			return fmt.Errorf("chroma plane block dimensions: %w", err)
		}
		chromaPX = chromaXForColor(lumaPX, st.color)
		chromaPY = chromaYForColor(lumaPY, st.color)
		chromaX4 = chromaX4ForColor(block.X4, st.color)
		chromaY4 = chromaY4ForColor(block.Y4, st.color)
		chromaWidth = chromaWidthForColor(src.Width, st.color)
		chromaHeight = chromaHeightForColor(src.Height, st.color)
	}

	// Motion estimation first: the skip decision needs the motion-compensated
	// residual. The partition decider already ran the full-pel searches and
	// cached them; fall back to a fresh search for blocks it never scored
	// (frame-edge nodes reach the leaf without a 16x16 decision). Subpel
	// refinement through the decoder's own convolve when the match is
	// imperfect.
	scaledReference := ref.Width != src.Width || ref.Height != src.Height
	var mv motion.Vector
	fullSAD := -1
	sadEpoch := st.sadCacheEpoch
	if scaledReference {
		if err := predictIntoScaled(st.sadScratch[:bw*bh], ref.Y, ref.YStride, ref.Width, ref.Height, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false, &st.scaledScratch); err != nil {
			return fmt.Errorf("scaled reference luma probe: %w", err)
		}
		fullSAD = sadRectDualBlock(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, st.sadScratch[:bw*bh], bw, bw, bh)
	} else {
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
				if sadCacheValid(st.sad16Grid[i0], sadEpoch) && sadCacheValid(st.sad16Grid[i1], sadEpoch) {
					mv = st.mv16Grid[i0]
					fullSAD = sadCacheValue(st.sad16Grid[i0]) + sadCacheValue(st.sad16Grid[i1])
				}
				break
			}
			i0 := (lumaPY/8)*st.grid8Cols + lumaPX/8
			i1 := i0 + 1
			if bh > bw {
				i1 = i0 + st.grid8Cols
			}
			if sadCacheValid(st.sad8Grid[i0], sadEpoch) && sadCacheValid(st.sad8Grid[i1], sadEpoch) {
				mv = st.mv8Grid[i0]
				fullSAD = sadCacheValue(st.sad8Grid[i0]) + sadCacheValue(st.sad8Grid[i1])
			}
		case n == 64:
			idx := (lumaPY/64)*st.grid64Cols + lumaPX/64
			if sadCacheValid(st.sad64Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv64Grid[idx], sadCacheValue(st.sad64Grid[idx])
			}
		case n == 32:
			idx := (lumaPY/32)*st.grid32Cols + lumaPX/32
			if sadCacheValid(st.sad32Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv32Grid[idx], sadCacheValue(st.sad32Grid[idx])
			}
		case n == 16:
			idx := (lumaPY/16)*st.grid16Cols + lumaPX/16
			if sadCacheValid(st.sad16Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv16Grid[idx], sadCacheValue(st.sad16Grid[idx])
			}
		default:
			if idx := (lumaPY/8)*st.grid8Cols + lumaPX/8; sadCacheValid(st.sad8Grid[idx], sadEpoch) {
				mv, fullSAD = st.mv8Grid[idx], sadCacheValue(st.sad8Grid[idx])
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
		if !st.forceIntegerMV && bw == bh && fullSAD > n*n*2 {
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
	if !scaledReference && golden != nil && golden.Y != nil && goldenEligible && fullSAD > bw*bh*4 {
		lastMV, lastSAD := mv, fullSAD
		var gmv motion.Vector
		gsad := 1 << 30
		if bw == 8 {
			gdx, gdy, s := fullPelDiamondSearch(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, 8)
			gmv, gsad = motion.Vector{Row: int16(gdy * 8), Col: int16(gdx * 8)}, s
			if !st.forceIntegerMV && gsad > 8*8*2 {
				gmv, gsad = st.subpelRefine(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, 8, gmv, gsad)
			}
		} else {
			base := lumaPY*src.YStride + lumaPX
			gsad = sadRectBlock(src.Y, golden.Y, base, base, src.YStride, bw, bh, lastSAD)
		}
		if bw == 8 && referenceMode == parser.ReferenceModeSelect && gsad+32 >= lastSAD && gsad <= lastSAD+8*8*4 {
			if err := predictCompoundInto(st.sadScratch[:64], ref.Y, ref.YStride, golden.Y, golden.YStride, src.Width, src.Height, lumaPX, lumaPY, 8, 8, lastMV, gmv, false, false, &st.compBuf0, &st.compBuf1, &st.compScratch); err == nil {
				compoundSAD := sad8x8Dual(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, st.sadScratch[:64], 8)
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
		ForceIntegerMV: st.forceIntegerMV,
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
			if predRefs, err := stack.Stack.ResolveInterMVReferences(tile.InterModeResult{Mode: tile.InterModeNearestMV}, 0, false, st.forceIntegerMV); err == nil {
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
								srcBlock := src.Y[lumaPY*src.YStride+lumaPX:]
								predBlock := st.sadScratch[:bw*bh]
								nearSAD := 0
								switch {
								case bw == 8 && bh == 8:
									nearSAD = sad8x8Dual(srcBlock, src.YStride, predBlock, bw)
								case bw == 16 && bh == 16:
									nearSAD = sad16x16Dual(srcBlock, src.YStride, predBlock, bw)
								case bw == 32 && bh == 32:
									nearSAD = sad32x32Dual(srcBlock, src.YStride, predBlock, bw)
								case bw == 64 && bh == 64:
									nearSAD = sadDualBlock(srcBlock, src.YStride, predBlock, bw, bw)
								case bw == 16 && bh == 8:
									nearSAD = sad8x8Dual(srcBlock, src.YStride, predBlock, bw) +
										sad8x8Dual(srcBlock[8:], src.YStride, predBlock[8:], bw)
								case bw == 8 && bh == 16:
									nearSAD = sad8x8Dual(srcBlock, src.YStride, predBlock, bw) +
										sad8x8Dual(srcBlock[8*src.YStride:], src.YStride, predBlock[8*bw:], bw)
								case bw == 32 && bh == 16:
									nearSAD = sad16x16Dual(srcBlock, src.YStride, predBlock, bw) +
										sad16x16Dual(srcBlock[16:], src.YStride, predBlock[16:], bw)
								case bw == 16 && bh == 32:
									nearSAD = sad16x16Dual(srcBlock, src.YStride, predBlock, bw) +
										sad16x16Dual(srcBlock[16*src.YStride:], src.YStride, predBlock[16*bw:], bw)
								default:
									nearSAD = sadRectDualBlock(srcBlock, src.YStride, predBlock, bw, bw, bh)
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
	if refs.Compound {
		if err := predictCompoundInto(st.predY[:bw*bh], ref.Y, ref.YStride, golden.Y, golden.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, compoundMV[0], compoundMV[1], false, false, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
			return fmt.Errorf("predict compound luma: %w", err)
		}
		if hasChroma {
			if err := predictCompoundInto(st.predU[:cbw*cbh], ref.U, ref.ChromaStride, golden.U, golden.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, compoundMV[0], compoundMV[1], st.color.SubsamplingX, st.color.SubsamplingY, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
				return fmt.Errorf("predict compound u: %w", err)
			}
			if err := predictCompoundInto(st.predV[:cbw*cbh], ref.V, ref.ChromaStride, golden.V, golden.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, compoundMV[0], compoundMV[1], st.color.SubsamplingX, st.color.SubsamplingY, &st.compBuf0, &st.compBuf1, &st.compScratch); err != nil {
				return fmt.Errorf("predict compound v: %w", err)
			}
		}
	} else {
		if refPlanes.Width != src.Width || refPlanes.Height != src.Height {
			refChromaW := chromaWidthForColor(refPlanes.Width, st.color)
			refChromaH := chromaHeightForColor(refPlanes.Height, st.color)
			if err := predictIntoScaled(st.predY[:bw*bh], refPlanes.Y, refPlanes.YStride, refPlanes.Width, refPlanes.Height, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false, &st.scaledScratch); err != nil {
				return fmt.Errorf("predict scaled luma: %w", err)
			}
			if hasChroma {
				if err := predictIntoScaled(st.predU[:cbw*cbh], refPlanes.U, refPlanes.ChromaStride, refChromaW, refChromaH, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, &st.scaledScratch); err != nil {
					return fmt.Errorf("predict scaled u: %w", err)
				}
				if err := predictIntoScaled(st.predV[:cbw*cbh], refPlanes.V, refPlanes.ChromaStride, refChromaW, refChromaH, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, &st.scaledScratch); err != nil {
					return fmt.Errorf("predict scaled v: %w", err)
				}
			}
		} else {
			if err := predictInto(st.predY[:bw*bh], refPlanes.Y, refPlanes.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false); err != nil {
				return fmt.Errorf("predict luma: %w", err)
			}
			if hasChroma {
				if err := predictInto(st.predU[:cbw*cbh], refPlanes.U, refPlanes.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY); err != nil {
					return fmt.Errorf("predict u: %w", err)
				}
				if err := predictInto(st.predV[:cbw*cbh], refPlanes.V, refPlanes.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY); err != nil {
					return fmt.Errorf("predict v: %w", err)
				}
			}
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
	dctRdD := int64(0)
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
		if hasChroma {
			uZero := st.prepareInterTXB(src.U, st.predU[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.uQuant, st.uQ[:cbw*cbh])
			vZero := st.prepareInterTXB(src.V, st.predV[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.vQuant, st.vQ[:cbw*cbh])
			skip = lumaZero && uZero && vZero
		} else {
			skip = lumaZero
		}
		dctRdD = st.rdDcode
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
				costFull := st.trialTXBBitsY16x16((*[256]int16)(st.lumaQ[:256])) + lumaRdD<<7
				costSplit := splitD << 7
				costSplit += st.trialTXBBitsY8x8((*[64]int16)(st.lumaQ2[0:64]))
				costSplit += st.trialTXBBitsY8x8((*[64]int16)(st.lumaQ2[64:128]))
				costSplit += st.trialTXBBitsY8x8((*[64]int16)(st.lumaQ2[128:192]))
				costSplit += st.trialTXBBitsY8x8((*[64]int16)(st.lumaQ2[192:256]))
				if costSplit < costFull {
					splitTX = true
				}
			}
		}
	}
	if !videoColorIs420(st.color) {
		skip = false
		splitTX = false
	}

	txType := transform.TypeDCTDCT
	if !skip && !splitTX && bw == 8 && bh == 8 && st.qIndex <= 96 && videoColorIs420(st.color) {
		txType = st.chooseInter8x8TXType(src, lumaPX, lumaPY, dctRdD)
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
		mvRefs, err = stack.Stack.ResolveInterMVReferences(modeResult, 0, false, st.forceIntegerMV)
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
		Precision:    tile.MVPrecision(false, st.forceIntegerMV),
	}, motionResult); err != nil {
		return fmt.Errorf("motion vector: %w", err)
	}

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
		Color: st.color, TransformMode: st.transformMode,
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

	if skip {
		// No residual symbols; reset the coefficient entropy contexts per
		// plane as the decoder's residual path does for skip blocks, and the
		// reconstruction is the motion-compensated prediction.
		if err := coeffCtx.ResetBlock(0, block.Size, int(block.X4), int(block.Y4)); err != nil {
			return fmt.Errorf("reset luma coeff ctx: %w", err)
		}
		if hasChroma {
			for plane := 1; plane <= 2; plane++ {
				if err := coeffCtx.ResetBlock(plane, chromaBlock, int(chromaX4), int(chromaY4)); err != nil {
					return fmt.Errorf("reset chroma %d coeff ctx: %w", plane, err)
				}
			}
		}
		copyPredScratch(recon.Y, st.predY[:bw*bh], src.YStride, lumaPX, lumaPY, bw, bh)
		if hasChroma {
			copyPredScratch(recon.U, st.predU[:cbw*cbh], src.ChromaStride, chromaPX, chromaPY, cbw, cbh)
			copyPredScratch(recon.V, st.predV[:cbw*cbh], src.ChromaStride, chromaPX, chromaPY, cbw, cbh)
		}
		if st.decisionStats != nil {
			st.decisionStats.noteInterBlock(block.Size, true, false, refs, modeResult, transform.TypeDCTDCT)
		}
		return nil
	}

	// Residual: largest-TX luma with the inter tx_type symbol, then chroma.
	lumaTX, lumaScan := tile.TransformSize8x8, st.scan8
	switch {
	case bw == 16 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize16x16, st.scan16
	case bw == 32 && bh == 32:
		lumaTX, lumaScan = tile.TransformSize32x32, st.scan32
	case bw == 16 && bh == 8:
		lumaTX, lumaScan = tile.TransformSize16x8, st.scan16x8
	case bw == 8 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize8x16, st.scan8x16
	case bw == 32 && bh == 16:
		lumaTX, lumaScan = tile.TransformSize32x16, st.scan32x16
	case bw == 16 && bh == 32:
		lumaTX, lumaScan = tile.TransformSize16x32, st.scan16x32
	}
	chromaTX, chromaScan := tile.TransformSize4x4, st.scan4
	if hasChroma {
		var err error
		chromaTX, err = tile.MaxTransformSize(block.Size, st.color, 1)
		if err != nil {
			return fmt.Errorf("chroma transform size: %w", err)
		}
		var ok bool
		chromaScan, ok = st.scanForTransformSize(chromaTX)
		if !ok {
			return fmt.Errorf("encoder: unsupported chroma transform %d", chromaTX)
		}
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
	if hasChroma {
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
			if err := st.finishInterTXBTyped(rdata, pred, cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, q, qc, tile.CoeffContextRequest{
				Plane:      uint8(plane),
				PlaneBlock: chromaBlock,
				Size:       chromaTX,
				X4:         chromaX4,
				Y4:         chromaY4,
			}, coeffCtx, chromaScan, nil, chromaTxType); err != nil {
				return fmt.Errorf("chroma %d txb: %w", plane, err)
			}
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
	hasChroma := !st.color.MonoChrome
	if hasChroma {
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
		if err := st.writeNoPaletteMode(modeCtx, block, mode, tile.ChromaIntraModeDC, true); err != nil {
			return err
		}
	} else if err := st.writeNoPaletteMode(modeCtx, block, mode, tile.ChromaIntraModeDC, false); err != nil {
		return err
	}

	lfTree, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: st.transformMode,
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
	if !hasChroma {
		if st.decisionStats != nil {
			st.decisionStats.noteIntraBlock(block.Size)
		}
		return nil
	}
	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("chroma plane block: %w", err)
	}
	cw, ch, err := planeBlockPixels(chromaBlock)
	if err != nil {
		return fmt.Errorf("chroma plane block dimensions: %w", err)
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
		if err := st.encodeTXBAvailRect(rdata, data, src.ChromaStride, chromaX, chromaY, cw, ch, q, tile.CoeffContextRequest{
			Plane:      uint8(plane),
			PlaneBlock: chromaBlock,
			Size:       chromaTX,
			X4:         chromaX4,
			Y4:         chromaY4,
		}, coeffCtx, chromaScan, nil, block.HaveTop, block.HaveLeft); err != nil {
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
	return predictIntoScaled(dst, refPlane, stride, width, height, width, height, px, py, bw, bh, mv, ssX, ssY, nil)
}

// predictIntoScaled runs the same inter predictor as predictInto, but allows
// the reference plane and current plane to have different dimensions. AV1's
// scaled inter prediction derives the reference taps from the reference-vs-
// current size ratio, so callers must pass the full current plane dimensions
// rather than the destination scratch block dimensions.
func predictIntoScaled(dst []byte, refPlane []byte, stride, refWidth, refHeight, curWidth, curHeight, px, py, bw, bh int, mv motion.Vector, ssX, ssY bool, scratch *motion.ScaledConvolveScratch) error {
	dstPlane := frame.Plane{Pix: dst, Stride: bw, Width: bw, Height: bh}
	ref := frame.Plane{Pix: refPlane, Stride: stride, Width: refWidth, Height: refHeight}
	filters := motion.RegularFilters
	if refWidth == curWidth && refHeight == curHeight {
		refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, ssX, ssY)
		if err != nil {
			return err
		}
		return motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstPlane, ref, 1, 8, 0, 0, refX, refY, bw, bh, subX, subY, filters)
	}
	sf, err := motion.NewScaleFactors(refWidth, refHeight, curWidth, curHeight)
	if err != nil {
		return err
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(px, py, mv, ssX, ssY)
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
	return motion.ConvolveScale2D8ClampedWithScratch(dstPlane, ref, 0, 0, bw, bh, startX, xStep, startY, yStep, xTable, yTable, scratch)
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
	srcOff := py*srcStride + px
	ref0Off := (py+dy0)*stride0 + px + dx0
	ref1Off := (py+dy1)*stride1 + px + dx1
	return sad8x8CompoundAvgBlock(src[srcOff:], srcStride, ref0[ref0Off:], stride0, ref1[ref1Off:], stride1)
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
	switch n {
	case 8:
		return st.subpelRefine8x8(src, refPlane, stride, width, height, px, py, mv, bestSAD)
	case 16:
		return st.subpelRefine16x16(src, refPlane, stride, width, height, px, py, mv, bestSAD)
	case 32:
		return st.subpelRefine32x32(src, refPlane, stride, width, height, px, py, mv, bestSAD)
	}
	return st.subpelRefineGeneric(src, refPlane, stride, width, height, px, py, n, mv, bestSAD)
}

func (st *lossyEncodeState) subpelRefine8x8(src, refPlane []byte, stride, width, height, px, py int, mv motion.Vector, bestSAD int) (motion.Vector, int) {
	st.prober.Init(frame.Plane{
		Pix: refPlane, Stride: stride, Width: width, Height: height,
	}, px+int(mv.Col)>>3, py+int(mv.Row)>>3, 8)
	start := mv
	center := bestSAD
	probe := st.sadScratch[:64]
	srcBlock := src[py*stride+px:]

	left := motion.Vector{Row: start.Row, Col: start.Col - 4}
	halfLeft := st.subpelExact8x8(probe, srcBlock, refPlane, stride, width, height, px, py, start, left)
	if halfLeft >= 0 && halfLeft < bestSAD {
		bestSAD, mv = halfLeft, left
	}
	right := motion.Vector{Row: start.Row, Col: start.Col + 4}
	halfRight := st.subpelExact8x8(probe, srcBlock, refPlane, stride, width, height, px, py, start, right)
	if halfRight >= 0 && halfRight < bestSAD {
		bestSAD, mv = halfRight, right
	}
	up := motion.Vector{Row: start.Row - 4, Col: start.Col}
	halfUp := st.subpelExact8x8(probe, srcBlock, refPlane, stride, width, height, px, py, start, up)
	if halfUp >= 0 && halfUp < bestSAD {
		bestSAD, mv = halfUp, up
	}
	down := motion.Vector{Row: start.Row + 4, Col: start.Col}
	halfDown := st.subpelExact8x8(probe, srcBlock, refPlane, stride, width, height, px, py, start, down)
	if halfDown >= 0 && halfDown < bestSAD {
		bestSAD, mv = halfDown, down
	}

	estX := subpelQuarterAxis(halfLeft, halfRight, center)
	estY := subpelQuarterAxis(halfUp, halfDown, center)
	qx, qy := estX&^1, estY&^1
	if qx != 0 || qy != 0 {
		cand := motion.Vector{Row: start.Row + int16(qy), Col: start.Col + int16(qx)}
		if cand != mv && cand != start {
			if s := st.subpelExact8x8(probe, srcBlock, refPlane, stride, width, height, px, py, start, cand); s >= 0 && s < bestSAD {
				bestSAD, mv = s, cand
			}
		}
	}
	qx, qy = (estX+1)&^1, (estY+1)&^1
	if qx != 0 || qy != 0 {
		cand := motion.Vector{Row: start.Row + int16(qy), Col: start.Col + int16(qx)}
		if cand != mv && cand != start {
			if s := st.subpelExact8x8(probe, srcBlock, refPlane, stride, width, height, px, py, start, cand); s >= 0 && s < bestSAD {
				bestSAD, mv = s, cand
			}
		}
	}
	return mv, bestSAD
}

func (st *lossyEncodeState) subpelRefine16x16(src, refPlane []byte, stride, width, height, px, py int, mv motion.Vector, bestSAD int) (motion.Vector, int) {
	st.prober.Init(frame.Plane{
		Pix: refPlane, Stride: stride, Width: width, Height: height,
	}, px+int(mv.Col)>>3, py+int(mv.Row)>>3, 16)
	start := mv
	center := bestSAD
	probe := st.sadScratch[:256]
	srcBlock := src[py*stride+px:]

	left := motion.Vector{Row: start.Row, Col: start.Col - 4}
	halfLeft := st.subpelExact16x16(probe, srcBlock, refPlane, stride, width, height, px, py, start, left)
	if halfLeft >= 0 && halfLeft < bestSAD {
		bestSAD, mv = halfLeft, left
	}
	right := motion.Vector{Row: start.Row, Col: start.Col + 4}
	halfRight := st.subpelExact16x16(probe, srcBlock, refPlane, stride, width, height, px, py, start, right)
	if halfRight >= 0 && halfRight < bestSAD {
		bestSAD, mv = halfRight, right
	}
	up := motion.Vector{Row: start.Row - 4, Col: start.Col}
	halfUp := st.subpelExact16x16(probe, srcBlock, refPlane, stride, width, height, px, py, start, up)
	if halfUp >= 0 && halfUp < bestSAD {
		bestSAD, mv = halfUp, up
	}
	down := motion.Vector{Row: start.Row + 4, Col: start.Col}
	halfDown := st.subpelExact16x16(probe, srcBlock, refPlane, stride, width, height, px, py, start, down)
	if halfDown >= 0 && halfDown < bestSAD {
		bestSAD, mv = halfDown, down
	}

	estX := subpelQuarterAxis(halfLeft, halfRight, center)
	estY := subpelQuarterAxis(halfUp, halfDown, center)
	qx, qy := estX&^1, estY&^1
	if qx != 0 || qy != 0 {
		cand := motion.Vector{Row: start.Row + int16(qy), Col: start.Col + int16(qx)}
		if cand != mv && cand != start {
			if s := st.subpelExact16x16(probe, srcBlock, refPlane, stride, width, height, px, py, start, cand); s >= 0 && s < bestSAD {
				bestSAD, mv = s, cand
			}
		}
	}
	qx, qy = (estX+1)&^1, (estY+1)&^1
	if qx != 0 || qy != 0 {
		cand := motion.Vector{Row: start.Row + int16(qy), Col: start.Col + int16(qx)}
		if cand != mv && cand != start {
			if s := st.subpelExact16x16(probe, srcBlock, refPlane, stride, width, height, px, py, start, cand); s >= 0 && s < bestSAD {
				bestSAD, mv = s, cand
			}
		}
	}
	return mv, bestSAD
}

func (st *lossyEncodeState) subpelRefine32x32(src, refPlane []byte, stride, width, height, px, py int, mv motion.Vector, bestSAD int) (motion.Vector, int) {
	st.prober.Init(frame.Plane{
		Pix: refPlane, Stride: stride, Width: width, Height: height,
	}, px+int(mv.Col)>>3, py+int(mv.Row)>>3, 32)
	start := mv
	center := bestSAD
	probe := st.sadScratch[:1024]
	srcBlock := src[py*stride+px:]

	left := motion.Vector{Row: start.Row, Col: start.Col - 4}
	halfLeft := st.subpelExact32x32(probe, srcBlock, refPlane, stride, width, height, px, py, start, left)
	if halfLeft >= 0 && halfLeft < bestSAD {
		bestSAD, mv = halfLeft, left
	}
	right := motion.Vector{Row: start.Row, Col: start.Col + 4}
	halfRight := st.subpelExact32x32(probe, srcBlock, refPlane, stride, width, height, px, py, start, right)
	if halfRight >= 0 && halfRight < bestSAD {
		bestSAD, mv = halfRight, right
	}
	up := motion.Vector{Row: start.Row - 4, Col: start.Col}
	halfUp := st.subpelExact32x32(probe, srcBlock, refPlane, stride, width, height, px, py, start, up)
	if halfUp >= 0 && halfUp < bestSAD {
		bestSAD, mv = halfUp, up
	}
	down := motion.Vector{Row: start.Row + 4, Col: start.Col}
	halfDown := st.subpelExact32x32(probe, srcBlock, refPlane, stride, width, height, px, py, start, down)
	if halfDown >= 0 && halfDown < bestSAD {
		bestSAD, mv = halfDown, down
	}

	estX := subpelQuarterAxis(halfLeft, halfRight, center)
	estY := subpelQuarterAxis(halfUp, halfDown, center)
	qx, qy := estX&^1, estY&^1
	if qx != 0 || qy != 0 {
		cand := motion.Vector{Row: start.Row + int16(qy), Col: start.Col + int16(qx)}
		if cand != mv && cand != start {
			if s := st.subpelExact32x32(probe, srcBlock, refPlane, stride, width, height, px, py, start, cand); s >= 0 && s < bestSAD {
				bestSAD, mv = s, cand
			}
		}
	}
	qx, qy = (estX+1)&^1, (estY+1)&^1
	if qx != 0 || qy != 0 {
		cand := motion.Vector{Row: start.Row + int16(qy), Col: start.Col + int16(qx)}
		if cand != mv && cand != start {
			if s := st.subpelExact32x32(probe, srcBlock, refPlane, stride, width, height, px, py, start, cand); s >= 0 && s < bestSAD {
				bestSAD, mv = s, cand
			}
		}
	}
	return mv, bestSAD
}

func (st *lossyEncodeState) subpelExact8x8(probe, srcBlock, refPlane []byte, stride, width, height, px, py int, startMV, cand motion.Vector) int {
	if !st.prober.Predict8x8(probe, motion.Vector{Row: cand.Row - startMV.Row, Col: cand.Col - startMV.Col}) {
		if err := predictInto(probe, refPlane, stride, width, height, px, py, 8, 8, cand, false, false); err != nil {
			return -1
		}
	}
	return sad8x8Dual(srcBlock, stride, probe, 8)
}

func (st *lossyEncodeState) subpelExact16x16(probe, srcBlock, refPlane []byte, stride, width, height, px, py int, startMV, cand motion.Vector) int {
	if !st.prober.Predict16x16(probe, motion.Vector{Row: cand.Row - startMV.Row, Col: cand.Col - startMV.Col}) {
		if err := predictInto(probe, refPlane, stride, width, height, px, py, 16, 16, cand, false, false); err != nil {
			return -1
		}
	}
	return sad16x16Dual(srcBlock, stride, probe, 16)
}

func (st *lossyEncodeState) subpelExact32x32(probe, srcBlock, refPlane []byte, stride, width, height, px, py int, startMV, cand motion.Vector) int {
	if !st.prober.Predict32x32(probe, motion.Vector{Row: cand.Row - startMV.Row, Col: cand.Col - startMV.Col}) {
		if err := predictInto(probe, refPlane, stride, width, height, px, py, 32, 32, cand, false, false); err != nil {
			return -1
		}
	}
	return sad32x32Dual(srcBlock, stride, probe, 32)
}

func subpelQuarterAxis(sl, sr, center int) int {
	if sl < 0 || sr < 0 {
		return 0
	}
	den := sl + sr - 2*center
	if den <= 0 {
		return 0
	}
	est := (sl - sr) * 2 / den // half-pel steps are 4 eighths
	if est > 4 {
		return 4
	}
	if est < -4 {
		return -4
	}
	return est
}

func (st *lossyEncodeState) subpelRefineGeneric(src, refPlane []byte, stride, width, height, px, py, n int, mv motion.Vector, bestSAD int) (motion.Vector, int) {
	// The probes sit within one pixel of the full-pel start, so geometry
	// validation hoists into the prober; blocks near the frame edge fall
	// back to the fully validated predictor per probe.
	st.prober.Init(frame.Plane{
		Pix: refPlane, Stride: stride, Width: width, Height: height,
	}, px+int(mv.Col)>>3, py+int(mv.Row)>>3, n)
	startMV := mv
	probe := st.sadScratch[:n*n]
	srcBlock := src[py*stride+px:]
	exact := func(cand motion.Vector) int {
		if !st.prober.Predict(probe, motion.Vector{Row: cand.Row - startMV.Row, Col: cand.Col - startMV.Col}) {
			if err := predictInto(probe, refPlane, stride, width, height, px, py, n, n, cand, false, false); err != nil {
				return -1
			}
		}
		switch n {
		case 8:
			return sad8x8Dual(srcBlock, stride, probe, n)
		case 16:
			return sad16x16Dual(srcBlock, stride, probe, n)
		case 32:
			return sad32x32Dual(srcBlock, stride, probe, n)
		default:
			return sadDualBlock(srcBlock, stride, probe, n, n)
		}
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
	estX := subpelQuarterAxis(half[0], half[1], center)
	estY := subpelQuarterAxis(half[2], half[3], center)
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
		return sad8x8(src[base:], ref[refBase:], stride, limit)
	case 16:
		return sad16x16(src[base:], ref[refBase:], stride)
	case 32:
		return sad32x32(src[base:], ref[refBase:], stride)
	case 64:
		return sad64x64(src[base:], ref[refBase:], stride)
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
	srcBlock := src[base : base+(bh-1)*stride+bw]
	refBlock := ref[refBase : refBase+(bh-1)*stride+bw]
	switch {
	case bw == 16 && bh == 8:
		return sad8x8(srcBlock, refBlock, stride, limit) +
			sad8x8(srcBlock[8:], refBlock[8:], stride, limit)
	case bw == 8 && bh == 16:
		return sad8x8(srcBlock, refBlock, stride, limit) +
			sad8x8(srcBlock[8*stride:], refBlock[8*stride:], stride, limit)
	case bw == 32 && bh == 16:
		return sad16x16(srcBlock, refBlock, stride) +
			sad16x16(srcBlock[16:], refBlock[16:], stride)
	case bw == 16 && bh == 32:
		return sad16x16(srcBlock, refBlock, stride) +
			sad16x16(srcBlock[16*stride:], refBlock[16*stride:], stride)
	}
	total := 0
	for r := range bh {
		row := r * stride
		for c := range bw {
			d := int(srcBlock[row+c]) - int(refBlock[row+c])
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

func sadDualBlock(src []byte, srcStride int, ref []byte, refStride int, n int) int {
	switch n {
	case 8:
		return sad8x8Dual(src, srcStride, ref, refStride)
	case 16:
		return sad16x16Dual(src, srcStride, ref, refStride)
	case 32:
		return sad32x32Dual(src, srcStride, ref, refStride)
	case 64:
		return sad32x32Dual(src, srcStride, ref, refStride) +
			sad32x32Dual(src[32:], srcStride, ref[32:], refStride) +
			sad32x32Dual(src[32*srcStride:], srcStride, ref[32*refStride:], refStride) +
			sad32x32Dual(src[32*srcStride+32:], srcStride, ref[32*refStride+32:], refStride)
	}
	total := 0
	for r := range n {
		srow := r * srcStride
		rrow := r * refStride
		for c := range n {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func sadRectDualBlock(src []byte, srcStride int, ref []byte, refStride int, bw, bh int) int {
	if bw == bh {
		return sadDualBlock(src, srcStride, ref, refStride, bw)
	}
	switch {
	case bw == 16 && bh == 8:
		return sad8x8Dual(src, srcStride, ref, refStride) +
			sad8x8Dual(src[8:], srcStride, ref[8:], refStride)
	case bw == 8 && bh == 16:
		return sad8x8Dual(src, srcStride, ref, refStride) +
			sad8x8Dual(src[8*srcStride:], srcStride, ref[8*refStride:], refStride)
	case bw == 32 && bh == 16:
		return sad16x16Dual(src, srcStride, ref, refStride) +
			sad16x16Dual(src[16:], srcStride, ref[16:], refStride)
	case bw == 16 && bh == 32:
		return sad16x16Dual(src, srcStride, ref, refStride) +
			sad16x16Dual(src[16*srcStride:], srcStride, ref[16*refStride:], refStride)
	}
	total := 0
	for r := range bh {
		srow := r * srcStride
		rrow := r * refStride
		for c := range bw {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
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
	if err := quantize.QuantizeBlockScaledB(qcoeff, h, tran[:n], h, w, h, q, ts); err != nil {
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

func (st *lossyEncodeState) chooseInter8x8TXType(src SourceFrame420, lumaPX, lumaPY int, dctDcode int64) transform.Type {
	if !st.armTrial() {
		return transform.TypeDCTDCT
	}
	hasChroma := !st.color.MonoChrome
	bestType := transform.TypeDCTDCT
	saveDcode, saveDskip, saveRcode := st.rdDcode, st.rdDskip, st.rdRcode
	baseTrialCDFs := &st.trial8x8CDFs
	baseTrialCDFs.save(&st.trialCDFs)
	bestCost := dctDcode << 7
	bestCost += st.trialTXBBitsInter8x8((*[64]int16)(st.lumaQ[:64]), transform.TypeDCTDCT)
	if hasChroma {
		bestCost += st.trialTXBBitsUV4x4((*[16]int16)(st.uQ[:16]))
		bestCost += st.trialTXBBitsUV4x4((*[16]int16)(st.vQ[:16]))
	}
	tmpY := st.lumaQ2[:64]
	tmpY64 := (*[64]int16)(tmpY)
	tmpU := st.lumaQ2[64:80]
	tmpV := st.lumaQ2[80:96]
	tmpU16 := (*[16]int16)(tmpU)
	tmpV16 := (*[16]int16)(tmpV)
	for _, typ := range [...]transform.Type{
		transform.TypeADSTDCT,
		transform.TypeDCTADST,
		transform.TypeADSTADST,
		transform.TypeIDTX,
	} {
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		lumaZero := st.prepareInterTXBTyped(src.Y, st.predY[:64], 8, src.YStride, lumaPX, lumaPY, 8, 8, st.yQuant, tmpY, typ)
		if typ != transform.TypeDCTDCT && lumaZero {
			continue
		}
		lumaDcode := st.rdDcode
		if lumaDcode<<7 >= bestCost {
			continue
		}
		baseTrialCDFs.restoreY(&st.trialCDFs)
		lumaBits := st.trialTXBBitsInter8x8(tmpY64, typ)
		if (lumaDcode<<7)+lumaBits >= bestCost {
			continue
		}
		cost := lumaDcode << 7
		cost += lumaBits
		if hasChroma {
			baseTrialCDFs.restoreUV(&st.trialCDFs)
			st.prepareInterTXBTyped(src.U, st.predU[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.uQuant, tmpU, typ)
			st.prepareInterTXBTyped(src.V, st.predV[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.vQuant, tmpV, typ)
			cost = st.rdDcode << 7
			cost += lumaBits
			cost += st.trialTXBBitsUV4x4(tmpU16)
			cost += st.trialTXBBitsUV4x4(tmpV16)
		}
		if cost < bestCost {
			bestCost = cost
			bestType = typ
			copy(st.lumaQ[:64], tmpY)
			if hasChroma {
				copy(st.uQ[:16], tmpU)
				copy(st.vQ[:16], tmpV)
			}
		}
	}
	st.rdDcode, st.rdDskip, st.rdRcode = saveDcode, saveDskip, saveRcode
	baseTrialCDFs.restore(&st.trialCDFs)
	return bestType
}

type coeffTrial8x8Snapshot struct {
	txbSkip4x4UV      entropy.CDF
	txbSkip8x8Y       entropy.CDF
	eobExtra4x4UV     [tile.EOBCoefContexts]entropy.CDF
	eobExtra8x8Y      [tile.EOBCoefContexts]entropy.CDF
	dcSignY0          entropy.CDF
	dcSignUV0         entropy.CDF
	coeffBR4x4UV      [tile.CoeffBRContexts]entropy.CDF
	coeffBR8x8Y       [tile.CoeffBRContexts]entropy.CDF
	coeffBase4x4UV    [tile.CoeffBaseContexts]entropy.CDF
	coeffBase8x8Y     [tile.CoeffBaseContexts]entropy.CDF
	coeffBaseEOB4x4UV [tile.EOBBaseContexts]entropy.CDF
	coeffBaseEOB8x8Y  [tile.EOBBaseContexts]entropy.CDF
	eobFlag16UV0      entropy.CDF
	eobFlag64Y0       entropy.CDF
}

func (s *coeffTrial8x8Snapshot) save(cdfs *tile.CoeffCDFs) {
	s.txbSkip4x4UV = cdfs.TXBSkip[0][0]
	s.txbSkip8x8Y = cdfs.TXBSkip[1][0]
	s.eobExtra4x4UV = cdfs.EOBExtra[0][tile.CoeffPlaneUV]
	s.eobExtra8x8Y = cdfs.EOBExtra[1][tile.CoeffPlaneY]
	s.dcSignY0 = cdfs.DCSign[tile.CoeffPlaneY][0]
	s.dcSignUV0 = cdfs.DCSign[tile.CoeffPlaneUV][0]
	s.coeffBR4x4UV = cdfs.CoeffBR[0][tile.CoeffPlaneUV]
	s.coeffBR8x8Y = cdfs.CoeffBR[1][tile.CoeffPlaneY]
	s.coeffBase4x4UV = cdfs.CoeffBase[0][tile.CoeffPlaneUV]
	s.coeffBase8x8Y = cdfs.CoeffBase[1][tile.CoeffPlaneY]
	s.coeffBaseEOB4x4UV = cdfs.CoeffBaseEOB[0][tile.CoeffPlaneUV]
	s.coeffBaseEOB8x8Y = cdfs.CoeffBaseEOB[1][tile.CoeffPlaneY]
	s.eobFlag16UV0 = cdfs.EOBFlag16[tile.CoeffPlaneUV][0]
	s.eobFlag64Y0 = cdfs.EOBFlag64[tile.CoeffPlaneY][0]
}

func (s *coeffTrial8x8Snapshot) restore(cdfs *tile.CoeffCDFs) {
	s.restoreUV(cdfs)
	s.restoreY(cdfs)
}

func (s *coeffTrial8x8Snapshot) restoreY(cdfs *tile.CoeffCDFs) {
	cdfs.TXBSkip[1][0] = s.txbSkip8x8Y
	cdfs.EOBExtra[1][tile.CoeffPlaneY] = s.eobExtra8x8Y
	cdfs.DCSign[tile.CoeffPlaneY][0] = s.dcSignY0
	cdfs.CoeffBR[1][tile.CoeffPlaneY] = s.coeffBR8x8Y
	cdfs.CoeffBase[1][tile.CoeffPlaneY] = s.coeffBase8x8Y
	cdfs.CoeffBaseEOB[1][tile.CoeffPlaneY] = s.coeffBaseEOB8x8Y
	cdfs.EOBFlag64[tile.CoeffPlaneY][0] = s.eobFlag64Y0
}

func (s *coeffTrial8x8Snapshot) restoreUV(cdfs *tile.CoeffCDFs) {
	cdfs.TXBSkip[0][0] = s.txbSkip4x4UV
	cdfs.EOBExtra[0][tile.CoeffPlaneUV] = s.eobExtra4x4UV
	cdfs.DCSign[tile.CoeffPlaneUV][0] = s.dcSignUV0
	cdfs.CoeffBR[0][tile.CoeffPlaneUV] = s.coeffBR4x4UV
	cdfs.CoeffBase[0][tile.CoeffPlaneUV] = s.coeffBase4x4UV
	cdfs.CoeffBaseEOB[0][tile.CoeffPlaneUV] = s.coeffBaseEOB4x4UV
	cdfs.EOBFlag16[tile.CoeffPlaneUV][0] = s.eobFlag16UV0
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
	if w == 4 && h == 4 && ctxReq.Plane == 0 && ctxReq.Size == tile.TransformSize4x4 && afterSkip != nil {
		qcoeff4 := (*[16]int16)(qcoeff)
		var txCDF *entropy.CDF
		txSymbol := 0
		ok := true
		if txb4x4NonZero(qcoeff4) {
			txCDF, txSymbol, ok = st.interTXCDFAndSymbol(ctxReq.Size, txType)
		}
		if ok {
			if coeffCtx == nil {
				return tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 1, H4: 1}, blockDims)
			result := tile.WriteCoefficientsTXB4x4Y2DContextTrustedTXTypeArray(st.w, &st.coeffCDFs, qcoeff4, txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
				return err
			}
		} else if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip); err != nil {
			return err
		}
	} else if w == 8 && h == 8 && ctxReq.Plane == 0 && ctxReq.Size == tile.TransformSize8x8 && afterSkip != nil {
		txCDF, txSymbol, ok := st.interTXCDFAndSymbol(ctxReq.Size, txType)
		if ok {
			if coeffCtx == nil {
				return tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 2, H4: 2}, blockDims)
			result := tile.WriteCoefficientsTXB8x8Y2DContextTrustedArray(st.w, &st.coeffCDFs, (*[64]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
				return err
			}
		} else if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip); err != nil {
			return err
		}
	} else if w == 16 && h == 16 && ctxReq.Plane == 0 && ctxReq.Size == tile.TransformSize16x16 && afterSkip != nil && txType == transform.TypeDCTDCT {
		txCDF, txSymbol, ok := st.interTXCDFAndSymbol(ctxReq.Size, txType)
		if ok {
			if coeffCtx == nil {
				return tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 4, H4: 4}, blockDims)
			result := tile.WriteCoefficientsTXB16x16Y2DContextTrustedArray(st.w, &st.coeffCDFs, (*[256]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
				return err
			}
		} else if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip); err != nil {
			return err
		}
	} else if w == 32 && h == 32 && ctxReq.Plane == 0 && ctxReq.Size == tile.TransformSize32x32 && afterSkip != nil && txType == transform.TypeDCTDCT {
		txCDF, txSymbol, ok := st.interTXCDFAndSymbol(ctxReq.Size, txType)
		if ok {
			if coeffCtx == nil {
				return tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 8, H4: 8}, blockDims)
			result := tile.WriteCoefficientsTXB32x32Y2DContextTrustedArray(st.w, &st.coeffCDFs, (*[1024]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
				return err
			}
		} else if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip); err != nil {
			return err
		}
	} else if w == 4 && h == 4 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize4x4 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 1, H4: 1}, blockDims)
		result := tile.WriteCoefficientsTXB4x4UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[16]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
			return err
		}
	} else if w == 8 && h == 8 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize8x8 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 2, H4: 2}, blockDims)
		result := tile.WriteCoefficientsTXB8x8UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[64]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
			return err
		}
	} else if w == 16 && h == 16 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize16x16 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 4, H4: 4}, blockDims)
		result := tile.WriteCoefficientsTXB16x16UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[256]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
			return err
		}
	} else if w == 32 && h == 32 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize32x32 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 8, H4: 8}, blockDims)
		result := tile.WriteCoefficientsTXB32x32UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[1024]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, result); err != nil {
			return err
		}
	} else {
		if _, err := tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip); err != nil {
			return err
		}
	}
	n := w * h
	dq := &st.dqScratch
	if err := quantize.DequantizeBlockScaledBitDepth(dq[:n], h, qcoeff, h, w, h, q, txScaleForSize(max(w, h)), 8); err != nil {
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

func txb4x4NonZero(coeffs *[16]int16) bool {
	var acc int16
	for _, c := range coeffs {
		acc |= c
	}
	return acc != 0
}

func (st *lossyEncodeState) interTXCDFAndSymbol(size tile.TransformSize, typ transform.Type) (*entropy.CDF, int, bool) {
	if size == tile.TransformSize8x8 {
		return st.inter8x8TXCDFAndSymbol(typ)
	}
	if st.qIndex == 0 {
		return nil, 0, typ == transform.TypeDCTDCT
	}
	set, err := tile.ExtTXSetTypeFor(size, true, false)
	if err != nil {
		return nil, 0, false
	}
	symbols, err := tile.ExtTXTypeCount(set)
	if err != nil {
		return nil, 0, false
	}
	if symbols <= 1 {
		return nil, 0, typ == transform.TypeDCTDCT
	}
	index, err := tile.ExtTXSetIndex(size, true, false)
	if err != nil {
		return nil, 0, false
	}
	square, err := tile.TransformSizeSquare(size)
	if err != nil {
		return nil, 0, false
	}
	cdf, err := st.txCDFs.InterCDF(index, square, symbols)
	if err != nil {
		return nil, 0, false
	}
	symbol, err := tile.ExtTXSymbolForType(set, typ)
	if err != nil {
		return nil, 0, false
	}
	return cdf, symbol, true
}

func (st *lossyEncodeState) inter8x8TXCDFAndSymbol(typ transform.Type) (*entropy.CDF, int, bool) {
	if st.qIndex == 0 {
		return nil, 0, typ == transform.TypeDCTDCT
	}
	symbol := 0
	switch typ {
	case transform.TypeIDTX:
		symbol = 0
	case transform.TypeDCTDCT:
		symbol = 7
	case transform.TypeADSTDCT:
		symbol = 8
	case transform.TypeDCTADST:
		symbol = 9
	case transform.TypeADSTADST:
		symbol = 12
	default:
		return nil, 0, false
	}
	cdf := &st.txCDFs.Inter[1][tile.TransformSize8x8]
	if cdf.Symbols() != 16 {
		return nil, 0, false
	}
	return cdf, symbol, true
}

// forwardTransformBlock dispatches the forward transform for every coded
// DCT_DCT shape and the small square hybrid transforms enabled in the realtime
// inter tx_type selector.
func forwardTransformBlock(tran []int32, residual []int16, scratch []int32, w, h int, typ transform.Type) error {
	if typ == transform.TypeDCTDCT {
		return forwardDCTBlock(tran, residual, w, h)
	}
	if w == 8 && h == 8 {
		switch typ {
		case transform.TypeADSTDCT:
			transform.ForwardBlock8x8ADSTDCTTrusted(tran, h, residual, w, scratch)
		case transform.TypeDCTADST:
			transform.ForwardBlock8x8DCTADSTTrusted(tran, h, residual, w, scratch)
		case transform.TypeADSTADST:
			transform.ForwardBlock8x8ADSTADSTTrusted(tran, h, residual, w, scratch)
		case transform.TypeIDTX:
			transform.ForwardBlock8x8IDTXTrusted(tran, h, residual, w, scratch)
		default:
			return transform.ErrInvalidTransform
		}
		return nil
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
	st.trialReady = true
	return true
}

// interHeaderCost prices one extra inter block's prefix symbols (mode,
// reference, vector residual) under the frame multiplier.
func (st *lossyEncodeState) interHeaderCost() int64 {
	return (int64(24) << 9) * st.rdMult >> 9
}

// trialInterCost8x8 prices coding one motion-compensated 8x8 block exactly:
// the true transform-quantize pass for distortion and the real coefficient
// coder for bits. The prediction lands in sadScratch.
func (st *lossyEncodeState) trialInterCost8x8(src SourceFrame420, ref SourceFrame420, px, py int, mv motion.Vector) int64 {
	pred := st.sadScratch[:64]
	if err := predictInto(pred, ref.Y, src.YStride, src.Width, src.Height, px, py, 8, 8, mv, false, false); err != nil {
		return 1 << 59
	}
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, pred, 8, src.YStride, px, py, 8, 8, st.yQuant, st.lumaQ2[:64])
	cost := st.trialTXBBitsY8x8((*[64]int16)(st.lumaQ2[:64])) + st.rdDcode<<7

	halfW, halfH := src.Width/2, src.Height/2
	cpred := st.sadScratch[:16]
	if err := predictInto(cpred, ref.U, src.ChromaStride, halfW, halfH, px/2, py/2, 4, 4, mv, true, true); err != nil {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(src.U, cpred, 4, src.ChromaStride, px/2, py/2, 4, 4, st.uQuant, st.uQ[:16])
	cost += st.trialTXBBitsUV4x4((*[16]int16)(st.uQ[:16])) + st.rdDcode<<7

	if err := predictInto(cpred, ref.V, src.ChromaStride, halfW, halfH, px/2, py/2, 4, 4, mv, true, true); err != nil {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(src.V, cpred, 4, src.ChromaStride, px/2, py/2, 4, 4, st.vQuant, st.vQ[:16])
	cost += st.trialTXBBitsUV4x4((*[16]int16)(st.vQ[:16])) + st.rdDcode<<7
	return cost
}

func (st *lossyEncodeState) trialInterCost16x16(src SourceFrame420, ref SourceFrame420, px, py int, mv motion.Vector) int64 {
	pred := st.sadScratch[:256]
	if err := predictInto(pred, ref.Y, src.YStride, src.Width, src.Height, px, py, 16, 16, mv, false, false); err != nil {
		return 1 << 59
	}
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, pred, 16, src.YStride, px, py, 16, 16, st.yQuant, st.lumaQ2[:256])
	cost := st.trialTXBBitsY16x16((*[256]int16)(st.lumaQ2[:256])) + st.rdDcode<<7

	halfW, halfH := src.Width/2, src.Height/2
	cpred := st.sadScratch[:64]
	if err := predictInto(cpred, ref.U, src.ChromaStride, halfW, halfH, px/2, py/2, 8, 8, mv, true, true); err != nil {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(src.U, cpred, 8, src.ChromaStride, px/2, py/2, 8, 8, st.uQuant, st.uQ[:64])
	cost += st.trialTXBBitsUV8x8((*[64]int16)(st.uQ[:64])) + st.rdDcode<<7

	if err := predictInto(cpred, ref.V, src.ChromaStride, halfW, halfH, px/2, py/2, 8, 8, mv, true, true); err != nil {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(src.V, cpred, 8, src.ChromaStride, px/2, py/2, 8, 8, st.vQuant, st.vQ[:64])
	cost += st.trialTXBBitsUV8x8((*[64]int16)(st.vQ[:64])) + st.rdDcode<<7
	return cost
}

// trialInterMergeWins prices the merged 16x16 against its four 8x8 children
// with real bits; the merge keeps three blocks' header savings.
func (st *lossyEncodeState) trialInterMergeWins(src SourceFrame420, ref SourceFrame420, px, py int) bool {
	if !st.armTrial() {
		return false
	}
	headerCost := st.interHeaderCost()
	idx8 := (py/8)*st.grid8Cols + px/8
	children := st.trialInterCost8x8(src, ref, px, py, st.mv8Grid[idx8]) + headerCost
	children += st.trialInterCost8x8(src, ref, px+8, py, st.mv8Grid[idx8+1]) + headerCost
	children += st.trialInterCost8x8(src, ref, px, py+8, st.mv8Grid[idx8+st.grid8Cols]) + headerCost
	children += st.trialInterCost8x8(src, ref, px+8, py+8, st.mv8Grid[idx8+st.grid8Cols+1]) + headerCost
	idx16 := (py/16)*st.grid16Cols + px/16
	merged := st.trialInterCost16x16(src, ref, px, py, st.mv16Grid[idx16]) + headerCost
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
	seedDX = minInt(maxInt(seedDX, -px), width-n-px) &^ 1
	seedDY = minInt(maxInt(seedDY, -py), height-n-py) &^ 1
	minDX := maxInt(seedDX-reach, -px)
	maxDX := minInt(seedDX+reach, width-n-px)
	minDY := maxInt(seedDY-reach, -py)
	maxDY := minInt(seedDY+reach, height-n-py)

	base := py*stride + px
	srcBlock := src[base:]
	switch n {
	case 8:
		return fullPelDiamondSearch8(srcBlock, ref, base, stride, minDX, maxDX, minDY, maxDY)
	case 16:
		return fullPelDiamondSearch16(srcBlock, ref, base, stride, minDX, maxDX, minDY, maxDY)
	case 32:
		return fullPelDiamondSearch32(srcBlock, ref, base, stride, minDX, maxDX, minDY, maxDY)
	case 64:
		return fullPelDiamondSearch64(srcBlock, ref, base, stride, minDX, maxDX, minDY, maxDY)
	default:
		return fullPelDiamondSearchGeneric(src, ref, base, stride, n, minDX, maxDX, minDY, maxDY)
	}
}

func fullPelDiamondSearch8(srcBlock []byte, ref []byte, base, stride, minDX, maxDX, minDY, maxDY int) (int, int, int) {
	bestDX, bestDY := 0, 0
	bestSAD := sad8x8(srcBlock, ref[base:], stride, 1<<30)
	// Static-content fast path: when zero motion is already a near-perfect
	// match (below ~2 per sample, the quantizer's noise floor at realtime
	// qindexes), searching cannot pay for its own cost.
	if bestSAD <= 8*8*2 {
		return 0, 0, bestSAD
	}
	// Coarse even-step raster with row-granular early exit, then a +-2 even
	// diamond refinement.
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		refRow := base + dy*stride
		dx := minDX &^ 1
		for ; dx+12 <= maxDX; dx += 16 {
			s0, s1, s2, s3 := sad8x8x4Step4(srcBlock, ref[refRow+dx:], stride)
			if s0 < bestSAD {
				bestSAD, bestDX, bestDY = s0, dx, dy
			}
			if s1 < bestSAD {
				bestSAD, bestDX, bestDY = s1, dx+4, dy
			}
			if s2 < bestSAD {
				bestSAD, bestDX, bestDY = s2, dx+8, dy
			}
			if s3 < bestSAD {
				bestSAD, bestDX, bestDY = s3, dx+12, dy
			}
		}
		for ; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sad8x8(srcBlock, ref[refRow+dx:], stride, bestSAD); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	for _, cand := range [4][2]int{{bestDX + 2, bestDY}, {bestDX - 2, bestDY}, {bestDX, bestDY + 2}, {bestDX, bestDY - 2}} {
		dx, dy := cand[0], cand[1]
		if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
			continue
		}
		if s := sad8x8(srcBlock, ref[base+dy*stride+dx:], stride, bestSAD); s < bestSAD {
			bestSAD, bestDX, bestDY = s, dx, dy
		}
	}
	return bestDX, bestDY, bestSAD
}

func fullPelDiamondSearch16(srcBlock []byte, ref []byte, base, stride, minDX, maxDX, minDY, maxDY int) (int, int, int) {
	bestDX, bestDY := 0, 0
	bestSAD := sad16x16(srcBlock, ref[base:], stride)
	if bestSAD <= 16*16*2 {
		return 0, 0, bestSAD
	}
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		refRow := base + dy*stride
		dx := minDX &^ 1
		for ; dx+12 <= maxDX; dx += 16 {
			s0, s1, s2, s3 := sad16x16x4Step4(srcBlock, ref[refRow+dx:], stride)
			if s0 < bestSAD {
				bestSAD, bestDX, bestDY = s0, dx, dy
			}
			if s1 < bestSAD {
				bestSAD, bestDX, bestDY = s1, dx+4, dy
			}
			if s2 < bestSAD {
				bestSAD, bestDX, bestDY = s2, dx+8, dy
			}
			if s3 < bestSAD {
				bestSAD, bestDX, bestDY = s3, dx+12, dy
			}
		}
		for ; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sad16x16(srcBlock, ref[refRow+dx:], stride); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	for _, cand := range [4][2]int{{bestDX + 2, bestDY}, {bestDX - 2, bestDY}, {bestDX, bestDY + 2}, {bestDX, bestDY - 2}} {
		dx, dy := cand[0], cand[1]
		if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
			continue
		}
		if s := sad16x16(srcBlock, ref[base+dy*stride+dx:], stride); s < bestSAD {
			bestSAD, bestDX, bestDY = s, dx, dy
		}
	}
	return bestDX, bestDY, bestSAD
}

func fullPelDiamondSearch32(srcBlock []byte, ref []byte, base, stride, minDX, maxDX, minDY, maxDY int) (int, int, int) {
	bestDX, bestDY := 0, 0
	bestSAD := sad32x32(srcBlock, ref[base:], stride)
	if bestSAD <= 32*32*2 {
		return 0, 0, bestSAD
	}
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		refRow := base + dy*stride
		dx := minDX &^ 1
		for ; dx+12 <= maxDX; dx += 16 {
			s0, s1, s2, s3 := sad32x32x4Step4(srcBlock, ref[refRow+dx:], stride)
			if s0 < bestSAD {
				bestSAD, bestDX, bestDY = s0, dx, dy
			}
			if s1 < bestSAD {
				bestSAD, bestDX, bestDY = s1, dx+4, dy
			}
			if s2 < bestSAD {
				bestSAD, bestDX, bestDY = s2, dx+8, dy
			}
			if s3 < bestSAD {
				bestSAD, bestDX, bestDY = s3, dx+12, dy
			}
		}
		for ; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sad32x32(srcBlock, ref[refRow+dx:], stride); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	refineDX, refineDY := bestDX, bestDY
	if refineDX+2 <= maxDX && refineDX-2 >= minDX && refineDY+2 <= maxDY && refineDY-2 >= minDY {
		s0, s1, s2, s3 := sad32x32x4(srcBlock,
			ref[base+refineDY*stride+refineDX+2:],
			ref[base+refineDY*stride+refineDX-2:],
			ref[base+(refineDY+2)*stride+refineDX:],
			ref[base+(refineDY-2)*stride+refineDX:],
			stride)
		if s0 < bestSAD {
			bestSAD, bestDX, bestDY = s0, refineDX+2, refineDY
		}
		if s1 < bestSAD {
			bestSAD, bestDX, bestDY = s1, refineDX-2, refineDY
		}
		if s2 < bestSAD {
			bestSAD, bestDX, bestDY = s2, refineDX, refineDY+2
		}
		if s3 < bestSAD {
			bestSAD, bestDX, bestDY = s3, refineDX, refineDY-2
		}
	} else {
		for _, cand := range [4][2]int{{refineDX + 2, refineDY}, {refineDX - 2, refineDY}, {refineDX, refineDY + 2}, {refineDX, refineDY - 2}} {
			dx, dy := cand[0], cand[1]
			if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
				continue
			}
			if s := sad32x32(srcBlock, ref[base+dy*stride+dx:], stride); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	return bestDX, bestDY, bestSAD
}

func fullPelDiamondSearch64(srcBlock []byte, ref []byte, base, stride, minDX, maxDX, minDY, maxDY int) (int, int, int) {
	bestDX, bestDY := 0, 0
	bestSAD := sad64x64(srcBlock, ref[base:], stride)
	if bestSAD <= 64*64*2 {
		return 0, 0, bestSAD
	}
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		refRow := base + dy*stride
		dx := minDX &^ 1
		for ; dx+12 <= maxDX; dx += 16 {
			s0, s1, s2, s3 := sad64x64x4Step4(srcBlock, ref[refRow+dx:], stride)
			if s0 < bestSAD {
				bestSAD, bestDX, bestDY = s0, dx, dy
			}
			if s1 < bestSAD {
				bestSAD, bestDX, bestDY = s1, dx+4, dy
			}
			if s2 < bestSAD {
				bestSAD, bestDX, bestDY = s2, dx+8, dy
			}
			if s3 < bestSAD {
				bestSAD, bestDX, bestDY = s3, dx+12, dy
			}
		}
		for ; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sad64x64(srcBlock, ref[refRow+dx:], stride); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	refineDX, refineDY := bestDX, bestDY
	if refineDX+2 <= maxDX && refineDX-2 >= minDX && refineDY+2 <= maxDY && refineDY-2 >= minDY {
		s0, s1, s2, s3 := sad64x64x4(srcBlock,
			ref[base+refineDY*stride+refineDX+2:],
			ref[base+refineDY*stride+refineDX-2:],
			ref[base+(refineDY+2)*stride+refineDX:],
			ref[base+(refineDY-2)*stride+refineDX:],
			stride)
		if s0 < bestSAD {
			bestSAD, bestDX, bestDY = s0, refineDX+2, refineDY
		}
		if s1 < bestSAD {
			bestSAD, bestDX, bestDY = s1, refineDX-2, refineDY
		}
		if s2 < bestSAD {
			bestSAD, bestDX, bestDY = s2, refineDX, refineDY+2
		}
		if s3 < bestSAD {
			bestSAD, bestDX, bestDY = s3, refineDX, refineDY-2
		}
	} else {
		for _, cand := range [4][2]int{{refineDX + 2, refineDY}, {refineDX - 2, refineDY}, {refineDX, refineDY + 2}, {refineDX, refineDY - 2}} {
			dx, dy := cand[0], cand[1]
			if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
				continue
			}
			if s := sad64x64(srcBlock, ref[base+dy*stride+dx:], stride); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	return bestDX, bestDY, bestSAD
}

func fullPelDiamondSearchGeneric(src, ref []byte, base, stride, n, minDX, maxDX, minDY, maxDY int) (int, int, int) {
	bestDX, bestDY := 0, 0
	bestSAD := sadBlock(src, ref, base, base, stride, n, 1<<30)
	if bestSAD <= n*n*2 {
		return 0, 0, bestSAD
	}
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		refRow := base + dy*stride
		for dx := minDX &^ 1; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sadBlock(src, ref, base, refRow+dx, stride, n, bestSAD); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	for _, cand := range [4][2]int{{bestDX + 2, bestDY}, {bestDX - 2, bestDY}, {bestDX, bestDY + 2}, {bestDX, bestDY - 2}} {
		dx, dy := cand[0], cand[1]
		if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
			continue
		}
		if s := sadBlock(src, ref, base, base+dy*stride+dx, stride, n, bestSAD); s < bestSAD {
			bestSAD, bestDX, bestDY = s, dx, dy
		}
	}
	return bestDX, bestDY, bestSAD
}
