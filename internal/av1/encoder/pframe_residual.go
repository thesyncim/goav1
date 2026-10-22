package encoder

import (
	"encoding/binary"
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
	// ownHME is the frame ME of coders not driven by a VideoEncoder.
	ownHME               *hmeState
	st                   lossyEncodeState
	effortLevel          int8
	partCDFs             tile.PartitionCDFs
	refCDFs              tile.InterRefCDFs
	modeCDFs             tile.InterModeCDFs
	interpCDFs           tile.InterpFilterCDFs
	scratch              tile.BlockLoopScratch
	carrier              tile.BlockLoopContextCarrier
	writerBuf            []byte
	writer               entropy.Writer
	decisionStats        EncoderDecisionStats
	decisionStatsEnabled bool

	// forceSplit runs the decision/write split pipeline (pframe_split.go) even
	// when it cannot parallelize, so the byte-identity oracle can compare the
	// serial split against the fused walk. In production the split is used
	// ONLY when the SB-row wavefront will actually run (>=2 lanes on the
	// single-tile coder); serial and multi-tile coders take the fused
	// single-pass walk, which records nothing and replays nothing.
	forceSplit bool
	// splitRowRecs holds one decision record per superblock row so wavefront
	// lanes fill disjoint records; the serial decision pass uses the same
	// layout and the write pass replays rows in raster order either way.
	splitRowRecs []pframeSplitRecord
	// wavefront, when non-nil with wavefrontLanes >= 2, runs the decision
	// pass as an SB-row wavefront (pframe_wavefront.go). Only the single-tile
	// coder gets one: tile columns already parallelize the multi-tile path.
	wavefront      *pframeWavefront
	wavefrontLanes int

	// cdefLog, when non-nil, makes the tile writer log its symbol stream
	// there (entropy WriterLog) so the frame driver can splice in the
	// cdef_idx literals the SVT CDEF search picks after the frame is
	// reconstructed (cdef_frame.go). The driver owns the logs per tile.
	cdefLog *entropy.WriterLog
}

const (
	sadCacheBits      = 20
	sadCacheValueMask = (uint32(1) << sadCacheBits) - 1
	sadCacheMaxEpoch  = (uint32(1) << (32 - sadCacheBits)) - 1
)

type realtimeSourceSAD uint8

const (
	realtimeSourceSADZero realtimeSourceSAD = iota
	realtimeSourceSADVeryLow
	realtimeSourceSADLow
	realtimeSourceSADMed
	realtimeSourceSADHigh
)

const realtimeSourceVarianceUnknown = ^uint32(0)

type realtimeContentStateSB struct {
	sourceSADNonRD realtimeSourceSAD
	sourceSADRD    realtimeSourceSAD
	lightingChange bool
	lowSumDiff     bool
}

func defaultRealtimeContentStateSB() realtimeContentStateSB {
	return realtimeContentStateSB{
		sourceSADNonRD: realtimeSourceSADMed,
		sourceSADRD:    realtimeSourceSADMed,
	}
}

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

func realtimeSourceContentSB(src, last SourceFrame420, px, py int, increaseSourceSADThresh bool) realtimeContentStateSB {
	state := defaultRealtimeContentStateSB()
	if src.Width != last.Width || src.Height != last.Height ||
		px < 0 || py < 0 || px+64 > src.Width || py+64 > src.Height {
		return state
	}
	srcOff := py*src.YStride + px
	lastOff := py*last.YStride + px
	tmpSSE, tmpVariance := sseVariance64x64(src.Y[srcOff:], src.YStride, last.Y[lastOff:], last.YStride)
	veryLowThresh := uint64(10000)
	lowThreshNonRD := uint64(100000)
	lowThreshRD := uint64(36000)
	highThresh := uint64(1000000)
	if increaseSourceSADThresh {
		highThresh <<= 1
		lowThreshNonRD <<= 1
		veryLowThresh <<= 1
	}
	if uint64(tmpSSE) < lowThreshRD {
		state.sourceSADRD = realtimeSourceSADLow
	}
	if tmpSSE == 0 {
		state.sourceSADNonRD = realtimeSourceSADZero
		return state
	}
	switch {
	case uint64(tmpSSE) < veryLowThresh:
		state.sourceSADNonRD = realtimeSourceSADVeryLow
	case uint64(tmpSSE) < lowThreshNonRD:
		state.sourceSADNonRD = realtimeSourceSADLow
	case uint64(tmpSSE) > highThresh:
		state.sourceSADNonRD = realtimeSourceSADHigh
	}
	sumSqThresh := uint32(10000)
	sumSq := tmpSSE - tmpVariance
	state.lightingChange = tmpVariance < tmpSSE>>1 && sumSq > sumSqThresh
	state.lowSumDiff = sumSq < sumSqThresh>>1
	return state
}

func (st *lossyEncodeState) prepareRealtimeSourceContent(src, last SourceFrame420, gridCols, gridRows int, miColStart, miColEnd uint16) {
	st.sourceContentCols = gridCols
	need := gridCols * gridRows
	if need <= 0 {
		return
	}
	if len(st.sourceContentGrid) < need {
		st.sourceContentGrid = make([]realtimeContentStateSB, need)
	}
	if len(st.sourceVarianceGrid) < need {
		st.sourceVarianceGrid = make([]uint32, need)
	}
	for i := range st.sourceContentGrid[:need] {
		st.sourceContentGrid[i] = defaultRealtimeContentStateSB()
		st.sourceVarianceGrid[i] = realtimeSourceVarianceUnknown
	}
	if src.Width != last.Width || src.Height != last.Height {
		return
	}
	startCol := (int(miColStart) * 4) / 64
	endCol := (int(miColEnd)*4 + 63) / 64
	if startCol < 0 {
		startCol = 0
	}
	if endCol > gridCols {
		endCol = gridCols
	}
	for py := 0; py+64 <= src.Height; py += 64 {
		row := py / 64
		if row >= gridRows {
			break
		}
		for col := startCol; col < endCol; col++ {
			px := col * 64
			if px+64 > src.Width {
				continue
			}
			idx := row*gridCols + col
			state := realtimeSourceContentSB(src, last, px, py, false)
			st.sourceContentGrid[idx] = state
			if state.sourceSADNonRD > realtimeSourceSADLow {
				st.sourceVarianceGrid[idx] = realtimeSourceVariancePerPixel(src.Y[py*src.YStride+px:], src.YStride, 64, 64)
			}
		}
	}
}

func realtimeStatic64METhreshold(effort int8) int {
	speed := realtimeLibaomSpeedForEffort(effort)
	if speed <= 12 {
		return 64 * 64
	}
	return 2 * 64 * 64
}

func (st *lossyEncodeState) prepareStatic64MEBypass(src, ref SourceFrame420, gridRows int, miColStart, miColEnd uint16) {
	if st == nil || st.grid64Cols <= 0 || gridRows <= 0 || st.sadCacheEpoch == 0 ||
		src.Width != ref.Width || src.Height != ref.Height ||
		src.Width <= 0 || src.Height <= 0 ||
		len(st.mv64Grid) < st.grid64Cols*gridRows ||
		len(st.sad64Grid) < st.grid64Cols*gridRows {
		return
	}
	threshold := realtimeStatic64METhreshold(st.effortLevel)
	if threshold <= 0 {
		return
	}
	startCol := (int(miColStart) * 4) / 64
	endCol := (int(miColEnd)*4 + 63) / 64
	if startCol < 0 {
		startCol = 0
	}
	if endCol > st.grid64Cols {
		endCol = st.grid64Cols
	}
	zero := motion.Vector{}
	for py := 0; py+64 <= src.Height; py += 64 {
		row64 := py / 64
		if row64 >= gridRows {
			break
		}
		for col64 := startCol; col64 < endCol; col64++ {
			px := col64 * 64
			if px+64 > src.Width {
				continue
			}
			base := py*src.YStride + px
			sad64 := sad64x64(src.Y[base:], ref.Y[base:], src.YStride)
			if sad64 >= threshold {
				continue
			}
			idx64 := row64*st.grid64Cols + col64
			st.mv64Grid[idx64] = zero
			st.sad64Grid[idx64] = sadCachePack(st.sadCacheEpoch, sad64)
			st.fillStatic64ChildSADCaches(src, ref, px, py)
		}
	}
}

func (st *lossyEncodeState) fillStatic64ChildSADCaches(src, ref SourceFrame420, px, py int) {
	zero := motion.Vector{}
	if st.grid32Cols > 0 && len(st.mv32Grid) > 0 && len(st.sad32Grid) == len(st.mv32Grid) {
		for dy := 0; dy < 64; dy += 32 {
			for dx := 0; dx < 64; dx += 32 {
				x, y := px+dx, py+dy
				idx := (y/32)*st.grid32Cols + x/32
				if uint(idx) >= uint(len(st.sad32Grid)) {
					continue
				}
				base := y*src.YStride + x
				st.mv32Grid[idx] = zero
				st.sad32Grid[idx] = sadCachePack(st.sadCacheEpoch, sad32x32(src.Y[base:], ref.Y[base:], src.YStride))
			}
		}
	}
	if st.grid16Cols > 0 && len(st.mv16Grid) > 0 && len(st.sad16Grid) == len(st.mv16Grid) {
		for dy := 0; dy < 64; dy += 16 {
			for dx := 0; dx < 64; dx += 16 {
				x, y := px+dx, py+dy
				idx := (y/16)*st.grid16Cols + x/16
				if uint(idx) >= uint(len(st.sad16Grid)) {
					continue
				}
				base := y*src.YStride + x
				st.mv16Grid[idx] = zero
				st.sad16Grid[idx] = sadCachePack(st.sadCacheEpoch, sad16x16(src.Y[base:], ref.Y[base:], src.YStride))
			}
		}
	}
	if st.grid8Cols > 0 && len(st.mv8Grid) > 0 && len(st.sad8Grid) == len(st.mv8Grid) {
		for dy := 0; dy < 64; dy += 8 {
			for dx := 0; dx < 64; dx += 8 {
				x, y := px+dx, py+dy
				idx := (y/8)*st.grid8Cols + x/8
				if uint(idx) >= uint(len(st.sad8Grid)) {
					continue
				}
				base := y*src.YStride + x
				st.mv8Grid[idx] = zero
				st.sad8Grid[idx] = sadCachePack(st.sadCacheEpoch, sad8x8(src.Y[base:], ref.Y[base:], src.YStride, 1<<30))
			}
		}
	}
}

func (st *lossyEncodeState) realtimeContentStateForBlock(px, py int) realtimeContentStateSB {
	if st == nil || st.sourceContentCols <= 0 || px < 0 || py < 0 {
		return defaultRealtimeContentStateSB()
	}
	col, row := px/64, py/64
	idx := row*st.sourceContentCols + col
	if idx < 0 || idx >= len(st.sourceContentGrid) {
		return defaultRealtimeContentStateSB()
	}
	return st.sourceContentGrid[idx]
}

func realtimeSourceVariancePerPixel(src []byte, stride, w, h int) uint32 {
	if variance, ok := realtimeSourceVarianceAgainstAV1VarOffs(src, stride, w, h); ok {
		shift := uint(bits.TrailingZeros(uint(w * h)))
		return roundPowerOfTwoUint32(variance, shift)
	}
	sse, sum := pixelStats128PureGo(src, stride, w, h)
	shift := uint(bits.TrailingZeros(uint(w * h)))
	variance := varianceFromStats(sse, sum, shift)
	return roundPowerOfTwoUint32(variance, shift)
}

var realtimeAV1VarOffs = [128]byte{
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128,
}

func realtimeSourceVarianceAgainstAV1VarOffs(src []byte, stride, w, h int) (uint32, bool) {
	ref := realtimeAV1VarOffs[:]
	switch {
	case w == 64 && h == 64:
		_, variance := sseVariance64x64(src, stride, ref, 0)
		return variance, true
	case w == 32 && h == 32:
		_, variance := sseVariance32x32(src, stride, ref, 0)
		return variance, true
	case w == 16 && h == 16:
		_, variance := sseVariance16x16(src, stride, ref, 0)
		return variance, true
	case w == 8 && h == 8:
		_, variance := sseVariance8x8(src, stride, ref, 0)
		return variance, true
	default:
		return 0, false
	}
}

func pixelStats128PureGo(src []byte, stride, w, h int) (sse uint32, sum int32) {
	total := 0
	sumInt := 0
	for r := range h {
		row := r * stride
		for c := range w {
			d := int(src[row+c]) - 128
			sumInt += d
			total += d * d
		}
	}
	return uint32(total), int32(sumInt)
}

func roundPowerOfTwoUint32(v uint32, shift uint) uint32 {
	if shift == 0 {
		return v
	}
	return (v + (1 << (shift - 1))) >> shift
}

var realtimeAvg8x8Impl = realtimeAvg8x8Wide
var realtimeAvg8x8QuadImpl = realtimeAvg8x8QuadWide

// realtimeAvg8x8Wide sums the 64 samples of an 8x8 block with SWAR: each row
// is one little-endian uint64, its even and odd bytes are split into 16-bit
// lanes, and the lanes accumulate over the eight rows. A lane holds at most
// 8*510 = 4080, so no carry crosses lanes and the sum is exact.
func realtimeAvg8x8Wide(src []byte, stride int) int {
	const lowBytes = 0x00ff00ff00ff00ff
	_ = src[7*stride+7]
	r1 := src[stride:]
	r2 := src[2*stride:]
	r3 := src[3*stride:]
	r4 := src[4*stride:]
	r5 := src[5*stride:]
	r6 := src[6*stride:]
	r7 := src[7*stride:]
	acc := rowPairSum(binary.LittleEndian.Uint64(src), lowBytes) +
		rowPairSum(binary.LittleEndian.Uint64(r1), lowBytes) +
		rowPairSum(binary.LittleEndian.Uint64(r2), lowBytes) +
		rowPairSum(binary.LittleEndian.Uint64(r3), lowBytes) +
		rowPairSum(binary.LittleEndian.Uint64(r4), lowBytes) +
		rowPairSum(binary.LittleEndian.Uint64(r5), lowBytes) +
		rowPairSum(binary.LittleEndian.Uint64(r6), lowBytes) +
		rowPairSum(binary.LittleEndian.Uint64(r7), lowBytes)
	sum := acc&0xffff + (acc>>16)&0xffff + (acc>>32)&0xffff + acc>>48
	return int(sum+32) >> 6
}

// rowPairSum returns the four 16-bit lanes of w's even and odd byte pairs.
func rowPairSum(w, lowBytes uint64) uint64 {
	return (w & lowBytes) + ((w >> 8) & lowBytes)
}

func realtimeAvg8x8QuadWide(src []byte, stride int) (int, int, int, int) {
	return realtimeAvg8x8Wide(src, stride),
		realtimeAvg8x8Wide(src[8:], stride),
		realtimeAvg8x8Wide(src[8*stride:], stride),
		realtimeAvg8x8Wide(src[8*stride+8:], stride)
}

func realtimeAvg8x8(src []byte, stride int) int {
	return realtimeAvg8x8Impl(src, stride)
}

func realtimeAvg8x8PureGo(src []byte, stride int) int {
	sum := 0
	for y := range 8 {
		row := y * stride
		for x := range 8 {
			sum += int(src[row+x])
		}
	}
	return (sum + 32) >> 6
}

func realtimeAvg8x8Quad(src []byte, stride int) (int, int, int, int) {
	return realtimeAvg8x8QuadImpl(src, stride)
}

func realtimeAvg8x8QuadPureGo(src []byte, stride int) (int, int, int, int) {
	return realtimeAvg8x8PureGo(src, stride),
		realtimeAvg8x8PureGo(src[8:], stride),
		realtimeAvg8x8PureGo(src[8*stride:], stride),
		realtimeAvg8x8PureGo(src[8*stride+8:], stride)
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
	st.effortLevel = pc.effortLevel
	if prev != nil {
		pc.partCDFs = prev.Partition
		pc.refCDFs = prev.InterRef
		pc.modeCDFs = prev.InterMode
		pc.interpCDFs = prev.Interp
		st.modeCDFs = prev.Mode
		st.intraCDFs = prev.Intra
		st.coeffCDFs = prev.Coeff
		st.txCDFs = prev.TransformType
		st.treeCDFs = prev.Transform
		st.mvCDFs = prev.MV
		st.deltaQCDF = prev.DeltaQ
	} else {
		if err := st.deltaQCDF.InitDefaultDelta(); err != nil {
			return err
		}
		if err := pc.partCDFs.InitDefault(); err != nil {
			return err
		}
		if err := pc.refCDFs.InitDefault(); err != nil {
			return err
		}
		if err := pc.modeCDFs.InitDefault(); err != nil {
			return err
		}
		if err := pc.interpCDFs.InitDefault(); err != nil {
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
	st.ensureSubpelProbeCache()
	if err := st.initScans(); err != nil {
		return err
	}
	if cap(pc.writerBuf) == 0 {
		pc.writerBuf = make([]byte, 1<<18)
	}
	pc.resetCarrierEdges(rootCols)
	return nil
}

// resetCarrierEdges clears the tile's above/left edge carriers so a fresh
// block-loop walk starts from decode-reset contexts. The split pipeline calls
// it again between the decision and write passes: the write walk must rebuild
// the contexts from scratch exactly as the fused single pass did.
func (pc *pframeCoder) resetCarrierEdges(rootCols int) {
	if len(pc.carrier.Above) < rootCols {
		pc.carrier.Above = make([]tile.BlockLoopRootAboveContext, rootCols)
	}
	pc.carrier.Left = tile.BlockLoopRootLeftContext{}
	for i := range pc.carrier.Above[:rootCols] {
		pc.carrier.Above[i] = tile.BlockLoopRootAboveContext{}
	}
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
		{&st.scan4x16, 4, 16}, {&st.scan16x4, 16, 4},
		{&st.scan8x32, 8, 32}, {&st.scan32x8, 32, 8},
		{&st.scan16x64, 16, 64}, {&st.scan64x16, 64, 16},
	} {
		size := transform.Size{Width: rs.w, Height: rs.h}
		scanSize, err := transform.ScanSize(size)
		if err != nil {
			return err
		}
		total := int(scanSize.Width) * int(scanSize.Height)
		*rs.dst = make([]int16, total)
		inv := make([]int16, total)
		if err := transform.FillDefaultScan(*rs.dst, inv, size, transform.Class2D); err != nil {
			return err
		}
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize32x32)
	if err != nil {
		return err
	}
	st.levels = make([]uint8, scratchLen)
	st.invScratch = make([]int32, 4096)
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
	case tile.TransformSize32x32, tile.TransformSize64x64, tile.TransformSize64x32, tile.TransformSize32x64:
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

//go:noinline
func (st *lossyEncodeState) scanForThinTransformSize(size tile.TransformSize) ([]int16, bool) {
	switch size {
	case tile.TransformSize4x16:
		return st.scan4x16, true
	case tile.TransformSize16x4:
		return st.scan16x4, true
	case tile.TransformSize8x32:
		return st.scan8x32, true
	case tile.TransformSize32x8:
		return st.scan32x8, true
	case tile.TransformSize16x64:
		return st.scan16x64, true
	case tile.TransformSize64x16:
		return st.scan64x16, true
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
	dst.Interp = pc.interpCDFs
	dst.Mode = pc.st.modeCDFs
	dst.Intra = pc.st.intraCDFs
	dst.Coeff = pc.st.coeffCDFs
	dst.TransformType = pc.st.txCDFs
	dst.Transform = pc.st.treeCDFs
	dst.MV = pc.st.mvCDFs
	dst.DeltaQ = pc.st.deltaQCDF
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
	pc.st.cntZeroMV = 0
	if qIndex == 0 {
		return pc.encodeLosslessIntraInterTile(src, recon, prev, forceIntegerMV, allowScreenContentTools, color, miColStart, miColEnd)
	}
	miCols := uint16(src.Width / 4)
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB
	if err := pc.reset(qIndex, rootCols, prev, color); err != nil {
		return nil, err
	}
	st := &pc.st
	st.sbQIndex, st.sbQCols = st.sbQPending, st.sbQPendingCols
	st.sbQPending = nil
	st.beginLPD1Tile(miColStart)
	st.forceIntegerMV = forceIntegerMV
	st.allowScreenContentTools = allowScreenContentTools
	st.transformMode = parser.TransformModeSwitchable
	// Freeze the MDS0 mode/DRL rate tables from the frame-start CDF state
	// (SVT md_rate_estimation.c svt_aom_estimate_syntax_rate).
	if st.mds0Level != 0 {
		pc.modeCDFs.EstimateRateTables(&st.mds0Rates)
		st.mds0IntraOK = st.intraCDFs.EstimateIntraDCRateTables(&st.mds0IntraRates) == nil
	}
	st.decisionStats = nil
	if pc.decisionStatsEnabled {
		pc.decisionStats.Reset()
		pc.decisionStats.Tiles = 1
		st.decisionStats = &pc.decisionStats
	}

	pc.writer.Reset(pc.writerBuf[:0])
	pc.armCDEFLog()
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
	// contexts; they re-arm CDFs lazily on first use each frame. Inter tiles
	// keep the trial tables frozen at their armed state across blocks so
	// trial-priced decisions are independent of coding order (wavefront
	// requirement; see trialBlockLocal).
	st.trialReady = false
	st.skipRateReady = false
	st.trialBlockLocal = true
	st.keyMIColEnd = uint32(miColEnd)
	st.keyMIRowEnd = uint32(miRows)
	st.keyVisW, st.keyVisH = src.Width, src.Height

	// av1_compute_rd_mult_based_on_qindex, inter shape at 8-bit: the DC step
	// squared scaled by the rate multiplier; RDCOST pairs it with rates in
	// 512-units-per-bit and distortion shifted by RDDIV_BITS.
	st.setInterLambda()
	st.quantQ = qIndex
	st.baseQAC = st.yQuant.AC
	st.deltaQCur = qIndex
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
	st.prepareRealtimeSourceContent(src, ref, st.grid64Cols, grid64Rows, miColStart, miColEnd)
	st.prepareStatic64MEBypass(src, ref, grid64Rows, miColStart, miColEnd)
	st.prepareSVTPD0(st.grid64Cols, grid64Rows)
	st.pd0Rates.estimate(&pc.partCDFs)
	// Coders driven without a VideoEncoder frame ME (the standalone P-frame
	// API, monochrome) run the open-loop ME here, with the reference
	// reconstruction standing in for the previous source.
	if !scaledReference && (st.hme == nil || st.hme == pc.ownHME) {
		if pc.ownHME == nil {
			pc.ownHME = new(hmeState)
		}
		pc.ownHME.primeFromRecon(ref)
		pc.ownHME.runSerial(src, true, 0, -1)
		st.hme = pc.ownHME
	}
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		partition, err := st.decideRealtimePartition(src, ref, scaledReference, level, miCol, miRow, haveRight, haveBottom)
		if err == nil && st.decisionStats != nil {
			st.decisionStats.notePartition(level, partition)
		}
		return partition, err
	}

	// The split pipeline only earns its record/replay overhead when the
	// decision pass runs in parallel; a serial or multi-tile coder does strictly
	// more work for no benefit. Use the fused single-pass walk unless the SB-row
	// wavefront will actually run (>=2 lanes), or the oracle forces the split.
	wavefrontActive := pc.wavefront != nil && pc.wavefrontLanes >= 2 && st.decisionStats == nil
	if !wavefrontActive && !pc.forceSplit {
		visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
			return st.encodePBlock(src, ref, golden, recon, block, scratch, &pc.refCDFs, &pc.modeCDFs, &pc.interpCDFs, referenceMode, walkReq, miCols, miRows)
		}
		st.bindRefmvs(scratch, walkReq, miCols, miRows)
		err := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit)
		st.unbindRefmvs(scratch)
		if err != nil {
			return nil, err
		}
		return pc.writer.Finish()
	}

	// Split pipeline (PERF_PLAN §6 E1): decision pass first — search,
	// prediction, quantize, recon, contexts marked, decisions recorded per SB
	// row — then a serial entropy pass replaying the records through the same
	// writers. The libaom encode_sb / write_modes split; see pframe_split.go.
	// The decision pass runs as an SB-row wavefront when the coder has lanes
	// (pframe_wavefront.go), byte-identical to the serial walk by
	// construction and by the oracle tests.
	tileMI4W := int(miColEnd - miColStart)
	rows := (int(miRows) + sbSizeMIB - 1) / sbSizeMIB
	if cap(pc.splitRowRecs) < rows {
		next := make([]pframeSplitRecord, rows)
		copy(next, pc.splitRowRecs)
		pc.splitRowRecs = next
	}
	pc.splitRowRecs = pc.splitRowRecs[:rows]
	for r := range pc.splitRowRecs {
		rowMI4H := int(miRows) - r*sbSizeMIB
		if rowMI4H > sbSizeMIB {
			rowMI4H = sbSizeMIB
		}
		pc.splitRowRecs[r].reset(tileMI4W, rowMI4H, st.color)
	}
	lanes := pc.wavefrontLanes
	if lanes > rows {
		lanes = rows
	}
	pipelined := pc.wavefront != nil && lanes >= 2 && st.decisionStats == nil
	if pipelined {
		// Decision lanes run on parked helpers; this goroutine immediately
		// starts the serial entropy pass below, consuming each SB row's
		// records once its decisions complete (SVT's entropy_coding_process
		// trailing enc_dec_kernel's finished segment rows).
		pc.wavefront.start(pc, lanes, src, ref, golden, recon, referenceMode, walkReq, miCols, miRows, rows, rootCols, scaledReference)
	} else {
		decideRecord := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
			partition, err := decide(level, ctx, miCol, miRow, haveRight, haveBottom)
			if err == nil {
				pc.splitRowRecs[int(miRow)/sbSizeMIB].appendPart(partition)
			}
			return partition, err
		}
		visitDecide := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
			return st.decidePBlock(src, ref, golden, recon, block, scratch, referenceMode, walkReq, miCols, miRows, &pc.splitRowRecs[int(block.MIRow)/sbSizeMIB])
		}
		st.bindRefmvs(scratch, walkReq, miCols, miRows)
		err := tile.WalkBlockLoopDecide(scratch, carrier, walkReq, sbSizeMIB, decideRecord, visitDecide)
		st.unbindRefmvs(scratch)
		if err != nil {
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
			// Forced-split edge superblocks reach their leaves without a
			// partition read, so the visit gates on the row too.
			if err := pc.wavefront.waitRowDecided(row); err != nil {
				return err
			}
		}
		return st.writePBlock(block, scratch, &pc.refCDFs, &pc.modeCDFs, &pc.interpCDFs, referenceMode, &pc.splitRowRecs[row])
	}
	var writeStart txWriteStatsStamp
	if txWriteStatsEnabled {
		writeStart = txWriteStatsStart()
	}
	writeErr := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decideReplay, visitWrite)
	if txWriteStatsEnabled {
		txWriteStatsFinish(writeStart)
	}
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

// decideRealtimePartition is the content-only partition decision for one tree
// node (the fused and split paths and every wavefront lane share it): 8x8
// leaves stop, non-420 and scaled-reference frames force splits down to the
// leaves, and everything else follows the SB's SVT VERY_LIGHT_PD0 tree
// (pframe_vlpd0.go).
func (st *lossyEncodeState) decideRealtimePartition(src, ref SourceFrame420, scaledReference bool, level tile.BlockLevel, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
	if level == tile.BlockLevel8x8 {
		return tile.PartitionNone, nil
	}
	if !videoColorIs420(st.color) {
		return tile.PartitionSplit, nil
	}
	if scaledReference {
		return tile.PartitionSplit, nil
	}
	px, py := int(miCol)*4, int(miRow)*4
	sbPX, sbPY := (px/64)*64, (py/64)*64
	sb, err := st.svtPD0ForSB(src, ref, sbPX, sbPY)
	if err != nil {
		return tile.PartitionSplit, err
	}
	n := 64
	if level == tile.BlockLevel32x32 {
		n = 32
	} else if level == tile.BlockLevel16x16 {
		n = 16
	}
	part := svtPD0Fallback(px, py, n, src.Width, src.Height)
	if sb != nil {
		relX, relY := px-sbPX, py-sbPY
		switch level {
		case tile.BlockLevel64x64:
			part = sb.part64
		case tile.BlockLevel32x32:
			part = sb.part32[(relY/32)*2+relX/32]
		case tile.BlockLevel16x16:
			part = sb.part16[(relY/16)*4+relX/16]
		}
	}
	// A node missing its lower (right) half codes only HORZ (VERT) or SPLIT.
	if (!haveBottom && part != tile.PartitionH && part != tile.PartitionSplit) ||
		(!haveRight && part != tile.PartitionV && part != tile.PartitionSplit) {
		return tile.PartitionSplit, nil
	}
	return part, nil
}

func (pc *pframeCoder) encodeLosslessIntraInterTile(src SourceFrame420, recon *SourceFrame420, prev *frameCDFs, forceIntegerMV bool, allowScreenContentTools bool, color parser.ColorConfig, miColStart, miColEnd uint16) ([]byte, error) {
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16
	rootCols := (int(miColEnd-miColStart) + sbSizeMIB - 1) / sbSizeMIB
	if err := pc.reset(0, rootCols, prev, color); err != nil {
		return nil, err
	}
	st := &pc.st
	st.forceIntegerMV = forceIntegerMV
	st.allowScreenContentTools = allowScreenContentTools
	st.transformMode = parser.TransformMode4x4Only
	st.lfMap = nil
	st.decisionStats = nil
	if pc.decisionStatsEnabled {
		pc.decisionStats.Reset()
		pc.decisionStats.Tiles = 1
		st.decisionStats = &pc.decisionStats
	}

	pc.writer.Reset(pc.writerBuf[:0])
	st.w = &pc.writer
	st.intraTxTypeReq = tile.IntraTransformTypeRequest{
		Size:        tile.TransformSize4x4,
		Mode:        tile.IntraModeDC,
		Lossless:    true,
		QIndexKnown: true,
		QIndex:      0,
	}
	if st.afterSkipIntra == nil {
		st.afterSkipIntra = func() error {
			return tile.WriteIntraTransformType(st.w, &st.txCDFs, st.intraTxTypeReq, transform.TypeDCTDCT)
		}
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
		return st.encodeLosslessIntraInterBlock(src, block, scratch)
	}
	if err := tile.WalkBlockLoopWrite(&pc.writer, &pc.partCDFs, scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	copyLosslessTileSourceToRecon(src, recon, color, miColStart, miColEnd)
	return pc.writer.Finish()
}

func (st *lossyEncodeState) encodeLosslessIntraInterBlock(src SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
	dims, ok := block.Size.Dimensions()
	if !ok || dims.W4 < 2 || dims.H4 < 2 {
		return fmt.Errorf("encoder: unexpected lossless intra-inter block %+v", block)
	}
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx
	const lumaMode = tile.IntraModeDC
	const chromaMode = tile.ChromaIntraModeDC

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("lossless intra-inter skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("lossless intra-inter mark prefix: %w", err)
	}
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
		return fmt.Errorf("lossless intra-inter intra flag: %w", err)
	}
	if err := tile.WriteLumaIntraMode(st.w, &st.intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		FrameType: parser.FrameTypeInter,
		Size:      block.Size, X4: block.X4, Y4: block.Y4,
	}, lumaMode); err != nil {
		return fmt.Errorf("lossless intra-inter luma mode: %w", err)
	}
	if err := modeCtx.MarkIntra(block.Size, int(block.X4), int(block.Y4), true, lumaMode); err != nil {
		return fmt.Errorf("lossless intra-inter mark intra: %w", err)
	}
	if err := tile.WriteIntraAngleDelta(st.w, &st.intraCDFs, tile.IntraAngleDeltaRequest{
		Size: block.Size, Mode: lumaMode,
	}, 0); err != nil {
		return fmt.Errorf("lossless intra-inter angle delta: %w", err)
	}
	hasChroma := !st.color.MonoChrome
	if hasChroma {
		cflAllowed, err := tile.ChromaIntraCFLAllowed(block.Size, st.color, false)
		if err != nil {
			return fmt.Errorf("lossless intra-inter cfl allowed: %w", err)
		}
		if err := tile.WriteChromaIntraMode(st.w, &st.intraCDFs, tile.ChromaIntraModeRequest{
			Size: block.Size, LumaMode: lumaMode, CFLAllowed: cflAllowed,
		}, chromaMode, tile.CFLAlphaResult{}); err != nil {
			return fmt.Errorf("lossless intra-inter chroma mode: %w", err)
		}
		if err := modeCtx.MarkChromaIntra(block.Size, int(block.X4), int(block.Y4), true, chromaMode); err != nil {
			return fmt.Errorf("lossless intra-inter mark chroma: %w", err)
		}
		if err := st.writeNoPaletteMode(modeCtx, block, lumaMode, chromaMode, true); err != nil {
			return err
		}
	} else if err := st.writeNoPaletteMode(modeCtx, block, lumaMode, chromaMode, false); err != nil {
		return err
	}

	if _, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
		VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
		HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
		Color: st.color, TransformMode: parser.TransformMode4x4Only,
		Lossless: true,
	}, tile.TransformTreeResult{Y: tile.TransformSize4x4}); err != nil {
		return fmt.Errorf("lossless intra-inter transform tree: %w", err)
	}

	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	st.intraTxTypeReq.Mode = lumaMode
	st.intraTxTypeReq.Size = tile.TransformSize4x4
	st.intraTxTypeReq.Lossless = true
	for ty := range int(block.VisibleH4) {
		for tx := range int(block.VisibleW4) {
			if err := encodeLosslessTXBWithHook(st.w, &st.coeffCDFs, coeffCtx, tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: block.Size,
				Size:       tile.TransformSize4x4,
				X4:         block.X4 + uint8(tx),
				Y4:         block.Y4 + uint8(ty),
			}, src.Y, src.YStride, lumaPX+tx*4, lumaPY+ty*4, st.scan4, st.levels, st.afterSkipIntra); err != nil {
				return fmt.Errorf("lossless intra-inter luma txb (%d,%d): %w", tx, ty, err)
			}
		}
	}
	if !hasChroma {
		if st.decisionStats != nil {
			st.decisionStats.noteIntraBlock(block.Size)
		}
		return nil
	}
	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return fmt.Errorf("lossless intra-inter chroma plane block: %w", err)
	}
	cw, ch, err := planeBlockPixels(chromaBlock)
	if err != nil {
		return fmt.Errorf("lossless intra-inter chroma plane block dimensions: %w", err)
	}
	chromaPX := chromaXForColor(lumaPX, st.color)
	chromaPY := chromaYForColor(lumaPY, st.color)
	chromaX4 := chromaX4ForColor(block.X4, st.color)
	chromaY4 := chromaY4ForColor(block.Y4, st.color)
	for plane := 1; plane <= 2; plane++ {
		data := src.U
		if plane == 2 {
			data = src.V
		}
		for py := 0; py < ch; py += 4 {
			for px := 0; px < cw; px += 4 {
				if err := encodeLosslessTXB(st.w, &st.coeffCDFs, coeffCtx, tile.CoeffContextRequest{
					Plane:      uint8(plane),
					PlaneBlock: chromaBlock,
					Size:       tile.TransformSize4x4,
					X4:         chromaX4 + uint8(px/4),
					Y4:         chromaY4 + uint8(py/4),
				}, data, src.ChromaStride, chromaPX+px, chromaPY+py, st.scan4, st.levels); err != nil {
					return fmt.Errorf("lossless intra-inter chroma %d txb (%d,%d): %w", plane, px/4, py/4, err)
				}
			}
		}
	}
	if st.decisionStats != nil {
		st.decisionStats.noteIntraBlock(block.Size)
	}
	return nil
}

func copyLosslessTileSourceToRecon(src SourceFrame420, recon *SourceFrame420, color parser.ColorConfig, miColStart, miColEnd uint16) {
	if recon == nil {
		return
	}
	x0 := int(miColStart) * 4
	x1 := int(miColEnd) * 4
	if x1 > src.Width {
		x1 = src.Width
	}
	for y := 0; y < src.Height; y++ {
		copy(recon.Y[y*recon.YStride+x0:y*recon.YStride+x1], src.Y[y*src.YStride+x0:y*src.YStride+x1])
	}
	if color.MonoChrome {
		return
	}
	cx0 := chromaXForColor(x0, color)
	cx1 := chromaXForColor(x1, color)
	ch := chromaHeightForColor(src.Height, color)
	for y := 0; y < ch; y++ {
		copy(recon.U[y*recon.ChromaStride+cx0:y*recon.ChromaStride+cx1], src.U[y*src.ChromaStride+cx0:y*src.ChromaStride+cx1])
		copy(recon.V[y*recon.ChromaStride+cx0:y*recon.ChromaStride+cx1], src.V[y*src.ChromaStride+cx0:y*src.ChromaStride+cx1])
	}
}

// encodePBlock codes one inter block (8x8 or 16x16) with residual: the inter
// mode symbols in the decoder's order, then the largest-TX luma residual (with
// the inter tx_type symbol after txb_skip) and two chroma residuals against
// the reference reconstruction.
func (st *lossyEncodeState) encodePBlock(src, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch,
	refCDFs *tile.InterRefCDFs, interModeCDFs *tile.InterModeCDFs, interpCDFs *tile.InterpFilterCDFs,
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
	case tile.BlockSize64x32:
		bw, bh = 64, 32
	case tile.BlockSize64x16:
		bw, bh = 64, 16
	case tile.BlockSize32x64:
		bw, bh = 32, 64
	case tile.BlockSize16x64:
		bw, bh = 16, 64
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
	st.predStats.valid = false
	// The MDS0 winner statistics belong to this block only (a block that
	// left MDS0 for intra must not hand them to the next).
	st.mds0BestRate = -1
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	if err := st.useSBQuant(lumaPX, lumaPY); err != nil {
		return err
	}
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
	// cached them; the shared motion stage (pframe_motion.go) adds the SVT
	// light-PD1 sub-pel refinement and the reference choice.
	mo, err := st.pblockMotionSearch(src, ref, golden, block, modeCtx, referenceMode, walkReq, miCols, miRows, lumaPX, lumaPY, bw, bh)
	if err != nil {
		return err
	}
	scaledReference := mo.scaledReference
	mv, fullSAD := mo.mv, mo.fullSAD
	refs, refPlanes := mo.refs, mo.refPlanes
	compoundMV := mo.compoundMV

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
	stack, err := st.pblockStack(&mo, block, modeCtx, walkReq, miCols, miRows)
	if err != nil {
		return err
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
	}
	drlIndex := 0
	mds0PredRetained := false
	if !refs.Compound {
		mds0Picked := false
		// The light-PD1 MDS0 candidate decision (SVT preset 12 shape; see
		// pframe_mds0.go) prices every block, low-distortion ones included:
		// SVT's generate_md_stage_0_cand_light_pd1 has no distortion gate,
		// and at high q those are the blocks where a vector-free NEARESTMV
		// undercuts NEWMV. Static superblocks keep the short path as
		// libaom's realtime pick cuts its mode loop on zero source SAD
		// (av1/encoder/nonrd_pickmode.c source-SAD shortcuts).
		if st.mds0Level != 0 && !scaledReference && refPlanes.Width == src.Width && refPlanes.Height == src.Height &&
			st.realtimeContentStateForBlock(lumaPX, lumaPY).sourceSADNonRD != realtimeSourceSADZero {
			if cand, ok, retained := st.mds0PickInterMode(src, refPlanes.Y, refPlanes.YStride, &stack, lumaPX, lumaPY, bw, bh, mv); ok {
				modeResult.Mode = cand.mode
				if cand.mv != mv {
					fullSAD = st.mds0WinnerSAD(src, lumaPX, lumaPY, bw, bh, retained, fullSAD)
				}
				mv = cand.mv
				drlIndex = int(cand.drl)
				mds0Picked = true
				mds0PredRetained = retained
			}
		}
		if !mds0Picked {
			modeResult.Mode = st.classifyInterMode(src, refPlanes, &stack, lumaPX, lumaPY, bw, bh, &mv, &fullSAD)
		} else if st.mds0IntraDCWins(src, recon, block, modeCtx, lumaPX, lumaPY, bw, bh) {
			return st.encodeIntraPBlock(src, recon, block, scratch)
		}
	}
	// Light-PD1 runs no interpolation filter search: md_stage_0_light_pd1
	// sets REGULAR on every candidate and svt_aom_inter_pu_prediction_av1_
	// light_pd1 never calls interpolation_filter_search (IFS_OFF is a
	// light-PD1 feature assumption, enc_dec_process.c
	// exaustive_light_pd1_features). Every block is REGULAR, so P-frame
	// headers carry the fixed EIGHTTAP filter instead of SWITCHABLE with
	// all-REGULAR blocks (identical predictions, no per-block symbols).
	blockFilters := motion.InterpFilters{}
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
			if !mds0PredictorMatchesCodedFilters(mds0PredRetained, mv, blockFilters) {
				if err := st.predictLuma(st.predY[:bw*bh], refPlanes.Y, refPlanes.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, blockFilters); err != nil {
					return fmt.Errorf("predict luma: %w", err)
				}
			}
			if hasChroma {
				if err := st.predictPlane(st.predU[:cbw*cbh], refPlanes.U, refPlanes.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, blockFilters); err != nil {
					return fmt.Errorf("predict u: %w", err)
				}
				if err := st.predictPlane(st.predV[:cbw*cbh], refPlanes.V, refPlanes.ChromaStride, chromaWidth, chromaHeight, chromaPX, chromaPY, cbw, cbh, mv, st.color.SubsamplingX, st.color.SubsamplingY, blockFilters); err != nil {
					return fmt.Errorf("predict v: %w", err)
				}
			}
		}
	}

	// Quantize all three transform blocks up front; a block whose residual
	// quantizes to zero everywhere is coded as skip (no residual symbols, the
	// reconstruction is the prediction itself). The shared decision helper
	// also applies the light-PD1 TX shortcut detectors (pframe_lpd1_tx.go).
	st.predYKey = lumaPredKey{
		valid:   !refs.Compound && refPlanes.Width == src.Width && refPlanes.Height == src.Height,
		mv:      mv,
		filters: blockFilters,
	}
	st.lpd1NbrSkipCtx, _ = scratch.Mode.SkipContext(int(block.X4), int(block.Y4))
	st.lpd1NoteIsInterRate(&scratch.Mode, block)
	st.lpd1NoteMVP64(lumaPX, lumaPY, bw, bh, modeResult)
	skip, splitTX, txPlan, txType, err := st.decideInterTXBlock(src, refPlanes, mv, block,
		lumaPX, lumaPY, bw, bh, cbw, cbh, chromaPX, chromaPY, hasChroma, fullSAD)
	st.predYKey.valid = false
	if err != nil {
		return err
	}

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, skip); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := st.writeSBDeltaQ(block, skip); err != nil {
		return fmt.Errorf("delta q: %w", err)
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
	if err := tile.WriteDRLIndex(st.w, interModeCDFs, drlReq, drlIndex); err != nil {
		return fmt.Errorf("drl: %w", err)
	}
	var mvRefs tile.InterMVReferenceSet
	if !interModeResultUsesGlobalOnly(modeResult) {
		mvRefs, err = stack.Stack.ResolveInterMVReferences(modeResult, drlIndex, false, st.forceIntegerMV)
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
	// Count zero motion exactly as libaom's update_state()
	// (av1/encoder/encodeframe_utils.c): inter blocks predicting from LAST
	// whose vector is under one pixel accumulate their MI area, feeding the
	// avg_frame_low_motion statistic behind adaptive golden refresh.
	if refs.Ref[0] == tile.ReferenceFrameLast {
		zmv := motionResult.MV[0]
		if zmv.Row > -8 && zmv.Row < 8 && zmv.Col > -8 && zmv.Col < 8 {
			st.cntZeroMV += int(block.VisibleW4) * int(block.VisibleH4)
		}
	}
	if err := tile.WriteInterMotion(st.w, &st.mvCDFs, tile.InterMotionRequest{
		References:   refs,
		Mode:         modeResult,
		ReferenceMVs: mvRefs,
		Precision:    tile.MVPrecision(false, st.forceIntegerMV),
	}, motionResult); err != nil {
		return fmt.Errorf("motion vector: %w", err)
	}

	if err := scratch.EncoderCommitInter(block, motionResult, blockFilters, hasChroma); err != nil {
		return err
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
		treeRes = txPlan.tree
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
		if err := st.markLoopFilterBlock(block, lfTree, skip, false, uint8(refs.Ref[0])+1, lfMode); err != nil {
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

	// Residual: luma with source-shaped realtime TX partitioning, then chroma.
	lumaTX, lumaScan := st.interLumaTXAndScan(bw, bh)
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
			chromaScan, ok = st.scanForThinTransformSize(chromaTX)
		}
		if !ok {
			return fmt.Errorf("encoder: unsupported chroma transform %d", chromaTX)
		}
	}
	if splitTX {
		childScan, ok := st.scanForTransformSize(txPlan.leafTX)
		if !ok {
			childScan, ok = st.scanForThinTransformSize(txPlan.leafTX)
		}
		if !ok {
			return fmt.Errorf("encoder: unsupported realtime luma transform %d", txPlan.leafTX)
		}
		st.interTxTypeReq.Size = txPlan.leafTX
		st.interTxType = transform.TypeDCTDCT
		leafArea := txPlan.leafSize * txPlan.leafSize
		if err := txPlan.ForEachLeaf(func(i, dx, dy int) error {
			return st.finishInterTXB(recon.Y, st.predY[dy*bw+dx:], bw, src.YStride, lumaPX+dx, lumaPY+dy, txPlan.leafSize, txPlan.leafSize, st.yQuant, st.lumaQ2[i*leafArea:(i+1)*leafArea], tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: block.Size,
				Size:       txPlan.leafTX,
				X4:         block.X4 + uint8(dx/4),
				Y4:         block.Y4 + uint8(dy/4),
			}, coeffCtx, childScan, st.afterSkipInter)
		}); err != nil {
			return fmt.Errorf("luma realtime tx: %w", err)
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
		st.noteInterTXBSizes(block.Size, splitTX, txPlan, bw, bh)
	}
	return nil
}

// classifyInterMode is the legacy single-reference mode choice by signaling
// cost: zero motion keeps the short GLOBALMV cascade; a vector the predictor
// stack already names codes as NEARESTMV/NEARMV with no motion residual at
// all; everything else pays for NEWMV plus the joint and component symbols,
// with one priced NEARESTMV predictor-swap probe. mv and fullSAD update in
// place when the swap wins.
func (st *lossyEncodeState) classifyInterMode(src, refPlanes SourceFrame420, stack *tile.ReferenceMVStackResult,
	lumaPX, lumaPY, bw, bh int, mv *motion.Vector, fullSAD *int) tile.InterMode {
	mode := tile.InterModeGlobalMV
	if mv.Row == 0 && mv.Col == 0 {
		return mode
	}
	mode = tile.InterModeNewMV
	predRefs, err := stack.Stack.ResolveInterMVReferences(tile.InterModeResult{Mode: tile.InterModeNearestMV}, 0, false, st.forceIntegerMV)
	if err != nil {
		return mode
	}
	switch *mv {
	case predRefs.Nearest[0]:
		return tile.InterModeNearestMV
	case predRefs.Near[0]:
		return tile.InterModeNearMV
	}
	nearest := predRefs.Nearest[0]
	// Pricing the predictor swap needs a prediction and a SAD;
	// blocks already in skip territory keep NEWMV without it.
	if (nearest.Row != 0 || nearest.Col != 0) && *fullSAD > bw*bh {
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
			if err := st.predictLuma(st.sadScratch[:bw*bh], refPlanes.Y, refPlanes.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, nearest, motion.RegularFilters); err == nil {
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
				if nearSAD+2*st.sadPerBit < *fullSAD+mvBits*st.sadPerBit {
					mode = tile.InterModeNearestMV
					*mv = nearest
					*fullSAD = nearSAD
				}
			}
		}
	}
	return mode
}

// encodeIntraPBlock codes one 8x8 intra block inside an inter frame: skip,
// the is_inter flag (intra), DC luma and chroma modes through the inter-frame
// y_mode CDFs, the decoder's MarkIntra/MarkChromaIntra context updates, then
// the same DC-predicted TXB pipeline the lossy keyframe path uses.
func (st *lossyEncodeState) encodeIntraPBlock(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
	st.lpd1NoteIntraBlock(block)
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("intra skip: %w", err)
	}
	if err := st.writeSBDeltaQ(block, false); err != nil {
		return fmt.Errorf("intra delta q: %w", err)
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
	n, lumaTX, lumaScan := intraPBlockGeometry(st, block.Size)
	pred := st.predY[:n*n]
	mode, angleDelta := tile.IntraModeDC, 0
	if n == 8 {
		mode = selectIntraMode8(src.Y, recon.Y, src.YStride, lumaPX, lumaPY, block.HaveTop, block.HaveLeft, pred)
		if !st.trialReady {
			if err := st.trialCDFs.InitDefault(st.qIndex); err != nil {
				return err
			}
			st.trialReady = true
		}
		mode, angleDelta = st.improveIntraModeDirectional(src, recon, block, mode, pred, lumaPX, lumaPY, 8)
	} else {
		fillIntraDCPred(pred, recon.Y, src.YStride, lumaPX, lumaPY, n, block.HaveTop, block.HaveLeft)
	}
	if err := tile.WriteLumaIntraMode(st.w, &st.intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		FrameType: parser.FrameTypeInter,
		Size:      block.Size, X4: block.X4, Y4: block.Y4,
	}, mode); err != nil {
		return fmt.Errorf("intra luma mode: %w", err)
	}
	if err := scratch.EncoderCommitIntra(block, mode); err != nil {
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
	}, tile.TransformTreeResult{Y: lumaTX})
	if err != nil {
		return fmt.Errorf("intra transform tree: %w", err)
	}
	if st.lfMap != nil {
		if err := st.markLoopFilterBlock(block, lfTree, false, true, 0, loopfilter.ModeDeltaClassZero); err != nil {
			return fmt.Errorf("mark intra loop filter: %w", err)
		}
	}

	st.intraTxTypeReq.Mode = mode
	st.intraTxTypeReq.Size = lumaTX
	err = st.encodeTXBPred(recon.Y, src.Y, src.YStride, lumaPX, lumaPY, n, st.yQuant, tile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: block.Size,
		Size:       lumaTX,
		X4:         block.X4,
		Y4:         block.Y4,
	}, coeffCtx, lumaScan, st.afterSkipIntra, pred)
	st.intraTxTypeReq.Size = tile.TransformSize8x8
	if err != nil {
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
		chromaScan, ok = st.scanForThinTransformSize(chromaTX)
	}
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
func predictInto(dst []byte, refPlane []byte, stride, width, height, px, py, bw, bh int, mv motion.Vector, ssX, ssY bool, scratch *motion.ScaledConvolveScratch) error {
	return predictIntoScaled(dst, refPlane, stride, width, height, width, height, px, py, bw, bh, mv, ssX, ssY, scratch)
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
		return motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dstPlane, ref, 1, 8, 0, 0, refX, refY, bw, bh, bw, bh, subX, subY, filters, scratch.Conv())
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
			// LAST's vector from the frame's open-loop ME (hme.go).
			var lmv motion.Vector
			if me := st.hme.sbME(px, py); me != nil {
				pu := svtMEBlockOffset(px&63, py&63, blockSize)
				lmv = motion.Vector{Row: me.bestMV[pu][1] * 8, Col: me.bestMV[pu][0] * 8}
			}
			ldx, ldy := int(lmv.Col>>3), int(lmv.Row>>3)
			if px+ldx < 0 || py+ldy < 0 || px+ldx+blockSize > ref.Width || py+ldy+blockSize > ref.Height {
				continue
			}
			lastSAD := st.leafFullPelSAD(src, ref, px, py, blockSize, blockSize, lmv)
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

func (st *lossyEncodeState) allowSubpelRefinement() bool {
	return !st.forceIntegerMV && st.effortLevel > WebRTCMinEffortLevel
}

// txScaleForSize is get_tx_scale for square transforms.
func txScaleForSize(n int) uint8 {
	if n >= 64 {
		return 2
	}
	if n >= 32 {
		return 1
	}
	return 0
}

// realtimeInterTXPlan is the low-bitdepth realtime nonrd_pickmode.c
// calculate_tx_size() result expressed as AV1 var-tx syntax. Libaom chooses a
// square TX_SIZE for the inter block; rectangular prediction blocks signal that
// choice by splitting the transform tree down to square 8x8 or 16x16 leaves.
type realtimeInterTXPlan struct {
	block    tile.BlockSize
	tree     tile.TransformTreeResult
	leafTX   tile.TransformSize
	leafSize int
}

func (p realtimeInterTXPlan) Variable() bool {
	return p.tree.Split[0] != 0 || p.tree.Split[1] != 0
}

func (p realtimeInterTXPlan) ForEachLeaf(visit func(i, dx, dy int) error) error {
	if visit == nil || p.leafSize != 8 && p.leafSize != 16 && p.leafSize != 32 {
		return tile.ErrInvalidDecodeState
	}
	if c := realtimeInterTXPlanCached(p.block, p.leafSize); c != nil && c.plan == p {
		for i := 0; i < c.leaves; i++ {
			if err := visit(i, int(c.dx[i]), int(c.dy[i])); err != nil {
				return err
			}
		}
		return nil
	}
	maxY, err := tile.MaxTransformSize(p.block, parser.ColorConfig{}, 0)
	if err != nil {
		return err
	}
	i := 0
	return p.walkLeaf(maxY, 0, 0, 0, 0, 0, &i, visit)
}

func (p realtimeInterTXPlan) walkLeaf(from tile.TransformSize, depth int, x4, y4 int, xOff, yOff int, i *int, visit func(i, dx, dy int) error) error {
	dims, ok := from.Dimensions()
	if !ok {
		return tile.ErrInvalidDecodeState
	}
	if int(dims.W4)*4 <= p.leafSize && int(dims.H4)*4 <= p.leafSize {
		if err := visit(*i, x4*4, y4*4); err != nil {
			return err
		}
		(*i)++
		return nil
	}
	sub := dims.Sub
	subDims, ok := sub.Dimensions()
	if !ok {
		return tile.ErrInvalidDecodeState
	}
	if err := p.walkLeaf(sub, depth+1, x4, y4, xOff*2, yOff*2, i, visit); err != nil {
		return err
	}
	if dims.Log2W >= dims.Log2H {
		if err := p.walkLeaf(sub, depth+1, x4+int(subDims.W4), y4, xOff*2+1, yOff*2, i, visit); err != nil {
			return err
		}
	}
	if dims.Log2H >= dims.Log2W {
		if err := p.walkLeaf(sub, depth+1, x4, y4+int(subDims.H4), xOff*2, yOff*2+1, i, visit); err != nil {
			return err
		}
		if dims.Log2W >= dims.Log2H {
			if err := p.walkLeaf(sub, depth+1, x4+int(subDims.W4), y4+int(subDims.H4), xOff*2+1, yOff*2+1, i, visit); err != nil {
				return err
			}
		}
	}
	return nil
}

func realtimeInterTXPlanForBlock(block tile.BlockSize, qIndex uint8, qAC int32, txSizeLevelBasedOnQstep int, sse, variance uint32) (realtimeInterTXPlan, error) {
	leafSize, err := realtimeInterTXLeafSizeForBlock(block, qIndex, qAC, txSizeLevelBasedOnQstep, sse, variance)
	if err != nil {
		return realtimeInterTXPlan{}, err
	}
	return realtimeInterTXPlanForLeafSize(block, leafSize)
}

func realtimeInterTXPlanForLeafSize(block tile.BlockSize, leafSize int) (realtimeInterTXPlan, error) {
	if c := realtimeInterTXPlanCached(block, leafSize); c != nil {
		return c.plan, nil
	}
	return realtimeInterTXPlanBuild(block, leafSize)
}

func realtimeInterTXPlanBuild(block tile.BlockSize, leafSize int) (realtimeInterTXPlan, error) {
	var leafTX tile.TransformSize
	switch leafSize {
	case 8:
		leafTX = tile.TransformSize8x8
	case 16:
		leafTX = tile.TransformSize16x16
	case 32:
		leafTX = tile.TransformSize32x32
	case 64:
		leafTX = tile.TransformSize64x64
	default:
		return realtimeInterTXPlan{}, fmt.Errorf("encoder: invalid realtime tx leaf %d", leafSize)
	}
	maxY, err := tile.MaxTransformSize(block, parser.ColorConfig{}, 0)
	if err != nil {
		return realtimeInterTXPlan{}, err
	}
	plan := realtimeInterTXPlan{block: block, leafTX: leafTX, leafSize: leafSize}
	if err := buildRealtimeInterTXTree(&plan.tree, maxY, leafSize, 0, 0, 0); err != nil {
		return realtimeInterTXPlan{}, err
	}
	return plan, nil
}

func buildRealtimeInterTXTree(tree *tile.TransformTreeResult, from tile.TransformSize, leafSize int, depth int, xOff int, yOff int) error {
	dims, ok := from.Dimensions()
	if !ok {
		return tile.ErrInvalidDecodeState
	}
	if int(dims.W4)*4 <= leafSize && int(dims.H4)*4 <= leafSize {
		return nil
	}
	if depth >= len(tree.Split) || yOff >= 4 || xOff >= 4 {
		return fmt.Errorf("encoder: realtime tx leaf %d is too small for %d", leafSize, from)
	}
	tree.Split[depth] |= 1 << (yOff*4 + xOff)
	sub := dims.Sub
	if _, ok := sub.Dimensions(); !ok {
		return tile.ErrInvalidDecodeState
	}
	if err := buildRealtimeInterTXTree(tree, sub, leafSize, depth+1, xOff*2, yOff*2); err != nil {
		return err
	}
	if dims.Log2W >= dims.Log2H {
		if err := buildRealtimeInterTXTree(tree, sub, leafSize, depth+1, xOff*2+1, yOff*2); err != nil {
			return err
		}
	}
	if dims.Log2H >= dims.Log2W {
		if err := buildRealtimeInterTXTree(tree, sub, leafSize, depth+1, xOff*2, yOff*2+1); err != nil {
			return err
		}
		if dims.Log2W >= dims.Log2H {
			if err := buildRealtimeInterTXTree(tree, sub, leafSize, depth+1, xOff*2+1, yOff*2+1); err != nil {
				return err
			}
		}
	}
	return nil
}

// realtimeLibaomSpeedForEffort maps the public WebRTC effort range onto
// libaom's realtime speed numbers. Effort 0 is the speed-8 baseline; lower
// effort selects speed 9/10 behavior, higher effort selects speed 7..4.
func realtimeLibaomSpeedForEffort(effort int8) int {
	if effort < WebRTCMinEffortLevel {
		effort = WebRTCMinEffortLevel
	} else if effort > WebRTCMaxEffortLevel {
		effort = WebRTCMaxEffortLevel
	}
	return 8 - int(effort)
}

// realtimeTXSizeLevelBasedOnQstep mirrors the
// speed_features.c set_rt_speed_feature_framesize_dependent() assignments for
// rt_sf.tx_size_level_based_on_qstep in low-bitdepth realtime mode.
func realtimeTXSizeLevelBasedOnQstep(width, height int, effort int8) int {
	speed := realtimeLibaomSpeedForEffort(effort)
	if minInt(width, height) < 360 {
		if speed >= 8 {
			return 1
		}
		return 0
	}
	if speed >= 10 {
		return 0
	}
	if speed >= 8 {
		return 2
	}
	return 0
}

// realtimeInterTXLeafSizeForBlock mirrors av1/encoder/nonrd_pickmode.c
// calculate_tx_size() for realtime TX_MODE_SELECT without cyclic-refresh AQ:
// qstep-scaled SSE/variance select TX_8X8 versus the block's max square TX when
// the speed feature is enabled, the result is capped at TX_16X16, and blocks
// larger than 32x32 force TX_16X16.
func realtimeInterTXLeafSizeForBlock(block tile.BlockSize, qIndex uint8, qAC int32, txSizeLevelBasedOnQstep int, sse, variance uint32) (int, error) {
	if txSizeLevelBasedOnQstep != 0 && qAC <= 0 {
		return 0, fmt.Errorf("encoder: invalid realtime tx qstep %d", qAC)
	}
	dims, ok := block.Dimensions()
	if !ok {
		return 0, tile.ErrInvalidDecodeState
	}
	w, h := int(dims.W4)*4, int(dims.H4)*4
	maxSquare := 4 << minInt(int(dims.Log2W), int(dims.Log2H))
	if maxSquare < 8 {
		maxSquare = 8
	}
	if w > 32 || h > 32 {
		return 16, nil
	}
	multiplier := uint32(8)
	varThresh := uint32(0)
	if txSizeLevelBasedOnQstep != 0 {
		qband := int(qIndex) >> 6
		multiplier = [...]uint32{8, 7, 6, 5}[qband]
		qstep := uint32(qAC >> 3)
		qstepSq := qstep * qstep
		varThresh = qstepSq * 2
	}
	if uint64(sse) > (uint64(variance)*uint64(multiplier))>>2 || variance < varThresh {
		if maxSquare > 16 {
			return 16, nil
		}
		return maxSquare, nil
	}
	return 8, nil
}

func realtimeInterResidualSSEVariance(src, pred []byte, srcStride, predStride, px, py, w, h int) (sse uint32, variance uint32) {
	base := py*srcStride + px
	switch {
	case w == 8 && h == 8:
		return sseVariance8x8(src[base:], srcStride, pred, predStride)
	case w == 16 && h == 8:
		return sseVariance16x8(src[base:], srcStride, pred, predStride)
	case w == 8 && h == 16:
		return sseVariance8x16(src[base:], srcStride, pred, predStride)
	case w == 16 && h == 16:
		return sseVariance16x16(src[base:], srcStride, pred, predStride)
	case w == 32 && h == 16:
		return sseVariance32x16(src[base:], srcStride, pred, predStride)
	case w == 16 && h == 32:
		return sseVariance16x32(src[base:], srcStride, pred, predStride)
	case w == 32 && h == 32:
		return sseVariance32x32(src[base:], srcStride, pred, predStride)
	case w == 64 && h == 64:
		return sseVariance64x64(src[base:], srcStride, pred, predStride)
	}
	sse, sum := pixelStatsPureGo(src[base:], srcStride, pred, predStride, w, h)
	return sse, varianceFromStats(sse, sum, uint(bits.TrailingZeros(uint(w*h))))
}

// Compatibility helpers keep the source-cap tests readable while the production
// path uses the generic 8x8/16x16 realtime plan above.
func realtimeInterTXUses16x16Leaves(w, h int) bool {
	return w > 16 || h > 16
}

func realtimeInterTX16Tree(w, h int) tile.TransformTreeResult {
	block, ok := realtimeInterBlockSizeForPixels(w, h)
	if !ok {
		return tile.TransformTreeResult{}
	}
	plan, err := realtimeInterTXPlanForLeafSize(block, 16)
	if err != nil {
		return tile.TransformTreeResult{}
	}
	return plan.tree
}

func forEachRealtimeInterTX16Leaf(w, h int, visit func(i, dx, dy int) error) error {
	block, ok := realtimeInterBlockSizeForPixels(w, h)
	if !ok {
		return fmt.Errorf("encoder: invalid realtime tx16 block %dx%d", w, h)
	}
	plan, err := realtimeInterTXPlanForLeafSize(block, 16)
	if err != nil {
		return err
	}
	return plan.ForEachLeaf(visit)
}

func realtimeInterBlockSizeForPixels(w, h int) (tile.BlockSize, bool) {
	switch {
	case w == 64 && h == 64:
		return tile.BlockSize64x64, true
	case w == 64 && h == 32:
		return tile.BlockSize64x32, true
	case w == 64 && h == 16:
		return tile.BlockSize64x16, true
	case w == 32 && h == 64:
		return tile.BlockSize32x64, true
	case w == 16 && h == 64:
		return tile.BlockSize16x64, true
	case w == 32 && h == 32:
		return tile.BlockSize32x32, true
	case w == 32 && h == 16:
		return tile.BlockSize32x16, true
	case w == 16 && h == 32:
		return tile.BlockSize16x32, true
	case w == 16 && h == 16:
		return tile.BlockSize16x16, true
	case w == 16 && h == 8:
		return tile.BlockSize16x8, true
	case w == 8 && h == 16:
		return tile.BlockSize8x16, true
	case w == 8 && h == 8:
		return tile.BlockSize8x8, true
	}
	return 0, false
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
		return sad64x64Dual(src, srcStride, ref, refStride)
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
	case bw == 64 && bh == 32:
		return sad32x32Dual(src, srcStride, ref, refStride) +
			sad32x32Dual(src[32:], srcStride, ref[32:], refStride)
	case bw == 32 && bh == 64:
		return sad32x32Dual(src, srcStride, ref, refStride) +
			sad32x32Dual(src[32*srcStride:], srcStride, ref[32*refStride:], refStride)
	case bw == 64 && bh == 16:
		return sad16x16Dual(src, srcStride, ref, refStride) +
			sad16x16Dual(src[16:], srcStride, ref[16:], refStride) +
			sad16x16Dual(src[32:], srcStride, ref[32:], refStride) +
			sad16x16Dual(src[48:], srcStride, ref[48:], refStride)
	case bw == 16 && bh == 64:
		return sad16x16Dual(src, srcStride, ref, refStride) +
			sad16x16Dual(src[16*srcStride:], srcStride, ref[16*refStride:], refStride) +
			sad16x16Dual(src[32*srcStride:], srcStride, ref[32*refStride:], refStride) +
			sad16x16Dual(src[48*srcStride:], srcStride, ref[48*refStride:], refStride)
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
	copyPredToReconBlock(reconPlane, pred, w, stride, px, py, w, h)
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
	geo, err := txBlockGeometryForDimensions(w, h)
	if err != nil {
		return false
	}
	residual := &st.resScratch
	residualBlockImpl(residual[:w*h], srcPlane, py*stride+px, stride, pred, predStride, w, h)
	n := geo.sampleCount
	cn := geo.coeffCount
	tran := &st.tranScratch
	if err := forwardTransformBlock8BitResidualTrusted(tran[:cn], residual[:n], st.dqScratch[:n], w, h, txType); err != nil {
		return false
	}
	if st.pfN4 {
		svtPFN4Zero(tran[:cn], w, h, geo.coeffWidth, geo.coeffHeight)
	}
	if len(qcoeff) < cn {
		return false
	}
	return st.quantizeInterTXBStats(&geo, q, qcoeff)
}

// quantizeInterTXBStats quantizes the transform coefficients in tranScratch
// into qcoeff and accumulates the block's skip-decision RD terms (see
// prepareInterTXB), reporting whether every level is zero.
func (st *lossyEncodeState) quantizeInterTXBStats(geo *txBlockGeometry, q quantize.Quantizer, qcoeff []int16) bool {
	cn := geo.coeffCount
	tran := &st.tranScratch
	if st.rdoqArm != 0 {
		st.rdoqQuantizeTX(tran[:cn], qcoeff[:cn], rdoqGeometry(geo.tx, int(st.rdoqArm-1)), q)
		return st.accumulateInterTXBStats(geo, q, qcoeff)
	}
	if err := quantize.QuantizeBlockScaledB(qcoeff[:cn], geo.coeffHeight, tran[:cn], geo.coeffHeight, geo.coeffWidth, geo.coeffHeight, q, geo.txScale); err != nil {
		return false
	}
	return st.accumulateInterTXBStats(geo, q, qcoeff)
}

// accumulateInterTXBStats adds the skip-decision RD terms of the levels in
// qcoeff over the coefficients in tranScratch (see prepareInterTXB) and
// reports whether every level is zero.
func (st *lossyEncodeState) accumulateInterTXBStats(geo *txBlockGeometry, q quantize.Quantizer, qcoeff []int16) bool {
	cn := geo.coeffCount
	tran := &st.tranScratch
	// The vector statistics apply the AC step to every lane; the DC
	// coefficient's coded distortion is then corrected with the DC step.
	dskip, dcode, rate, allZero := rdStatsBlockImpl(tran[:cn], qcoeff[:cn], cn, q.AC, geo.txScale)
	if v := qcoeff[0]; v != 0 {
		c := int64(tran[0])
		eAC := c - ((int64(v) * int64(q.AC)) >> geo.txScale)
		eDC := c - ((int64(v) * int64(q.DC)) >> geo.txScale)
		dcode += eDC*eDC - eAC*eAC
	}
	st.rdDskip += dskip
	st.rdDcode += dcode
	st.rdRcode += rate
	return allZero
}

// prepareInterTXBDecide is prepareInterTXB for decideInterTXBlock, whose RD
// terms only ever feed the skip comparison rdDskip<<7 <= rate + rdDcode<<7.
// A coefficient that quantizes to zero adds the same c*c to rdDskip and
// rdDcode, so its terms cancel there; square 8x8, 16x16 and 32x32 DCT_DCT
// blocks therefore take the row-skipping forward DCT, which leaves out the
// coefficient rows that provably quantize to zero (and the whole
// quantize/stats pass when every row does). The levels, the all-zero result
// and the skip decision are exactly those of prepareInterTXB; only the
// cancelling distortion terms are omitted (rdDskip carries the skip-minus-code
// difference of the non-zero levels), so no caller may read the absolute
// rdDcode/rdDskip after it (the 8x8 TX-type trial's DCT baseline must keep
// using prepareInterTXB).
func (st *lossyEncodeState) prepareInterTXBDecide(srcPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16) bool {
	var geo *txBlockGeometry
	switch {
	case w == 8 && h == 8:
		geo = &txBlockGeometry8x8
	case w == 16 && h == 16:
		geo = &txBlockGeometry16x16
	case w == 32 && h == 32:
		geo = &txBlockGeometry32x32
	}
	if geo == nil || len(qcoeff) < geo.coeffCount {
		return st.prepareInterTXB(srcPlane, pred, predStride, stride, px, py, w, h, q, qcoeff)
	}
	residualBlockImpl(st.resScratch[:geo.sampleCount], srcPlane, py*stride+px, stride, pred, predStride, w, h)
	if st.rdoqArm != 0 {
		return st.prepareInterTXBDecideRDOQ(geo, w, q, qcoeff)
	}
	dcZero, acZero := quantize.ZeroBinB(q, geo.txScale)
	thr0, thr := min(dcZero, acZero), acZero
	var mask uint8
	switch w {
	case 8:
		mask = transform.ForwardDCT8x8RowSkip8BitTrusted((*[64]int32)(st.tranScratch[:64]), (*[64]int16)(st.resScratch[:64]), thr0, thr)
	case 16:
		mask = transform.ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(st.tranScratch[:256]), (*[256]int16)(st.resScratch[:256]), thr0, thr)
	default:
		mask = transform.ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(st.tranScratch[:1024]), (*[1024]int16)(st.resScratch[:1024]), thr0, thr)
	}
	if st.pfN4 {
		svtPFN4Zero(st.tranScratch[:geo.coeffCount], w, h, geo.coeffWidth, geo.coeffHeight)
		mask = svtPFN4Mask(mask, h)
	}
	if mask == 0 {
		clear(qcoeff[:geo.coeffCount])
		return true
	}
	// Keep the quantizer's loaded coefficients and levels for the following
	// skip-only statistics when the architecture can fuse the two passes.
	if diff, rate, handled := quantize.TryQuantizeBRowGroupsDiffTrusted(qcoeff[:geo.coeffCount], st.tranScratch[:geo.coeffCount], w, mask, q, geo.txScale); handled {
		st.rdDskip += diff
		st.rdRcode += rate
		return rate == 0
	}
	// Quantize only the live rows; when every level is zero the block adds
	// nothing to the skip comparison.
	if !quantize.QuantizeBRowGroupsTrusted(qcoeff[:geo.coeffCount], st.tranScratch[:geo.coeffCount], w, mask, q, geo.txScale) {
		return true
	}
	return st.accumulateInterTXBStatsDiff(geo, q, qcoeff)
}

// accumulateInterTXBStatsDiff is accumulateInterTXBStats for the skip
// comparison alone: only rdDskip - rdDcode and the rate matter there, and a
// zero level adds equally to both distortions, so the non-zero levels'
// difference terms go to rdDskip (rdStatsDiffImpl) and rdDcode is left as is.
func (st *lossyEncodeState) accumulateInterTXBStatsDiff(geo *txBlockGeometry, q quantize.Quantizer, qcoeff []int16) bool {
	cn := geo.coeffCount
	tran := &st.tranScratch
	diff, rate, allZero := rdStatsDiffImpl(tran[:cn], qcoeff[:cn], cn, q.AC, geo.txScale)
	if v := qcoeff[0]; v != 0 {
		// The vector pass used the AC step for DC too.
		c := int64(tran[0])
		eAC := c - ((int64(v) * int64(q.AC)) >> geo.txScale)
		eDC := c - ((int64(v) * int64(q.DC)) >> geo.txScale)
		diff += eAC*eAC - eDC*eDC
	}
	st.rdDskip += diff
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
		st.setRDOQArm(0)
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
			st.setRDOQArm(1)
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
	st.rdoqArm = 0
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
	geo, err := txBlockGeometryForTransformSize(ctxReq.Size)
	if err != nil {
		return err
	}
	if !geo.matchesDimensions(w, h) || len(qcoeff) < geo.coeffCount {
		return tile.ErrInvalidDecodeState
	}
	qcoeff = qcoeff[:geo.coeffCount]
	txbResult, err := st.emitInterTXBTyped(w, h, qcoeff, ctxReq, coeffCtx, scan, afterSkip, txType)
	if err != nil {
		return err
	}
	return st.reconInterTXBResult(reconPlane, pred, predStride, stride, px, py, w, h, q, qcoeff, geo, scan, txType, txbResult)
}

// emitInterTXBTyped is the entropy-write half of finishInterTXBTyped: it codes
// the prepared coefficients (dispatching to the specialized trusted-context
// writers where available) and marks the coefficient contexts, without touching
// the reconstruction. The libaom pack phase (av1_write_coeffs_txb in
// av1/encoder/bitstream.c) performs exactly this work when replaying stored
// tokens after encode_sb.
func (st *lossyEncodeState) emitInterTXBTyped(w, h int, qcoeff []int16,
	ctxReq tile.CoeffContextRequest, coeffCtx *tile.CoeffEntropyContext, scan []int16, afterSkip func() error, txType transform.Type) (tile.TXBDecodeResult, error) {
	var txbResult tile.TXBDecodeResult
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
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 1, H4: 1}, blockDims)
			txbResult = tile.WriteCoefficientsTXB4x4Y2DContextTrustedTXTypeArray(st.w, &st.coeffCDFs, qcoeff4, txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
				return tile.TXBDecodeResult{}, err
			}
		} else {
			var err error
			txbResult, err = tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip)
			if err != nil {
				return tile.TXBDecodeResult{}, err
			}
		}
	} else if w == 8 && h == 8 && ctxReq.Plane == 0 && ctxReq.Size == tile.TransformSize8x8 && afterSkip != nil {
		txCDF, txSymbol, ok := st.interTXCDFAndSymbol(ctxReq.Size, txType)
		if ok {
			if coeffCtx == nil {
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 2, H4: 2}, blockDims)
			txbResult = tile.WriteCoefficientsTXB8x8Y2DContextTrustedArray(st.w, &st.coeffCDFs, (*[64]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
				return tile.TXBDecodeResult{}, err
			}
		} else {
			var err error
			txbResult, err = tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip)
			if err != nil {
				return tile.TXBDecodeResult{}, err
			}
		}
	} else if w == 16 && h == 16 && ctxReq.Plane == 0 && ctxReq.Size == tile.TransformSize16x16 && afterSkip != nil && txType == transform.TypeDCTDCT {
		txCDF, txSymbol, ok := st.interTXCDFAndSymbol(ctxReq.Size, txType)
		if ok {
			if coeffCtx == nil {
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 4, H4: 4}, blockDims)
			txbResult = tile.WriteCoefficientsTXB16x16Y2DContextTrustedArray(st.w, &st.coeffCDFs, (*[256]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
				return tile.TXBDecodeResult{}, err
			}
		} else {
			var err error
			txbResult, err = tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip)
			if err != nil {
				return tile.TXBDecodeResult{}, err
			}
		}
	} else if w == 32 && h == 32 && ctxReq.Plane == 0 && ctxReq.Size == tile.TransformSize32x32 && afterSkip != nil && txType == transform.TypeDCTDCT {
		txCDF, txSymbol, ok := st.interTXCDFAndSymbol(ctxReq.Size, txType)
		if ok {
			if coeffCtx == nil {
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			blockDims, ok := ctxReq.PlaneBlock.Dimensions()
			if !ok {
				return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
			}
			txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 8, H4: 8}, blockDims)
			txbResult = tile.WriteCoefficientsTXB32x32Y2DContextTrustedZeroedLevels(st.w, &st.coeffCDFs, (*[1024]int16)(qcoeff), st.levels32Zeroed[:], txbCtx.TXBSkipContext, txbCtx.DCSignContext, txCDF, txSymbol)
			if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
				return tile.TXBDecodeResult{}, err
			}
		} else {
			var err error
			txbResult, err = tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip)
			if err != nil {
				return tile.TXBDecodeResult{}, err
			}
		}
	} else if w == 4 && h == 4 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize4x4 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 1, H4: 1}, blockDims)
		txbResult = tile.WriteCoefficientsTXB4x4UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[16]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
			return tile.TXBDecodeResult{}, err
		}
	} else if w == 8 && h == 8 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize8x8 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 2, H4: 2}, blockDims)
		txbResult = tile.WriteCoefficientsTXB8x8UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[64]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
			return tile.TXBDecodeResult{}, err
		}
	} else if w == 16 && h == 16 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize16x16 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 4, H4: 4}, blockDims)
		txbResult = tile.WriteCoefficientsTXB16x16UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[256]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
			return tile.TXBDecodeResult{}, err
		}
	} else if w == 32 && h == 32 && ctxReq.Plane != 0 && ctxReq.Size == tile.TransformSize32x32 && afterSkip == nil && txType == transform.TypeDCTDCT {
		if coeffCtx == nil {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		blockDims, ok := ctxReq.PlaneBlock.Dimensions()
		if !ok {
			return tile.TXBDecodeResult{}, tile.ErrInvalidDecodeState
		}
		txbCtx := coeffCtx.TXBContextTrusted(ctxReq, tile.TransformDimensions{W4: 8, H4: 8}, blockDims)
		txbResult = tile.WriteCoefficientsTXB32x32UV2DContextTrustedArray(st.w, &st.coeffCDFs, (*[1024]int16)(qcoeff), txbCtx.TXBSkipContext, txbCtx.DCSignContext)
		if err := coeffCtx.MarkTXB(ctxReq, txbResult); err != nil {
			return tile.TXBDecodeResult{}, err
		}
	} else {
		var err error
		txbResult, err = tile.WriteCoefficientsTXBWithContextHook(st.w, &st.coeffCDFs, coeffCtx, ctxReq, transform.Class2D, qcoeff, scan, st.levels, afterSkip)
		if err != nil {
			return tile.TXBDecodeResult{}, err
		}
	}
	return txbResult, nil
}

// reconInterTXBResult is the reconstruction half of finishInterTXBTyped: it
// rebuilds the recon block from the prediction through the decoder's dequant +
// inverse transform, driven by the TXB result (AllZero/EOB) the entropy writer
// produced. The decision pass derives the same result from the coefficients
// alone (interTXBResult), so search/recon never needs the bitstream writer.
func (st *lossyEncodeState) reconInterTXBResult(reconPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16,
	geo txBlockGeometry, scan []int16, txType transform.Type, txbResult tile.TXBDecodeResult) error {
	n := geo.sampleCount
	if txbResult.AllZero {
		copyPredToReconBlock(reconPlane, pred, predStride, stride, px, py, w, h)
		return nil
	}
	if txbResult.EOB == 1 && txType == transform.TypeDCTDCT {
		dc := quantize.DequantizeDCCoeffBitDepthTrusted(qcoeff[0], q, geo.txScale, nil, 8)
		sample, err := transform.InverseDCTDCOnlySampleBitDepth(dc, st.invScratch[:n], geo.size, 8)
		if err != nil {
			return err
		}
		addConstantResidualToPredReconBlock(reconPlane, pred, predStride, stride, px, py, w, h, sample)
		return nil
	}

	dq := &st.dqScratch
	eob := int(txbResult.EOB)
	if txType != transform.TypeIDTX {
		// Collect the same scan-prefix row bound while dequantization already
		// visits those positions, including entries whose coefficient is zero.
		activeRows := quantize.DequantizeBlockScaledBitDepthEOBRowsFlatTrusted(dq[:geo.coeffCount], qcoeff, scan, eob, geo.coeffHeight, q, geo.txScale, 8)
		raw := st.invScratch[:n]
		if err := transform.InverseBlockBitDepthRaw(raw, dq[:geo.coeffCount], geo.coeffHeight, geo.size, txType, 8, activeRows); err != nil {
			return err
		}
		dst := reconPlane[py*stride+px:]
		addRawResidualToPredReconBlock(dst, stride, pred, predStride, raw, w, h)
		return nil
	}

	quantize.DequantizeBlockScaledBitDepthEOBTrusted(dq[:geo.coeffCount], geo.coeffHeight, qcoeff, geo.coeffHeight, scan, eob, geo.coeffWidth, geo.coeffHeight, q, geo.txScale, 8)
	res := &st.invResidual
	if err := transform.InverseBlock(res[:n], w, dq[:geo.coeffCount], geo.coeffHeight, st.invScratch[:n], geo.size, txType); err != nil {
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

func addConstantResidualToPredReconBlockPureGo(reconPlane, pred []byte, predStride, stride, px, py, w, h int, residual int16) {
	rv := int(residual)
	for r := range h {
		row := (py+r)*stride + px
		for c := range w {
			v := int(pred[r*predStride+c]) + rv
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			reconPlane[row+c] = uint8(v)
		}
	}
}

func copyPredToReconBlockPureGo(reconPlane, pred []byte, predStride, stride, px, py, w, h int) {
	for r := range h {
		row := (py+r)*stride + px
		copy(reconPlane[row:row+w], pred[r*predStride:r*predStride+w])
	}
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

// realtimeSADPerBit mirrors libaom's ME LUT setup:
//
//	av1/encoder/rd.c init_me_luts_bd()
//	av1/encoder/ratectrl.c av1_convert_qindex_to_q()
//
// Motion-estimation pricing uses the AC quantizer converted back to the old Q
// scale; the DC step is only used by the RD multiplier path.
func realtimeSADPerBit(q quantize.Quantizer, bitDepth uint8) int {
	scale := 4.0
	switch bitDepth {
	case 10:
		scale = 16.0
	case 12:
		scale = 64.0
	}
	oldQ := float64(q.AC) / scale
	return int(0.0418*oldQ + 2.4107)
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

// forwardTransformBlock8BitResidualTrusted is used only after residuals have
// been formed from 8-bit source and prediction planes. The residual values are
// therefore bounded to [-255, 255], allowing narrow transform kernels to skip
// their defensive full-block range scan.
func forwardTransformBlock8BitResidualTrusted(tran []int32, residual []int16, scratch []int32, w, h int, typ transform.Type) error {
	return transform.ForwardBlock8BitResidualTrusted(
		tran, min(h, 32), residual, w, scratch,
		transform.Size{Width: uint8(w), Height: uint8(h)}, typ,
	)
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
	if err := predictInto(pred, ref.Y, src.YStride, src.Width, src.Height, px, py, 8, 8, mv, false, false, &st.scaledScratch); err != nil {
		return 1 << 59
	}
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, pred, 8, src.YStride, px, py, 8, 8, st.yQuant, st.lumaQ2[:64])
	cost := st.trialTXBBitsY8x8((*[64]int16)(st.lumaQ2[:64])) + st.rdDcode<<7

	halfW, halfH := src.Width/2, src.Height/2
	cpred := st.sadScratch[:16]
	if err := predictInto(cpred, ref.U, src.ChromaStride, halfW, halfH, px/2, py/2, 4, 4, mv, true, true, &st.scaledScratch); err != nil {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(src.U, cpred, 4, src.ChromaStride, px/2, py/2, 4, 4, st.uQuant, st.uQ[:16])
	cost += st.trialTXBBitsUV4x4((*[16]int16)(st.uQ[:16])) + st.rdDcode<<7

	if err := predictInto(cpred, ref.V, src.ChromaStride, halfW, halfH, px/2, py/2, 4, 4, mv, true, true, &st.scaledScratch); err != nil {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(src.V, cpred, 4, src.ChromaStride, px/2, py/2, 4, 4, st.vQuant, st.vQ[:16])
	cost += st.trialTXBBitsUV4x4((*[16]int16)(st.vQ[:16])) + st.rdDcode<<7
	return cost
}

func (st *lossyEncodeState) trialInterCost16x16(src SourceFrame420, ref SourceFrame420, px, py int, mv motion.Vector) int64 {
	pred := st.sadScratch[:256]
	if err := predictInto(pred, ref.Y, src.YStride, src.Width, src.Height, px, py, 16, 16, mv, false, false, &st.scaledScratch); err != nil {
		return 1 << 59
	}
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, pred, 16, src.YStride, px, py, 16, 16, st.yQuant, st.lumaQ2[:256])
	cost := st.trialTXBBitsY16x16((*[256]int16)(st.lumaQ2[:256])) + st.rdDcode<<7

	halfW, halfH := src.Width/2, src.Height/2
	cpred := st.sadScratch[:64]
	if err := predictInto(cpred, ref.U, src.ChromaStride, halfW, halfH, px/2, py/2, 8, 8, mv, true, true, &st.scaledScratch); err != nil {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(src.U, cpred, 8, src.ChromaStride, px/2, py/2, 8, 8, st.uQuant, st.uQ[:64])
	cost += st.trialTXBBitsUV8x8((*[64]int16)(st.uQ[:64])) + st.rdDcode<<7

	if err := predictInto(cpred, ref.V, src.ChromaStride, halfW, halfH, px/2, py/2, 8, 8, mv, true, true, &st.scaledScratch); err != nil {
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
	case w == 4 && h == 16:
		return transform.ForwardBlock(tran, 16, residual, 4, tran, transform.Size{Width: 4, Height: 16}, transform.TypeDCTDCT)
	case w == 16 && h == 4:
		return transform.ForwardBlock(tran, 4, residual, 16, tran, transform.Size{Width: 16, Height: 4}, transform.TypeDCTDCT)
	case w == 8 && h == 32:
		return transform.ForwardBlock(tran, 32, residual, 8, tran, transform.Size{Width: 8, Height: 32}, transform.TypeDCTDCT)
	case w == 32 && h == 8:
		return transform.ForwardBlock(tran, 8, residual, 32, tran, transform.Size{Width: 32, Height: 8}, transform.TypeDCTDCT)
	case w == 16 && h == 64:
		return transform.ForwardBlock(tran, 32, residual, 16, tran, transform.Size{Width: 16, Height: 64}, transform.TypeDCTDCT)
	case w == 64 && h == 16:
		return transform.ForwardBlock(tran, 16, residual, 64, tran, transform.Size{Width: 64, Height: 16}, transform.TypeDCTDCT)
	case w == 64 && h == 64:
		return transform.ForwardDCT64x64(tran, 32, residual, 64)
	case w == 64 && h == 32:
		return transform.ForwardDCT64x32(tran, 32, residual, 64)
	case w == 32 && h == 64:
		return transform.ForwardDCT32x64(tran, 32, residual, 32)
	}
	return transform.ErrInvalidTransform
}

// fullPelDiamondSearch finds the full-pel offset (dx, dy) that minimizes the
// luma SAD of the n x n block at (px, py) against the reference plane, keeping
// the offset window fully inside the frame.
func fullPelDiamondSearch(src, ref []byte, stride, width, height, px, py, n int) (int, int, int) {
	return fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, n, 0, 0, fullPelReach)
}

// fullPelReach is the libaom mesh-search radius of the golden probe.
const fullPelReach = 8

// goldenRefPruneMaxDevPct ports SVT-AV1's dist_based_ref_pruning
// (svt_aom_set_dist_based_ref_pruning_controls + perform_md_reference_pruning,
// Source/Lib/Codec/enc_mode_config.c / product_coding_loop.c). SVT prunes the
// expensive MD evaluation of a reference whose ME distortion deviates from the
// best reference by more than max_dev_to_best[PA_ME_GROUP]; for preset-12
// low-delay leaves that threshold is 30 (dist_based_ref_pruning level 6). Here
// LAST is the closest/best reference, so once GOLDEN's full-pel ME distortion is
// more than 30% worse than LAST it can no longer overtake it — SVT drops its
// NEW-MV group, so we skip GOLDEN's subpel refinement and compound evaluation.
const goldenRefPruneMaxDevPct = 30

// fullPelDiamondSearchSeeded recenters libaom's mesh search on a
// coarse-search seed (the hierarchical pre-pass vector).
func fullPelDiamondSearchSeeded(src, ref []byte, stride, width, height, px, py, n int, seedDX, seedDY, reach int) (int, int, int) {
	return fullPelLibaomMeshSearch(src, ref, stride, width, height, px, py, n, n, seedDX, seedDY, reach)
}

func fullPelLibaomMeshSearch(src, ref []byte, stride, width, height, px, py, bw, bh int, seedDX, seedDY, reach int) (int, int, int) {
	seedDX = minInt(maxInt(seedDX, -px), width-bw-px)
	seedDY = minInt(maxInt(seedDY, -py), height-bh-py)
	bestDX, bestDY, bestSAD := fullPelLibaomExhaustiveMeshSearch(src, ref, stride, width, height, px, py, bw, bh, seedDX, seedDY, reach, 4, -1)
	// The refinement mesh is centred on the first mesh's winner, whose SAD it
	// already has.
	return fullPelLibaomExhaustiveMeshSearch(src, ref, stride, width, height, px, py, bw, bh, bestDX, bestDY, 2, 2, bestSAD)
}

// Ported from libaom av1/encoder/mcomp.c exhaustive_mesh_search. The local
// caller supplies the mesh pattern used by the realtime HME-seeded search.
// centerSAD, when not negative, is the SAD at (startDX, startDY) and is reused
// instead of recomputed. The centre itself is never re-scored inside the mesh:
// its SAD cannot be below the running best, which starts there.
func fullPelLibaomExhaustiveMeshSearch(src, ref []byte, stride, width, height, px, py, bw, bh int, startDX, startDY, searchRange, step, centerSAD int) (int, int, int) {
	if step < 1 {
		step = 1
	}
	clampedDX := minInt(maxInt(startDX, -px), width-bw-px)
	clampedDY := minInt(maxInt(startDY, -py), height-bh-py)
	if clampedDX != startDX || clampedDY != startDY {
		centerSAD = -1
	}
	startDX, startDY = clampedDX, clampedDY
	minDX := -px
	maxDX := width - bw - px
	minDY := -py
	maxDY := height - bh - py

	startCol := maxInt(-searchRange, minDX-startDX)
	endCol := minInt(searchRange, maxDX-startDX)
	startRow := maxInt(-searchRange, minDY-startDY)
	endRow := minInt(searchRange, maxDY-startDY)

	base := py*stride + px
	srcBlock := src[base:]
	bestDX, bestDY := startDX, startDY
	bestSAD := centerSAD
	if bestSAD < 0 {
		bestSAD = fullPelSearchSAD(srcBlock, ref[base+startDY*stride+startDX:], stride, bw, bh, 1<<30)
	}
	colStep := step
	if step <= 1 {
		colStep = 4
	}
	for row := startRow; row <= endRow; row += step {
		col := startCol
		if step == 4 && bw == bh {
			for ; col+3*step <= endCol; col += step * 4 {
				dx := startDX + col
				dy := startDY + row
				s0, s1, s2, s3 := fullPelSearchSAD4Step4(srcBlock, ref, base, stride, bw, bh, dx, dy)
				if s0 < bestSAD {
					bestSAD, bestDX, bestDY = s0, dx, dy
				}
				if s1 < bestSAD {
					bestSAD, bestDX, bestDY = s1, dx+step, dy
				}
				if s2 < bestSAD {
					bestSAD, bestDX, bestDY = s2, dx+2*step, dy
				}
				if s3 < bestSAD {
					bestSAD, bestDX, bestDY = s3, dx+3*step, dy
				}
			}
		}
		for ; col <= endCol; col += colStep {
			dx := startDX + col
			dy := startDY + row
			if step <= 1 && col+3 <= endCol && bw == bh {
				s0, s1, s2, s3 := fullPelSearchSAD4(srcBlock, ref, base, stride, bw, bh,
					dx, dy, dx+1, dy, dx+2, dy, dx+3, dy)
				if s0 < bestSAD {
					bestSAD, bestDX, bestDY = s0, dx, dy
				}
				if s1 < bestSAD {
					bestSAD, bestDX, bestDY = s1, dx+1, dy
				}
				if s2 < bestSAD {
					bestSAD, bestDX, bestDY = s2, dx+2, dy
				}
				if s3 < bestSAD {
					bestSAD, bestDX, bestDY = s3, dx+3, dy
				}
				continue
			}
			if col == 0 && row == 0 {
				continue
			}
			if s := fullPelSearchSAD(srcBlock, ref[base+dy*stride+dx:], stride, bw, bh, bestSAD); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	return bestDX, bestDY, bestSAD
}

func fullPelSearchSAD(srcBlock, refBlock []byte, stride, bw, bh, limit int) int {
	if bw == bh {
		switch bw {
		case 8:
			return sad8x8(srcBlock, refBlock, stride, limit)
		case 16:
			return sad16x16(srcBlock, refBlock, stride)
		case 32:
			return sad32x32(srcBlock, refBlock, stride)
		case 64:
			return sad64x64(srcBlock, refBlock, stride)
		}
	}
	return sadRectDualBlock(srcBlock, stride, refBlock, stride, bw, bh)
}

func fullPelSearchSAD4Step4(srcBlock, ref []byte, base, stride, bw, bh int, dx, dy int) (int, int, int, int) {
	ref0 := ref[base+dy*stride+dx:]
	if bw == bh {
		switch bw {
		case 8:
			return sad8x8x4Step4(srcBlock, ref0, stride)
		case 16:
			return sad16x16x4Step4(srcBlock, ref0, stride)
		case 32:
			return sad32x32x4Step4(srcBlock, ref0, stride)
		case 64:
			return sad64x64x4Step4(srcBlock, ref0, stride)
		}
	}
	return fullPelSearchSAD(srcBlock, ref0, stride, bw, bh, 1<<30),
		fullPelSearchSAD(srcBlock, ref0[4:], stride, bw, bh, 1<<30),
		fullPelSearchSAD(srcBlock, ref0[8:], stride, bw, bh, 1<<30),
		fullPelSearchSAD(srcBlock, ref0[12:], stride, bw, bh, 1<<30)
}

func fullPelSearchSAD4(srcBlock, ref []byte, base, stride, bw, bh int, dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3 int) (int, int, int, int) {
	ref0 := ref[base+dy0*stride+dx0:]
	ref1 := ref[base+dy1*stride+dx1:]
	ref2 := ref[base+dy2*stride+dx2:]
	ref3 := ref[base+dy3*stride+dx3:]
	if bw == bh {
		switch bw {
		case 8:
			return sad8x8x4(srcBlock, ref0, ref1, ref2, ref3, stride)
		case 16:
			return sad16x16x4(srcBlock, ref0, ref1, ref2, ref3, stride)
		case 32:
			return sad32x32x4(srcBlock, ref0, ref1, ref2, ref3, stride)
		case 64:
			return sad64x64x4(srcBlock, ref0, ref1, ref2, ref3, stride)
		}
	}
	return fullPelSearchSAD(srcBlock, ref0, stride, bw, bh, 1<<30),
		fullPelSearchSAD(srcBlock, ref1, stride, bw, bh, 1<<30),
		fullPelSearchSAD(srcBlock, ref2, stride, bw, bh, 1<<30),
		fullPelSearchSAD(srcBlock, ref3, stride, bw, bh, 1<<30)
}
