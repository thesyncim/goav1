package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// keyframe.go assembles a complete decodable LOSSLESS all-intra keyframe
// temporal unit. The tile symbol stream mirrors the decoder's keyframe block
// loop exactly — the superblock iteration reuses the decoder's carrier
// load/store via tile.WalkBlockLoopWrite, and the per-block symbols follow
// tile/block_loop.go decodeBlockVisit order — using the round-trip-verified
// writers. Headers come from this package's byte-verified OBU emitters.
//
// Scope (extended incrementally): 8-bit 4:2:0 and monochrome, frame dimensions
// multiples of 8 (single tile, edge superblocks split down to fully visible
// 8x8 blocks), DC intra everywhere, qindex 0 (lossless: WHT 4x4 transforms,
// recon == source so prediction neighbors come straight from the source
// planes).

// SourceFrame420 is one caller-owned 8-bit 4:2:0 source picture.
type SourceFrame420 struct {
	Y, U, V               []byte
	YStride, ChromaStride int
	Width, Height         int
}

// SourceFrameMono is one caller-owned 8-bit monochrome source picture.
type SourceFrameMono struct {
	Y             []byte
	YStride       int
	Width, Height int
}

// SourceFrameMono16 is one caller-owned 10/12-bit monochrome source picture.
// Samples are stored as uint16 values in the low bits and YStride is in
// samples, not bytes.
type SourceFrameMono16 struct {
	Y             []uint16
	YStride       int
	Width, Height int
	BitDepth      uint8
}

// SourceFrame42016 is one caller-owned 10/12-bit 4:2:0 source picture.
// Samples are stored as uint16 values in the low bits and strides are in
// samples, not bytes.
type SourceFrame42016 struct {
	Y, U, V               []uint16
	YStride, ChromaStride int
	Width, Height         int
	BitDepth              uint8
}

// EncodeLosslessKeyframe encodes src (dimensions must be multiples of 8) as
// one low-overhead temporal unit: temporal delimiter, sequence header, complete
// lossless keyframe header, and a single-tile tile group carrying the coded
// blocks. The returned bytes decode in the goav1 decoder to a frame that is
// bit-exactly src.
func EncodeLosslessKeyframe(src SourceFrame420) ([]byte, error) {
	// Multiples of 8 guarantee an even MI grid, so the partition walk always
	// terminates in fully-inside blocks of at least 8x8 (no frame-edge block
	// overhang and chroma present on every block at 4:2:0).
	if src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0 {
		return nil, fmt.Errorf("encoder: frame dimensions must be positive multiples of 8, got %dx%d", src.Width, src.Height)
	}
	tilePayload, err := encodeLosslessKeyframeTile(src)
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}

	seq := losslessKeyframeSequence(src.Width, src.Height)
	header := losslessKeyframeHeader(src.Width, src.Height)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, fmt.Errorf("size header TU: %w", err)
	}

	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, err
	}

	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, nil
}

// EncodeLosslessMonochromeKeyframe encodes src as one low-overhead temporal
// unit carrying a native AV1 monochrome lossless keyframe.
func EncodeLosslessMonochromeKeyframe(src SourceFrameMono) ([]byte, error) {
	if err := validateSourceFrameMono(src); err != nil {
		return nil, err
	}
	tilePayload, err := encodeLosslessMonochromeKeyframeTile(src)
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}

	seq := losslessMonochromeKeyframeSequence(src.Width, src.Height)
	header := losslessKeyframeHeaderForSequence(seq, src.Width, src.Height)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, fmt.Errorf("size header TU: %w", err)
	}

	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, err
	}

	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, nil
}

// EncodeLosslessHighBitDepthMonochromeKeyframe encodes src as one native
// 10/12-bit AV1 monochrome lossless keyframe temporal unit.
func EncodeLosslessHighBitDepthMonochromeKeyframe(src SourceFrameMono16) ([]byte, error) {
	if err := validateSourceFrameMono16(src); err != nil {
		return nil, err
	}
	tilePayload, err := encodeLosslessHighBitDepthMonochromeKeyframeTile(src)
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}

	seq := losslessHighBitDepthMonochromeKeyframeSequence(src.Width, src.Height, src.BitDepth)
	header := losslessKeyframeHeaderForSequence(seq, src.Width, src.Height)

	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, fmt.Errorf("size header TU: %w", err)
	}

	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	group := make([]byte, 0, groupSize)
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, []TilePayload{{Data: tilePayload}})
	if err != nil {
		return nil, fmt.Errorf("append tile group: %w", err)
	}
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, err
	}

	out := make([]byte, 0, headerSize+groupOBUSize)
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, fmt.Errorf("append tile group OBU: %w", err)
	}
	return out, nil
}

func validateSourceFrameMono(src SourceFrameMono) error {
	if src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0 {
		return fmt.Errorf("encoder: frame dimensions must be positive multiples of 8, got %dx%d", src.Width, src.Height)
	}
	if src.YStride < src.Width {
		return fmt.Errorf("encoder: monochrome Y stride %d is smaller than width %d", src.YStride, src.Width)
	}
	if src.Height > 0 && src.YStride > (int(^uint(0)>>1)-(src.Width-1))/(src.Height-1) {
		return fmt.Errorf("encoder: monochrome Y plane dimensions overflow int")
	}
	need := (src.Height-1)*src.YStride + src.Width
	if len(src.Y) < need {
		return fmt.Errorf("encoder: monochrome Y plane is too short: got %d bytes, need %d", len(src.Y), need)
	}
	return nil
}

func validateSourceFrameMono16(src SourceFrameMono16) error {
	if src.BitDepth != 10 && src.BitDepth != 12 {
		return fmt.Errorf("encoder: high-bit-depth monochrome bit depth must be 10 or 12, got %d", src.BitDepth)
	}
	if src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0 {
		return fmt.Errorf("encoder: frame dimensions must be positive multiples of 8, got %dx%d", src.Width, src.Height)
	}
	if src.YStride < src.Width {
		return fmt.Errorf("encoder: monochrome Y stride %d is smaller than width %d", src.YStride, src.Width)
	}
	if src.Height > 0 && src.YStride > (int(^uint(0)>>1)-(src.Width-1))/(src.Height-1) {
		return fmt.Errorf("encoder: monochrome Y plane dimensions overflow int")
	}
	need := (src.Height-1)*src.YStride + src.Width
	if len(src.Y) < need {
		return fmt.Errorf("encoder: monochrome Y plane is too short: got %d samples, need %d", len(src.Y), need)
	}
	maxSample := uint16((1 << src.BitDepth) - 1)
	for y := range src.Height {
		row := src.Y[y*src.YStride : y*src.YStride+src.Width]
		for x, sample := range row {
			if sample > maxSample {
				return fmt.Errorf("encoder: monochrome Y sample (%d,%d)=%d exceeds %d-bit maximum %d", x, y, sample, src.BitDepth, maxSample)
			}
		}
	}
	return nil
}

func validateSourceFrame42016(src SourceFrame42016) error {
	return validateSourceFrameColor16(src, parser.ColorConfig{BitDepth: src.BitDepth, SubsamplingX: true, SubsamplingY: true}, "4:2:0")
}

func validateSourceFrameColor16(src SourceFrame42016, color parser.ColorConfig, label string) error {
	return validateSourceFrameColor16WithGeometry(src, color, label, true)
}

func validateSourceRenderFrameColor16(src SourceFrame42016, color parser.ColorConfig, label string) error {
	return validateSourceFrameColor16WithGeometry(src, color, label, false)
}

func validateSourceFrameColor16WithGeometry(src SourceFrame42016, color parser.ColorConfig, label string, requireCodedMultiple bool) error {
	if src.BitDepth != 10 && src.BitDepth != 12 {
		return fmt.Errorf("encoder: high-bit-depth %s bit depth must be 10 or 12, got %d", label, src.BitDepth)
	}
	if requireCodedMultiple && (src.Width <= 0 || src.Height <= 0 || src.Width%8 != 0 || src.Height%8 != 0) {
		return fmt.Errorf("encoder: frame dimensions must be positive multiples of 8, got %dx%d", src.Width, src.Height)
	}
	if !requireCodedMultiple && (src.Width <= 0 || src.Height <= 0) {
		return fmt.Errorf("encoder: frame dimensions must be positive, got %dx%d", src.Width, src.Height)
	}
	if color.MonoChrome || color.BitDepth != src.BitDepth {
		return ErrInvalidFrame
	}
	chromaWidth := chromaWidthForColor(src.Width, color)
	chromaHeight := chromaHeightForColor(src.Height, color)
	if src.YStride < src.Width {
		return fmt.Errorf("encoder: %s Y stride %d is smaller than width %d", label, src.YStride, src.Width)
	}
	if src.ChromaStride < chromaWidth {
		return fmt.Errorf("encoder: %s chroma stride %d is smaller than chroma width %d", label, src.ChromaStride, chromaWidth)
	}
	maxInt := int(^uint(0) >> 1)
	if src.Height > 0 && src.YStride > (maxInt-(src.Width-1))/(src.Height-1) {
		return fmt.Errorf("encoder: %s Y plane dimensions overflow int", label)
	}
	if chromaHeight > 0 && src.ChromaStride > (maxInt-(chromaWidth-1))/(chromaHeight-1) {
		return fmt.Errorf("encoder: %s chroma plane dimensions overflow int", label)
	}
	yNeed := (src.Height-1)*src.YStride + src.Width
	chromaNeed := (chromaHeight-1)*src.ChromaStride + chromaWidth
	if len(src.Y) < yNeed {
		return fmt.Errorf("encoder: %s Y plane is too short: got %d samples, need %d", label, len(src.Y), yNeed)
	}
	if len(src.U) < chromaNeed {
		return fmt.Errorf("encoder: %s U plane is too short: got %d samples, need %d", label, len(src.U), chromaNeed)
	}
	if len(src.V) < chromaNeed {
		return fmt.Errorf("encoder: %s V plane is too short: got %d samples, need %d", label, len(src.V), chromaNeed)
	}
	maxSample := uint16((1 << src.BitDepth) - 1)
	if err := validateSourcePlaneColor16Samples(label, "Y", src.Y, src.YStride, src.Width, src.Height, maxSample, src.BitDepth); err != nil {
		return err
	}
	if err := validateSourcePlaneColor16Samples(label, "U", src.U, src.ChromaStride, chromaWidth, chromaHeight, maxSample, src.BitDepth); err != nil {
		return err
	}
	return validateSourcePlaneColor16Samples(label, "V", src.V, src.ChromaStride, chromaWidth, chromaHeight, maxSample, src.BitDepth)
}

func validateSourcePlane42016Samples(name string, samples []uint16, stride, width, height int, maxSample uint16, bitDepth uint8) error {
	return validateSourcePlaneColor16Samples("4:2:0", name, samples, stride, width, height, maxSample, bitDepth)
}

func validateSourcePlaneColor16Samples(label string, name string, samples []uint16, stride, width, height int, maxSample uint16, bitDepth uint8) error {
	for y := range height {
		row := samples[y*stride : y*stride+width]
		for x, sample := range row {
			if sample > maxSample {
				return fmt.Errorf("encoder: %s %s sample (%d,%d)=%d exceeds %d-bit maximum %d", label, name, x, y, sample, bitDepth, maxSample)
			}
		}
	}
	return nil
}

func losslessKeyframeSequence(width, height int) SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        uint32(width),
		MaxFrameHeight:       uint32(height),
		Use128x128Superblock: false,
		EnableCDEF:           true,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func losslessMonochromeKeyframeSequence(width, height int) SequenceHeader {
	seq := losslessKeyframeSequence(width, height)
	seq.ColorConfig = SequenceColorConfig{
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
	}
	return seq
}

func losslessHighBitDepthMonochromeKeyframeSequence(width, height int, bitDepth uint8) SequenceHeader {
	seq := losslessMonochromeKeyframeSequence(width, height)
	if bitDepth == 12 {
		seq.Profile = Profile2
	}
	seq.ColorConfig.BitDepth = bitDepth
	return seq
}

func losslessKeyframeHeader(width, height int) IntraFrameHeaderParams {
	return losslessKeyframeHeaderForSequence(losslessKeyframeSequence(width, height), width, height)
}

func losslessKeyframeHeaderForSequence(seq SequenceHeader, width, height int) IntraFrameHeaderParams {
	sbCols := uint16((width + 63) / 64)
	sbRows := uint16((height + 63) / 64)
	tiles := TileInfo{
		RefreshContext: true,
		SBCols:         sbCols,
		SBRows:         sbRows,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [MaxTileCols + 1]uint16{0, sbCols},
		RowStartSB:     [MaxTileRows + 1]uint16{0, sbRows},
	}
	// validateTileInfo requires the caller to restate the derived min/max
	// tile-log2 bounds; compute them with the package's own derivation.
	if derived, err := deriveEncoderTileInfo(seq, uint32(width), uint32(height), tiles); err == nil {
		tiles.MinLog2Cols = derived.MinLog2Cols
		tiles.MaxLog2Cols = derived.MaxLog2Cols
		tiles.MaxLog2Rows = derived.MaxLog2Rows
		tiles.MinLog2Tiles = derived.MinLog2Tiles
	}
	return IntraFrameHeaderParams{
		Prefix: FrameHeaderPrefix{
			FrameType:          FrameHeaderTypeKey,
			ShowFrame:          true,
			ErrorResilientMode: true,
			ForceIntegerMV:     true, // inferred 1 for key/intra frames
			FrameSizeOverride:  uint32(width) != seq.MaxFrameWidth || uint32(height) != seq.MaxFrameHeight,
			PrimaryRefFrame:    EncoderPrimaryRefNone,
		},
		Size: IntraFrameSize{
			UpscaledWidth:       uint32(width),
			Height:              uint32(height),
			RenderWidth:         uint32(width),
			RenderHeight:        uint32(height),
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		Tile:         tiles,
		Quantization: QuantizationParams{BaseQIdx: 0},
		LoopFilter: LoopFilterParams{
			ModeRefDeltaEnabled: true,
			ModeRefDeltaUpdate:  true,
			Deltas:              defaultLoopFilterDeltas(),
		},
		TransformRef: TransformReferenceParams{
			TransformMode: TransformMode4x4Only,
			ReferenceMode: ReferenceModeSingle,
		},
		AllLossless: true,
	}
}

// encodeLosslessKeyframeTile codes the tile: one PARTITION_NONE 64x64 DC-intra
// block per superblock, residuals as 4x4 WHT transform blocks, with the
// decoder's own cross-superblock context carry.
func encodeLosslessKeyframeTile(src SourceFrame420) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	var modeCDFs tile.BlockModeCDFs
	var intraCDFs tile.IntraModeCDFs
	var coeffCDFs tile.CoeffCDFs
	if err := partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := modeCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := intraCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := coeffCDFs.InitDefault(0); err != nil {
		return nil, err
	}

	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := transform.FillDefaultScan(scan, inverse, transform.Size{Width: 4, Height: 4}, transform.Class2D); err != nil {
		return nil, err
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize4x4)
	if err != nil {
		return nil, err
	}
	levels := make([]uint8, scratchLen)

	w := entropy.NewWriter(make([]byte, 0, 1<<18))

	miCols := uint16(src.Width / 4)
	miRows := uint16(src.Height / 4)
	const sbSizeMIB = 16 // 64x64 superblocks
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
	// Largest-block tree: PARTITION_NONE wherever the node's halves start
	// inside the frame (overhanging leaves are legal; their residual codes
	// only the visible transform blocks), SPLIT when only one direction is
	// available. The walker forces SPLIT when neither is, without consulting
	// the decider.
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if haveRight && haveBottom {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return encodeLosslessKeyframeBlock(&w, src, block, &modeCDFs, &intraCDFs, &coeffCDFs, scratch, scan, levels)
	}
	if err := tile.WalkBlockLoopWrite(&w, &partCDFs, &scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

// encodeLosslessMonochromeKeyframeTile mirrors encodeLosslessKeyframeTile but
// emits only luma intra mode and luma residual symbols.
func encodeLosslessMonochromeKeyframeTile(src SourceFrameMono) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	var modeCDFs tile.BlockModeCDFs
	var intraCDFs tile.IntraModeCDFs
	var coeffCDFs tile.CoeffCDFs
	if err := partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := modeCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := intraCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := coeffCDFs.InitDefault(0); err != nil {
		return nil, err
	}

	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := transform.FillDefaultScan(scan, inverse, transform.Size{Width: 4, Height: 4}, transform.Class2D); err != nil {
		return nil, err
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize4x4)
	if err != nil {
		return nil, err
	}
	levels := make([]uint8, scratchLen)

	w := entropy.NewWriter(make([]byte, 0, 1<<18))

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
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if haveRight && haveBottom {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return encodeLosslessMonochromeKeyframeBlock(&w, src, block, &modeCDFs, &intraCDFs, &coeffCDFs, scratch, scan, levels)
	}
	if err := tile.WalkBlockLoopWrite(&w, &partCDFs, &scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

// encodeLosslessHighBitDepthMonochromeKeyframeTile mirrors the 8-bit
// monochrome lossless path while reading uint16 samples and signaling the
// high-bit-depth sequence header.
func encodeLosslessHighBitDepthMonochromeKeyframeTile(src SourceFrameMono16) ([]byte, error) {
	var partCDFs tile.PartitionCDFs
	var modeCDFs tile.BlockModeCDFs
	var intraCDFs tile.IntraModeCDFs
	var coeffCDFs tile.CoeffCDFs
	if err := partCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := modeCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := intraCDFs.InitDefault(); err != nil {
		return nil, err
	}
	if err := coeffCDFs.InitDefault(0); err != nil {
		return nil, err
	}

	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := transform.FillDefaultScan(scan, inverse, transform.Size{Width: 4, Height: 4}, transform.Class2D); err != nil {
		return nil, err
	}
	scratchLen, err := tile.CoeffLevelsScratchLen(tile.TransformSize4x4)
	if err != nil {
		return nil, err
	}
	levels := make([]uint8, scratchLen)

	w := entropy.NewWriter(make([]byte, 0, 1<<18))

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
	decide := func(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
		if haveRight && haveBottom {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	visit := func(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
		return encodeLosslessHighBitDepthMonochromeKeyframeBlock(&w, src, block, &modeCDFs, &intraCDFs, &coeffCDFs, scratch, scan, levels)
	}
	if err := tile.WalkBlockLoopWrite(&w, &partCDFs, &scratch, carrier, walkReq, sbSizeMIB, decide, visit); err != nil {
		return nil, err
	}
	return w.Finish()
}

// encodeLosslessKeyframeBlock codes one keyframe block in the decoder's
// decodeBlockVisit symbol order: skip_transform, mode-context mark, luma DC
// mode, chroma DC mode, then the residual transform blocks (all luma 4x4 TXBs
// in raster order, then U, then V — the <=64px single-unit residual order).
func encodeLosslessKeyframeBlock(w *entropy.Writer, src SourceFrame420, block tile.BlockVisit,
	modeCDFs *tile.BlockModeCDFs, intraCDFs *tile.IntraModeCDFs, coeffCDFs *tile.CoeffCDFs,
	scratch *tile.BlockLoopScratch, scan []int16, levels []uint8) error {

	dims, ok := block.Size.Dimensions()
	if !ok || dims.W4 < 2 || dims.H4 < 2 {
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	color := parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	// 1) skip_transform = 0 (residual present).
	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(w, modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	// 2) CDEF disabled (lossless), no symbol. 3) Mark the prefix exactly as the
	// decoder does after reading it.
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	// 4) deltas disabled, no symbols. 5) keyframe intra: no intrabc symbol
	// (allow_intrabc=0), luma mode then chroma mode. DC is non-directional so
	// no angle deltas; palette and filter-intra are gated off by the headers.
	if err := tile.WriteLumaIntraMode(w, intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
	}, tile.IntraModeDC); err != nil {
		return fmt.Errorf("luma mode: %w", err)
	}
	cflAllowed, err := tile.ChromaIntraCFLAllowed(block.Size, color, true)
	if err != nil {
		return fmt.Errorf("cfl allowed: %w", err)
	}
	if err := tile.WriteChromaIntraMode(w, intraCDFs, tile.ChromaIntraModeRequest{
		Size: block.Size, LumaMode: tile.IntraModeDC, CFLAllowed: cflAllowed,
	}, tile.ChromaIntraModeDC, tile.CFLAlphaResult{}); err != nil {
		return fmt.Errorf("chroma mode: %w", err)
	}

	// 6) residual: 4x4 WHT TXBs. Luma raster over the block, then U, then V
	// (every block here is <=64px, a single residual unit). Frame-edge blocks
	// may overhang the frame: only the visible transform blocks are coded,
	// exactly as the decoder's residual loop skips rows/cols past the MI grid.
	// With dimensions that are multiples of 8 the visible extent is whole 4x4
	// units, so every coded TXB is fully inside the frame.
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	for ty := range int(block.VisibleH4) {
		for tx := range int(block.VisibleW4) {
			if err := encodeLosslessTXB(w, coeffCDFs, coeffCtx, tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: block.Size,
				Size:       tile.TransformSize4x4,
				X4:         block.X4 + uint8(tx),
				Y4:         block.Y4 + uint8(ty),
			}, src.Y, src.YStride, lumaPX+tx*4, lumaPY+ty*4, scan, levels); err != nil {
				return fmt.Errorf("luma txb (%d,%d): %w", tx, ty, err)
			}
		}
	}
	chromaBlock, err := tile.PlaneBlockSize(block.Size, color, 1)
	if err != nil {
		return fmt.Errorf("chroma plane block: %w", err)
	}
	chromaPX := lumaPX / 2
	chromaPY := lumaPY / 2
	chromaW4 := int(block.VisibleW4) / 2
	chromaH4 := int(block.VisibleH4) / 2
	for plane := 1; plane <= 2; plane++ {
		data := src.U
		if plane == 2 {
			data = src.V
		}
		for ty := range chromaH4 {
			for tx := range chromaW4 {
				if err := encodeLosslessTXB(w, coeffCDFs, coeffCtx, tile.CoeffContextRequest{
					Plane:      uint8(plane),
					PlaneBlock: chromaBlock,
					Size:       tile.TransformSize4x4,
					X4:         block.X4/2 + uint8(tx),
					Y4:         block.Y4/2 + uint8(ty),
				}, data, src.ChromaStride, chromaPX+tx*4, chromaPY+ty*4, scan, levels); err != nil {
					return fmt.Errorf("chroma %d txb (%d,%d): %w", plane, tx, ty, err)
				}
			}
		}
	}
	return nil
}

func encodeLosslessMonochromeKeyframeBlock(w *entropy.Writer, src SourceFrameMono, block tile.BlockVisit,
	modeCDFs *tile.BlockModeCDFs, intraCDFs *tile.IntraModeCDFs, coeffCDFs *tile.CoeffCDFs,
	scratch *tile.BlockLoopScratch, scan []int16, levels []uint8) error {

	dims, ok := block.Size.Dimensions()
	if !ok || dims.W4 < 2 || dims.H4 < 2 {
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(w, modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	if err := tile.WriteLumaIntraMode(w, intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
	}, tile.IntraModeDC); err != nil {
		return fmt.Errorf("luma mode: %w", err)
	}

	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	for ty := range int(block.VisibleH4) {
		for tx := range int(block.VisibleW4) {
			if err := encodeLosslessTXB(w, coeffCDFs, coeffCtx, tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: block.Size,
				Size:       tile.TransformSize4x4,
				X4:         block.X4 + uint8(tx),
				Y4:         block.Y4 + uint8(ty),
			}, src.Y, src.YStride, lumaPX+tx*4, lumaPY+ty*4, scan, levels); err != nil {
				return fmt.Errorf("luma txb (%d,%d): %w", tx, ty, err)
			}
		}
	}
	return nil
}

func encodeLosslessHighBitDepthMonochromeKeyframeBlock(w *entropy.Writer, src SourceFrameMono16, block tile.BlockVisit,
	modeCDFs *tile.BlockModeCDFs, intraCDFs *tile.IntraModeCDFs, coeffCDFs *tile.CoeffCDFs,
	scratch *tile.BlockLoopScratch, scan []int16, levels []uint8) error {

	dims, ok := block.Size.Dimensions()
	if !ok || dims.W4 < 2 || dims.H4 < 2 {
		return fmt.Errorf("encoder: unexpected block %+v", block)
	}
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx

	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(w, modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	if err := tile.WriteLumaIntraMode(w, intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
	}, tile.IntraModeDC); err != nil {
		return fmt.Errorf("luma mode: %w", err)
	}

	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	for ty := range int(block.VisibleH4) {
		for tx := range int(block.VisibleW4) {
			if err := encodeLosslessTXB16(w, coeffCDFs, coeffCtx, tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: block.Size,
				Size:       tile.TransformSize4x4,
				X4:         block.X4 + uint8(tx),
				Y4:         block.Y4 + uint8(ty),
			}, src.Y, src.YStride, lumaPX+tx*4, lumaPY+ty*4, src.BitDepth, scan, levels); err != nil {
				return fmt.Errorf("luma txb (%d,%d): %w", tx, ty, err)
			}
		}
	}
	return nil
}

// encodeLosslessTXB DC-predicts one 4x4 transform block from the source plane
// (lossless: reconstruction equals source, so source neighbors are the
// decoder's reconstructed neighbors), forward-WHT-transforms the residual,
// quantizes at qindex 0 (exact >>2 of the x4-scaled WHT output), and codes the
// coefficients through the carrier-context writer.
func encodeLosslessTXB(w *entropy.Writer, cdfs *tile.CoeffCDFs, ctx *tile.CoeffEntropyContext,
	ctxReq tile.CoeffContextRequest, plane []byte, stride, px, py int,
	scan []int16, levels []uint8) error {

	dc := dcPredictN(plane, stride, px, py, 4, py > 0, px > 0)

	var residual [16]int16
	for r := range 4 {
		row := (py+r)*stride + px
		for c := range 4 {
			residual[r*4+c] = int16(plane[row+c]) - int16(dc)
		}
	}
	var wht [16]int32
	if err := transform.ForwardWHT4x4(wht[:], 4, residual[:], 4); err != nil {
		return err
	}
	var coeffs [16]int16
	for i, v := range wht {
		coeffs[i] = int16(v >> 2) // qindex 0: dequant 4 restores the x4 WHT scale
	}
	_, err := tile.WriteCoefficientsTXBWithContext(w, cdfs, ctx, ctxReq, transform.Class2D, coeffs[:], scan, levels)
	return err
}

func encodeLosslessTXB16(w *entropy.Writer, cdfs *tile.CoeffCDFs, ctx *tile.CoeffEntropyContext,
	ctxReq tile.CoeffContextRequest, plane []uint16, stride, px, py int, bitDepth uint8,
	scan []int16, levels []uint8) error {

	dc := dcPredictN16(plane, stride, px, py, 4, py > 0, px > 0, bitDepth)

	var residual [16]int16
	for r := range 4 {
		row := (py+r)*stride + px
		for c := range 4 {
			residual[r*4+c] = int16(plane[row+c]) - int16(dc)
		}
	}
	var wht [16]int32
	if err := transform.ForwardWHT4x4(wht[:], 4, residual[:], 4); err != nil {
		return err
	}
	var coeffs [16]int16
	for i, v := range wht {
		coeffs[i] = int16(v >> 2)
	}
	_, err := tile.WriteCoefficientsTXBWithContext(w, cdfs, ctx, ctxReq, transform.Class2D, coeffs[:], scan, levels)
	return err
}

func dcPredictN16(plane []uint16, stride, px, py, n int, haveTop, haveLeft bool, bitDepth uint8) uint16 {
	return dcPredictRect16(plane, stride, px, py, n, n, haveTop, haveLeft, bitDepth)
}

func dcPredictRect16(plane []uint16, stride, px, py, w, h int, haveTop, haveLeft bool, bitDepth uint8) uint16 {
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
		return uint16(1 << (bitDepth - 1))
	}
	return uint16((sum + count/2) / count)
}
