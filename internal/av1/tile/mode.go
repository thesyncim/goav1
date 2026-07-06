package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	BlockModeContexts   = 3
	MaxBlockModeSlots   = 32
	cdefUnitsPerSB      = 4
	segmentIDCDFSymbols = parser.MaxSegments
	maxCDEFBits         = 3

	// intrabcCrossSBHistory is the depth (in 4x4 mi units) of cross-superblock
	// motion history retained for libaom's outer mv-ref scan in IBC. The outer
	// scan reads rows -3 and -5 (and column equivalents) into the previous SB's
	// interior; libaom's frame-level mi grid covers those cells directly. Five
	// rows are sufficient because MVREF_ROW_COLS=3 caps the outer scan at
	// row/col offset -5 from the current block, while in-SB scans handle the
	// remaining cells via the live grid.
	intrabcCrossSBHistory = 5
)

// BlockModeCDFs contains the caller-owned CDFs used by the block syntax prefix:
// segmentation prediction/id, skip_mode, and skip_transform.
type BlockModeCDFs struct {
	Skip        [BlockModeContexts]entropy.CDF
	SkipMode    [BlockModeContexts]entropy.CDF
	SegmentPred [BlockModeContexts]entropy.CDF
	SegmentID   [BlockModeContexts]entropy.CDF
}

// BlockModeContext is the caller-owned top/left block context for one
// superblock. Slots are addressed in 4x4 units, matching dav1d's bx4/by4.
type BlockModeContext struct {
	AboveSkip        [MaxBlockModeSlots]uint8
	LeftSkip         [MaxBlockModeSlots]uint8
	AboveSkipMode    [MaxBlockModeSlots]uint8
	LeftSkipMode     [MaxBlockModeSlots]uint8
	AboveSegmentPred [MaxBlockModeSlots]uint8
	LeftSegmentPred  [MaxBlockModeSlots]uint8
	AboveIntra       [MaxBlockModeSlots]uint8
	LeftIntra        [MaxBlockModeSlots]uint8
	AboveMode        [MaxBlockModeSlots]IntraMode
	LeftMode         [MaxBlockModeSlots]IntraMode
	AboveChromaIntra [MaxBlockModeSlots]uint8
	LeftChromaIntra  [MaxBlockModeSlots]uint8
	AboveChromaMode  [MaxBlockModeSlots]ChromaIntraMode
	LeftChromaMode   [MaxBlockModeSlots]ChromaIntraMode
	AboveTxIntra     [MaxBlockModeSlots]uint8
	LeftTxIntra      [MaxBlockModeSlots]uint8
	AboveTx          [MaxBlockModeSlots]uint8
	LeftTx           [MaxBlockModeSlots]uint8
	AboveRef         [2][MaxBlockModeSlots]ReferenceFrame
	LeftRef          [2][MaxBlockModeSlots]ReferenceFrame
	AboveCompound    [MaxBlockModeSlots]uint8
	LeftCompound     [MaxBlockModeSlots]uint8
	AboveCompGroup   [MaxBlockModeSlots]uint8
	LeftCompGroup    [MaxBlockModeSlots]uint8
	AboveCompIndex   [MaxBlockModeSlots]uint8
	LeftCompIndex    [MaxBlockModeSlots]uint8
	// AboveInterIntra/LeftInterIntra mirror libaom's per-MI inter-intra
	// status. libaom records inter-intra blocks with
	// mbmi->ref_frame[1] = INTRA_FRAME while goav1 keeps Ref[1] =
	// ReferenceFrameNone for them. Neighbor scans that need to know
	// "has any second ref" (ref_frame[1] != NONE_FRAME) - distinct from
	// "has an inter second ref" (has_second_ref, i.e. ref_frame[1] >
	// INTRA_FRAME) - consult this flag. Notably av1_findSamples rejects
	// inter-intra neighbors as warp-affine sample sources because their
	// ref_frame[1] is INTRA_FRAME, not NONE_FRAME.
	AboveInterIntra  [MaxBlockModeSlots]uint8
	LeftInterIntra   [MaxBlockModeSlots]uint8
	AboveInterMotion [MaxBlockModeSlots]InterMotionResult
	LeftInterMotion  [MaxBlockModeSlots]InterMotionResult
	AboveMotionValid [MaxBlockModeSlots]uint8
	LeftMotionValid  [MaxBlockModeSlots]uint8

	// SBTopInterMotion mirrors libaom's frame-level mi grid for the row of
	// cells immediately above the current superblock. Unlike AboveInterMotion
	// it is loaded once at superblock start (from the caller's carrier) and
	// never overwritten while the superblock is being decoded. Cross-SB
	// diagonal/outer mv-ref scans consult this snapshot instead of the live
	// AboveInterMotion array, so an in-SB block does not mask the prior SB
	// row's neighbor when its column overlaps the diagonal lookup.
	SBTopInterMotion [MaxBlockModeSlots]InterMotionResult
	SBTopMotionValid [MaxBlockModeSlots]uint8
	SBTopBlockSize   [MaxBlockModeSlots]BlockSize

	// SBTopInterMotionGrid extends SBTopInterMotion with additional rows of the
	// prior superblock above the current one. Depth 0 is the row immediately
	// above the SB (mirrors SBTopInterMotion) and deeper indices correspond to
	// progressively higher mi rows inside the prior SB. This is the carrier
	// for libaom's outer mv-ref row scan (offsets -3 and -5) which would
	// otherwise read into the prior SB's interior. Cells outside the prior
	// SB's actual content keep MotionValid==0.
	SBTopInterMotionGrid      [intrabcCrossSBHistory][MaxBlockModeSlots]InterMotionResult
	SBTopMotionValidGrid      [intrabcCrossSBHistory][MaxBlockModeSlots]uint8
	SBTopBlockSizeGrid        [intrabcCrossSBHistory][MaxBlockModeSlots]BlockSize
	SBTopBlockSizeVisitedGrid [intrabcCrossSBHistory][MaxBlockModeSlots]uint8

	// SBTopRightInterMotion mirrors the bottom row of the SB that lives above
	// and to the right of the current superblock. Its sole consumer is the
	// topright (mi_col+W4, mi_row-1) lookup when X4+W4 == MaxBlockModeSlots
	// (the block touches the current SB's right edge with sbSizeMIB ==
	// MaxBlockModeSlots): the cell lives across the right SB boundary, which
	// AboveInterMotion cannot represent because slot indices saturate at
	// MaxBlockModeSlots. Slot i holds the cell at frame
	// mi(mi_row-1, rightSBStartCol + i) where rightSBStartCol is the mi_col
	// of the SB column immediately right of the current one. libaom's
	// frame-wide mi grid covers this cell directly via xd->mi[-stride +
	// xd->width]; goav1 stages it from carrier.Above[rootColIndex+1] at SB
	// load time.
	SBTopRightInterMotion [MaxBlockModeSlots]InterMotionResult
	SBTopRightMotionValid [MaxBlockModeSlots]uint8
	SBTopRightBlockSize   [MaxBlockModeSlots]BlockSize
	SBTopRightIntra       [MaxBlockModeSlots]uint8
	SBTopRightValid       bool

	// SBLeftInterMotion is the SB-row-equivalent snapshot of the column
	// immediately to the left of the current superblock. LeftInterMotion is
	// overwritten as in-SB blocks decode, but cross-SB scans need the prior
	// SB column's per-row state to find diagonal/outer-column neighbors.
	SBLeftInterMotion [MaxBlockModeSlots]InterMotionResult
	SBLeftMotionValid [MaxBlockModeSlots]uint8
	SBLeftBlockSize   [MaxBlockModeSlots]BlockSize

	// SBLeftInterMotionGrid extends SBLeftInterMotion with additional columns
	// of the prior superblock to the left of the current one. Depth 0 is the
	// column immediately to the left (mirrors SBLeftInterMotion) and deeper
	// indices correspond to progressively earlier mi columns inside the prior
	// SB. Carrier for libaom's outer mv-ref column scan (offsets -3 and -5).
	SBLeftInterMotionGrid      [intrabcCrossSBHistory][MaxBlockModeSlots]InterMotionResult
	SBLeftMotionValidGrid      [intrabcCrossSBHistory][MaxBlockModeSlots]uint8
	SBLeftBlockSizeGrid        [intrabcCrossSBHistory][MaxBlockModeSlots]BlockSize
	SBLeftBlockSizeVisitedGrid [intrabcCrossSBHistory][MaxBlockModeSlots]uint8

	// SBDiagonalInterMotionGrid snapshots the bottom-right corner of the SB
	// diagonally up-left of the current SB. Indexed [rowDepth][colDepth]:
	// depth (d, e) carries libaom's mi grid cell at (mi_col-1-e, mi_row-1-d).
	// Used by the IBC outer block scan at (-1, -1) when both x4<0 and y4<0
	// (the diagonal cell is in the SB above the SB to the left, which the
	// carrier's same-row Left store would otherwise have clobbered).
	SBDiagonalInterMotionGrid      [intrabcCrossSBHistory][intrabcCrossSBHistory]InterMotionResult
	SBDiagonalMotionValidGrid      [intrabcCrossSBHistory][intrabcCrossSBHistory]uint8
	SBDiagonalBlockSizeGrid        [intrabcCrossSBHistory][intrabcCrossSBHistory]BlockSize
	SBDiagonalBlockSizeVisitedGrid [intrabcCrossSBHistory][intrabcCrossSBHistory]uint8

	AboveInterp      [MaxBlockModeSlots]motion.InterpFilters
	LeftInterp       [MaxBlockModeSlots]motion.InterpFilters
	AboveInterpValid [MaxBlockModeSlots]uint8
	LeftInterpValid  [MaxBlockModeSlots]uint8
	AboveBlockSize   [MaxBlockModeSlots]BlockSize
	LeftBlockSize    [MaxBlockModeSlots]BlockSize

	// TxNeighborSnapshot captures the above/left neighbor state at slots
	// touched by the current block, taken before the current block's
	// MarkIntra/MarkIntrabcMotion overwrites those slots. It is consumed by
	// the intra tx_size context which, in libaom, reads xd->above_mbmi /
	// xd->left_mbmi — pointers to neighbor mbmi structs that are not
	// affected by writes to the current block's mbmi. Without this snapshot
	// Go's slot-shared AboveIntra/AboveBlockSize/LeftIntra/LeftBlockSize
	// would be the current block's just-written flags by the time tx_size
	// reads them inside DecodeBlockCoefficients.
	TxNeighborValid          bool
	TxAboveNeighborIntra     uint8
	TxAboveNeighborBlockSize BlockSize
	TxLeftNeighborIntra      uint8
	TxLeftNeighborBlockSize  BlockSize
	AbovePaletteY            [MaxBlockModeSlots]paletteContext
	LeftPaletteY             [MaxBlockModeSlots]paletteContext
	AbovePaletteUV           [MaxBlockModeSlots]paletteContext
	LeftPaletteUV            [MaxBlockModeSlots]paletteContext

	GridInterMotion [MaxBlockModeSlots][MaxBlockModeSlots]InterMotionResult
	GridMotionValid [MaxBlockModeSlots][MaxBlockModeSlots]uint8
	GridBlockSize   [MaxBlockModeSlots][MaxBlockModeSlots]BlockSize

	// GridInterp records each inter block's decoded interpolation filter
	// pair per MI cell, mirroring libaom's per-MB this_mbmi->interp_filters.
	// The sub8x8 chroma predictor (build_inter_predictors_sub8x8) reads each
	// covered luma sub-block's own filters, so it consults this grid for the
	// neighbor cells rather than the collapsed 1D Above/Left filter contexts
	// (which only retain the last block written to a row/column slot and can
	// therefore report a different neighbor's filter). Written by
	// MarkInterFilters alongside GridInterMotion; read under the same
	// GridMotionValid guard.
	GridInterp      [MaxBlockModeSlots][MaxBlockModeSlots]motion.InterpFilters
	GridInterpValid [MaxBlockModeSlots][MaxBlockModeSlots]uint8

	// GridBlockSizeVisited mirrors GridBlockSize but signals whether the
	// containing slot was ever recorded by markGridInterMotion or
	// clearGridInterMotion. Outer ref-MV scans use it to distinguish "slot
	// was a real intra/inter neighbor of known size" (advance by mi_size
	// of GridBlockSize, matching libaom's scan_row/col stride) from "slot
	// is past the decoded region" (skip cell-by-cell). Without this flag a
	// zeroed GridBlockSize (BlockSize128x128) collides with unvisited cells
	// and forces the scan to fall back to a 4x4 stride past intra neighbors,
	// oversampling deeper cells libaom never visits.
	GridBlockSizeVisited [MaxBlockModeSlots][MaxBlockModeSlots]uint8
}

// CDEFIndexContext caches the cdef_idx values already read for the 64x64 CDEF
// units inside the current superblock. A 128x128 superblock has four units; a
// 64x64 superblock only uses slot 0.
type CDEFIndexContext struct {
	Index [cdefUnitsPerSB]uint8
	Read  [cdefUnitsPerSB]bool
}

// BlockModeRequest describes the block currently being decoded for the AV1
// block syntax prefix.
type BlockModeRequest struct {
	Size BlockSize

	SkipMode parser.SkipModeParams
	CDEF     parser.CDEFParams

	SegmentationEnabled bool
	Segment             parser.SegmentData

	X4 uint8
	Y4 uint8
}

// BlockModeResult is the entropy-decoded block syntax prefix.
type BlockModeResult struct {
	SegmentPredicted bool
	SkipMode         bool
	SkipTransform    bool
	CDEFIndex        uint8
}

var defaultBlockModeCDFs = makeDefaultBlockModeCDFs()

func makeDefaultBlockModeCDFs() BlockModeCDFs {
	var c BlockModeCDFs
	if err := c.initDefault(); err != nil {
		panic(err)
	}
	return c
}

// InitDefault seeds c with dav1d/libaom's default block-prefix CDFs.
func (c *BlockModeCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	*c = defaultBlockModeCDFs
	return nil
}

func (c *BlockModeCDFs) initDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next BlockModeCDFs
	for i, cumulative := range [...]uint16{31671, 16515, 4576} {
		if err := next.Skip[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	for i, cumulative := range [...]uint16{32621, 20708, 8127} {
		if err := next.SkipMode[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	for i := range BlockModeContexts {
		if err := next.SegmentPred[i].Init([]uint16{16384}); err != nil {
			return err
		}
	}
	defaultSegmentID := [...][parser.MaxSegments - 1]uint16{
		{5622, 7893, 16093, 18233, 27809, 28373, 32533},
		{14274, 18230, 22557, 24935, 29980, 30851, 32344},
		{27527, 28487, 28723, 28890, 32397, 32647, 32679},
	}
	for i := range defaultSegmentID {
		if err := next.SegmentID[i].Init(defaultSegmentID[i][:]); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

// SkipCDF returns the initialized skip_transform CDF for ctx.
func (c *BlockModeCDFs) SkipCDF(ctx int) (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return blockModeCDF(&c.Skip, ctx, 2)
}

// SkipModeCDF returns the initialized skip_mode CDF for ctx.
func (c *BlockModeCDFs) SkipModeCDF(ctx int) (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return blockModeCDF(&c.SkipMode, ctx, 2)
}

// SegmentPredCDF returns the initialized temporal segment prediction CDF for ctx.
func (c *BlockModeCDFs) SegmentPredCDF(ctx int) (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return blockModeCDF(&c.SegmentPred, ctx, 2)
}

// SegmentIDCDF returns the initialized spatial segment-id CDF for ctx.
func (c *BlockModeCDFs) SegmentIDCDF(ctx int) (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return blockModeCDF(&c.SegmentID, ctx, segmentIDCDFSymbols)
}

// Reset marks all CDEF-unit indices in c as unread for a new superblock.
func (c *CDEFIndexContext) Reset() {
	if c == nil {
		return
	}
	*c = CDEFIndexContext{}
}

// SkipContext returns libaom's av1_get_skip_txfm_context() value.
func (c *BlockModeContext) SkipContext(x4 int, y4 int) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	return int(c.AboveSkip[x4]) + int(c.LeftSkip[y4]), nil
}

// SkipModeContext returns libaom's av1_get_skip_mode_context() value.
func (c *BlockModeContext) SkipModeContext(x4 int, y4 int) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	return int(c.AboveSkipMode[x4]) + int(c.LeftSkipMode[y4]), nil
}

// SegmentPredContext returns libaom's av1_get_pred_context_seg_id() value.
func (c *BlockModeContext) SegmentPredContext(x4 int, y4 int) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	return int(c.AboveSegmentPred[x4]) + int(c.LeftSegmentPred[y4]), nil
}

// Mark updates the top/left block prefix context after a decoded block.
func (c *BlockModeContext) Mark(size BlockSize, x4 int, y4 int, result BlockModeResult) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	if x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}

	segPred := boolByte(result.SegmentPredicted)
	skipMode := boolByte(result.SkipMode)
	skip := boolByte(result.SkipTransform)
	for i := 0; i < int(dims.W4); i++ {
		c.AboveSegmentPred[x4+i] = segPred
		c.AboveSkipMode[x4+i] = skipMode
		c.AboveSkip[x4+i] = skip
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftSegmentPred[y4+i] = segPred
		c.LeftSkipMode[y4+i] = skipMode
		c.LeftSkip[y4+i] = skip
	}
	return nil
}

// ReadSegmentPrediction decodes the temporal segmentation prediction flag.
func (s *DecodeState) ReadSegmentPrediction(cdfs *BlockModeCDFs, ctx *BlockModeContext, x4 int, y4 int) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	context, err := ctx.SegmentPredContext(x4, y4)
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.SegmentPredCDF(context)
	if err != nil {
		return false, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return false, err
	}
	return symbol != 0, nil
}

// ReadSegmentID decodes one spatial segment id using AV1's negative
// deinterleave around the current-frame spatial predictor.
func (s *DecodeState) ReadSegmentID(cdfs *BlockModeCDFs, pred uint8, context int, lastActiveID int8, skip bool) (uint8, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	max := int(lastActiveID) + 1
	if max <= 0 || max > parser.MaxSegments || int(pred) >= max {
		return 0, ErrInvalidDecodeState
	}
	if skip {
		return pred, nil
	}
	cdf, err := cdfs.SegmentIDCDF(context)
	if err != nil {
		return 0, err
	}
	diff, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	segmentID, err := negDeinterleaveSegmentID(diff, int(pred), max)
	if err != nil {
		return 0, err
	}
	if segmentID < 0 || segmentID >= max {
		return 0, ErrInvalidDecodeState
	}
	return uint8(segmentID), nil
}

// ReadSkipMode decodes skip_mode for one block, including the segment and
// block-size conditions that make the syntax implicitly zero.
func (s *DecodeState) ReadSkipMode(cdfs *BlockModeCDFs, ctx *BlockModeContext, req BlockModeRequest) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	read, err := shouldReadSkipMode(req)
	if err != nil || !read {
		return false, err
	}
	context, err := ctx.SkipModeContext(int(req.X4), int(req.Y4))
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.SkipModeCDF(context)
	if err != nil {
		return false, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return false, err
	}
	return symbol != 0, nil
}

// ReadSkipTransform decodes skip_transform for one block, including the
// skip_mode and segment-skip conditions that force it to one.
func (s *DecodeState) ReadSkipTransform(cdfs *BlockModeCDFs, ctx *BlockModeContext, req BlockModeRequest, skipMode bool) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	if skipMode || segmentForcesSkip(req) {
		return true, nil
	}
	context, err := ctx.SkipContext(int(req.X4), int(req.Y4))
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.SkipCDF(context)
	if err != nil {
		return false, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return false, err
	}
	return symbol != 0, nil
}

// ReadCDEFIndex decodes the cdef_idx syntax for one block. AV1 omits this
// field for skipped blocks and when the frame has a single CDEF strength.
func (s *DecodeState) ReadCDEFIndex(params parser.CDEFParams, skipTransform bool) (uint8, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	if params.Bits > maxCDEFBits {
		return 0, ErrInvalidDecodeState
	}
	if skipTransform || params.Bits == 0 {
		return 0, nil
	}
	index, err := s.Reader.ReadBits(params.Bits)
	if err != nil {
		return 0, err
	}
	if index >= parser.MaxCDEFStrengths {
		return 0, ErrInvalidDecodeState
	}
	return uint8(index), nil
}

// ReadCDEFIndexForBlock decodes cdef_idx using libaom/dav1d's 64x64 CDEF-unit
// cache: the first non-skip block in a unit reads the syntax element, and later
// blocks in the same unit reuse it without consuming entropy bits.
func (s *DecodeState) ReadCDEFIndexForBlock(params parser.CDEFParams, ctx *CDEFIndexContext, size BlockSize, x4 int, y4 int, skipTransform bool) (uint8, error) {
	if s == nil || ctx == nil {
		return 0, ErrInvalidDecodeState
	}
	if params.Bits > maxCDEFBits {
		return 0, ErrInvalidDecodeState
	}
	if skipTransform || params.Bits == 0 {
		return 0, nil
	}

	units, count, err := cdefUnitsForBlock(size, x4, y4)
	if err != nil {
		return 0, err
	}
	if ctx.Read[units[0]] {
		cached := ctx.Index[units[0]]
		if cached >= parser.MaxCDEFStrengths {
			return 0, ErrInvalidDecodeState
		}
		return cached, nil
	}

	index, err := s.ReadCDEFIndex(params, false)
	if err != nil {
		return 0, err
	}
	for i := range count {
		ctx.Index[units[i]] = index
		ctx.Read[units[i]] = true
	}
	return index, nil
}

// ReadBlockModePrefix decodes skip_mode, skip_transform, and cdef_idx for one
// block and updates the caller-owned top/left contexts.
func (s *DecodeState) ReadBlockModePrefix(cdfs *BlockModeCDFs, ctx *BlockModeContext, req BlockModeRequest) (BlockModeResult, error) {
	if ctx == nil {
		return BlockModeResult{}, ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok {
		return BlockModeResult{}, ErrInvalidDecodeState
	}
	skipMode, err := s.ReadSkipMode(cdfs, ctx, req)
	if err != nil {
		return BlockModeResult{}, err
	}
	skip, err := s.ReadSkipTransform(cdfs, ctx, req, skipMode)
	if err != nil {
		return BlockModeResult{}, err
	}
	cdefIndex, err := s.ReadCDEFIndex(req.CDEF, skip)
	if err != nil {
		return BlockModeResult{}, err
	}
	result := BlockModeResult{
		SkipMode:      skipMode,
		SkipTransform: skip,
		CDEFIndex:     cdefIndex,
	}
	if err := ctx.Mark(req.Size, int(req.X4), int(req.Y4), result); err != nil {
		return BlockModeResult{}, err
	}
	return result, nil
}

// PredictCurrentSegmentID ports dav1d's get_cur_frame_segid() helper.
func PredictCurrentSegmentID(cur []uint8, stride uint16, x4 int, y4 int, haveTop bool, haveLeft bool) (uint8, int, error) {
	strideInt := int(stride)
	if stride == 0 || x4 < 0 || y4 < 0 || x4 >= strideInt {
		return 0, 0, ErrInvalidDecodeState
	}
	idx := y4*strideInt + x4
	if idx < 0 || idx >= len(cur) {
		return 0, 0, ErrInvalidDecodeState
	}
	if haveLeft && x4 == 0 || haveTop && y4 == 0 {
		return 0, 0, ErrInvalidDecodeState
	}
	if haveLeft && haveTop {
		if idx-strideInt-1 < 0 || idx-1 >= len(cur) || idx-strideInt >= len(cur) {
			return 0, 0, ErrInvalidDecodeState
		}
		left := cur[idx-1]
		above := cur[idx-strideInt]
		aboveLeft := cur[idx-strideInt-1]
		if left >= parser.MaxSegments || above >= parser.MaxSegments || aboveLeft >= parser.MaxSegments {
			return 0, 0, ErrInvalidDecodeState
		}

		context := 0
		switch {
		case left == above && aboveLeft == left:
			context = 2
		case left == above || aboveLeft == left || above == aboveLeft:
			context = 1
		}
		if above == aboveLeft {
			return above, context, nil
		}
		return left, context, nil
	}

	if haveLeft {
		left := cur[idx-1]
		if left >= parser.MaxSegments {
			return 0, 0, ErrInvalidDecodeState
		}
		return left, 0, nil
	}
	if haveTop {
		above := cur[idx-strideInt]
		if above >= parser.MaxSegments {
			return 0, 0, ErrInvalidDecodeState
		}
		return above, 0, nil
	}
	return 0, 0, nil
}

// MinPreviousSegmentID ports dav1d/libaom's previous-frame segment-map block
// prediction: the minimum segment id over the block footprint.
func MinPreviousSegmentID(prev []uint8, stride uint16, x4 int, y4 int, w4 int, h4 int) (uint8, error) {
	strideInt := int(stride)
	if stride == 0 || x4 < 0 || y4 < 0 || w4 <= 0 || h4 <= 0 || x4+w4 > strideInt {
		return 0, ErrInvalidDecodeState
	}
	best := uint8(parser.MaxSegments)
	for y := range h4 {
		row := (y4+y)*strideInt + x4
		if row < 0 || row+w4 > len(prev) {
			return 0, ErrInvalidDecodeState
		}
		for x := range w4 {
			id := prev[row+x]
			if id >= parser.MaxSegments {
				return 0, ErrInvalidDecodeState
			}
			if id < best {
				best = id
			}
		}
		if best == 0 {
			return 0, nil
		}
	}
	if best >= parser.MaxSegments {
		return 0, ErrInvalidDecodeState
	}
	return best, nil
}

// FillSegmentID writes a decoded segment id into a current-frame segment map.
func FillSegmentID(dst []uint8, stride uint16, x4 int, y4 int, w4 int, h4 int, segmentID uint8) error {
	strideInt := int(stride)
	if segmentID >= parser.MaxSegments || stride == 0 || x4 < 0 || y4 < 0 ||
		w4 <= 0 || h4 <= 0 || x4+w4 > strideInt {
		return ErrInvalidDecodeState
	}
	for y := range h4 {
		row := (y4+y)*strideInt + x4
		if row < 0 || row+w4 > len(dst) {
			return ErrInvalidDecodeState
		}
		for x := range w4 {
			dst[row+x] = segmentID
		}
	}
	return nil
}

func blockModeCDF(table *[BlockModeContexts]entropy.CDF, ctx int, symbols int) (*entropy.CDF, error) {
	if table == nil || ctx < 0 || ctx >= BlockModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &table[ctx]
	if cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
}

func shouldReadSkipMode(req BlockModeRequest) (bool, error) {
	dims, ok := req.Size.Dimensions()
	if !ok {
		return false, ErrInvalidDecodeState
	}
	if !req.SkipMode.Enabled || minUint8(dims.W4, dims.H4) <= 1 {
		return false, nil
	}
	return !segmentPrecludesSkipMode(req), nil
}

func segmentPrecludesSkipMode(req BlockModeRequest) bool {
	return req.SegmentationEnabled &&
		(req.Segment.Skip || req.Segment.GlobalMV || req.Segment.RefFrame >= 0)
}

func segmentForcesSkip(req BlockModeRequest) bool {
	return req.SegmentationEnabled && req.Segment.Skip
}

func validateBlockModeSlot(x4 int, y4 int) error {
	if x4 < 0 || y4 < 0 || x4 >= MaxBlockModeSlots || y4 >= MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	return nil
}

func cdefUnitsForBlock(size BlockSize, x4 int, y4 int) ([cdefUnitsPerSB]uint8, int, error) {
	dims, ok := size.Dimensions()
	if !ok || x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return [cdefUnitsPerSB]uint8{}, 0, ErrInvalidDecodeState
	}
	xUnit0 := x4 >> 4
	yUnit0 := y4 >> 4
	xUnit1 := (x4 + int(dims.W4) - 1) >> 4
	yUnit1 := (y4 + int(dims.H4) - 1) >> 4
	if xUnit0 < 0 || yUnit0 < 0 || xUnit1 >= 2 || yUnit1 >= 2 {
		return [cdefUnitsPerSB]uint8{}, 0, ErrInvalidDecodeState
	}

	var units [cdefUnitsPerSB]uint8
	count := 0
	for uy := yUnit0; uy <= yUnit1; uy++ {
		for ux := xUnit0; ux <= xUnit1; ux++ {
			units[count] = uint8(uy*2 + ux)
			count++
		}
	}
	return units, count, nil
}

func negDeinterleaveSegmentID(diff int, ref int, max int) (int, error) {
	if diff < 0 || ref < 0 || max <= 0 || ref >= max {
		return 0, ErrInvalidDecodeState
	}
	if ref == 0 {
		return diff, nil
	}
	if ref >= max-1 {
		return max - diff - 1, nil
	}
	if 2*ref < max {
		if diff <= 2*ref {
			if diff&1 != 0 {
				return ref + ((diff + 1) >> 1), nil
			}
			return ref - (diff >> 1), nil
		}
		return diff, nil
	}
	if diff <= 2*(max-ref-1) {
		if diff&1 != 0 {
			return ref + ((diff + 1) >> 1), nil
		}
		return ref - (diff >> 1), nil
	}
	return max - (diff + 1), nil
}

func boolByte(v bool) uint8 {
	if v {
		return 1
	}
	return 0
}

func minUint8(a uint8, b uint8) uint8 {
	if a < b {
		return a
	}
	return b
}
