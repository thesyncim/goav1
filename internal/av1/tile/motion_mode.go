package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// MotionMode identifies AV1's motion variation mode in libaom/dav1d order.
type MotionMode uint8

const (
	MotionModeTranslation MotionMode = iota
	MotionModeOBMC
	MotionModeWarp
	motionModeCount
)

// MotionModeCDFs contains caller-owned CDFs for motion mode syntax.
type MotionModeCDFs struct {
	MotionMode [blockSizeCount]entropy.CDF
	OBMC       [blockSizeCount]entropy.CDF
}

// MotionModeRequest describes the already-derived block/frame conditions used
// by libaom's motion_mode_allowed() gate.
type MotionModeRequest struct {
	Size BlockSize
	Mode InterMode

	Compound   bool
	SkipMode   bool
	InterIntra bool

	SwitchableMotionMode bool
	AllowWarpedMotion    bool
	ForceIntegerMV       bool
	GlobalMotionType     parser.GlobalMotionType
	ScaledReference      bool

	OverlappableNeighbors int
	NumProjRef            int
}

// OverlappableNeighborRequest describes the current block for libaom's
// av1_count_overlappable_neighbors() scan over the already decoded top/left
// mode context.
type OverlappableNeighborRequest struct {
	Size BlockSize

	X4 int
	Y4 int

	VisibleW4 uint8
	VisibleH4 uint8

	HaveTop      bool
	HaveLeft     bool
	HaveTopRight bool
}

const MaxOverlappableNeighbors = 4

// OverlappableNeighbor snapshots one already-decoded neighbor used by OBMC.
type OverlappableNeighbor struct {
	RelX4 int
	RelY4 int
	Span4 uint8

	Size BlockSize

	Motion             InterMotionResult
	MotionValid        bool
	InterpFilters      motion.InterpFilters
	InterpFiltersValid bool
}

// OverlappableNeighborSet carries the above and left OBMC neighbor lists.
type OverlappableNeighborSet struct {
	Above         [MaxOverlappableNeighbors]OverlappableNeighbor
	AboveCount    int
	Left          [MaxOverlappableNeighbors]OverlappableNeighbor
	LeftCount     int
	TopLeft       OverlappableNeighbor
	TopLeftValid  bool
	TopRight      OverlappableNeighbor
	TopRightValid bool
}

func (s OverlappableNeighborSet) MotionModeCount() int {
	if s.AboveCount != 0 {
		return s.AboveCount
	}
	return s.LeftCount
}

// WarpSampleCount returns the count of neighboring single-reference samples
// that can seed AV1's warped-motion projection for a block using ref.
func (s OverlappableNeighborSet) WarpSampleCount(ref ReferenceFrame) int {
	if !ref.Valid() {
		return 0
	}
	count := 0
	for i := 0; i < s.AboveCount && count < maxWarpSamples; i++ {
		if warpSampleNeighborMatches(s.Above[i], ref) {
			count++
		}
	}
	for i := 0; i < s.LeftCount && count < maxWarpSamples; i++ {
		if warpSampleNeighborMatches(s.Left[i], ref) {
			count++
		}
	}
	if s.TopLeftValid && count < maxWarpSamples && warpSampleNeighborMatches(s.TopLeft, ref) {
		count++
	}
	if s.TopRightValid && count < maxWarpSamples && warpSampleNeighborMatches(s.TopRight, ref) {
		count++
	}
	return count
}

func (s OverlappableNeighborSet) WarpProjectionInvalid(block BlockVisit, ref ReferenceFrame, mv motion.Vector) (bool, error) {
	_, invalid, err := s.WarpProjection(block, ref, mv)
	return invalid, err
}

var motionModeDefaultCDF = [blockSizeCount][2]uint16{
	BlockSize128x128: {32507, 32558},
	BlockSize128x64:  {30878, 31335},
	BlockSize64x128:  {28898, 30397},
	BlockSize64x64:   {29516, 30701},
	BlockSize64x32:   {21679, 26830},
	BlockSize64x16:   {29742, 31203},
	BlockSize32x64:   {20360, 28062},
	BlockSize32x32:   {26260, 29116},
	BlockSize32x16:   {11606, 24308},
	BlockSize32x8:    {28799, 31390},
	BlockSize16x64:   {28973, 31594},
	BlockSize16x32:   {5123, 23606},
	BlockSize16x16:   {19419, 26810},
	BlockSize16x8:    {5391, 25528},
	BlockSize8x32:    {26431, 30774},
	BlockSize8x16:    {4738, 24765},
	BlockSize8x8:     {7651, 24760},
}

var obmcDefaultCDF = [blockSizeCount]uint16{
	BlockSize128x128: 32638,
	BlockSize128x64:  31560,
	BlockSize64x128:  31014,
	BlockSize64x64:   30128,
	BlockSize64x32:   22083,
	BlockSize64x16:   26879,
	BlockSize32x64:   22823,
	BlockSize32x32:   25817,
	BlockSize32x16:   15142,
	BlockSize32x8:    23664,
	BlockSize16x64:   24008,
	BlockSize16x32:   14423,
	BlockSize16x16:   17432,
	BlockSize16x8:    9301,
	BlockSize8x32:    20901,
	BlockSize8x16:    9371,
	BlockSize8x8:     10437,
}

func (mode MotionMode) Valid() bool {
	return mode < motionModeCount
}

// InitDefault seeds c with dav1d/libaom's default motion-mode CDFs.
func (c *MotionModeCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next MotionModeCDFs
	for size := BlockSize(0); size < blockSizeCount; size++ {
		if !motionModeBlockSizeAllowed(size) {
			continue
		}
		if err := next.MotionMode[size].Init(motionModeDefaultCDF[size][:]); err != nil {
			return err
		}
		if err := next.OBMC[size].Init([]uint16{obmcDefaultCDF[size]}); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

func (c *MotionModeCDFs) MotionModeCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || !motionModeBlockSizeAllowed(size) {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.MotionMode[size]
	if cdf.Symbols() != int(motionModeCount) {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (c *MotionModeCDFs) OBMCCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || !motionModeBlockSizeAllowed(size) {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.OBMC[size])
}

// CountOverlappableNeighbors ports libaom's
// av1_count_overlappable_neighbors() decision for the locally tracked
// immediate above/left block contexts.
func (c *BlockModeContext) CountOverlappableNeighbors(req OverlappableNeighborRequest) (int, error) {
	visibleW, visibleH, err := c.validateOverlappableNeighborRequest(req)
	if err != nil {
		return 0, err
	}
	if !motionModeBlockSizeAllowed(req.Size) {
		return 0, nil
	}

	if req.HaveTop {
		if count := c.countOverlappableAbove(req.X4, visibleW); count != 0 {
			return count, nil
		}
	}
	if req.HaveLeft {
		return c.countOverlappableLeft(req.Y4, visibleH), nil
	}
	return 0, nil
}

// CollectOverlappableNeighbors ports the foreach_overlappable_nb_above/left
// scans used by libaom's OBMC predictor builder.
func (c *BlockModeContext) CollectOverlappableNeighbors(req OverlappableNeighborRequest) (OverlappableNeighborSet, error) {
	visibleW, visibleH, err := c.validateOverlappableNeighborRequest(req)
	if err != nil {
		return OverlappableNeighborSet{}, err
	}
	if !motionModeBlockSizeAllowed(req.Size) {
		return OverlappableNeighborSet{}, nil
	}
	aboveMax, leftMax, err := MaxOBMCNeighborsForSize(req.Size)
	if err != nil {
		return OverlappableNeighborSet{}, err
	}
	var set OverlappableNeighborSet
	if req.HaveTop && aboveMax > 0 {
		c.collectOverlappableAbove(req.X4, visibleW, aboveMax, &set)
	}
	if req.HaveLeft && leftMax > 0 {
		c.collectOverlappableLeft(req.Y4, visibleH, leftMax, &set)
	}
	if req.HaveTop && req.HaveLeft {
		if neighbor, ok := c.gridOverlappableNeighbor(req.X4-1, req.Y4-1, 0, 0); ok {
			set.TopLeft = neighbor
			set.TopLeftValid = true
		}
	}
	if req.HaveTopRight {
		if neighbor, ok := c.gridOverlappableNeighbor(req.X4+visibleW, req.Y4-1, visibleW, 0); ok {
			set.TopRight = neighbor
			set.TopRightValid = true
		}
	}
	return set, nil
}

func MaxOBMCNeighborsForSize(size BlockSize) (above int, left int, err error) {
	dims, ok := size.Dimensions()
	if !ok {
		return 0, 0, ErrInvalidDecodeState
	}
	return maxOBMCNeighborCount(dims.Log2W), maxOBMCNeighborCount(dims.Log2H), nil
}

var maxOBMCNeighbors = [6]int{0, 1, 2, 3, 4, 4}

func maxOBMCNeighborCount(log2 uint8) int {
	if int(log2) >= len(maxOBMCNeighbors) {
		return MaxOverlappableNeighbors
	}
	return maxOBMCNeighbors[log2]
}

func (c *BlockModeContext) validateOverlappableNeighborRequest(req OverlappableNeighborRequest) (int, int, error) {
	if c == nil {
		return 0, 0, ErrInvalidDecodeState
	}
	dims, ok := req.Size.Dimensions()
	if !ok {
		return 0, 0, ErrInvalidDecodeState
	}
	visibleW := int(req.VisibleW4)
	if visibleW == 0 {
		visibleW = int(dims.W4)
	}
	visibleH := int(req.VisibleH4)
	if visibleH == 0 {
		visibleH = int(dims.H4)
	}
	if req.X4 < 0 || req.Y4 < 0 ||
		visibleW <= 0 || visibleH <= 0 ||
		visibleW > int(dims.W4) || visibleH > int(dims.H4) ||
		req.X4+visibleW > MaxBlockModeSlots ||
		req.Y4+visibleH > MaxBlockModeSlots {
		return 0, 0, ErrInvalidDecodeState
	}
	return visibleW, visibleH, nil
}

func (c *BlockModeContext) countOverlappableAbove(x4 int, visibleW4 int) int {
	count := 0
	end := x4 + visibleW4
	for col := x4; col < end; {
		step := c.aboveOverlappableStep(col)
		if step == 1 {
			pairStart := col &^ 1
			if c.aboveOverlappable(pairStart + 1) {
				count++
			}
			col = pairStart + 2
			continue
		}
		if c.aboveOverlappable(col) {
			count++
		}
		col += step
	}
	return count
}

func (c *BlockModeContext) countOverlappableLeft(y4 int, visibleH4 int) int {
	count := 0
	end := y4 + visibleH4
	for row := y4; row < end; {
		step := c.leftOverlappableStep(row)
		if step == 1 {
			pairStart := row &^ 1
			if c.leftOverlappable(pairStart + 1) {
				count++
			}
			row = pairStart + 2
			continue
		}
		if c.leftOverlappable(row) {
			count++
		}
		row += step
	}
	return count
}

func (c *BlockModeContext) collectOverlappableAbove(x4 int, visibleW4 int, maxNeighbors int, set *OverlappableNeighborSet) {
	end := x4 + visibleW4
	for col := x4; col < end && set.AboveCount < maxNeighbors; {
		step := c.aboveOverlappableStep(col)
		rel := col - x4
		span := minInt(step, end-col)
		slot := col
		if step == 1 {
			pairStart := col &^ 1
			rel = pairStart - x4
			span = minInt(2, end-pairStart)
			slot = pairStart + 1
			col = pairStart + 2
		} else {
			col += step
		}
		if c.aboveOverlappable(slot) {
			set.Above[set.AboveCount] = c.aboveOverlappableNeighbor(slot, rel, span)
			set.AboveCount++
		}
	}
}

func (c *BlockModeContext) collectOverlappableLeft(y4 int, visibleH4 int, maxNeighbors int, set *OverlappableNeighborSet) {
	end := y4 + visibleH4
	for row := y4; row < end && set.LeftCount < maxNeighbors; {
		step := c.leftOverlappableStep(row)
		rel := row - y4
		span := minInt(step, end-row)
		slot := row
		if step == 1 {
			pairStart := row &^ 1
			rel = pairStart - y4
			span = minInt(2, end-pairStart)
			slot = pairStart + 1
			row = pairStart + 2
		} else {
			row += step
		}
		if c.leftOverlappable(slot) {
			set.Left[set.LeftCount] = c.leftOverlappableNeighbor(slot, rel, span)
			set.LeftCount++
		}
	}
}

func (c *BlockModeContext) aboveOverlappableNeighbor(slot int, rel int, span int) OverlappableNeighbor {
	return OverlappableNeighbor{
		RelX4:              rel,
		Span4:              uint8(maxInt(0, span)),
		Size:               c.AboveBlockSize[slot],
		Motion:             c.AboveInterMotion[slot],
		MotionValid:        c.AboveMotionValid[slot] != 0,
		InterpFilters:      c.AboveInterp[slot],
		InterpFiltersValid: c.AboveInterpValid[slot] != 0,
	}
}

func (c *BlockModeContext) leftOverlappableNeighbor(slot int, rel int, span int) OverlappableNeighbor {
	return OverlappableNeighbor{
		RelY4:              rel,
		Span4:              uint8(maxInt(0, span)),
		Size:               c.LeftBlockSize[slot],
		Motion:             c.LeftInterMotion[slot],
		MotionValid:        c.LeftMotionValid[slot] != 0,
		InterpFilters:      c.LeftInterp[slot],
		InterpFiltersValid: c.LeftInterpValid[slot] != 0,
	}
}

func (c *BlockModeContext) gridOverlappableNeighbor(x4 int, y4 int, relX4 int, relY4 int) (OverlappableNeighbor, bool) {
	motionResult, size, ok := c.gridInterMotion(x4, y4)
	if !ok {
		return OverlappableNeighbor{}, false
	}
	return OverlappableNeighbor{
		RelX4:       relX4,
		RelY4:       relY4,
		Span4:       1,
		Size:        size,
		Motion:      motionResult,
		MotionValid: true,
	}, true
}

const maxWarpSamples = 8
const warpLeastSquaresMVMax = 256

func warpSampleNeighborMatches(n OverlappableNeighbor, ref ReferenceFrame) bool {
	return n.MotionValid &&
		!n.Motion.InterIntra &&
		!n.Motion.References.Compound &&
		n.Motion.References.Ref[0] == ref &&
		n.Motion.References.Ref[1] == ReferenceFrameNone
}

type warpSample struct {
	X      int
	Y      int
	RefX   int
	RefY   int
	Motion motion.Vector
}

func (s OverlappableNeighborSet) warpSamples(block BlockVisit, ref ReferenceFrame, samples *[maxWarpSamples]warpSample) (int, error) {
	if !ref.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if samples == nil {
		return 0, ErrInvalidDecodeState
	}
	count := 0
	if block.HaveTop {
		for i := 0; i < s.AboveCount && count < maxWarpSamples; i++ {
			sample, ok, err := warpSampleAbove(s.Above[i], ref)
			if err != nil {
				return 0, err
			}
			if ok {
				samples[count] = sample
				count++
			}
		}
	}
	if block.HaveLeft {
		for i := 0; i < s.LeftCount && count < maxWarpSamples; i++ {
			sample, ok, err := warpSampleLeft(s.Left[i], ref)
			if err != nil {
				return 0, err
			}
			if ok {
				samples[count] = sample
				count++
			}
		}
	}
	if s.TopLeftValid && count < maxWarpSamples {
		sample, ok, err := warpSampleTopLeft(s.TopLeft, ref)
		if err != nil {
			return 0, err
		}
		if ok {
			samples[count] = sample
			count++
		}
	}
	if s.TopRightValid && count < maxWarpSamples {
		sample, ok, err := warpSampleTopRight(s.TopRight, ref)
		if err != nil {
			return 0, err
		}
		if ok {
			samples[count] = sample
			count++
		}
	}
	return count, nil
}

func warpSampleAbove(n OverlappableNeighbor, ref ReferenceFrame) (warpSample, bool, error) {
	if !warpSampleNeighborMatches(n, ref) {
		return warpSample{}, false, nil
	}
	dims, ok := n.Size.Dimensions()
	if !ok {
		return warpSample{}, false, ErrInvalidDecodeState
	}
	return recordWarpSample(n.Motion.MV[0], int(n.RelX4), 0, 1, -1, int(dims.W4), int(dims.H4)), true, nil
}

func warpSampleLeft(n OverlappableNeighbor, ref ReferenceFrame) (warpSample, bool, error) {
	if !warpSampleNeighborMatches(n, ref) {
		return warpSample{}, false, nil
	}
	dims, ok := n.Size.Dimensions()
	if !ok {
		return warpSample{}, false, ErrInvalidDecodeState
	}
	return recordWarpSample(n.Motion.MV[0], 0, int(n.RelY4), -1, 1, int(dims.W4), int(dims.H4)), true, nil
}

func warpSampleTopLeft(n OverlappableNeighbor, ref ReferenceFrame) (warpSample, bool, error) {
	if !warpSampleNeighborMatches(n, ref) {
		return warpSample{}, false, nil
	}
	dims, ok := n.Size.Dimensions()
	if !ok {
		return warpSample{}, false, ErrInvalidDecodeState
	}
	return recordWarpSample(n.Motion.MV[0], 0, 0, -1, -1, int(dims.W4), int(dims.H4)), true, nil
}

func warpSampleTopRight(n OverlappableNeighbor, ref ReferenceFrame) (warpSample, bool, error) {
	if !warpSampleNeighborMatches(n, ref) {
		return warpSample{}, false, nil
	}
	dims, ok := n.Size.Dimensions()
	if !ok {
		return warpSample{}, false, ErrInvalidDecodeState
	}
	return recordWarpSample(n.Motion.MV[0], int(n.RelX4), 0, 1, -1, int(dims.W4), int(dims.H4)), true, nil
}

func recordWarpSample(mv motion.Vector, colOffset4 int, rowOffset4 int, signC int, signR int, w4 int, h4 int) warpSample {
	x := colOffset4*4 + signC*(w4*4)/2 - 1
	y := rowOffset4*4 + signR*(h4*4)/2 - 1
	return warpSample{
		X:      x * motion.SubpelScale,
		Y:      y * motion.SubpelScale,
		RefX:   x*motion.SubpelScale + int(mv.Col),
		RefY:   y*motion.SubpelScale + int(mv.Row),
		Motion: mv,
	}
}

func selectWarpSamples(samples []warpSample, size BlockSize, mv motion.Vector) int {
	dims, ok := size.Dimensions()
	if !ok {
		return len(samples)
	}
	thresh := maxInt(int(dims.W4)*4, int(dims.H4)*4)
	thresh = minInt(maxInt(thresh, 16), 112)
	ret := 0
	for i := range samples {
		diff := absInt(samples[i].RefX-samples[i].X-int(mv.Col)) +
			absInt(samples[i].RefY-samples[i].Y-int(mv.Row))
		if diff > thresh {
			continue
		}
		if ret != i {
			samples[ret] = samples[i]
		}
		ret++
	}
	if ret == 0 {
		return 1
	}
	return ret
}

func warpLeastSquaresSquare(a int) int {
	return (a*a*4 + a*4*motion.SubpelScale + motion.SubpelScale*motion.SubpelScale*2) >> 4
}

func warpLeastSquaresProduct1(a int, b int) int {
	return (a*b*4 + (a+b)*2*motion.SubpelScale + motion.SubpelScale*motion.SubpelScale) >> 4
}

func (c *BlockModeContext) aboveOverlappable(slot int) bool {
	return slot >= 0 && slot < MaxBlockModeSlots && c.AboveIntra[slot] == 0 && c.AboveRef[0][slot].Valid()
}

func (c *BlockModeContext) leftOverlappable(slot int) bool {
	return slot >= 0 && slot < MaxBlockModeSlots && c.LeftIntra[slot] == 0 && c.LeftRef[0][slot].Valid()
}

func (c *BlockModeContext) aboveOverlappableStep(slot int) int {
	if slot < 0 || slot >= MaxBlockModeSlots {
		return 1
	}
	if dims, ok := c.AboveBlockSize[slot].Dimensions(); ok {
		return maxInt(1, minInt(int(dims.W4), 16))
	}
	return 1
}

func (c *BlockModeContext) leftOverlappableStep(slot int) int {
	if slot < 0 || slot >= MaxBlockModeSlots {
		return 1
	}
	if dims, ok := c.LeftBlockSize[slot].Dimensions(); ok {
		return maxInt(1, minInt(int(dims.H4), 16))
	}
	return 1
}

// LastMotionModeAllowed ports libaom's motion_mode_allowed() result. It returns
// the largest syntax mode that may be read for this block.
func LastMotionModeAllowed(req MotionModeRequest) (MotionMode, error) {
	if _, ok := req.Size.Dimensions(); !ok {
		return 0, ErrInvalidDecodeState
	}
	if !req.Compound && !req.Mode.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if !globalMotionTypeValid(req.GlobalMotionType) || req.OverlappableNeighbors < 0 || req.NumProjRef < 0 {
		return 0, ErrInvalidDecodeState
	}
	if !req.SwitchableMotionMode || req.SkipMode || req.InterIntra || req.OverlappableNeighbors == 0 {
		return MotionModeTranslation, nil
	}
	if !motionModeBlockSizeAllowed(req.Size) || req.Compound {
		return MotionModeTranslation, nil
	}
	if !req.ForceIntegerMV && req.Mode == InterModeGlobalMV && req.GlobalMotionType > parser.GlobalMotionTranslation {
		return MotionModeTranslation, nil
	}
	if req.NumProjRef >= 1 && req.AllowWarpedMotion && !req.ForceIntegerMV && !req.ScaledReference {
		return MotionModeWarp, nil
	}
	return MotionModeOBMC, nil
}

func (s *DecodeState) ReadMotionMode(cdfs *MotionModeCDFs, req MotionModeRequest) (MotionMode, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	last, err := LastMotionModeAllowed(req)
	if err != nil {
		return 0, err
	}
	switch last {
	case MotionModeTranslation:
		return MotionModeTranslation, nil
	case MotionModeOBMC:
		cdf, err := cdfs.OBMCCDF(req.Size)
		if err != nil {
			return 0, err
		}
		obmc, err := readBoolCDF(s, cdf)
		if err != nil {
			return 0, err
		}
		if obmc {
			return MotionModeOBMC, nil
		}
		return MotionModeTranslation, nil
	case MotionModeWarp:
		cdf, err := cdfs.MotionModeCDF(req.Size)
		if err != nil {
			return 0, err
		}
		symbol, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return 0, err
		}
		mode := MotionMode(symbol)
		if !mode.Valid() {
			return 0, ErrInvalidDecodeState
		}
		return mode, nil
	default:
		return 0, ErrInvalidDecodeState
	}
}

func motionModeBlockSizeAllowed(size BlockSize) bool {
	dims, ok := size.Dimensions()
	return ok && dims.W4 >= 2 && dims.H4 >= 2
}

func globalMotionTypeValid(t parser.GlobalMotionType) bool {
	return t <= parser.GlobalMotionAffine
}
