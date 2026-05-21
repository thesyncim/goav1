package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	LumaIntraModeContexts     = 4
	KeyframeIntraModeContexts = 5
)

// IntraMode identifies an AV1 luma intra prediction mode.
type IntraMode uint8

const (
	IntraModeDC IntraMode = iota
	IntraModeVertical
	IntraModeHorizontal
	IntraModeD45
	IntraModeD135
	IntraModeD113
	IntraModeD157
	IntraModeD203
	IntraModeD67
	IntraModeSmooth
	IntraModeSmoothVertical
	IntraModeSmoothHorizontal
	IntraModePaeth
	intraModeCount
)

// IntraModeCDFs contains caller-owned CDFs for AV1 intra/inter entry and luma
// intra mode syntax.
type IntraModeCDFs struct {
	Intra         [LumaIntraModeContexts]entropy.CDF
	Intrabc       entropy.CDF
	YMode         [LumaIntraModeContexts]entropy.CDF
	KeyframeYMode [KeyframeIntraModeContexts][KeyframeIntraModeContexts]entropy.CDF
}

// IntraFlagRequest describes the frame/block conditions used to decide whether
// intra is implicit or entropy-coded.
type IntraFlagRequest struct {
	FrameType    parser.FrameType
	AllowIntrabc bool
	SkipMode     bool

	SegmentationEnabled bool
	Segment             parser.SegmentData

	X4       int
	Y4       int
	HaveTop  bool
	HaveLeft bool
}

// LumaIntraModeRequest describes one luma intra mode symbol.
type LumaIntraModeRequest struct {
	FrameType parser.FrameType
	Size      BlockSize
	X4        int
	Y4        int
}

// BlockPredictionModeResult is the luma entry/mode syntax decoded after the
// block prefix. Inter-specific reference and MV syntax is decoded by later
// mode stages.
type BlockPredictionModeResult struct {
	Valid bool
	Intra bool

	LumaMode IntraMode

	InterReferences      InterReferencesResult
	InterReferencesValid bool
}

var yModeSizeContext = [blockSizeCount]int{
	BlockSize128x128: 3,
	BlockSize128x64:  3,
	BlockSize64x128:  3,
	BlockSize64x64:   3,
	BlockSize64x32:   3,
	BlockSize64x16:   2,
	BlockSize32x64:   3,
	BlockSize32x32:   3,
	BlockSize32x16:   2,
	BlockSize32x8:    1,
	BlockSize16x64:   2,
	BlockSize16x32:   2,
	BlockSize16x16:   2,
	BlockSize16x8:    1,
	BlockSize16x4:    0,
	BlockSize8x32:    1,
	BlockSize8x16:    1,
	BlockSize8x8:     1,
	BlockSize8x4:     0,
	BlockSize4x16:    0,
	BlockSize4x8:     0,
	BlockSize4x4:     0,
}

var intraModeContext = [intraModeCount]int{
	IntraModeDC:               0,
	IntraModeVertical:         1,
	IntraModeHorizontal:       2,
	IntraModeD45:              3,
	IntraModeD135:             4,
	IntraModeD113:             4,
	IntraModeD157:             4,
	IntraModeD203:             4,
	IntraModeD67:              3,
	IntraModeSmooth:           0,
	IntraModeSmoothVertical:   1,
	IntraModeSmoothHorizontal: 2,
	IntraModePaeth:            0,
}

// Valid reports whether mode is an AV1 luma intra prediction mode.
func (mode IntraMode) Valid() bool {
	return mode < intraModeCount
}

// InitDefault seeds c with dav1d/libaom's default intra-mode CDFs.
func (c *IntraModeCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next IntraModeCDFs
	for i, cumulative := range [...]uint16{806, 16662, 20186, 26538} {
		if err := next.Intra[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	if err := next.Intrabc.Init([]uint16{30531}); err != nil {
		return err
	}
	defaultYMode := [...][intraModeCount - 1]uint16{
		{22801, 23489, 24293, 24756, 25601, 26123, 26606, 27418, 27945, 29228, 29685, 30349},
		{18673, 19845, 22631, 23318, 23950, 24649, 25527, 27364, 28152, 29701, 29984, 30852},
		{19770, 20979, 23396, 23939, 24241, 24654, 25136, 27073, 27830, 29360, 29730, 30659},
		{20155, 21301, 22838, 23178, 23261, 23533, 23703, 24804, 25352, 26575, 27016, 28049},
	}
	for i := range defaultYMode {
		if err := next.YMode[i].Init(defaultYMode[i][:]); err != nil {
			return err
		}
	}
	defaultKeyframeYMode := [...][KeyframeIntraModeContexts][intraModeCount - 1]uint16{
		{
			{15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466},
			{12016, 18066, 19516, 20303, 20719, 21444, 21888, 23032, 24434, 28658, 30172, 31409},
			{10052, 10771, 22296, 22788, 23055, 23239, 24133, 25620, 26160, 29336, 29929, 31567},
			{14091, 15406, 16442, 18808, 19136, 19546, 19998, 22096, 24746, 29585, 30958, 32462},
			{12122, 13265, 15603, 16501, 18609, 20033, 22391, 25583, 26437, 30261, 31073, 32475},
		},
		{
			{10023, 19585, 20848, 21440, 21832, 22760, 23089, 24023, 25381, 29014, 30482, 31436},
			{5983, 24099, 24560, 24886, 25066, 25795, 25913, 26423, 27610, 29905, 31276, 31794},
			{7444, 12781, 20177, 20728, 21077, 21607, 22170, 23405, 24469, 27915, 29090, 30492},
			{8537, 14689, 15432, 17087, 17408, 18172, 18408, 19825, 24649, 29153, 31096, 32210},
			{7543, 14231, 15496, 16195, 17905, 20717, 21984, 24516, 26001, 29675, 30981, 31994},
		},
		{
			{12613, 13591, 21383, 22004, 22312, 22577, 23401, 25055, 25729, 29538, 30305, 32077},
			{9687, 13470, 18506, 19230, 19604, 20147, 20695, 22062, 23219, 27743, 29211, 30907},
			{6183, 6505, 26024, 26252, 26366, 26434, 27082, 28354, 28555, 30467, 30794, 32086},
			{10718, 11734, 14954, 17224, 17565, 17924, 18561, 21523, 23878, 28975, 30287, 32252},
			{9194, 9858, 16501, 17263, 18424, 19171, 21563, 25961, 26561, 30072, 30737, 32463},
		},
		{
			{12602, 14399, 15488, 18381, 18778, 19315, 19724, 21419, 25060, 29696, 30917, 32409},
			{8203, 13821, 14524, 17105, 17439, 18131, 18404, 19468, 25225, 29485, 31158, 32342},
			{8451, 9731, 15004, 17643, 18012, 18425, 19070, 21538, 24605, 29118, 30078, 32018},
			{7714, 9048, 9516, 16667, 16817, 16994, 17153, 18767, 26743, 30389, 31536, 32528},
			{8843, 10280, 11496, 15317, 16652, 17943, 19108, 22718, 25769, 29953, 30983, 32485},
		},
		{
			{12578, 13671, 15979, 16834, 19075, 20913, 22989, 25449, 26219, 30214, 31150, 32477},
			{9563, 13626, 15080, 15892, 17756, 20863, 22207, 24236, 25380, 29653, 31143, 32277},
			{8356, 8901, 17616, 18256, 19350, 20106, 22598, 25947, 26466, 29900, 30523, 32261},
			{10835, 11815, 13124, 16042, 17018, 18039, 18947, 22753, 24615, 29489, 30883, 32482},
			{7618, 8288, 9859, 10509, 15386, 18657, 22903, 28776, 29180, 31355, 31802, 32593},
		},
	}
	for above := range defaultKeyframeYMode {
		for left := range defaultKeyframeYMode[above] {
			if err := next.KeyframeYMode[above][left].Init(defaultKeyframeYMode[above][left][:]); err != nil {
				return err
			}
		}
	}
	*c = next
	return nil
}

// IntraCDF returns the initialized intra/inter flag CDF for ctx.
func (c *IntraModeCDFs) IntraCDF(ctx int) (*entropy.CDF, error) {
	return intraModeCDF(c, cdfIntra, ctx, 2)
}

// IntrabcCDF returns the initialized intrabc flag CDF.
func (c *IntraModeCDFs) IntrabcCDF() (*entropy.CDF, error) {
	if c == nil || c.Intrabc.Symbols() != 2 {
		return nil, entropy.ErrInvalidCDF
	}
	return &c.Intrabc, c.Intrabc.Validate()
}

// YModeCDF returns the initialized inter-frame luma intra-mode CDF for ctx.
func (c *IntraModeCDFs) YModeCDF(ctx int) (*entropy.CDF, error) {
	return intraModeCDF(c, cdfYMode, ctx, int(intraModeCount))
}

// KeyframeYModeCDF returns the initialized keyframe luma intra-mode CDF for
// the above/left mode contexts.
func (c *IntraModeCDFs) KeyframeYModeCDF(above int, left int) (*entropy.CDF, error) {
	if c == nil || above < 0 || above >= KeyframeIntraModeContexts ||
		left < 0 || left >= KeyframeIntraModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.KeyframeYMode[above][left]
	if cdf.Symbols() != int(intraModeCount) {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// IntraContext returns dav1d's get_intra_ctx() value.
func (c *BlockModeContext) IntraContext(x4 int, y4 int, haveTop bool, haveLeft bool) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	if haveLeft {
		left := int(c.LeftIntra[y4])
		if haveTop {
			ctx := left + int(c.AboveIntra[x4])
			if ctx == 2 {
				return 3, nil
			}
			return ctx, nil
		}
		return left * 2, nil
	}
	if haveTop {
		return int(c.AboveIntra[x4]) * 2, nil
	}
	return 0, nil
}

// KeyframeYModeContext returns libaom's intra_mode_context[] pair for the
// above/left luma prediction modes.
func (c *BlockModeContext) KeyframeYModeContext(x4 int, y4 int) (int, int, error) {
	if c == nil {
		return 0, 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, 0, err
	}
	above := c.AboveMode[x4]
	left := c.LeftMode[y4]
	if !above.Valid() || !left.Valid() {
		return 0, 0, ErrInvalidDecodeState
	}
	return intraModeContext[above], intraModeContext[left], nil
}

// MarkIntra updates the top/left intra contexts after a decoded intra/inter
// block mode decision.
func (c *BlockModeContext) MarkIntra(size BlockSize, x4 int, y4 int, intra bool, mode IntraMode) error {
	if c == nil || !mode.Valid() {
		return ErrInvalidDecodeState
	}
	if err := c.MarkIntraEntry(size, x4, y4, intra, mode); err != nil {
		return err
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	ref0 := ReferenceFrameNone
	ref1 := ReferenceFrameNone
	if !intra {
		ref0 = ReferenceFrameLast
	}
	for i := 0; i < int(dims.W4); i++ {
		c.AboveRef[0][x4+i] = ref0
		c.AboveRef[1][x4+i] = ref1
		c.AboveCompound[x4+i] = 0
		c.AboveInterMotion[x4+i] = InterMotionResult{}
		c.AboveMotionValid[x4+i] = 0
		c.AboveBlockSize[x4+i] = size
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftRef[0][y4+i] = ref0
		c.LeftRef[1][y4+i] = ref1
		c.LeftCompound[y4+i] = 0
		c.LeftInterMotion[y4+i] = InterMotionResult{}
		c.LeftMotionValid[y4+i] = 0
		c.LeftBlockSize[y4+i] = size
	}
	return nil
}

// MarkIntraEntry updates only the intra/inter entry context and, for intra
// blocks, the luma prediction mode context. Inter reference contexts are left
// for the inter-reference reader to fill with the actual decoded refs.
func (c *BlockModeContext) MarkIntraEntry(size BlockSize, x4 int, y4 int, intra bool, mode IntraMode) error {
	if c == nil || (intra && !mode.Valid()) {
		return ErrInvalidDecodeState
	}
	if !intra {
		mode = IntraModeDC
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
	intraValue := boolByte(intra)
	for i := 0; i < int(dims.W4); i++ {
		c.AboveIntra[x4+i] = intraValue
		c.AboveMode[x4+i] = mode
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftIntra[y4+i] = intraValue
		c.LeftMode[y4+i] = mode
	}
	return nil
}

// ReadIntraFlag decodes whether one block is intra-coded.
func (s *DecodeState) ReadIntraFlag(cdfs *IntraModeCDFs, ctx *BlockModeContext, req IntraFlagRequest) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	if req.SkipMode {
		return false, nil
	}
	if frameTypeIsInterOrSwitch(req.FrameType) {
		if req.SegmentationEnabled && (req.Segment.RefFrame >= 0 || req.Segment.GlobalMV) {
			return req.Segment.RefFrame == 0 && !req.Segment.GlobalMV, nil
		}
		context, err := ctx.IntraContext(req.X4, req.Y4, req.HaveTop, req.HaveLeft)
		if err != nil {
			return false, err
		}
		cdf, err := cdfs.IntraCDF(context)
		if err != nil {
			return false, err
		}
		isInter, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return false, err
		}
		return isInter == 0, nil
	}
	if req.AllowIntrabc {
		cdf, err := cdfs.IntrabcCDF()
		if err != nil {
			return false, err
		}
		intrabc, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return false, err
		}
		return intrabc == 0, nil
	}
	return true, nil
}

// ReadLumaIntraMode decodes one luma intra prediction mode.
func (s *DecodeState) ReadLumaIntraMode(cdfs *IntraModeCDFs, ctx *BlockModeContext, req LumaIntraModeRequest) (IntraMode, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok {
		return 0, ErrInvalidDecodeState
	}

	var cdf *entropy.CDF
	var err error
	if frameTypeIsInterOrSwitch(req.FrameType) {
		cdf, err = cdfs.YModeCDF(yModeSizeContext[req.Size])
	} else {
		above, left, ctxErr := ctx.KeyframeYModeContext(req.X4, req.Y4)
		if ctxErr != nil {
			return 0, ctxErr
		}
		cdf, err = cdfs.KeyframeYModeCDF(above, left)
	}
	if err != nil {
		return 0, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	mode := IntraMode(symbol)
	if !mode.Valid() {
		return 0, ErrInvalidDecodeState
	}
	return mode, nil
}

type intraCDFKind uint8

const (
	cdfIntra intraCDFKind = iota
	cdfYMode
)

func intraModeCDF(c *IntraModeCDFs, kind intraCDFKind, ctx int, symbols int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= LumaIntraModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	var cdf *entropy.CDF
	switch kind {
	case cdfIntra:
		cdf = &c.Intra[ctx]
	case cdfYMode:
		cdf = &c.YMode[ctx]
	default:
		return nil, entropy.ErrInvalidCDF
	}
	if cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func frameTypeIsInterOrSwitch(frameType parser.FrameType) bool {
	return frameType == parser.FrameTypeInter || frameType == parser.FrameTypeSwitch
}
