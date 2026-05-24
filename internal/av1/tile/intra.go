package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	LumaIntraModeContexts     = 4
	KeyframeIntraModeContexts = 5
	CFLAllowedTypes           = 2
	DirectionalIntraModes     = 8
	FilterIntraModes          = 5
	CFLJointSigns             = 8
	CFLAlphaContexts          = 6
	CFLAlphabetSize           = 16
	AngleDeltaMax             = 3
	AngleDeltaStep            = 3
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

// FilterIntraMode identifies libaom's FILTER_INTRA_MODE values.
type FilterIntraMode uint8

const (
	FilterIntraModeDC FilterIntraMode = iota
	FilterIntraModeVertical
	FilterIntraModeHorizontal
	FilterIntraModeD157
	FilterIntraModePaeth
)

// ChromaIntraMode identifies an AV1 UV intra prediction mode.
type ChromaIntraMode uint8

const (
	ChromaIntraModeDC ChromaIntraMode = iota
	ChromaIntraModeVertical
	ChromaIntraModeHorizontal
	ChromaIntraModeD45
	ChromaIntraModeD135
	ChromaIntraModeD113
	ChromaIntraModeD157
	ChromaIntraModeD203
	ChromaIntraModeD67
	ChromaIntraModeSmooth
	ChromaIntraModeSmoothVertical
	ChromaIntraModeSmoothHorizontal
	ChromaIntraModePaeth
	ChromaIntraModeCFL
	chromaIntraModeCount
)

// CFLAlphaResult carries libaom's packed CfL alpha index and joint-sign symbol.
type CFLAlphaResult struct {
	Index     uint8
	JointSign int8
}

// IntraModeCDFs contains caller-owned CDFs for AV1 intra/inter entry, luma/UV
// intra mode, directional angle delta, and CfL alpha syntax.
type IntraModeCDFs struct {
	Intra           [LumaIntraModeContexts]entropy.CDF
	Intrabc         entropy.CDF
	YMode           [LumaIntraModeContexts]entropy.CDF
	KeyframeYMode   [KeyframeIntraModeContexts][KeyframeIntraModeContexts]entropy.CDF
	UVMode          [CFLAllowedTypes][intraModeCount]entropy.CDF
	AngleDelta      [DirectionalIntraModes]entropy.CDF
	CFLSign         entropy.CDF
	CFLAlpha        [CFLAlphaContexts]entropy.CDF
	FilterIntra     [blockSizeCount]entropy.CDF
	FilterIntraMode entropy.CDF
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

// IntraAngleDeltaRequest describes one directional intra angle delta symbol.
type IntraAngleDeltaRequest struct {
	Size BlockSize
	Mode IntraMode
}

// ChromaIntraModeRequest describes one UV intra mode symbol.
type ChromaIntraModeRequest struct {
	Size       BlockSize
	LumaMode   IntraMode
	CFLAllowed bool
}

// FilterIntraRequest describes one optional filter-intra mode symbol.
type FilterIntraRequest struct {
	EnableFilterIntra bool
	Size              BlockSize
	LumaMode          IntraMode
	PaletteYSize      uint8
}

// BlockPredictionModeResult is the luma entry/mode syntax decoded after the
// block prefix. Inter MV residuals and reconstruction are decoded by later
// mode stages.
type BlockPredictionModeResult struct {
	Valid bool
	Intra bool

	LumaMode       IntraMode
	LumaAngleDelta int8

	IntraEdgeSmoothNeighbor       bool
	ChromaIntraEdgeSmoothNeighbor bool

	FilterIntraMode  FilterIntraMode
	FilterIntraValid bool

	ChromaMode       ChromaIntraMode
	ChromaModeValid  bool
	ChromaAngleDelta int8
	CFLAlpha         CFLAlphaResult
	CFLAlphaValid    bool

	InterReferences      InterReferencesResult
	InterReferencesValid bool

	InterMode      InterModeResult
	InterModeValid bool

	ReferenceMVStack      ReferenceMVStackResult
	ReferenceMVStackValid bool

	DRLIndex      int
	DRLIndexValid bool

	InterMVReferences      InterMVReferenceSet
	InterMVReferencesValid bool

	InterMotion      InterMotionResult
	InterMotionValid bool

	MVResiduals     [2]MVResidualResult
	MVResidualValid [2]bool

	InterpFilters      motion.InterpFilters
	InterpFiltersValid bool
	InterpFilterReads  int

	InterIntra      InterIntraResult
	InterIntraValid bool

	MotionMode      MotionMode
	MotionModeValid bool

	CompoundBlend      CompoundBlendResult
	CompoundBlendValid bool
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

// Valid reports whether mode is an AV1 filter-intra prediction mode.
func (mode FilterIntraMode) Valid() bool {
	return mode < FilterIntraModes
}

// Valid reports whether mode is an AV1 UV intra prediction mode.
func (mode ChromaIntraMode) Valid() bool {
	return mode < chromaIntraModeCount
}

// LumaMode returns the luma predictor that libaom uses for this UV mode.
func (mode ChromaIntraMode) LumaMode() (IntraMode, error) {
	if !mode.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if mode == ChromaIntraModeCFL {
		return IntraModeDC, nil
	}
	return IntraMode(mode), nil
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
	defaultUVNoCFL := [...][chromaIntraModeCount - 2]uint16{
		{22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923, 28244, 30059, 30941, 31961},
		{9513, 26881, 26973, 27046, 27118, 27664, 27739, 27824, 28359, 29505, 29800, 31796},
		{9845, 9915, 28663, 28704, 28757, 28780, 29198, 29822, 29854, 30764, 31777, 32029},
		{13639, 13897, 14171, 25331, 25606, 25727, 25953, 27148, 28577, 30612, 31355, 32493},
		{9764, 9835, 9930, 9954, 25386, 27053, 27958, 28148, 28243, 31101, 31744, 32363},
		{11825, 13589, 13677, 13720, 15048, 29213, 29301, 29458, 29711, 31161, 31441, 32550},
		{14175, 14399, 16608, 16821, 17718, 17775, 28551, 30200, 30245, 31837, 32342, 32667},
		{12885, 13038, 14978, 15590, 15673, 15748, 16176, 29128, 29267, 30643, 31961, 32461},
		{12026, 13661, 13874, 15305, 15490, 15726, 15995, 16273, 28443, 30388, 30767, 32416},
		{19052, 19840, 20579, 20916, 21150, 21467, 21885, 22719, 23174, 28861, 30379, 32175},
		{18627, 19649, 20974, 21219, 21492, 21816, 22199, 23119, 23527, 27053, 31397, 32148},
		{17026, 19004, 19997, 20339, 20586, 21103, 21349, 21907, 22482, 25896, 26541, 31819},
		{12124, 13759, 14959, 14992, 15007, 15051, 15078, 15166, 15255, 15753, 16039, 16606},
	}
	defaultUVCFL := [...][chromaIntraModeCount - 1]uint16{
		{10407, 11208, 12900, 13181, 13823, 14175, 14899, 15656, 15986, 20086, 20995, 22455, 24212},
		{4532, 19780, 20057, 20215, 20428, 21071, 21199, 21451, 22099, 24228, 24693, 27032, 29472},
		{5273, 5379, 20177, 20270, 20385, 20439, 20949, 21695, 21774, 23138, 24256, 24703, 26679},
		{6740, 7167, 7662, 14152, 14536, 14785, 15034, 16741, 18371, 21520, 22206, 23389, 24182},
		{4987, 5368, 5928, 6068, 19114, 20315, 21857, 22253, 22411, 24911, 25380, 26027, 26376},
		{5370, 6889, 7247, 7393, 9498, 21114, 21402, 21753, 21981, 24780, 25386, 26517, 27176},
		{4816, 4961, 7204, 7326, 8765, 8930, 20169, 20682, 20803, 23188, 23763, 24455, 24940},
		{6608, 6740, 8529, 9049, 9257, 9356, 9735, 18827, 19059, 22336, 23204, 23964, 24793},
		{5998, 7419, 7781, 8933, 9255, 9549, 9753, 10417, 18898, 22494, 23139, 24764, 25989},
		{10660, 11298, 12550, 12957, 13322, 13624, 14040, 15004, 15534, 20714, 21789, 23443, 24861},
		{10522, 11530, 12552, 12963, 13378, 13779, 14245, 15235, 15902, 20102, 22696, 23774, 25838},
		{10099, 10691, 12639, 13049, 13386, 13665, 14125, 15163, 15636, 19676, 20474, 23519, 25208},
		{3144, 5087, 7382, 7504, 7593, 7690, 7801, 8064, 8232, 9248, 9875, 10521, 29048},
	}
	for y := range defaultUVNoCFL {
		if err := next.UVMode[0][y].Init(defaultUVNoCFL[y][:]); err != nil {
			return err
		}
		if err := next.UVMode[1][y].Init(defaultUVCFL[y][:]); err != nil {
			return err
		}
	}
	defaultAngleDelta := [...][2 * AngleDeltaMax]uint16{
		{2180, 5032, 7567, 22776, 26989, 30217},
		{2301, 5608, 8801, 23487, 26974, 30330},
		{3780, 11018, 13699, 19354, 23083, 31286},
		{4581, 11226, 15147, 17138, 21834, 28397},
		{1737, 10927, 14509, 19588, 22745, 28823},
		{2664, 10176, 12485, 17650, 21600, 30495},
		{2240, 11096, 15453, 20341, 22561, 28917},
		{3605, 10428, 12459, 17676, 21244, 30655},
	}
	for i := range defaultAngleDelta {
		if err := next.AngleDelta[i].Init(defaultAngleDelta[i][:]); err != nil {
			return err
		}
	}
	if err := next.CFLSign.Init([]uint16{1418, 2123, 13340, 18405, 26972, 28343, 32294}); err != nil {
		return err
	}
	defaultCFLAlpha := [...][CFLAlphabetSize - 1]uint16{
		{7637, 20719, 31401, 32481, 32657, 32688, 32692, 32696, 32700, 32704, 32708, 32712, 32716, 32720, 32724},
		{14365, 23603, 28135, 31168, 32167, 32395, 32487, 32573, 32620, 32647, 32668, 32672, 32676, 32680, 32684},
		{11532, 22380, 28445, 31360, 32349, 32523, 32584, 32649, 32673, 32677, 32681, 32685, 32689, 32693, 32697},
		{26990, 31402, 32282, 32571, 32692, 32696, 32700, 32704, 32708, 32712, 32716, 32720, 32724, 32728, 32732},
		{17248, 26058, 28904, 30608, 31305, 31877, 32126, 32321, 32394, 32464, 32516, 32560, 32576, 32593, 32622},
		{14738, 21678, 25779, 27901, 29024, 30302, 30980, 31843, 32144, 32413, 32520, 32594, 32622, 32656, 32660},
	}
	for i := range defaultCFLAlpha {
		if err := next.CFLAlpha[i].Init(defaultCFLAlpha[i][:]); err != nil {
			return err
		}
	}
	defaultFilterIntra := [blockSizeCount]uint16{
		BlockSize128x128: 16384,
		BlockSize128x64:  16384,
		BlockSize64x128:  16384,
		BlockSize64x64:   16384,
		BlockSize64x32:   16384,
		BlockSize64x16:   16384,
		BlockSize32x64:   16384,
		BlockSize32x32:   22343,
		BlockSize32x16:   12756,
		BlockSize32x8:    18101,
		BlockSize16x64:   16384,
		BlockSize16x32:   14301,
		BlockSize16x16:   12408,
		BlockSize16x8:    9394,
		BlockSize16x4:    10368,
		BlockSize8x32:    20229,
		BlockSize8x16:    12551,
		BlockSize8x8:     7866,
		BlockSize8x4:     5893,
		BlockSize4x16:    12770,
		BlockSize4x8:     6743,
		BlockSize4x4:     4621,
	}
	for i := range defaultFilterIntra {
		if err := next.FilterIntra[i].Init([]uint16{defaultFilterIntra[i]}); err != nil {
			return err
		}
	}
	if err := next.FilterIntraMode.Init([]uint16{8949, 12776, 17211, 29558}); err != nil {
		return err
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

// UVModeCDF returns the initialized UV intra-mode CDF for luma mode and CfL
// availability.
func (c *IntraModeCDFs) UVModeCDF(cflAllowed bool, lumaMode IntraMode) (*entropy.CDF, error) {
	if c == nil || !lumaMode.Valid() {
		return nil, entropy.ErrInvalidCDF
	}
	allowed := 0
	symbols := int(chromaIntraModeCount - 1)
	if cflAllowed {
		allowed = 1
		symbols = int(chromaIntraModeCount)
	}
	cdf := &c.UVMode[allowed][lumaMode]
	if cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// AngleDeltaCDF returns the initialized directional intra angle-delta CDF.
func (c *IntraModeCDFs) AngleDeltaCDF(mode IntraMode) (*entropy.CDF, error) {
	if c == nil || !isDirectionalIntraMode(mode) {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.AngleDelta[mode-IntraModeVertical]
	if cdf.Symbols() != 2*AngleDeltaMax+1 {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// CFLSignCDF returns the initialized CfL joint-sign CDF.
func (c *IntraModeCDFs) CFLSignCDF() (*entropy.CDF, error) {
	if c == nil || c.CFLSign.Symbols() != CFLJointSigns {
		return nil, entropy.ErrInvalidCDF
	}
	return &c.CFLSign, c.CFLSign.Validate()
}

// CFLAlphaCDF returns the initialized CfL alpha-magnitude CDF for ctx.
func (c *IntraModeCDFs) CFLAlphaCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= CFLAlphaContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.CFLAlpha[ctx]
	if cdf.Symbols() != CFLAlphabetSize {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// FilterIntraCDF returns the initialized filter-intra use flag CDF for size.
func (c *IntraModeCDFs) FilterIntraCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || size >= blockSizeCount {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.FilterIntra[size]
	if cdf.Symbols() != 2 {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// FilterIntraModeCDF returns the initialized filter-intra mode CDF.
func (c *IntraModeCDFs) FilterIntraModeCDF() (*entropy.CDF, error) {
	if c == nil || c.FilterIntraMode.Symbols() != FilterIntraModes {
		return nil, entropy.ErrInvalidCDF
	}
	return &c.FilterIntraMode, c.FilterIntraMode.Validate()
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

// IntraEdgeSmoothNeighbor reports libaom's luma get_intra_edge_filter_type()
// decision: whether the immediate above or left luma neighbor is an intra
// smooth mode. It must be called before the current block is marked into the
// top/left mode context.
func (c *BlockModeContext) IntraEdgeSmoothNeighbor(x4 int, y4 int, haveTop bool, haveLeft bool) (bool, error) {
	if c == nil {
		return false, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return false, err
	}
	if haveTop && c.AboveIntra[x4] != 0 {
		smooth, err := intraModeIsSmooth(c.AboveMode[x4])
		if err != nil {
			return false, err
		}
		if smooth {
			return true, nil
		}
	}
	if haveLeft && c.LeftIntra[y4] != 0 {
		smooth, err := intraModeIsSmooth(c.LeftMode[y4])
		if err != nil {
			return false, err
		}
		if smooth {
			return true, nil
		}
	}
	return false, nil
}

// ChromaIntraEdgeSmoothNeighbor reports libaom's chroma
// get_intra_edge_filter_type() decision: whether the immediate above or left
// chroma neighbor is an intra smooth UV mode. It must be called before the
// current block is marked into the chroma top/left mode context.
func (c *BlockModeContext) ChromaIntraEdgeSmoothNeighbor(x4 int, y4 int, haveTop bool, haveLeft bool, subsamplingX bool, subsamplingY bool) (bool, error) {
	if c == nil {
		return false, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return false, err
	}
	ssX := int(boolToShift(subsamplingX))
	ssY := int(boolToShift(subsamplingY))
	baseX4 := x4 - (x4 & ssX)
	baseY4 := y4 - (y4 & ssY)
	aboveX4 := baseX4 + ssX
	leftY4 := baseY4 + ssY
	if aboveX4 < 0 || aboveX4 >= MaxBlockModeSlots || leftY4 < 0 || leftY4 >= MaxBlockModeSlots {
		return false, ErrInvalidDecodeState
	}
	if haveTop && c.AboveChromaIntra[aboveX4] != 0 {
		smooth, err := chromaModeIsSmooth(c.AboveChromaMode[aboveX4])
		if err != nil {
			return false, err
		}
		if smooth {
			return true, nil
		}
	}
	if haveLeft && c.LeftChromaIntra[leftY4] != 0 {
		smooth, err := chromaModeIsSmooth(c.LeftChromaMode[leftY4])
		if err != nil {
			return false, err
		}
		if smooth {
			return true, nil
		}
	}
	return false, nil
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

func intraModeIsSmooth(mode IntraMode) (bool, error) {
	switch mode {
	case IntraModeSmooth, IntraModeSmoothVertical, IntraModeSmoothHorizontal:
		return true, nil
	case IntraModeDC, IntraModeVertical, IntraModeHorizontal, IntraModeD45,
		IntraModeD135, IntraModeD113, IntraModeD157, IntraModeD203,
		IntraModeD67, IntraModePaeth:
		return false, nil
	default:
		return false, ErrInvalidDecodeState
	}
}

func chromaModeIsSmooth(mode ChromaIntraMode) (bool, error) {
	switch mode {
	case ChromaIntraModeSmooth, ChromaIntraModeSmoothVertical, ChromaIntraModeSmoothHorizontal:
		return true, nil
	case ChromaIntraModeDC, ChromaIntraModeVertical, ChromaIntraModeHorizontal, ChromaIntraModeD45,
		ChromaIntraModeD135, ChromaIntraModeD113, ChromaIntraModeD157, ChromaIntraModeD203,
		ChromaIntraModeD67, ChromaIntraModePaeth, ChromaIntraModeCFL:
		return false, nil
	default:
		return false, ErrInvalidDecodeState
	}
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
		c.AboveInterp[x4+i] = motion.InterpFilters{}
		c.AboveInterpValid[x4+i] = 0
		c.AboveBlockSize[x4+i] = size
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftRef[0][y4+i] = ref0
		c.LeftRef[1][y4+i] = ref1
		c.LeftCompound[y4+i] = 0
		c.LeftInterMotion[y4+i] = InterMotionResult{}
		c.LeftMotionValid[y4+i] = 0
		c.LeftInterp[y4+i] = motion.InterpFilters{}
		c.LeftInterpValid[y4+i] = 0
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

// MarkChromaIntra updates the top/left chroma intra mode context for blocks
// that carry chroma samples.
func (c *BlockModeContext) MarkChromaIntra(size BlockSize, x4 int, y4 int, intra bool, mode ChromaIntraMode) error {
	if c == nil || (intra && !mode.Valid()) {
		return ErrInvalidDecodeState
	}
	if !intra {
		mode = ChromaIntraModeDC
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
		c.AboveChromaIntra[x4+i] = intraValue
		c.AboveChromaMode[x4+i] = mode
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftChromaIntra[y4+i] = intraValue
		c.LeftChromaMode[y4+i] = mode
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

// ReadIntraAngleDelta decodes the optional directional intra angle delta.
func (s *DecodeState) ReadIntraAngleDelta(cdfs *IntraModeCDFs, req IntraAngleDeltaRequest) (int8, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	read, err := shouldReadIntraAngleDelta(req.Size, req.Mode)
	if err != nil || !read {
		return 0, err
	}
	cdf, err := cdfs.AngleDeltaCDF(req.Mode)
	if err != nil {
		return 0, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	return int8(symbol - AngleDeltaMax), nil
}

// ReadChromaIntraMode decodes the UV intra mode and optional CfL alphas.
func (s *DecodeState) ReadChromaIntraMode(cdfs *IntraModeCDFs, req ChromaIntraModeRequest) (ChromaIntraMode, CFLAlphaResult, error) {
	if s == nil || !req.LumaMode.Valid() {
		return 0, CFLAlphaResult{}, ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok {
		return 0, CFLAlphaResult{}, ErrInvalidDecodeState
	}
	cdf, err := cdfs.UVModeCDF(req.CFLAllowed, req.LumaMode)
	if err != nil {
		return 0, CFLAlphaResult{}, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, CFLAlphaResult{}, err
	}
	mode := ChromaIntraMode(symbol)
	if !mode.Valid() || (!req.CFLAllowed && mode == ChromaIntraModeCFL) {
		return 0, CFLAlphaResult{}, ErrInvalidDecodeState
	}
	if mode != ChromaIntraModeCFL {
		return mode, CFLAlphaResult{}, nil
	}
	alpha, err := s.ReadCFLAlphas(cdfs)
	if err != nil {
		return 0, CFLAlphaResult{}, err
	}
	return mode, alpha, nil
}

// ReadCFLAlphas decodes libaom's packed CfL alpha index and joint sign.
func (s *DecodeState) ReadCFLAlphas(cdfs *IntraModeCDFs) (CFLAlphaResult, error) {
	if s == nil {
		return CFLAlphaResult{}, ErrInvalidDecodeState
	}
	signCDF, err := cdfs.CFLSignCDF()
	if err != nil {
		return CFLAlphaResult{}, err
	}
	jointSign, err := s.Reader.ReadCDF(signCDF)
	if err != nil {
		return CFLAlphaResult{}, err
	}
	if jointSign < 0 || jointSign >= CFLJointSigns {
		return CFLAlphaResult{}, ErrInvalidDecodeState
	}
	var idx uint8
	if cflSignU(jointSign) != cflSignZero {
		cdf, err := cdfs.CFLAlphaCDF(cflContextU(jointSign))
		if err != nil {
			return CFLAlphaResult{}, err
		}
		alpha, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return CFLAlphaResult{}, err
		}
		idx = uint8(alpha << 4)
	}
	if cflSignV(jointSign) != cflSignZero {
		cdf, err := cdfs.CFLAlphaCDF(cflContextV(jointSign))
		if err != nil {
			return CFLAlphaResult{}, err
		}
		alpha, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return CFLAlphaResult{}, err
		}
		idx += uint8(alpha)
	}
	return CFLAlphaResult{Index: idx, JointSign: int8(jointSign)}, nil
}

// ReadFilterIntraMode decodes libaom's read_filter_intra_mode_info().
func (s *DecodeState) ReadFilterIntraMode(cdfs *IntraModeCDFs, req FilterIntraRequest) (FilterIntraMode, bool, error) {
	if s == nil {
		return 0, false, ErrInvalidDecodeState
	}
	allowed, err := FilterIntraAllowed(req)
	if err != nil || !allowed {
		return 0, false, err
	}
	cdf, err := cdfs.FilterIntraCDF(req.Size)
	if err != nil {
		return 0, false, err
	}
	useFilter, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, false, err
	}
	if useFilter == 0 {
		return 0, false, nil
	}
	modeCDF, err := cdfs.FilterIntraModeCDF()
	if err != nil {
		return 0, false, err
	}
	symbol, err := s.Reader.ReadCDF(modeCDF)
	if err != nil {
		return 0, false, err
	}
	mode := FilterIntraMode(symbol)
	if !mode.Valid() {
		return 0, false, ErrInvalidDecodeState
	}
	return mode, true, nil
}

// FilterIntraAllowed reports libaom's av1_filter_intra_allowed() gate.
func FilterIntraAllowed(req FilterIntraRequest) (bool, error) {
	dims, ok := req.Size.Dimensions()
	if !ok || !req.LumaMode.Valid() {
		return false, ErrInvalidDecodeState
	}
	return req.EnableFilterIntra &&
		req.LumaMode == IntraModeDC &&
		req.PaletteYSize == 0 &&
		int(dims.W4)*4 <= 32 &&
		int(dims.H4)*4 <= 32, nil
}

// ChromaIntraCFLAllowed reports libaom/dav1d's CfL availability for a block.
func ChromaIntraCFLAllowed(size BlockSize, color parser.ColorConfig, lossless bool) (bool, error) {
	dims, ok := size.Dimensions()
	if !ok || color.MonoChrome {
		return false, ErrInvalidDecodeState
	}
	if lossless {
		planeSize, err := PlaneBlockSize(size, color, 1)
		if err != nil {
			return false, err
		}
		return planeSize == BlockSize4x4, nil
	}
	return dims.W4 <= 8 && dims.H4 <= 8, nil
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

const (
	cflSignZero     = 0
	cflSignNegative = 1
	cflSignPositive = 2
	cflSigns        = 3
)

func shouldReadIntraAngleDelta(size BlockSize, mode IntraMode) (bool, error) {
	dims, ok := size.Dimensions()
	if !ok || !mode.Valid() {
		return false, ErrInvalidDecodeState
	}
	return dims.Log2W+dims.Log2H >= 2 && isDirectionalIntraMode(mode), nil
}

func isDirectionalIntraMode(mode IntraMode) bool {
	return mode >= IntraModeVertical && mode <= IntraModeD67
}

func cflSignU(jointSign int) int {
	return ((jointSign + 1) * 11) >> 5
}

func cflSignV(jointSign int) int {
	return (jointSign + 1) - cflSigns*cflSignU(jointSign)
}

func cflContextU(jointSign int) int {
	return jointSign + 1 - cflSigns
}

func cflContextV(jointSign int) int {
	return cflSignV(jointSign)*cflSigns + cflSignU(jointSign) - cflSigns
}
