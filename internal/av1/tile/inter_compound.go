package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

const (
	CompoundBlendContexts = 6
	InterIntraGroups      = 4
	WedgeContexts         = 9
	MaxWedgeTypes         = 16
)

// InterIntraMode identifies AV1's inter-intra prediction mode.
type InterIntraMode uint8

const (
	InterIntraModeDC InterIntraMode = iota
	InterIntraModeVertical
	InterIntraModeHorizontal
	InterIntraModeSmooth
	interIntraModeCount
)

// CompoundType identifies AV1's inter-inter compound blending type.
type CompoundType uint8

const (
	CompoundTypeAverage CompoundType = iota
	CompoundTypeDistWtd
	CompoundTypeWedge
	CompoundTypeDiffWtd
	compoundTypeCount
)

// DiffWtdMaskType identifies AV1's diff-wtd mask orientation.
type DiffWtdMaskType uint8

const (
	DiffWtdMaskType38 DiffWtdMaskType = iota
	DiffWtdMaskType38Inv
	diffWtdMaskTypeCount
)

// CompoundBlendCDFs contains caller-owned CDFs for inter-intra and compound
// blend syntax.
type CompoundBlendCDFs struct {
	CompoundIndex     [CompoundBlendContexts]entropy.CDF
	CompoundGroup     [CompoundBlendContexts]entropy.CDF
	CompoundType      [blockSizeCount]entropy.CDF
	WedgeIndex        [blockSizeCount]entropy.CDF
	InterIntra        [InterIntraGroups]entropy.CDF
	WedgeInterIntra   [blockSizeCount]entropy.CDF
	InterIntraModeCDF [InterIntraGroups]entropy.CDF
}

type InterIntraRequest struct {
	Size BlockSize
	Mode InterMode

	EnableInterIntraCompound bool
	SkipMode                 bool
	Compound                 bool
}

type InterIntraResult struct {
	Enabled    bool
	Mode       InterIntraMode
	UseWedge   bool
	WedgeIndex uint8
}

type CompoundBlendRequest struct {
	Size       BlockSize
	Compound   bool
	SkipMode   bool
	MotionMode MotionMode

	EnableMaskedCompound  bool
	EnableDistWtdCompound bool
	EnableOrderHint       bool
	OrderHintBits         uint8
	CurrentOrderHint      uint8
	RefOrderHint          [2]uint8

	X4       uint8
	Y4       uint8
	HaveTop  bool
	HaveLeft bool
}

type CompoundBlendResult struct {
	Type               CompoundType
	CompoundIndex      uint8
	CompoundGroupIndex uint8
	WedgeIndex         uint8
	WedgeSign          bool
	MaskType           DiffWtdMaskType
}

var compoundIndexDefaultCDF = [CompoundBlendContexts]uint16{18244, 12865, 7053, 13259, 9334, 4644}
var compoundGroupDefaultCDF = [CompoundBlendContexts]uint16{26607, 22891, 18840, 24594, 19934, 22674}
var interIntraDefaultCDF = [InterIntraGroups]uint16{16384, 26887, 27597, 30237}
var interIntraModeDefaultCDF = [InterIntraGroups][interIntraModeCount - 1]uint16{
	{8192, 16384, 24576},
	{1875, 11082, 27332},
	{2473, 9996, 26388},
	{4238, 11537, 25926},
}

var wedgeContextForBlock = [blockSizeCount]int8{
	BlockSize128x128: -1,
	BlockSize128x64:  -1,
	BlockSize64x128:  -1,
	BlockSize64x64:   -1,
	BlockSize64x32:   -1,
	BlockSize64x16:   -1,
	BlockSize32x64:   -1,
	BlockSize32x32:   6,
	BlockSize32x16:   5,
	BlockSize32x8:    8,
	BlockSize16x64:   -1,
	BlockSize16x32:   4,
	BlockSize16x16:   3,
	BlockSize16x8:    2,
	BlockSize16x4:    -1,
	BlockSize8x32:    7,
	BlockSize8x16:    1,
	BlockSize8x8:     0,
	BlockSize8x4:     -1,
	BlockSize4x16:    -1,
	BlockSize4x8:     -1,
	BlockSize4x4:     -1,
}

var wedgeCompoundDefaultCDF = [WedgeContexts]uint16{23431, 13171, 11470, 9770, 9100, 8233, 6172, 11820, 7701}
var interIntraWedgeDefaultCDF = [7]uint16{20036, 24957, 26704, 27530, 29564, 29444, 26872}
var wedgeIndexDefaultCDF = [WedgeContexts][MaxWedgeTypes - 1]uint16{
	{2438, 4440, 6599, 8663, 11005, 12874, 15751, 18094, 20359, 22362, 24127, 25702, 27752, 29450, 31171},
	{806, 3266, 6005, 6738, 7218, 7367, 7771, 14588, 16323, 17367, 18452, 19422, 22839, 26127, 29629},
	{2779, 3738, 4683, 7213, 7775, 8017, 8655, 14357, 17939, 21332, 24520, 27470, 29456, 30529, 31656},
	{1684, 3625, 5675, 7108, 9302, 11274, 14429, 17144, 19163, 20961, 22884, 24471, 26719, 28714, 30877},
	{1142, 3491, 6277, 7314, 8089, 8355, 9023, 13624, 15369, 16730, 18114, 19313, 22521, 26012, 29550},
	{2742, 4195, 5727, 8035, 8980, 9336, 10146, 14124, 17270, 20533, 23434, 25972, 27944, 29570, 31416},
	{1727, 3948, 6101, 7796, 9841, 12344, 15766, 18944, 20638, 22038, 23963, 25311, 26988, 28766, 31012},
	{154, 987, 1925, 2051, 2088, 2111, 2151, 23033, 23703, 24284, 24985, 25684, 27259, 28883, 30911},
	{1135, 1322, 1493, 2635, 2696, 2737, 2770, 21016, 22935, 25057, 27251, 29173, 30089, 30960, 31933},
}

func (mode InterIntraMode) Valid() bool {
	return mode < interIntraModeCount
}

func (typ CompoundType) Valid() bool {
	return typ < compoundTypeCount
}

func (typ DiffWtdMaskType) Valid() bool {
	return typ < diffWtdMaskTypeCount
}

// InitDefault seeds c with dav1d/libaom's default compound-blend CDFs.
func (c *CompoundBlendCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next CompoundBlendCDFs
	for ctx := range CompoundBlendContexts {
		if err := next.CompoundIndex[ctx].Init([]uint16{compoundIndexDefaultCDF[ctx]}); err != nil {
			return err
		}
		if err := next.CompoundGroup[ctx].Init([]uint16{compoundGroupDefaultCDF[ctx]}); err != nil {
			return err
		}
	}
	for group := range InterIntraGroups {
		if err := next.InterIntra[group].Init([]uint16{interIntraDefaultCDF[group]}); err != nil {
			return err
		}
		if err := next.InterIntraModeCDF[group].Init(interIntraModeDefaultCDF[group][:]); err != nil {
			return err
		}
	}
	for size := range blockSizeCount {
		wedgeCtx, ok := wedgeContext(size)
		if !ok {
			continue
		}
		if err := next.CompoundType[size].Init([]uint16{wedgeCompoundDefaultCDF[wedgeCtx]}); err != nil {
			return err
		}
		if err := next.WedgeIndex[size].Init(wedgeIndexDefaultCDF[wedgeCtx][:]); err != nil {
			return err
		}
		if wedgeCtx < len(interIntraWedgeDefaultCDF) {
			if err := next.WedgeInterIntra[size].Init([]uint16{interIntraWedgeDefaultCDF[wedgeCtx]}); err != nil {
				return err
			}
		}
	}
	*c = next
	return nil
}

func (c *CompoundBlendCDFs) CompoundIndexCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= CompoundBlendContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.CompoundIndex[ctx])
}

func (c *CompoundBlendCDFs) CompoundGroupCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= CompoundBlendContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.CompoundGroup[ctx])
}

func (c *CompoundBlendCDFs) CompoundTypeCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || !WedgeUsed(size) {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.CompoundType[size])
}

func (c *CompoundBlendCDFs) WedgeIndexCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || !WedgeUsed(size) {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.WedgeIndex[size]
	if cdf.Symbols() != MaxWedgeTypes {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (c *CompoundBlendCDFs) InterIntraCDF(group int) (*entropy.CDF, error) {
	if c == nil || group < 0 || group >= InterIntraGroups {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.InterIntra[group])
}

func (c *CompoundBlendCDFs) WedgeInterIntraCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || !InterIntraAllowedBlockSize(size) || !WedgeUsed(size) {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.WedgeInterIntra[size])
}

func (c *CompoundBlendCDFs) InterIntraModeCDFForGroup(group int) (*entropy.CDF, error) {
	if c == nil || group < 0 || group >= InterIntraGroups {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.InterIntraModeCDF[group]
	if cdf.Symbols() != int(interIntraModeCount) {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (s *DecodeState) ReadInterIntra(cdfs *CompoundBlendCDFs, req InterIntraRequest) (InterIntraResult, error) {
	if s == nil {
		return InterIntraResult{}, ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok || !req.Mode.Valid() {
		return InterIntraResult{}, ErrInvalidDecodeState
	}
	if !InterIntraAllowed(req) {
		return InterIntraResult{}, nil
	}
	group := yModeSizeContext[req.Size]
	cdf, err := cdfs.InterIntraCDF(group)
	if err != nil {
		return InterIntraResult{}, err
	}
	enabled, err := readBoolCDF(s, cdf)
	if err != nil || !enabled {
		return InterIntraResult{}, err
	}
	modeCDF, err := cdfs.InterIntraModeCDFForGroup(group)
	if err != nil {
		return InterIntraResult{}, err
	}
	symbol, err := s.Reader.ReadCDF(modeCDF)
	if err != nil {
		return InterIntraResult{}, err
	}
	mode := InterIntraMode(symbol)
	if !mode.Valid() {
		return InterIntraResult{}, ErrInvalidDecodeState
	}
	result := InterIntraResult{Enabled: true, Mode: mode}
	if WedgeUsed(req.Size) {
		wedgeCDF, err := cdfs.WedgeInterIntraCDF(req.Size)
		if err != nil {
			return InterIntraResult{}, err
		}
		useWedge, err := readBoolCDF(s, wedgeCDF)
		if err != nil {
			return InterIntraResult{}, err
		}
		result.UseWedge = useWedge
		if useWedge {
			wedgeIndex, err := s.readWedgeIndex(cdfs, req.Size)
			if err != nil {
				return InterIntraResult{}, err
			}
			result.WedgeIndex = wedgeIndex
		}
	}
	return result, nil
}

func (s *DecodeState) ReadCompoundBlend(cdfs *CompoundBlendCDFs, ctx *BlockModeContext, req CompoundBlendRequest) (CompoundBlendResult, error) {
	if s == nil || ctx == nil {
		return CompoundBlendResult{}, ErrInvalidDecodeState
	}
	if err := validateCompoundBlendRequest(req); err != nil {
		return CompoundBlendResult{}, err
	}
	result := CompoundBlendResult{Type: CompoundTypeAverage, CompoundIndex: 1}
	if !req.Compound || req.SkipMode {
		return result, nil
	}
	if !compoundReferenceAllowed(req.Size) {
		return CompoundBlendResult{}, ErrInvalidDecodeState
	}
	maskedCompoundUsed := req.EnableMaskedCompound && req.MotionMode == MotionModeTranslation && AnyMaskedCompoundUsed(req.Size)
	if maskedCompoundUsed {
		groupCtx, err := ctx.CompoundGroupIndexContext(req)
		if err != nil {
			return CompoundBlendResult{}, err
		}
		cdf, err := cdfs.CompoundGroupCDF(groupCtx)
		if err != nil {
			return CompoundBlendResult{}, err
		}
		group, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return CompoundBlendResult{}, err
		}
		result.CompoundGroupIndex = uint8(group)
	}
	if result.CompoundGroupIndex == 0 {
		if req.EnableDistWtdCompound {
			indexCtx, err := ctx.CompoundIndexContext(req)
			if err != nil {
				return CompoundBlendResult{}, err
			}
			cdf, err := cdfs.CompoundIndexCDF(indexCtx)
			if err != nil {
				return CompoundBlendResult{}, err
			}
			index, err := s.Reader.ReadCDF(cdf)
			if err != nil {
				return CompoundBlendResult{}, err
			}
			result.CompoundIndex = uint8(index)
			if index == 0 {
				result.Type = CompoundTypeDistWtd
			}
		}
		return result, nil
	}
	if WedgeUsed(req.Size) {
		cdf, err := cdfs.CompoundTypeCDF(req.Size)
		if err != nil {
			return CompoundBlendResult{}, err
		}
		symbol, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return CompoundBlendResult{}, err
		}
		result.Type = CompoundTypeWedge + CompoundType(symbol)
	} else {
		result.Type = CompoundTypeDiffWtd
	}
	if result.Type == CompoundTypeWedge {
		wedgeIndex, err := s.readWedgeIndex(cdfs, req.Size)
		if err != nil {
			return CompoundBlendResult{}, err
		}
		result.WedgeIndex = wedgeIndex
		bit, err := s.Reader.ReadBit()
		if err != nil {
			return CompoundBlendResult{}, err
		}
		result.WedgeSign = bit != 0
		return result, nil
	}
	bit, err := s.Reader.ReadBit()
	if err != nil {
		return CompoundBlendResult{}, err
	}
	result.MaskType = DiffWtdMaskType(bit)
	if !result.MaskType.Valid() {
		return CompoundBlendResult{}, ErrInvalidDecodeState
	}
	return result, nil
}

func (s *DecodeState) readWedgeIndex(cdfs *CompoundBlendCDFs, size BlockSize) (uint8, error) {
	cdf, err := cdfs.WedgeIndexCDF(size)
	if err != nil {
		return 0, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	if symbol < 0 || symbol >= MaxWedgeTypes {
		return 0, ErrInvalidDecodeState
	}
	return uint8(symbol), nil
}

func (c *BlockModeContext) CompoundGroupIndexContext(req CompoundBlendRequest) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateCompoundBlendRequest(req); err != nil {
		return 0, err
	}
	aboveCtx := 0
	if req.HaveTop {
		aboveCtx = compoundGroupNeighborContext(c.AboveIntra[req.X4], c.AboveCompound[req.X4], c.AboveRef[0][req.X4], c.AboveCompGroup[req.X4])
	}
	leftCtx := 0
	if req.HaveLeft {
		leftCtx = compoundGroupNeighborContext(c.LeftIntra[req.Y4], c.LeftCompound[req.Y4], c.LeftRef[0][req.Y4], c.LeftCompGroup[req.Y4])
	}
	return minInt(5, aboveCtx+leftCtx), nil
}

func (c *BlockModeContext) CompoundIndexContext(req CompoundBlendRequest) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateCompoundBlendRequest(req); err != nil {
		return 0, err
	}
	equalDistance, err := compoundReferenceDistancesEqual(req)
	if err != nil {
		return 0, err
	}
	aboveCtx := 0
	if req.HaveTop {
		aboveCtx = compoundIndexNeighborContext(c.AboveIntra[req.X4], c.AboveCompound[req.X4], c.AboveRef[0][req.X4], c.AboveCompIndex[req.X4])
	}
	leftCtx := 0
	if req.HaveLeft {
		leftCtx = compoundIndexNeighborContext(c.LeftIntra[req.Y4], c.LeftCompound[req.Y4], c.LeftRef[0][req.Y4], c.LeftCompIndex[req.Y4])
	}
	return aboveCtx + leftCtx + 3*boolInt(equalDistance), nil
}

func (c *BlockModeContext) MarkCompoundBlend(size BlockSize, x4 int, y4 int, result CompoundBlendResult) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok || !result.Type.Valid() || result.CompoundGroupIndex > 1 || result.CompoundIndex > 1 {
		return ErrInvalidDecodeState
	}
	if x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	for i := 0; i < int(dims.W4); i++ {
		c.AboveCompGroup[x4+i] = result.CompoundGroupIndex
		c.AboveCompIndex[x4+i] = result.CompoundIndex
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftCompGroup[y4+i] = result.CompoundGroupIndex
		c.LeftCompIndex[y4+i] = result.CompoundIndex
	}
	return nil
}

func InterIntraAllowed(req InterIntraRequest) bool {
	if !req.EnableInterIntraCompound || req.SkipMode || req.Compound || !req.Mode.Valid() {
		return false
	}
	return InterIntraAllowedBlockSize(req.Size)
}

func InterIntraAllowedBlockSize(size BlockSize) bool {
	switch size {
	case BlockSize32x32, BlockSize32x16, BlockSize16x32, BlockSize16x16, BlockSize16x8, BlockSize8x16, BlockSize8x8:
		return true
	default:
		return false
	}
}

func AnyMaskedCompoundUsed(size BlockSize) bool {
	return compoundReferenceAllowed(size)
}

func WedgeUsed(size BlockSize) bool {
	_, ok := wedgeContext(size)
	return ok
}

func wedgeContext(size BlockSize) (int, bool) {
	if size >= blockSizeCount {
		return 0, false
	}
	ctx := wedgeContextForBlock[size]
	if ctx < 0 {
		return 0, false
	}
	return int(ctx), true
}

func validateCompoundBlendRequest(req CompoundBlendRequest) error {
	if _, ok := req.Size.Dimensions(); !ok || !req.MotionMode.Valid() {
		return ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(int(req.X4), int(req.Y4)); err != nil {
		return err
	}
	if req.OrderHintBits > 31 {
		return ErrInvalidDecodeState
	}
	return nil
}

func compoundGroupNeighborContext(intra uint8, compound uint8, ref0 ReferenceFrame, group uint8) int {
	if intra != 0 {
		return 0
	}
	if compound != 0 {
		return int(group)
	}
	if ref0 == ReferenceFrameAltref {
		return 3
	}
	return 0
}

func compoundIndexNeighborContext(intra uint8, compound uint8, ref0 ReferenceFrame, index uint8) int {
	if intra != 0 {
		return 0
	}
	if compound != 0 {
		return int(index)
	}
	if ref0 == ReferenceFrameAltref {
		return 1
	}
	return 0
}

func compoundReferenceDistancesEqual(req CompoundBlendRequest) (bool, error) {
	d0, err := compoundRelativeOrderHint(req.EnableOrderHint, req.OrderHintBits, req.RefOrderHint[1], req.CurrentOrderHint)
	if err != nil {
		return false, err
	}
	d1, err := compoundRelativeOrderHint(req.EnableOrderHint, req.OrderHintBits, req.CurrentOrderHint, req.RefOrderHint[0])
	if err != nil {
		return false, err
	}
	if d0 < 0 {
		d0 = -d0
	}
	if d1 < 0 {
		d1 = -d1
	}
	return d0 == d1, nil
}

func compoundRelativeOrderHint(enabled bool, bits uint8, a uint8, b uint8) (int, error) {
	if !enabled {
		return 0, nil
	}
	if bits == 0 || bits > 8 {
		return 0, ErrInvalidDecodeState
	}
	limit := uint32(1) << bits
	ua := uint32(a)
	ub := uint32(b)
	if ua >= limit || ub >= limit {
		return 0, ErrInvalidDecodeState
	}
	mask := int32(1 << (bits - 1))
	diff := int32(ua) - int32(ub)
	return int((diff & (mask - 1)) - (diff & mask)), nil
}
