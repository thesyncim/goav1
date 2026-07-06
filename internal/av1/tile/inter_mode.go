package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	NewMVModeContexts    = 6
	GlobalMVModeContexts = 2
	RefMVModeContexts    = 6
	DRLModeContexts      = 3
	InterModeContexts    = 8

	globalMVOffset = 3
	refMVOffset    = 4
	newMVMask      = (1 << globalMVOffset) - 1
	globalMVMask   = (1 << (refMVOffset - globalMVOffset)) - 1
	refMVMask      = (1 << (8 - refMVOffset)) - 1
)

// InterMode identifies one single-reference inter prediction mode in dav1d
// order.
type InterMode uint8

const (
	InterModeNearestMV InterMode = iota
	InterModeNearMV
	InterModeGlobalMV
	InterModeNewMV
	interModeCount
)

// CompoundInterMode identifies one compound inter prediction mode in
// dav1d/libaom order.
type CompoundInterMode uint8

const (
	CompoundInterModeNearestNearest CompoundInterMode = iota
	CompoundInterModeNearNear
	CompoundInterModeNearestNew
	CompoundInterModeNewNearest
	CompoundInterModeNearNew
	CompoundInterModeNewNear
	CompoundInterModeGlobalGlobal
	CompoundInterModeNewNew
	compoundInterModeCount
)

type CompoundInterModeComponents struct {
	First  InterMode
	Second InterMode
}

// InterModeCDFs contains caller-owned CDFs for inter prediction mode and DRL
// syntax.
type InterModeCDFs struct {
	NewMV    [NewMVModeContexts]entropy.CDF
	GlobalMV [GlobalMVModeContexts]entropy.CDF
	RefMV    [RefMVModeContexts]entropy.CDF
	DRL      [DRLModeContexts]entropy.CDF
	Compound [InterModeContexts]entropy.CDF
}

type InterModeRequest struct {
	Compound bool
	SkipMode bool

	SegmentationEnabled bool
	Segment             parser.SegmentData

	ModeContext uint16
}

type InterModeResult struct {
	Mode         InterMode
	CompoundMode CompoundInterMode
	Compound     bool
}

type DRLRequest struct {
	Mode         InterMode
	CompoundMode CompoundInterMode
	Compound     bool
	RefMVCount   uint8
	Contexts     [3]uint8
}

var compoundInterModeComponents = [compoundInterModeCount]CompoundInterModeComponents{
	CompoundInterModeNearestNearest: {First: InterModeNearestMV, Second: InterModeNearestMV},
	CompoundInterModeNearNear:       {First: InterModeNearMV, Second: InterModeNearMV},
	CompoundInterModeNearestNew:     {First: InterModeNearestMV, Second: InterModeNewMV},
	CompoundInterModeNewNearest:     {First: InterModeNewMV, Second: InterModeNearestMV},
	CompoundInterModeNearNew:        {First: InterModeNearMV, Second: InterModeNewMV},
	CompoundInterModeNewNear:        {First: InterModeNewMV, Second: InterModeNearMV},
	CompoundInterModeGlobalGlobal:   {First: InterModeGlobalMV, Second: InterModeGlobalMV},
	CompoundInterModeNewNew:         {First: InterModeNewMV, Second: InterModeNewMV},
}

var compoundModeContextMap = [3][5]uint8{
	{0, 1, 1, 1, 1},
	{1, 2, 3, 4, 4},
	{4, 4, 5, 6, 7},
}

func (mode InterMode) Valid() bool {
	return mode < interModeCount
}

func (mode CompoundInterMode) Valid() bool {
	return mode < compoundInterModeCount
}

func (mode CompoundInterMode) Components() (CompoundInterModeComponents, error) {
	if !mode.Valid() {
		return CompoundInterModeComponents{}, ErrInvalidDecodeState
	}
	return compoundInterModeComponents[mode], nil
}

var defaultInterModeCDFs = makeDefaultInterModeCDFs()

func makeDefaultInterModeCDFs() InterModeCDFs {
	var c InterModeCDFs
	if err := c.initDefault(); err != nil {
		panic(err)
	}
	return c
}

// InitDefault seeds c with dav1d/libaom's default inter-mode CDFs.
func (c *InterModeCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	*c = defaultInterModeCDFs
	return nil
}

func (c *InterModeCDFs) initDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next InterModeCDFs
	for i, cumulative := range [...]uint16{24035, 16630, 15339, 8386, 12222, 4676} {
		if err := next.NewMV[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	for i, cumulative := range [...]uint16{2175, 1054} {
		if err := next.GlobalMV[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	for i, cumulative := range [...]uint16{23974, 24188, 17848, 28622, 24312, 19923} {
		if err := next.RefMV[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	for i, cumulative := range [...]uint16{13104, 24560, 18945} {
		if err := next.DRL[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	compound := [...][compoundInterModeCount - 1]uint16{
		{7760, 13823, 15808, 17641, 19156, 20666, 26891},
		{10730, 19452, 21145, 22749, 24039, 25131, 28724},
		{10664, 20221, 21588, 22906, 24295, 25387, 28436},
		{13298, 16984, 20471, 24182, 25067, 25736, 26422},
		{18904, 23325, 25242, 27432, 27898, 28258, 30758},
		{10725, 17454, 20124, 22820, 24195, 25168, 26046},
		{17125, 24273, 25814, 27492, 28214, 28704, 30592},
		{13046, 23214, 24505, 25942, 27435, 28442, 29330},
	}
	for i := range compound {
		if err := next.Compound[i].Init(compound[i][:]); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

func (c *InterModeCDFs) NewMVCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= NewMVModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.NewMV[ctx])
}

func (c *InterModeCDFs) GlobalMVCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= GlobalMVModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.GlobalMV[ctx])
}

func (c *InterModeCDFs) RefMVCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= RefMVModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.RefMV[ctx])
}

func (c *InterModeCDFs) DRLCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= DRLModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.DRL[ctx])
}

func (c *InterModeCDFs) CompoundModeCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= InterModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.Compound[ctx]
	if cdf.Symbols() != int(compoundInterModeCount) {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
}

func AnalyzeInterModeContext(raw uint16, compound bool) (int, error) {
	if !compound {
		newCtx := int(raw & newMVMask)
		globalCtx := int((raw >> globalMVOffset) & globalMVMask)
		refCtx := int((raw >> refMVOffset) & refMVMask)
		if newCtx >= NewMVModeContexts || globalCtx >= GlobalMVModeContexts || refCtx >= RefMVModeContexts {
			return 0, ErrInvalidDecodeState
		}
		return int(raw), nil
	}
	newCtx := int(raw & newMVMask)
	if newCtx >= len(compoundModeContextMap[0]) {
		newCtx = len(compoundModeContextMap[0]) - 1
	}
	refCtx := int((raw >> refMVOffset) & refMVMask)
	row := refCtx >> 1
	if row < 0 || row >= len(compoundModeContextMap) {
		return 0, ErrInvalidDecodeState
	}
	return int(compoundModeContextMap[row][newCtx]), nil
}

func (s *DecodeState) ReadBlockInterMode(cdfs *InterModeCDFs, req InterModeRequest) (InterModeResult, error) {
	if s == nil {
		return InterModeResult{}, ErrInvalidDecodeState
	}
	if req.SkipMode {
		if !req.Compound {
			return InterModeResult{}, ErrInvalidDecodeState
		}
		return InterModeResult{Compound: true, CompoundMode: CompoundInterModeNearestNearest}, nil
	}
	if req.SegmentationEnabled && (req.Segment.Skip || req.Segment.GlobalMV) {
		return InterModeResult{Mode: InterModeGlobalMV}, nil
	}
	context, err := AnalyzeInterModeContext(req.ModeContext, req.Compound)
	if err != nil {
		return InterModeResult{}, err
	}
	if req.Compound {
		mode, err := s.ReadCompoundInterMode(cdfs, context)
		if err != nil {
			return InterModeResult{}, err
		}
		return InterModeResult{Compound: true, CompoundMode: mode}, nil
	}
	mode, err := s.ReadSingleInterMode(cdfs, uint16(context))
	if err != nil {
		return InterModeResult{}, err
	}
	return InterModeResult{Mode: mode}, nil
}

func (s *DecodeState) ReadSingleInterMode(cdfs *InterModeCDFs, modeContext uint16) (InterMode, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	newCtx := int(modeContext & newMVMask)
	cdf, err := cdfs.NewMVCDF(newCtx)
	if err != nil {
		return 0, err
	}
	newSymbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	if newSymbol == 0 {
		return InterModeNewMV, nil
	}

	globalCtx := int((modeContext >> globalMVOffset) & globalMVMask)
	cdf, err = cdfs.GlobalMVCDF(globalCtx)
	if err != nil {
		return 0, err
	}
	globalSymbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	if globalSymbol == 0 {
		return InterModeGlobalMV, nil
	}

	refCtx := int((modeContext >> refMVOffset) & refMVMask)
	cdf, err = cdfs.RefMVCDF(refCtx)
	if err != nil {
		return 0, err
	}
	refSymbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	if refSymbol == 0 {
		return InterModeNearestMV, nil
	}
	return InterModeNearMV, nil
}

func (s *DecodeState) ReadCompoundInterMode(cdfs *InterModeCDFs, context int) (CompoundInterMode, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	cdf, err := cdfs.CompoundModeCDF(context)
	if err != nil {
		return 0, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	mode := CompoundInterMode(symbol)
	if !mode.Valid() {
		return 0, ErrInvalidDecodeState
	}
	return mode, nil
}

// ReadDRLIndex decodes libaom's ref_mv_idx value, not dav1d's absolute
// mv-stack slot. Callers provide already-computed av1_drl_ctx()/get_drl_context
// contexts from the reference-MV search.
func (s *DecodeState) ReadDRLIndex(cdfs *InterModeCDFs, req DRLRequest) (int, error) {
	if s == nil || req.RefMVCount > MaxRefMVStackSize {
		return 0, ErrInvalidDecodeState
	}
	if req.Compound {
		if !req.CompoundMode.Valid() {
			return 0, ErrInvalidDecodeState
		}
	} else if !req.Mode.Valid() {
		return 0, ErrInvalidDecodeState
	}
	refMVCount := int(req.RefMVCount)

	if req.usesNewMV() {
		refMVIndex := 0
		for idx := range 2 {
			if refMVCount <= idx+1 {
				return refMVIndex, nil
			}
			bit, err := s.readDRLBit(cdfs, req.Contexts[idx])
			if err != nil {
				return 0, err
			}
			refMVIndex = idx + boolInt(bit)
			if !bit {
				return refMVIndex, nil
			}
		}
		return refMVIndex, nil
	}

	if req.usesNearMV() {
		refMVIndex := 0
		for idx := 1; idx < 3; idx++ {
			if refMVCount <= idx+1 {
				return refMVIndex, nil
			}
			bit, err := s.readDRLBit(cdfs, req.Contexts[idx])
			if err != nil {
				return 0, err
			}
			refMVIndex = idx + boolInt(bit) - 1
			if !bit {
				return refMVIndex, nil
			}
		}
		return refMVIndex, nil
	}

	return 0, nil
}

func (s *DecodeState) readDRLBit(cdfs *InterModeCDFs, context uint8) (bool, error) {
	cdf, err := cdfs.DRLCDF(int(context))
	if err != nil {
		return false, err
	}
	return readBoolCDF(s, cdf)
}

func (req DRLRequest) usesNewMV() bool {
	if !req.Compound {
		return req.Mode == InterModeNewMV
	}
	return req.CompoundMode == CompoundInterModeNewNew
}

func (req DRLRequest) usesNearMV() bool {
	if !req.Compound {
		return req.Mode == InterModeNearMV
	}
	components, err := req.CompoundMode.Components()
	if err != nil {
		return false
	}
	return components.First == InterModeNearMV || components.Second == InterModeNearMV
}
